"""SQLite-Ablage für Dokumente und Textstücke.

Eine Datei, zwei Tabellen. SQLite reicht für ein paar hundert Seiten völlig und
macht die Zwischenstände mit normalem SQL inspizierbar (sqlite3 data/cite.sqlite).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_FILE = Path("data") / "cite.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    name     TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    url      TEXT,
    pages    INTEGER NOT NULL,
    sha256   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc         TEXT NOT NULL REFERENCES documents(name),
    config      TEXT NOT NULL,          -- Chunking-Variante, z. B. "c1200", damit Varianten nebeneinander liegen
    ordinal     INTEGER NOT NULL,       -- Position im Dokument
    section     TEXT,                   -- zuletzt gesehene Überschrift, z. B. "2.2.2 Primary and Secondary Variables"
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_config ON chunks(doc, config);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id  INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,                -- float32, little-endian
    PRIMARY KEY (chunk_id, model)
);
"""


@dataclass
class Chunk:
    doc: str
    section: str | None
    page_start: int
    page_end: int
    text: str
    id: int | None = None
    config: str = ""
    ordinal: int = 0

    @property
    def ref(self) -> str:
        """Kurzer Beleg für Antworten: Dokument, Abschnitt, Seite."""
        sec = f", {self.section[:57]}…" if self.section and len(self.section) > 60 else (f", {self.section}" if self.section else "")
        pages = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return f"[{self.doc}{sec}, {pages}]"


def connect(path: Path = DB_FILE) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert_document(con: sqlite3.Connection, name: str, title: str, url: str | None, pages: int, sha256: str) -> None:
    con.execute(
        "INSERT INTO documents(name, title, url, pages, sha256) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET title=excluded.title, url=excluded.url, pages=excluded.pages, sha256=excluded.sha256",
        (name, title, url, pages, sha256),
    )


def replace_chunks(con: sqlite3.Connection, doc: str, config: str, chunks: list[Chunk]) -> int:
    """Alle Stücke eines Dokuments für eine Chunking-Variante ersetzen (samt ihrer Vektoren)."""
    con.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc = ? AND config = ?)", (doc, config))
    con.execute("DELETE FROM chunks WHERE doc = ? AND config = ?", (doc, config))
    con.executemany(
        "INSERT INTO chunks(doc, config, ordinal, section, page_start, page_end, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(doc, config, i, c.section, c.page_start, c.page_end, c.text) for i, c in enumerate(chunks)],
    )
    con.commit()
    return len(chunks)


def load_chunks(con: sqlite3.Connection, config: str, doc: str | None = None) -> list[Chunk]:
    sql = "SELECT * FROM chunks WHERE config = ?"
    args: list = [config]
    if doc:
        sql += " AND doc = ?"
        args.append(doc)
    rows = con.execute(sql + " ORDER BY doc, ordinal", args).fetchall()
    return [
        Chunk(doc=r["doc"], section=r["section"], page_start=r["page_start"], page_end=r["page_end"],
              text=r["text"], id=r["id"], config=r["config"], ordinal=r["ordinal"])
        for r in rows
    ]


def stats(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT d.name, d.title, d.pages, c.config, COUNT(c.id) AS chunks, "
        "ROUND(AVG(LENGTH(c.text))) AS avg_chars "
        "FROM documents d LEFT JOIN chunks c ON c.doc = d.name GROUP BY d.name, c.config ORDER BY d.name, c.config"
    ).fetchall()
    return [dict(r) for r in rows]


def save_vectors(con: sqlite3.Connection, model: str, items: list[tuple[int, "np.ndarray"]]) -> None:
    import numpy as np  # noqa: F811  (nur hier gebraucht)

    con.executemany(
        "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?, ?, ?, ?)",
        [(cid, model, int(len(v)), np.asarray(v, dtype=np.float32).tobytes()) for cid, v in items],
    )
    con.commit()


def load_vectors(con: sqlite3.Connection, config: str, model: str) -> list[tuple[int, "np.ndarray"]]:
    import numpy as np

    rows = con.execute(
        "SELECT e.chunk_id, e.vector FROM embeddings e JOIN chunks c ON c.id = e.chunk_id "
        "WHERE c.config = ? AND e.model = ? ORDER BY c.doc, c.ordinal",
        (config, model),
    ).fetchall()
    return [(r["chunk_id"], np.frombuffer(r["vector"], dtype=np.float32)) for r in rows]

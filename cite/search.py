"""Schritt 2: Suche über die Stücke. Drei Wege, damit Schritt 4 sie vergleichen kann.

- bm25:   klassische Stichwortsuche (Okapi BM25). Kein Modell, keine Vorbereitung, die Grundlinie.
- embed:  dichte Suche. Jedes Stück wird einmal mit einem kleinen lokalen Modell (fastembed, läuft auf
          der CPU) in einen Vektor übersetzt und in SQLite abgelegt; die Frage wird genauso übersetzt,
          Treffer sind die Stücke mit der größten Kosinus-Ähnlichkeit.
- hybrid: beide Ranglisten per Reciprocal Rank Fusion zusammengelegt. Kein Gewicht zu wählen,
          ein Stück gewinnt, wenn es in beiden Listen weit oben steht.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from cite.store import Chunk, load_chunks, load_vectors, save_vectors

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 Dimensionen, ~130 MB, englisch

STOPWORDS = set("""a an and are as at be been by for from has have how in is it its of on or should that the
their there these this to was were what when where which who will with would""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Kleinbuchstaben, Wörter, Stoppwörter raus, grobe Pluralform weg ("trials" -> "trial").
    Bewusst einfach; ob es reicht, misst Schritt 4."""
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    method: str

    @property
    def snippet(self) -> str:
        return re.sub(r"\s+", " ", self.chunk.text)[:160]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(f"{c.section or ''} {c.text}") for c in chunks])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), rank + 1, "bm25") for rank, i in enumerate(order) if scores[i] > 0]


class Embedder:
    """Dünne Hülle um fastembed, damit Tests ein Fake-Modell einsetzen können."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # erst hier, der Import lädt das Modell nicht, ist aber langsam

        self.model_name = model_name
        # Modell unter data/models ablegen, nicht in /tmp, damit es einen Neustart überlebt
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(Path("data") / "models"))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(list(self._model.query_embed([text]))[0], dtype=np.float32)


class EmbeddingIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder, method: str = "embed"):
        assert len(chunks) == len(vectors)
        self.chunks = chunks
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.where(norms == 0, 1, norms)
        self.embedder = embedder
        self.method = method

    @classmethod
    def build(cls, con: sqlite3.Connection, config: str, embedder, *, model_name: str | None = None,
              batch: int = 32, log=print) -> "EmbeddingIndex":
        """Fehlende Vektoren rechnen und speichern, dann Index laden. Zweiter Aufruf rechnet nichts mehr."""
        model_name = model_name or getattr(embedder, "model_name", "unknown")
        chunks = load_chunks(con, config)
        have = {cid for cid, _ in load_vectors(con, config, model_name)}
        todo = [c for c in chunks if c.id not in have]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            vecs = embedder.embed_passages([f"{c.section or ''}\n{c.text}" for c in part])
            save_vectors(con, model_name, [(c.id, v) for c, v in zip(part, vecs)])
            log(f"[ok] embedded {min(i + batch, len(todo))}/{len(todo)} ({config}, {model_name})")
        return cls.load(con, config, embedder, model_name=model_name)

    @classmethod
    def load(cls, con: sqlite3.Connection, config: str, embedder, *, model_name: str | None = None) -> "EmbeddingIndex":
        model_name = model_name or getattr(embedder, "model_name", "unknown")
        by_id = {c.id: c for c in load_chunks(con, config)}
        rows = [(cid, vec) for cid, vec in load_vectors(con, config, model_name) if cid in by_id]
        if not rows:
            raise RuntimeError(f"no vectors for {config}/{model_name}; run `python -m cite index` first")
        chunks = [by_id[cid] for cid, _ in rows]
        return cls(chunks, np.vstack([v for _, v in rows]), embedder)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.embedder.embed_query(query)
        q = q / (np.linalg.norm(q) or 1)
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i]), rank + 1, self.method) for rank, i in enumerate(order)]


def reciprocal_rank_fusion(*rankings: list[Hit], k: int = 10, rrf_k: int = 60) -> list[Hit]:
    """Score eines Stücks = Summe über Listen von 1/(rrf_k + Rang). Standardwert 60 aus der Literatur."""
    scores: dict[int, float] = {}
    chunks: dict[int, Chunk] = {}
    for ranking in rankings:
        for hit in ranking:
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (rrf_k + hit.rank)
            chunks[hit.chunk.id] = hit.chunk
    order = sorted(scores, key=lambda cid: -scores[cid])[:k]
    return [Hit(chunks[cid], scores[cid], rank + 1, "hybrid") for rank, cid in enumerate(order)]


class Searcher:
    """Bündelt die drei Wege hinter einer Methode, damit Antwort- und Messschritt nur eine Schnittstelle brauchen."""

    def __init__(self, con: sqlite3.Connection, config: str = "c1200", *, embedder=None, model_name: str = DEFAULT_MODEL):
        self.con = con
        self.config = config
        self.chunks = load_chunks(con, config)
        if not self.chunks:
            raise RuntimeError(f"no chunks for {config}; run `python -m cite ingest` first")
        self.bm25 = BM25Index(self.chunks)
        self._embedder = embedder
        self._model_name = model_name
        self._embed_index: EmbeddingIndex | None = None

    @property
    def embed_index(self) -> EmbeddingIndex:
        if self._embed_index is None:
            embedder = self._embedder or Embedder(self._model_name)
            self._embed_index = EmbeddingIndex.load(self.con, self.config, embedder, model_name=self._model_name)
        return self._embed_index

    def search(self, query: str, method: str = "hybrid", k: int = 5) -> list[Hit]:
        if method == "bm25":
            return self.bm25.search(query, k)
        if method == "embed":
            return self.embed_index.search(query, k)
        if method == "hybrid":
            depth = max(k * 4, 20)  # beide Listen tiefer holen, dann verschmelzen
            return reciprocal_rank_fusion(self.bm25.search(query, depth), self.embed_index.search(query, depth), k=k)
        raise ValueError(f"unknown method {method}")

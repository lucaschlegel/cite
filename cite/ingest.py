"""Schritt 1: PDFs laden, Text je Seite herausziehen, in Stücke schneiden, in SQLite ablegen.

Warum Stücke (chunks)? Das Modell bekommt später nur die paar Stücke, die zur Frage passen,
nicht das ganze Dokument. Jedes Stück merkt sich Dokument, Seiten und die zuletzt gesehene
Überschrift, damit eine Antwort auf eine Stelle zeigen kann, die man nachschlagen kann.

Die Stückgröße ist ein Parameter (config "c1200" = etwa 1200 Zeichen), weil Schritt 4 misst,
welche Größe die besseren Treffer liefert.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml
from pypdf import PdfReader

from cite.store import Chunk

USER_AGENT = "cite/0.1 (document QA experiment; contact via GitHub)"
DATA_DIR = Path("data")

# Überschriften: "2.2.2 Primary and Secondary Variables", "II. PRINCIPLES OF ICH GCP", "ANNEX 1", "1.1 Scope", "A.3.1 Estimands".
# Ein Satz, der zufällig mit einer Zahl beginnt ("5 percent of ..."), fällt durch: Titel beginnt groß, endet ohne Satzzeichen.
HEADING_RE = re.compile(
    r"^(?:(?P<num>(?:[A-Z]\.)?\d{1,2}(?:\.\d{1,2}){0,3})\.?|(?P<roman>[IVX]{1,5})\.|(?P<annex>ANNEX\s+\d+|APPENDIX\s+[A-Z0-9]+))"
    r"(?:\s+(?P<title>[A-Z][^\n]{1,90}?[^.,;:\s]))?\s*$"
)
# Zeilen, die nur Seitenzahl oder Kopfzeile sind
NOISE_RE = re.compile(r"^(?:\d{1,3}|page \d+|ICH .{0,40}Guideline)\s*$", re.I)
# Inhaltsverzeichnis: "3.8.1 Notification ........ 26". Solche Zeilen würden sonst als Überschriften zählen.
TOC_RE = re.compile(r"\.{3,}\s*\d{1,3}\s*$")
DATE_RE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)


@dataclass
class Page:
    number: int  # 1-basiert, wie im PDF-Viewer
    text: str


def download(url: str, target: Path, *, timeout: int = 60) -> Path:
    """PDF laden, wenn noch nicht da. Kein erneuter Download bei jedem Lauf."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def read_pages(pdf_path: Path) -> list[Page]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(Page(number=i, text=normalize(text)))
    return pages


def normalize(text: str) -> str:
    """Typische PDF-Artefakte glätten: Mehrfachleerzeichen, getrennte Wörter am Zeilenende."""
    text = text.replace("­", "")  # weiches Trennzeichen
    text = re.sub(r"-\n(?=[a-z])", "", text)  # "thera-\npeutic" -> "therapeutic"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_heading(line: str) -> str | None:
    """Gibt die Überschrift zurück, wenn die Zeile eine ist, sonst None."""
    m = HEADING_RE.match(line.strip())
    if not m:
        return None
    label = m.group("num") or m.group("roman") or m.group("annex")
    if m.group("num") is None and m.group("annex") is None and not m.group("title"):
        return None  # römische Zahl ohne Titel ist eher eine Aufzählung
    if m.group("num") and not m.group("title"):
        return None  # nackte Zahl ist eine Seitenzahl
    title = m.group("title") or ""
    if DATE_RE.match(title):
        return None  # "27 April 1995" ist ein Datum, keine Überschrift
    return f"{label} {title}".strip()


SENTENCE_END = re.compile(r"[.!?:;]\s*$")


def heading_level(heading: str) -> int:
    """"2 INVESTIGATOR" -> 1, "2.1 Qualifications" -> 2, "2.1.1 The investigator" -> 3; römisch und Annex -> 1."""
    label = heading.split(" ", 1)[0]
    if label[0].isdigit():
        return label.count(".") + 1
    if len(label) > 2 and label[1] == "." and label[2].isdigit():  # "A.3.1" -> wie "3.1"
        return label.count(".")
    return 1


def chunk_pages(doc: str, pages: list[Page], *, max_chars: int = 1200, min_chars: int | None = None,
                overlap_lines: int = 2) -> list[Chunk]:
    """Zeilen sammeln, bis etwa max_chars erreicht sind. Jeder Chunk kennt Seitenbereich und Überschrift.

    PDF-Text hat keine verlässlichen Absätze, deshalb ist die Zeile die kleinste Einheit. Ein Chunk wird
    bevorzugt an einem Satzende geschlossen (ab 80 % der Zielgröße), spätestens bei 125 %.
    Eine Hauptüberschrift beginnt immer einen neuen Chunk, eine Unterüberschrift erst, wenn der aktuelle
    Chunk mindestens min_chars hat (Standard 40 % von max_chars, Ebene 2 die Hälfte davon); kürzere
    Abschnitte werden zusammengelegt, der Chunk trägt dann die erste Überschrift.
    overlap_lines Zeilen wandern in den nächsten Chunk hinüber, nicht aber über eine Überschrift hinweg."""
    if min_chars is None:
        min_chars = int(max_chars * 0.4)
    chunks: list[Chunk] = []
    buffer: list[tuple[str, int]] = []  # (Zeile, Seite)
    section: str | None = None        # zuletzt gesehene Überschrift im Dokument
    chunk_section: str | None = None  # Überschrift, die dieser Chunk trägt

    def size() -> int:
        return sum(len(t) + 1 for t, _ in buffer)

    def flush(*, carry: bool) -> None:
        nonlocal buffer, chunk_section
        if not buffer:
            return
        text = "\n".join(t for t, _ in buffer).strip()
        if len(text) >= 40:  # Krümel verwerfen
            chunks.append(Chunk(doc=doc, section=chunk_section, page_start=buffer[0][1], page_end=buffer[-1][1], text=text))
        buffer = buffer[-overlap_lines:] if carry and overlap_lines else []
        chunk_section = section if buffer else None

    for page in pages:
        for raw in page.text.split("\n"):
            line = raw.strip()
            if not line or NOISE_RE.match(line) or TOC_RE.search(line):
                continue
            heading = find_heading(line)
            if heading:
                # Hauptabschnitte ("2. INVESTIGATOR", "II PRINCIPLES", "ANNEX 1") beginnen immer ein neues Stück,
                # Unterabschnitte erst, wenn das aktuelle Stück groß genug ist (Ebene 2 halb so streng).
                threshold = {1: 0, 2: min_chars // 2}.get(heading_level(heading), min_chars)
                if size() >= threshold:
                    flush(carry=False)
                section = heading
                if chunk_section is None or not buffer:
                    chunk_section = heading
            elif size() >= max_chars * 1.25 or (size() >= max_chars * 0.8 and SENTENCE_END.search(buffer[-1][0] if buffer else "")):
                flush(carry=True)
            if chunk_section is None:
                chunk_section = section
            buffer.append((line, page.number))
    flush(carry=False)
    return chunks


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(path: str | Path = "sources.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["documents"]


def ingest_document(source: dict, *, data_dir: Path = DATA_DIR, max_chars: int = 1200) -> tuple[dict, list[Chunk]]:
    """Ein Dokument komplett: laden, lesen, schneiden. Gibt Metadaten und Chunks zurück, speichert nichts."""
    pdf = download(source["url"], data_dir / f"{source['name']}.pdf") if source.get("url") else Path(source["path"])
    pages = read_pages(pdf)
    chunks = chunk_pages(source["name"], pages, max_chars=max_chars)
    meta = {"name": source["name"], "title": source["title"], "url": source.get("url"),
            "pages": len(pages), "sha256": sha256_of(pdf)}
    return meta, chunks

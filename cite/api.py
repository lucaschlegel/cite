"""Schritt 5: kleine HTTP-Schnittstelle.

  python -m cite serve            # http://127.0.0.1:8000/docs
  GET  /health
  GET  /search?q=...&method=hybrid&k=5
  POST /ask   {"question": "...", "method": "hybrid", "k": 5}

Suche braucht kein Modell, /ask ruft Claude (ANTHROPIC_API_KEY). Index und Datenbank werden beim ersten
Aufruf geladen und dann wiederverwendet. create_app() nimmt fertige Objekte entgegen, damit Tests ohne
Netz und ohne Modell laufen.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cite.answer import AnswerRecord, Answerer
from cite.search import DEFAULT_MODEL, Searcher
from cite.store import connect

DB_FILE = Path(os.environ.get("CITE_DB", "data/cite.sqlite"))
CONFIG = os.environ.get("CITE_CONFIG", "c1200")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    method: str = Field(default="hybrid", pattern="^(bm25|embed|hybrid)$")
    k: int = Field(default=5, ge=1, le=20)


def _hit_json(h) -> dict:
    return {"rank": h.rank, "score": round(h.score, 4), "chunk_id": h.chunk.id, "doc": h.chunk.doc,
            "section": h.chunk.section, "pages": [h.chunk.page_start, h.chunk.page_end], "ref": h.chunk.ref,
            "text": h.chunk.text}


def _record_json(rec: AnswerRecord) -> dict:
    return {
        "question": rec.question,
        "answerable": rec.answer.answerable,
        "answer": rec.answer.answer,
        "grounded": rec.grounded,
        "citations": [{"passage": c.passage, "ref": c.ref, "quote": c.quote, "ok": c.ok, "near_verbatim": c.fuzzy,
                       "similarity": c.similarity} for c in rec.checked],
        "passages": [_hit_json(h) for h in rec.hits],
        "model": rec.model,
    }


def create_app(searcher: Searcher | None = None, answerer: Answerer | None = None) -> FastAPI:
    app = FastAPI(title="cite", version="0.1", description="Fragen über Dokumente, Antworten mit geprüften Belegen.")
    state = {"searcher": searcher, "answerer": answerer}

    def get_searcher() -> Searcher:
        if state["searcher"] is None:
            state["searcher"] = Searcher(connect(DB_FILE), CONFIG, model_name=DEFAULT_MODEL)
        return state["searcher"]

    def get_answerer() -> Answerer:
        if state["answerer"] is None:
            state["answerer"] = Answerer(get_searcher())
        return state["answerer"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "db": str(DB_FILE), "config": CONFIG}

    @app.get("/search")
    def search(q: str = Query(min_length=2), method: str = Query("hybrid", pattern="^(bm25|embed|hybrid)$"),
               k: int = Query(5, ge=1, le=20)) -> list[dict]:
        try:
            return [_hit_json(h) for h in get_searcher().search(q, method=method, k=k)]
        except RuntimeError as exc:  # z. B. keine Vektoren
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        try:
            return _record_json(get_answerer().ask(req.question, method=req.method, k=req.k))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    return app


app = create_app()

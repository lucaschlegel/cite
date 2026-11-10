"""Schritt 3: Antwort aus den gefundenen Stücken, mit Belegen, die sich nachprüfen lassen.

Das Modell bekommt die Frage und die k besten Stücke, jedes mit einer Nummer. Es muss das Werkzeug
record_answer aufrufen (Schema aus Pydantic, wie bei der Extraktion in sift). Jeder Beleg nennt die
Nummer des Stücks und ein wörtliches Zitat daraus. Das erlaubt zwei Prüfungen ohne Modell:
  1. Zeigt jeder Beleg auf ein Stück, das wirklich vorgelegt wurde?
  2. Steht das Zitat wörtlich in diesem Stück?
Ein Beleg, der eine der beiden Prüfungen nicht besteht, ist erfunden. Das ist der Teil von RAG, der
in der Praxis zählt: nicht die Antwort, sondern ob man ihr trauen kann.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from cite.search import Hit

DEFAULT_MODEL = os.environ.get("CITE_MODEL", "claude-haiku-5-5")
PROMPT_VERSION = "p1"


class Citation(BaseModel):
    passage: int = Field(description="Number of the passage as given in the prompt, e.g. 3 for [3]")
    quote: str = Field(description="Verbatim quote from that passage, 5 to 40 words, supporting the claim")


class Answer(BaseModel):
    answerable: bool = Field(description="false if the passages do not contain the information needed")
    answer: str = Field(description="The answer in two to five plain sentences; if not answerable, say what is missing")
    citations: list[Citation] = Field(default_factory=list, description="One citation per claim; empty if not answerable")


SYSTEM_PROMPT = """You answer questions strictly from the numbered passages you are given.
Rules:
- Use only the passages. Do not add knowledge from elsewhere, even if you are sure.
- If the passages do not contain the information, set answerable=false and say briefly what is missing. Do not guess.
- Every factual claim in the answer needs a citation: the passage number and a verbatim quote from that passage (copy the words exactly as they appear, 5 to 40 words). Do not paraphrase inside quotes.
- Prefer the passage that states the point most directly. Cite several passages if the answer draws on several.
- Write plainly, two to five sentences. No preamble.
Then call record_answer exactly once."""

TOOL = {
    "name": "record_answer",
    "description": "Record the answer and its citations.",
    "input_schema": Answer.model_json_schema(),
}


def build_user_message(question: str, hits: list[Hit]) -> str:
    parts = [f"QUESTION: {question}", "", "PASSAGES:"]
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] {h.chunk.ref}\n{h.chunk.text}\n")
    return "\n".join(parts)


@dataclass
class CheckedCitation:
    passage: int
    quote: str
    valid_passage: bool  # Nummer existiert
    quote_found: bool    # Zitat steht wörtlich im Stück (nach Normalisierung von Leerraum)
    ref: str = ""

    @property
    def ok(self) -> bool:
        return self.valid_passage and self.quote_found


@dataclass
class AnswerRecord:
    question: str
    answer: Answer
    hits: list[Hit]
    checked: list[CheckedCitation] = field(default_factory=list)
    model: str = ""
    prompt_version: str = PROMPT_VERSION

    @property
    def grounded(self) -> bool:
        """Antwort gilt als belegt, wenn sie beantwortbar ist, mindestens einen Beleg hat und alle Belege stimmen."""
        return self.answer.answerable and bool(self.checked) and all(c.ok for c in self.checked)


def _squash(text: str) -> str:
    """Leerraum und Zeilenumbrüche egalisieren, Anführungszeichen vereinheitlichen, damit ein Zitat über
    einen Zeilenumbruch im PDF-Text hinweg trotzdem gefunden wird."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def check_citations(answer: Answer, hits: list[Hit]) -> list[CheckedCitation]:
    out = []
    for c in answer.citations:
        valid = 1 <= c.passage <= len(hits)
        found = False
        ref = ""
        if valid:
            chunk = hits[c.passage - 1].chunk
            ref = chunk.ref
            found = _squash(c.quote) in _squash(chunk.text)
        out.append(CheckedCitation(passage=c.passage, quote=c.quote, valid_passage=valid, quote_found=found, ref=ref))
    return out


def parse_tool_response(response) -> Answer:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL["name"]:
            return Answer.model_validate(block.input)
    raise ValueError("model did not call record_answer")


class Answerer:
    def __init__(self, searcher, *, model: str | None = None, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.searcher = searcher

    def ask(self, question: str, *, method: str = "hybrid", k: int = 5) -> AnswerRecord:
        hits = self.searcher.search(question, method=method, k=k)
        return self.answer_from_hits(question, hits)

    def answer_from_hits(self, question: str, hits: list[Hit]) -> AnswerRecord:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": TOOL["name"]},
            messages=[{"role": "user", "content": build_user_message(question, hits)}],
        )
        answer = parse_tool_response(response)
        return AnswerRecord(question=question, answer=answer, hits=hits, checked=check_citations(answer, hits),
                            model=self.model)


def format_record(rec: AnswerRecord) -> str:
    """Lesbare Ausgabe für die Kommandozeile: Antwort, dann die Belege mit Prüfergebnis."""
    lines = [rec.answer.answer, ""]
    if not rec.answer.answerable:
        lines.append("(not answerable from the documents)")
    for c in rec.checked:
        mark = "ok " if c.ok else ("?? " if c.valid_passage else "!! ")
        where = c.ref if c.valid_passage else f"passage {c.passage} does not exist"
        lines.append(f"{mark}[{c.passage}] {where}")
        lines.append(f"    \"{c.quote}\"" + ("" if c.quote_found or not c.valid_passage else "   <- quote not found verbatim"))
    lines.append("")
    lines.append(f"grounded: {'yes' if rec.grounded else 'no'}   model: {rec.model}   passages: {len(rec.hits)}")
    return "\n".join(lines)

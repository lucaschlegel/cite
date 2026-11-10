"""Schritt 4: Messen statt glauben.

Fragenkatalog data/questions.csv, von Hand gesetzt: Frage, Dokument, Abschnitt (zum Nachschlagen) und
eine Gold-Passage, ein kurzer wörtlicher Ausschnitt der Stelle, die die Antwort enthält (mehrere
akzeptable Stellen mit " | " getrennt). Dazu einige
Fragen, die die Dokumente nicht beantworten (answerable=no), um zu messen, ob das Modell dann auch
wirklich "steht nicht drin" sagt.

Die Gold-Passage statt einer Chunk-Nummer macht die Labels unabhängig von der Stückgröße: ein Treffer
ist jedes Stück, das die Passage enthält (Vergleich wie bei den Zitaten, ohne Leerraum und Satzzeichen).

Retrieval (ohne Modell, kostenlos): recall@1, @3, @5 und MRR je Suchweg.
Antworten (mit Modell, Cent-Beträge): Anteil belegter Antworten, Anteil korrekter Belege, Anteil der
Antworten, die die Gold-Stelle zitieren, und bei unbeantwortbaren Fragen der Anteil, bei dem das
Modell richtigerweise ablehnt.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from cite.answer import AnswerRecord, _key
from cite.search import Hit

QUESTIONS_FILE = Path("data") / "questions.csv"


@dataclass
class Question:
    id: int
    doc: str
    section: str
    question: str
    gold_phrase: str
    answerable: bool

    @property
    def gold_phrases(self) -> list[str]:
        """Mehrere akzeptable Stellen mit " | " trennen, z. B. Definition im Glossar oder im Kapitel."""
        return [p.strip() for p in self.gold_phrase.split("|") if p.strip()]

    def matches(self, hit: Hit) -> bool:
        if hit.chunk.doc != self.doc:
            return False
        text = _key(hit.chunk.text)
        return any(_key(p) in text for p in self.gold_phrases)


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(Question(id=int(row["id"]), doc=row["doc"], section=row["section"], question=row["question"],
                                gold_phrase=row["gold_phrase"], answerable=row["answerable"].strip().lower() == "yes"))
    return out


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    question_id: int
    method: str
    rank: int | None  # Rang des ersten Treffers mit der Gold-Passage, None = nicht in den Top-k


@dataclass
class RetrievalReport:
    method: str
    config: str
    k: int
    n: int
    recall_at: dict[int, float]
    mrr: float
    misses: list[int]  # Fragen ohne Treffer in den Top-k


def first_hit_rank(question: Question, hits: list[Hit]) -> int | None:
    for h in hits:
        if question.matches(h):
            return h.rank
    return None


def evaluate_retrieval(searcher, questions: list[Question], *, methods=("bm25", "embed", "hybrid"), k: int = 10,
                       log=None) -> tuple[list[RetrievalReport], list[RetrievalResult]]:
    answerable = [q for q in questions if q.answerable]
    reports, results = [], []
    for method in methods:
        ranks = []
        for q in answerable:
            rank = first_hit_rank(q, searcher.search(q.question, method=method, k=k))
            ranks.append(rank)
            results.append(RetrievalResult(q.id, method, rank))
            if log:
                log(f"  {method:6} q{q.id:<3} rank {rank}")
        n = len(answerable)
        recall = {c: sum(1 for r in ranks if r is not None and r <= c) / n for c in (1, 3, 5) if c <= k}
        mrr = sum(1.0 / r for r in ranks if r is not None) / n
        misses = [q.id for q, r in zip(answerable, ranks) if r is None]
        reports.append(RetrievalReport(method=method, config=searcher.config, k=k, n=n, recall_at=recall,
                                       mrr=round(mrr, 3), misses=misses))
    return reports, results


# ---------------------------------------------------------------------------
# Antworten
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    question_id: int
    answerable_expected: bool
    answerable_said: bool
    grounded: bool
    citations: int
    citations_ok: int
    gold_cited: bool
    answer: str


@dataclass
class AnswerReport:
    method: str
    k: int
    model: str
    n_answerable: int
    grounded_rate: float        # belegte Antworten / beantwortbare Fragen
    citation_precision: float   # korrekte Belege / alle Belege (beantwortbare Fragen)
    gold_cited_rate: float      # Antworten, die die Gold-Stelle zitieren / beantwortbare Fragen
    n_unanswerable: int
    refusal_rate: float         # Modell sagt "nicht beantwortbar" oder gibt keinen Beleg / unbeantwortbare Fragen
    results: list[AnswerResult] = field(default_factory=list)


def judge_answer(question: Question, rec: AnswerRecord) -> AnswerResult:
    gold_cited = any(c.ok and question.matches(rec.hits[c.passage - 1]) for c in rec.checked if c.valid_passage)
    return AnswerResult(
        question_id=question.id, answerable_expected=question.answerable, answerable_said=rec.answer.answerable,
        grounded=rec.grounded, citations=len(rec.checked), citations_ok=sum(1 for c in rec.checked if c.ok),
        gold_cited=gold_cited, answer=rec.answer.answer,
    )


def evaluate_answers(answerer, questions: list[Question], *, method: str = "hybrid", k: int = 5, log=None) -> AnswerReport:
    results = []
    for q in questions:
        try:
            rec = answerer.ask(q.question, method=method, k=k)
            res = judge_answer(q, rec)
        except Exception as exc:  # eine kaputte Antwort soll die anderen 40 nicht kosten
            res = AnswerResult(question_id=q.id, answerable_expected=q.answerable, answerable_said=False, grounded=False,
                               citations=0, citations_ok=0, gold_cited=False, answer=f"ERROR: {exc}")
        results.append(res)
        if log:
            log(f"  q{q.id:<3} expected={'yes' if q.answerable else 'no':3} said={'yes' if res.answerable_said else 'no':3} "
                f"grounded={res.grounded!s:5} cites {res.citations_ok}/{res.citations} gold={res.gold_cited}")
    ans = [r for r in results if r.answerable_expected]
    una = [r for r in results if not r.answerable_expected]
    total_cites = sum(r.citations for r in ans)
    return AnswerReport(
        method=method, k=k, model=getattr(answerer, "model", ""),
        n_answerable=len(ans),
        grounded_rate=round(sum(r.grounded for r in ans) / len(ans), 3) if ans else 0.0,
        citation_precision=round(sum(r.citations_ok for r in ans) / total_cites, 3) if total_cites else 0.0,
        gold_cited_rate=round(sum(r.gold_cited for r in ans) / len(ans), 3) if ans else 0.0,
        n_unanswerable=len(una),
        refusal_rate=round(sum((not r.answerable_said) or r.citations == 0 for r in una) / len(una), 3) if una else 0.0,
        results=results,
    )


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def format_retrieval(reports: list[RetrievalReport]) -> str:
    lines = [f"Retrieval ({reports[0].config}, {reports[0].n} answerable questions, top-{reports[0].k})",
             f"{'method':8} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}  misses"]
    for r in reports:
        ra = r.recall_at
        lines.append(f"{r.method:8} {ra.get(1, 0):6.0%} {ra.get(3, 0):6.0%} {ra.get(5, 0):6.0%} {r.mrr:6.3f}  {r.misses or '-'}")
    return "\n".join(lines)


def format_answers(rep: AnswerReport) -> str:
    lines = [f"Answers ({rep.method}, top-{rep.k}, {rep.model})",
             f"  grounded            {rep.grounded_rate:.0%}  ({rep.n_answerable} answerable questions)",
             f"  citation precision  {rep.citation_precision:.0%}",
             f"  gold passage cited  {rep.gold_cited_rate:.0%}",
             f"  refusal when unanswerable  {rep.refusal_rate:.0%}  ({rep.n_unanswerable} questions)"]
    bad = [r for r in rep.results if (r.answerable_expected and not r.grounded) or (not r.answerable_expected and r.grounded)]
    if bad:
        lines.append("  look at: " + ", ".join(f"q{r.question_id}" for r in bad))
    return "\n".join(lines)


def save_report(path: Path, **parts) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for name, value in parts.items():
        if isinstance(value, list):
            payload[name] = [asdict(v) for v in value]
        elif value is not None:
            payload[name] = asdict(value)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

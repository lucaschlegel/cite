from pathlib import Path

from cite.answer import Answer, AnswerRecord, check_citations
from cite.evaluate import (Question, evaluate_answers, evaluate_retrieval, first_hit_rank, format_answers,
                           format_retrieval, judge_answer, load_questions, save_report)
from cite.search import Hit
from cite.store import Chunk


def chunk(i, doc, text):
    return Chunk(doc=doc, section=None, page_start=1, page_end=1, text=text, id=i)


Q_YES = Question(1, "e9", "2.2.2", "How many primary variables?", "only one primary variable", True)
Q_NO = Question(2, "", "", "Maximum fee?", "", False)
HIT_GOLD = Hit(chunk(1, "e9", "There should generally be\nonly one  primary variable."), 1.0, 1, "x")
HIT_OTHER = Hit(chunk(2, "e9", "Sample size should be large enough."), 0.5, 2, "x")
HIT_WRONG_DOC = Hit(chunk(3, "e3", "only one primary variable"), 0.4, 3, "x")


def test_match_ignores_whitespace_but_requires_document():
    assert Q_YES.matches(HIT_GOLD)
    assert not Q_YES.matches(HIT_OTHER)
    assert not Q_YES.matches(HIT_WRONG_DOC)
    assert not Q_NO.matches(HIT_GOLD)
    assert first_hit_rank(Q_YES, [HIT_OTHER, HIT_GOLD]) == 1  # Rang kommt aus dem Hit


def test_multiple_gold_phrases():
    q = Question(9, "e9", "x", "q", "nothing here | only one primary variable", True)
    assert q.matches(HIT_GOLD) and not q.matches(HIT_OTHER)


def test_questions_csv_loads_and_is_consistent():
    qs = load_questions(Path("data/questions.csv"))
    assert len(qs) >= 36
    assert all(q.gold_phrase and q.doc for q in qs if q.answerable)
    assert all(not q.gold_phrase for q in qs if not q.answerable)
    assert len({q.id for q in qs}) == len(qs)


class FakeSearcher:
    config = "cX"

    def search(self, query, method="hybrid", k=5):
        if method == "bm25":
            return [HIT_GOLD, HIT_OTHER]
        return [HIT_OTHER, Hit(HIT_GOLD.chunk, 0.9, 2, method)]


def test_retrieval_metrics():
    reports, results = evaluate_retrieval(FakeSearcher(), [Q_YES, Q_NO], methods=("bm25", "embed"), k=5)
    bm25, embed = reports
    assert bm25.n == 1 and bm25.recall_at[1] == 1.0 and bm25.mrr == 1.0 and bm25.misses == []
    assert embed.recall_at[1] == 0.0 and embed.recall_at[3] == 1.0 and embed.mrr == 0.5
    assert len(results) == 2  # nur beantwortbare Fragen, je Methode eine Zeile
    assert "bm25" in format_retrieval(reports)


class FakeAnswerer:
    model = "test"

    def __init__(self, searcher):
        self.searcher = searcher

    def ask(self, question, method="hybrid", k=5):
        hits = [HIT_GOLD, HIT_OTHER]
        if "fee" in question:
            answer = Answer(answerable=False, answer="Not in the documents.", citations=[])
        else:
            answer = Answer(answerable=True, answer="One.", citations=[{"passage": 1, "quote": "only one primary variable"}])
        return AnswerRecord(question=question, answer=answer, hits=hits, checked=check_citations(answer, hits), model="test")


def test_answer_metrics_and_refusal():
    rep = evaluate_answers(FakeAnswerer(FakeSearcher()), [Q_YES, Q_NO])
    assert rep.grounded_rate == 1.0 and rep.citation_precision == 1.0 and rep.gold_cited_rate == 1.0
    assert rep.refusal_rate == 1.0 and rep.n_unanswerable == 1
    assert "refusal when unanswerable  100%" in format_answers(rep)


def test_judge_flags_fabricated_citation_as_not_gold():
    answer = Answer(answerable=True, answer="x", citations=[{"passage": 1, "quote": "made up words here"}])
    hits = [HIT_GOLD]
    rec = AnswerRecord(question="q", answer=answer, hits=hits, checked=check_citations(answer, hits))
    res = judge_answer(Q_YES, rec)
    assert not res.grounded and not res.gold_cited and res.citations == 1 and res.citations_ok == 0


def test_save_report_writes_json(tmp_path):
    reports, results = evaluate_retrieval(FakeSearcher(), [Q_YES], methods=("bm25",), k=3)
    out = save_report(tmp_path / "r.json", retrieval=reports, retrieval_results=results, answers=None)
    text = out.read_text(encoding="utf-8")
    assert '"method": "bm25"' in text and '"timestamp"' in text and "answers" not in text


def test_evaluate_answers_survives_a_failing_question():
    class Boom(FakeAnswerer):
        def ask(self, question, method="hybrid", k=5):
            if "fee" in question:
                raise RuntimeError("api down")
            return super().ask(question, method, k)

    rep = evaluate_answers(Boom(FakeSearcher()), [Q_YES, Q_NO])
    assert rep.grounded_rate == 1.0
    assert rep.results[1].answer.startswith("ERROR") and rep.refusal_rate == 1.0

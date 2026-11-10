from types import SimpleNamespace

from cite.answer import TOOL, Answer, Answerer, build_user_message, check_citations, format_record
from cite.search import Hit
from cite.store import Chunk


def hits():
    a = Chunk(doc="e9", section="2.2.2 Primary and Secondary Variables", page_start=9, page_end=9, id=11,
              text="The primary variable should be the variable capable of providing\nthe most clinically relevant evidence. "
                   "There should generally be only one primary variable.")
    b = Chunk(doc="e9", section="3.5 Sample Size", page_start=20, page_end=20, id=12,
              text="The number of subjects in a clinical trial should always be large enough to provide a reliable answer.")
    return [Hit(a, 0.9, 1, "hybrid"), Hit(b, 0.8, 2, "hybrid")]


class FakeMessages:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="record_answer", input=self.payload)])


class FakeClient:
    def __init__(self, payload):
        self.messages = FakeMessages(payload)


class FakeSearcher:
    def search(self, query, method="hybrid", k=5):
        return hits()


def test_user_message_numbers_passages():
    msg = build_user_message("How many primary variables?", hits())
    assert "QUESTION: How many primary variables?" in msg
    assert "[1] [e9, 2.2.2 Primary and Secondary Variables, p. 9]" in msg
    assert "[2] [e9, 3.5 Sample Size, p. 20]" in msg


def test_tool_schema_requires_citations_list():
    props = TOOL["input_schema"]["properties"]
    assert set(props) == {"answerable", "answer", "citations"}


def test_grounded_answer_with_verbatim_quote_across_linebreak():
    payload = {"answerable": True, "answer": "Generally only one primary variable.",
               "citations": [{"passage": 1, "quote": "capable of providing the most clinically relevant evidence"}]}
    answerer = Answerer(FakeSearcher(), model="test", client=FakeClient(payload))
    rec = answerer.ask("How many primary variables?")
    assert rec.grounded and rec.checked[0].ok
    call = answerer.client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "record_answer"}
    assert "ok [1] [e9, 2.2.2" in format_record(rec)


def test_fabricated_quote_and_bad_passage_are_flagged():
    payload = {"answerable": True, "answer": "Three primary variables are allowed.",
               "citations": [{"passage": 1, "quote": "three primary variables are allowed"},
                             {"passage": 7, "quote": "anything"}]}
    rec = Answerer(FakeSearcher(), model="test", client=FakeClient(payload)).ask("q")
    assert not rec.grounded
    assert rec.checked[0].valid_passage and not rec.checked[0].quote_found
    assert not rec.checked[1].valid_passage
    out = format_record(rec)
    assert "quote not found" in out and "passage 7 does not exist" in out


def test_not_answerable_is_not_grounded():
    payload = {"answerable": False, "answer": "The passages do not say who signs the report.", "citations": []}
    rec = Answerer(FakeSearcher(), model="test", client=FakeClient(payload)).ask("Who signs?")
    assert not rec.grounded and "(not answerable" in format_record(rec)


def test_check_citations_normalises_quotes_and_whitespace():
    answer = Answer(answerable=True, answer="x", citations=[{"passage": 2, "quote": "ALWAYS   be large enough"}])
    assert check_citations(answer, hits())[0].ok


def test_pdf_artifacts_and_small_deviations_are_tolerated():
    from cite.answer import locate_quote

    text = "Where signatures of the principal or c oordinating investigators are required by\nregulatory authorities, these should be included in appendix 16.1.5"
    found, sim = locate_quote("principal or coordinating investigators are required by regulatory authorities", text)
    assert found and sim == 1.0  # Leerzeichen im Wort stört nicht
    found, sim = locate_quote("principal or coordinating investigators are needed by regulatory authorities", text)
    assert found and 0.85 <= sim < 1.0  # ein Wort anders: noch akzeptiert, aber als fuzzy markiert
    found, sim = locate_quote("the sponsor must sign the report within thirty days", text)
    assert not found and sim < 0.85


def test_missing_answerable_defaults_to_true():
    payload = {"answer": "Only one.", "citations": [{"passage": 1, "quote": "only one primary variable"}]}
    rec = Answerer(FakeSearcher(), model="test", client=FakeClient(payload)).ask("q")
    assert rec.answer.answerable and rec.grounded


def test_citations_given_as_json_string_are_parsed():
    a = Answer.model_validate({"answer": "x", "citations": '[{"passage": 1, "quote": "only one primary variable"}]'})
    assert len(a.citations) == 1 and a.citations[0].passage == 1


def test_truncated_citation_string_keeps_the_readable_items():
    broken = '[{"passage": 1, "quote": "only one primary variable"}, {"passage": 4, "quote": "confirmatory trials", "}]'
    a = Answer.model_validate({"answer": "x", "citations": broken})
    assert [c.passage for c in a.citations] == [1]


def test_stringified_citations_are_parsed():
    payload = {"answerable": True, "answer": "Only one.",
               "citations": '[{"passage": 1, "quote": "only one primary variable"}]'}
    rec = Answerer(FakeSearcher(), model="test", client=FakeClient(payload)).ask("q")
    assert rec.grounded and rec.checked[0].ok

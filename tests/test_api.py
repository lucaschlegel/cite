from pathlib import Path

from fastapi.testclient import TestClient

from cite.api import create_app
from cite.ingest import chunk_pages, read_pages
from cite.search import Searcher
from cite.store import connect, replace_chunks, upsert_document
from tests.test_answer import FakeClient
from tests.test_search import FakeEmbedder

FIX = Path(__file__).parent / "fixtures"


def make_client(tmp_path):
    con = connect(tmp_path / "t.sqlite")
    chunks = chunk_pages("sample", read_pages(FIX / "sample.pdf"), max_chars=500)
    upsert_document(con, "sample", "Sample", None, 3, "x" * 64)
    replace_chunks(con, "sample", "c500", chunks)
    searcher = Searcher(con, "c500", embedder=FakeEmbedder(), model_name="fake-4d")
    from cite.answer import Answerer

    payload = {"answerable": True, "answer": "Thirty widgets.", "citations": [{"passage": 1, "quote": "thirty widgets"}]}
    answerer = Answerer(searcher, model="test", client=FakeClient(payload))
    return TestClient(create_app(searcher=searcher, answerer=answerer))


def test_health_and_search(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    hits = client.get("/search", params={"q": "minimum number of widgets per batch", "method": "bm25", "k": 2}).json()
    assert len(hits) == 2 and "thirty widgets" in hits[0]["text"] and hits[0]["rank"] == 1


def test_search_without_vectors_returns_503(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/search", params={"q": "widgets", "method": "embed"}).status_code == 503


def test_ask_returns_checked_citations(tmp_path):
    client = make_client(tmp_path)
    body = client.post("/ask", json={"question": "How many widgets per batch?", "method": "bm25", "k": 3}).json()
    assert body["grounded"] is True and body["citations"][0]["ok"] is True
    assert body["passages"][0]["ref"].startswith("[sample")
    assert client.post("/ask", json={"question": "x"}).status_code == 422  # zu kurz

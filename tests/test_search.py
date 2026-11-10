from pathlib import Path

import numpy as np

from cite.ingest import chunk_pages, read_pages
from cite.search import BM25Index, EmbeddingIndex, Hit, Searcher, reciprocal_rank_fusion, tokenize
from cite.store import connect, load_vectors, replace_chunks, upsert_document

FIX = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministische 'Embeddings': Zählt vier Stichwörter. Reicht, um Suche und Speicherung zu prüfen."""

    model_name = "fake-4d"
    WORDS = ("sample", "report", "deviation", "glossary")

    def _vec(self, text: str) -> np.ndarray:
        t = text.lower()
        return np.array([t.count(w) for w in self.WORDS], dtype=np.float32) + 1e-3

    def embed_passages(self, texts):
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def fixture_db(tmp_path):
    con = connect(tmp_path / "t.sqlite")
    chunks = chunk_pages("sample", read_pages(FIX / "sample.pdf"), max_chars=500)
    upsert_document(con, "sample", "Sample", None, 3, "x" * 64)
    replace_chunks(con, "sample", "c500", chunks)
    return con


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("The widgets of the batch are tested") == ["widget", "batch", "tested"]
    assert "class" in tokenize("class")  # "ss" bleibt


def test_bm25_finds_sample_size_section(tmp_path):
    con = fixture_db(tmp_path)
    searcher = Searcher(con, "c500", embedder=FakeEmbedder(), model_name="fake-4d")
    hits = searcher.search("minimum number of widgets per batch", method="bm25", k=3)
    assert hits and "thirty widgets" in hits[0].chunk.text
    assert hits[0].rank == 1 and hits[0].method == "bm25"


def test_embedding_index_build_stores_vectors_and_searches(tmp_path):
    con = fixture_db(tmp_path)
    emb = FakeEmbedder()
    index = EmbeddingIndex.build(con, "c500", emb, log=lambda m: None)
    n = len(index.chunks)
    assert len(load_vectors(con, "c500", "fake-4d")) == n
    # zweiter Build rechnet nichts neu
    again = EmbeddingIndex.build(con, "c500", emb, log=lambda m: (_ for _ in ()).throw(AssertionError("recomputed")))
    assert len(again.chunks) == n
    hits = index.search("glossary", k=2)
    assert hits[0].chunk.section == "ANNEX 1"


def test_hybrid_fuses_both_lists(tmp_path):
    con = fixture_db(tmp_path)
    emb = FakeEmbedder()
    EmbeddingIndex.build(con, "c500", emb, log=lambda m: None)
    searcher = Searcher(con, "c500", embedder=emb, model_name="fake-4d")
    hits = searcher.search("report archiving", method="hybrid", k=3)
    assert hits[0].method == "hybrid"
    assert {h.chunk.section for h in hits} & {"3 REPORTING", "3.1 Archiving"}


def test_rrf_prefers_items_high_in_both_lists():
    from cite.store import Chunk

    def c(i):
        return Chunk(doc="d", section=None, page_start=1, page_end=1, text=str(i), id=i)

    a = [Hit(c(1), 9, 1, "bm25"), Hit(c(2), 8, 2, "bm25"), Hit(c(3), 7, 3, "bm25")]
    b = [Hit(c(2), 0.9, 1, "embed"), Hit(c(4), 0.8, 2, "embed"), Hit(c(1), 0.7, 3, "embed")]
    fused = reciprocal_rank_fusion(a, b, k=4)
    assert [h.chunk.id for h in fused][:2] == [2, 1] or [h.chunk.id for h in fused][:2] == [1, 2]
    assert fused[0].chunk.id in (1, 2) and fused[-1].chunk.id in (3, 4)

from pathlib import Path

from cite.ingest import Page, chunk_pages, find_heading, ingest_document, normalize, read_pages
from cite.store import connect, load_chunks, replace_chunks, stats, upsert_document

FIX = Path(__file__).parent / "fixtures"


def test_find_heading_accepts_numbered_roman_and_annex():
    assert find_heading("1.1 Scope") == "1.1 Scope"
    assert find_heading("2.2.2 Primary and Secondary Variables") == "2.2.2 Primary and Secondary Variables"
    assert find_heading("II. PRINCIPLES OF ICH GCP") == "II PRINCIPLES OF ICH GCP"
    assert find_heading("ANNEX 1") == "ANNEX 1"


def test_find_heading_rejects_sentences_numbers_and_dates():
    assert find_heading("5 percent of the widgets failed.") is None
    assert find_heading("2 Participants were enrolled.") is None  # endet mit Punkt
    assert find_heading("12") is None
    assert find_heading("27 April 1995") is None
    assert find_heading("The sample size should be large enough.") is None


def test_normalize_joins_hyphenated_words_and_collapses_spaces():
    assert normalize("thera-\npeutic   agent") == "therapeutic agent"


def test_chunk_pages_on_fixture_pdf():
    pages = read_pages(FIX / "sample.pdf")
    assert len(pages) == 3
    chunks = chunk_pages("sample", pages, max_chars=500)
    sections = [c.section for c in chunks]
    assert "2.1 Sample Size" in sections
    assert "3 REPORTING" in sections
    assert all(len(c.text) <= 500 * 1.25 + 200 for c in chunks)  # Zeile ist atomar, daher kleiner Puffer
    sample_size = next(c for c in chunks if c.section == "2.1 Sample Size")
    assert "thirty widgets" in sample_size.text
    assert sample_size.page_start == 1
    assert chunks[-1].page_end == 3 and chunks[-1].section == "ANNEX 1"
    assert chunks[0].ref == "[sample, 1 INTRODUCTION, p. 1]"


def test_short_sections_are_merged_until_min_chars():
    pages = read_pages(FIX / "sample.pdf")
    big = chunk_pages("sample", pages, max_chars=4000)  # alles passt in wenige Stücke
    assert len(big) <= 3
    assert big[0].section == "1 INTRODUCTION"  # trägt die erste Überschrift


def test_toc_lines_and_page_numbers_are_skipped():
    page = Page(1, "3.8.1 Notification to Authority ............ 26\n12\nReal content line that is long enough to keep.\n"
                   "Another real line so the chunk is not a crumb.")
    chunks = chunk_pages("x", [page], max_chars=500)
    assert len(chunks) == 1
    assert "Notification" not in chunks[0].text
    assert chunks[0].section is None


def test_ingest_document_and_store_roundtrip(tmp_path):
    meta, chunks = ingest_document({"name": "sample", "title": "Sample", "path": str(FIX / "sample.pdf")}, max_chars=500)
    assert meta["pages"] == 3 and len(meta["sha256"]) == 64
    con = connect(tmp_path / "t.sqlite")
    upsert_document(con, **meta)
    assert replace_chunks(con, "sample", "c500", chunks) == len(chunks)
    assert replace_chunks(con, "sample", "c500", chunks) == len(chunks)  # zweiter Lauf ersetzt, verdoppelt nicht
    loaded = load_chunks(con, "c500", "sample")
    assert [c.text for c in loaded] == [c.text for c in chunks]
    assert loaded[0].id is not None
    row = stats(con)[0]
    assert row["chunks"] == len(chunks) and row["pages"] == 3

# cite

Question answering over a set of PDF documents, with every answer tied to the passage it came from. A small end-to-end retrieval-augmented generation (RAG) application in Python: ingest PDFs into section-aware chunks, retrieve with keyword search and embeddings, answer with an LLM that must cite its sources and must say so when the documents do not contain the answer, and measure retrieval and answer quality against a hand-labelled question set. Read-only FastAPI on top.

The example corpus are five public ICH guidelines on clinical trials (ich.org), chosen because they are public, well structured and long enough (about 220 pages) to make retrieval non-trivial. The pipeline does not care what the documents are about.

*Deutsche Kurzfassung unten.*

## Pipeline

1. **Ingest** (done): download the PDFs, extract text per page, cut it into chunks that respect numbered headings (a heading starts a new chunk once the current one has reached a minimum size; short sections are merged), skip table-of-contents lines, and store documents and chunks in SQLite. Chunk size is a parameter, so several chunkings can live side by side for the evaluation.
2. **Index and search**: BM25 keyword search as the baseline, dense retrieval with a small local embedding model, and a hybrid of both.
3. **Answer**: the model gets the question and the top chunks and answers with explicit references `[document, section, page]`; if the chunks do not contain the answer it says so instead of guessing.
4. **Evaluate**: a hand-labelled question set (question, gold document and section). Retrieval: recall@k and MRR, compared across BM25 / embeddings / hybrid and across chunk sizes. Answers: does every reference point to a retrieved chunk, and does the answer stay within the retrieved text.
5. **Serve**: FastAPI with `/search` and `/ask`; Dockerfile.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                         # fixture PDF only, no network
python -m cite ingest          # downloads the PDFs into data/ (about 2 MB), writes data/cite.sqlite
python -m cite stats
python -m cite show 42         # look at one chunk
```

`python -m cite ingest --max-chars 600` adds a second chunking (`c600`) next to the default `c1200`.

## Kurzfassung

Fragen über eine Sammlung von PDF-Dokumenten beantworten, jede Antwort mit Beleg auf die Stelle, aus der sie stammt. Eine kleine RAG-Anwendung von Anfang bis Ende: PDFs in abschnittstreue Stücke zerlegen, mit Stichwortsuche und Embeddings suchen, mit einem Sprachmodell antworten, das zitieren muss und „steht nicht in den Dokumenten" sagen darf, und das Ganze gegen einen von Hand gesetzten Fragenkatalog messen. Beispielkorpus sind fünf öffentliche ICH-Richtlinien zu klinischen Studien; der Pipeline ist der Inhalt egal.

Stand: Schritt 1 (Einlesen, Zerlegen, SQLite) fertig, 7 Tests.

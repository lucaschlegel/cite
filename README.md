# cite

Question answering over a set of PDF documents, with every answer tied to the passage it came from. A small end-to-end retrieval-augmented generation (RAG) application in Python: ingest PDFs into section-aware chunks, retrieve with keyword search and embeddings, answer with an LLM that must cite its sources and must say so when the documents do not contain the answer, and measure retrieval and answer quality against a hand-labelled question set. Read-only FastAPI on top.

The example corpus are five public ICH guidelines on clinical trials (ich.org), chosen because they are public, well structured and long enough (about 220 pages) to make retrieval non-trivial. The pipeline does not care what the documents are about.

*Deutsche Kurzfassung unten.*

## Pipeline

1. **Ingest** (done): download the PDFs, extract text per page, cut it into chunks that respect numbered headings (a heading starts a new chunk once the current one has reached a minimum size; short sections are merged), skip table-of-contents lines, and store documents and chunks in SQLite. Chunk size is a parameter, so several chunkings can live side by side for the evaluation.
2. **Index and search** (done): BM25 keyword search as the baseline (no model, no preparation), dense retrieval with a small local embedding model (`BAAI/bge-small-en-v1.5` via fastembed, runs on the CPU, vectors stored in SQLite next to the chunks), and a hybrid that merges both rankings with reciprocal rank fusion.
3. **Answer** (done): the model (`claude-haiku-5-5` by default) gets the question and the top chunks, numbered, and must call a tool whose schema is a Pydantic model: `answerable`, `answer`, and `citations`, each a passage number plus a verbatim quote. Two checks then run without any model: does the passage number exist, and does the quote appear in that chunk. The comparison ignores whitespace and punctuation, because pypdf leaves spaces inside words ("c oordinating", "mark eting") that the model silently repairs when quoting; a quote still counts if at least 85 % of it is found in order (difflib matching blocks), and is then marked as near-verbatim. In the first live run, one of three citations failed the strict check for exactly this reason. An answer counts as grounded only if it is answerable, has at least one citation and every citation passes both checks. If the chunks do not contain the answer the model says so instead of guessing.
4. **Evaluate** (done): `data/questions.csv`, 36 hand-written questions with a gold passage each (a short verbatim excerpt of the passage that answers the question, so the labels do not depend on chunk boundaries) plus 5 questions the documents cannot answer. Retrieval: recall@1/3/5 and MRR per method and per chunking, no model involved, free. Answers (`--answers`, a few cents): share of grounded answers, citation precision, share of answers that cite the gold passage, and refusal rate on the unanswerable questions.
5. **Serve** (done): FastAPI with `/health`, `/search` and `/ask` (answer, verified citations, passages as JSON); Dockerfile, `data/` mounted as a volume.

## Results

Retrieval, 36 answerable questions, chunking `c1200` (about 1200 characters per chunk, 667 chunks):

| method | R@1 | R@3 | R@5 | MRR |
| --- | --- | --- | --- | --- |
| BM25 | 69 % | 81 % | 86 % | 0.771 |
| embeddings (bge-small) | 53 % | 78 % | 86 % | 0.679 |
| hybrid (RRF) | 72 % | 89 % | 94 % | 0.806 |

Keyword search is a strong baseline on this corpus: the questions reuse the guidelines' own vocabulary, and BM25 puts the right passage first in two of three cases. The small embedding model alone is weaker at rank 1 but finds different passages, so fusing the two lists helps at every cutoff. The two remaining misses are instructive: one question names the document ("according to ICH E6(R3)"), which pulls the cover page to the top for every method; the other asks what "quality by design" identifies, and the corpus has a dozen passages about quality by design in two documents, so the specific one lands on rank 7 to 15.

Smaller chunks (`c600`, about 600 characters, 1180 chunks) are not better here:

| method | R@1 | R@3 | R@5 | MRR |
| --- | --- | --- | --- | --- |
| BM25 | 61 % | 78 % | 86 % | 0.731 |
| embeddings (bge-small) | 47 % | 72 % | 86 % | 0.619 |
| hybrid (RRF) | 64 % | 86 % | 94 % | 0.752 |

Recall at 5 is identical, but the right passage lands on rank 1 less often: with half-size chunks a section splits into two or three pieces that all look similar, and they share the score. For this corpus, section-aware chunks of about 1200 characters are the better unit.

Answers (hybrid, top-5 passages, `claude-haiku-5-5`, 41 questions, about one cent):

| metric | value |
| --- | --- |
| grounded answers (answerable, at least one citation, every citation verified) | 92 % (33/36) |
| citation precision (verified citations / all citations) | 100 % |
| answers that cite the gold passage | 92 % |
| refusals on the 5 unanswerable questions | 100 % (5/5) |

No fabricated citation in 36 answers, and the model declined all five questions the documents do not answer. The three answers that are not grounded are the two retrieval misses (the right passage was not among the five) plus one answer with no citation at all. In other words, on this corpus the answer step is only as good as retrieval, which is where the remaining work is.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                         # fixture PDF only, no network
python -m cite ingest          # downloads the PDFs into data/ (about 2 MB), writes data/cite.sqlite
python -m cite stats
python -m cite show 42         # look at one chunk
python -m cite index           # embeds all chunks once (~130 MB model download, a few minutes on CPU)
python -m cite search "How many primary variables should a confirmatory trial have?" --method bm25
python -m cite search "How many primary variables should a confirmatory trial have?" --method embed
python -m cite search "How many primary variables should a confirmatory trial have?"   # hybrid
export ANTHROPIC_API_KEY=sk-ant-...
python -m cite ask "How many primary variables should a confirmatory trial have?"       # a fraction of a cent
python -m cite ask "Who signs the clinical study report?"                                 # probably not answerable
python -m cite evaluate                    # retrieval metrics, free
python -m cite evaluate --answers          # plus answer metrics, 41 model calls, a few cents
python -m cite serve                       # http://127.0.0.1:8000/docs
```

Docker: `docker build -t cite . && docker run -v $PWD/data:/app/data -p 8000:8000 -e ANTHROPIC_API_KEY cite`

Lessons from the real runs, in the order they happened: `pypdf` leaves spaces inside words, so quote checks must ignore whitespace; a forced tool call guarantees valid JSON, not that every field is present (`answerable` went missing, `citations` arrived as a string, once truncated), so validate and repair after the call; re-ingesting documents silently dropped their vectors, so the index now warns; labels need checking too (one gold passage pointed at the glossary while the chapter definition was equally right).

`python -m cite ingest --max-chars 600` adds a second chunking (`c600`) next to the default `c1200`; `index` and `search` take `--config c600` to use it.

## Kurzfassung

Fragen über eine Sammlung von PDF-Dokumenten beantworten, jede Antwort mit Beleg auf die Stelle, aus der sie stammt. Eine kleine RAG-Anwendung von Anfang bis Ende: PDFs in abschnittstreue Stücke zerlegen, mit Stichwortsuche und Embeddings suchen, mit einem Sprachmodell antworten, das zitieren muss und „steht nicht in den Dokumenten" sagen darf, und das Ganze gegen einen von Hand gesetzten Fragenkatalog messen. Beispielkorpus sind fünf öffentliche ICH-Richtlinien zu klinischen Studien; der Pipeline ist der Inhalt egal.

Stand: alle fünf Schritte fertig (Einlesen und Zerlegen; Stichwortsuche, Embeddings, Hybrid; Antwort mit nachgeprüften Belegen; Messung gegen 41 Fragen; FastAPI und Dockerfile), 35 Tests. Suche: Hybrid findet die richtige Stelle in 94 Prozent der Fälle unter den ersten fünf. Antworten: 92 Prozent belegt, kein einziger erfundener Beleg, alle fünf unbeantwortbaren Fragen abgelehnt.

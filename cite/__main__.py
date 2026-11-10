"""Kommandozeile.

  python -m cite ingest [--max-chars N]     Schritt 1: PDFs laden, schneiden, in SQLite ablegen
  python -m cite stats                      Was liegt in der Datenbank?
  python -m cite show <chunk-id>            Ein Stück ansehen
  python -m cite index [--config c1200]     Schritt 2: Vektoren für alle Stücke rechnen (einmalig, lokal)
  python -m cite search "frage" [--method]  Schritt 2: bm25 | embed | hybrid
  python -m cite ask "frage" [--method -k]  Schritt 3: Antwort mit geprüften Belegen (braucht ANTHROPIC_API_KEY)
  python -m cite evaluate [--answers]       Schritt 4: Suche gegen data/questions.csv messen, mit --answers auch die Antworten
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cite.answer import Answerer, format_record
from cite.evaluate import (QUESTIONS_FILE, evaluate_answers, evaluate_retrieval, format_answers, format_retrieval,
                           load_questions, save_report)
from cite.ingest import ingest_document, load_sources
from cite.search import DEFAULT_MODEL, Embedder, EmbeddingIndex, Searcher
from cite.store import DB_FILE, connect, replace_chunks, stats, upsert_document


def cmd_ingest(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    config = f"c{args.max_chars}"
    sources = load_sources(args.sources)
    if args.only:
        sources = [s for s in sources if s["name"] in args.only]
    for source in sources:
        try:
            meta, chunks = ingest_document(source, data_dir=Path(args.data_dir), max_chars=args.max_chars)
        except Exception as exc:  # ein Dokument soll den Lauf nicht stoppen
            print(f"[warn] {source['name']}: {exc}", file=sys.stderr)
            continue
        upsert_document(con, **meta)
        n = replace_chunks(con, meta["name"], config, chunks)
        print(f"[ok] {meta['name']}: {meta['pages']} pages, {n} chunks ({config})", file=sys.stderr)
    return cmd_stats(args)


def cmd_stats(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    rows = stats(con)
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        for r in rows:
            print(r)
        return 0
    table = Table(title=f"Dokumente und Stücke ({args.db})")
    for col in ("Dokument", "Titel", "Seiten", "Config", "Stücke", "Ø Zeichen"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["name"], r["title"], str(r["pages"]), r["config"] or "", str(r["chunks"]), str(r["avg_chars"] or ""))
    Console().print(table)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    row = con.execute("SELECT * FROM chunks WHERE id = ?", (args.chunk_id,)).fetchone()
    if row is None:
        print(f"no chunk {args.chunk_id}", file=sys.stderr)
        return 1
    print(f"{row['doc']} | {row['section']} | pages {row['page_start']}-{row['page_end']} | {row['config']}\n")
    print(row["text"])
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    embedder = Embedder(args.model)
    index = EmbeddingIndex.build(con, args.config, embedder, model_name=args.model,
                                 log=lambda m: print(m, file=sys.stderr))
    print(f"[ok] {len(index.chunks)} vectors ready ({args.config}, {args.model})", file=sys.stderr)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    searcher = Searcher(con, args.config, model_name=args.model)
    hits = searcher.search(args.query, method=args.method, k=args.k)
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        for h in hits:
            print(f"{h.rank:2} {h.score:7.3f} {h.chunk.ref} {h.snippet}")
        return 0
    from rich.markup import escape

    table = Table(title=escape(f"{args.method}: {args.query}"))
    for col in ("#", "Score", "Beleg", "Anfang"):
        table.add_column(col, overflow="fold")
    for h in hits:
        table.add_row(str(h.rank), f"{h.score:.3f}", escape(h.chunk.ref), escape(h.snippet))
    Console().print(table)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    searcher = Searcher(con, args.config, model_name=args.embed_model)
    answerer = Answerer(searcher, model=args.model)
    rec = answerer.ask(args.question, method=args.method, k=args.k)
    print(format_record(rec))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    questions = load_questions(Path(args.questions))
    searcher = Searcher(con, args.config, model_name=args.embed_model)
    log = (lambda m: print(m, file=sys.stderr)) if args.verbose else None
    reports, results = evaluate_retrieval(searcher, questions, methods=args.methods, k=args.k, log=log)
    print(format_retrieval(reports))
    answer_report = None
    if args.answers:
        answerer = Answerer(searcher, model=args.model)
        answer_report = evaluate_answers(answerer, questions, method=args.answer_method, k=args.answer_k, log=log)
        print()
        print(format_answers(answer_report))
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H%M")
    out = save_report(Path(args.db).parent / f"eval_{args.config}_{stamp}.json",
                      retrieval=reports, retrieval_results=results, answers=answer_report)
    print(f"\n[ok] saved {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cite")
    parser.add_argument("--db", default=str(DB_FILE))
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="PDFs laden, schneiden, speichern")
    ingest.add_argument("--sources", default="sources.yaml")
    ingest.add_argument("--data-dir", default="data")
    ingest.add_argument("--max-chars", type=int, default=1200, help="Zielgröße eines Stücks in Zeichen")
    ingest.add_argument("--only", nargs="*", help="nur diese Dokumentnamen")
    ingest.set_defaults(func=cmd_ingest)

    st = sub.add_parser("stats", help="Inhalt der Datenbank")
    st.set_defaults(func=cmd_stats)

    show = sub.add_parser("show", help="ein Stück anzeigen")
    show.add_argument("chunk_id", type=int)
    show.set_defaults(func=cmd_show)

    index = sub.add_parser("index", help="Vektoren rechnen und speichern")
    index.add_argument("--config", default="c1200")
    index.add_argument("--model", default=DEFAULT_MODEL)
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="Stücke zu einer Frage finden")
    search.add_argument("query")
    search.add_argument("--method", choices=["bm25", "embed", "hybrid"], default="hybrid")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--config", default="c1200")
    search.add_argument("--model", default=DEFAULT_MODEL)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="Frage beantworten, Belege prüfen")
    ask.add_argument("question")
    ask.add_argument("--method", choices=["bm25", "embed", "hybrid"], default="hybrid")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--config", default="c1200")
    ask.add_argument("--model", default=None, help="Antwortmodell, Standard aus CITE_MODEL oder claude-haiku-5-5")
    ask.add_argument("--embed-model", default=DEFAULT_MODEL)
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("evaluate", help="Suche (und Antworten) gegen den Fragenkatalog messen")
    ev.add_argument("--questions", default=str(QUESTIONS_FILE))
    ev.add_argument("--config", default="c1200")
    ev.add_argument("--methods", nargs="+", default=["bm25", "embed", "hybrid"])
    ev.add_argument("-k", type=int, default=10, help="Suchtiefe für recall/MRR")
    ev.add_argument("--answers", action="store_true", help="auch Antworten erzeugen und prüfen (API-Kosten, Cent-Beträge)")
    ev.add_argument("--answer-method", default="hybrid")
    ev.add_argument("--answer-k", type=int, default=5)
    ev.add_argument("--model", default=None)
    ev.add_argument("--embed-model", default=DEFAULT_MODEL)
    ev.add_argument("--verbose", "-v", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

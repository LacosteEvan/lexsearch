"""lexsearch: retrieval and search-agent experiments on BSARD."""
from __future__ import annotations

import argparse

from dotenv import load_dotenv


def build(spec: str):
    """Retriever from a spec: bm25 | dense:<model> | hybrid:<spec>,<spec> | rerank:<model>@<spec>."""
    from . import data
    from .retrievers import BM25, Dense, Hybrid, Rerank

    articles = data.articles()
    if spec == "bm25":
        return BM25(list(articles.values()))
    if spec.startswith("dense:"):
        return Dense(spec[6:], list(articles.values()))
    if spec.startswith("hybrid:"):
        return Hybrid(*(build(s) for s in spec[7:].split(",")))
    if spec.startswith("rerank:"):
        model, first = spec[7:].split("@", 1)
        return Rerank(build(first), model, articles)
    raise SystemExit(f"unknown retriever {spec!r}")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="lexsearch")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieve", help="run retrievers on a split and save rankings + metrics")
    r.add_argument("specs", nargs="+")
    r.add_argument("--split", default="dev", choices=["fit", "dev", "test"])

    f = sub.add_parser("finetune", help="fine-tune an embedding model on the fit split")
    f.add_argument("model")
    f.add_argument("--out", required=True)
    f.add_argument("--synthetic", type=int, default=0, help="also train on this many synthetic questions")
    f.add_argument("--epochs", type=int, default=3)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--lr", type=float, default=2e-5)

    a = sub.add_parser("answer", help="answer questions by citing articles (paid: needs --max-cost)")
    a.add_argument("mode", choices=["closed_book", "one_shot", "agent"])
    a.add_argument("--model", default="haiku")
    a.add_argument("--retriever", help="retriever spec, for one_shot and agent")
    a.add_argument("--split", default="dev", choices=["fit", "dev", "test"])
    a.add_argument("--depth", type=int, default=20, help="articles shown to one_shot")
    a.add_argument("--limit", type=int)
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--max-cost", type=float, required=True)

    args = p.parse_args()
    if args.cmd == "retrieve":
        from .evaluate import line, run_retrieval

        for spec in args.specs:
            summary = run_retrieval(build(spec), args.split)
            print(line(summary), f"  {summary['seconds_per_query'] * 1000:.0f} ms/q", flush=True)
    elif args.cmd == "answer":
        from .runs import line, run, summarize

        retriever = build(args.retriever) if args.retriever else None
        path = run(args.mode, args.model, args.split, args.max_cost, retriever, args.depth, args.workers, args.limit)
        print(line(summarize(path, args.split)))
    elif args.cmd == "finetune":
        from pathlib import Path

        from .finetune import train

        train(args.model, Path(args.out), synthetic=args.synthetic, epochs=args.epochs, batch_size=args.batch_size,
              lr=args.lr)


if __name__ == "__main__":
    main()

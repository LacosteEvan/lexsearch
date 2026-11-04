"""Run an answering mode on a split, in parallel, resumable: one JSON line per question under results/answers/."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import data
from .answer import Answerer, Resolver
from .llm import BudgetExceeded, Chat
from .metrics import bootstrap_ci, set_scores

RESULTS = Path("results/answers")


def run_name(mode: str, model: str, retriever: str | None) -> str:
    parts = [mode, model] + ([retriever.replace("/", "__")] if retriever else [])
    return "__".join(parts)


def run(mode: str, model: str, split: str, max_cost: float, retriever=None, depth: int = 20, workers: int = 8,
        limit: int | None = None) -> Path:
    path = RESULTS / split / f"{run_name(mode, model, retriever.name if retriever else None)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.open()}
    qs = [q for q in data.questions(split)[:limit] if q.id not in done]
    answerer = Answerer(Chat(model, max_cost), retriever, depth)
    if mode == "closed_book":
        resolver = Resolver(answerer.corpus)
        fn = lambda q: answerer.closed_book(q, resolver)  # noqa: E731
    else:
        fn = getattr(answerer, mode)
    with path.open("a") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fn, q): q for q in qs}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except BudgetExceeded as e:
                print(f"stopped: {e}", flush=True)
                for other in futures:
                    other.cancel()
                break
            except Exception as e:  # one failed question must not lose the others
                print(f"{futures[fut].id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{n}/{len(qs)}  ${answerer.chat.spent:.3f}", flush=True)
    print(f"spent ${answerer.chat.spent:.3f}", flush=True)
    return path


def summarize(path: Path, split: str) -> dict:
    gold = {q.id: q.gold for q in data.questions(split)}
    rows = [json.loads(line) for line in path.open()]
    scores = {k: np.array([set_scores(frozenset(r["cited"]), gold[r["id"]])[k] for r in rows])
              for k in ("precision", "recall", "f1")}
    # Each gold article, for the decomposition: cited, shown but not cited (reading), never shown (search).
    cited = shown = missed = 0
    for r in rows:
        for a in gold[r["id"]]:
            if a in r["cited"]:
                cited += 1
            elif a in r["shown"]:
                shown += 1
            else:
                missed += 1
    total = cited + shown + missed
    out = {"run": path.stem, "n": len(rows), "errors": sum(bool(r["error"]) for r in rows)}
    for k, v in scores.items():
        out[k] = {"mean": float(v.mean()), "ci": list(bootstrap_ci(v))}
    out |= {
        "gold_cited": cited / total, "gold_shown_not_cited": shown / total, "gold_never_shown": missed / total,
        "articles_cited": float(np.mean([len(r["cited"]) for r in rows])),
        "cost_per_1k": 1000 * float(np.mean([r["cost_usd"] for r in rows])),
        "latency_s": float(np.median([r["latency_s"] for r in rows])),
        "steps": float(np.mean([r["steps"] for r in rows])),
        "searches": float(np.mean([len(r["queries"]) for r in rows])),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(out, indent=1))
    return out


def line(s: dict) -> str:
    return (f"{s['run']:<60} F1 {100 * s['f1']['mean']:.1f}  P {100 * s['precision']['mean']:.1f}  "
            f"R {100 * s['recall']['mean']:.1f}  cited {s['articles_cited']:.1f}  "
            f"gold: cited {100 * s['gold_cited']:.0f}% / shown {100 * s['gold_shown_not_cited']:.0f}% / "
            f"missed {100 * s['gold_never_shown']:.0f}%  ${s['cost_per_1k']:.2f}/1k  {s['latency_s']:.1f}s  "
            f"err {s['errors']}")

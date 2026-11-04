"""Run a retriever on a split, save its rankings and a summary under results/retrieval/<split>/."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import data
from .metrics import bootstrap_ci, ranking_scores

RESULTS = Path("results")


def run_path(split: str, name: str) -> Path:
    return RESULTS / "retrieval" / split / f"{name.replace('/', '__')}.jsonl"


def load_run(split: str, name: str) -> dict[str, list[str]]:
    with run_path(split, name).open() as f:
        return {r["id"]: r["ranked"] for r in map(json.loads, f)}


def summarize(run: dict[str, list[str]], split: str) -> dict:
    qs = data.questions(split)
    gold = {q.id: q.gold for q in qs}
    scores = ranking_scores(run, gold)
    seen = data.seen_articles()
    unseen = np.array([not (q.gold & seen) for q in qs])  # no answer article was ever a training label
    out = {"n": len(qs), "n_unseen": int(unseen.sum())}
    # Recall restricted to the answer articles that were (seen) or were never (unseen) a training label, averaged
    # over the questions that have some.
    for name, keep in (("seen", True), ("unseen", False)):
        parts = [(q.id, frozenset(a for a in q.gold if (a in seen) == keep)) for q in qs]
        parts = [(qid, g) for qid, g in parts if g]
        for k in (10, 100):
            out[f"recall@{k}_{name}_articles"] = float(np.mean([len(g.intersection(run[qid][:k])) / len(g)
                                                                for qid, g in parts]))
        out[f"n_{name}_articles"] = len(parts)
    for metric, values in scores.items():
        lo, hi = bootstrap_ci(values)
        out[metric] = {"mean": float(values.mean()), "ci": [lo, hi],
                       "unseen": float(values[unseen].mean()) if unseen.any() else None}
    return out


def run_retrieval(retriever, split: str, k: int = 100) -> dict:
    qs = data.questions(split)
    start = time.monotonic()
    ranked = retriever.search([q.text for q in qs], k)
    seconds = time.monotonic() - start
    run = {q.id: r for q, r in zip(qs, ranked, strict=True)}
    path = run_path(split, retriever.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid, r in run.items():
            f.write(json.dumps({"id": qid, "ranked": r}) + "\n")
    summary = summarize(run, split) | {"retriever": retriever.name, "seconds_per_query": seconds / len(qs)}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def line(summary: dict) -> str:
    cells = [f"{m} {100 * summary[m]['mean']:.1f}" for m in ("recall@10", "recall@100", "ndcg@10", "map@100")]
    return f"{summary['retriever']:<45} " + "  ".join(cells)

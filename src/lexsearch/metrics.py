"""Ranking metrics against the lawyers' labels, set metrics for the articles an agent cites, and a paired bootstrap."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def recall_at(ranked: Sequence[str], gold: frozenset[str], k: int) -> float:
    return len(gold.intersection(ranked[:k])) / len(gold)


def ndcg_at(ranked: Sequence[str], gold: frozenset[str], k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, a in enumerate(ranked[:k]) if a in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal


def average_precision(ranked: Sequence[str], gold: frozenset[str], k: int = 100) -> float:
    hits, total = 0, 0.0
    for i, a in enumerate(ranked[:k]):
        if a in gold:
            hits += 1
            total += hits / (i + 1)
    return total / len(gold)


RANKING = {
    "recall@10": lambda r, g: recall_at(r, g, 10),
    "recall@20": lambda r, g: recall_at(r, g, 20),
    "recall@100": lambda r, g: recall_at(r, g, 100),
    "ndcg@10": ndcg_at,
    "map@100": average_precision,
}


def ranking_scores(runs: dict[str, list[str]], gold: dict[str, frozenset[str]]) -> dict[str, np.ndarray]:
    """Per-question scores, in the order of `gold`, for every ranking metric."""
    ids = list(gold)
    return {name: np.array([f(runs[q], gold[q]) for q in ids]) for name, f in RANKING.items()}


def set_scores(cited: frozenset[str], gold: frozenset[str]) -> dict[str, float]:
    """For an answer that cites articles: how many of the cited ones are right, how many of the right ones are cited."""
    hits = len(cited & gold)
    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def bootstrap_ci(values: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean, resampling questions."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a: np.ndarray, b: np.ndarray, **kw) -> tuple[float, float, float]:
    """b minus a on the same questions: mean and 95% interval."""
    d = b - a
    lo, hi = bootstrap_ci(d, **kw)
    return float(d.mean()), lo, hi


def mean_ci(values: np.ndarray, fmt: Callable[[float], str] = lambda x: f"{100 * x:.1f}") -> str:
    lo, hi = bootstrap_ci(values)
    return f"{fmt(values.mean())} ({fmt(lo)}–{fmt(hi)})"

"""Paired differences between retrieval runs on the same questions: python scripts/compare.py <split> <a> <b> ...

Each pair is a/b on the command line; names are run file names without .jsonl.
"""

import sys

import numpy as np

from lexsearch import data
from lexsearch.evaluate import load_run
from lexsearch.metrics import paired_diff, ranking_scores, recall_at

split, names = sys.argv[1], sys.argv[2:]
qs = data.questions(split)
gold = {q.id: q.gold for q in qs}
seen = data.seen_articles()
unseen_q = [q.id for q in qs if q.gold - seen]


def unseen_recall(run, k=100):
    return np.array([recall_at(run[q], gold[q] - seen, k) for q in unseen_q])


for a, b in zip(names[::2], names[1::2]):
    ra, rb = load_run(split, a), load_run(split, b)
    sa, sb = ranking_scores(ra, gold), ranking_scores(rb, gold)
    print(f"{b}  vs  {a}")
    for m in ("recall@10", "recall@100", "ndcg@10"):
        d, lo, hi = paired_diff(sa[m], sb[m])
        print(f"  {m:<12} {100 * d:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    d, lo, hi = paired_diff(unseen_recall(ra), unseen_recall(rb))
    print(f"  {'unseen@100':<12} {100 * d:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]  (n={len(unseen_q)})")

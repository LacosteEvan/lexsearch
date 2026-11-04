"""Recompute every retrieval summary from the saved rankings (after a change to the metrics)."""
import json
import sys
from pathlib import Path

from lexsearch.evaluate import RESULTS, summarize

for path in sorted((RESULTS / "retrieval").glob("*/*.jsonl")):
    split = path.parent.name
    run = {r["id"]: r["ranked"] for r in map(json.loads, path.open())}
    old = path.with_suffix(".summary.json")
    keep = {k: v for k, v in json.loads(old.read_text()).items() if k in ("retriever", "seconds_per_query")}
    new = summarize(run, split) | keep
    old.write_text(json.dumps(new, indent=1))
    print(f"{split:5} {keep['retriever']:<40} R@100 seen {100 * new['recall@100_seen_articles']:.1f} "
          f"unseen {100 * new['recall@100_unseen_articles']:.1f} (n={new['n_unseen_articles']})", file=sys.stdout)

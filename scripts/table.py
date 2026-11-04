"""Print every retrieval summary of a split, best recall@100 first."""
import json
import sys
from pathlib import Path

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
rows = [json.loads(p.read_text()) for p in Path("results/retrieval", split).glob("*.summary.json")]
for d in sorted(rows, key=lambda d: -d["recall@100"]["mean"]):
    print(f"{d['retriever']:<52} R@10 {100 * d['recall@10']['mean']:5.1f}  R@100 {100 * d['recall@100']['mean']:5.1f}  "
          f"nDCG@10 {100 * d['ndcg@10']['mean']:5.1f}  unseen R@100 {100 * d['recall@100_unseen_articles']:5.1f}")

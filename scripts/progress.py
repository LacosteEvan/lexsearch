"""Progress of the answer runs on a split: python scripts/progress.py [split]"""

import json
import sys
from pathlib import Path

split = sys.argv[1] if len(sys.argv) > 1 else "test"
total_cost = 0.0
for path in sorted(Path("results/answers", split).glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
    rows = [json.loads(line) for line in path.open()]
    cost = sum(r["cost_usd"] for r in rows)
    total_cost += cost
    print(f"{path.stem:<40} {len(rows):>4}/222  ${cost:6.2f}  errors {sum(bool(r['error']) for r in rows)}")
print(f"{'total':<40} {'':>8}  ${total_cost:6.2f}")

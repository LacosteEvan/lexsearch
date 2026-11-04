"""Remove failed rows from answer runs so the next run retries them: python scripts/drop_errors.py <files...>"""

import json
import sys
from pathlib import Path

for name in sys.argv[1:]:
    path = Path(name)
    rows = [json.loads(line) for line in path.open()]
    kept = [r for r in rows if not r["error"]]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    print(f"{path.stem}: dropped {len(rows) - len(kept)}")

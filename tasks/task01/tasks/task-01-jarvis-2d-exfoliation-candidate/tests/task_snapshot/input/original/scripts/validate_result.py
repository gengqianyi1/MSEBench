from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import evaluate_result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/validate_result.py <result.json>")
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    workspace = path.parent if path.name == "result.json" else None
    print(json.dumps(evaluate_result(data, workspace=workspace), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


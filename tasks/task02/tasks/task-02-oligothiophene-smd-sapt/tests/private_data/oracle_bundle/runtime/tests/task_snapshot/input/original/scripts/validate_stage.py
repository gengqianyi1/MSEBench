from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage_validation import validate_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a benchmark stage")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--stage", choices=["A", "B", "C", "D"], required=True)
    args = parser.parse_args()
    errors = validate_stage(Path(args.workspace), args.stage)
    payload = {"stage": args.stage, "valid": not errors, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

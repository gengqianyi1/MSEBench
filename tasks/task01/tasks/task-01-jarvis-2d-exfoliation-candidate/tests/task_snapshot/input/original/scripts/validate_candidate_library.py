from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate_validator import validate_candidate_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate candidate-library file format and record schema.")
    parser.add_argument("candidate_library")
    parser.add_argument(
        "--schema",
        default=str(ROOT / "tasks/jarvis_2d_exfoliation/candidate_record_schema.json"),
    )
    parser.add_argument("--required-count", type=int, default=1000)
    args = parser.parse_args()
    report = validate_candidate_library(
        Path(args.candidate_library),
        Path(args.schema),
        required_count=args.required_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

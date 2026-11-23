"""Recompute all immutable v1 and v1.1 historical commitments for v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from package_tools.historical_integrity import (
    build_historical_integrity_receipt,
    validate_historical_integrity_receipt,
)
from package_tools.provenance import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "HISTORICAL_RELEASE_INTEGRITY_POST.json"


def _safe_output(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(
            "HISTORICAL_RECEIPT_OUTPUT_MUST_STAY_IN_PROJECT"
        ) from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    payload = build_historical_integrity_receipt(ROOT.parent)
    if payload["status"] == "PASS":
        validate_historical_integrity_receipt(payload)
    if not args.check_only:
        output = _safe_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
    summary = {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "historical_write_operations": payload[
            "historical_write_operations"
        ],
        "receipt_written": not args.check_only,
        "v1": payload["v1"],
        "v1_archive": payload["v1_archive"],
        "v1_1": payload["v1_1"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

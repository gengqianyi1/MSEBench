#!/usr/bin/env python3
"""Offline Task 02 verifier-replay Oracle check."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETAILS = ROOT / "parity" / "oracle_task02" / "logs" / "verifier" / "details.json"
OUTPUT = ROOT / "parity" / "task02_offline_oracle.json"


def main() -> int:
    if not DETAILS.is_file():
        print(json.dumps({"status": "FAIL", "error": "recorded oracle evidence missing"}))
        return 1
    details = json.loads(DETAILS.read_text(encoding="utf-8"))
    legacy_result = details.get("legacy_result") or {}
    score = float(legacy_result.get("score"))
    failure_codes = legacy_result.get("failure_codes") or []
    ok = details.get("status") == "scored" and score > 0 and not failure_codes
    OUTPUT.write_text(json.dumps({"status": "PASS" if ok else "FAIL", "harbor_score": score, "failure_codes": failure_codes}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if ok else "FAIL", "harbor_score": score, "failure_codes": failure_codes}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

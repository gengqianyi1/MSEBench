#!/usr/bin/env python3
"""Offline Task 01 verifier-replay Oracle check.

This verifies the already-recorded Harbor verifier replay and the native
legacy scorer replay. It does not require Docker at check time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETAILS = ROOT / "parity" / "oracle_task01" / "logs" / "verifier" / "details.json"
NATIVE = ROOT / "parity" / "oracle_task01" / "run" / "native_private_score.json"
OUTPUT = ROOT / "parity" / "task01_offline_oracle.json"


def main() -> int:
    if not DETAILS.is_file() or not NATIVE.is_file():
        print(json.dumps({"status": "FAIL", "error": "recorded oracle evidence missing"}))
        return 1
    details = json.loads(DETAILS.read_text(encoding="utf-8"))
    native = json.loads(NATIVE.read_text(encoding="utf-8"))
    legacy_result = details.get("legacy_result") or {}
    harbor_score = float(legacy_result.get("score"))
    native_score = float(native.get("score"))
    failure_codes = legacy_result.get("failure_codes") or []
    ok = (
        details.get("status") == "scored"
        and harbor_score > 0
        and not failure_codes
        and math.isclose(harbor_score, native_score, abs_tol=1.0e-9)
    )
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "PASS" if ok else "FAIL",
                "harbor_score": harbor_score,
                "native_score": native_score,
                "failure_codes": failure_codes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS" if ok else "FAIL", "harbor_score": harbor_score, "native_score": native_score, "failure_codes": failure_codes}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

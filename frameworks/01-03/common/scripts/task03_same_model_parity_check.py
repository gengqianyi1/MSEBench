#!/usr/bin/env python3
"""Validate Task03 same-model parity against the fixed xTB runtime.

The historical Harbor job used an xTB build whose DIPRO subprogram crashed and
therefore reported zero.  The exact same five DeepSeek candidates were
independently recomputed with the pinned working build.  This check compares
that canonical recomputation with the legacy grader adapter and records that
the historical zero is invalidated rather than silently reused.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "parity" / "task03_fixed_build_same_model_recheck_20260830"
NATIVE_RESULT = EVIDENCE / "workspace" / "result.json"
DETAILS = EVIDENCE / "logs" / "verifier" / "details.json"
OLD_JOB = ROOT / "jobs_fresh_task03b" / "fresh-task03b" / "result.json"
OUTPUT = ROOT / "parity" / "task03_same_model_parity.json"
TOLERANCE = 1.0e-9


def main() -> int:
    required = (NATIVE_RESULT, DETAILS, OLD_JOB)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print(json.dumps({"status": "FAIL", "missing": missing}))
        return 1

    native = json.loads(NATIVE_RESULT.read_text(encoding="utf-8"))
    details = json.loads(DETAILS.read_text(encoding="utf-8"))
    old_job = json.loads(OLD_JOB.read_text(encoding="utf-8"))
    metrics = [
        (name, metric)
        for name, stats in (old_job.get("stats", {}).get("evals") or {}).items()
        for metric in (stats.get("metrics") or [])
    ]
    if len(metrics) != 1:
        print(json.dumps({"status": "FAIL", "error": "expected exactly one historical model metric"}))
        return 1

    agent, old_metric = metrics[0]
    native_score = float(native["score"]["total_score"])
    legacy_score = float(details["legacy_result"]["score"])
    successful = sum(item.get("status") == "success" for item in native.get("evaluations", []))
    eligible = bool(native.get("score", {}).get("eligible"))
    difference = abs(native_score - legacy_score)
    ok = (
        details.get("status") == "scored"
        and math.isfinite(native_score)
        and difference <= TOLERANCE
        and successful == 5
        and eligible
    )
    report = {
        "status": "PASS" if ok else "FAIL",
        "agent": agent,
        "runtime": "xtb=6.7.1=h8876d29_4",
        "historical_score_invalidated": float(old_metric["raw_score"]),
        "invalidation_reason": "historical xTB DIPRO build crashed",
        "harbor_independent_score": native_score,
        "legacy_adapter_score": legacy_score,
        "absolute_difference": difference,
        "tolerance": TOLERANCE,
        "successful_candidates": successful,
        "recommended_candidate_eligible": eligible,
        "evidence": str(EVIDENCE.relative_to(ROOT)).replace("\\", "/"),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

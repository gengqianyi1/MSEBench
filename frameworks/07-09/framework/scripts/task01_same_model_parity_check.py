#!/usr/bin/env python3
"""Same-model Task 01 parity evidence from the recorded DeepSeek Harbor run.

The recorded Harbor job used DeepSeek-V4-Flash and produced evidence ID
``kjlbSSdIg-cyPW5O4RrZvBgFlrt6C1f4``. The native legacy scorer was replayed on
that evidence workspace. This script verifies the job reward and the native
legacy score agree exactly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB_RESULT = ROOT / "jobs_fresh_task01h" / "fresh-task01h" / "result.json"
NATIVE = ROOT / "parity" / "oracle_task01" / "run" / "native_private_score.json"
OUTPUT = ROOT / "parity" / "task01_same_model_parity.json"


def main() -> int:
    if not JOB_RESULT.is_file() or not NATIVE.is_file():
        print(json.dumps({"status": "FAIL", "error": "parity evidence missing"}))
        return 1
    job = json.loads(JOB_RESULT.read_text(encoding="utf-8"))
    metrics = []
    for agent_name, stats in (job.get("stats", {}).get("evals") or {}).items():
        for metric in stats.get("metrics") or []:
            metrics.append((agent_name, metric))
    if not metrics:
        print(json.dumps({"status": "FAIL", "error": "no job metrics"}))
        return 1
    harbor_score = float(metrics[0][1]["raw_score"])
    native_score = float(json.loads(NATIVE.read_text(encoding="utf-8"))["score"])
    ok = math.isclose(harbor_score, native_score, abs_tol=1.0e-9)
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "PASS" if ok else "FAIL",
                "agent": metrics[0][0],
                "harbor_job_raw_score": harbor_score,
                "legacy_native_score": native_score,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS" if ok else "FAIL", "agent": metrics[0][0], "harbor_job_raw_score": harbor_score, "legacy_native_score": native_score}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

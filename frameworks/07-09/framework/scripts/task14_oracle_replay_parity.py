#!/usr/bin/env python3
"""Verify Task 14 synthetic oracle candidates equal the calibrated replay."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "task-14-multiobjective-mol-design"
CORPUS = TASK / "tests" / "task_snapshot" / "private_grader" / "replay_corpus" / "deepseek-v4-pro_q1_multiobjective_mol_design_20260802_004103"

def main() -> int:
    expected = json.loads((CORPUS / "expected.json").read_text(encoding="utf-8"))
    records = [json.loads(x) for x in (CORPUS / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    errors = []
    for rec in records:
        n = int(rec["round"])
        oracle_path = TASK / "steps" / f"round-{n}" / "solution" / "candidate.json"
        if not oracle_path.is_file():
            errors.append(f"round-{n}: missing {oracle_path.relative_to(ROOT)}")
            continue
        got = json.loads(oracle_path.read_text(encoding="utf-8")).get("candidate")
        want = rec.get("submitted_candidate")
        if got != want:
            errors.append(f"round-{n}: oracle candidate differs from calibrated replay")
    report = {
        "task": "task-14-multiobjective-mol-design",
        "source_corpus": CORPUS.name,
        "rounds_checked": len(records),
        "expected_legacy_raw_score": expected.get("total_score"),
        "expected_harbor_reward": float(expected.get("total_score", 0)) / 100.0,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    print(json.dumps(report, indent=2))
    return 1 if errors else 0

if __name__ == "__main__":
    raise SystemExit(main())

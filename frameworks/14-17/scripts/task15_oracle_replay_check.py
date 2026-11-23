#!/usr/bin/env python3
"""Verify Task 15 step Oracles exactly match the calibrated positive replay."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/task-15-lipophilic-sulfonamide-edit-budget"
CORPUS = TASK / "tests/task_snapshot/private_grader/replay_corpus/deepseek-v4-pro_q1_lipophilic_sulfonamide_edit_budget_20260801_104757"


def main() -> int:
    expected = json.loads((CORPUS / "expected.json").read_text(encoding="utf-8"))["total_score"]
    errors=[]; checked=0
    for line in (CORPUS / "trajectory.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        record=json.loads(line); candidate=record.get("submitted_candidate")
        if not candidate: continue
        checked+=1; number=int(record["round"])
        path=TASK / "steps" / f"round-{number}" / "solution/candidate.json"
        actual=json.loads(path.read_text(encoding="utf-8")).get("candidate") if path.is_file() else None
        if actual != candidate: errors.append(f"round-{number}: candidate mismatch")
    print(json.dumps({"task":"task-15","rounds_checked":checked,"expected_raw_score":expected,"errors":errors,"status":"PASS" if not errors else "FAIL"},indent=2))
    return 1 if errors else 0


if __name__ == "__main__": raise SystemExit(main())

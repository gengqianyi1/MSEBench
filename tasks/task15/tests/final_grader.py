#!/usr/bin/env python3
"""Reconstruct the legacy Task 15 scoring surface from five Harbor step submissions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import yaml

WORK = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
TASK = Path(os.environ.get("CHEMBENCH_TASK_ROOT", "/task"))


def _read_candidate_json(path):
    """Read one round's candidate file, treating a corrupt one as a round with no candidate.

    A truncated or malformed `candidate.json` is the single most common way a real model fails a
    round -- the archived legacy runs for this task contain `UNPARSEABLE_SUBMISSION` rounds. Letting
    `json.loads` raise here makes the whole verifier exit non-zero, which `run_legacy_grader.py`
    records as `verifier_error` with `reward 0.0`: an INFRASTRUCTURE failure. That silently converts
    a bad submission into "re-run this", which is exactly the misattribution Evaluation §16.5 and
    §10 forbid. Returning None instead lets the round be counted as missing, the score be computed
    as a genuine lower bound, and `protocol_completeness` report it as INCOMPLETE_MULTI_ROUND.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError):
        return None


def main() -> int:
    records = []
    for number in range(1, 6):
        path = WORK / "rounds" / f"round-{number}" / "candidate.json"
        if not path.is_file():
            continue
        value = _read_candidate_json(path)
        if value is None:
            continue
        candidate = value.get("candidate", value) if isinstance(value, dict) else None
        if isinstance(candidate, dict) and str(candidate.get("smiles", "")).strip():
            records.append({"round": number, "submitted_candidate": candidate})
    (WORK / "trajectory.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    config = yaml.safe_load((TASK / "input/tasks/q1_lipophilic/task_config.yaml").read_text(encoding="utf-8"))
    summary = {
        "reconstructed_from_harbor_steps": True,
        "n_rounds_recorded": len(records),
        "max_rounds": 5,
        "prompt_version": config.get("prompt_version"),
        "ruler_version": config.get("ruler_version"),
    }
    (WORK / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # environment/public/rubric.json marks BOTH final_answer_present and
    # final_answer_declares_task CRITICAL, but this grader never wrote final_answer.json at
    # all until 2026-09-10 -- a stronger form of the gap a round-5 reviewer found on task 17.
    # Nothing caught it because no step executed the rubric; tests/rubric_check.py does now.
    (WORK / "final_answer.json").write_text(
        json.dumps({"task_id": "task_15_lipophilic_sulfonamide_edit_budget",
                    "rounds": [r["submitted_candidate"] for r in records]},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

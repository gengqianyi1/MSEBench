#!/bin/bash
set -euo pipefail
# final_grader.py rebuilds the legacy submission surface (trajectory.jsonl, run_summary.json,
# final_answer.json, and where the task defines it final_report.json) from the per-round candidate
# files Harbor left in /workspace/rounds.
python /tests/final_grader.py
# rubric_check.py executes environment/public/rubric.json against that surface. Advisory: it
# always exits 0, because a structural mismatch is a submission problem and must not become a
# Harbor verifier_error. Its verdict lands in /logs/verifier/rubric_check.json.
python /tests/rubric_check.py
# run_legacy_grader.py scores the surface with the task's own private grader and applies the
# completeness gate declared in the public task_config.yaml before writing reward.json.
python /tests/run_legacy_grader.py

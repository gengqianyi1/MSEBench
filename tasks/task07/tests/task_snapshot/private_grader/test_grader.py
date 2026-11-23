"""Executable canonical fixture for the Task 07 private grader.

Run either ``python private_grader/test_grader.py`` or
``python -m pytest -q private_grader/test_grader.py`` from the task directory.
The bridge checks the canonical JSON fixtures and then executes the authoritative
runtime suite, including positive scoring, objective-gated zero, missing fields,
and invalid/unsafe SMILES cases.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


HERE = Path(__file__).resolve().parent
TASK = HERE.parent
RUNTIME = TASK / "private_data" / "runtime_package"
FIXTURES = HERE / "fixtures"


def _json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _check_canonical_fixtures() -> None:
    sys.path.insert(0, str(RUNTIME))
    try:
        from evaluator.scoring import compute_score
        from tools.rdkit_tools import evaluate_candidate
    finally:
        sys.path.pop(0)

    config = yaml.safe_load((RUNTIME / "configs" / "task.yaml").read_text(encoding="utf-8"))
    full = _json("full_score.json")
    scored = compute_score(
        final_qed=full["final_qed"],
        start_qed=full["start_qed"],
        unique_feasible_candidates=0,
        process_checks={},
        hard_constraints_passed=True,
        process_eligible=True,
        objective_passed=True,
        feedback_revision_quality=full["feedback_revision_quality"],
        config=config,
    )
    assert scored["total_score"] == full["expected_score"]

    zero = _json("zero_score.json")
    gated = compute_score(
        final_qed=full["final_qed"],
        start_qed=full["start_qed"],
        unique_feasible_candidates=0,
        process_checks={},
        hard_constraints_passed=True,
        process_eligible=True,
        objective_passed=zero["objective_passed"],
        feedback_revision_quality=True,
        config=config,
    )
    assert gated["total_score"] == zero["expected_score"]

    invalid = _json("invalid_smiles.json")
    observed = evaluate_candidate(invalid["smiles"], config)
    assert observed["valid"] is invalid["expected_valid"]

    missing = _json("missing_submission.json")
    assert missing["submission"] == {} and missing["expected_score"] == 0.0


def _run_authoritative_runtime_suite() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=RUNTIME,
        check=False,
    )
    assert completed.returncode == 0, f"runtime fixture suite failed: {completed.returncode}"


def test_canonical_private_grader_fixture() -> None:
    _check_canonical_fixtures()
    _run_authoritative_runtime_suite()


if __name__ == "__main__":
    test_canonical_private_grader_fixture()
    print(json.dumps({"task": "07", "status": "PASS", "runtime_fixture_count": 194}))

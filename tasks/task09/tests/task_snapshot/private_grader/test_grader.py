"""Executable canonical fixture bridge for Task 09 coverage and grading."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / "private_data" / "runtime_package"
FIXTURE = HERE / "fixtures" / "coverage_contract.json"


def _check_coverage_contract() -> None:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    suite = json.loads((RUNTIME / "organizer_private" / "suite.json").read_text(encoding="utf-8"))
    assert len(suite["cases"]) == contract["expected_case_count"]
    policy = contract["missing_case_policy"]
    assert policy == {
        "retain_completed_cases": True,
        "impute_missing_as_zero": False,
        "ranking_requires_full_coverage": True,
        "different_partial_subsets_are_comparable": False,
    }


def _run_authoritative_runtime_suite() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=RUNTIME,
        check=False,
    )
    assert completed.returncode == 0, f"runtime fixture suite failed: {completed.returncode}"


def test_canonical_private_grader_fixture() -> None:
    _check_coverage_contract()
    _run_authoritative_runtime_suite()


if __name__ == "__main__":
    test_canonical_private_grader_fixture()
    print(json.dumps({"task": "09", "status": "PASS", "runtime_fixture_count": 485}))

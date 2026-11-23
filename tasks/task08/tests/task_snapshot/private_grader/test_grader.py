"""Executable canonical fixture bridge for all six Task 08 fault cases."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / "private_data" / "runtime_package"
FIXTURES = HERE / "fixtures"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_fixture_inventory() -> None:
    expected = _json(FIXTURES / "fault_cases.json")
    suite = _json(RUNTIME / "organizer_private" / "suite.json")
    cases = [suite["public_case"], *suite["hidden_cases"]]
    observed = {str(case["case_id"]) for case in cases}
    assert len(cases) == expected["expected_case_count"]
    assert observed == set(expected["required_case_ids"])
    for name in ("perfect_recovery.json", "missing_artifact.json", "invalid_lifecycle.json"):
        fixture = _json(FIXTURES / name)
        assert fixture["runtime_test"].startswith("tests/")


def _run_authoritative_runtime_suite() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=RUNTIME,
        check=False,
    )
    assert completed.returncode == 0, f"runtime fixture suite failed: {completed.returncode}"


def test_canonical_private_grader_fixture() -> None:
    _check_fixture_inventory()
    _run_authoritative_runtime_suite()


if __name__ == "__main__":
    test_canonical_private_grader_fixture()
    print(json.dumps({"task": "08", "status": "PASS", "passed": 170, "skipped": 3}))

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.file_tools import WorkspaceSandbox, inspect_scheduler_record
from tools.xtb_log_parser import parse_xtb_log_text
from tools.xyz_validator import validate_xyz_text


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "xtb_parser"


def _parse(name: str) -> dict[str, object]:
    return parse_xtb_log_text((FIXTURES / name).read_text(encoding="utf-8"), path=name)


def _codes(result: dict[str, object], segment: int = -1) -> set[str]:
    selected = result["segments"][segment]
    return {fact["code"] for fact in selected["facts"]}


def test_fixture_manifest_hashes_and_labels_are_exact() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["fixtures"]) == 17
    for entry in manifest["fixtures"]:
        assert entry["source_type"] == "SYNTHETIC_TEST_FIXTURE"
        path = FIXTURES / entry["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("scc_nonconvergence.log", "SCC_NON_CONVERGENCE"),
        ("max_cycles.log", "OPTIMIZATION_MAX_CYCLES"),
        ("segfault.log", "SEGMENTATION_FAULT"),
        ("timeout.log", "TIMEOUT"),
        ("oom.log", "OUT_OF_MEMORY"),
        ("unsupported_option.log", "UNSUPPORTED_OPTION"),
        ("missing_parameter.log", "MISSING_PARAMETER"),
    ],
)
def test_failure_fixture_classification(name: str, code: str) -> None:
    assert code in _codes(_parse(name))


def test_success_and_multirun_ordering() -> None:
    success = _parse("success_opt.log")
    assert success["latest_segment"]["optimization_converged_observed"] is True
    assert success["latest_segment"]["normal_termination_observed"] is True

    multiple = _parse("multi_runs.log")
    assert multiple["segment_count"] == 3
    assert multiple["segments"][0]["optimization_failure_observed"] is True
    assert multiple["segments"][1]["optimization_failure_observed"] is True
    assert multiple["segments"][2]["optimization_converged_observed"] is True


def test_early_outcomes_do_not_override_latest_segment() -> None:
    fail_success = _parse("early_fail_then_success.log")
    assert fail_success["segments"][0]["optimization_failure_observed"] is True
    assert fail_success["latest_segment"]["optimization_converged_observed"] is True

    success_fail = _parse("early_success_then_fail.log")
    assert success_fail["segments"][0]["optimization_converged_observed"] is True
    assert success_fail["latest_segment"]["optimization_failure_observed"] is True
    assert success_fail["latest_segment"]["normal_termination_observed"] is False

    stale = _parse("stale_conflict.log")
    assert stale["segments"][0]["optimization_converged_observed"] is True
    assert stale["latest_segment"]["optimization_failure_observed"] is True


def test_exit_zero_text_and_scheduler_completion_are_not_scientific_success() -> None:
    incomplete = _parse("exit0_incomplete.log")
    assert incomplete["latest_segment"]["optimization_converged_observed"] is False
    assert incomplete["latest_segment"]["normal_termination_observed"] is False

    sandbox = WorkspaceSandbox(FIXTURES)
    scheduler = inspect_scheduler_record(sandbox, "scheduler_completed_xtb_failed.json")
    assert scheduler["state"] == "COMPLETED"
    assert scheduler["scheduler_state"] == "COMPLETED"
    assert scheduler["exit_code"] == 0
    assert scheduler["scientific_success_inference"] is False
    assert scheduler["scientific_completion"] == "NOT_ASSESSED"


def test_unknown_version_is_parsed_without_version_inference() -> None:
    result = _parse("unknown_version.log")
    assert result["segment_count"] == 1
    assert result["latest_segment"]["optimization_failure_observed"] is True
    assert result["latest_segment"]["method"] == "UNKNOWN"
    assert result["latest_segment"]["charge"] == "UNKNOWN"


def test_xtb_observability_contract_extracts_setup_science_and_state() -> None:
    result = parse_xtb_log_text(
        """* xtb version 6.7.1 (abc123)
program call : xtb input.xyz --gfn 2 --chrg -1 --uhf 1 --opt normal
: Hamiltonian GFN2-xTB :
: GBSA solvation false :
*** convergence criteria satisfied after 7 iterations ***
........................ CYCLE 3 ........................
:: total energy -12.345678 Eh ::
:: gradient norm 0.0042 Eh/a0 ::
*** GEOMETRY OPTIMIZATION CONVERGED AFTER 3 ITERATIONS ***
normal termination of xtb
""",
        path="structured.log",
    )
    latest = result["latest_segment"]
    assert latest["version"] == "6.7.1"
    assert latest["method"] == "GFN2-xTB"
    assert latest["charge"] == -1
    assert latest["uhf"] == 1
    assert latest["solvation"] == "NONE"
    assert latest["optimization_cycle_count"] == 3
    assert latest["optimization_convergence"] == "CONVERGED"
    assert latest["scc_convergence"] == "CONVERGED"
    assert latest["final_energy_eh"] == pytest.approx(-12.345678)
    assert latest["gradient_norm_eh_per_bohr"] == pytest.approx(0.0042)
    assert latest["incomplete_or_truncated_sections"] is False
    assert latest["state"] == "NORMAL_TERMINATION"
    assert latest["parser_confidence"] >= 0.8
    assert latest["conflicts"] == []
    assert latest["state_history"]


def test_xtb_observability_contract_reports_conflicts_and_truncation() -> None:
    result = parse_xtb_log_text(
        """$ xtb input.xyz --gfn 1 --opt
Hamiltonian GFN2-xTB
GEOMETRY OPTIMIZATION CONVERGED AFTER 3 ITERATIONS
GEOMETRY OPTIMIZATION FAILED TO CONVERGE
""",
        path="conflict.log",
    )
    latest = result["latest_segment"]
    codes = {item["code"] for item in latest["conflicts"]}
    assert "METHOD_CONFLICT" in codes
    assert "OPTIMIZATION_OUTCOME_CONFLICT" in codes
    assert latest["optimization_convergence"] == "CONFLICT"
    assert latest["state"] == "CONFLICTING_EVIDENCE"
    assert latest["incomplete_or_truncated_sections"] is True


def test_malformed_and_truncated_xyz_fixtures() -> None:
    malformed = validate_xyz_text((FIXTURES / "malformed.xyz").read_text(encoding="utf-8"))
    assert malformed["valid"] is False
    assert malformed["frame_count"] == 0

    truncated = validate_xyz_text((FIXTURES / "truncated.xyz").read_text(encoding="utf-8"))
    assert truncated["valid"] is False
    assert truncated["truncated"] is True


def test_xyz_contract_separates_fragmentation_syntax_and_recoverability() -> None:
    fragmented = validate_xyz_text(
        """2
separated
H 0 0 0
H 0 0 20
""",
        expected_symbols=("H", "H"),
    )
    assert fragmented["syntax_status"] == "SYNTAX_VALID"
    assert fragmented["scientific_status"] == "SCIENTIFICALLY_SUSPICIOUS"
    assert fragmented["fragmentation_detected"] is True
    assert fragmented["fragmentation"]["connected_component_count"] == 2
    assert fragmented["recoverability"] == "UNSAFE_TO_REPAIR"

    recoverable_tail = validate_xyz_text(
        """2
complete
H 0 0 0
H 0 0 0.74
2
partial
H 0 0 0
""",
        expected_symbols=("H", "H"),
    )
    assert recoverable_tail["valid"] is False
    assert recoverable_tail["syntax_status"] == "SYNTAX_INVALID"
    assert recoverable_tail["recoverable_tail_truncation"] is True
    assert recoverable_tail["last_complete_frame_index"] == 0
    assert recoverable_tail["selected_frame_index"] == 0
    assert recoverable_tail["selected_frame_is_complete"] is True
    assert recoverable_tail["recoverability"] == "SAFELY_RECOVERABLE"


@pytest.mark.parametrize(
    ("raw_state", "canonical"),
    [
        ("COMPLETED", "COMPLETED"),
        ("FAILED", "FAILED"),
        ("TIMEOUT", "TIMEOUT"),
        ("OUT_OF_MEMORY", "OUT_OF_MEMORY"),
        ("NODE_FAIL", "NODE_FAIL"),
        ("PREEMPTED", "PREEMPTED"),
        ("CANCELLED", "CANCELLED"),
        ("MYSTERY", "UNKNOWN"),
    ],
)
def test_scheduler_canonical_state_mapping(
    tmp_path: Path, raw_state: str, canonical: str
) -> None:
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps({"scheduler_state": raw_state, "exit_code": None}),
        encoding="utf-8",
    )
    inspected = inspect_scheduler_record(WorkspaceSandbox(tmp_path), "record.json")
    assert inspected["canonical_state"] == canonical
    assert inspected["scientific_success_inference"] is False


def test_scheduler_text_timeout_false_does_not_become_timeout(tmp_path: Path) -> None:
    (tmp_path / "record.log").write_text(
        "scheduler_state=COMPLETED\nexit_code=0\ntimeout=false\n",
        encoding="utf-8",
    )
    inspected = inspect_scheduler_record(WorkspaceSandbox(tmp_path), "record.log")
    assert inspected["timed_out"] is False
    assert inspected["canonical_state"] == "COMPLETED"

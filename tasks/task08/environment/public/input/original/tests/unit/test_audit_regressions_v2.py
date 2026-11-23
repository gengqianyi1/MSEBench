from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from baselines.run_suite_baselines_v2 import deliberately_wrong_diagnosis
from framework.redaction import LOCAL_PATH, sanitize_for_persistence
from tools.file_tools import WorkspaceSandbox, inspect_scheduler_record
from tools.xtb_log_parser import parse_xtb_log_text
from tools.xyz_validator import parse_xyz_text, validate_xyz_text


def test_real_xtb_scc_nonconvergence_marker_is_detected() -> None:
    parsed = parse_xtb_log_text(
        "xtb version 6.7.1\n"
        "*** convergence criteria cannot be satisfied within 250 iterations ***\n"
    )
    latest = parsed["latest_segment"]
    assert latest["scc_convergence"] == "NOT_CONVERGED"
    assert latest["optimization_failure_observed"] is True


def test_segment_detection_is_linear_enough_for_truncated_logs() -> None:
    text = "\n".join(["CYCLE 1 gradient norm 1.0 Eh/α"] * 8_000)
    started = time.perf_counter()
    parsed = parse_xtb_log_text(text)
    elapsed = time.perf_counter() - started
    assert parsed["segment_count"] == 1
    assert elapsed < 2.0


def test_model_hessian_does_not_turn_optimization_into_mixed_phase() -> None:
    parsed = parse_xtb_log_text(
        "xtb version 6.7.1\ngeometry optimization\nmodel hessian updated\n"
    )
    assert parsed["latest_segment"]["phase"] == "geometry_optimization"


def test_xtb_gradient_alpha_unit_is_normalized() -> None:
    parsed = parse_xtb_log_text("gradient norm 2.5E-3 Eh/α\n")
    assert parsed["latest_segment"]["gradient_norm_eh_per_bohr"] == pytest.approx(0.0025)


def test_inp_is_readable_and_utf8_bom_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "xcontrol.inp").write_bytes(b"\xef\xbb\xbf$opt\nmaxcycle=1\n$end\n")
    sandbox = WorkspaceSandbox(tmp_path)
    result = sandbox.read_text_range("xcontrol.inp", 1, 3)
    assert result["content"].startswith("$opt")


def test_scheduler_ignores_incidental_cancel_text_and_numeric_timeout(tmp_path: Path) -> None:
    (tmp_path / "scheduler.json").write_text(
        json.dumps(
            {"state": "COMPLETED", "note": "job was not cancelled by the user", "timeout": 240}
        ),
        encoding="utf-8",
    )
    result = inspect_scheduler_record(WorkspaceSandbox(tmp_path), "scheduler.json")
    assert result["canonical_state"] == "COMPLETED"
    assert result["timed_out"] is False


def test_xyz_rejects_python_numeric_underscore_and_035_angstrom_overlap() -> None:
    assert parse_xyz_text("1_0\ncomment\n").errors
    result = validate_xyz_text(
        "2\nclose contact\nC 0.0 0.0 0.0\nH 0.35 0.0 0.0\n"
    )
    assert result["scientifically_plausible"] is False


def test_redaction_covers_common_linux_and_macos_roots() -> None:
    value = " /opt/app /srv/job /root/x /data/a /var/folders/aa/bb "
    sanitized = sanitize_for_persistence(value)
    assert sanitized.count(LOCAL_PATH) == 5


def test_adversarial_baseline_changes_core_diagnosis_and_scheduler_state() -> None:
    for correct in (
        {
            "primary_category": "OPTIMIZATION_FAILURE",
            "primary_cause_code": "OPTIMIZATION_MAX_CYCLES",
            "scheduler_assessment": "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE",
            "conflict_resolution": "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION",
        },
        {
            "primary_category": "INPUT_FAILURE",
            "primary_cause_code": "UNSUPPORTED_OPTION",
            "scheduler_assessment": "EXECUTION_FAILED",
            "conflict_resolution": "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION",
        },
    ):
        wrong = deliberately_wrong_diagnosis(correct)
        assert wrong["primary_cause_code"] != correct["primary_cause_code"]
        assert wrong["scheduler_assessment"] != correct["scheduler_assessment"]

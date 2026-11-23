from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import pytest

from tools.auto_profile import select_profile
from tools.sampling_state import (
    SamplingStateError,
    canonical_angle,
    reconstruct_batch_state,
)
from tools.tool_registry import ToolSession


def _fake_native(
    *,
    reference_path: Path,
    angle_deg: int,
    attempt_root: Path,
    timeout_seconds: int,
) -> dict:
    attempt_root.mkdir(parents=True, exist_ok=False)
    (attempt_root / "input.xyz").write_bytes(reference_path.read_bytes())
    signed = angle_deg if angle_deg <= 180 else angle_deg - 360
    (attempt_root / "xcontrol.inp").write_text(
        "$constrain\n"
        "  force constant=1.0\n"
        f"  dihedral: 6, 7, 8, 9, {signed:.1f}\n"
        "$end\n",
        encoding="ascii",
        newline="\n",
    )
    (attempt_root / "xtbopt.xyz").write_bytes(reference_path.read_bytes())
    (attempt_root / "stdout.log").write_text(
        "GEOMETRY OPTIMIZATION CONVERGED\n", encoding="ascii"
    )
    (attempt_root / "stderr.log").write_bytes(b"")
    (attempt_root / "attempt_manifest.json").write_text(
        '{"algorithm":"SHA-256","files":[],"schema_version":"1.0"}',
        encoding="ascii",
    )
    energy = -100.0 + 0.001 * (
        1.0 - math.cos(math.radians(4.0 * angle_deg))
    )
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest().upper()
    input_sha = hashlib.sha256(
        (attempt_root / "input.xyz").read_bytes()
    ).hexdigest().upper()
    executable = "XTB_FIXTURE"
    return {
        "method": "GFN2-xTB",
        "phase": "gas",
        "charge": 0,
        "uhf": 0,
        "threads": 1,
        "requested_final_angle_tolerance_deg": 2.0,
        "argv": [
            executable,
            "input.xyz",
            "--gfn",
            "2",
            "--chrg",
            "0",
            "--uhf",
            "0",
            "--opt",
            "tight",
            "--input",
            "xcontrol.inp",
        ],
        "shell": False,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "xtb": {
            "executable": executable,
            "version": "6.7.1",
            "binary_sha256": (
                "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
            ),
            "version_exit_code": 0,
        },
        "return_code": 0,
        "timed_out": False,
        "converged": True,
        "total_energy_hartree": energy,
        "final_angle_deg": float(angle_deg),
        "angle_error_deg": 0.0,
        "structure_validation": {
            "atom_count_ok": True,
            "atom_order_ok": True,
            "composition_ok": True,
            "formal_charge_ok": True,
            "connectivity_ok": True,
        },
        "profile_eligible": True,
        "reference_sha256": reference_sha,
        "prepared_input": {
            "reference_sha256": reference_sha,
            "input_sha256": input_sha,
        },
        "timeout_seconds": timeout_seconds,
    }


def _scripted_native(
    failures_before_success: dict[int, int],
) -> Callable[..., dict]:
    """Return a native stub that fails the requested leading attempts.

    A value of ``2`` exhausts the frozen per-angle attempt budget.  The stub
    still writes a complete immutable raw-attempt directory so retry and
    inspection tests exercise the same projection path as successful points.
    """

    calls: dict[int, int] = {}

    def runner(
        *,
        reference_path: Path,
        angle_deg: int,
        attempt_root: Path,
        timeout_seconds: int,
    ) -> dict:
        record = _fake_native(
            reference_path=reference_path,
            angle_deg=angle_deg,
            attempt_root=attempt_root,
            timeout_seconds=timeout_seconds,
        )
        calls[angle_deg] = calls.get(angle_deg, 0) + 1
        if calls[angle_deg] <= failures_before_success.get(angle_deg, 0):
            (attempt_root / "stdout.log").write_text(
                "GEOMETRY OPTIMIZATION FAILED\n", encoding="ascii"
            )
            record.update(
                {
                    "return_code": 1,
                    "converged": False,
                    "total_energy_hartree": None,
                    "final_angle_deg": None,
                    "angle_error_deg": None,
                    "profile_eligible": False,
                }
            )
        return record

    return runner


def test_native_metadata_invariant_breach_is_never_retried(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(**kwargs) -> dict:
        nonlocal calls
        calls += 1
        record = _fake_native(**kwargs)
        if calls == 1:
            record["method"] = "FORBIDDEN-METHOD"
        return record

    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        native_runner=runner,
        raw_provenance_root=tmp_path / "raw",
    )
    result = session.dispatch(
        _action([0, 60, 120, 180, 240, 300]),
        assistant_turn=1,
    )

    assert result["ok"] is False
    assert result["status"] == "NATIVE_INVARIANT_BREACH"
    assert result["budget"]["native_attempts"] == 1
    assert result["budget"]["terminal_failure"] == "NATIVE_INVARIANT_BREACH"
    assert calls == 1


def _action(angles: list[int], *, finalize: bool = False) -> dict:
    return {
        "action": "run_scan_batch",
        "angles_deg": angles,
        "finalize": finalize,
        "confidence": "high" if finalize else None,
        "rationale": "Frozen deterministic test decision.",
    }


@pytest.mark.parametrize("angle", list(range(0, 360, 5)))
def test_all_72_grid_angles_are_canonical(angle: int) -> None:
    assert canonical_angle(angle) == angle


@pytest.mark.parametrize(
    "angle",
    [-360, -5, 1, 4, 6, 356, 360, 365, True, False, "5", None],
)
def test_non_grid_angles_are_rejected(angle: object) -> None:
    with pytest.raises(SamplingStateError):
        canonical_angle(angle)


def test_three_phase_session_terminally_finalizes(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-three-phase",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_fake_native,
    )
    first = session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )
    assert first["status"] == "BATCH_COMPLETE"
    assert first["profile"]["grid_size"] == 72
    assert "predicted_energies" not in first["profile"]
    second = session.dispatch(
        _action([30, 150, 270]), assistant_turn=2
    )
    assert second["state"]["feedback_rounds"] == 1
    third = session.dispatch(
        _action([90, 210, 330], finalize=True), assistant_turn=3
    )
    assert third["status"] == "FINAL_ACCEPTED"
    assert third["terminal"] == {
        "requested": True,
        "accepted": True,
        "provider_must_stop": True,
    }
    state = session.sampling_state()
    assert state["unique_angle_count"] == 12
    assert state["valid_unique_angle_count"] == 12
    assert state["feedback_rounds"] == 2
    assert state["profile_fit_rounds"] == 3
    assert state["eligible"] is True
    assert len(session.profile(state["profile_ids"][-1])["predicted_energies"]) == 72
    assert all(len(records) == 1 for records in session.attempt_summaries.values())
    assert "audit_tmp" not in json.dumps(third, sort_keys=True)
    assert str(tmp_path) not in json.dumps(third, sort_keys=True)


def test_early_finalize_is_rejected_before_native(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-early-final",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_fake_native,
    )
    result = session.dispatch(
        _action([0, 60, 120, 180, 240, 300], finalize=True),
        assistant_turn=1,
    )
    assert result["ok"] is False
    assert result["budget"]["native_attempts"] == 0


def test_duplicate_prior_angle_is_rejected_before_native(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-duplicate",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_fake_native,
    )
    assert session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )["ok"]
    result = session.dispatch(_action([0, 30]), assistant_turn=2)
    assert result["status"] == "PRIOR_ANGLE_DUPLICATE"
    assert result["budget"]["native_attempts"] == 6


def test_auto_selector_is_deterministic_and_oracle_blind() -> None:
    samples = {
        angle: 1.0 - math.cos(math.radians(4.0 * angle))
        for angle in [0, 60, 120, 180, 240, 300]
    }
    first = select_profile(samples)
    second = select_profile(dict(reversed(list(samples.items()))))
    assert first["profile_hash"] == second["profile_hash"]
    assert first["selector"]["selector_id"] == "auto_periodic_fit_v2"
    assert len(first["predicted_energies"]) == 72
    assert "oracle" not in json.dumps(first).lower()


def test_independent_reconstruction_detects_phase_size_violation() -> None:
    malformed = {
        "phase": "phase0",
        "angles_deg": [0, 60],
        "valid_angles_deg": [0, 60],
        "failed_angles_deg": [],
        "profile_id": "profile",
        "profile_hash": "A" * 64,
        "finalize": False,
        "terminal_accepted": False,
    }
    state = reconstruct_batch_state([malformed])
    assert "batch_0:invalid_batch_size" in state["violations"]
    assert state["eligible"] is False


def test_first_native_failure_is_automatically_retried_and_can_succeed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-auto-retry-success",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_scripted_native({0: 1}),
    )

    result = session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )

    point = next(item for item in result["point_results"] if item["angle_deg"] == 0)
    attempts = session.attempt_summaries[0]
    assert result["status"] == "BATCH_COMPLETE"
    assert point == {
        "angle_deg": 0,
        "status": "valid",
        "attempt_count": 2,
        "final_angle_deg": 0.0,
        "angle_error_deg": 0.0,
    }
    assert [item["profile_eligible"] for item in attempts] == [False, True]
    assert result["budget"]["native_attempts"] == 7
    assert result["state"]["valid_unique_angles"] == 6
    assert result["state"]["failed_angles_deg"] == []

    inspection = session.dispatch(
        {"action": "inspect_failed_attempt", "angle_deg": 0},
        assistant_turn=2,
    )
    assert inspection["ok"] is False
    assert inspection["status"] == "FAILED_ATTEMPT_NOT_INSPECTABLE"


def test_exhausted_two_attempt_failure_is_counted_and_only_then_inspectable(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-auto-retry-exhausted",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_scripted_native({0: 2}),
    )

    result = session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )

    assert result["status"] == "BATCH_COMPLETE"
    assert result["budget"]["unique_angles_used"] == 6
    assert result["budget"]["valid_unique_angles"] == 5
    assert result["budget"]["failed_unique_angles"] == 1
    assert result["budget"]["native_attempts"] == 7
    assert result["point_results"][0]["status"] == "failed"
    assert result["point_results"][0]["attempt_count"] == 2
    assert all(
        item["profile_eligible"] is False
        for item in session.attempt_summaries[0]
    )

    inspection = session.dispatch(
        {"action": "inspect_failed_attempt", "angle_deg": 0},
        assistant_turn=2,
    )
    assert inspection["ok"] is True
    assert inspection["status"] == "FAILED_ATTEMPT_INSPECTION"
    assert inspection["attempt_count"] == 2
    assert len(inspection["attempts"]) == 2

    successful_angle = session.dispatch(
        {"action": "inspect_failed_attempt", "angle_deg": 60},
        assistant_turn=3,
    )
    assert successful_angle["ok"] is False
    assert successful_angle["status"] == "FAILED_ATTEMPT_NOT_INSPECTABLE"


@pytest.mark.parametrize(
    ("failures", "replacement_angles"),
    [
        ({0: 2}, [270]),
        ({0: 2, 30: 2}, [270, 330]),
    ],
)
def test_recovery_accepts_only_needed_one_or_two_replacement_angles(
    tmp_path: Path,
    failures: dict[int, int],
    replacement_angles: list[int],
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id=f"unit-recovery-{len(replacement_angles)}",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_scripted_native(failures),
    )

    assert session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )["status"] == "BATCH_COMPLETE"
    assert session.dispatch(
        _action([30, 150]), assistant_turn=2
    )["status"] == "BATCH_COMPLETE"
    phase2 = session.dispatch(
        _action([90, 210], finalize=True), assistant_turn=3
    )

    assert phase2["status"] == "RECOVERY_REQUIRED"
    assert phase2["ok"] is True
    assert phase2["terminal"]["accepted"] is False
    assert phase2["state"]["recovery_allowed"] is True
    assert phase2["state"]["valid_unique_angles"] == 10 - len(failures)

    recovery = session.dispatch(
        _action(replacement_angles, finalize=True), assistant_turn=4
    )
    state = session.sampling_state()
    assert recovery["status"] == "FINAL_ACCEPTED"
    assert recovery["phase"] == "recovery"
    assert recovery["terminal"]["provider_must_stop"] is True
    assert state["unique_angle_count"] == 10 + len(replacement_angles)
    assert state["valid_unique_angle_count"] == 10
    assert state["profile_fit_rounds"] == 4
    assert state["feedback_rounds"] == 2
    assert state["eligible"] is True


def test_budget_exhaustion_with_too_few_valid_points_fails_eligibility(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-exhausted-ineligible",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_scripted_native({0: 2, 60: 2, 30: 2}),
    )

    session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )
    session.dispatch(_action([30, 150, 270]), assistant_turn=2)
    terminal = session.dispatch(
        _action([90, 210, 330], finalize=True), assistant_turn=3
    )

    assert terminal["status"] == "FINAL_ELIGIBILITY_FAILED"
    assert terminal["ok"] is False
    assert terminal["terminal"]["accepted"] is False
    assert terminal["state"]["unique_angles_used"] == 12
    assert terminal["state"]["unique_angles_remaining"] == 0
    assert terminal["state"]["valid_unique_angles"] == 9
    assert terminal["state"]["recovery_allowed"] is False
    assert session.sampling_state()["eligible"] is False

    extra = session.dispatch(
        _action([45], finalize=True), assistant_turn=4
    )
    assert extra["ok"] is False
    assert extra["status"] == "TERMINAL_FAILURE_ALREADY_RECORDED"
    assert extra["budget"]["native_attempts"] == 15


def test_phase0_native_failures_that_make_ten_valid_points_impossible_fail_closed(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-insufficient-valid-phase0",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_scripted_native({0: 2, 60: 2, 120: 2}),
    )

    result = session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )

    assert result["ok"] is False
    assert result["status"] == "NATIVE_ELIGIBILITY_IMPOSSIBLE"
    assert result["terminal"]["provider_must_stop"] is True
    assert result["budget"]["unique_angles_used"] == 6
    assert result["budget"]["valid_unique_angles"] == 3
    assert result["budget"]["failed_unique_angles"] == 3
    assert result["budget"]["native_attempts"] == 9
    assert result["budget"]["profile_fits"] == 0
    assert result["state"]["eligibility_mathematically_possible"] is False
    assert (run_root / "artifacts" / "batches" / "batch_01.json").is_file()
    assert len(session.ledger) == 1
    assert session.ledger[0]["profile_id"] is None

    later = session.dispatch(_action([30, 330]), assistant_turn=2)
    assert later["status"] == "TERMINAL_FAILURE_ALREADY_RECORDED"
    assert later["budget"]["native_attempts"] == 9


def test_terminal_finalization_blocks_all_later_public_actions(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="unit-terminal-block",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_fake_native,
    )
    session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )
    session.dispatch(_action([30, 150]), assistant_turn=2)
    accepted = session.dispatch(
        _action([90, 210], finalize=True), assistant_turn=3
    )
    assert accepted["status"] == "FINAL_ACCEPTED"

    later_batch = session.dispatch(
        _action([270], finalize=True), assistant_turn=4
    )
    later_inspection = session.dispatch(
        {"action": "inspect_failed_attempt", "angle_deg": 0},
        assistant_turn=5,
    )
    assert later_batch["status"] == "TERMINAL_FINALIZATION_ALREADY_ACCEPTED"
    assert later_inspection["status"] == "TERMINAL_FINALIZATION_ALREADY_ACCEPTED"
    assert session.sampling_state()["unique_angle_count"] == 10
    assert later_inspection["budget"]["native_attempts"] == 10


@pytest.mark.parametrize("invalid_size", [5, 7])
def test_phase0_rejects_invalid_action_sizes_before_native(
    tmp_path: Path,
    invalid_size: int,
) -> None:
    run_root = tmp_path / f"run-{invalid_size}"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id=f"unit-phase0-size-{invalid_size}",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_fake_native,
    )
    result = session.dispatch(
        _action(list(range(0, invalid_size * 5, 5))), assistant_turn=1
    )
    assert result["status"] == "PHASE0_BATCH_SIZE_INVALID"
    assert result["budget"]["native_attempts"] == 0
    assert result["budget"]["unique_angles_used"] == 0


@pytest.mark.parametrize(
    ("phase", "invalid_angles", "finalize", "expected_status"),
    [
        ("phase1", [30], False, "PHASE1_BATCH_SIZE_INVALID"),
        ("phase1", [30, 90, 150, 210], False, "PHASE1_BATCH_SIZE_INVALID"),
        ("phase2", [90], True, "PHASE2_BATCH_SIZE_INVALID"),
        ("phase2", [90, 150, 210, 270], True, "PHASE2_BATCH_SIZE_INVALID"),
    ],
)
def test_refinement_phases_reject_invalid_action_sizes_before_native(
    tmp_path: Path,
    phase: str,
    invalid_angles: list[int],
    finalize: bool,
    expected_status: str,
) -> None:
    run_root = tmp_path / phase
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id=f"unit-{phase}-invalid-size",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_fake_native,
    )
    session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )
    if phase == "phase2":
        session.dispatch(_action([30, 330]), assistant_turn=2)

    before = session.budget_snapshot()["native_attempts"]
    result = session.dispatch(
        _action(invalid_angles, finalize=finalize),
        assistant_turn=2 if phase == "phase1" else 3,
    )
    assert result["status"] == expected_status
    assert result["budget"]["native_attempts"] == before


@pytest.mark.parametrize("replacement_angles", [[], [45, 75, 105]])
def test_recovery_rejects_action_sizes_outside_one_or_two_before_native(
    tmp_path: Path,
    replacement_angles: list[int],
) -> None:
    run_root = tmp_path / f"run-{len(replacement_angles)}"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id=f"unit-recovery-size-{len(replacement_angles)}",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_scripted_native({0: 2}),
    )
    session.dispatch(
        _action([0, 60, 120, 180, 240, 300]), assistant_turn=1
    )
    session.dispatch(_action([30, 150]), assistant_turn=2)
    phase2 = session.dispatch(
        _action([90, 210], finalize=True), assistant_turn=3
    )
    assert phase2["status"] == "RECOVERY_REQUIRED"

    before = session.budget_snapshot()["native_attempts"]
    result = session.dispatch(
        _action(replacement_angles, finalize=True), assistant_turn=4
    )
    assert result["status"] == "RECOVERY_BATCH_SIZE_INVALID"
    assert result["budget"]["native_attempts"] == before

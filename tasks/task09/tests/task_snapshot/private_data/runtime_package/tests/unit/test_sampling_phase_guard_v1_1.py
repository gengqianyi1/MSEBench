from __future__ import annotations

from pathlib import Path

import pytest

import tools.tool_registry as registry
from framework.schemas import CASE_ID
from tools.science import canonical_json_bytes, sha256_file
from tools.tool_registry import ToolSession


def _fake_native(
    *,
    reference_path: Path,
    angle_deg: int,
    attempt_root: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    attempt_root.mkdir(parents=True, exist_ok=False)
    (attempt_root / "result.json").write_bytes(
        canonical_json_bytes({"return_code": 0, "angle_deg": angle_deg})
    )
    (attempt_root / "attempt_manifest.json").write_bytes(
        canonical_json_bytes({"angle_deg": angle_deg})
    )
    return {
        "return_code": 0,
        "timed_out": False,
        "converged": True,
        "total_energy_hartree": -10.0 + angle_deg * 0.00001,
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
        "reference_sha256": sha256_file(reference_path),
        "prepared_input": {"input_sha256": "A" * 64},
        "attempt_manifest_sha256": sha256_file(
            attempt_root / "attempt_manifest.json"
        ),
        "xtb": {"binary_sha256": "B" * 64},
    }


def _fit_analyze(
    session: ToolSession,
    *,
    turn: int,
) -> tuple[dict[str, object], dict[str, object]]:
    fit = session.dispatch(
        "fit_periodic_profile",
        {"method": "periodic_cubic"},
        assistant_turn=turn,
    )
    assert fit["ok"] is True
    profile_id = fit["profile"]["profile_id"]
    analysis = session.dispatch(
        "analyze_profile",
        {"profile_id": profile_id},
        assistant_turn=turn,
    )
    assert analysis["ok"] is True
    return fit, analysis


def _submission(profile_id: str, analysis: dict[str, object]) -> dict[str, object]:
    scientific = analysis["analysis"]
    return {
        "case_id": CASE_ID,
        "final_status": "completed",
        "profile_id": profile_id,
        "global_minimum_deg": scientific["global_minimum"]["angle_deg"],
        "local_minima_deg": [
            item["angle_deg"] for item in scientific["local_minima"]
        ],
        "local_maxima_deg": [
            item["angle_deg"] for item in scientific["local_maxima"]
        ],
        "forward_barrier_kcal_mol": scientific["forward_barrier"],
        "reverse_barrier_kcal_mol": scientific["reverse_barrier"],
        "confidence": 0.8,
        "rationale": "Three guarded feedback refinements completed.",
    }


def test_initial_six_are_canonical_and_fit_analysis_unlocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(registry, "run_native_scan", _fake_native)
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(run_root)

    for angle in (0, 5, 10, 15, 20, 25):
        result = session.dispatch(
            "run_scan_point", {"angle_deg": angle}, assistant_turn=2
        )
        assert result["ok"] is True
    state = session.budget_snapshot()
    assert state["new_unique_angles_this_round"] == 6
    assert state["cumulative_unique_angles"] == 6
    assert state["unique_angles_remaining"] == 10
    assert state["scan_feedback_rounds"] == 0
    assert state["profile_fit_rounds"] == 0
    assert state["current_sampling_phase"] == "initial"
    assert state["first_sampling_turn_new_angles"] == 6

    rejected = session.dispatch(
        "run_scan_point", {"angle_deg": 30}, assistant_turn=3
    )
    assert rejected["status"] == "FIRST_ROUND_NEW_ANGLE_LIMIT"
    fit = session.dispatch(
        "fit_periodic_profile",
        {"method": "periodic_cubic"},
        assistant_turn=3,
    )
    assert fit["ok"] is True
    pending = fit["budget"]
    assert pending["phase_fit_pending_analysis"] is True
    assert pending["current_sampling_phase"] == "initial"

    stale_scan = session.dispatch(
        "run_scan_point", {"angle_deg": 30}, assistant_turn=3
    )
    assert stale_scan["status"] == "PHASE_ANALYSIS_REQUIRED"
    duplicate_fit = session.dispatch(
        "fit_periodic_profile",
        {"method": "periodic_cubic"},
        assistant_turn=3,
    )
    assert duplicate_fit["status"] == "PHASE_ANALYSIS_REQUIRED"

    profile_id = fit["profile"]["profile_id"]
    analyzed = session.dispatch(
        "analyze_profile",
        {"profile_id": profile_id},
        assistant_turn=3,
    )
    assert analyzed["budget"]["current_sampling_phase"] == "refinement_1"
    assert analyzed["budget"]["initial_sampling_new_unique_angles"] == 6
    assert analyzed["budget"]["new_unique_angles_this_round"] == 0

    same_turn = session.dispatch(
        "run_scan_point", {"angle_deg": 30}, assistant_turn=3
    )
    assert same_turn["status"] == "REFINEMENT_REQUIRES_LATER_ASSISTANT_TURN"
    later = session.dispatch(
        "run_scan_point", {"angle_deg": 30}, assistant_turn=4
    )
    assert later["ok"] is True
    assert later["budget"]["new_unique_angles_this_round"] == 1


def test_refinement_caps_submit_guard_and_finalization_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(registry, "run_native_scan", _fake_native)
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(run_root)

    for angle in (0, 5, 10, 15, 20, 25):
        assert session.dispatch(
            "run_scan_point", {"angle_deg": angle}, assistant_turn=1
        )["ok"]
    _, initial_analysis = _fit_analyze(session, turn=1)
    assert initial_analysis["budget"]["scan_feedback_rounds"] == 0

    last_fit: dict[str, object] | None = None
    last_analysis: dict[str, object] | None = None
    for round_number, (turn, angles) in enumerate(
        (
            (2, (30, 35, 40)),
            (3, (45, 50, 55)),
            (4, (60, 65, 70)),
        ),
        start=1,
    ):
        for angle in angles:
            assert session.dispatch(
                "run_scan_point", {"angle_deg": angle}, assistant_turn=turn
            )["ok"]
        if round_number == 1:
            capped = session.dispatch(
                "run_scan_point", {"angle_deg": 75}, assistant_turn=turn
            )
            assert capped["status"] == "SAMPLING_PHASE_NEW_ANGLE_LIMIT"
            assert capped["budget"]["phase_new_unique_angles_remaining"] == 0
        last_fit, last_analysis = _fit_analyze(session, turn=turn)
        assert last_analysis["budget"]["scan_feedback_rounds"] == round_number

    assert last_fit is not None and last_analysis is not None
    state = session.budget_snapshot()
    assert state["scan_feedback_rounds"] == 3
    assert state["profile_fit_rounds"] == 4
    assert state["current_sampling_phase"] == "post_feedback"
    assert state["third_feedback_round_reachable"] is True

    sixteenth = session.dispatch(
        "run_scan_point", {"angle_deg": 75}, assistant_turn=5
    )
    assert sixteenth["ok"] is True
    assert sixteenth["budget"]["cumulative_unique_angles"] == 16
    assert sixteenth["budget"]["unique_angles_remaining"] == 0
    assert sixteenth["budget"]["current_sampling_phase"] == "finalization_only"
    assert sixteenth["budget"]["finalization_only"] is True

    blocked = session.dispatch(
        "run_scan_point", {"angle_deg": 80}, assistant_turn=5
    )
    assert blocked["status"] == "FINALIZATION_ONLY"
    listed = session.dispatch("list_scan_points", {}, assistant_turn=5)
    assert listed["ok"] is True
    analyzed_again = session.dispatch(
        "analyze_profile",
        {"profile_id": last_fit["profile"]["profile_id"]},
        assistant_turn=5,
    )
    assert analyzed_again["ok"] is True
    submitted = session.dispatch(
        "submit_final",
        _submission(last_fit["profile"]["profile_id"], last_analysis),
        assistant_turn=5,
    )
    assert submitted["ok"] is True
    assert submitted["budget"]["finalization_only"] is True


def test_submit_is_rejected_until_three_completed_feedback_rounds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(registry, "run_native_scan", _fake_native)
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(run_root)
    for angle in (0, 60, 120, 180, 240, 300):
        assert session.dispatch(
            "run_scan_point", {"angle_deg": angle}, assistant_turn=1
        )["ok"]
    fit, analysis = _fit_analyze(session, turn=1)
    rejected = session.dispatch(
        "submit_final",
        _submission(fit["profile"]["profile_id"], analysis),
        assistant_turn=1,
    )
    assert rejected["ok"] is False
    assert rejected["status"] == "FEEDBACK_ROUNDS_REQUIRED"
    assert rejected["eligibility_gaps"]["scan_feedback_rounds_remaining"] == 3
    assert session.final_submission is None

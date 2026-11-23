from __future__ import annotations

from pathlib import Path

import pytest

import tools.tool_registry as registry
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
    (attempt_root / "result.json").write_bytes(canonical_json_bytes({"ok": True}))
    (attempt_root / "attempt_manifest.json").write_bytes(
        canonical_json_bytes({"angle_deg": angle_deg})
    )
    return {
        "return_code": 0,
        "timed_out": False,
        "converged": True,
        "total_energy_hartree": -5.0 + angle_deg * 0.00001,
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


def test_scan_fit_analysis_phase_transitions_are_enforced(
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
    fit = session.dispatch(
        "fit_periodic_profile", {"method": "periodic_cubic"}, assistant_turn=1
    )
    assert session.dispatch(
        "analyze_profile",
        {"profile_id": fit["profile"]["profile_id"]},
        assistant_turn=1,
    )["ok"]

    for expected_round, (turn, angle) in enumerate(
        ((2, 30), (3, 90), (4, 270)),
        start=1,
    ):
        scan = session.dispatch(
            "run_scan_point", {"angle_deg": angle}, assistant_turn=turn
        )
        assert scan["budget"]["new_unique_angles_this_round"] == 1
        fit = session.dispatch(
            "fit_periodic_profile",
            {"method": "periodic_cubic"},
            assistant_turn=turn,
        )
        analyzed = session.dispatch(
            "analyze_profile",
            {"profile_id": fit["profile"]["profile_id"]},
            assistant_turn=turn,
        )
        assert analyzed["budget"]["scan_feedback_rounds"] == expected_round

    final_state = session.budget_snapshot()
    assert final_state["current_sampling_phase"] == "post_feedback"
    assert final_state["scan_feedback_rounds"] == 3
    assert final_state["profile_fit_rounds"] == 4
    assert final_state["third_feedback_round_reachable"] is True


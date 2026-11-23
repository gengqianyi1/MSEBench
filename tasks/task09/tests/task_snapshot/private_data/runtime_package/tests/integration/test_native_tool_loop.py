from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework.schemas import CASE_ID
from tools.science import run_native_scan, verify_attempt_manifest
from tools.tool_registry import ToolSession


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.native
def test_native_single_scan_point(tmp_path: Path) -> None:
    attempt = tmp_path / "angle_035_attempt_01"
    record = run_native_scan(
        reference_path=ROOT / "benchmark_case" / "reference.xyz",
        angle_deg=35,
        attempt_root=attempt,
        timeout_seconds=180,
    )
    assert record["return_code"] == 0
    assert record["profile_eligible"] is True
    assert record["angle_error_deg"] <= 2.0
    assert verify_attempt_manifest(attempt)


@pytest.mark.native
def test_scan_fit_analyze_submit_loop(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(run_root)
    for angle in [0, 60, 120, 180, 240, 300]:
        result = session.dispatch(
            "run_scan_point", {"angle_deg": angle}, assistant_turn=1
        )
        assert result["ok"] is True
    fit = session.dispatch(
        "fit_periodic_profile", {"method": "periodic_cubic"}, assistant_turn=1
    )
    profile_id = fit["profile"]["profile_id"]
    analysis = session.dispatch(
        "analyze_profile", {"profile_id": profile_id}, assistant_turn=1
    )
    assert analysis["ok"] is True
    for turn, angle in ((2, 90), (3, 270), (4, 30)):
        result = session.dispatch(
            "run_scan_point", {"angle_deg": angle}, assistant_turn=turn
        )
        assert result["ok"] is True
        fit = session.dispatch(
            "fit_periodic_profile",
            {"method": "periodic_cubic"},
            assistant_turn=turn,
        )
        profile_id = fit["profile"]["profile_id"]
        analysis = session.dispatch(
            "analyze_profile", {"profile_id": profile_id}, assistant_turn=turn
        )
        assert analysis["ok"] is True
    scientific = analysis["analysis"]
    final = session.dispatch(
        "submit_final",
        {
            "case_id": CASE_ID,
            "final_status": "completed",
            "profile_id": profile_id,
            "global_minimum_deg": scientific["global_minimum"]["angle_deg"],
            "local_minima_deg": [item["angle_deg"] for item in scientific["local_minima"]],
            "local_maxima_deg": [item["angle_deg"] for item in scientific["local_maxima"]],
            "forward_barrier_kcal_mol": scientific["forward_barrier"],
            "reverse_barrier_kcal_mol": scientific["reverse_barrier"],
            "confidence": 0.7,
            "rationale": "Native points, two fits and three feedback sampling rounds were completed.",
        },
        assistant_turn=4,
    )
    assert final["ok"] is True
    assert session.budget_snapshot()["unique_valid_scan_angles"] == 9
    assert session.budget_snapshot()["scan_feedback_rounds"] == 3
    assert session.budget_snapshot()["profile_fits"] == 4

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.evaluate_v2 import evaluate_run
from framework.trusted_final import validate_final_envelope
from package_tools.provenance import validate_package_safe_projection
from tools.science import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def test_oracle_72_point_inheritance_is_replayed_and_package_safe() -> None:
    profile = json.loads(
        (ROOT / "evaluator_private" / "oracle_profile.json").read_text()
    )
    report = json.loads(
        (ROOT / "reports" / "oracle_independent_validation.json").read_text()
    )
    assert profile["grid_angles_deg"] == list(range(0, 360, 5))
    assert len(profile["relative_energies_kcal_mol"]) == 72
    assert report["status"] == "PASS"
    assert report["checks"]["point_count"] == 72
    assert report["checks"]["replay_max_abs_error_kcal_mol"] == 0
    projections = sorted(
        (ROOT / "evaluator_private" / "oracle_projections").glob("*.json")
    )
    assert len(projections) == 72
    for path in projections:
        assert validate_package_safe_projection(json.loads(path.read_text()))


def test_gate1_smoke_is_four_native_successes() -> None:
    receipt = json.loads(
        (ROOT / "reports" / "gate1_scientific_smoke.json").read_text()
    )
    assert receipt["passed"] is True
    assert [item["angle_deg"] for item in receipt["smoke_angles"]] == [
        0,
        90,
        180,
        270,
    ]
    assert all(item["profile_eligible"] for item in receipt["smoke_angles"])


def test_baseline_live_and_replay_are_identical_and_eligible() -> None:
    run = ROOT / "trajectories" / "baseline"
    frozen = json.loads((run / "evaluation.json").read_text())
    replay = evaluate_run(run, write=False)
    assert replay == frozen
    assert replay["run_status"] == "PASS"
    assert replay["eligible"] is True
    assert replay["reconstructed"]["state"]["unique_angle_count"] == 12
    assert replay["reconstructed"]["state"]["feedback_rounds"] == 2
    assert replay["reconstructed"]["state"]["profile_fit_rounds"] == 3


def test_baseline_final_envelope_is_closed_and_bound() -> None:
    run = ROOT / "trajectories" / "baseline"
    final = json.loads((run / "final_answer.json").read_text())
    assert validate_final_envelope(final)
    assert final["trusted_runtime"]["actual_model_identity"] == (
        "deterministic-baseline/2.0"
    )
    assert final["trusted_runtime"]["unique_angles"] == 12
    assert len(
        final["trusted_runtime"]["profile"]["predicted_energy_kcal_mol"]
    ) == 72


def test_baseline_usage_has_zero_provider_turns_and_twelve_native_attempts() -> None:
    usage = json.loads(
        (ROOT / "trajectories" / "baseline" / "usage_report.json").read_text()
    )
    assert usage["provider_turns"] == 0
    assert usage["native_xtb_attempts"] == 12
    assert usage["successful_unique_angles"] == 12
    assert usage["failed_unique_angles"] == 0


@pytest.mark.parametrize("label", ["kimi", "deepseek", "glm"])
def test_v3_canonical_campaign_status_is_current(label: str) -> None:
    release = json.loads((ROOT / "configs" / "release.json").read_text(encoding="utf-8"))
    if release.get("task_id") == "adaptive_xtb_torsion_profile_agent_v3":
        # v2 single-case formal records are retained in the editable audit tree,
        # but the changed suite identity makes replaying them as v3 evidence invalid.
        assert release["legacy_v2_trajectories_valid_for_v3"] is False
        model_status = json.loads(
            (ROOT / "reports" / "MODEL_RUN_STATUS.json").read_text(encoding="utf-8")
        )
        status = model_status["models"][label]
        assert status["requested_model_id"]
        assert status["ranking_eligible"] is False
        if label == "deepseek":
            assert status["pass"] == 1
            assert status["blocked"] == 3
            assert status["capability_mean_excluding_blocked"] == pytest.approx(
                68.21965443164
            )
        elif label == "glm":
            assert status["pass"] == 3
            assert status["blocked"] == 1
            assert status["capability_mean_excluding_blocked"] == pytest.approx(
                88.875779651515
            )
        else:
            assert status["pass"] == 0
            assert status["blocked"] == 4
            assert status["capability_mean_excluding_blocked"] is None
        return
    run = ROOT / "trajectories" / label
    trajectory = run / "trajectory.jsonl"
    if not trajectory.exists():
        assert not [path for path in run.rglob("*") if path.is_file()]
        return
    evaluation = evaluate_run(run, write=False)
    assert evaluation["run_status"] in {"PASS", "FAIL", "BLOCKED"}
    if evaluation.get("legal_failure_envelope") is True:
        assert evaluation["run_status"] in {"FAIL", "BLOCKED"}
        trajectory_audit = evaluation["trajectory"]
        assert evaluation["usage_accounting"]["passed"] is True
    else:
        trajectory_audit = evaluation["reconstructed"]["trajectory"]
    assert sha256_file(trajectory) == trajectory_audit["file_sha256"]

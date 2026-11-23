from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.evaluate import evaluate_run, evaluation_bytes
from framework.schemas import FORMAL_MODELS
from framework.trajectory import load_jsonl
from tools.campaign import CAMPAIGN_IDENTITY, validate_campaign_snapshot
from tools.science import sha256_file


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_LABELS = ("kimi", "deepseek", "glm")


def _complete_run(label: str) -> Path | None:
    run = ROOT / "trajectories" / label
    required = {
        "trajectory.jsonl",
        "final_answer.json",
        "evaluation.json",
        "run_manifest.json",
    }
    if not run.is_dir():
        return None
    assert required <= {path.name for path in run.iterdir() if path.is_file()}
    return run


@pytest.mark.e2e
def test_oracle_is_72_point_native_and_committed() -> None:
    oracle_path = ROOT / "evaluator_private" / "oracle_profile.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    commitment = json.loads(
        (ROOT / "evaluator_private" / "oracle_commitment.json").read_text(
            encoding="utf-8"
        )
    )
    validation = json.loads(
        (ROOT / "reports" / "ORACLE_V1_1_ASSET_VALIDATION.json").read_text(
            encoding="utf-8"
        )
    )
    assert oracle["grid_angles_deg"] == list(range(0, 360, 5))
    assert len(oracle["total_energies_hartree"]) == 72
    assert validation["status"] == "PASS"
    assert validation["oracle_validation"]["successful_attempt_count"] == 72
    assert validation["oracle_validation"]["failed_attempt_count"] == 0
    assert validation["oracle_validation"]["retry_count"] == 0
    assert commitment["oracle_profile_sha256"] == sha256_file(oracle_path)


@pytest.mark.e2e
def test_campaign_snapshot_is_absent_before_freeze_or_valid_after_freeze() -> None:
    snapshot = ROOT / "configs" / "campaign_snapshot.json"
    if not snapshot.exists():
        assert not any((ROOT / "trajectories").iterdir())
        return
    validation = validate_campaign_snapshot(ROOT)
    for field, expected in CAMPAIGN_IDENTITY.items():
        assert validation[field] == expected


@pytest.mark.e2e
def test_baseline_live_evaluation_is_fresh_or_not_yet_created() -> None:
    run = _complete_run("baseline")
    if run is None:
        assert not (ROOT / "reports" / "baseline_live_validation.json").exists()
        return
    evaluation = evaluate_run(run)
    stored = (run / "evaluation.json").read_bytes()
    final = json.loads((run / "final_answer.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert evaluation_bytes(evaluation) == stored
    assert evaluation["status"] == "PASS"
    assert evaluation["eligible"] is True
    assert set(final) == {"model_submission", "trusted_runtime"}
    assert manifest["run_id"] == "baseline-live-v1-1"
    assert manifest["requested_model_id"] == "deterministic-baseline/1.1"
    reconstruction = evaluation["independent_reconstruction"]
    assert reconstruction["budget"]["first_sampling_turn_new_angles"] == 6
    assert reconstruction["budget"]["feedback_round_count"] >= 3
    assert reconstruction["budget"]["valid_profile_fit_count"] >= 2


@pytest.mark.e2e
def test_baseline_runner_and_evaluator_sampling_metadata_match() -> None:
    run = _complete_run("baseline")
    if run is None:
        assert not (ROOT / "trajectories" / "baseline").exists()
        return
    final = json.loads((run / "final_answer.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    runner = final["trusted_runtime"]["process_recomputed_by_runner"]
    evaluator = evaluation["independent_reconstruction"]["budget"]
    for field in (
        "first_sampling_turn_new_angles",
        "new_unique_angles_this_round",
        "cumulative_unique_angles",
        "unique_angles_remaining",
        "scan_feedback_rounds",
        "profile_fit_rounds",
        "current_sampling_phase",
        "finalization_only",
        "completed_sampling_rounds",
    ):
        assert runner[field] == evaluator[field]
    assert runner["first_sampling_turn_new_angles"] == 6


@pytest.mark.e2e
def test_baseline_offline_replay_is_fresh_or_not_yet_created() -> None:
    replay_path = ROOT / "reports" / "baseline_replay_validation.json"
    if not replay_path.exists():
        assert _complete_run("baseline") is None
        return
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["status"] == "PASS"
    assert len(replay["results"]) == 1
    checks = replay["results"][0]["checks"]
    assert checks["evaluation_byte_identical"] is True
    assert checks["double_recompute_byte_identical"] is True
    assert checks["final_answer_unchanged"] is True
    assert checks["provider_called"] is False
    assert checks["native_xtb_called"] is False


@pytest.mark.e2e
@pytest.mark.parametrize("label", ACTIVE_LABELS)
def test_formal_v1_1_run_is_independently_recomputed_or_not_yet_created(
    label: str,
) -> None:
    run = _complete_run(label)
    if run is None:
        assert not (ROOT / "reports" / f"{label}_formal_validation.json").exists()
        return
    requested = FORMAL_MODELS[label]
    events = load_jsonl(run / "trajectory.jsonl")
    stored = (run / "evaluation.json").read_bytes()
    evaluation = evaluate_run(run)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    final = json.loads((run / "final_answer.json").read_text(encoding="utf-8"))
    assert evaluation_bytes(evaluation) == stored
    assert evaluation["status"] in {"PASS", "FAIL", "BLOCKED"}
    assert manifest["run_id"] == f"{label}-formal_campaign_v1_1_rc1"
    assert manifest["requested_model_id"] == requested
    assert manifest["fallback_used"] is False
    assert manifest["model_override_used"] is False
    assert set(final) == {"model_submission", "trusted_runtime"}
    assert events[-1]["event"] in {"run_complete", "run_failure"}
    if events[-1]["event"] == "run_complete":
        assert manifest["actual_model_id"] == requested
        assert manifest["exact_identity"] is True
        assert evaluation["independent_reconstruction"]["identity"]["exact"] is True
    else:
        assert evaluation["status"] == "BLOCKED"
        assert final["model_submission"] == {}


@pytest.mark.e2e
@pytest.mark.parametrize("label", ACTIVE_LABELS)
def test_formal_v1_1_process_metadata_agrees_when_completed(label: str) -> None:
    run = _complete_run(label)
    if run is None:
        assert not (ROOT / "trajectories" / label).exists()
        return
    final = json.loads((run / "final_answer.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    if evaluation["status"] == "BLOCKED":
        assert final["model_submission"] == {}
        return
    runner = final["trusted_runtime"]["process_recomputed_by_runner"]
    evaluator = evaluation["independent_reconstruction"]["budget"]
    for field in (
        "first_sampling_turn_new_angles",
        "new_unique_angles_this_round",
        "cumulative_unique_angles",
        "unique_angles_remaining",
        "scan_feedback_rounds",
        "profile_fit_rounds",
        "current_sampling_phase",
        "finalization_only",
        "completed_sampling_rounds",
    ):
        assert runner[field] == evaluator[field]
    assert runner["first_sampling_turn_new_angles"] <= 6


@pytest.mark.e2e
def test_campaign_replay_is_fresh_or_not_yet_created() -> None:
    replay_path = ROOT / "reports" / "replay_validation.json"
    if not replay_path.exists():
        assert any(_complete_run(label) is None for label in ACTIVE_LABELS)
        return
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["status"] == "PASS"
    results = {item["label"]: item for item in replay["results"]}
    assert set(results) == {"baseline", *ACTIVE_LABELS}
    for item in results.values():
        assert item["status"] == "PASS"
        checks = item["checks"]
        assert checks["evaluation_byte_identical"] is True
        assert checks["double_recompute_byte_identical"] is True
        assert checks["trajectory_valid"] is True
        assert checks["provider_called"] is False
        assert checks["native_xtb_called"] is False


@pytest.mark.e2e
def test_v1_results_are_not_present_as_v1_1_history() -> None:
    assert not (ROOT / "trajectories" / "history").exists()
    invalidation = (ROOT / "reports" / "RC1_INVALIDATION_REPORT.md").read_text(
        encoding="utf-8"
    )
    assert "permanently classified **FAIL**" in invalidation
    assert "F96C2938473B1791F9F027CA80E140C550C345F6C978C6C7769A82EC59F053DE" in invalidation

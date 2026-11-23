from __future__ import annotations

import json
from pathlib import Path

from evaluator.evaluate import evaluate_files
from evaluator.trusted_validation import canonical_json_bytes
from tests.unit.test_evaluator_helpers import (
    build_complete_fixture,
    refresh_final_answer_envelope,
    rehash_events,
)


def _evaluate(paths: dict[str, Path]) -> dict:
    return evaluate_files(
        final_answer_path=paths["final"],
        trajectory_path=paths["trajectory"],
        artifacts_path=paths["case"],
        config_path=paths["config"],
    )


def test_frozen_category_weights_total_100(tmp_path: Path) -> None:
    result = _evaluate(build_complete_fixture(tmp_path))

    assert {name: category["maximum"] for name, category in result["scores"].items()} == {
        "failure_diagnosis_and_evidence": 40,
        "restart_strategy_and_provenance": 20,
        "actual_task_recovery": 30,
        "compliance_and_efficiency": 10,
    }
    assert sum(category["score"] for category in result["scores"].values()) == result["ungated_score"]


def test_wrong_diagnosis_loses_only_derived_diagnosis_points(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)
    final_answer = json.loads(paths["final"].read_text(encoding="utf-8"))
    model_submission = final_answer["model_submission"]
    model_submission["diagnosis"]["primary_category"] = "UNKNOWN"
    model_submission["diagnosis"]["primary_cause_code"] = "UNKNOWN"
    paths["final"].write_bytes(canonical_json_bytes(final_answer, newline=True, indent=2))

    def update_final(events: list[dict]) -> None:
        for event in events:
            if event["event"] == "tool_call" and event["payload"]["name"] == "submit_final":
                event["payload"]["arguments"] = model_submission
            elif event["event"] == "final_submitted":
                event["payload"]["final_answer"] = model_submission

    rehash_events(paths["trajectory"], update_final)
    refresh_final_answer_envelope(paths)
    result = _evaluate(paths)

    diagnosis = result["scores"]["failure_diagnosis_and_evidence"]
    assert diagnosis["components"]["primary_failure_cause"]["awarded"] == 0
    assert diagnosis["components"]["scheduler_vs_scientific_state"]["awarded"] == 5
    assert result["raw_score_before_scientific_cap"] == 80
    assert result["ungated_score"] == 20
    assert result["scientific_core_cap_applied"] is True
    # Scientific completion remains independently derived even when model text is wrong.
    assert result["trusted_evidence"]["artifacts"]["optimization_converged"] is True
    assert result["trusted_evidence"]["artifacts"]["hessian_completed"] is True


def test_hard_gate_keeps_diagnostic_ungated_score(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)

    def mismatch(events: list[dict]) -> None:
        for event in events:
            if event["event"] == "provider_response_metadata":
                event["payload"]["actual_model"] = "fallback/model"

    rehash_events(paths["trajectory"], mismatch)
    result = _evaluate(paths)

    assert result["ungated_score"] == 100
    assert result["official_score"] == 0
    assert result["eligibility"]["eligible"] is False


def test_scoring_uses_artifacts_not_claimed_success(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)
    hessian = paths["case"] / "attempts" / "attempt_002" / "hessian"
    hessian.write_text("tampered\n", encoding="utf-8")

    result = _evaluate(paths)

    recovery = result["scores"]["actual_task_recovery"]
    assert recovery["components"]["hessian_completed"]["awarded"] == 0
    assert result["official_score"] == 0
    assert "hessian_completed" in result["eligibility"]["failed_gates"]

"""Security, evidence-gating, and determinism tests for the evaluator."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from evaluator.evaluate import (
    EvaluatorConfigurationError,
    evaluate_submission,
    load_config,
)
from tools.rdkit_tools import calculate_properties, evaluate_candidate


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task.yaml"
EVALUATOR_PATH = ROOT / "evaluator" / "evaluate.py"
IMPROVED_SMILES = "CCOC(=O)c1ccc(NC(=O)C)c(C)c1"
LOWER_FEASIBLE_SMILES = "COC(=O)c1ccc(NC(=O)C)cc1"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _candidate(name: str, smiles: str) -> dict[str, str]:
    return {
        "name": name,
        "smiles": smiles,
        "rationale": "A deterministic evaluator test candidate.",
    }


def _canonical(smiles: str, config: dict[str, Any]) -> str:
    result = evaluate_candidate(smiles, config)
    canonical = result["canonical_smiles"]
    assert result["ok"] is True and result["valid"] is True
    assert isinstance(canonical, str)
    return canonical


def _process_events(
    config: dict[str, Any], entries: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Create current-runner-shaped events from actual RDKit results."""

    events: list[dict[str, Any]] = [
        {"sequence": 1, "event": "run_start", "mode": "TEST"}
    ]
    for kind, smiles in entries:
        sequence = len(events) + 1
        if kind == "evaluate":
            events.append(
                {
                    "sequence": sequence,
                    "event": "tool_call",
                    "call_id": f"call-{sequence}",
                    "tool": "evaluate_candidate",
                    "arguments": {"smiles": smiles},
                    "status": "SUCCESS",
                    "result": evaluate_candidate(smiles, config),
                }
            )
        elif kind == "other":
            events.append(
                {
                    "sequence": sequence,
                    "event": "tool_call",
                    "call_id": f"call-{sequence}",
                    "tool": "calculate_properties",
                    "arguments": {"smiles": smiles},
                    "status": "SUCCESS",
                    "result": calculate_properties(smiles, config),
                }
            )
        else:  # pragma: no cover - test helper misuse
            raise AssertionError(kind)
    events.append(
        {"sequence": len(events) + 1, "event": "run_complete"}
    )
    return events


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
        newline="\n",
    )


@pytest.fixture()
def valid_submission(config: dict[str, Any]) -> dict[str, Any]:
    start = config["task"]["start_smiles"]
    start_canonical = _canonical(start, config)
    improved_canonical = _canonical(IMPROVED_SMILES, config)
    return {
        "schema_version": config["output"]["schema_version"],
        "task_id": config["task"]["id"],
        "final_candidate": _candidate("improved", IMPROVED_SMILES),
        "candidate_history": [
            _candidate("start", start),
            _candidate("improved", IMPROVED_SMILES),
        ],
        "run_metadata": {
            "tool_call_count": 3,
            "budget_exceeded": False,
            "unique_evaluated_candidates": [
                start_canonical,
                improved_canonical,
            ],
            "final_candidate_evaluated": True,
        },
    }


@pytest.fixture()
def valid_process_events(config: dict[str, Any]) -> list[dict[str, Any]]:
    return _process_events(
        config,
        [
            ("evaluate", config["task"]["start_smiles"]),
            ("other", config["task"]["start_smiles"]),
            ("evaluate", IMPROVED_SMILES),
        ],
    )


def test_normal_submission_is_recomputed_and_scored_from_process_evidence(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    result = evaluate_submission(
        valid_submission, config, process_events=valid_process_events
    )

    assert result["status"] == "ok"
    assert result["valid_submission"] is True
    assert result["hard_constraints_passed"] is True
    assert result["objective_passed"] is True
    assert result["process_eligible"] is True
    assert result["eligible"] is True
    assert 0.0 < result["score"] <= 100.0
    assert result["metrics"]["final_qed"] == result["final_candidate"]["qed"]
    assert result["metrics"]["tool_call_count"] == 3
    assert result["process_evidence"]["source"] == "validated_process_events"
    assert result["score"] == pytest.approx(
        sum(result["score_breakdown"].values()), abs=0.0002
    )


def test_final_qed_must_strictly_improve_even_when_other_gates_pass(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    tied = copy.deepcopy(valid_submission)
    tied["final_candidate"] = _candidate(
        "start_as_final", config["task"]["start_smiles"]
    )

    result = evaluate_submission(
        tied, config, process_events=valid_process_events
    )

    assert result["status"] == "ok"
    assert result["hard_constraints_passed"] is True
    assert result["process_eligible"] is True
    assert result["objective_passed"] is False
    assert result["metrics"]["final_qed"] == result["metrics"]["start_qed"]
    assert result["eligible"] is False
    assert result["score"] == 0.0
    assert result["total_score"] == 0.0
    assert any("strictly greater" in item for item in result["violations"])


def test_lower_qed_feasible_candidate_fails_objective_gate(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
) -> None:
    lower = copy.deepcopy(valid_submission)
    lower["final_candidate"] = _candidate(
        "lower_but_feasible", LOWER_FEASIBLE_SMILES
    )
    lower["candidate_history"][-1] = _candidate(
        "lower_but_feasible", LOWER_FEASIBLE_SMILES
    )
    lower["run_metadata"]["unique_evaluated_candidates"] = [
        _canonical(config["task"]["start_smiles"], config),
        _canonical(LOWER_FEASIBLE_SMILES, config),
    ]
    process_events = _process_events(
        config,
        [
            ("evaluate", config["task"]["start_smiles"]),
            ("other", config["task"]["start_smiles"]),
            ("evaluate", LOWER_FEASIBLE_SMILES),
        ],
    )

    result = evaluate_submission(lower, config, process_events=process_events)

    assert result["status"] == "ok"
    assert result["hard_constraints_passed"] is True
    assert result["process_eligible"] is True
    assert result["objective_passed"] is False
    assert result["metrics"]["final_qed"] < result["metrics"]["start_qed"]
    assert result["eligible"] is False
    assert result["score"] == 0.0
    assert result["total_score"] == 0.0
    assert any("strictly greater" in item for item in result["violations"])


def test_forged_metadata_without_evidence_cannot_earn_process_credit(
    config: dict[str, Any], valid_submission: dict[str, Any]
) -> None:
    forged = copy.deepcopy(valid_submission)
    forged["run_metadata"]["tool_call_count"] = config["budget"][
        "max_tool_calls"
    ]

    result = evaluate_submission(forged, config)
    diagnostic = evaluate_submission(
        forged, config, require_process_evidence=False
    )

    for value in (result, diagnostic):
        assert value["status"] == "ok"
        assert value["hard_constraints_passed"] is True
        assert value["process_eligible"] is False
        assert value["eligible"] is False
        assert value["score"] == 0.0
        assert value["metrics"]["tool_call_count"] == 0
        assert value["metrics"]["reported_tool_call_count"] == config["budget"][
            "max_tool_calls"
        ]
        assert any("process evidence" in item for item in value["violations"])
    assert diagnostic["process_evidence"]["required"] is False


def test_recomputed_ok_false_attempt_does_not_satisfy_success_minimum(
    config: dict[str, Any], valid_submission: dict[str, Any]
) -> None:
    events = _process_events(
        config,
        [
            ("evaluate", "not-a-smiles"),
            ("evaluate", config["task"]["start_smiles"]),
            ("evaluate", IMPROVED_SMILES),
        ],
    )
    result = evaluate_submission(
        valid_submission, config, process_events=events
    )

    assert result["process_evidence"]["tool_call_count"] == 3
    assert result["process_evidence"]["evaluate_candidate_attempt_count"] == 3
    assert result["process_evidence"]["successful_public_tool_call_count"] == 2
    assert (
        result["process_evidence"][
            "successful_evaluate_candidate_call_count"
        ]
        == 2
    )
    assert len(result["process_evidence"]["unique_evaluated_candidates"]) == 2
    assert result["process_evidence"]["public_tool_events_valid"] is True
    assert result["process_checks"]["minimum_calls_met"] is False
    assert result["process_eligible"] is False
    assert result["score"] == 0.0


def test_tampered_result_and_reported_count_are_gated(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    tampered_events = copy.deepcopy(valid_process_events)
    final_evaluation = next(
        event
        for event in reversed(tampered_events)
        if event.get("tool") == "evaluate_candidate"
    )
    final_evaluation["result"]["canonical_smiles"] = "CCO"
    tampered = evaluate_submission(
        valid_submission, config, process_events=tampered_events
    )
    assert tampered["process_evidence"]["valid"] is True
    assert tampered["process_evidence"]["public_tool_events_valid"] is False
    assert tampered["process_eligible"] is False
    assert tampered["score"] == 0.0

    count_forgery = copy.deepcopy(valid_submission)
    count_forgery["run_metadata"]["tool_call_count"] = 4
    count_result = evaluate_submission(
        count_forgery, config, process_events=valid_process_events
    )
    assert count_result["process_evidence"]["valid"] is True
    assert (
        count_result["process_checks"][
            "process_tool_call_count_matches_metadata"
        ]
        is False
    )
    assert count_result["process_eligible"] is False
    assert count_result["score"] == 0.0


def test_unknown_success_tool_is_budgeted_but_cannot_satisfy_minimum_calls(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    events = copy.deepcopy(valid_process_events)
    other = next(
        event
        for event in events
        if event.get("tool") == "calculate_properties"
    )
    other["tool"] = "invented_success_tool"

    result = evaluate_submission(
        valid_submission, config, process_events=events
    )

    assert result["process_evidence"]["tool_call_count"] == 3
    assert result["process_evidence"]["successful_public_tool_call_count"] == 2
    assert result["process_evidence"]["public_tool_events_valid"] is False
    assert result["process_checks"]["minimum_calls_met"] is False
    assert result["process_eligible"] is False


@pytest.mark.parametrize("damage", ["arguments", "result", "truncated_result"])
def test_incomplete_success_event_cannot_qualify_as_public_execution(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
    damage: str,
) -> None:
    events = copy.deepcopy(valid_process_events)
    other = next(
        event
        for event in events
        if event.get("tool") == "calculate_properties"
    )
    if damage in {"arguments", "result"}:
        other.pop(damage)
    else:
        other["result"] = {"ok": True}

    result = evaluate_submission(
        valid_submission, config, process_events=events
    )

    assert result["process_evidence"]["tool_call_count"] == 3
    assert result["process_evidence"]["successful_public_tool_call_count"] == 2
    assert result["process_evidence"]["public_tool_events_valid"] is False
    assert result["process_checks"]["minimum_calls_met"] is False
    assert result["process_eligible"] is False


def test_tool_only_and_post_terminal_trajectories_fail_lifecycle_gate(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    tool_only = copy.deepcopy(valid_process_events[1:-1])
    for sequence, event in enumerate(tool_only, 1):
        event["sequence"] = sequence
    tool_only_result = evaluate_submission(
        valid_submission, config, process_events=tool_only
    )
    assert tool_only_result["process_evidence"]["lifecycle_valid"] is False
    assert tool_only_result["process_evidence"][
        "successful_public_tool_call_count"
    ] == 3
    assert tool_only_result["process_eligible"] is False

    post_terminal = copy.deepcopy(valid_process_events)
    post_terminal.append(
        {
            "sequence": len(post_terminal) + 1,
            "event": "message",
            "message": {"role": "assistant", "content": "late"},
        }
    )
    post_terminal_result = evaluate_submission(
        valid_submission, config, process_events=post_terminal
    )
    assert post_terminal_result["process_evidence"]["lifecycle_valid"] is False
    assert any(
        "after terminal" in issue
        for issue in post_terminal_result["process_evidence"]["issues"]
    )
    assert post_terminal_result["process_eligible"] is False


def test_terminal_run_failure_is_valid_evidence_but_ineligible(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    events = copy.deepcopy(valid_process_events)
    events[-1] = {
        "sequence": events[-1]["sequence"],
        "event": "run_failure",
        "terminal_status": "FAILED",
        "error": {"type": "RuntimeError", "message": "controlled failure"},
    }

    result = evaluate_submission(
        valid_submission, config, process_events=events
    )

    assert result["process_evidence"]["valid"] is True
    assert result["process_evidence"]["lifecycle_valid"] is True
    assert result["process_evidence"]["terminal_event"] == "run_failure"
    assert result["process_evidence"]["run_completed"] is False
    assert result["process_eligible"] is False


def test_config_path_must_be_regular_and_confined_to_project(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-task.yaml"
    outside.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(EvaluatorConfigurationError, match="inside"):
        load_config(outside)
    with pytest.raises(EvaluatorConfigurationError, match="regular file"):
        load_config(ROOT / "configs")


def test_minimum_unique_successful_candidates_is_enforced(
    config: dict[str, Any]
) -> None:
    start = config["task"]["start_smiles"]
    canonical = _canonical(start, config)
    submission = {
        "schema_version": config["output"]["schema_version"],
        "task_id": config["task"]["id"],
        "final_candidate": _candidate("start", start),
        "candidate_history": [_candidate("start", start)],
        "run_metadata": {
            "tool_call_count": 3,
            "budget_exceeded": False,
            "unique_evaluated_candidates": [canonical],
            "final_candidate_evaluated": True,
        },
    }
    events = _process_events(
        config,
        [("evaluate", start), ("other", start), ("evaluate", start)],
    )

    result = evaluate_submission(submission, config, process_events=events)

    assert (
        result["process_checks"][
            "minimum_unique_evaluated_candidates_met"
        ]
        is False
    )
    assert result["process_eligible"] is False


def test_trajectory_auto_discovery_and_explicit_cli(
    tmp_path: Path,
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    submission_path = tmp_path / "final_answer.json"
    trajectory_path = tmp_path / "trajectory.jsonl"
    output_path = tmp_path / "evaluation.json"
    submission_path.write_text(
        json.dumps(valid_submission), encoding="utf-8"
    )
    _write_jsonl(trajectory_path, valid_process_events)

    automatic = evaluate_submission(submission_path, config)
    in_memory = evaluate_submission(
        valid_submission, config, process_events=valid_process_events
    )
    assert automatic["process_eligible"] is True
    assert automatic["process_evidence"]["source"] == "validated_process_events"
    assert automatic == in_memory

    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR_PATH),
            "--submission",
            str(submission_path),
            "--trajectory",
            str(trajectory_path),
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "process_eligible"
    ] is True


def test_trajectory_duplicate_keys_and_outside_path_fail_safely(
    tmp_path: Path,
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    submission_dir = tmp_path / "submission"
    outside_dir = tmp_path / "outside"
    submission_dir.mkdir()
    outside_dir.mkdir()
    submission_path = submission_dir / "final_answer.json"
    submission_path.write_text(
        json.dumps(valid_submission), encoding="utf-8"
    )

    duplicate_path = submission_dir / "duplicate.jsonl"
    duplicate_path.write_text(
        '{"sequence":1,"sequence":2,"event":"run_start"}\n',
        encoding="utf-8",
    )
    duplicate = evaluate_submission(
        submission_path, config, trajectory_path=duplicate_path
    )
    assert duplicate["status"] == "ok"
    assert duplicate["process_evidence"]["valid"] is False
    assert "duplicate" in duplicate["process_evidence"]["errors"][0]

    outside_path = outside_dir / "trajectory.jsonl"
    _write_jsonl(outside_path, valid_process_events)
    outside = evaluate_submission(
        submission_path, config, trajectory_path=outside_path
    )
    assert outside["status"] == "ok"
    assert outside["process_eligible"] is False
    assert "submission directory" in outside["process_evidence"]["errors"][0]


def test_malformed_json_cli_always_writes_failure_document(
    tmp_path: Path,
) -> None:
    submission_path = tmp_path / "broken.json"
    output_path = tmp_path / "evaluation.json"
    submission_path.write_text('{"task_id": ', encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(EVALUATOR_PATH),
            "--submission",
            str(submission_path),
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "invalid_submission"
    assert result["total_score"] == 0.0
    assert result["errors"]


def test_missing_or_unexpected_fields_are_rejected(
    config: dict[str, Any], valid_submission: dict[str, Any]
) -> None:
    missing = copy.deepcopy(valid_submission)
    del missing["final_candidate"]["rationale"]
    assert evaluate_submission(missing, config)["status"] == "invalid_submission"

    injected = copy.deepcopy(valid_submission)
    injected["final_candidate"]["qed"] = 1.0
    assert evaluate_submission(injected, config)["status"] == "invalid_submission"


def test_canonical_duplicate_history_is_detected_and_gated(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    duplicate = copy.deepcopy(valid_submission)
    duplicate["candidate_history"].append(
        _candidate(
            "same graph alternate SMILES",
            "CC(=O)Nc1ccc(C(=O)OCC)cc1",
        )
    )

    result = evaluate_submission(
        duplicate, config, process_events=valid_process_events
    )

    assert result["duplicate_candidates"]
    assert (
        result["process_checks"]["history_candidates_canonically_unique"]
        is False
    )
    assert result["process_eligible"] is False
    assert result["score"] == 0.0


def test_budget_rejection_is_derived_and_gated(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    events = copy.deepcopy(valid_process_events[:-1])
    counted = 3
    while counted < config["budget"]["max_tool_calls"]:
        sequence = len(events) + 1
        events.append(
            {
                "sequence": sequence,
                "event": "tool_call",
                "call_id": f"budget-fill-{sequence}",
                "tool": "calculate_qed",
                "arguments": {"smiles": IMPROVED_SMILES},
                "status": "SUCCESS",
                "result": {"ok": True},
            }
        )
        counted += 1
    sequence = len(events) + 1
    events.append(
        {
            "sequence": sequence,
            "event": "tool_call",
            "call_id": "rejected-over-budget",
            "tool": "calculate_qed",
            "status": "REJECTED",
            "error": {"type": "ToolBudgetExceeded", "message": "guarded"},
        }
    )
    events.append(
        {"sequence": len(events) + 1, "event": "run_complete"}
    )
    submission = copy.deepcopy(valid_submission)
    submission["run_metadata"]["tool_call_count"] = counted
    submission["run_metadata"]["budget_exceeded"] = True

    result = evaluate_submission(submission, config, process_events=events)

    assert result["process_evidence"]["tool_call_count"] == counted
    assert result["process_evidence"]["rejected_tool_call_count"] == 1
    assert result["process_evidence"]["budget_exceeded"] is True
    assert result["process_checks"]["within_tool_budget"] is False
    assert result["process_eligible"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value, _config: value["run_metadata"].__setitem__(
            "tool_call_count", True
        ),
        lambda value, _config: value["final_candidate"].__setitem__(
            "smiles", ["not", "a", "string"]
        ),
        lambda value, config: value["final_candidate"].__setitem__(
            "rationale", "x" * (config["output"]["max_rationale_length"] + 1)
        ),
        lambda value, _config: value.__setitem__("unknown_payload", {}),
    ],
)
def test_unsafe_types_and_payload_shapes_fail_safely(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    mutation: Any,
) -> None:
    unsafe = copy.deepcopy(valid_submission)
    mutation(unsafe, config)
    result = evaluate_submission(unsafe, config)
    assert result["status"] == "invalid_submission"
    assert result["valid_submission"] is False
    assert result["score"] == 0.0


def test_oversized_submission_file_fails_before_json_parsing(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (config["output"]["max_submission_bytes"] + 1))

    result = evaluate_submission(oversized, config)

    assert result["status"] == "invalid_submission"
    assert result["score"] == 0.0
    assert "maximum size" in result["errors"][0]


def test_hard_constraint_failure_gates_score_with_valid_process(
    config: dict[str, Any]
) -> None:
    start = config["task"]["start_smiles"]
    ethanol = "CCO"
    submission = {
        "schema_version": config["output"]["schema_version"],
        "task_id": config["task"]["id"],
        "final_candidate": _candidate("ethanol", ethanol),
        "candidate_history": [
            _candidate("start", start),
            _candidate("ethanol", ethanol),
        ],
        "run_metadata": {
            "tool_call_count": 3,
            "budget_exceeded": False,
            "unique_evaluated_candidates": [
                _canonical(start, config),
                _canonical(ethanol, config),
            ],
            "final_candidate_evaluated": True,
        },
    }
    events = _process_events(
        config,
        [("evaluate", start), ("other", start), ("evaluate", ethanol)],
    )

    result = evaluate_submission(submission, config, process_events=events)

    assert result["status"] == "ok"
    assert result["process_eligible"] is True
    assert result["final_candidate"]["valid"] is True
    assert result["hard_constraints_passed"] is False
    assert result["score"] == 0.0


def test_task_id_and_final_evaluated_metadata_are_enforced(
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    wrong_task = copy.deepcopy(valid_submission)
    wrong_task["task_id"] = "a-different-task"
    assert evaluate_submission(wrong_task, config)["status"] == "invalid_submission"

    not_evaluated = copy.deepcopy(valid_submission)
    not_evaluated["run_metadata"]["final_candidate_evaluated"] = False
    result = evaluate_submission(
        not_evaluated, config, process_events=valid_process_events
    )
    assert result["process_checks"][
        "process_final_evaluation_matches_metadata"
    ] is False
    assert result["process_eligible"] is False


def test_repeated_evaluation_is_byte_deterministic(
    tmp_path: Path,
    config: dict[str, Any],
    valid_submission: dict[str, Any],
    valid_process_events: list[dict[str, Any]],
) -> None:
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    first = evaluate_submission(
        valid_submission,
        config,
        first_output,
        process_events=valid_process_events,
    )
    second = evaluate_submission(
        valid_submission,
        config,
        second_output,
        process_events=valid_process_events,
    )

    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()

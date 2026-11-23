from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from baselines.baseline_agent import run_baseline
from evaluator.evaluate import evaluate_submission
from tools import evaluate_candidate


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_baseline_is_deterministic_budgeted_and_schema_compliant() -> None:
    config = _config()
    first_events: list[dict] = []
    second_events: list[dict] = []
    first = run_baseline(config, process_events=first_events)
    second = run_baseline(config, process_events=second_events)

    assert first == second
    assert first_events == second_events
    assert set(first) == {
        "schema_version",
        "task_id",
        "final_candidate",
        "candidate_history",
        "run_metadata",
    }
    assert set(first["final_candidate"]) == {"name", "smiles", "rationale"}
    assert set(first["run_metadata"]) == {
        "tool_call_count",
        "budget_exceeded",
        "unique_evaluated_candidates",
        "final_candidate_evaluated",
    }
    assert first["schema_version"] == config["output"]["schema_version"]
    assert first["task_id"] == config["task"]["id"]
    assert first["run_metadata"]["tool_call_count"] == config["budget"][
        "max_unique_evaluated_candidates"
    ]
    assert first["run_metadata"]["budget_exceeded"] is False
    assert first["run_metadata"]["final_candidate_evaluated"] is True

    unique = first["run_metadata"]["unique_evaluated_candidates"]
    assert len(unique) == len(set(unique)) == len(first["candidate_history"])
    assert [item["smiles"] for item in first["candidate_history"]] == unique
    assert first_events[0]["provenance"] == "OFFLINE_BASELINE"
    assert first_events[0]["is_llm"] is False
    assert sum(event.get("event") == "tool_call" for event in first_events) == config[
        "budget"
    ]["max_unique_evaluated_candidates"]


def test_baseline_finds_a_feasible_measured_improvement() -> None:
    config = _config()
    result = run_baseline(config)
    start_eval = evaluate_candidate(config["task"]["start_smiles"], config)
    final_eval = evaluate_candidate(result["final_candidate"]["smiles"], config)

    assert start_eval["feasible"] is True
    assert final_eval["feasible"] is True
    assert final_eval["qed"] > start_eval["qed"]
    assert result["final_candidate"]["smiles"] in result["run_metadata"][
        "unique_evaluated_candidates"
    ]
    assert any(
        item["smiles"] != config["task"]["start_smiles"]
        for item in result["candidate_history"]
    )


def test_baseline_respects_a_smaller_candidate_budget() -> None:
    config = _config()
    config["budget"]["max_tool_calls"] = 4
    config["budget"]["max_unique_evaluated_candidates"] = 4
    result = run_baseline(config)

    assert result["run_metadata"]["tool_call_count"] <= 4
    assert len(result["run_metadata"]["unique_evaluated_candidates"]) <= 4
    assert result["run_metadata"]["budget_exceeded"] is False
    assert result["run_metadata"]["final_candidate_evaluated"] is True


def test_baseline_respects_public_halogen_cap() -> None:
    config = _config()
    result = run_baseline(config)

    assert all(
        evaluate_candidate(smiles, config)["properties"]["halogen_count"] == 0
        for smiles in result["run_metadata"]["unique_evaluated_candidates"]
    )


def test_direct_cli_writes_runner_friendly_final_answer(tmp_path: Path) -> None:
    script = ROOT / "baselines" / "baseline_agent.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    destination = tmp_path / "final_answer.json"
    trajectory = tmp_path / "trajectory.jsonl"
    assert destination.is_file()
    assert trajectory.is_file()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == run_baseline(_config())
    assert str(destination) in completed.stdout
    events = [
        json.loads(line)
        for line in trajectory.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["execution_mode"] == "OFFLINE_BASELINE"
    assert events[0]["provider"] == "OFFLINE_BASELINE"
    assert events[0]["model"] == "OFFLINE_BASELINE"
    assert events[0]["is_llm"] is False
    assert all(
        event.get("provenance") == "OFFLINE_BASELINE" for event in events
    )
    tool_events = [event for event in events if event["event"] == "tool_call"]
    assert len(tool_events) == payload["run_metadata"]["tool_call_count"]
    assert all(event["tool"] == "evaluate_candidate" for event in tool_events)
    assert all(event["status"] == "SUCCESS" for event in tool_events)

    reevaluated = evaluate_submission(destination, CONFIG_PATH)
    assert reevaluated["status"] == "ok"
    assert (
        reevaluated["process_evidence"]["source"]
        == "validated_process_events"
    )
    assert reevaluated["process_evidence"]["lifecycle_valid"] is True
    assert reevaluated["process_evidence"]["run_completed"] is True
    assert (
        reevaluated["process_evidence"]["successful_public_tool_call_count"]
        == len(tool_events)
    )
    assert reevaluated["process_eligible"] is True
    assert reevaluated["eligible"] is True

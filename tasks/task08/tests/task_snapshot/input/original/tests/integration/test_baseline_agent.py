from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.baseline_agent import (
    BaselineError,
    _special_event_payload,
    run_live,
    run_replay,
)
from evaluator.evaluate import evaluation_bytes


def test_live_refuses_to_overwrite_before_constructing_registry(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    called = False

    def registry_factory(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("registry must not be constructed")

    with pytest.raises(FileExistsError, match="REFUSING_TO_OVERWRITE"):
        run_live(existing, registry_factory=registry_factory)
    assert called is False


def test_live_rejects_new_output_outside_baselines(tmp_path: Path) -> None:
    target = tmp_path / "new-output"
    with pytest.raises(BaselineError, match="OUTPUT_MUST_BE_INSIDE_BASELINES"):
        run_live(target)
    assert not target.exists()


def test_special_event_projection_has_frozen_contract_fields() -> None:
    payload = _special_event_payload(
        "prepare_restart",
        {
            "attempt_id": "attempt_001",
            "phase": "geometry_optimization",
            "state": "PREPARED",
            "source_geometry": {
                "path": "attempts/attempt_000_initial_failure/xtbopt.log",
                "frame_index": 0,
                "sha256": "A" * 64,
                "provenance_state": "CURRENT_TRAJECTORY_FRAME",
            },
            "prepared_command_path": "attempts/attempt_001/prepared_command.json",
            "input_manifest_path": "attempts/attempt_001/input_manifest.json",
        },
    )
    assert set(payload) == {
        "attempt_id",
        "phase",
        "state",
        "source_geometry",
        "prepared_command_path",
        "input_manifest_path",
    }
    assert payload["source_geometry"]["frame_index"] == 0


def test_replay_calls_evaluator_twice_and_requires_stored_bytes(tmp_path: Path) -> None:
    source = tmp_path / "frozen"
    workspace = source / "workspace"
    workspace.mkdir(parents=True)
    (source / "final_answer.json").write_text("{}\n", encoding="utf-8", newline="\n")
    (source / "trajectory.jsonl").write_text("{}\n", encoding="utf-8", newline="\n")
    result = {
        "schema_version": "1.0",
        "evaluation_status": "COMPLETED",
        "eligibility": {"eligible": True},
        "official_score": 100,
    }
    (source / "evaluation.json").write_bytes(evaluation_bytes(result))
    calls = []

    def evaluator(**kwargs):
        calls.append(kwargs)
        return json.loads(json.dumps(result))

    replay = run_replay(source, evaluator_fn=evaluator)
    assert len(calls) == 2
    assert replay["byte_identical"] is True
    assert replay["official_score"] == 100


def test_replay_rejects_changed_stored_evaluation(tmp_path: Path) -> None:
    source = tmp_path / "frozen"
    (source / "workspace").mkdir(parents=True)
    (source / "final_answer.json").write_text("{}\n", encoding="utf-8")
    (source / "trajectory.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "evaluation.json").write_text("{}\n", encoding="utf-8")
    result = {"evaluation_status": "COMPLETED", "official_score": 100}

    with pytest.raises(BaselineError, match="FROZEN_EVALUATION_MISMATCH"):
        run_replay(source, evaluator_fn=lambda **_kwargs: result)

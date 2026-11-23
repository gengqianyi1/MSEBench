from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from evaluator.evaluate import evaluate_files
from evaluator.trusted_validation import canonical_json_bytes
from tests.security.test_tool_security import _registry
from tests.unit.test_evaluator_helpers import (
    build_complete_fixture,
    refresh_final_answer_envelope,
    rehash_events,
)
from tests.unit.test_xtb_tools import FakeXtb, _prepare_args, _project


def _evaluate(paths: dict[str, Path]) -> dict:
    return evaluate_files(
        final_answer_path=paths["final"],
        trajectory_path=paths["trajectory"],
        artifacts_path=paths["case"],
        config_path=paths["config"],
    )


def _rehash_without_resequencing(path: Path) -> None:
    """Repair only hashes so a deliberately bad sequence remains isolated."""

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    previous = "0" * 64
    for event in events:
        event["prev_event_hash"] = previous
        event.pop("event_hash", None)
        event["event_hash"] = hashlib.sha256(canonical_json_bytes(event)).hexdigest()
        previous = event["event_hash"]
    path.write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in events))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_non_contiguous_sequence_is_rejected_even_with_repaired_hashes(
    tmp_path: Path,
) -> None:
    paths = build_complete_fixture(tmp_path)
    events = [
        json.loads(line)
        for line in paths["trajectory"].read_text(encoding="utf-8").splitlines()
    ]
    events[3]["sequence"] += 7
    paths["trajectory"].write_bytes(
        b"".join(canonical_json_bytes(event) + b"\n" for event in events)
    )
    _rehash_without_resequencing(paths["trajectory"])

    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert "trace_hash_chain_valid" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "TRACE_SEQUENCE" for issue in result["issues"])


def test_multiple_terminal_events_are_rejected(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)

    def duplicate_terminal(events: list[dict]) -> None:
        assert events[-1]["event"] == "run_complete"
        events.insert(-1, copy.deepcopy(events[-1]))

    rehash_events(paths["trajectory"], duplicate_terminal)
    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert "complete_lifecycle" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "TRACE_LIFECYCLE" for issue in result["issues"])


def test_missing_run_start_is_rejected(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)

    def remove_start(events: list[dict]) -> None:
        assert events[0]["event"] == "run_start"
        events.pop(0)

    rehash_events(paths["trajectory"], remove_start)
    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert "complete_lifecycle" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "TRACE_LIFECYCLE" for issue in result["issues"])


def test_explicit_fallback_marker_fails_with_an_exact_model_name(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)

    def mark_fallback(events: list[dict]) -> None:
        response = next(
            event for event in events if event["event"] == "provider_response_metadata"
        )
        response["payload"]["fallback"] = True

    rehash_events(paths["trajectory"], mark_fallback)
    refresh_final_answer_envelope(paths)
    result = _evaluate(paths)

    assert result["trusted_evidence"]["run"]["exact_model_identity"] is True
    assert result["official_score"] == 0
    assert "no_fallback_or_manual_rewrite" in result["eligibility"]["failed_gates"]
    assert result["trusted_evidence"]["security"]["fallback_or_manual_marker"] is True


def test_replay_cannot_claim_real_api_mode(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)

    def lie_about_replay(events: list[dict]) -> None:
        start = events[0]["payload"]
        start["run_type"] = "REPLAY"
        start["execution_mode"] = "REAL_API"
        start["native_mode"] = "LIVE_XTB"

    rehash_events(paths["trajectory"], lie_about_replay)
    refresh_final_answer_envelope(paths)
    result = _evaluate(paths)

    assert result["eligibility"]["identity_gate_exempt"] is True
    assert result["trusted_evidence"]["run"]["mode_truthful"] is False
    assert result["official_score"] == 0
    assert "formal_mode_truthful" in result["eligibility"]["failed_gates"]


def test_mock_transport_marker_cannot_hide_behind_live_mode_labels(
    tmp_path: Path,
) -> None:
    paths = build_complete_fixture(tmp_path)

    def mark_mock_transport(events: list[dict]) -> None:
        response = next(
            event for event in events if event["event"] == "provider_response_metadata"
        )
        response["payload"]["transport"] = "scripted_stub"

    rehash_events(paths["trajectory"], mark_mock_transport)
    refresh_final_answer_envelope(paths)
    result = _evaluate(paths)

    assert result["trusted_evidence"]["run"]["execution_mode"] == "REAL_API"
    assert result["trusted_evidence"]["run"]["native_mode"] == "LIVE_XTB"
    assert result["official_score"] == 0
    assert "no_fallback_or_manual_rewrite" in result["eligibility"]["failed_gates"]
    assert result["trusted_evidence"]["security"]["fallback_or_manual_marker"] is True


def test_project_source_files_are_outside_the_model_read_root(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    for index, path in enumerate(
        (
            "../framework/agent_runner.py",
            "../evaluator/evaluate.py",
            "../scripts/run_model.py",
            "../configs/tool_contracts.json",
        )
    ):
        result = registry.dispatch(
            "read_text_range",
            {"path": path, "start_line": 1, "end_line": 1},
            call_id=f"project_source_{index}",
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "PARENT_PATH_REJECTED"


def test_reserved_attempt_zero_old_attempts_and_failure_logs_are_immutable(
    tmp_path: Path,
) -> None:
    registry = _project(
        tmp_path,
        FakeXtb(),
        include_initial_failure_trajectory=True,
    )
    initial_log = (
        registry.sandbox.root
        / "attempts"
        / "attempt_000_initial_failure"
        / "xtbopt.log"
    )
    initial_bytes = initial_log.read_bytes()

    reserved_run = registry.dispatch(
        "run_xtb", {"attempt_id": "attempt_000"}, call_id="reserved_run"
    )
    reserved_validate = registry.dispatch(
        "validate_attempt", {"attempt_id": "attempt_000"}, call_id="reserved_validate"
    )
    assert reserved_run["ok"] is False
    assert reserved_validate["ok"] is False
    assert reserved_run["error"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert reserved_validate["error"]["code"] == "SCHEMA_VALIDATION_FAILED"

    first = registry.dispatch(
        "prepare_restart",
        _prepare_args("geometry_optimization", "input.xyz", "original_input"),
        call_id="prepare_first",
    )
    assert first["ok"] is True
    assert first["result"]["attempt_id"] == "attempt_001"
    first_root = registry.sandbox.root / "attempts" / "attempt_001"
    first_snapshot = _tree_bytes(first_root)

    second_args = _prepare_args("geometry_optimization", "input.xyz", "original_input")
    second_args["optimization_cycle_limit"] = 120
    second = registry.dispatch(
        "prepare_restart", second_args, call_id="prepare_second"
    )
    assert second["ok"] is True
    assert second["result"]["attempt_id"] == "attempt_002"

    for index, tool_name in enumerate(("delete_file", "remove_attempt", "write_text")):
        denied = registry.dispatch(tool_name, {}, call_id=f"generic_mutator_{index}")
        assert denied["ok"] is False
        assert denied["error"]["code"] == "UNKNOWN_TOOL"

    assert _tree_bytes(first_root) == first_snapshot
    assert initial_log.read_bytes() == initial_bytes
    assert not (registry.sandbox.root / "attempts" / "attempt_000").exists()

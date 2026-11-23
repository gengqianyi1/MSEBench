from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from framework import (
    GENESIS_HASH,
    AgentRunner,
    ConfigurationError,
    LifecycleError,
    ModelConfig,
    ProtocolConfig,
    ScriptedTransport,
    SiliconFlowChatTransport,
    TrajectoryRecorder,
    canonical_json,
    canonical_sha256,
    render_dialogue,
    validate_hash_chain,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL = "Pro/moonshotai/Kimi-K2.6"


def _task(*, minimum_successes: int = 1, retries: int = 0) -> dict:
    value = yaml.safe_load((ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    value = copy.deepcopy(value)
    value["budget"]["min_successful_tool_calls"] = minimum_successes
    value["budget"]["max_provider_retries"] = retries
    return value


def _model() -> ModelConfig:
    return ModelConfig.load_registry(ROOT / "configs" / "models.example.yaml", "kimi")


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _response(request_id: str, tool_calls: list[dict], *, model: str = MODEL) -> dict:
    return {
        "id": request_id,
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
            }
        ],
    }


def _final_answer() -> dict:
    return {
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "final_status": "STOPPED_UNSAFE",
        "diagnosis": {
            "primary_category": "UNKNOWN",
            "primary_cause_code": "UNKNOWN",
            "scheduler_assessment": "UNKNOWN",
            "conflict_resolution": "UNRESOLVED",
            "legacy_disposition": "NOT_INSPECTED",
            "secondary_findings": [],
            "confidence": 0.25,
            "evidence": [
                {
                    "tool_call_id": "call-001",
                    "path": "initial_failure/stdout.log",
                    "line_start": 1,
                    "line_end": 1,
                    "content_sha256": "a" * 64,
                    "interpretation": "The bounded stub run has one inspected evidence reference.",
                }
            ],
        },
        "restart_summary": {
            "starting_geometry_source_kind": "NONE",
            "starting_geometry_source": {
                "attempt_id": "attempt_000_initial_failure",
                "path": "initial_failure/input.xyz",
                "frame_index": 0,
                "file_sha256": "a" * 64,
                "frame_sha256": "a" * 64,
            },
            "changes": [],
            "preserved_invariants": [],
        },
        "result_summary": {
            "optimization_completed": False,
            "hessian_completed": False,
            "final_attempt_id": "",
            "warnings": ["No native execution was requested in the deterministic stub run."],
        },
        "rationale": "Stop safely because this unit test only verifies the protocol lifecycle.",
    }


def _load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_canonical_chain_is_contiguous_tamper_evident_and_frozen() -> None:
    assert canonical_json({"timeout": 120.0, "temperature": 0.2}) == (
        '{"temperature":0.2,"timeout":120}'
    )
    recorder = TrajectoryRecorder(
        protocol=ProtocolConfig("xtb_failure_recovery_tool_loop", "1.0"),
        run_id="unit-chain",
        timestamp_fn=lambda: "2026-01-01T00:00:00Z",
    )
    first = recorder.append("run_start", {"logical_model": "kimi"})
    assert first["prev_event_hash"] == GENESIS_HASH
    recorder.append("final_submitted", {"final_answer": {"status": "STOPPED_UNSAFE"}})
    recorder.append("run_complete", {"status": "COMPLETE"})
    events = recorder.events
    validate_hash_chain(events)
    assert events[-1]["event_hash"] == canonical_sha256(
        {key: value for key, value in events[-1].items() if key != "event_hash"}
    )
    with pytest.raises(LifecycleError):
        recorder.append("message", {"message": {"role": "assistant", "content": "late"}})
    events[0]["payload"]["logical_model"] = "changed"
    with pytest.raises(LifecycleError):
        validate_hash_chain(events)


def test_adapter_disables_sdk_retries_and_sends_exact_model() -> None:
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return {"ok": True}

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": Completions()})()

    def factory(**kwargs):
        captured["client"] = kwargs
        return Client()

    adapter = SiliconFlowChatTransport(_model(), "test-only-key", client_factory=factory)
    adapter.complete(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
        timeout_seconds=5,
    )
    assert captured["client"]["max_retries"] == 0
    assert captured["request"]["model"] == MODEL
    assert captured["request"]["stream"] is False
    assert "tool_choice" not in captured["request"]


def test_runner_exposes_every_enforced_public_resource_limit(tmp_path: Path) -> None:
    runner = AgentRunner(
        task_config=_task(),
        model_config=_model(),
        logical_model="kimi",
        output_dir=tmp_path / "unused-output",
        prompt="Use the strict tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry={},
        transport=ScriptedTransport([]),
    )
    payload = json.loads(runner._initial_messages()[1]["content"])
    assert payload["budget"]["single_xtb_timeout_seconds"] == 240
    assert payload["budget"]["max_text_read_lines_per_call"] == 160
    assert payload["restart_resource_limits"] == {
        "optimization_cycle_limit_min": 2,
        "optimization_cycle_limit_max": 200,
        "thread_count_min": 1,
        "thread_count_max": 1,
        "walltime_seconds_max": 240,
    }


def test_read_tool_message_supplies_exact_copyable_evidence_reference() -> None:
    message = AgentRunner._tool_message(
        "functions.read_text_range:16",
        "read_text_range",
        True,
        {
            "result": {
                "path": "attempts/attempt_000_initial_failure/stdout.log",
                "line_start": 284,
                "line_end": 284,
                "content_sha256": "a" * 64,
                "requested_end_line": 400,
            }
        },
    )
    content = json.loads(message["content"])
    assert content["call_id"] == "functions.read_text_range:16"
    assert content["evidence_reference"] == {
        "tool_call_id": "functions.read_text_range:16",
        "path": "attempts/attempt_000_initial_failure/stdout.log",
        "line_start": 284,
        "line_end": 284,
        "content_sha256": "a" * 64,
    }


def test_runner_writes_exact_four_outputs_and_freezes_after_final(tmp_path: Path) -> None:
    answer = _final_answer()
    # Integral floats canonicalize to JSON integers.  The dialogue must be
    # rendered from the persisted canonical events, not pre-write objects.
    answer["diagnosis"]["confidence"] = 1.0
    transport = ScriptedTransport(
        [
            _response("req-1", [_call("call-001", "list_case_files", {})]),
            _response("req-2", [_call("call-002", "submit_final", answer)]),
        ]
    )
    registry = {
        "list_case_files": lambda: {"ok": True, "files": []},
        "submit_final": lambda **_kwargs: {"ok": True, "status": "SUBMITTED"},
    }
    destination = tmp_path / "formal-shape"
    result = AgentRunner(
        task_config=_task(),
        model_config=_model(),
        logical_model="kimi",
        output_dir=destination,
        prompt="Use the strict tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry=registry,
        transport=transport,
        run_id="unit-success",
        timestamp_fn=lambda: "2026-01-01T00:00:00Z",
        sleep_fn=lambda _seconds: None,
    ).run()

    assert result.complete
    assert {item.name for item in destination.iterdir()} == {
        "trajectory.jsonl",
        "dialogue.md",
        "metadata.json",
        "final_answer.json",
    }
    persisted = json.loads(
        (destination / "final_answer.json").read_text(encoding="utf-8")
    )
    assert persisted["model_submission"] == answer
    assert persisted["schema_version"] == "1.0"
    assert persisted["trusted_runtime"]["trace"]["trace_sha256"] == hashlib.sha256(
        (destination / "trajectory.jsonl").read_bytes()
    ).hexdigest().upper()
    events = _load_events(destination / "trajectory.jsonl")
    validate_hash_chain(events)
    assert (destination / "dialogue.md").read_text(encoding="utf-8") == render_dialogue(events)
    assert events[-2]["event"] == "final_submitted"
    assert events[-1]["event"] == "run_complete"
    assert sum(event["event"] == "final_submitted" for event in events) == 1
    assert events[-2]["payload"]["final_answer"] == answer
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["requested_model"] == MODEL == metadata["actual_model"]
    assert metadata["hash_chain"]["valid"] is True


def test_real_api_mode_rejects_injected_transport(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="forbids injected transports"):
        AgentRunner(
            task_config=_task(),
            model_config=_model(),
            logical_model="kimi",
            output_dir=tmp_path / "real-api-stub",
            prompt="Use tools.",
            tool_contracts=ROOT / "configs" / "tool_contracts.json",
            tool_registry={"submit_final": lambda **_kwargs: {}},
            transport=ScriptedTransport([]),
            execution_mode="REAL_API",
            native_mode="NO_NATIVE_EXECUTION",
            run_type="TEST",
        )


def test_live_xtb_mode_rejects_mapping_registry(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="live native tool registry"):
        AgentRunner(
            task_config=_task(),
            model_config=_model(),
            logical_model="kimi",
            output_dir=tmp_path / "live-xtb-mock",
            prompt="Use tools.",
            tool_contracts=ROOT / "configs" / "tool_contracts.json",
            tool_registry={"submit_final": lambda **_kwargs: {}},
            transport=ScriptedTransport([]),
            execution_mode="TEST_STUB",
            native_mode="LIVE_XTB",
            run_type="TEST",
        )


def test_runner_owns_bounded_provider_retry_and_records_budget_state(tmp_path: Path) -> None:
    answer = _final_answer()
    transport = ScriptedTransport(
        [
            RuntimeError("opaque provider detail"),
            _response("req-after-retry", [_call("call-001", "list_case_files", {})]),
            _response("req-final", [_call("call-final", "submit_final", answer)]),
        ]
    )
    destination = tmp_path / "runner-retry"
    result = AgentRunner(
        task_config=_task(retries=1),
        model_config=_model(),
        logical_model="kimi",
        output_dir=destination,
        prompt="Use tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry={
            "list_case_files": lambda: {"ok": True, "files": []},
            "submit_final": lambda **_kwargs: {"ok": True, "status": "SUBMITTED"},
        },
        transport=transport,
        run_id="unit-retry",
        timestamp_fn=lambda: "2026-01-01T00:00:00Z",
        sleep_fn=lambda _seconds: None,
    ).run()
    assert result.complete
    events = _load_events(destination / "trajectory.jsonl")
    assert sum(event["event"] == "provider_retry" for event in events) == 1
    final_budget = [event["payload"] for event in events if event["event"] == "budget_state"][-1]
    assert final_budget["provider_retries"] == 1
    assert final_budget["tool_calls"] == 2
    assert final_budget["successful_tool_calls"] == 2
    assert result.metadata["counts"]["provider_retries"] == 1


def test_runner_rejects_model_identity_mismatch_without_fallback(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [_response("wrong-model-response", [], model="substituted/model")]
    )
    destination = tmp_path / "identity-mismatch"
    result = AgentRunner(
        task_config=_task(minimum_successes=0, retries=2),
        model_config=_model(),
        logical_model="kimi",
        output_dir=destination,
        prompt="Use tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry={"list_case_files": lambda: {"ok": True}},
        transport=transport,
        run_id="unit-identity-mismatch",
        timestamp_fn=lambda: "2026-01-01T00:00:00Z",
    ).run()
    assert result.status == "FAILED"
    assert len(transport.requests) == 1
    events = _load_events(destination / "trajectory.jsonl")
    error = next(event for event in events if event["event"] == "provider_error")
    assert error["payload"]["will_retry"] is False
    assert error["payload"]["error"]["status"] == "MODEL_IDENTITY_REJECTED"


def test_runner_never_overwrites_an_existing_output_directory(tmp_path: Path) -> None:
    destination = tmp_path / "already-exists"
    destination.mkdir()
    marker = destination / "owned.txt"
    marker.write_text("preserve", encoding="utf-8")
    runner = AgentRunner(
        task_config=_task(),
        model_config=_model(),
        logical_model="kimi",
        output_dir=destination,
        prompt="Use tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry={"list_case_files": lambda: {"ok": True}},
        transport=ScriptedTransport([]),
        run_id="unit-no-overwrite",
    )
    with pytest.raises(ConfigurationError):
        runner.run()
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in destination.iterdir()} == {"owned.txt"}


def test_runner_rejects_duplicate_tool_call_ids_and_still_seals_failure(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            _response(
                "req-duplicate",
                [
                    _call("same-id", "list_case_files", {}),
                    _call("same-id", "list_case_files", {}),
                ],
            )
        ]
    )
    result = AgentRunner(
        task_config=_task(minimum_successes=0),
        model_config=_model(),
        logical_model="kimi",
        output_dir=tmp_path / "duplicate-id",
        prompt="Use tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry={"list_case_files": lambda: {"ok": True, "files": []}},
        transport=transport,
        run_id="unit-duplicate",
        timestamp_fn=lambda: "2026-01-01T00:00:00Z",
    ).run()
    assert result.status == "FAILED"
    events = _load_events(tmp_path / "duplicate-id" / "trajectory.jsonl")
    validate_hash_chain(events)
    call_ids = [event["payload"]["call_id"] for event in events if event["event"] == "tool_call"]
    assert len(call_ids) == len(set(call_ids))
    assert events[-1]["event"] == "run_failure"


def test_runner_unwraps_registry_results_into_frozen_attempt_events(tmp_path: Path) -> None:
    answer = _final_answer()
    prepare_arguments = {
        "phase": "geometry_optimization",
        "source_geometry": {
            "path": "attempts/attempt_000_initial_failure/xtb.trj",
            "frame_index": 0,
        },
        "optimization_cycle_limit": 100,
        "thread_count": 1,
        "walltime_seconds": 180,
        "restart_policy": "last_complete_current_frame",
        "rationale": "Retry the frozen max-cycle failure with the configured safe limit.",
        "preserved_invariants": [
            "method",
            "charge",
            "uhf",
            "solvation",
            "atom_order",
            "composition",
            "required_phases",
        ],
    }
    scripted = ScriptedTransport(
        [
            _response("req-p", [_call("call-p", "prepare_restart", prepare_arguments)]),
            _response("req-r", [_call("call-r", "run_xtb", {"attempt_id": "attempt_001"})]),
            _response("req-v", [_call("call-v", "validate_attempt", {"attempt_id": "attempt_001"})]),
            _response("req-f", [_call("call-f", "submit_final", answer)]),
        ]
    )

    class Registry:
        def dispatch(self, name, arguments, *, call_id):
            values = {
                "prepare_restart": {
                    "attempt_id": "attempt_001",
                    "phase": "geometry_optimization",
                    "state": "PREPARED",
                    "source_geometry": {
                            "path": arguments.get("source_geometry", {}).get("path"),
                        "frame_index": 0,
                        "sha256": "1" * 64,
                        "provenance_state": "CURRENT_COMPLETE_FRAME",
                    },
                    "prepared_command_path": "attempts/attempt_001/prepared.json",
                    "input_manifest_path": "attempts/attempt_001/input_manifest.json",
                },
                "run_xtb": {
                    "attempt_id": "attempt_001",
                    "phase": "geometry_optimization",
                    "argv": ["xtb", "input.xyz", "--opt", "--gfn", "2"],
                    "shell": False,
                    "process_exit_code": 0,
                    "timed_out": False,
                    "execution_summary_path": "attempts/attempt_001/run_record.json",
                    "artifact_manifest_path": "attempts/attempt_001/artifact_manifest.json",
                    "artifact_hashes": {"attempts/attempt_001/stdout.log": "2" * 64},
                },
                "validate_attempt": {
                    "attempt_id": "attempt_001",
                    "phase": "geometry_optimization",
                    "classification": "VALIDATED_OPTIMIZATION",
                    "valid": True,
                    "optimization_converged": True,
                    "hessian_completed": False,
                    "current_artifacts": True,
                    "scientific_invariants_preserved": True,
                    "source_provenance": {"state": "CURRENT"},
                    "validation_path": "attempts/attempt_001/validation.json",
                },
                "submit_final": {"status": "SUBMITTED", "frozen": True},
            }
            return {
                "ok": True,
                "tool": name,
                "call_id": call_id,
                "result": values[name],
            }

    AgentRunner(
        task_config=_task(minimum_successes=3),
        model_config=_model(),
        logical_model="kimi",
        output_dir=tmp_path / "attempt-events",
        prompt="Use tools.",
        tool_contracts=ROOT / "configs" / "tool_contracts.json",
        tool_registry=Registry(),
        transport=scripted,
        run_id="unit-attempt-events",
        timestamp_fn=lambda: "2026-01-01T00:00:00Z",
    ).run()
    events = _load_events(tmp_path / "attempt-events" / "trajectory.jsonl")
    by_name = {event["event"]: event["payload"] for event in events}
    assert set(by_name["attempt_prepared"]) == {
        "attempt_id",
        "phase",
        "state",
        "source_geometry",
        "prepared_command_path",
        "input_manifest_path",
    }
    assert by_name["native_xtb_execution"]["source_type"] == "NATIVE_XTB"
    assert by_name["native_xtb_execution"]["shell"] is False
    assert by_name["attempt_validated"]["optimization_converged"] is True
    result_event = next(
        event for event in events
        if event["event"] == "tool_result" and event["payload"]["call_id"] == "call-p"
    )
    assert "ok" not in result_event["payload"]["result"]
    assert "tool" not in result_event["payload"]["result"]

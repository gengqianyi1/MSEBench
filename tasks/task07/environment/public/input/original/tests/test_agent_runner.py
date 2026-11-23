from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback
from typing import Any

import pytest
import yaml

from framework import (
    AgentRunner,
    ConfigurationError,
    OpenAIChatTransport,
    ProviderError,
    ScriptedTransport,
    SiliconFlowChatTransport,
)
from framework.agent_runner import (
    ARTIFACT_NAMES,
    CANONICAL_DEPENDENCY_IDENTIFIERS,
    PUBLIC_TOOL_NAMES,
)
from framework.schemas import (
    ModelConfig,
    SILICONFLOW_API_KEY_ENV,
    SILICONFLOW_MODEL_IDS,
)
from scripts.run_model import SILICONFLOW_LOGICAL_MODELS, _model_config, build_parser


FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def task_config(**budget_overrides: Any) -> dict[str, Any]:
    budget = {
        "max_tool_calls": 6,
        "max_unique_evaluated_candidates": 3,
        "max_agent_turns": 5,
        "timeout_seconds": 30,
    }
    budget.update(budget_overrides)
    return {
        "schema_version": "1.0",
        "task": {"id": "runner_test", "start_smiles": "CCO"},
        "constraints": {
            "allowed_atomic_numbers": [6, 7, 8],
            "similarity": {"radius": 2, "n_bits": 128, "min": 0.1},
            "scaffold": {"smarts": "C"},
        },
        "budget": budget,
        "output": {
            "max_submission_bytes": 10000,
            "max_smiles_length": 100,
            "max_name_length": 40,
            "max_rationale_length": 200,
        },
    }


def model_config(**overrides: Any) -> dict[str, Any]:
    value = {
        "provider": "custom",
        "transport": "openai_compatible",
        "model": "scripted-offline-model",
        "base_url": "https://invalid.example.test/v1",
        "api_key_env": "TEST_AGENT_RUNNER_API_KEY",
        "temperature": 0,
        "max_tokens": 128,
        "timeout_seconds": 5,
        "max_retries": 1,
        "retry_backoff_seconds": 0,
    }
    value.update(overrides)
    return value


def siliconflow_model_config(
    model: str = "zai-org/GLM-5.2", **overrides: Any
) -> dict[str, Any]:
    value = model_config(
        provider="siliconflow",
        model=model,
        base_url="https://api.siliconflow.cn/v1",
        api_key_env=SILICONFLOW_API_KEY_ENV,
    )
    value.update(overrides)
    return value


def tool_call(call_id: str, name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, sort_keys=True)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assistant_tool_response(*calls: dict[str, Any], response_id: str = "response-tools") -> dict[str, Any]:
    return {
        "id": response_id,
        "created": 123,
        "model": "scripted-offline-model",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "private continuation state",
                    "tool_calls": list(calls),
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def final_response(smiles: str = "CCN", *, name: str = "candidate") -> dict[str, Any]:
    content = json.dumps(
        {
            "final_candidate": {
                "name": name,
                "smiles": smiles,
                "rationale": "Selected from measured tool feedback.",
            }
        },
        sort_keys=True,
    )
    return {
        "id": "response-final",
        "created": 124,
        "model": "scripted-offline-model",
        "system_fingerprint": "fp_test_2",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }


def fake_tools(call_log: list[tuple[str, Any]] | None = None) -> dict[str, Any]:
    log = call_log if call_log is not None else []

    def validate_smiles(smiles: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        log.append(("validate_smiles", smiles))
        return {"ok": True, "valid": bool(smiles)}

    def canonicalize_smiles(smiles: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        # Accounting canonicalization is intentionally not included in the
        # public-call log; it is runner bookkeeping, not an LLM tool call.
        return {"ok": True, "canonical_smiles": smiles.strip()}

    def calculate_qed(smiles: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        log.append(("calculate_qed", smiles))
        return {"ok": True, "qed": 0.8}

    def calculate_properties(smiles: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        log.append(("calculate_properties", smiles))
        return {"ok": True, "canonical_smiles": smiles, "molecular_weight": 123.4}

    def calculate_similarity(
        smiles_a: str,
        smiles_b: str,
        config: dict[str, Any] | None = None,
        *,
        radius: int | None = None,
        n_bits: int | None = None,
    ) -> dict[str, Any]:
        log.append(("calculate_similarity", (smiles_a, smiles_b)))
        return {"ok": True, "similarity": 0.7}

    def check_scaffold(
        smiles: str,
        scaffold_smarts: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        log.append(("check_scaffold", smiles))
        return {"ok": True, "matches": True}

    def check_substructure(
        smiles: str, smarts: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        log.append(("check_substructure", smiles))
        return {"ok": True, "matches": True}

    def evaluate_candidate(smiles: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        log.append(("evaluate_candidate", smiles))
        return {
            "ok": True,
            "valid": True,
            "canonical_smiles": smiles.strip(),
            "qed": 0.81,
            "feasible": True,
            "constraint_checks": {"all": True},
        }

    registry = locals()
    return {name: registry[name] for name in PUBLIC_TOOL_NAMES}


def fake_evaluator(
    submission: Path,
    config: Any,
    output_path: Path | None = None,
    *,
    process_events: list[dict[str, Any]],
    require_process_evidence: bool,
) -> dict[str, Any]:
    assert require_process_evidence is True
    assert isinstance(process_events, list)
    value = json.loads(Path(submission).read_text(encoding="utf-8"))
    valid = set(value) == {
        "schema_version",
        "task_id",
        "final_candidate",
        "candidate_history",
        "run_metadata",
    }
    return {
        "status": "ok" if valid else "invalid_submission",
        "valid_submission": valid,
        "total_score": 50.0 if valid else 0.0,
    }


def make_runner(
    output: Path,
    responses: list[Any],
    *,
    task: dict[str, Any] | None = None,
    tools: dict[str, Any] | None = None,
    transport: Any | None = None,
    evaluator: Any = fake_evaluator,
) -> tuple[AgentRunner, Any]:
    scripted = transport or ScriptedTransport(responses)
    runner = AgentRunner(
        task_config=task or task_config(),
        model_config=model_config(),
        output_dir=output,
        prompt="Offline runner test prompt. Use tools and emit strict JSON.",
        transport=scripted,
        tool_registry=tools or fake_tools(),
        evaluator=evaluator,
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
        sleep_fn=lambda _delay: None,
    )
    return runner, scripted


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_multi_step_tool_loop_preserves_continuation_and_provider_metadata(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []
    responses = [
        RuntimeError("temporary provider failure"),
        assistant_tool_response(
            tool_call("call-props", "calculate_properties", {"smiles": "CCN"}),
            response_id="provider-id-1",
        ),
        assistant_tool_response(
            tool_call("call-eval", "evaluate_candidate", {"smiles": "CCN"}),
            response_id="provider-id-2",
        ),
        final_response("CCN"),
    ]
    runner, transport = make_runner(
        tmp_path / "run", responses, tools=fake_tools(calls)
    )

    result = runner.run()

    assert result.complete
    assert result.evaluation["valid_submission"] is True
    assert calls == [("calculate_properties", "CCN"), ("evaluate_candidate", "CCN")]
    assert result.final_answer["run_metadata"] == {
        "tool_call_count": 2,
        "budget_exceeded": False,
        "unique_evaluated_candidates": ["CCN"],
        "final_candidate_evaluated": True,
    }
    assert result.final_answer["candidate_history"] == [
        {
            "name": "candidate",
            "smiles": "CCN",
            "rationale": "Selected from measured tool feedback.",
        }
    ]
    assert result.metadata["execution_mode"] == "MOCK"
    assert result.metadata["is_real_model_run"] is False
    assert result.metadata["counts"]["provider_attempts"] == 4
    assert result.metadata["provider_response_metadata"][0]["id"] == "provider-id-1"
    assert result.metadata["provider_response_metadata"][0]["usage"]["total_tokens"] == 14
    assert result.metadata["provider_response_metadata"][0]["system_fingerprint"] == "fp_test"

    # The next provider request receives the complete assistant tool call,
    # including provider reasoning continuation state, followed by its tool result.
    second_turn_messages = transport.calls[2]["messages"]
    assistant = next(message for message in second_turn_messages if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "private continuation state"
    assert assistant["tool_calls"][0]["id"] == "call-props"
    assert second_turn_messages[-1]["role"] == "tool"
    assert second_turn_messages[-1]["tool_call_id"] == "call-props"

    events = read_jsonl(tmp_path / "run" / "trajectory.jsonl")
    retry = next(event for event in events if event["event"] == "provider_failure")
    assert retry["will_retry"] is True
    assert any(
        event["event"] == "tool_call" and event["result"]["molecular_weight"] == 123.4
        for event in events
    )


def test_tool_call_budget_is_enforced_and_every_call_gets_a_tool_message(tmp_path: Path) -> None:
    log: list[tuple[str, Any]] = []
    response = assistant_tool_response(
        tool_call("one", "calculate_qed", {"smiles": "CCO"}),
        tool_call("two", "calculate_qed", {"smiles": "CCN"}),
        tool_call("three", "validate_smiles", {"smiles": "CCC"}),
    )
    runner, transport = make_runner(
        tmp_path / "budget",
        [response, final_response("CCO")],
        task=task_config(max_tool_calls=1),
        tools=fake_tools(log),
    )

    result = runner.run()

    assert result.complete
    assert log == [("calculate_qed", "CCO")]
    assert result.final_answer["run_metadata"]["tool_call_count"] == 1
    assert result.final_answer["run_metadata"]["budget_exceeded"] is True
    assert result.metadata["counts"]["rejected_tool_calls"] == 2
    continuation = transport.calls[1]["messages"]
    tool_messages = [message for message in continuation if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["one", "two", "three"]
    assert "ToolBudgetExceeded" in tool_messages[1]["content"]


def test_unique_evaluate_candidate_budget_rejects_only_new_candidates(tmp_path: Path) -> None:
    log: list[tuple[str, Any]] = []
    response = assistant_tool_response(
        tool_call("first", "evaluate_candidate", {"smiles": "CCN"}),
        tool_call("duplicate", "evaluate_candidate", {"smiles": "CCN"}),
        tool_call("new", "evaluate_candidate", {"smiles": "CCC"}),
    )
    runner, _ = make_runner(
        tmp_path / "candidate-budget",
        [response, final_response("CCN")],
        task=task_config(max_unique_evaluated_candidates=1),
        tools=fake_tools(log),
    )

    result = runner.run()

    assert result.complete
    assert log == [("evaluate_candidate", "CCN"), ("evaluate_candidate", "CCN")]
    assert result.final_answer["run_metadata"]["unique_evaluated_candidates"] == ["CCN"]
    assert result.final_answer["run_metadata"]["budget_exceeded"] is True


def test_pre_final_guard_requires_feedback_across_provider_turns(
    tmp_path: Path,
) -> None:
    task = task_config(max_agent_turns=5, max_tool_calls=6)
    task["budget"].update(
        {
                "min_tool_calls_for_eligibility": 3,
                "min_unique_evaluated_candidates_for_eligibility": 2,
                "min_candidate_feedback_turns_for_eligibility": 2,
                "final_candidate_must_be_tool_evaluated": True,
        }
    )
    responses = [
        assistant_tool_response(
            tool_call("batch-a", "evaluate_candidate", {"smiles": "CCN"}),
            tool_call("batch-b", "evaluate_candidate", {"smiles": "CCO"}),
        ),
        final_response("CCN", name="too_early"),
        assistant_tool_response(
            tool_call("later-c", "evaluate_candidate", {"smiles": "CCC"})
        ),
        final_response("CCC", name="measured_later"),
    ]
    runner, transport = make_runner(
        tmp_path / "process-guard", responses, task=task
    )

    result = runner.run()

    assert result.complete
    assert len(transport.calls) == 4
    first_progress_guidance = transport.calls[1]["messages"][-1]
    assert first_progress_guidance["role"] == "user"
    assert "Measured process progress is incomplete" in first_progress_guidance["content"]
    assert "Do not return final JSON yet" in first_progress_guidance["content"]
    assert "evaluate_candidate" in first_progress_guidance["content"]
    assert result.metadata["counts"]["successful_public_tool_calls"] == 3
    assert result.metadata["counts"]["candidate_feedback_turns"] == 2
    assert result.final_answer["run_metadata"]["final_candidate_evaluated"] is True
    events = read_jsonl(tmp_path / "process-guard" / "trajectory.jsonl")
    rejected = next(
        event for event in events if event["event"] == "final_answer_process_guard"
    )
    assert rejected["turn"] == 2
    assert rejected["will_retry"] is True
    assert rejected["candidate_feedback_turns"] == [1]
    assert {
        issue["code"] for issue in rejected["issues"]
    } == {
        "insufficient_successful_public_tool_calls",
        "insufficient_candidate_feedback_turns",
    }
    successful_evaluations = [
        event
        for event in events
        if event["event"] == "tool_call"
        and event["tool"] == "evaluate_candidate"
        and event["status"] == "SUCCESS"
    ]
    assert [event["turn"] for event in successful_evaluations] == [1, 1, 3]
    assert any(
        message["role"] == "user"
        and "measured-process guard" in str(message.get("content"))
        for message in transport.calls[2]["messages"]
    )
    assert "native function-calling channel" in transport.calls[2]["messages"][-1]["content"]
    assert "evaluate_candidate" in transport.calls[2]["messages"][-1]["content"]


def test_full_candidate_budget_prompts_immediate_final_answer(tmp_path: Path) -> None:
    task = task_config(
        max_agent_turns=3,
        max_tool_calls=3,
        max_unique_evaluated_candidates=2,
    )
    task["budget"].update(
        {
            "min_tool_calls_for_eligibility": 2,
            "min_unique_evaluated_candidates_for_eligibility": 2,
            "min_candidate_feedback_turns_for_eligibility": 2,
            "final_candidate_must_be_tool_evaluated": True,
        }
    )
    responses = [
        assistant_tool_response(
            tool_call("candidate-a", "evaluate_candidate", {"smiles": "CCN"})
        ),
        assistant_tool_response(
            tool_call("candidate-b", "evaluate_candidate", {"smiles": "CCO"})
        ),
        final_response("CCO", name="measured_best"),
    ]
    runner, transport = make_runner(
        tmp_path / "candidate-cap-guidance", responses, task=task
    )

    result = runner.run()

    assert result.complete
    guidance = transport.calls[2]["messages"][-1]
    assert guidance["role"] == "user"
    assert "unique evaluate_candidate budget is full" in guidance["content"]
    assert "Do not call any more tools" in guidance["content"]
    assert result.final_answer["run_metadata"]["budget_exceeded"] is False


def test_pre_final_guard_stops_when_turn_budget_cannot_support_tool_then_final(
    tmp_path: Path,
) -> None:
    task = task_config(max_agent_turns=2, max_tool_calls=6)
    task["budget"].update(
        {
            "min_tool_calls_for_eligibility": 3,
            "min_unique_evaluated_candidates_for_eligibility": 2,
            "final_candidate_must_be_tool_evaluated": True,
        }
    )
    runner, transport = make_runner(
        tmp_path / "process-guard-bounded",
        [
            assistant_tool_response(
                tool_call("batch-a", "evaluate_candidate", {"smiles": "CCN"}),
                tool_call("batch-b", "evaluate_candidate", {"smiles": "CCO"}),
            ),
            final_response("CCN", name="too_early"),
        ],
        task=task,
    )

    result = runner.run()

    assert result.status == "FAILED"
    assert result.metadata["failure"]["type"] == "BudgetExceededError"
    assert len(transport.calls) == 2
    events = read_jsonl(
        tmp_path / "process-guard-bounded" / "trajectory.jsonl"
    )
    guard_events = [
        event for event in events if event["event"] == "final_answer_process_guard"
    ]
    assert len(guard_events) == 1
    assert guard_events[0]["will_retry"] is False


def test_pre_final_guard_does_not_count_structured_ok_false_as_success(
    tmp_path: Path,
) -> None:
    task = task_config(max_agent_turns=5, max_tool_calls=6)
    task["budget"].update(
        {
            "min_tool_calls_for_eligibility": 3,
            "min_unique_evaluated_candidates_for_eligibility": 2,
            "final_candidate_must_be_tool_evaluated": True,
        }
    )
    tools = fake_tools()

    def structured_failure(
        smiles: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {"ok": False, "valid": False, "smiles": smiles}

    tools["validate_smiles"] = structured_failure
    runner, transport = make_runner(
        tmp_path / "process-guard-result-ok",
        [
            assistant_tool_response(
                tool_call("eval-a", "evaluate_candidate", {"smiles": "CCN"})
            ),
            assistant_tool_response(
                tool_call("eval-b", "evaluate_candidate", {"smiles": "CCO"}),
                tool_call("normal-false", "validate_smiles", {"smiles": "bad"}),
            ),
            final_response("CCO", name="still_too_early"),
            assistant_tool_response(
                tool_call("true-third", "calculate_qed", {"smiles": "CCO"})
            ),
            final_response("CCO", name="eligible_final"),
        ],
        task=task,
        tools=tools,
    )

    result = runner.run()

    assert result.complete
    assert len(transport.calls) == 5
    assert result.metadata["counts"]["tool_calls"] == 4
    assert result.metadata["counts"]["successful_public_tool_calls"] == 3
    events = read_jsonl(
        tmp_path / "process-guard-result-ok" / "trajectory.jsonl"
    )
    structured_false = next(
        event for event in events if event.get("call_id") == "normal-false"
    )
    assert structured_false["status"] == "SUCCESS"
    assert structured_false["result_ok"] is False
    assert structured_false["counted_as_successful_public_tool_call"] is False
    rejected = next(
        event for event in events if event["event"] == "final_answer_process_guard"
    )
    assert rejected["turn"] == 3
    assert rejected["successful_public_tool_calls"] == 2
    assert [issue["code"] for issue in rejected["issues"]] == [
        "insufficient_successful_public_tool_calls"
    ]
    assert rejected["will_retry"] is True
    correction = transport.calls[3]["messages"][-1]
    assert correction["role"] == "user"
    assert "native function-calling channel" in correction["content"]
    assert "calculate_qed" in correction["content"]


def test_malformed_tool_arguments_are_returned_as_a_complete_failure_message(tmp_path: Path) -> None:
    log: list[tuple[str, Any]] = []
    malformed = assistant_tool_response(
        tool_call("bad-json", "evaluate_candidate", '{"smiles":')
    )
    runner, transport = make_runner(
        tmp_path / "malformed",
        [malformed, final_response("CCN")],
        tools=fake_tools(log),
    )

    result = runner.run()

    assert result.complete
    assert log == []
    assert result.metadata["counts"]["tool_calls"] == 1
    assert result.metadata["counts"]["tool_failures"] == 1
    failure_message = transport.calls[1]["messages"][-1]
    assert failure_message["role"] == "tool"
    assert failure_message["tool_call_id"] == "bad-json"
    assert "malformed JSON tool arguments" in failure_message["content"]
    events = read_jsonl(tmp_path / "malformed" / "trajectory.jsonl")
    failed = next(event for event in events if event["event"] == "tool_call")
    assert failed["status"] == "FAILED"
    assert failed["raw_arguments"] == '{"smiles":'


class CredentialProbeTransport:
    requires_api_key = True
    is_mock = True

    def __init__(self) -> None:
        self.called = False

    def complete(self, **_kwargs: Any) -> Any:
        self.called = True
        raise AssertionError("network-like transport must not be called without a key")


def test_missing_environment_key_fails_before_transport_and_still_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_AGENT_RUNNER_API_KEY", raising=False)
    probe = CredentialProbeTransport()
    runner, _ = make_runner(tmp_path / "missing-key", [], transport=probe)

    result = runner.run()

    assert result.status == "FAILED"
    assert result.metadata["failure"]["type"] == "MissingCredentialError"
    assert probe.called is False
    assert {path.name for path in (tmp_path / "missing-key").iterdir()} == set(ARTIFACT_NAMES)
    assert result.evaluation["status"] == "invalid_submission"


def test_artifact_set_and_contents_are_deterministic_for_scripted_runs(tmp_path: Path) -> None:
    def responses() -> list[Any]:
        return [
            assistant_tool_response(
                tool_call("eval", "evaluate_candidate", {"smiles": "CCN"})
            ),
            final_response("CCN"),
        ]

    first, _ = make_runner(tmp_path / "first", responses())
    second, _ = make_runner(tmp_path / "second", responses())
    assert first.run().complete
    assert second.run().complete

    assert {path.name for path in (tmp_path / "first").iterdir()} == set(ARTIFACT_NAMES)
    assert {path.name for path in (tmp_path / "second").iterdir()} == set(ARTIFACT_NAMES)
    for name in ARTIFACT_NAMES:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()

    metadata = json.loads((tmp_path / "first" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETE"
    assert metadata["artifacts"] == list(ARTIFACT_NAMES)
    for group in metadata["hashes"].values():
        assert group
        assert all(len(value) == 64 for value in group.values())


def test_strict_final_json_rejects_markdown_then_retries_with_full_history(tmp_path: Path) -> None:
    invalid = {
        "role": "assistant",
        "content": "```json\n{\"final_candidate\": {\"name\": \"x\", \"smiles\": \"CCN\", \"rationale\": \"x\"}}\n```",
    }
    runner, transport = make_runner(
        tmp_path / "strict", [invalid, final_response("CCN")]
    )

    result = runner.run()

    assert result.complete
    retry_messages = transport.calls[1]["messages"]
    assert [message["role"] for message in retry_messages][-2:] == ["assistant", "user"]
    assert "not the required strict JSON" in retry_messages[-1]["content"]
    assert "first non-whitespace character must be {" in retry_messages[-1]["content"]
    assert "do not use backticks" in retry_messages[-1]["content"]


def test_scripted_transport_integrates_with_real_rdkit_tools_and_evaluator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    task = root / "configs" / "task.yaml"
    prompt = root / "prompts" / "agent_prompt.md"
    start_smiles = yaml.safe_load(task.read_text(encoding="utf-8"))["task"]["start_smiles"]
    improved_smiles = "CCOC(=O)c1cc(F)c(NC(=O)C)cc1"
    transport = ScriptedTransport(
        [
            assistant_tool_response(
                tool_call("actual-eval", "evaluate_candidate", {"smiles": start_smiles}),
                tool_call("actual-props", "calculate_properties", {"smiles": start_smiles}),
            ),
            assistant_tool_response(
                tool_call("second-turn-eval", "evaluate_candidate", {"smiles": improved_smiles})
            ),
            final_response(improved_smiles, name="measured_improvement"),
        ]
    )
    runner = AgentRunner(
        task_config=task,
        model_config=model_config(max_retries=0),
        output_dir=tmp_path / "actual-integration",
        prompt=prompt,
        transport=transport,
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
    )

    result = runner.run()

    assert result.complete
    assert result.evaluation["valid_submission"] is True
    assert result.final_answer["run_metadata"]["tool_call_count"] == 3
    assert result.final_answer["run_metadata"]["final_candidate_evaluated"] is True


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("moonshot", "kimi-k2.6", "https://api.moonshot.ai/v1"),
        ("deepseek", "deepseek-v4-pro", "https://api.deepseek.com"),
        ("zhipu", "glm-5.2", "https://open.bigmodel.cn/api/paas/v4"),
        ("siliconflow", "zai-org/GLM-5.2", "https://api.siliconflow.cn/v1"),
    ],
)
def test_trusted_provider_identity_requires_exact_official_endpoint_and_model_prefix(
    provider: str, model: str, base_url: str
) -> None:
    api_key_env = (
        SILICONFLOW_API_KEY_ENV
        if provider == "siliconflow"
        else "TEST_AGENT_RUNNER_API_KEY"
    )
    trusted = ModelConfig.from_mapping(
        model_config(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
        )
    )
    assert trusted.is_trusted_endpoint is True
    assert trusted.response_model_compatible(model) is True
    assert trusted.response_model_compatible(model + "-20260718") is (
        provider != "siliconflow"
    )

    alternate = ModelConfig.from_mapping(
        model_config(
            provider=provider,
            model=model,
            base_url=base_url + "/",
            api_key_env=api_key_env,
        )
    )
    assert alternate.is_trusted_endpoint is False

    wrong_prefix = {
        "moonshot": "deepseek-v4-pro",
        "deepseek": "glm-5.2",
        "zhipu": "kimi-k2.6",
        "siliconflow": "kimi-k2.6",
    }[provider]
    with pytest.raises(ConfigurationError, match="incompatible with provider"):
        ModelConfig.from_mapping(
            model_config(
                provider=provider,
                model=wrong_prefix,
                base_url=base_url,
                api_key_env=api_key_env,
            )
        )


@pytest.mark.parametrize("exact_model", sorted(SILICONFLOW_MODEL_IDS))
def test_siliconflow_response_model_requires_exact_configured_id(
    exact_model: str,
) -> None:
    config = ModelConfig.from_mapping(siliconflow_model_config(model=exact_model))
    alternate = next(model for model in SILICONFLOW_MODEL_IDS if model != exact_model)
    assert config.response_model_compatible(exact_model) is True
    assert config.response_model_compatible(exact_model + "-20260718") is False
    assert config.response_model_compatible(alternate) is False


def test_provider_and_transport_must_be_explicit_and_custom_endpoints_are_unverified() -> None:
    for missing in ("provider", "transport"):
        raw = model_config()
        raw.pop(missing)
        with pytest.raises(ConfigurationError, match=missing):
            ModelConfig.from_mapping(raw)

    custom = ModelConfig.from_mapping(model_config())
    assert custom.provider == "custom"
    assert custom.transport == "openai_compatible"
    assert custom.is_trusted_endpoint is False

    unverified_moonshot_domain = ModelConfig.from_mapping(
        model_config(
            provider="moonshot",
            model="kimi-k2.6",
            base_url="https://api.moonshot.cn/v1",
        )
    )
    assert unverified_moonshot_domain.is_trusted_endpoint is False


@pytest.mark.parametrize(
    "reserved",
    [
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "max_tokens",
        "timeout",
        "stream",
        "extra_body",
    ],
)
def test_provider_options_cannot_override_reserved_request_fields(reserved: str) -> None:
    with pytest.raises(ConfigurationError, match="cannot override"):
        ModelConfig.from_mapping(
            model_config(provider_options={reserved: "attacker-controlled"})
        )


def test_unknown_top_level_or_provider_options_cannot_be_request_passthroughs() -> None:
    raw = model_config()
    raw["extra_request_parameters"] = {"messages": [{"role": "user"}]}
    with pytest.raises(ConfigurationError, match="unknown model configuration"):
        ModelConfig.from_mapping(raw)

    with pytest.raises(ConfigurationError, match="unsupported field"):
        ModelConfig.from_mapping(model_config(provider_options={"n": 99}))


def test_transport_copies_only_validated_safe_options_after_fixed_request_fields() -> None:
    config = ModelConfig.from_mapping(
        model_config(
            provider_options={
                "seed": 17,
                "top_p": 0.8,
                "thinking": {"type": "enabled"},
            }
        )
    )

    class CompletionsProbe:
        def __init__(self) -> None:
            self.request: dict[str, Any] | None = None

        def create(self, **kwargs: Any) -> dict[str, Any]:
            self.request = kwargs
            return {"ok": True}

    class ClientProbe:
        def __init__(self) -> None:
            self.chat = type("ChatProbe", (), {})()
            self.chat.completions = CompletionsProbe()

    transport = object.__new__(OpenAIChatTransport)
    transport._config = config
    transport._client = ClientProbe()
    transport.complete(messages=[{"role": "user", "content": "x"}], tools=[], timeout_seconds=3)
    request = transport._client.chat.completions.request
    assert request is not None
    assert request["model"] == "scripted-offline-model"
    assert request["tool_choice"] == "auto"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 128
    assert request["timeout"] == 3
    assert request["stream"] is False
    assert request["seed"] == 17
    assert request["top_p"] == 0.8
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}


def test_siliconflow_example_uses_exact_logical_models_and_no_override() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "models.example.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert set(document["models"]) == set(SILICONFLOW_LOGICAL_MODELS)
    assert set(SILICONFLOW_LOGICAL_MODELS.values()) == set(SILICONFLOW_MODEL_IDS)

    for logical_name, exact_model in SILICONFLOW_LOGICAL_MODELS.items():
        entry = document["models"][logical_name]
        assert entry["provider"] == "siliconflow"
        assert entry["transport"] == "openai_compatible"
        assert entry["model"] == exact_model
        assert entry["base_url"] == "https://api.siliconflow.cn/v1"
        assert entry["api_key_env"] == SILICONFLOW_API_KEY_ENV
        assert "model_env" not in entry
        selected = _model_config(config_path, logical_name)
        assert ModelConfig.from_mapping(selected).is_trusted_endpoint is True


def test_run_model_normalizes_provider_before_siliconflow_logical_mapping(
    tmp_path: Path,
) -> None:
    entry = siliconflow_model_config(model=SILICONFLOW_LOGICAL_MODELS["glm"])
    entry["provider"] = " SiliconFlow "
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump({"models": {"glm": entry}}, sort_keys=True),
        encoding="utf-8",
    )
    selected = _model_config(config_path, "glm")
    assert selected["provider"] == "siliconflow"
    assert selected["model"] == SILICONFLOW_LOGICAL_MODELS["glm"]

    entry["model"] = SILICONFLOW_LOGICAL_MODELS["kimi"]
    config_path.write_text(
        yaml.safe_dump({"models": {"glm": entry}}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="must be exactly"):
        _model_config(config_path, "glm")


def test_siliconflow_schema_rejects_substitution_endpoint_and_credential_env() -> None:
    with pytest.raises(ConfigurationError, match="exact configured model IDs"):
        ModelConfig.from_mapping(
            siliconflow_model_config(model="THUDM/a-substitute-model")
        )
    with pytest.raises(ConfigurationError, match="api_key_env must be exactly"):
        ModelConfig.from_mapping(
            siliconflow_model_config(api_key_env="SOME_OTHER_KEY")
        )

    alternate = ModelConfig.from_mapping(
        siliconflow_model_config(base_url="https://api.siliconflow.cn/v1/")
    )
    assert alternate.is_trusted_endpoint is False


def test_siliconflow_adapter_builds_non_streaming_structured_tool_request() -> None:
    secret = "adapter-" + "credential-probe"
    captured_init: dict[str, Any] = {}
    captured_request: dict[str, Any] = {}
    response = {
        "id": "sf-response-1",
        "model": "zai-org/GLM-5.2",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("sf-call-1", "calculate_qed", {"smiles": "CCO"})
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    class CompletionsProbe:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            captured_request.update(kwargs)
            return response

    class ClientProbe:
        def __init__(self) -> None:
            self.chat = type("ChatProbe", (), {})()
            self.chat.completions = CompletionsProbe()

    def client_factory(**kwargs: Any) -> ClientProbe:
        captured_init.update(kwargs)
        return ClientProbe()

    config = ModelConfig.from_mapping(
        siliconflow_model_config(
            max_retries=2,
            provider_options={"thinking": {"type": "enabled"}},
        )
    )
    transport = SiliconFlowChatTransport(
        config, secret, client_factory=client_factory
    )
    tools = [{"type": "function", "function": {"name": "calculate_qed"}}]
    returned = transport.complete(
        messages=[{"role": "user", "content": "measure"}],
        tools=tools,
        timeout_seconds=3,
    )

    assert returned is response
    assert captured_init == {
        "api_key": secret,
        "base_url": "https://api.siliconflow.cn/v1",
        "timeout": 5.0,
        "max_retries": 0,
    }
    assert captured_request["model"] == "zai-org/GLM-5.2"
    assert captured_request["tools"] == tools
    assert captured_request["tool_choice"] == "auto"
    assert captured_request["stream"] is False
    assert captured_request["timeout"] == 3
    assert captured_request["temperature"] == 0
    assert captured_request["max_tokens"] == 128
    assert captured_request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert secret not in json.dumps(captured_request, sort_keys=True)


def test_siliconflow_adapter_error_does_not_reflect_credential() -> None:
    secret = "exception-" + "credential-probe"

    class ExplodingCompletions:
        def create(self, **_kwargs: Any) -> None:
            raise RuntimeError("request failed with " + secret)

    class ClientProbe:
        def __init__(self) -> None:
            self.chat = type("ChatProbe", (), {})()
            self.chat.completions = ExplodingCompletions()

    config = ModelConfig.from_mapping(siliconflow_model_config())
    transport = SiliconFlowChatTransport(
        config, secret, client_factory=lambda **_kwargs: ClientProbe()
    )
    with pytest.raises(ProviderError) as captured:
        transport.complete(messages=[], tools=[], timeout_seconds=1)
    rendered = "".join(
        traceback.format_exception(
            type(captured.value), captured.value, captured.value.__traceback__
        )
    )
    assert secret not in rendered
    assert "RuntimeError" in str(captured.value)


def test_siliconflow_adapter_uses_runner_retry_without_leaking_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "retry-" + "credential-probe"
    actual_model = SILICONFLOW_LOGICAL_MODELS["glm"]

    class RetryingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure for " + secret)
            response = final_response("CCN", name="retried_candidate")
            response["model"] = actual_model
            return response

    class ClientProbe:
        def __init__(self) -> None:
            self.chat = type("ChatProbe", (), {})()
            self.chat.completions = RetryingCompletions()

    client = ClientProbe()
    config_mapping = siliconflow_model_config(model=actual_model, max_retries=1)
    config = ModelConfig.from_mapping(config_mapping)
    transport = SiliconFlowChatTransport(
        config, secret, client_factory=lambda **_kwargs: client
    )
    monkeypatch.setenv(SILICONFLOW_API_KEY_ENV, secret)
    runner = AgentRunner(
        task_config=task_config(),
        model_config=config_mapping,
        output_dir=tmp_path / "siliconflow-retry",
        prompt="Offline SiliconFlow retry test.",
        transport=transport,
        tool_registry=fake_tools(),
        evaluator=fake_evaluator,
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
        sleep_fn=lambda _delay: None,
    )
    result = runner.run()

    assert result.complete
    assert client.chat.completions.calls == 2
    assert result.metadata["counts"]["provider_attempts"] == 2
    events = read_jsonl(tmp_path / "siliconflow-retry" / "trajectory.jsonl")
    assert any(
        event["event"] == "provider_failure" and event["will_retry"] is True
        for event in events
    )
    for artifact_name in ARTIFACT_NAMES:
        assert secret not in (
            tmp_path / "siliconflow-retry" / artifact_name
        ).read_text(encoding="utf-8")


def test_provider_model_mismatch_fails_run_and_records_returned_response(tmp_path: Path) -> None:
    mismatched = final_response("CCN")
    mismatched["model"] = "different-model"
    runner, _ = make_runner(tmp_path / "model-mismatch", [mismatched])

    result = runner.run()

    assert result.status == "FAILED"
    assert result.metadata["failure"]["type"] == "ProviderError"
    assert "incompatible model ID" in result.metadata["failure"]["message"]
    assert result.metadata["actual_model_id"] == "different-model"
    assert result.metadata["model"] == "different-model"
    assert result.metadata["requested_model"] == "scripted-offline-model"
    assert result.metadata["provider_response_metadata"][0]["model"] == "different-model"
    assert result.metadata["is_real_model_run"] is False
    events = read_jsonl(tmp_path / "model-mismatch" / "trajectory.jsonl")
    response_event = next(event for event in events if event["event"] == "provider_response")
    assert response_event["response"]["model"] == "different-model"


@pytest.mark.parametrize(
    "evaluation",
    [
        {"status": "invalid_submission", "valid_submission": False, "total_score": 0.0},
        {"status": "evaluation_error", "valid_submission": False, "total_score": 0.0},
        {"valid_submission": True, "total_score": 1.0},
        {"status": 7, "valid_submission": True, "total_score": 1.0},
    ],
)
def test_evaluator_error_invalid_or_unparseable_status_cannot_complete(
    tmp_path: Path, evaluation: dict[str, Any]
) -> None:
    def evaluator(
        _submission: Path,
        _config: Any,
        output_path: Path | None = None,
        *,
        process_events: list[dict[str, Any]],
        require_process_evidence: bool,
    ) -> dict[str, Any]:
        assert process_events
        assert require_process_evidence is True
        return dict(evaluation)

    runner, _ = make_runner(
        tmp_path / f"evaluation-{len(str(evaluation))}",
        [final_response("CCN")],
        evaluator=evaluator,
    )
    result = runner.run()
    assert result.status == "FAILED"
    assert result.metadata["is_real_model_run"] is False


def test_low_quality_status_ok_evaluation_can_complete(tmp_path: Path) -> None:
    def evaluator(
        _submission: Path,
        _config: Any,
        output_path: Path | None = None,
        *,
        process_events: list[dict[str, Any]],
        require_process_evidence: bool,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "valid_submission": True,
            "eligible": False,
            "total_score": 0.0,
        }

    runner, _ = make_runner(
        tmp_path / "low-quality", [final_response("CCN")], evaluator=evaluator
    )
    result = runner.run()
    assert result.complete
    assert result.evaluation["eligible"] is False
    assert result.evaluation["total_score"] == 0.0


def test_invalid_candidate_attempt_is_retained_but_excluded_from_valid_history(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, Any]] = []
    tools = fake_tools(log)

    def evaluate_candidate(smiles: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        log.append(("evaluate_candidate", smiles))
        if smiles == "not-a-smiles":
            return {
                "ok": False,
                "valid": False,
                "canonical_smiles": None,
                "error": "SMILES parse failed",
            }
        return {
            "ok": True,
            "valid": True,
            "canonical_smiles": smiles,
            "qed": 0.8,
            "feasible": True,
        }

    tools["evaluate_candidate"] = evaluate_candidate
    response = assistant_tool_response(
        tool_call("invalid", "evaluate_candidate", {"smiles": "not-a-smiles"}),
        tool_call("valid", "evaluate_candidate", {"smiles": "CCN"}),
    )
    runner, _ = make_runner(
        tmp_path / "invalid-then-valid",
        [response, final_response("CCN")],
        task=task_config(max_unique_evaluated_candidates=2),
        tools=tools,
    )
    result = runner.run()

    assert result.complete
    assert log == [
        ("evaluate_candidate", "not-a-smiles"),
        ("evaluate_candidate", "CCN"),
    ]
    assert result.metadata["counts"]["unique_attempted_candidates"] == 2
    assert result.metadata["counts"]["unique_evaluated_candidates"] == 1
    assert result.final_answer["run_metadata"]["unique_evaluated_candidates"] == ["CCN"]
    assert [item["smiles"] for item in result.final_answer["candidate_history"]] == ["CCN"]
    events = read_jsonl(tmp_path / "invalid-then-valid" / "trajectory.jsonl")
    invalid = next(event for event in events if event.get("call_id") == "invalid")
    assert invalid["status"] == "SUCCESS"
    assert invalid["result"] == {
        "ok": False,
        "valid": False,
        "canonical_smiles": None,
        "error": "SMILES parse failed",
    }
    assert invalid["recorded_valid_candidate"] is None


def test_distinct_invalid_attempts_consume_candidate_budget_but_duplicates_do_not(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, Any]] = []
    tools = fake_tools(log)

    def invalid_candidate(smiles: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        log.append(("evaluate_candidate", smiles))
        return {"ok": False, "valid": False, "canonical_smiles": None}

    tools["evaluate_candidate"] = invalid_candidate
    response = assistant_tool_response(
        tool_call("bad-1", "evaluate_candidate", {"smiles": "bad"}),
        tool_call("bad-duplicate", "evaluate_candidate", {"smiles": "bad"}),
        tool_call("new-after-budget", "evaluate_candidate", {"smiles": "CCN"}),
    )
    runner, _ = make_runner(
        tmp_path / "invalid-budget",
        [response, final_response("CCN")],
        task=task_config(max_unique_evaluated_candidates=1),
        tools=tools,
    )
    result = runner.run()

    assert result.complete
    assert log == [("evaluate_candidate", "bad"), ("evaluate_candidate", "bad")]
    assert result.metadata["counts"]["unique_attempted_candidates"] == 1
    assert result.metadata["counts"]["unique_evaluated_candidates"] == 0
    assert result.metadata["counts"]["rejected_tool_calls"] == 1
    assert result.final_answer["candidate_history"] == []


def test_metadata_aggregates_tokens_and_only_interface_reported_cost(tmp_path: Path) -> None:
    first = assistant_tool_response(
        tool_call("eval", "evaluate_candidate", {"smiles": "CCN"})
    )
    first["usage"]["cost"] = 0.01
    second = final_response("CCN")
    second["total_cost"] = 0.02
    runner, _ = make_runner(tmp_path / "usage-cost", [first, second])
    result = runner.run()

    assert result.complete
    assert result.metadata["token_usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
    }
    assert result.metadata["cost"] == pytest.approx(0.03)

    no_cost_runner, _ = make_runner(
        tmp_path / "usage-no-cost", [final_response("CCN")]
    )
    no_cost = no_cost_runner.run()
    assert "cost" not in no_cost.metadata


def test_injected_tools_and_evaluator_make_trusted_endpoint_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OfflineRealTransport:
        def __init__(self, config: ModelConfig, api_key: str):
            assert api_key == "test-key"
            self.config = config

        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            response = final_response("CCN")
            response["model"] = self.config.model
            response["usage"]["cost"] = 0.125
            return response

    monkeypatch.setattr("framework.agent_runner.OpenAIChatTransport", OfflineRealTransport)
    monkeypatch.setenv("TRUSTED_TEST_KEY", "test-key")
    runner = AgentRunner(
        task_config=task_config(),
        model_config=model_config(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            api_key_env="TRUSTED_TEST_KEY",
            max_retries=0,
        ),
        output_dir=tmp_path / "trusted-real",
        prompt="prompt",
        tool_registry=fake_tools(),
        evaluator=fake_evaluator,
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
    )
    result = runner.run()
    assert result.complete
    assert result.metadata["execution_mode"] == "UNVERIFIED"
    assert result.metadata["dependency_mode"] == "UNVERIFIED"
    assert result.metadata["canonical_dependencies"] is False
    assert result.metadata["dependency_provenance"]["transport"]["mode"] == "default"
    assert result.metadata["dependency_provenance"]["tool_registry"]["mode"] == "injected"
    assert result.metadata["dependency_provenance"]["evaluator"]["mode"] == "injected"
    assert result.metadata["is_real_model_run"] is False
    assert result.metadata["provider"] == "deepseek"
    assert result.metadata["base_url"] == "https://api.deepseek.com"
    assert result.metadata["endpoint"] == "https://api.deepseek.com"
    assert result.metadata["request_endpoint"] == (
        "https://api.deepseek.com/chat/completions"
    )
    assert result.metadata["actual_model_id"] == "deepseek-v4-pro"
    assert result.metadata["cost"] == pytest.approx(0.125)


def test_custom_endpoint_is_never_a_real_model_run_even_with_provider_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OfflineUnverifiedTransport:
        def __init__(self, config: ModelConfig, api_key: str):
            self.config = config

        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            response = final_response("CCN")
            response["model"] = self.config.model
            return response

    monkeypatch.setattr(
        "framework.agent_runner.OpenAIChatTransport", OfflineUnverifiedTransport
    )
    monkeypatch.setenv("TEST_AGENT_RUNNER_API_KEY", "test-key")
    runner = AgentRunner(
        task_config=task_config(),
        model_config=model_config(),
        output_dir=tmp_path / "custom-unverified",
        prompt="prompt",
        tool_registry=fake_tools(),
        evaluator=fake_evaluator,
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
    )
    result = runner.run()
    assert result.complete
    assert result.metadata["execution_mode"] == "UNVERIFIED"
    assert result.metadata["is_real_model_run"] is False


def test_existing_artifacts_are_never_overwritten_even_with_legacy_flag(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    artifact = output / "metadata.json"
    artifact.write_text("do-not-replace", encoding="utf-8")
    runner = AgentRunner(
        task_config=task_config(),
        model_config=model_config(),
        output_dir=output,
        prompt="prompt",
        transport=ScriptedTransport([final_response("CCN")]),
        tool_registry=fake_tools(),
        evaluator=fake_evaluator,
        allow_overwrite=True,
    )
    with pytest.raises(ConfigurationError, match="overwrite is disabled"):
        runner.run()
    assert artifact.read_text(encoding="utf-8") == "do-not-replace"
    assert list(output.iterdir()) == [artifact]


def test_run_model_cli_has_no_overwrite_escape_hatch() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "kimi", "--overwrite"])


def test_source_hashes_cover_scoring_and_configuration_files(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path / "source-hashes", [final_response("CCN")])
    result = runner.run()
    sources = result.metadata["hashes"]["sources"]
    assert {
        "evaluator/scoring.py",
        "framework/siliconflow_adapter.py",
        "configs/task.yaml",
        "configs/models.example.yaml",
    }.issubset(sources)
    assert all(len(sources[name]) == 64 for name in sources)


def test_siliconflow_default_transport_records_required_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "runner-" + "credential-probe"
    actual_model = SILICONFLOW_LOGICAL_MODELS["glm"]

    class OfflineSiliconFlowTransport:
        requires_api_key = True
        is_mock = False

        def __init__(self, config: ModelConfig, api_key: str):
            assert config.provider == "siliconflow"
            assert api_key == secret
            self._config = config

        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            response = final_response("CCN", name="siliconflow_candidate")
            response["model"] = actual_model
            return response

    monkeypatch.setattr(
        "framework.siliconflow_adapter.SiliconFlowChatTransport",
        OfflineSiliconFlowTransport,
    )
    monkeypatch.setattr("evaluator.evaluate.evaluate_submission", fake_evaluator)
    monkeypatch.setenv(SILICONFLOW_API_KEY_ENV, secret)
    runner = AgentRunner(
        task_config=task_config(),
        model_config=siliconflow_model_config(model=actual_model, max_retries=0),
        output_dir=tmp_path / "siliconflow-metadata",
        prompt="Offline SiliconFlow metadata test.",
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
    )
    result = runner.run()

    assert result.complete
    metadata = result.metadata
    assert metadata["provider"] == "siliconflow"
    assert metadata["model"] == actual_model
    assert metadata["actual_model_id"] == actual_model
    assert metadata["requested_model"] == actual_model
    assert metadata["model_config"]["model"] == actual_model
    assert metadata["endpoint"] == "https://api.siliconflow.cn/v1"
    assert metadata["request_endpoint"] == (
        "https://api.siliconflow.cn/v1/chat/completions"
    )
    assert metadata["timestamp"] == "2026-01-02T03:04:05Z"
    assert metadata["temperature"] == 0.0
    assert metadata["max_tokens"] == 128
    assert metadata["dependency_provenance"]["transport"]["identifier"] == (
        runner._object_identifier(OfflineSiliconFlowTransport)
    )
    assert metadata["dependency_provenance"]["tool_registry"]["identifier"] == (
        "tools.rdkit_tools.TOOL_REGISTRY"
    )
    assert metadata["execution_mode"] == "UNVERIFIED"
    assert metadata["canonical_dependencies"] is False
    assert metadata["dependency_provenance"]["transport"]["canonical"] is False
    assert metadata["is_real_model_run"] is False
    assert secret not in json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def test_monkeypatched_default_transport_cannot_self_attest_as_real_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    start_smiles = yaml.safe_load(
        (root / "configs" / "task.yaml").read_text(encoding="utf-8")
    )["task"]["start_smiles"]
    improved_smiles = "CCOC(=O)c1cc(F)c(NC(=O)C)cc1"

    class OfflineDefaultTransport:
        requires_api_key = True
        is_mock = False

        def __init__(self, config: ModelConfig, api_key: str):
            assert api_key == "test-key"
            self._config = config
            self.calls = 0

        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                response = assistant_tool_response(
                    tool_call("default-eval-1", "evaluate_candidate", {"smiles": start_smiles}),
                    tool_call("default-props", "calculate_properties", {"smiles": start_smiles}),
                )
            elif self.calls == 2:
                response = assistant_tool_response(
                    tool_call("default-eval-2", "evaluate_candidate", {"smiles": improved_smiles})
                )
            else:
                response = final_response(improved_smiles, name="measured_improvement")
            response["model"] = self._config.model
            return response

    monkeypatch.setattr("framework.agent_runner.OpenAIChatTransport", OfflineDefaultTransport)
    monkeypatch.setenv("CANONICAL_TEST_KEY", "test-key")
    runner = AgentRunner(
        task_config=root / "configs" / "task.yaml",
        model_config=model_config(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            api_key_env="CANONICAL_TEST_KEY",
            max_retries=0,
        ),
        output_dir=tmp_path / "canonical-real",
        prompt=root / "prompts" / "agent_prompt.md",
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
    )
    result = runner.run()

    assert result.complete
    assert result.evaluation["status"] == "ok"
    assert result.metadata["execution_mode"] == "UNVERIFIED"
    assert result.metadata["dependency_mode"] == "UNVERIFIED"
    assert result.metadata["canonical_dependencies"] is False
    provenance = result.metadata["dependency_provenance"]
    assert provenance["transport"]["mode"] == "default"
    assert provenance["transport"]["canonical"] is False
    assert provenance["transport"]["attestation"] == "instance_unverified"
    assert provenance["tool_registry"]["canonical"] is True
    assert provenance["evaluator"]["canonical"] is True
    assert result.metadata["integrity"]["verified"] is True
    assert result.metadata["is_real_model_run"] is False
    events = read_jsonl(tmp_path / "canonical-real" / "trajectory.jsonl")
    assert events[0]["event"] == "run_start"
    assert events[0]["sequence"] == 1
    assert events[-1]["event"] == "run_complete"
    assert events[-1]["status"] == "COMPLETE"
    assert [event["event"] for event in events].count("run_start") == 1
    assert [event["event"] for event in events].count("run_complete") == 1
    assert not any(event["event"] == "run_failure" for event in events)


def test_fake_dependencies_cannot_self_attest_as_canonical(tmp_path: Path) -> None:
    runner = AgentRunner(
        task_config=task_config(),
        model_config=model_config(),
        output_dir=tmp_path / "false-attestation",
        prompt="prompt",
        transport=ScriptedTransport([final_response("CCN")]),
        tool_registry=fake_tools(),
        evaluator=fake_evaluator,
        canonical_dependency_provenance=CANONICAL_DEPENDENCY_IDENTIFIERS,
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
    )
    result = runner.run()
    assert result.complete
    assert result.metadata["dependency_mode"] == "UNVERIFIED"
    assert result.metadata["canonical_dependencies"] is False
    assert all(
        entry["mode"] == "injected" and entry["attestation"] == "unverified"
        for entry in result.metadata["dependency_provenance"].values()
    )
    assert result.metadata["is_real_model_run"] is False


def test_output_directory_claim_is_atomic_under_concurrent_runners(tmp_path: Path) -> None:
    output = tmp_path / "one-claim-only"

    def attempt() -> str:
        runner, _ = make_runner(output, [final_response("CCN")])
        try:
            return runner.run().status
        except ConfigurationError as exc:
            assert "already exists" in str(exc)
            return "CLAIM_REJECTED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: attempt(), range(2)))

    assert sorted(outcomes) == ["CLAIM_REJECTED", "COMPLETE"]
    assert {path.name for path in output.iterdir()} == set(ARTIFACT_NAMES)


@pytest.mark.parametrize("bad_id", [None, "", "   ", 17, False, ["task"]])
def test_malformed_task_id_still_produces_five_safe_failure_artifacts(
    tmp_path: Path, bad_id: Any
) -> None:
    task = task_config()
    if bad_id is None:
        task["task"].pop("id")
    else:
        task["task"]["id"] = bad_id
    output = tmp_path / f"bad-task-id-{repr(bad_id)}"
    runner, _ = make_runner(output, [final_response("CCN")], task=task)

    result = runner.run()

    assert result.status == "FAILED"
    assert result.metadata["task_id"] is None
    assert result.metadata["failure"]["type"] == "ConfigurationError"
    assert {path.name for path in output.iterdir()} == set(ARTIFACT_NAMES)
    assert result.final_answer["task_id"] is None
    for artifact_name in ("metadata.json", "final_answer.json", "evaluation.json"):
        json.loads(
            (output / artifact_name).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    events = read_jsonl(output / "trajectory.jsonl")
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_failure"
    assert events[-1]["status"] == "FAILED"
    assert not any(event["event"] == "run_complete" for event in events)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0.2"),
        ("temperature", True),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", -0.01),
        ("temperature", 2.01),
        ("max_tokens", "128"),
        ("max_tokens", True),
        ("max_tokens", 128.0),
        ("max_tokens", 0),
        ("max_tokens", 1_000_001),
        ("timeout_seconds", "5"),
        ("timeout_seconds", False),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", 0),
        ("timeout_seconds", 86_401),
        ("max_retries", 1.0),
        ("max_retries", True),
        ("max_retries", -1),
        ("max_retries", 11),
        ("retry_backoff_seconds", "1"),
        ("retry_backoff_seconds", True),
        ("retry_backoff_seconds", float("-inf")),
        ("retry_backoff_seconds", -1),
        ("retry_backoff_seconds", 3_601),
        ("enabled", 1),
    ],
)
def test_model_config_rejects_inexact_nonfinite_or_out_of_range_values(
    field: str, value: Any
) -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig.from_mapping(model_config(**{field: value}))


@pytest.mark.parametrize(
    "options",
    [
        {"top_p": float("nan")},
        {"top_p": 1.01},
        {"frequency_penalty": float("inf")},
        {"frequency_penalty": -2.01},
        {"presence_penalty": 2.01},
        {"seed": True},
        {"seed": 2**63},
        {"parallel_tool_calls": 1},
    ],
)
def test_provider_options_reject_nonfinite_or_inexact_values(
    options: dict[str, Any]
) -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig.from_mapping(model_config(provider_options=options))


def test_nonfinite_provider_values_never_reach_json_artifacts(tmp_path: Path) -> None:
    nonfinite = final_response("CCN")
    nonfinite["usage"]["total_tokens"] = float("nan")
    runner, _ = make_runner(
        tmp_path / "nonfinite-provider",
        [nonfinite, final_response("CCN")],
    )
    result = runner.run()

    assert result.complete
    assert result.metadata["counts"]["provider_attempts"] == 2
    events = read_jsonl(tmp_path / "nonfinite-provider" / "trajectory.jsonl")
    assert any(
        event["event"] == "provider_failure"
        and "non-finite" in event["error"]["message"]
        for event in events
    )
    for artifact_name in ARTIFACT_NAMES:
        text = (tmp_path / "nonfinite-provider" / artifact_name).read_text(
            encoding="utf-8"
        )
        assert "NaN" not in text
        assert "Infinity" not in text
    for artifact_name in (
        "metadata.json",
        "final_answer.json",
        "evaluation.json",
        "trajectory.jsonl",
    ):
        for line in (tmp_path / "nonfinite-provider" / artifact_name).read_text(
            encoding="utf-8"
        ).splitlines() or ["{}"]:
            if artifact_name == "trajectory.jsonl":
                json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                )


def test_integrity_change_before_completion_fails_and_is_attested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_smiles = "CCCCN(CC)CCOC(=O)c1ccc(NC(C)=O)cc1"
    holder: dict[str, AgentRunner] = {}

    class MutatingDefaultTransport:
        def __init__(self, config: ModelConfig, api_key: str):
            self.config = config

        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            holder["runner"].prompt += "\nmutated-during-run"
            response = final_response(start_smiles, name="mutated")
            response["model"] = self.config.model
            return response

    monkeypatch.setattr(
        "framework.agent_runner.OpenAIChatTransport", MutatingDefaultTransport
    )
    monkeypatch.setenv("INTEGRITY_TEST_KEY", "test-key")
    root = Path(__file__).resolve().parents[1]
    runner = AgentRunner(
        task_config=root / "configs" / "task.yaml",
        model_config=model_config(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            api_key_env="INTEGRITY_TEST_KEY",
            max_retries=0,
        ),
        output_dir=tmp_path / "integrity-change",
        prompt=root / "prompts" / "agent_prompt.md",
        now_fn=lambda: FIXED_TIME,
        monotonic_fn=lambda: 100.0,
    )
    holder["runner"] = runner
    result = runner.run()

    assert result.status == "FAILED"
    assert result.metadata["execution_mode"] == "UNVERIFIED"
    assert result.metadata["canonical_dependencies"] is False
    assert result.metadata["integrity"]["verified"] is False
    assert "configs.agent_prompt" in result.metadata["integrity"]["changed"]
    assert result.metadata["is_real_model_run"] is False
    events = read_jsonl(tmp_path / "integrity-change" / "trajectory.jsonl")
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_failure"
    assert not any(event["event"] == "run_complete" for event in events)


def test_failed_tentative_evaluation_is_replayed_with_final_failure_lifecycle(
    tmp_path: Path,
) -> None:
    observed: list[list[dict[str, Any]]] = []

    def rejecting_evaluator(
        _submission: Path,
        _config: Any,
        output_path: Path | None = None,
        *,
        process_events: list[dict[str, Any]],
        require_process_evidence: bool,
    ) -> dict[str, Any]:
        observed.append(process_events)
        return {
            "status": "evaluation_error",
            "valid_submission": False,
            "total_score": 0.0,
        }

    runner, _ = make_runner(
        tmp_path / "tentative-terminal",
        [final_response("CCN")],
        evaluator=rejecting_evaluator,
    )
    result = runner.run()

    assert result.status == "FAILED"
    assert len(observed) == 2
    assert observed[0][-1]["event"] == "run_complete"
    assert observed[1][-1]["event"] == "run_failure"
    disk_events = read_jsonl(tmp_path / "tentative-terminal" / "trajectory.jsonl")
    assert disk_events == observed[1]
    assert disk_events[-1]["event"] == "run_failure"


def test_machine_local_exception_paths_are_redacted_from_all_artifacts(
    tmp_path: Path,
) -> None:
    # Assemble machine-like paths from fragments so the package's runtime
    # absolute-path scanner can still reject accidental literals in shipped
    # test source while this regression exercises redaction at runtime.
    windows_python_root = "E:" + r"\Python312"
    windows_user_path = "C:" + r"\Users\alice\.cache\token.txt"
    unc_path = "\\" + r"\build-host\private\credential.txt"
    posix_home_path = "/" + "home/alice/.cache/token"
    posix_library_path = "/" + "usr/local/lib/python3.12/site-packages/vendor_sdk/client.py"
    external_filename = windows_python_root + r"\Lib\site-packages\vendor_sdk\client.py"
    leaked_message = (
        f"SDK cache {windows_user_path}; "
        f"UNC cache {unc_path}; "
        f"POSIX cache {posix_home_path}; "
        f"library {posix_library_path}; "
        "docs https://docs.example.test/help; report reports/test_report.md"
    )
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "def fail():\n    raise RuntimeError(" + repr(leaked_message) + ")\n",
            external_filename,
            "exec",
        ),
        namespace,
    )

    class PathLeakingTransport:
        requires_api_key = False
        is_mock = True

        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            namespace["fail"]()
            raise AssertionError("unreachable")

    task = task_config()
    task["task"]["start_smiles"] = "C/C=C/C"
    output = tmp_path / "redacted-paths"
    runner, _ = make_runner(
        output,
        [],
        task=task,
        transport=PathLeakingTransport(),
    )

    result = runner.run()

    assert result.status == "FAILED"
    assert {path.name for path in output.iterdir()} == set(ARTIFACT_NAMES)
    artifact_text = {
        name: (output / name).read_text(encoding="utf-8") for name in ARTIFACT_NAMES
    }
    windows_absolute = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
    posix_local = re.compile(
        r"(?<![A-Za-z0-9:/>])/(?:home|usr|opt|var|tmp|etc|private)(?:/|$)"
    )
    for text in artifact_text.values():
        unescaped = text.replace("\\\\", "\\")
        assert windows_absolute.search(unescaped) is None
        assert posix_local.search(unescaped) is None
        assert unc_path.rsplit("\\", 1)[0] not in unescaped
        assert windows_user_path.rsplit("\\", 2)[0] not in unescaped
        assert windows_python_root not in unescaped

    combined = "\n".join(artifact_text.values())
    assert "<LOCAL_PATH>" in combined
    assert "<PROJECT_ROOT>" in artifact_text["trajectory.jsonl"]
    assert "https://docs.example.test/help" in combined
    assert "https://invalid.example.test/v1" in combined
    assert "reports/test_report.md" in combined
    assert "C/C=C/C" in combined

"""Bounded, auditable, tool-calling agent runner for xTB recovery cases."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from typing import Any
import uuid

import yaml

from .redaction import safe_exception_record, sanitize_for_persistence, to_plain
from .schemas import (
    BudgetExceededError,
    ConfigurationError,
    FrameworkError,
    LifecycleError,
    MissingCredentialError,
    ModelConfig,
    ProtocolViolationError,
    ProviderError,
    RunResult,
    TaskConfig,
    ToolContractError,
    ToolContractSet,
)
from .siliconflow_adapter import SiliconFlowChatTransport, load_runtime_credential
from .trajectory import (
    HASH_ALGORITHM,
    TrajectoryRecorder,
    canonical_json,
    render_dialogue,
    verify_hash_chain,
)
from .trusted_final import build_final_answer_envelope


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = (
    "trajectory.jsonl",
    "dialogue.md",
    "metadata.json",
    "final_answer.json",
)
_TOOL_CALL_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")


class ExactModelMismatchError(ProviderError):
    """A response identified a model other than the exact requested ID."""


class ScriptedTransport:
    """Deterministic transport used to exercise the exact runner without I/O."""

    requires_api_key = False
    is_mock = True
    transport_name = "scripted_stub"

    def __init__(self, responses: Iterable[Any]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> Any:
        self.requests.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": [dict(tool) for tool in tools],
                "timeout_seconds": timeout_seconds,
            }
        )
        try:
            item = next(self._responses)
        except StopIteration as exc:
            raise ProviderError("SCRIPTED_RESPONSES_EXHAUSTED") from exc
        if isinstance(item, BaseException):
            raise item
        return item


@dataclass(frozen=True)
class ToolExecutionContext:
    """Minimal optional context for mapping-based test tool callables."""

    call_id: str
    tool_name: str
    task_config: Mapping[str, Any]
    emit: Callable[[str, Mapping[str, Any]], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError("configuration file is missing")
    try:
        raw = path.read_bytes()
        if len(raw) > 1_048_576 or b"\x00" in raw:
            raise ConfigurationError("configuration file exceeds safe limits")
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError("configuration file is not safe UTF-8 YAML") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError("configuration root must be an object")
    return loaded


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_run_id(logical_model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", logical_model).strip("-.") or "run"
    return f"{safe}-{uuid.uuid4().hex}"


class AgentRunner:
    """Run one exact model through the bounded recovery tool protocol."""

    def __init__(
        self,
        *,
        task_config: TaskConfig | Mapping[str, Any] | str | Path,
        model_config: ModelConfig | Mapping[str, Any] | str | Path | None = None,
        logical_model: str | None = None,
        output_dir: str | Path,
        prompt: str | Path | None = None,
        tool_contracts: ToolContractSet | Mapping[str, Any] | str | Path | None = None,
        tool_registry: Any | None = None,
        transport: Any | None = None,
        execution_mode: str | None = None,
        native_mode: str | None = None,
        run_type: str | None = None,
        run_id: str | None = None,
        artifact_root: str | Path | None = None,
        timestamp_fn: Callable[[], Any] = _utc_now,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.task = self._task_config(task_config)
        self.logical_model = logical_model or self._default_logical_model(self.task)
        self.model = self._model_config(model_config, self.logical_model, self.task)
        if self.model.logical_name != self.logical_model:
            raise ConfigurationError("logical model and selected model configuration differ")
        self._assert_exact_registered_model(self.model, self.logical_model, self.task)
        self.contracts = self._tool_contracts(tool_contracts, self.task)
        if self.contracts.protocol != self.task.protocol:
            raise ConfigurationError("task and tool-contract protocol identities differ")
        if set(self.contracts.names) != set(self.task.allowed_tools):
            raise ConfigurationError("task allowed tools and tool contracts differ")

        self.output_dir = Path(output_dir)
        self.prompt = self._prompt(prompt)
        self.tool_registry = tool_registry
        self._injected_transport = transport
        is_test_transport = transport is not None and bool(getattr(transport, "is_mock", True))
        models_raw = self.task.raw.get("models") if isinstance(self.task.raw, Mapping) else {}
        models_raw = models_raw if isinstance(models_raw, Mapping) else {}
        self.execution_mode = execution_mode or (
            "TEST_STUB" if is_test_transport else str(models_raw.get("formal_execution_mode", "REAL_API"))
        )
        self.native_mode = native_mode or (
            "NO_NATIVE_EXECUTION" if is_test_transport else str(models_raw.get("formal_native_mode", "LIVE_XTB"))
        )
        self.run_type = run_type or ("TEST" if is_test_transport else "FORMAL_MODEL")
        if self.execution_mode == "REAL_API" and transport is not None:
            raise ConfigurationError("REAL_API mode forbids injected transports")
        if self.native_mode == "LIVE_XTB" and (
            tool_registry is None
            or getattr(tool_registry, "live_native_execution", False) is not True
        ):
            raise ConfigurationError("LIVE_XTB mode requires the live native tool registry")
        if artifact_root is None:
            if self.run_type == "FORMAL_MODEL":
                raise ConfigurationError("formal runs require an explicit artifact root")
            self.artifact_root: Path | None = None
        else:
            try:
                resolved_artifact_root = Path(artifact_root).resolve(strict=True)
            except OSError as exc:
                raise ConfigurationError("artifact root is unavailable") from exc
            if not resolved_artifact_root.is_dir():
                raise ConfigurationError("artifact root must be a directory")
            self.artifact_root = resolved_artifact_root
        self.run_id = run_id or _safe_run_id(self.logical_model)
        if not isinstance(self.run_id, str) or not self.run_id or len(self.run_id) > 128:
            raise ConfigurationError("run_id must be a non-empty bounded identifier")
        if any(character in self.run_id for character in ("/", "\\", ":")):
            raise ConfigurationError("run_id must not contain path syntax")

        self._timestamp_fn = timestamp_fn
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self._started_monotonic = 0.0
        self._recorder: TrajectoryRecorder | None = None
        self._messages: list[dict[str, Any]] = []
        self._secrets: tuple[str, ...] = ()
        self._seen_tool_call_ids: set[str] = set()
        self._tool_call_count = 0
        self._successful_tool_calls = 0
        self._failed_tool_calls = 0
        self._provider_attempts = 0
        self._provider_failures = 0
        self._provider_retries = 0
        self._agent_turns = 0
        self._execution_attempts = 0
        self._native_executions = 0
        self._actual_model: str | None = None
        self._request_ids: list[str] = []
        self._submitted = False
        self._model_submission: dict[str, Any] = {}
        self._caught: BaseException | None = None

    @staticmethod
    def _task_config(value: TaskConfig | Mapping[str, Any] | str | Path) -> TaskConfig:
        if isinstance(value, TaskConfig):
            return value
        if isinstance(value, Mapping):
            return TaskConfig.from_mapping(value)
        return TaskConfig.load(_resolve_project_path(value))

    @staticmethod
    def _default_logical_model(task: TaskConfig) -> str:
        models = task.raw.get("models") if isinstance(task.raw, Mapping) else None
        order = models.get("formal_order") if isinstance(models, Mapping) else None
        if isinstance(order, list) and order and isinstance(order[0], str):
            return order[0]
        raise ConfigurationError("logical_model is required")

    @staticmethod
    def _model_config(
        value: ModelConfig | Mapping[str, Any] | str | Path | None,
        logical_model: str,
        task: TaskConfig,
    ) -> ModelConfig:
        if isinstance(value, ModelConfig):
            return value
        if value is None:
            models = task.raw.get("models") if isinstance(task.raw, Mapping) else None
            registry = models.get("registry_path") if isinstance(models, Mapping) else None
            if not isinstance(registry, str):
                raise ConfigurationError("model registry path is missing")
            return ModelConfig.load_registry(_resolve_project_path(registry), logical_model)
        if isinstance(value, Mapping):
            if "provider_defaults" in value and "models" in value:
                return ModelConfig.from_registry(value, logical_model)
            return ModelConfig.from_mapping(value, logical_name=logical_model)
        document = _safe_yaml(_resolve_project_path(value))
        return ModelConfig.from_registry(document, logical_model)

    @staticmethod
    def _assert_exact_registered_model(
        selected: ModelConfig,
        logical_model: str,
        task: TaskConfig,
    ) -> None:
        models = task.raw.get("models") if isinstance(task.raw, Mapping) else None
        registry = models.get("registry_path") if isinstance(models, Mapping) else None
        if not isinstance(registry, str) or not registry:
            raise ConfigurationError("authoritative model registry path is missing")
        expected = ModelConfig.load_registry(_resolve_project_path(registry), logical_model)
        if selected != expected:
            raise ConfigurationError("selected model differs from the authoritative exact registry entry")

    @staticmethod
    def _tool_contracts(
        value: ToolContractSet | Mapping[str, Any] | str | Path | None,
        task: TaskConfig,
    ) -> ToolContractSet:
        if isinstance(value, ToolContractSet):
            return value
        if isinstance(value, Mapping):
            return ToolContractSet.from_mapping(value)
        if value is None:
            tools = task.raw.get("tools") if isinstance(task.raw, Mapping) else None
            contract_path = tools.get("contract_path") if isinstance(tools, Mapping) else None
            if not isinstance(contract_path, str):
                raise ConfigurationError("tool contract path is missing")
            value = contract_path
        return ToolContractSet.load(_resolve_project_path(value))

    @staticmethod
    def _prompt(value: str | Path | None) -> str:
        if value is None:
            value = PROJECT_ROOT / "prompts" / "agent_prompt.md"
        candidate: Path | None = value if isinstance(value, Path) else None
        if isinstance(value, str) and "\n" not in value and "\r" not in value:
            possible = _resolve_project_path(value)
            if possible.is_file():
                candidate = possible
        if candidate is not None:
            try:
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ConfigurationError("agent prompt could not be read") from exc
        else:
            text = str(value)
        if not text.strip() or len(text.encode("utf-8")) > 1_048_576:
            raise ConfigurationError("agent prompt must be non-empty and bounded")
        return text

    @property
    def messages(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._messages, ensure_ascii=False))

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._recorder.events if self._recorder is not None else []

    @property
    def submission_frozen(self) -> bool:
        return self._submitted

    def _claim_output_dir(self) -> None:
        try:
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise ConfigurationError("output directory already exists; overwrite is forbidden") from exc
        except OSError as exc:
            raise ConfigurationError("output directory could not be claimed") from exc

    def _elapsed(self) -> float:
        return max(0.0, self._monotonic() - self._started_monotonic)

    def _remaining(self) -> float:
        return self.task.budget.total_timeout_seconds - self._elapsed()

    def _check_timeout(self) -> None:
        if self._remaining() <= 0:
            raise BudgetExceededError("TOTAL_TIMEOUT_EXCEEDED")

    def _safe(self, value: Any) -> Any:
        return sanitize_for_persistence(value, secrets=self._secrets)

    def _emit(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._recorder is None:
            raise LifecycleError("trajectory recorder is unavailable")
        return self._recorder.append(event, self._safe(dict(payload)))

    def _append_message(self, message: Mapping[str, Any], *, source: str) -> None:
        safe = self._safe(dict(message))
        if not isinstance(safe, dict):
            raise ProtocolViolationError("MESSAGE_SANITIZATION_FAILED")
        self._messages.append(safe)
        # Only provider assistant messages belong in the public trajectory.
        # System/user/tool context remains represented by run/tool events and
        # is retained in memory solely for the next provider request.
        if safe.get("role") == "assistant":
            public_message: dict[str, Any] = {
                "role": "assistant",
                "content": safe.get("content"),
            }
            calls = safe.get("tool_calls")
            if isinstance(calls, list):
                public_calls: list[dict[str, Any]] = []
                for raw_call in calls:
                    function = raw_call.get("function") if isinstance(raw_call, Mapping) else None
                    public_calls.append(
                        {
                            "id": raw_call.get("id") if isinstance(raw_call, Mapping) else None,
                            "type": "function",
                            "function": {
                                "name": function.get("name") if isinstance(function, Mapping) else None
                            },
                        }
                    )
                public_message["tool_calls"] = public_calls
            self._emit("assistant_message", {"source": source, "message": public_message})

    def _emit_budget_state(self) -> None:
        self._emit(
            "budget_state",
            {
                "tool_calls": self._tool_call_count,
                "successful_tool_calls": self._successful_tool_calls,
                "agent_turns": self._agent_turns,
                "execution_attempts": self._execution_attempts,
                "provider_retries": self._provider_retries,
                "elapsed_seconds": self._elapsed(),
            },
        )

    def _initial_messages(self) -> list[dict[str, Any]]:
        raw_budget = self.task.raw.get("budget") if isinstance(self.task.raw, Mapping) else {}
        raw_budget = raw_budget if isinstance(raw_budget, Mapping) else {}
        runtime = self.task.raw.get("runtime") if isinstance(self.task.raw, Mapping) else {}
        runtime = runtime if isinstance(runtime, Mapping) else {}
        xtb = runtime.get("xtb") if isinstance(runtime, Mapping) else {}
        xtb = xtb if isinstance(xtb, Mapping) else {}
        task_payload = {
            "task_id": self.task.task_id,
            "case_id": self.task.case_id,
            "protocol": self.task.protocol.to_dict(),
            # These are public policy limits the model is required to obey.
            # Passing the complete configured budget avoids forcing the model
            # to guess hidden read/native-execution bounds.
            "budget": dict(raw_budget),
            "restart_resource_limits": {
                "optimization_cycle_limit_min": xtb.get("restart_cycle_limit_min"),
                "optimization_cycle_limit_max": xtb.get("restart_cycle_limit_max"),
                "thread_count_min": xtb.get("thread_count_min"),
                "thread_count_max": xtb.get("thread_count_max"),
                "walltime_seconds_max": xtb.get("walltime_seconds_max"),
            },
            "allowed_tools": list(self.task.allowed_tools),
            "instruction": "Use only declared tools and finish by calling submit_final exactly once.",
        }
        return [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": json.dumps(task_payload, ensure_ascii=False, sort_keys=True, indent=2),
            },
        ]

    @staticmethod
    def _transport_name(transport: Any) -> str:
        name = getattr(transport, "transport_name", None)
        return str(name) if isinstance(name, str) and name else type(transport).__name__

    def _extract_provider_response(
        self, response: Any, transport: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        plain = to_plain(response)
        if not isinstance(plain, Mapping):
            raise ProviderError("PROVIDER_RESPONSE_NOT_OBJECT")
        actual_model = plain.get("model")
        choices = plain.get("choices")
        if not isinstance(actual_model, str) or not actual_model:
            raise ProviderError("PROVIDER_RESPONSE_MODEL_MISSING")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise ProviderError("PROVIDER_RESPONSE_CHOICES_INVALID")
        choice = choices[0]
        raw_message = choice.get("message")
        if not isinstance(raw_message, Mapping):
            raise ProviderError("PROVIDER_RESPONSE_MESSAGE_MISSING")

        safe_message_source = self._safe(dict(raw_message))
        if not isinstance(safe_message_source, Mapping):
            raise ProviderError("PROVIDER_MESSAGE_SANITIZATION_FAILED")
        content = safe_message_source.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        tool_calls = safe_message_source.get("tool_calls")
        if tool_calls is None:
            tool_calls = []
        if not isinstance(tool_calls, list):
            raise ProviderError("PROVIDER_TOOL_CALLS_INVALID")
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls

        request_id = plain.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ProviderError("PROVIDER_RESPONSE_REQUEST_ID_MISSING")
        usage = plain.get("usage")
        safe_usage = self._safe(usage if isinstance(usage, Mapping) else {})
        metadata = {
            "request_id": request_id,
            "actual_model": actual_model,
            "finish_reason": choice.get("finish_reason"),
            "usage": safe_usage,
            "transport": self._transport_name(transport),
        }
        return assistant, metadata

    def _provider_call(self, transport: Any, turn: int) -> dict[str, Any]:
        retry_budget = min(self.task.budget.max_provider_retries, self.model.max_retries)
        attempts = retry_budget + 1
        for attempt in range(1, attempts + 1):
            self._check_timeout()
            self._provider_attempts += 1
            timeout = min(self._remaining(), self.model.timeout_seconds)
            self._emit(
                "provider_request",
                {
                    "turn": turn,
                    "attempt": attempt,
                    "requested_model": self.model.model,
                    "message_count": len(self._messages),
                    "tool_names": list(self.contracts.names),
                    "timeout_seconds": timeout,
                    "transport": self._transport_name(transport),
                },
            )
            try:
                response = transport.complete(
                    messages=self._messages,
                    tools=self.contracts.provider_definitions(),
                    timeout_seconds=timeout,
                )
                self._check_timeout()
                assistant, metadata = self._extract_provider_response(response, transport)
                self._emit("provider_response_metadata", metadata)
                actual = metadata["actual_model"]
                if actual != self.model.model:
                    raise ExactModelMismatchError("EXACT_RESPONSE_MODEL_MISMATCH")
                if self._actual_model is None:
                    self._actual_model = actual
                elif self._actual_model != actual:
                    raise ExactModelMismatchError("RESPONSE_MODEL_CHANGED")
                self._request_ids.append(metadata["request_id"])
                return assistant
            except ExactModelMismatchError as exc:
                self._provider_failures += 1
                self._emit(
                    "provider_error",
                    {
                        "turn": turn,
                        "attempt": attempt,
                        "will_retry": False,
                        "error": safe_exception_record(exc, status="MODEL_IDENTITY_REJECTED"),
                    },
                )
                raise
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                self._provider_failures += 1
                will_retry = attempt < attempts
                self._emit(
                    "provider_error",
                    {
                        "turn": turn,
                        "attempt": attempt,
                        "will_retry": will_retry,
                        "error": safe_exception_record(exc, status="PROVIDER_REQUEST_FAILED"),
                    },
                )
                if not will_retry:
                    raise ProviderError("PROVIDER_RETRIES_EXHAUSTED") from None
                self._provider_retries += 1
                self._emit(
                    "provider_retry",
                    {
                        "turn": turn,
                        "failed_attempt": attempt,
                        "next_attempt": attempt + 1,
                        "error": safe_exception_record(
                            exc, status="PROVIDER_REQUEST_FAILED"
                        ),
                    },
                )
                delay = min(2 ** (attempt - 1), max(0.0, self._remaining()))
                if delay > 0:
                    self._sleep(delay)
        raise ProviderError("PROVIDER_RETRIES_EXHAUSTED")

    @staticmethod
    def _decode_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)
        if not isinstance(raw, str):
            raise ToolContractError("tool arguments must be a JSON string or object")

        def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ToolContractError("tool arguments contain duplicate keys")
                result[key] = value
            return result

        try:
            value = json.loads(
                raw,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ToolContractError("tool arguments contain a non-finite value")
                ),
            )
        except ToolContractError:
            raise
        except json.JSONDecodeError as exc:
            raise ToolContractError("tool arguments are malformed JSON") from exc
        if not isinstance(value, dict):
            raise ToolContractError("tool arguments must decode to an object")
        return value

    def _call_identity(self, raw_call: Any) -> tuple[str, str, Any, bool]:
        plain = to_plain(raw_call)
        if not isinstance(plain, Mapping):
            synthetic = f"invalid_tool_call_{self._tool_call_count:03d}"
            return synthetic, "", None, False
        raw_id = plain.get("id")
        valid_id = isinstance(raw_id, str) and bool(_TOOL_CALL_ID.fullmatch(raw_id))
        if not valid_id or raw_id in self._seen_tool_call_ids:
            prefix = "duplicate" if valid_id else "invalid"
            synthetic = f"{prefix}_tool_call_{self._tool_call_count:03d}"
            return synthetic, "", None, False
        self._seen_tool_call_ids.add(raw_id)
        if plain.get("type") != "function":
            return raw_id, "", None, False
        function = plain.get("function")
        if not isinstance(function, Mapping):
            return raw_id, "", None, False
        name = function.get("name")
        return raw_id, str(name) if isinstance(name, str) else "", function.get("arguments"), True

    @staticmethod
    def _tool_message(call_id: str, name: str, ok: bool, value: Mapping[str, Any]) -> dict[str, Any]:
        content_value: dict[str, Any] = {"ok": ok, "call_id": call_id, **dict(value)}
        nested = value.get("result")
        if ok and name == "read_text_range" and isinstance(nested, Mapping):
            # This is a lossless projection of the trusted read result, not a
            # model claim.  It makes the exact citation contract copyable and
            # prevents providers from guessing call IDs or requested EOF lines.
            content_value["evidence_reference"] = {
                "tool_call_id": call_id,
                "path": nested.get("path"),
                "line_start": nested.get("line_start"),
                "line_end": nested.get("line_end"),
                "content_sha256": nested.get("content_sha256"),
            }
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name or "invalid_tool",
            "content": json.dumps(
                content_value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
        }

    def _dispatch_mapping_tool(self, name: str, arguments: dict[str, Any], call_id: str) -> Any:
        if not isinstance(self.tool_registry, Mapping):
            raise ToolContractError("tool registry does not support dispatch")
        function = self.tool_registry.get(name)
        if not callable(function):
            raise ToolContractError("tool implementation is unavailable")
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return function(**arguments)
        parameters = signature.parameters
        context = ToolExecutionContext(
            call_id=call_id,
            tool_name=name,
            task_config=self.task.raw,
            emit=lambda event, payload: self._emit(event, payload),
        )
        extras: dict[str, Any] = {}
        if "context" in parameters:
            extras["context"] = context
        if "config" in parameters:
            extras["config"] = self.task.raw
        elif "task_config" in parameters:
            extras["task_config"] = self.task.raw
        if "call_id" in parameters:
            extras["call_id"] = call_id
        if "arguments" in parameters and not set(arguments).intersection(parameters):
            return function(arguments=arguments, **extras)
        return function(**arguments, **extras)

    def _dispatch_tool(self, name: str, arguments: dict[str, Any], call_id: str) -> Any:
        dispatch = getattr(self.tool_registry, "dispatch", None)
        if callable(dispatch):
            return dispatch(name, arguments, call_id=call_id)
        return self._dispatch_mapping_tool(name, arguments, call_id)

    @staticmethod
    def _normalize_tool_result(value: Mapping[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
        """Normalize registry envelopes and small injected-stub responses.

        The production registry returns ``{ok, tool, call_id, result|error}``.
        The canonical trajectory records the trusted result itself, not that
        transport envelope.  Mapping-based stubs may return either shape.
        """

        ok = value.get("ok", True) is True
        if not ok:
            error = value.get("error")
            code = error.get("code") if isinstance(error, Mapping) else None
            safe_code = (
                code
                if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code)
                else None
            )
            return False, {}, safe_code
        nested = value.get("result")
        envelope_shape = (
            {"tool", "call_id"}.issubset(value)
            or set(value).issubset({"ok", "result"})
        )
        if isinstance(nested, Mapping) and envelope_shape:
            return True, dict(nested), None
        result = dict(value)
        result.pop("ok", None)
        return True, result, None

    @staticmethod
    def _special_event_payload(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
        """Extract the immutable attempt evidence fields from a successful result."""

        if name == "prepare_restart":
            source = result.get("source_geometry")
            source = source if isinstance(source, Mapping) else {}
            return {
                "attempt_id": result.get("attempt_id"),
                "phase": result.get("phase"),
                "state": result.get("state"),
                "source_geometry": {
                    "path": source.get("path"),
                    "frame_index": source.get("frame_index"),
                    "sha256": source.get("sha256", source.get("file_sha256")),
                    "provenance_state": source.get(
                        "provenance_state", result.get("source_provenance")
                    ),
                },
                "prepared_command_path": result.get(
                    "prepared_command_path", result.get("prepared_record")
                ),
                "input_manifest_path": result.get(
                    "input_manifest_path", result.get("provenance_record")
                ),
            }
        if name == "run_xtb":
            hashes = result.get("artifact_hashes")
            if not isinstance(hashes, Mapping):
                files = result.get("files")
                hashes = {
                    str(path): metadata.get("sha256")
                    for path, metadata in files.items()
                    if isinstance(path, str)
                    and isinstance(metadata, Mapping)
                    and isinstance(metadata.get("sha256"), str)
                } if isinstance(files, Mapping) else {}
            return {
                "attempt_id": result.get("attempt_id"),
                "phase": result.get("phase"),
                "source_type": "NATIVE_XTB",
                "argv": result.get("argv"),
                "shell": result.get("shell"),
                "process_exit_code": result.get(
                    "process_exit_code", result.get("exit_code")
                ),
                "timed_out": result.get("timed_out"),
                "execution_summary_path": result.get(
                    "execution_summary_path", result.get("run_record")
                ),
                "artifact_manifest_path": result.get("artifact_manifest_path"),
                "artifact_hashes": dict(hashes),
            }
        if name == "validate_attempt":
            source_provenance = result.get("source_provenance")
            if not isinstance(source_provenance, (str, Mapping)):
                source_provenance = {"current": result.get("source_current") is True}
            return {
                "attempt_id": result.get("attempt_id"),
                "phase": result.get("phase"),
                "classification": result.get("classification", result.get("state")),
                "valid": result.get("valid", result.get("validation_passed")),
                "optimization_converged": result.get(
                    "optimization_converged", result.get("optimization_completed")
                ),
                "hessian_completed": result.get("hessian_completed"),
                "current_artifacts": result.get(
                    "current_artifacts", result.get("artifacts_current")
                ),
                "scientific_invariants_preserved": result.get(
                    "scientific_invariants_preserved", result.get("invariants_preserved")
                ),
                "source_provenance": source_provenance,
                "validation_path": result.get(
                    "validation_path", result.get("validation_record")
                ),
            }
        raise ProtocolViolationError("UNKNOWN_SPECIAL_TOOL_EVENT")

    def _failed_tool_result(
        self,
        *,
        call_id: str,
        name: str,
        error_type: str,
        status: str,
    ) -> None:
        self._failed_tool_calls += 1
        error = {"type": error_type, "status": status}
        self._emit("tool_result", {"call_id": call_id, "name": name, "ok": False, "error": error})
        self._emit_budget_state()
        self._append_message(
            self._tool_message(call_id, name, False, {"error": error}),
            source="runner_tool_failure",
        )

    def _execute_tool_call(self, raw_call: Any) -> bool:
        if self._submitted:
            raise ProtocolViolationError("SUBMISSION_ALREADY_FROZEN")
        self._check_timeout()
        self._tool_call_count += 1
        call_id, name, raw_arguments, identity_valid = self._call_identity(raw_call)
        if self._tool_call_count > self.task.budget.max_tool_calls:
            self._emit("tool_call", {"call_id": call_id, "name": name, "arguments": {}})
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type="ToolBudgetExceeded",
                status="REJECTED",
            )
            raise BudgetExceededError("TOOL_CALL_BUDGET_EXCEEDED")
        if not identity_valid:
            self._emit("tool_call", {"call_id": call_id, "name": name, "arguments": {}})
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type="InvalidOrDuplicateToolCallId",
                status="REJECTED",
            )
            raise ProtocolViolationError("TOOL_CALL_ID_NOT_UNIQUE")

        try:
            arguments = self._decode_arguments(raw_arguments)
            arguments = self.contracts.validate_arguments(name, arguments)
        except ToolContractError as exc:
            self._emit("tool_call", {"call_id": call_id, "name": name, "arguments": {}})
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type=type(exc).__name__,
                status="REJECTED",
            )
            return False

        self._emit("tool_call", {"call_id": call_id, "name": name, "arguments": arguments})
        if (
            name == "submit_final"
            and self._successful_tool_calls + 1 < self.task.budget.min_successful_tool_calls
        ):
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type="MinimumSuccessfulToolsNotMet",
                status="REJECTED",
            )
            return False
        if name == "prepare_restart" and self._execution_attempts >= self.task.budget.max_execution_attempts:
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type="ExecutionAttemptBudgetExceeded",
                status="REJECTED",
            )
            raise BudgetExceededError("EXECUTION_ATTEMPT_BUDGET_EXCEEDED")
        if name == "run_xtb" and self._native_executions >= self.task.budget.max_execution_attempts:
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type="NativeExecutionBudgetExceeded",
                status="REJECTED",
            )
            raise BudgetExceededError("NATIVE_EXECUTION_BUDGET_EXCEEDED")

        try:
            raw_result = self._dispatch_tool(name, arguments, call_id)
            result = self._safe(raw_result)
            if not isinstance(result, Mapping):
                result = {"value": result}
            reported_ok, normalized_result, tool_error_code = self._normalize_tool_result(result)
            if not reported_ok:
                self._failed_tool_result(
                    call_id=call_id,
                    name=name,
                    error_type=tool_error_code or "ToolReportedFailure",
                    status="REJECTED",
                )
                return False
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type=type(exc).__name__,
                status="FAILED",
            )
            return False

        safe_result = normalized_result
        self._successful_tool_calls += 1
        if name == "prepare_restart":
            self._execution_attempts += 1
        elif name == "run_xtb":
            self._native_executions += 1
        self._emit("tool_result", {"call_id": call_id, "name": name, "ok": True, "result": safe_result})
        self._append_message(
            self._tool_message(call_id, name, True, {"result": safe_result}),
            source="tool_registry",
        )
        special_events = {
            "prepare_restart": "attempt_prepared",
            "run_xtb": "native_xtb_execution",
            "validate_attempt": "attempt_validated",
        }
        if name in special_events:
            self._emit(special_events[name], self._special_event_payload(name, safe_result))
        self._emit_budget_state()
        if name == "submit_final":
            self._model_submission = self._safe(arguments)
            if not isinstance(self._model_submission, dict):
                raise ProtocolViolationError("FINAL_ANSWER_SANITIZATION_FAILED")
            self._emit("final_submitted", {"final_answer": self._model_submission})
            self._submitted = True
            return True
        return False

    @staticmethod
    def _tool_name(raw_call: Any) -> str:
        plain = to_plain(raw_call)
        function = plain.get("function") if isinstance(plain, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        return str(name) if isinstance(name, str) else ""

    def _reject_mixed_final_calls(self, tool_calls: Sequence[Any]) -> None:
        for raw_call in tool_calls:
            self._tool_call_count += 1
            call_id, name, _raw_arguments, identity_valid = self._call_identity(raw_call)
            if self._tool_call_count > self.task.budget.max_tool_calls:
                raise BudgetExceededError("TOOL_CALL_BUDGET_EXCEEDED")
            self._emit("tool_call", {"call_id": call_id, "name": name, "arguments": {}})
            self._failed_tool_result(
                call_id=call_id,
                name=name,
                error_type=("MixedSubmitFinalBatch" if identity_valid else "InvalidOrDuplicateToolCallId"),
                status="REJECTED",
            )
        raise ProtocolViolationError("SUBMIT_FINAL_MUST_BE_THE_ONLY_TOOL_CALL")

    def _run_loop(self, transport: Any) -> None:
        for message in self._initial_messages():
            self._append_message(message, source="runner")

        for turn in range(1, self.task.budget.max_agent_turns + 1):
            self._check_timeout()
            self._agent_turns = turn
            assistant = self._provider_call(transport, turn)
            self._append_message(assistant, source="provider")
            tool_calls = assistant.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ProtocolViolationError("ASSISTANT_TOOL_CALLS_NOT_LIST")
            names = [self._tool_name(call) for call in tool_calls]
            if "submit_final" in names and len(tool_calls) != 1:
                self._reject_mixed_final_calls(tool_calls)
            if tool_calls:
                for raw_call in tool_calls:
                    submitted = self._execute_tool_call(raw_call)
                    if submitted:
                        return
                continue
            self._emit_budget_state()
            self._append_message(
                {
                    "role": "user",
                    "content": (
                        "A final text response is not accepted. Continue with declared tools, "
                        "or call submit_final exactly once with its strict object."
                    ),
                },
                source="runner_protocol_guard",
            )
        raise BudgetExceededError("AGENT_TURN_BUDGET_EXCEEDED")

    def _failure_answer(self, exc: BaseException) -> dict[str, Any]:
        return {
            "status": "RUN_FAILED",
            "error": safe_exception_record(exc, status="RUN_FAILED"),
        }

    def _metadata(
        self,
        *,
        status: str,
        failure: Mapping[str, Any] | None,
        artifact_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        if self._recorder is None:
            raise ProtocolViolationError("RECORDER_NOT_INITIALIZED")
        events = self._recorder.events
        usage_totals: dict[str, int] = {}
        for event in events:
            if event.get("event") != "provider_response_metadata":
                continue
            payload = event.get("payload")
            usage = payload.get("usage") if isinstance(payload, Mapping) else None
            if not isinstance(usage, Mapping):
                continue
            for key, value in usage.items():
                if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
                    usage_totals[key] = usage_totals.get(key, 0) + value
        started_at = events[0]["timestamp_utc"] if events else None
        ended_at = events[-1]["timestamp_utc"] if events else None
        duration_seconds = (
            max(0.0, self._monotonic() - self._started_monotonic)
            if self._started_monotonic is not None
            else None
        )
        xtb_runtime = self.task.raw.get("runtime", {}).get("xtb", {})
        expected_xtb_version = (
            xtb_runtime.get("version") if isinstance(xtb_runtime, Mapping) else None
        )
        return {
            "schema_version": "1.0",
            "status": status,
            "run_id": self.run_id,
            "task_id": self.task.task_id,
            "case_id": self.task.case_id,
            "protocol": self.task.protocol.to_dict(),
            "run_type": self.run_type,
            "execution_mode": self.execution_mode,
            "native_mode": self.native_mode,
            "logical_model": self.logical_model,
            "requested_model": self.model.model,
            "actual_model": self._actual_model,
            "provider": self.model.provider,
            "transport": self.model.transport,
            "endpoint": self.model.base_url,
            "temperature": self.model.temperature,
            "seed": None,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(duration_seconds, 6) if duration_seconds is not None else None,
            "token_usage": usage_totals,
            "environment_versions": {
                "python": platform.python_version(),
                "xtb": str(expected_xtb_version) if expected_xtb_version is not None else None,
            },
            "model_config": self.model.sanitized_dict(),
            "counts": {
                "agent_turns": self._agent_turns,
                "provider_attempts": self._provider_attempts,
                "provider_failures": self._provider_failures,
                "provider_retries": self._provider_retries,
                "tool_calls": self._tool_call_count,
                "successful_tool_calls": self._successful_tool_calls,
                "failed_tool_calls": self._failed_tool_calls,
                "execution_attempts": self._execution_attempts,
                "native_executions": self._native_executions,
            },
            "limits": self.task.budget.to_dict(),
            "submitted": self._submitted,
            "submission_frozen": self._submitted,
            "hash_chain": {
                "algorithm": HASH_ALGORITHM,
                "genesis_hash": self.task.genesis_hash,
                "head_hash": self._recorder.head_hash,
                "event_count": len(self._recorder.events),
                "valid": verify_hash_chain(
                    self._recorder.events,
                    genesis_hash=self.task.genesis_hash,
                    require_terminal=True,
                ),
            },
            "artifact_sha256": dict(artifact_hashes),
            "request_ids": list(self._request_ids),
            "hidden_reasoning_persisted": False,
            "failure": dict(failure) if isinstance(failure, Mapping) else None,
            "artifacts": list(ARTIFACT_NAMES),
        }

    def _write_artifacts(
        self,
        *,
        status: str,
        failure: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._recorder is None or not self._recorder.terminal:
            raise ProtocolViolationError("TERMINAL_TRAJECTORY_REQUIRED")
        model_submission = self._safe(self._model_submission)
        if not isinstance(model_submission, dict):
            raise ProtocolViolationError("FINAL_ANSWER_NOT_OBJECT")
        trajectory_path = self.output_dir / "trajectory.jsonl"
        self._recorder.write(trajectory_path)
        # Generate every derived view from the exact canonical bytes that were
        # persisted.  In-memory values such as 1.0 may canonicalize to 1 in
        # JSONL; rendering the pre-serialization objects would then make the
        # human view fail deterministic replay despite identical semantics.
        persisted_events = [
            json.loads(line)
            for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        if not persisted_events or any(not isinstance(event, dict) for event in persisted_events):
            raise ProtocolViolationError("PERSISTED_TRAJECTORY_NOT_OBJECTS")
        final_answer = build_final_answer_envelope(
            model_submission=model_submission,
            events=persisted_events,
            trajectory_path=trajectory_path,
            artifact_root=self.artifact_root,
            genesis_hash=self.task.genesis_hash,
        )
        final_answer = self.contracts.validate_final_answer_artifact(
            final_answer,
            require_valid_submission=self._submitted,
        )
        final_path = self.output_dir / "final_answer.json"
        _write_json_exclusive(final_path, final_answer)
        dialogue_path = self.output_dir / "dialogue.md"
        with dialogue_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(render_dialogue(persisted_events))
        hashes = {
            "final_answer.json": _sha256_file(final_path),
            "trajectory.jsonl": _sha256_file(trajectory_path),
            "dialogue.md": _sha256_file(dialogue_path),
        }
        metadata = self._safe(self._metadata(status=status, failure=failure, artifact_hashes=hashes))
        if not isinstance(metadata, dict):
            raise ProtocolViolationError("METADATA_NOT_OBJECT")
        _write_json_exclusive(self.output_dir / "metadata.json", metadata)
        return metadata, final_answer

    def run(self, *, raise_on_failure: bool = False) -> RunResult:
        self._claim_output_dir()
        self._started_monotonic = self._monotonic()
        self._recorder = TrajectoryRecorder(
            protocol=self.task.protocol,
            run_id=self.run_id,
            genesis_hash=self.task.genesis_hash,
            timestamp_fn=self._timestamp_fn,
        )
        self._emit(
            "run_start",
            {
                "task_id": self.task.task_id,
                "case_id": self.task.case_id,
                "protocol": self.task.protocol.to_dict(),
                "run_type": self.run_type,
                "logical_model": self.logical_model,
                "requested_model": self.model.model,
                "provider": self.model.provider,
                "transport": self.model.transport,
                "execution_mode": self.execution_mode,
                "native_mode": self.native_mode,
            },
        )
        status = "FAILED"
        failure: dict[str, Any] | None = None
        try:
            if self._injected_transport is None:
                api_key = load_runtime_credential(PROJECT_ROOT)
                self._secrets = (api_key,)
                # Register the secret before any object that can reflect it is
                # inspected or persisted.
                self._recorder.register_secret(api_key)
                transport = SiliconFlowChatTransport(self.model, api_key)
                del api_key
            else:
                transport = self._injected_transport
            if self.tool_registry is None:
                raise ConfigurationError("TOOL_REGISTRY_MISSING")
            self._run_loop(transport)
            if not self._submitted:
                raise ProtocolViolationError("FINAL_SUBMISSION_MISSING")
            self._emit(
                "run_complete",
                {
                    "status": "COMPLETE",
                    "terminal_status": "COMPLETE",
                    "actual_model": self._actual_model,
                    "tool_calls": self._tool_call_count,
                    "successful_tool_calls": self._successful_tool_calls,
                },
            )
            status = "COMPLETE"
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                self._caught = exc
            else:
                self._caught = exc
            failure = safe_exception_record(exc, status="RUN_FAILED")
            if not self._submitted and not self._recorder.terminal:
                self._emit_budget_state()
                self._emit(
                    "run_failure",
                    {
                        "status": "FAILED",
                        "terminal_status": "FAILED",
                        "error": failure,
                        "tool_calls": self._tool_call_count,
                        "successful_tool_calls": self._successful_tool_calls,
                    },
                )
            if not self._model_submission:
                self._model_submission = self._failure_answer(exc)
        try:
            metadata, final_answer = self._write_artifacts(status=status, failure=failure)
        finally:
            self._secrets = ()
        result = RunResult(
            status=status,
            run_id=self.run_id,
            final_answer=final_answer,
            metadata=metadata,
            artifacts=ARTIFACT_NAMES,
        )
        if self._caught is not None and (
            raise_on_failure or isinstance(self._caught, (KeyboardInterrupt, SystemExit))
        ):
            raise self._caught
        return result


def run_agent(**kwargs: Any) -> RunResult:
    raise_on_failure = bool(kwargs.pop("raise_on_failure", False))
    return AgentRunner(**kwargs).run(raise_on_failure=raise_on_failure)


__all__ = [
    "ARTIFACT_NAMES",
    "AgentRunner",
    "ExactModelMismatchError",
    "ScriptedTransport",
    "ToolExecutionContext",
    "run_agent",
]

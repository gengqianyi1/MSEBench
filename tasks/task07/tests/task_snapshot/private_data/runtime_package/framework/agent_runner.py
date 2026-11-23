"""Auditable OpenAI-compatible tool-calling runner for the QED benchmark.

The provider SDK is imported only when a real transport is constructed.  Tests
can consequently exercise the exact agent loop with :class:`ScriptedTransport`
without installing ``openai`` or making network requests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any

from .schemas import (
    BudgetConfig,
    BudgetExceededError,
    ConfigurationError,
    FinalCandidate,
    MissingCredentialError,
    ModelConfig,
    OutputParseError,
    ProviderError,
    RunResult,
)


PUBLIC_TOOL_NAMES = (
    "validate_smiles",
    "canonicalize_smiles",
    "calculate_qed",
    "calculate_properties",
    "calculate_similarity",
    "check_scaffold",
    "check_substructure",
    "evaluate_candidate",
)


def _function_tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


_SMILES = {"type": "string", "minLength": 1, "maxLength": 512}
_SMARTS = {"type": "string", "minLength": 1, "maxLength": 512}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _function_tool(
        "validate_smiles",
        "Validate parsing, connectedness, allowed elements, and formal charge.",
        {"smiles": _SMILES},
        ["smiles"],
    ),
    _function_tool(
        "canonicalize_smiles",
        "Return the canonical isomeric SMILES for a molecule.",
        {"smiles": _SMILES},
        ["smiles"],
    ),
    _function_tool(
        "calculate_qed",
        "Calculate RDKit QED for a molecule.",
        {"smiles": _SMILES},
        ["smiles"],
    ),
    _function_tool(
        "calculate_properties",
        "Calculate the public physicochemical property panel.",
        {"smiles": _SMILES},
        ["smiles"],
    ),
    _function_tool(
        "calculate_similarity",
        "Calculate Morgan-fingerprint Tanimoto similarity between two molecules.",
        {"smiles_a": _SMILES, "smiles_b": _SMILES},
        ["smiles_a", "smiles_b"],
    ),
    _function_tool(
        "check_scaffold",
        "Check the public required scaffold. The SMARTS defaults to the task scaffold.",
        {"smiles": _SMILES, "scaffold_smarts": _SMARTS},
        ["smiles"],
    ),
    _function_tool(
        "check_substructure",
        "Check whether a molecule contains a specified SMARTS substructure.",
        {"smiles": _SMILES, "smarts": _SMARTS},
        ["smiles", "smarts"],
    ),
    _function_tool(
        "evaluate_candidate",
        "Evaluate all public properties and hard constraints for one candidate.",
        {"smiles": _SMILES},
        ["smiles"],
    ),
]

_TOOL_SCHEMAS = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in TOOL_DEFINITIONS
}

ARTIFACT_NAMES = (
    "trajectory.jsonl",
    "dialogue.md",
    "metadata.json",
    "final_answer.json",
    "evaluation.json",
)

CANONICAL_DEPENDENCY_IDENTIFIERS = {
    "transport": "framework.agent_runner.OpenAIChatTransport",
    "tool_registry": "tools.rdkit_tools.TOOL_REGISTRY",
    "evaluator": "evaluator.evaluate.evaluate_submission",
}

SILICONFLOW_TRANSPORT_IDENTIFIER = (
    "framework.siliconflow_adapter.SiliconFlowChatTransport"
)

# Match only syntactically absolute machine paths.  The POSIX expression
# deliberately requires a non-URL/non-SMILES boundary before '/', so
# ``https://...``, ``C/C=C/C``, and relative report paths are untouched.
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])[^\s\"'<>|,;)\]}]*"
)
_UNC_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\\A-Za-z0-9])\\\\[A-Za-z0-9._$-]+[\\/][^\s\"'<>|,;)\]}]+"
)
_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:/>])/(?!/)[^\s\"'<>|,;)\]}]+"
)


def _to_plain(value: Any) -> Any:
    """Recursively convert SDK/Pydantic objects to JSON-safe Python values."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floating-point values are forbidden in run artifacts")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_plain(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _to_plain(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(key): _to_plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _to_plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_time(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(
        _to_plain(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - package environment includes PyYAML
        raise ConfigurationError("PyYAML is required to load benchmark configuration") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration {path.name!r} must contain an object")
    return value


def _mapping_or_yaml(value: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value)
    return _load_yaml(path), path


class OpenAIChatTransport:
    """Thin adapter around the OpenAI SDK's Chat Completions endpoint.

    ``openai`` is intentionally imported here rather than at module import time.
    """

    requires_api_key = True
    is_mock = False

    def __init__(self, config: ModelConfig, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError(
                "the optional 'openai' package is required for real model runs"
            ) from exc
        self._config = config
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,  # retries are captured explicitly by AgentRunner
        )

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self._config.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "timeout": min(timeout_seconds, self._config.timeout_seconds),
            "stream": False,
        }
        for name, value in self._config.provider_options.items():
            if name == "thinking":
                # The OpenAI SDK exposes provider-specific body extensions via
                # extra_body.  Config validation permits only this exact nested
                # option, never caller-controlled arbitrary request fields.
                request["extra_body"] = {"thinking": _to_plain(value)}
            else:
                request[name] = _to_plain(value)
        return self._client.chat.completions.create(**request)


# Keep an immutable reference to the implementation loaded from this source
# file.  The public name remains patchable for tests and integrations, but a
# patched default must never inherit canonical/REAL_API attestation merely
# because the caller did not inject it through the constructor.
_CANONICAL_OPENAI_CHAT_TRANSPORT = OpenAIChatTransport


class ScriptedTransport:
    """Deterministic offline Chat Completions fake used by smoke tests.

    Each item is either an OpenAI-shaped response, an assistant-message mapping,
    or an exception to raise.  Metadata always identifies this transport as a
    mock; its artifacts can never be mistaken for evidence of a real API run.
    """

    requires_api_key = False
    is_mock = True

    def __init__(self, responses: Iterable[Any]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> Any:
        self.calls.append(
            {
                "messages": _to_plain(messages),
                "tools": _to_plain(tools),
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self._responses:
            raise RuntimeError("scripted transport has no response remaining")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Mapping) and "choices" not in item:
            return {
                "id": f"mock-response-{len(self.calls)}",
                "created": 0,
                "model": "scripted-offline-model",
                "system_fingerprint": "mock",
                "choices": [{"index": 0, "message": dict(item), "finish_reason": None}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        return item


class AgentRunner:
    """Run and audit one bounded model/tool/evaluator interaction."""

    def __init__(
        self,
        *,
        task_config: Mapping[str, Any] | str | Path,
        model_config: Mapping[str, Any] | str | Path,
        output_dir: str | Path,
        prompt: str | Path,
        transport: Any | None = None,
        tool_registry: Mapping[str, Callable[..., Any]] | None = None,
        evaluator: Callable[..., Any] | None = None,
        now_fn: Callable[[], Any] = _utc_now,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        run_label: str | None = None,
        model_config_source: str | Path | None = None,
        canonical_dependency_provenance: Mapping[str, str] | None = None,
        allow_overwrite: bool = False,
    ):
        self.task_config, self.task_config_path = _mapping_or_yaml(task_config)
        raw_model, self.model_config_path = _mapping_or_yaml(model_config)
        self.model_config = ModelConfig.from_mapping(raw_model)
        self.budget = BudgetConfig.from_task_config(self.task_config)
        self._configure_final_process_guard()
        self.output_dir = Path(output_dir)
        self._canonical_dependency_identifiers = dict(
            CANONICAL_DEPENDENCY_IDENTIFIERS
        )
        if self.model_config.provider == "siliconflow":
            self._canonical_dependency_identifiers["transport"] = (
                SILICONFLOW_TRANSPORT_IDENTIFIER
            )

        candidate_prompt_path: Path | None = None
        if isinstance(prompt, Path):
            candidate_prompt_path = prompt
        elif isinstance(prompt, str) and "\n" not in prompt and "\r" not in prompt:
            try:
                possible_path = Path(prompt)
                if possible_path.exists():
                    candidate_prompt_path = possible_path
            except OSError:
                candidate_prompt_path = None
        if candidate_prompt_path is not None and candidate_prompt_path.exists():
            self.prompt = candidate_prompt_path.read_text(encoding="utf-8")
            self.prompt_path: Path | None = candidate_prompt_path
        else:
            self.prompt = str(prompt)
            self.prompt_path = None
        if not self.prompt.strip():
            raise ConfigurationError("agent prompt must not be empty")

        self._injected_transport = transport
        self._injected_tools = dict(tool_registry) if tool_registry is not None else None
        self._evaluator = evaluator
        self._now = now_fn
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn
        self.run_label = run_label
        self.model_config_source_path = (
            Path(model_config_source) if model_config_source is not None else None
        )
        raw_provenance = dict(canonical_dependency_provenance or {})
        if any(not isinstance(name, str) for name in raw_provenance):
            raise ConfigurationError("canonical dependency provenance keys must be strings")
        unknown_provenance = sorted(
            set(raw_provenance) - set(self._canonical_dependency_identifiers)
        )
        if unknown_provenance:
            raise ConfigurationError(
                "unknown canonical dependency provenance field(s): "
                + ", ".join(unknown_provenance)
            )
        for name, identifier in raw_provenance.items():
            expected_identifier = self._canonical_dependency_identifiers[name]
            if identifier != expected_identifier:
                raise ConfigurationError(
                    f"canonical dependency provenance for {name} must be exactly "
                    f"{expected_identifier!r}"
                )
        self._declared_canonical_provenance = raw_provenance
        # Retained only as a source-compatible constructor argument.  Existing
        # artifacts are never overwritten, even when older callers pass True.
        self.allow_overwrite = bool(allow_overwrite)

        self._messages: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._provider_responses: list[dict[str, Any]] = []
        self._actual_model_id: str | None = None
        self._observed_model_ids: list[str] = []
        self._provider_models_verified = True
        self._provider_attempts = 0
        self._tool_call_count = 0
        self._successful_public_tool_call_count = 0
        self._tool_failures = 0
        self._rejected_tool_calls = 0
        self._turn_count = 0
        self._budget_exceeded = False
        self._unique_candidates: list[str] = []
        self._unique_candidate_set: set[str] = set()
        self._attempted_candidates: list[str] = []
        self._attempted_candidate_set: set[str] = set()
        self._candidate_history: list[dict[str, str]] = []
        self._candidate_first_turn: dict[str, int] = {}
        self._started_monotonic = 0.0
        self._secret_for_redaction = ""
        self._tools: dict[str, Callable[..., Any]] = {}
        self._resolved_evaluator: Callable[..., Any] | None = None
        self._dependency_provenance: dict[str, dict[str, Any]] = {}
        self._dependency_mode = "UNVERIFIED"
        self._canonical_dependencies = False
        self._integrity_start: dict[str, Any] | None = None
        self._integrity_end: dict[str, Any] | None = None
        self._integrity_verified = False
        self._integrity_changes: list[str] = []
        self._integrity_error: str | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        return _to_plain(self._messages)

    def _configure_final_process_guard(self) -> None:
        raw = self.task_config.get("budget")
        assert isinstance(raw, Mapping)  # validated by BudgetConfig
        required = {
            "min_tool_calls_for_eligibility",
            "min_unique_evaluated_candidates_for_eligibility",
            "final_candidate_must_be_tool_evaluated",
        }
        present = required.intersection(raw)
        self._final_process_guard_enabled = bool(present)
        self._minimum_successful_public_tool_calls = 0
        self._minimum_unique_valid_candidates = 0
        self._minimum_candidate_feedback_turns = 0
        self._final_candidate_must_be_evaluated = False
        if not present:
            return
        missing = sorted(required - present)
        if missing:
            raise ConfigurationError(
                "process eligibility budget is incomplete; missing: "
                + ", ".join(missing)
            )
        minimum_calls = raw["min_tool_calls_for_eligibility"]
        minimum_candidates = raw[
            "min_unique_evaluated_candidates_for_eligibility"
        ]
        minimum_feedback_turns = raw.get(
            "min_candidate_feedback_turns_for_eligibility", 0
        )
        final_must_be_evaluated = raw[
            "final_candidate_must_be_tool_evaluated"
        ]
        for label, value in (
            ("min_tool_calls_for_eligibility", minimum_calls),
            (
                "min_unique_evaluated_candidates_for_eligibility",
                minimum_candidates,
            ),
            (
                "min_candidate_feedback_turns_for_eligibility",
                minimum_feedback_turns,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"budget.{label} must be a non-negative integer")
        if not isinstance(final_must_be_evaluated, bool):
            raise ConfigurationError(
                "budget.final_candidate_must_be_tool_evaluated must be a boolean"
            )
        self._minimum_successful_public_tool_calls = minimum_calls
        self._minimum_unique_valid_candidates = minimum_candidates
        self._minimum_candidate_feedback_turns = minimum_feedback_turns
        self._final_candidate_must_be_evaluated = final_must_be_evaluated
        if self._minimum_successful_public_tool_calls > self.budget.max_tool_calls:
            raise ConfigurationError(
                "process eligibility requires more successful tool calls than "
                "budget.max_tool_calls permits"
            )
        if (
            self._minimum_unique_valid_candidates
            > self.budget.max_unique_evaluated_candidates
        ):
            raise ConfigurationError(
                "process eligibility requires more unique evaluated candidates than "
                "budget.max_unique_evaluated_candidates permits"
            )

    @staticmethod
    def _object_identifier(value: Any) -> str:
        target = value if callable(value) else type(value)
        module = getattr(target, "__module__", type(value).__module__)
        qualname = getattr(target, "__qualname__", type(value).__qualname__)
        return f"{module}.{qualname}"

    def _default_transport_class(self) -> type[Any]:
        if self.model_config.provider == "siliconflow":
            from .siliconflow_adapter import SiliconFlowChatTransport

            return SiliconFlowChatTransport
        return OpenAIChatTransport

    def _canonical_default_transport_class(self) -> type[Any]:
        if self.model_config.provider == "siliconflow":
            module = importlib.import_module("framework.siliconflow_adapter")
            return getattr(module, "_CANONICAL_SILICONFLOW_CHAT_TRANSPORT")
        return _CANONICAL_OPENAI_CHAT_TRANSPORT

    @staticmethod
    def _source_is_expected(value: Any, relative_path: str) -> bool:
        expected = (Path(__file__).resolve().parents[1] / relative_path).resolve()
        try:
            source = inspect.getsourcefile(value)
            return source is not None and Path(source).resolve() == expected
        except (OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _module_source_is_expected(module: Any, relative_path: str) -> bool:
        expected = (Path(__file__).resolve().parents[1] / relative_path).resolve()
        try:
            source = getattr(module, "__file__", None)
            return source is not None and Path(source).resolve() == expected
        except (OSError, TypeError, ValueError):
            return False

    def _default_transport_class_is_canonical(self) -> bool:
        try:
            current = self._default_transport_class()
            canonical = self._canonical_default_transport_class()
            relative = (
                "framework/siliconflow_adapter.py"
                if self.model_config.provider == "siliconflow"
                else "framework/agent_runner.py"
            )
            return (
                current is canonical
                and self._object_identifier(current)
                == self._canonical_dependency_identifiers["transport"]
                and self._source_is_expected(current, relative)
            )
        except Exception:
            return False

    def _resolve_default_tool_registry(
        self,
    ) -> tuple[dict[str, Callable[..., Any]], bool]:
        module = importlib.import_module("tools.rdkit_tools")
        raw_registry = getattr(module, "TOOL_REGISTRY", None)
        if not isinstance(raw_registry, Mapping):
            return {}, False
        registry = dict(raw_registry)
        canonical = (
            self._module_source_is_expected(module, "tools/rdkit_tools.py")
            and set(registry) == set(PUBLIC_TOOL_NAMES)
            and all(
                callable(registry.get(name))
                and registry[name] is getattr(module, name, None)
                and self._object_identifier(registry[name])
                == f"tools.rdkit_tools.{name}"
                and self._source_is_expected(registry[name], "tools/rdkit_tools.py")
                for name in PUBLIC_TOOL_NAMES
            )
        )
        return registry, canonical

    def _resolve_default_evaluator(
        self,
    ) -> tuple[Callable[..., Any] | None, bool]:
        module = importlib.import_module("evaluator.evaluate")
        evaluator = getattr(module, "evaluate_submission", None)
        canonical = (
            callable(evaluator)
            and self._module_source_is_expected(module, "evaluator/evaluate.py")
            and self._object_identifier(evaluator)
            == self._canonical_dependency_identifiers["evaluator"]
            and self._source_is_expected(evaluator, "evaluator/evaluate.py")
        )
        return evaluator if callable(evaluator) else None, canonical

    def _injected_dependency_is_canonical(self, name: str) -> bool:
        if name not in self._declared_canonical_provenance:
            return False
        try:
            if name == "transport":
                expected_type = self._canonical_default_transport_class()
                relative = (
                    "framework/siliconflow_adapter.py"
                    if self.model_config.provider == "siliconflow"
                    else "framework/agent_runner.py"
                )
                return (
                    type(self._injected_transport) is expected_type
                    and self._source_is_expected(expected_type, relative)
                    and getattr(self._injected_transport, "_config", None)
                    == self.model_config
                    and getattr(self._injected_transport, "is_mock", True) is False
                    and (
                        self.model_config.provider != "siliconflow"
                        or getattr(
                            self._injected_transport,
                            "uses_canonical_client",
                            False,
                        )
                        is True
                    )
                )
            if name == "tool_registry":
                registry, canonical = self._resolve_default_tool_registry()
                if (
                    not canonical
                    or self._injected_tools is None
                    or set(self._injected_tools) != set(PUBLIC_TOOL_NAMES)
                ):
                    return False
                return all(
                    self._injected_tools[tool_name] is registry[tool_name]
                    for tool_name in PUBLIC_TOOL_NAMES
                )
            if name == "evaluator":
                evaluator, canonical = self._resolve_default_evaluator()
                return canonical and self._evaluator is evaluator
        except Exception:
            return False
        return False

    def _build_dependency_provenance(self) -> None:
        injected = {
            "transport": self._injected_transport,
            "tool_registry": self._injected_tools,
            "evaluator": self._evaluator,
        }
        report: dict[str, dict[str, Any]] = {}
        for name, value in injected.items():
            expected = self._canonical_dependency_identifiers[name]
            if value is None:
                if name == "transport":
                    verified = self._default_transport_class_is_canonical()
                    try:
                        actual_identifier = self._object_identifier(
                            self._default_transport_class()
                        )
                    except Exception:
                        actual_identifier = "unavailable_default_transport"
                    report[name] = {
                        "mode": "default",
                        # Transport attestation is incomplete until the exact
                        # instance (and, for SiliconFlow, default SDK client)
                        # has been constructed and inspected.
                        "canonical": False,
                        "identifier": expected if verified else actual_identifier,
                        "attestation": (
                            "class_source_verified_instance_pending"
                            if verified
                            else "class_source_unverified"
                        ),
                    }
                    continue
                if name == "tool_registry":
                    try:
                        _registry, verified = self._resolve_default_tool_registry()
                    except Exception:
                        verified = False
                else:
                    try:
                        _evaluator, verified = self._resolve_default_evaluator()
                    except Exception:
                        verified = False
                report[name] = {
                    "mode": "default",
                    "canonical": verified,
                    "identifier": expected,
                    "attestation": (
                        "identity_source_verified" if verified else "identity_source_unverified"
                    ),
                }
                continue
            if self._injected_dependency_is_canonical(name):
                report[name] = {
                    "mode": "injected_canonical",
                    "canonical": True,
                    "identifier": expected,
                    "attestation": "identity_verified",
                }
                continue
            identifier = (
                "injected_mapping"
                if name == "tool_registry"
                else self._object_identifier(value)
            )
            report[name] = {
                "mode": "injected",
                "canonical": False,
                "identifier": identifier,
                "attestation": "unverified",
            }
        self._dependency_provenance = report
        self._refresh_dependency_state()

    def _refresh_dependency_state(self) -> None:
        self._canonical_dependencies = all(
            entry["canonical"] is True
            for entry in self._dependency_provenance.values()
        )
        self._dependency_mode = (
            "CANONICAL" if self._canonical_dependencies else "UNVERIFIED"
        )

    def _execution_mode(self) -> str:
        if (
            self._injected_transport is not None
            and not self._dependency_provenance.get("transport", {}).get("canonical")
        ):
            return "MOCK"
        if self.model_config.is_trusted_endpoint and self._canonical_dependencies:
            return "REAL_API"
        return "UNVERIFIED"

    def _finalize_transport_provenance(self, transport: Any) -> None:
        if self._injected_transport is not None:
            self._refresh_dependency_state()
            return
        try:
            expected_type = self._canonical_default_transport_class()
            relative = (
                "framework/siliconflow_adapter.py"
                if self.model_config.provider == "siliconflow"
                else "framework/agent_runner.py"
            )
            canonical = (
                self._default_transport_class_is_canonical()
                and type(transport) is expected_type
                and self._source_is_expected(type(transport), relative)
                and getattr(transport, "_config", None) == self.model_config
                and getattr(transport, "is_mock", True) is False
                and (
                    self.model_config.provider != "siliconflow"
                    or getattr(transport, "uses_canonical_client", False) is True
                )
            )
        except Exception:
            canonical = False
        entry = self._dependency_provenance["transport"]
        entry["canonical"] = canonical
        entry["attestation"] = (
            "class_source_instance_client_verified"
            if canonical and self.model_config.provider == "siliconflow"
            else "class_source_instance_verified"
            if canonical
            else "instance_unverified"
        )
        if not canonical:
            entry["identifier"] = self._object_identifier(transport)
        self._refresh_dependency_state()

    def validate_credentials(self) -> str | None:
        """Read a real provider credential exclusively from its configured env var."""

        transport = self._injected_transport
        requires_key = transport is None or bool(getattr(transport, "requires_api_key", False))
        if not requires_key:
            return None
        api_key = os.environ.get(self.model_config.api_key_env, "")
        if not api_key.strip():
            raise MissingCredentialError(
                f"required environment variable {self.model_config.api_key_env!r} is not set"
            )
        return api_key

    def _timestamp(self) -> str:
        return _iso_time(self._now())

    def _elapsed(self) -> float:
        return max(0.0, self._monotonic() - self._started_monotonic)

    def _remaining(self) -> float:
        return self.budget.timeout_seconds - self._elapsed()

    def _check_timeout(self) -> None:
        if self._remaining() <= 0:
            self._budget_exceeded = True
            raise BudgetExceededError(
                f"run exceeded timeout_seconds={self.budget.timeout_seconds:g}"
            )

    def _redact(self, value: Any) -> Any:
        plain = _to_plain(value)
        secret = self._secret_for_redaction
        project_root = str(Path(__file__).resolve().parents[1])
        try:
            run_dir = str(self.output_dir.resolve())
        except OSError:
            run_dir = str(self.output_dir)

        def visit(item: Any) -> Any:
            if isinstance(item, str):
                if secret:
                    item = item.replace(secret, "[REDACTED]")
                for original, replacement in (
                    (project_root, "<PROJECT_ROOT>"),
                    (project_root.replace("\\", "/"), "<PROJECT_ROOT>"),
                    (run_dir, "<RUN_DIR>"),
                    (run_dir.replace("\\", "/"), "<RUN_DIR>"),
                ):
                    if original:
                        item = item.replace(original, replacement)
                item = _UNC_ABSOLUTE_PATH_RE.sub("<LOCAL_PATH>", item)
                item = _WINDOWS_ABSOLUTE_PATH_RE.sub("<LOCAL_PATH>", item)
                item = _POSIX_ABSOLUTE_PATH_RE.sub("<LOCAL_PATH>", item)
                return item
            if isinstance(item, dict):
                return {key: visit(val) for key, val in item.items()}
            if isinstance(item, list):
                return [visit(val) for val in item]
            return item

        return visit(plain)

    def _event(self, event_type: str, **data: Any) -> None:
        self._events.append(
            {
                "sequence": len(self._events) + 1,
                "timestamp": self._timestamp(),
                "event": event_type,
                **self._redact(data),
            }
        )

    def _append_message(self, message: Mapping[str, Any], *, source: str) -> None:
        plain = self._redact(dict(message))
        self._messages.append(plain)
        self._event("message", source=source, message=plain)

    def _initial_messages(self) -> list[dict[str, Any]]:
        task_id = self._task_id()
        user_payload = {
            "task_id": task_id,
            "start_smiles": (self.task_config.get("task") or {}).get("start_smiles"),
            "objective": (self.task_config.get("task") or {}).get("objective"),
            "constraints": self.task_config.get("constraints", {}),
            "budget": self.task_config.get("budget", {}),
            "required_final_json": {
                "final_candidate": {
                    "name": "short_identifier",
                    "smiles": "one_connected_molecule",
                    "rationale": "brief explanation",
                }
            },
        }
        return [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": (
                    "Run the constrained molecular-design task below. Use the provided "
                    "RDKit tools for measured feedback, then return only the required strict JSON.\n\n"
                    + json.dumps(
                        user_payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                ),
            },
        ]

    def _task_id(self) -> str:
        task = self.task_config.get("task")
        if (
            isinstance(task, Mapping)
            and isinstance(task.get("id"), str)
            and task["id"].strip()
        ):
            return task["id"]
        task_id = self.task_config.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            return task_id
        raise ConfigurationError("task configuration must define task.id")

    def _task_id_or_none(self) -> str | None:
        try:
            return self._task_id()
        except ConfigurationError:
            return None

    def _load_tools(self) -> dict[str, Callable[..., Any]]:
        if self._injected_tools is not None:
            registry = dict(self._injected_tools)
            canonical = self._injected_dependency_is_canonical("tool_registry")
            entry = self._dependency_provenance["tool_registry"]
            entry["canonical"] = canonical
            entry["attestation"] = (
                "identity_source_verified" if canonical else "unverified"
            )
        else:
            registry, canonical = self._resolve_default_tool_registry()
            entry = self._dependency_provenance["tool_registry"]
            entry["canonical"] = canonical
            entry["attestation"] = (
                "identity_source_verified"
                if canonical
                else "identity_source_unverified"
            )
        self._refresh_dependency_state()
        missing = [name for name in PUBLIC_TOOL_NAMES if not callable(registry.get(name))]
        extra = sorted(set(registry) - set(PUBLIC_TOOL_NAMES))
        if missing or extra:
            raise ConfigurationError(
                "RDKit TOOL_REGISTRY must contain exactly the public functions; "
                f"missing={missing}, extra={extra}"
            )
        return {name: registry[name] for name in PUBLIC_TOOL_NAMES}

    def _load_evaluator(self) -> Callable[..., Any]:
        if self._evaluator is not None:
            evaluator = self._evaluator
            canonical = self._injected_dependency_is_canonical("evaluator")
            entry = self._dependency_provenance["evaluator"]
            entry["canonical"] = canonical
            entry["attestation"] = (
                "identity_source_verified" if canonical else "unverified"
            )
        else:
            evaluator, canonical = self._resolve_default_evaluator()
            entry = self._dependency_provenance["evaluator"]
            entry["canonical"] = canonical
            entry["attestation"] = (
                "identity_source_verified"
                if canonical
                else "identity_source_unverified"
            )
        self._refresh_dependency_state()
        if not callable(evaluator):
            raise ConfigurationError("canonical evaluator entry point is unavailable")
        return evaluator

    def _provider_call(self, transport: Any, turn: int) -> Any:
        last_error: BaseException | None = None
        attempts = self.model_config.max_retries + 1
        for attempt in range(1, attempts + 1):
            self._check_timeout()
            self._provider_attempts += 1
            started = self._monotonic()
            timeout = min(self._remaining(), self.model_config.timeout_seconds)
            self._event(
                "provider_request",
                turn=turn,
                attempt=attempt,
                model=self.model_config.model,
                message_count=len(self._messages),
                messages=self._messages,
                tools=TOOL_DEFINITIONS,
                timeout_seconds=timeout,
            )
            try:
                response = transport.complete(
                    messages=self._messages,
                    tools=TOOL_DEFINITIONS,
                    timeout_seconds=timeout,
                )
                if self._elapsed() > self.budget.timeout_seconds:
                    self._budget_exceeded = True
                    raise BudgetExceededError(
                        f"provider response arrived after timeout_seconds={self.budget.timeout_seconds:g}"
                    )
                plain = self._redact(_to_plain(response))
                self._event(
                    "provider_response",
                    turn=turn,
                    attempt=attempt,
                    duration_seconds=max(0.0, self._monotonic() - started),
                    response=plain,
                )
                return plain
            except BudgetExceededError:
                raise
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                last_error = exc
                self._event(
                    "provider_failure",
                    turn=turn,
                    attempt=attempt,
                    duration_seconds=max(0.0, self._monotonic() - started),
                    will_retry=attempt < attempts,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    traceback=traceback.format_exc(),
                )
                if attempt < attempts:
                    delay = self.model_config.retry_backoff_seconds * (2 ** (attempt - 1))
                    delay = min(delay, max(0.0, self._remaining()))
                    if delay > 0:
                        self._sleep(delay)
        assert last_error is not None
        raise ProviderError(
            f"provider failed after {attempts} attempt(s): {type(last_error).__name__}: {last_error}"
        ) from last_error

    def _extract_assistant(self, response: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        returned_model = response.get("model")
        metadata = {
            "id": response.get("id"),
            "created": response.get("created"),
            "model": returned_model,
            "system_fingerprint": response.get("system_fingerprint"),
            "service_tier": response.get("service_tier"),
            "usage": response.get("usage"),
            "finish_reason": None,
        }
        for cost_key in ("cost", "total_cost"):
            if cost_key in response:
                metadata[cost_key] = response.get(cost_key)

        stored_returned_model = (
            self._redact(returned_model)
            if isinstance(returned_model, str)
            else None
        )
        if isinstance(stored_returned_model, str):
            self._observed_model_ids.append(stored_returned_model)
            if self._actual_model_id is None:
                self._actual_model_id = stored_returned_model
            elif stored_returned_model != self._actual_model_id:
                self._provider_models_verified = False
                self._provider_responses.append(self._redact(metadata))
                raise ProviderError(
                    "provider model ID changed during the run: "
                    f"first={self._actual_model_id!r}, returned={returned_model!r}"
                )
        if not self.model_config.response_model_compatible(returned_model):
            self._provider_models_verified = False
            self._provider_responses.append(self._redact(metadata))
            raise ProviderError(
                "provider returned an incompatible model ID: "
                f"configured={self.model_config.model!r}, returned={returned_model!r}, "
                f"provider={self.model_config.provider!r}"
            )
        assert isinstance(returned_model, str)

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("provider response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ProviderError("provider response choice is not an object")
        raw_message = choice.get("message")
        if not isinstance(raw_message, Mapping):
            raise ProviderError("provider response choice has no assistant message")
        message = dict(_to_plain(raw_message))
        message["role"] = "assistant"
        if "content" not in message:
            message["content"] = None
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ProviderError("assistant tool_calls must be a list")

        metadata["finish_reason"] = choice.get("finish_reason")
        safe_metadata = self._redact(metadata)
        self._provider_responses.append(safe_metadata)
        return message, safe_metadata

    def _validate_tool_arguments(self, name: str, value: Any) -> dict[str, Any]:
        if name not in _TOOL_SCHEMAS:
            raise ValueError(f"unknown tool {name!r}")
        if not isinstance(value, dict):
            raise ValueError("tool arguments must decode to a JSON object")
        schema = _TOOL_SCHEMAS[name]
        required = set(schema["required"])
        allowed = set(schema["properties"])
        missing = sorted(required - set(value))
        extra = sorted(set(value) - allowed)
        if missing or extra:
            raise ValueError(f"invalid arguments: missing={missing}, extra={extra}")
        for key, item in value.items():
            expected = schema["properties"][key].get("type")
            if expected == "string" and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f"argument {key!r} must be a non-empty string")
            maximum = schema["properties"][key].get("maxLength")
            if isinstance(item, str) and maximum is not None and len(item) > maximum:
                raise ValueError(f"argument {key!r} exceeds {maximum} characters")
        return dict(value)

    def _task_defaults_for_tool(self, name: str, function: Callable[..., Any]) -> dict[str, Any]:
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            parameters = {}
        defaults: dict[str, Any] = {}
        task = self.task_config.get("task") or {}
        constraints = self.task_config.get("constraints") or {}
        similarity = constraints.get("similarity") or {}
        scaffold = constraints.get("scaffold") or {}
        start_smiles = task.get("start_smiles") or self.task_config.get("start_smiles")

        if "task_config" in parameters:
            defaults["task_config"] = self.task_config
        if "config" in parameters:
            defaults["config"] = self.task_config
        if "constraints" in parameters:
            defaults["constraints"] = constraints
        if "start_smiles" in parameters:
            defaults["start_smiles"] = start_smiles
        if "radius" in parameters and name == "calculate_similarity":
            defaults["radius"] = int(similarity.get("radius", 2))
        if "n_bits" in parameters and name == "calculate_similarity":
            defaults["n_bits"] = int(similarity.get("n_bits", 2048))
        if "scaffold_smarts" in parameters and name == "check_scaffold":
            defaults["scaffold_smarts"] = scaffold.get("smarts")
        return defaults

    def _invoke_tool_function(self, name: str, arguments: Mapping[str, Any]) -> Any:
        function = self._tools[name]
        kwargs = self._task_defaults_for_tool(name, function)
        kwargs.update(arguments)
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if parameters and not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return function(**kwargs)

    @staticmethod
    def _canonical_from_result(result: Any) -> str | None:
        if isinstance(result, str) and result.strip():
            return result.strip()
        if isinstance(result, Mapping):
            for key in ("canonical_smiles", "canonical", "smiles"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            nested = result.get("result")
            if nested is not result:
                return AgentRunner._canonical_from_result(nested)
        return None

    def _canonicalize_for_accounting(self, smiles: str) -> str:
        raw = smiles.strip()
        try:
            result = self._invoke_tool_function("canonicalize_smiles", {"smiles": raw})
            return self._canonical_from_result(_to_plain(result)) or raw
        except Exception:
            return raw

    def _tool_error_message(
        self, call_id: str, name: str, error_type: str, message: str
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(
                {"ok": False, "error": {"type": error_type, "message": message}},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    def _record_evaluated_candidate(self, key: str) -> None:
        if key in self._unique_candidate_set:
            return
        self._unique_candidate_set.add(key)
        self._unique_candidates.append(key)
        self._candidate_first_turn[key] = self._turn_count
        index = len(self._unique_candidates)
        self._candidate_history.append(
            {
                "name": f"eval_{index:02d}",
                "smiles": key,
                "rationale": "Tool-evaluated candidate.",
            }
        )

    def _record_attempted_candidate(self, key: str) -> None:
        if key in self._attempted_candidate_set:
            return
        self._attempted_candidate_set.add(key)
        self._attempted_candidates.append(key)

    def _execute_tool_call(self, call: Any) -> None:
        started_at = self._timestamp()
        started = self._monotonic()
        plain_call = _to_plain(call)
        call_id = ""
        name = ""
        raw_arguments: Any = None
        if isinstance(plain_call, Mapping):
            call_id = str(plain_call.get("id") or f"missing-call-id-{self._tool_call_count + 1}")
            function = plain_call.get("function")
            if isinstance(function, Mapping):
                name = str(function.get("name") or "")
                raw_arguments = function.get("arguments")

        if self._tool_call_count >= self.budget.max_tool_calls:
            self._budget_exceeded = True
            self._rejected_tool_calls += 1
            message = self._tool_error_message(
                call_id,
                name,
                "ToolBudgetExceeded",
                f"max_tool_calls={self.budget.max_tool_calls} has been reached",
            )
            self._event(
                "tool_call",
                turn=self._turn_count,
                call_id=call_id,
                tool=name,
                raw_arguments=raw_arguments,
                status="REJECTED",
                error={"type": "ToolBudgetExceeded", "message": "tool-call budget exhausted"},
                started_at=started_at,
                completed_at=self._timestamp(),
                duration_seconds=max(0.0, self._monotonic() - started),
            )
            self._append_message(message, source="runner_budget_guard")
            return

        self._tool_call_count += 1
        arguments: dict[str, Any] | None = None
        try:
            if not name:
                raise ValueError("tool call is missing function.name")
            if isinstance(raw_arguments, str):
                try:
                    decoded = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSON tool arguments: {exc.msg}") from exc
            elif isinstance(raw_arguments, Mapping):
                decoded = dict(raw_arguments)
            else:
                raise ValueError("tool function.arguments must be a JSON string or object")
            arguments = self._validate_tool_arguments(name, decoded)

            reserved_candidate: str | None = None
            if name == "evaluate_candidate":
                reserved_candidate = self._canonicalize_for_accounting(arguments["smiles"])
                if (
                    reserved_candidate not in self._attempted_candidate_set
                    and len(self._attempted_candidates)
                    >= self.budget.max_unique_evaluated_candidates
                ):
                    self._budget_exceeded = True
                    self._rejected_tool_calls += 1
                    message_text = (
                        "max_unique_evaluated_candidates="
                        f"{self.budget.max_unique_evaluated_candidates} has been reached"
                    )
                    message = self._tool_error_message(
                        call_id, name, "CandidateBudgetExceeded", message_text
                    )
                    self._event(
                        "tool_call",
                        turn=self._turn_count,
                        call_id=call_id,
                        tool=name,
                        raw_arguments=raw_arguments,
                        arguments=arguments,
                        status="REJECTED",
                        error={"type": "CandidateBudgetExceeded", "message": message_text},
                        started_at=started_at,
                        completed_at=self._timestamp(),
                        duration_seconds=max(0.0, self._monotonic() - started),
                    )
                    self._append_message(message, source="runner_budget_guard")
                    return
                # A syntactically accepted evaluate_candidate call consumes the
                # distinct-attempt budget even when RDKit reports ok/valid false
                # or the tool later raises.  This prevents an infinite stream of
                # distinct invalid SMILES from bypassing the candidate budget.
                self._record_attempted_candidate(reserved_candidate)

            result = _to_plain(self._invoke_tool_function(name, arguments))
            result_ok = isinstance(result, Mapping) and result.get("ok") is True
            if result_ok:
                self._successful_public_tool_call_count += 1
            recorded_candidate: str | None = None
            if name == "evaluate_candidate":
                if (
                    isinstance(result, Mapping)
                    and result.get("ok") is True
                    and result.get("valid") is True
                    and isinstance(result.get("canonical_smiles"), str)
                    and result["canonical_smiles"].strip()
                ):
                    recorded_candidate = result["canonical_smiles"].strip()
                    self._record_evaluated_candidate(recorded_candidate)
            message = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(
                    {"ok": True, "result": result},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            self._event(
                "tool_call",
                turn=self._turn_count,
                call_id=call_id,
                tool=name,
                raw_arguments=raw_arguments,
                arguments=arguments,
                status="SUCCESS",
                result=result,
                result_ok=result_ok,
                counted_as_successful_public_tool_call=result_ok,
                attempted_candidate=reserved_candidate,
                recorded_valid_candidate=recorded_candidate,
                started_at=started_at,
                completed_at=self._timestamp(),
                duration_seconds=max(0.0, self._monotonic() - started),
            )
            self._append_message(message, source="rdkit_tool")
        except Exception as exc:
            self._tool_failures += 1
            error_message = str(exc)
            message = self._tool_error_message(
                call_id, name, type(exc).__name__, error_message
            )
            self._event(
                "tool_call",
                turn=self._turn_count,
                call_id=call_id,
                tool=name,
                raw_arguments=raw_arguments,
                arguments=arguments,
                status="FAILED",
                error={"type": type(exc).__name__, "message": error_message},
                traceback=traceback.format_exc(),
                started_at=started_at,
                completed_at=self._timestamp(),
                duration_seconds=max(0.0, self._monotonic() - started),
            )
            self._append_message(message, source="rdkit_tool_failure")

    def _strict_final_candidate(self, content: Any) -> FinalCandidate:
        if not isinstance(content, str):
            raise OutputParseError("final assistant content must be a JSON string")
        max_bytes = int((self.task_config.get("output") or {}).get("max_submission_bytes", 1048576))
        if len(content.encode("utf-8")) > max_bytes:
            raise OutputParseError(f"final assistant content exceeds {max_bytes} bytes")

        def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise OutputParseError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        try:
            decoded = json.loads(
                content,
                object_pairs_hook=exact_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    OutputParseError(f"non-standard JSON constant {token!r}")
                ),
            )
        except OutputParseError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise OutputParseError(f"assistant content is not strict JSON: {exc}") from exc
        if not isinstance(decoded, dict) or set(decoded) != {"final_candidate"}:
            actual = sorted(decoded) if isinstance(decoded, dict) else type(decoded).__name__
            raise OutputParseError(
                "top-level JSON keys must be exactly ['final_candidate']; "
                f"received {actual}"
            )
        return FinalCandidate.from_exact_object(
            decoded["final_candidate"], self.task_config.get("output") or {}
        )

    def _final_process_guard_state(
        self, candidate: FinalCandidate
    ) -> dict[str, Any]:
        final_key = self._canonicalize_for_accounting(candidate.smiles)
        candidate_turns = sorted(set(self._candidate_first_turn.values()))
        issues: list[dict[str, Any]] = []
        if (
            self._successful_public_tool_call_count
            < self._minimum_successful_public_tool_calls
        ):
            issues.append(
                {
                    "code": "insufficient_successful_public_tool_calls",
                    "required": self._minimum_successful_public_tool_calls,
                    "observed": self._successful_public_tool_call_count,
                }
            )
        if len(self._unique_candidates) < self._minimum_unique_valid_candidates:
            issues.append(
                {
                    "code": "insufficient_novel_valid_evaluated_candidates",
                    "required": self._minimum_unique_valid_candidates,
                    "observed": len(self._unique_candidates),
                }
            )
        if len(candidate_turns) < self._minimum_candidate_feedback_turns:
            issues.append(
                {
                    "code": "insufficient_candidate_feedback_turns",
                    "required": self._minimum_candidate_feedback_turns,
                    "observed": len(candidate_turns),
                }
            )
        final_evaluated = final_key in self._unique_candidate_set
        if self._final_candidate_must_be_evaluated and not final_evaluated:
            issues.append(
                {
                    "code": "final_candidate_not_tool_evaluated",
                    "required": True,
                    "observed": False,
                }
            )
        return {
            "issues": issues,
            "successful_public_tool_calls": self._successful_public_tool_call_count,
            "unique_valid_evaluated_candidates": len(self._unique_candidates),
            "candidate_feedback_turns": candidate_turns,
            "evaluated_candidates": list(self._unique_candidates),
            "proposed_final_canonical_smiles": final_key,
            "proposed_final_evaluated": final_evaluated,
        }

    def _process_guard_retry_possible(
        self, *, turn: int, state: Mapping[str, Any]
    ) -> bool:
        issue_codes = {
            issue.get("code")
            for issue in state.get("issues", [])
            if isinstance(issue, Mapping)
        }
        evidence_codes = {
            "insufficient_successful_public_tool_calls",
            "insufficient_novel_valid_evaluated_candidates",
            "insufficient_candidate_feedback_turns",
        }
        needs_tool_feedback = bool(issue_codes.intersection(evidence_codes))
        remaining_turns = self.budget.max_agent_turns - turn
        missing_feedback_turns = max(
            0,
            self._minimum_candidate_feedback_turns
            - len(set(self._candidate_first_turn.values())),
        )
        # New tool evidence consumes one provider turn; a later provider turn
        # is then required for the final JSON.  A final-candidate-only fix can
        # instead select an already evaluated candidate on the next turn.
        future_tool_turns = (
            max(1, missing_feedback_turns) if needs_tool_feedback else 0
        )
        if remaining_turns < future_tool_turns + 1:
            return False
        if not needs_tool_feedback:
            return True

        needed_successes = max(
            0,
            self._minimum_successful_public_tool_calls
            - self._successful_public_tool_call_count,
        )
        needed_candidates = max(
            0,
            self._minimum_unique_valid_candidates - len(self._unique_candidates),
        )
        # A new feedback turn must introduce a novel valid candidate; replaying
        # a prior candidate on a later turn does not establish adaptive search.
        required_new_candidates = max(
            needed_candidates, missing_feedback_turns
        )
        required_future_calls = max(
            needed_successes, required_new_candidates
        )
        available_calls = self.budget.max_tool_calls - self._tool_call_count
        available_candidates = (
            self.budget.max_unique_evaluated_candidates
            - len(self._attempted_candidates)
        )
        return (
            required_future_calls <= available_calls
            and required_new_candidates <= available_candidates
        )

    def _process_guard_correction(self, state: Mapping[str, Any]) -> str:
        issue_codes = [
            str(issue.get("code"))
            for issue in state.get("issues", [])
            if isinstance(issue, Mapping)
        ]
        issue_set = set(issue_codes)
        needs_candidate_measurement = bool(
            issue_set
            & {
                "insufficient_novel_valid_evaluated_candidates",
                "insufficient_candidate_feedback_turns",
                "final_candidate_not_tool_evaluated",
            }
        )
        if needs_candidate_measurement:
            next_action = (
                "Your next response must use the native function-calling channel to "
                "call evaluate_candidate for a novel candidate (or for the proposed "
                "final candidate if only its measurement is missing), with exactly one "
                "argument named smiles."
            )
        else:
            next_action = (
                "Your next response must use the native function-calling channel to "
                "call calculate_qed for one already evaluated candidate, with exactly "
                "one argument named smiles."
            )
        return (
            "The proposed final JSON was rejected by the runner's measured-process "
            "guard. Resolve these issues: "
            + ", ".join(issue_codes)
            + ". "
            + next_action
            + " Do not place a smiles object in ordinary assistant text and do not "
            "return final JSON on that tool-call turn. Before returning final JSON, "
            "complete at least "
            f"{self._minimum_successful_public_tool_calls} successful public tool calls "
            f"and evaluate at least {self._minimum_unique_valid_candidates} novel valid "
            "candidates across at least two separate provider turns. The final SMILES "
            "must be one of the tool-evaluated candidates. Do not repeat already "
            "measured candidates merely to consume budget."
        )

    def _required_process_tool_name(self) -> str | None:
        if not self._final_process_guard_enabled:
            return None
        candidate_turn_count = len(set(self._candidate_first_turn.values()))
        if (
            len(self._unique_candidates) < self._minimum_unique_valid_candidates
            or candidate_turn_count < self._minimum_candidate_feedback_turns
        ):
            return "evaluate_candidate"
        if (
            self._successful_public_tool_call_count
            < self._minimum_successful_public_tool_calls
        ):
            return "calculate_qed"
        return None

    def _process_progress_guidance(self) -> str | None:
        """Give public, measured progress after tool turns before a premature final.

        Some providers aggressively try to finish immediately after the first tool
        result.  The benchmark requirements are already public, so reporting the
        observed counters here neither reveals evaluator internals nor supplies a
        candidate.  It simply gives the model a timely, request-anchored instruction
        to continue the required feedback loop with a valid public tool call.
        """

        if not self._final_process_guard_enabled:
            return None
        candidate_turn_count = len(set(self._candidate_first_turn.values()))
        calls_missing = max(
            0,
            self._minimum_successful_public_tool_calls
            - self._successful_public_tool_call_count,
        )
        candidates_missing = max(
            0,
            self._minimum_unique_valid_candidates - len(self._unique_candidates),
        )
        feedback_turns_missing = max(
            0,
            self._minimum_candidate_feedback_turns - candidate_turn_count,
        )
        if calls_missing == candidates_missing == feedback_turns_missing == 0:
            candidate_budget_full = (
                len(self._unique_candidates)
                >= self.budget.max_unique_evaluated_candidates
            )
            tool_budget_full = self._tool_call_count >= self.budget.max_tool_calls
            if candidate_budget_full or tool_budget_full:
                full_budgets = []
                if candidate_budget_full:
                    full_budgets.append(
                        "the unique evaluate_candidate budget is full"
                    )
                if tool_budget_full:
                    full_budgets.append("the public tool-call budget is full")
                return (
                    "Measured process requirements are complete and "
                    + " and ".join(full_budgets)
                    + ". Do not call any more tools: a rejected over-budget call "
                    "would make the run ineligible. In your next response, return "
                    "only the required strict final JSON and select the best feasible, "
                    "strictly improved candidate that was already measured."
                )
            return None

        next_tool = self._required_process_tool_name()
        if next_tool == "evaluate_candidate":
            next_action = (
                "call the public evaluate_candidate tool (not a made-up tool name) "
                "for a novel, connected candidate using exactly one JSON argument "
                "named smiles"
            )
        else:
            next_action = (
                "call the public calculate_qed tool for one already evaluated "
                "candidate using exactly one JSON argument named smiles"
            )

        return (
            "Measured process progress is incomplete: "
            f"{self._successful_public_tool_call_count}/"
            f"{self._minimum_successful_public_tool_calls} successful public tool calls, "
            f"{len(self._unique_candidates)}/"
            f"{self._minimum_unique_valid_candidates} novel valid evaluated candidates, "
            f"and {candidate_turn_count}/"
            f"{self._minimum_candidate_feedback_turns} candidate-feedback turns. "
            "Do not return final JSON yet. In your next response, use the native "
            "function-calling channel to "
            + next_action
            + ". Use the returned measurement to decide the following step."
        )

    def _build_final_answer(self, candidate: FinalCandidate) -> dict[str, Any]:
        final_key = self._canonicalize_for_accounting(candidate.smiles)
        final_evaluated = final_key in self._unique_candidate_set
        history: list[dict[str, str]] = []
        for item in self._candidate_history:
            if item["smiles"] == final_key:
                history.append(
                    {
                        "name": candidate.name,
                        "smiles": final_key,
                        "rationale": candidate.rationale,
                    }
                )
            else:
                history.append(dict(item))
        output_config = self.task_config.get("output", {})
        schema_version = (
            output_config.get("schema_version", "1.0")
            if isinstance(output_config, Mapping)
            else "1.0"
        )
        return {
            "schema_version": schema_version,
            "task_id": self._task_id(),
            "final_candidate": candidate.to_dict(),
            "candidate_history": history,
            "run_metadata": {
                "tool_call_count": self._tool_call_count,
                "budget_exceeded": self._budget_exceeded,
                "unique_evaluated_candidates": list(self._unique_candidates),
                "final_candidate_evaluated": final_evaluated,
            },
        }

    def _failure_answer(self, error: BaseException) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "task_id": self._task_id_or_none(),
            "error": {
                "type": type(error).__name__,
                "message": self._redact(str(error)),
            },
        }

    def _invoke_evaluator(self, final_path: Path, evaluation_path: Path) -> dict[str, Any]:
        evaluator = self._resolved_evaluator or self._load_evaluator()
        config_argument: Any = self.task_config_path or self.task_config
        try:
            parameters = inspect.signature(evaluator).parameters
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "evaluator signature cannot be inspected for process-evidence support"
            ) from exc
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        required_extensions = {"process_events", "require_process_evidence"}
        if not accepts_kwargs and not required_extensions.issubset(parameters):
            raise ConfigurationError(
                "evaluator must accept process_events and require_process_evidence"
            )
        result = evaluator(
            final_path,
            config_argument,
            output_path=evaluation_path,
            process_events=_to_plain(self._events),
            require_process_evidence=True,
        )
        plain = self._redact(_to_plain(result))
        if not isinstance(plain, dict):
            raise RuntimeError("evaluator must return a JSON object")
        # The returned object is authoritative.  Rewriting also prevents a
        # stale evaluation from surviving an explicitly allowed rerun when an
        # injected evaluator chooses not to write its output_path itself.
        _write_json(evaluation_path, plain)
        return plain

    @staticmethod
    def _require_successful_evaluation(evaluation: Mapping[str, Any]) -> None:
        status = evaluation.get("status")
        if not isinstance(status, str):
            raise RuntimeError("evaluator returned an unparseable status")
        if status != "ok":
            raise RuntimeError(f"evaluator did not complete successfully: status={status!r}")
        if evaluation.get("valid_submission") is not True:
            raise RuntimeError(
                "evaluator status is 'ok' but valid_submission is not true"
            )

    def _dialogue_markdown(self) -> str:
        lines = ["# Agent dialogue", ""]
        for index, message in enumerate(self._messages, 1):
            role = str(message.get("role", "unknown")).capitalize()
            suffix = ""
            if message.get("name"):
                suffix = f" — `{message['name']}`"
            lines.extend([f"## {index}. {role}{suffix}", ""])
            content = message.get("content")
            if content is None:
                lines.extend(["_No text content._", ""])
            else:
                lines.extend(["```text", str(content), "```", ""])
            details = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content", "name"}
            }
            if details:
                lines.extend(
                    [
                        "Message metadata:",
                        "",
                        "```json",
                        json.dumps(
                            details,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        ),
                        "```",
                        "",
                    ]
                )
        return "\n".join(lines).rstrip() + "\n"

    def _hashes(self) -> dict[str, Any]:
        config_hashes = {
            "task_config": _sha256_bytes(_canonical_json(self.task_config).encode("utf-8")),
            "model_config_sanitized": _sha256_bytes(
                _canonical_json(self.model_config.sanitized_dict()).encode("utf-8")
            ),
            "agent_prompt": _sha256_bytes(self.prompt.encode("utf-8")),
        }
        for label, candidate in (
            ("task_config_file", self.task_config_path),
            ("model_config_file", self.model_config_path),
            ("model_config_source_file", self.model_config_source_path),
            ("agent_prompt_file", self.prompt_path),
        ):
            if candidate is not None and candidate.is_file():
                config_hashes[label] = _sha256_file(candidate)
        source_paths = {
            "framework/agent_runner.py": Path(__file__),
            "framework/schemas.py": Path(__file__).with_name("schemas.py"),
        }
        project_root = Path(__file__).resolve().parents[1]
        for relative in (
            "framework/siliconflow_adapter.py",
            "tools/rdkit_tools.py",
            "evaluator/evaluate.py",
            "evaluator/scoring.py",
            "configs/task.yaml",
            "configs/models.example.yaml",
        ):
            candidate = project_root / relative
            if candidate.is_file():
                source_paths[relative] = candidate
        return {
            "configs": config_hashes,
            "sources": {
                label: _sha256_file(path) for label, path in source_paths.items()
            },
        }

    @staticmethod
    def _snapshot_changes(
        start: Mapping[str, Any], end: Mapping[str, Any]
    ) -> list[str]:
        changes: list[str] = []
        group_names = sorted(set(start) | set(end))
        for group_name in group_names:
            start_group = start.get(group_name)
            end_group = end.get(group_name)
            if not isinstance(start_group, Mapping) or not isinstance(
                end_group, Mapping
            ):
                if start_group != end_group:
                    changes.append(str(group_name))
                continue
            for label in sorted(set(start_group) | set(end_group)):
                if start_group.get(label) != end_group.get(label):
                    changes.append(f"{group_name}.{label}")
        return changes

    def _capture_integrity_end(self) -> bool:
        try:
            self._integrity_end = self._hashes()
            self._integrity_error = None
        except Exception as exc:
            self._integrity_end = None
            self._integrity_verified = False
            self._integrity_changes = []
            self._integrity_error = self._redact(f"{type(exc).__name__}: {exc}")
            return False
        if self._integrity_start is None:
            self._integrity_verified = False
            self._integrity_changes = []
            self._integrity_error = "start integrity snapshot is unavailable"
            return False
        self._integrity_changes = self._snapshot_changes(
            self._integrity_start, self._integrity_end
        )
        self._integrity_verified = not self._integrity_changes
        return self._integrity_verified

    def _aggregate_token_usage(self) -> dict[str, Any]:
        aggregate: dict[str, Any] = {}

        def merge(
            source: Mapping[str, Any], target: dict[str, Any], *, token_context: bool = False
        ) -> None:
            for raw_key, value in source.items():
                key = str(raw_key)
                child_context = token_context or "token" in key.lower()
                if isinstance(value, Mapping):
                    child: dict[str, Any] = {}
                    merge(value, child, token_context=child_context)
                    if child:
                        existing = target.setdefault(key, {})
                        if isinstance(existing, dict):
                            merge(child, existing, token_context=True)
                    continue
                if (
                    child_context
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    existing_value = target.get(key, 0)
                    if existing_value == "NON_FINITE_AGGREGATE":
                        continue
                    combined = existing_value + value
                    if isinstance(combined, float) and not math.isfinite(combined):
                        target[key] = "NON_FINITE_AGGREGATE"
                    else:
                        target[key] = combined

        for response in self._provider_responses:
            usage = response.get("usage")
            if isinstance(usage, Mapping):
                merge(usage, aggregate)
        return aggregate

    def _aggregate_reported_cost(self) -> Any | None:
        reports: list[Any] = []
        for response in self._provider_responses:
            value: Any = None
            for key in ("cost", "total_cost"):
                if key in response and response.get(key) is not None:
                    value = response.get(key)
                    break
            if value is None and isinstance(response.get("usage"), Mapping):
                usage = response["usage"]
                for key in ("total_cost", "cost"):
                    if key in usage and usage.get(key) is not None:
                        value = usage.get(key)
                        break
            if value is not None:
                reports.append(_to_plain(value))
        if not reports:
            return None
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in reports
        ):
            total = sum(reports)
            if isinstance(total, float) and not math.isfinite(total):
                return {"reported_values": reports}
            return total
        if all(
            isinstance(value, Mapping)
            and isinstance(value.get("amount"), (int, float))
            and not isinstance(value.get("amount"), bool)
            for value in reports
        ):
            currencies = {value.get("currency") for value in reports}
            if len(currencies) == 1:
                amount = sum(value["amount"] for value in reports)
                if isinstance(amount, float) and not math.isfinite(amount):
                    return {"reported_values": reports}
                return {
                    "amount": amount,
                    "currency": next(iter(currencies)),
                }
        # Preserve non-numeric provider reports verbatim rather than inventing a
        # conversion or currency.  The wrapper makes their non-aggregated nature
        # explicit while still proving that cost came from the interface.
        return {"reported_values": reports}

    def _metadata(
        self,
        *,
        status: str,
        started_at: str,
        completed_at: str,
        failure: dict[str, Any] | None,
        execution_mode: str,
    ) -> dict[str, Any]:
        metadata = {
            "schema_version": "1.0",
            "status": status,
            "execution_mode": execution_mode,
            "is_real_model_run": (
                execution_mode == "REAL_API"
                and status == "COMPLETE"
                and self.model_config.is_trusted_endpoint
                and self._canonical_dependencies
                and self._integrity_verified
                and bool(self._provider_responses)
                and self._provider_models_verified
                and self._actual_model_id is not None
            ),
            "task_id": self._task_id_or_none(),
            "run_label": self.run_label,
            "dependency_mode": self._dependency_mode,
            "canonical_dependencies": self._canonical_dependencies,
            "dependency_provenance": self._dependency_provenance,
            "provider": self.model_config.provider,
            "transport": self.model_config.transport,
            "base_url": self.model_config.base_url,
            "endpoint": self.model_config.base_url,
            "request_endpoint": (
                self.model_config.base_url.rstrip("/") + "/chat/completions"
            ),
            "requested_model": self.model_config.model,
            "requested_model_id": self.model_config.model,
            "model": self._actual_model_id,
            "actual_model_id": self._actual_model_id,
            "observed_model_ids": list(self._observed_model_ids),
            "model_config": self.model_config.sanitized_dict(),
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            "timestamp": completed_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": self._elapsed(),
            "limits": self.budget.to_dict(),
            "counts": {
                "agent_turns": self._turn_count,
                "provider_attempts": self._provider_attempts,
                "provider_responses": len(self._provider_responses),
                "tool_calls": self._tool_call_count,
                "successful_public_tool_calls": self._successful_public_tool_call_count,
                "tool_failures": self._tool_failures,
                "rejected_tool_calls": self._rejected_tool_calls,
                "unique_attempted_candidates": len(self._attempted_candidates),
                "unique_evaluated_candidates": len(self._unique_candidates),
                "candidate_feedback_turns": len(
                    set(self._candidate_first_turn.values())
                ),
            },
            "budget_exceeded": self._budget_exceeded,
            "failure": failure,
            "token_usage": self._aggregate_token_usage(),
            "provider_response_metadata": self._provider_responses,
            "hashes": self._integrity_start or {},
            "integrity": {
                "start_snapshot": self._integrity_start,
                "end_snapshot": self._integrity_end,
                "verified": self._integrity_verified,
                "changed": list(self._integrity_changes),
                "error": self._integrity_error,
            },
            "artifacts": list(ARTIFACT_NAMES),
        }
        cost = self._aggregate_reported_cost()
        if cost is not None:
            metadata["cost"] = cost
        return self._redact(metadata)

    def _write_trajectory(self, path: Path) -> None:
        text = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for event in self._events
        )
        path.write_text(text, encoding="utf-8", newline="\n")

    def _claim_output_dir(self) -> None:
        try:
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise ConfigurationError(
                "output directory already exists; overwrite is disabled, so choose a new directory"
            ) from exc
        except OSError as exc:
            raise ConfigurationError(
                f"output directory could not be atomically claimed: {exc}"
            ) from exc

    def run(self, *, raise_on_failure: bool = False) -> RunResult:
        # The leaf directory itself is the claim token.  There is no exists()
        # preflight window in which another process could create artifacts.
        self._claim_output_dir()
        paths = {name: self.output_dir / name for name in ARTIFACT_NAMES}
        self._started_monotonic = self._monotonic()
        started_at = self._timestamp()
        status = "FAILED"
        failure: dict[str, Any] | None = None
        caught: BaseException | None = None
        final_answer: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        tentative_complete_event: dict[str, Any] | None = None
        fatal_interrupt = False
        self._build_dependency_provenance()
        execution_mode = self._execution_mode()

        try:
            self._event(
                "run_start",
                status="RUNNING",
                task_id=self._task_id_or_none(),
                provider=self.model_config.provider,
                transport=self.model_config.transport,
                model=self.model_config.model,
                requested_model=self.model_config.model,
                endpoint=self.model_config.base_url,
                request_endpoint=(
                    self.model_config.base_url.rstrip("/") + "/chat/completions"
                ),
                execution_mode=execution_mode,
                dependency_mode=self._dependency_mode,
                canonical_dependencies=self._canonical_dependencies,
                dependency_provenance=self._dependency_provenance,
                integrity_start_snapshot=None,
            )
            run_start_event = self._events[-1]
            self._integrity_start = self._hashes()
            run_start_event["integrity_start_snapshot"] = _to_plain(
                self._integrity_start
            )

            api_key = self.validate_credentials()
            self._secret_for_redaction = api_key or ""
            if self._injected_transport is None:
                assert api_key is not None
                transport_class = self._default_transport_class()
                transport = transport_class(self.model_config, api_key)
                del api_key
            else:
                transport = self._injected_transport

            self._finalize_transport_provenance(transport)
            self._tools = self._load_tools()
            self._resolved_evaluator = self._load_evaluator()
            execution_mode = self._execution_mode()
            run_start_event["execution_mode"] = execution_mode
            run_start_event["dependency_mode"] = self._dependency_mode
            run_start_event["canonical_dependencies"] = self._canonical_dependencies
            run_start_event["dependency_provenance"] = self._redact(
                self._dependency_provenance
            )
            for message in self._initial_messages():
                self._append_message(message, source="runner")

            candidate: FinalCandidate | None = None
            last_parse_error: OutputParseError | None = None
            for turn in range(1, self.budget.max_agent_turns + 1):
                self._check_timeout()
                self._turn_count = turn
                response = self._provider_call(transport, turn)
                assistant, response_metadata = self._extract_assistant(response)
                self._append_message(assistant, source="provider")
                self._event(
                    "provider_response_metadata",
                    turn=turn,
                    metadata=response_metadata,
                )

                tool_calls = assistant.get("tool_calls") or []
                if tool_calls:
                    for call in tool_calls:
                        self._check_timeout()
                        self._execute_tool_call(call)
                    progress_guidance = self._process_progress_guidance()
                    if progress_guidance is not None:
                        self._append_message(
                            {"role": "user", "content": progress_guidance},
                            source="runner_process_progress",
                        )
                    continue

                try:
                    candidate = self._strict_final_candidate(assistant.get("content"))
                    last_parse_error = None
                    if self._final_process_guard_enabled:
                        guard_state = self._final_process_guard_state(candidate)
                        issues = guard_state["issues"]
                        if issues:
                            will_retry = self._process_guard_retry_possible(
                                turn=turn, state=guard_state
                            )
                            self._event(
                                "final_answer_process_guard",
                                turn=turn,
                                status="REJECTED",
                                will_retry=will_retry,
                                **guard_state,
                            )
                            candidate = None
                            if not will_retry:
                                self._budget_exceeded = True
                                raise BudgetExceededError(
                                    "final answer arrived before the required measured "
                                    "multi-turn process evidence and no bounded retry "
                                    "can satisfy the remaining requirements"
                                )
                            self._append_message(
                                {
                                    "role": "user",
                                    "content": self._process_guard_correction(
                                        guard_state
                                    ),
                                },
                                source="runner_process_guard",
                            )
                            continue
                    break
                except OutputParseError as exc:
                    last_parse_error = exc
                    self._event(
                        "final_answer_parse_failure",
                        turn=turn,
                        error={"type": type(exc).__name__, "message": str(exc)},
                        will_retry=turn < self.budget.max_agent_turns,
                    )
                    if turn < self.budget.max_agent_turns:
                        self._append_message(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was not the required strict JSON: "
                                    f"{exc}. Return only an object with exactly final_candidate "
                                    "and exactly name, smiles, rationale. The first non-whitespace "
                                    "character must be { and the last must be }; do not use "
                                    "backticks, a Markdown fence, a json label, or extra text."
                                ),
                            },
                            source="runner_parse_retry",
                        )

            if candidate is None:
                self._budget_exceeded = True
                if last_parse_error is not None:
                    raise BudgetExceededError(
                        "maximum agent turns reached without a valid final answer; "
                        f"last parse error: {last_parse_error}"
                    )
                raise BudgetExceededError(
                    "maximum agent turns reached before the model returned a final answer"
                )

            final_answer = self._build_final_answer(candidate)
            _write_json(paths["final_answer.json"], final_answer)
            if not self._capture_integrity_end():
                raise RuntimeError(
                    "run source/config integrity changed before evaluation: "
                    + ", ".join(self._integrity_changes or [self._integrity_error or "unknown"])
                )
            self._event(
                "run_complete",
                status="COMPLETE",
                terminal_status="COMPLETE",
                dependency_mode=self._dependency_mode,
                canonical_dependencies=self._canonical_dependencies,
                integrity_verified=True,
            )
            tentative_complete_event = self._events[-1]
            evaluation = self._invoke_evaluator(
                paths["final_answer.json"], paths["evaluation.json"]
            )
            self._require_successful_evaluation(evaluation)
            if not self._events or self._events[-1] is not tentative_complete_event:
                raise RuntimeError("events were appended after terminal run_complete")
            if not self._capture_integrity_end():
                raise RuntimeError(
                    "run source/config integrity changed during evaluation: "
                    + ", ".join(self._integrity_changes or [self._integrity_error or "unknown"])
                )
            status = "COMPLETE"
        except BaseException as exc:
            fatal_interrupt = isinstance(exc, (KeyboardInterrupt, SystemExit))
            if tentative_complete_event is not None:
                for index, event in enumerate(self._events):
                    if event is tentative_complete_event:
                        self._events = self._events[:index]
                        break
                tentative_complete_event = None
            caught = exc
            failure = {
                "type": type(exc).__name__,
                "message": self._redact(str(exc)),
                "traceback": self._redact(traceback.format_exc()),
            }
            self._event(
                "run_failure",
                status="FAILED",
                terminal_status="FAILED",
                dependency_mode=self._dependency_mode,
                canonical_dependencies=self._canonical_dependencies,
                integrity_verified=self._integrity_verified,
                error=failure,
            )
            terminal_failure_event = self._events[-1]
            if not final_answer:
                try:
                    final_answer = self._failure_answer(exc)
                except Exception:
                    final_answer = {
                        "schema_version": "1.0",
                        "task_id": "unknown",
                        "error": {"type": type(exc).__name__, "message": self._redact(str(exc))},
                    }
            _write_json(paths["final_answer.json"], final_answer)
            try:
                # Re-evaluate against the final failure lifecycle even when a
                # tentative successful evaluation already ran.  This keeps the
                # in-memory evidence identical to the JSONL replay evidence.
                evaluation = self._invoke_evaluator(
                    paths["final_answer.json"], paths["evaluation.json"]
                )
            except Exception as evaluator_exc:
                evaluation = {
                    "status": "evaluation_failed",
                    "valid_submission": False,
                    "total_score": 0.0,
                    "error": {
                        "type": type(evaluator_exc).__name__,
                        "message": self._redact(str(evaluator_exc)),
                    },
                }
                _write_json(paths["evaluation.json"], evaluation)
            if any(event is terminal_failure_event for event in self._events):
                terminal_index = next(
                    index
                    for index, event in enumerate(self._events)
                    if event is terminal_failure_event
                )
                self._events = self._events[: terminal_index + 1]
        finally:
            self._capture_integrity_end()
            execution_mode = self._execution_mode()
            # All five artifacts are produced for both success and failure.
            if not paths["final_answer.json"].exists():
                _write_json(paths["final_answer.json"], final_answer)
            if not paths["evaluation.json"].exists():
                _write_json(paths["evaluation.json"], evaluation)
            self._write_trajectory(paths["trajectory.jsonl"])
            paths["dialogue.md"].write_text(
                self._dialogue_markdown(), encoding="utf-8", newline="\n"
            )
            completed_at = self._timestamp()
            metadata = self._metadata(
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                failure=failure,
                execution_mode=execution_mode,
            )
            _write_json(paths["metadata.json"], metadata)
            self._secret_for_redaction = ""

        result = RunResult(
            status=status,
            output_dir=self.output_dir,
            final_answer=final_answer,
            evaluation=evaluation,
            metadata=metadata,
        )
        if fatal_interrupt and caught is not None:
            raise caught
        if caught is not None and raise_on_failure:
            raise caught
        return result


def run_agent(**kwargs: Any) -> RunResult:
    """Convenience entry point mirroring :class:`AgentRunner` construction."""

    raise_on_failure = bool(kwargs.pop("raise_on_failure", False))
    return AgentRunner(**kwargs).run(raise_on_failure=raise_on_failure)

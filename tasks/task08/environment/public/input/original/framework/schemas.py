"""Strict, dependency-light configuration and protocol schemas.

The benchmark keeps task, model, and tool contracts outside the runner.  This
module validates those documents before any provider or native tool can run.
It intentionally stores credential *environment variable names* only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import yaml


class FrameworkError(RuntimeError):
    """Base class for expected benchmark framework failures."""


class ConfigurationError(FrameworkError):
    """A trusted configuration or contract is missing, unsafe, or invalid."""


class MissingCredentialError(FrameworkError):
    """The configured runtime credential is unavailable."""

    safe_status = "MISSING_GJLD_API_KEY"


class ProviderError(FrameworkError):
    """A provider call failed or returned an incompatible response."""


class ProtocolViolationError(FrameworkError):
    """The model violated the bounded tool/final-submission protocol."""


class BudgetExceededError(FrameworkError):
    """A configured turn, tool, execution, retry, or time budget was exceeded."""


class ToolContractError(FrameworkError):
    """A tool call does not satisfy the published JSON contract."""


class LifecycleError(FrameworkError):
    """An event would violate the append-only trajectory lifecycle."""


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEX_64 = re.compile(r"^[0-9A-Fa-f]{64}$")
_SAFE_SCHEMA_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_API_KEY_ENV = "GJLD_API_KEY"
GENESIS_HASH = "0" * 64


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be an object")
    return value


def _string(value: Any, path: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ConfigurationError(f"{path} must be a non-empty bounded string")
    return value.strip()


def _integer(value: Any, path: str, *, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer")
    if value < minimum or value > maximum:
        raise ConfigurationError(f"{path} is outside the supported range")
    return value


def _number(value: Any, path: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ConfigurationError(f"{path} is outside the supported range")
    return result


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ConfigurationError(f"{path} must be a boolean")
    return value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every depth."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key, value in loader.construct_pairs(node, deep=deep):
        if not isinstance(key, str):
            raise ConfigurationError("configuration YAML keys must be strings")
        if key in result:
            raise ConfigurationError("configuration YAML contains duplicate keys")
        result[key] = value
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ConfigurationError("configuration file is missing")
    try:
        raw = candidate.read_bytes()
        if len(raw) > 1_048_576 or b"\x00" in raw:
            raise ConfigurationError("configuration file exceeds safe limits")
        loaded = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError("configuration file is not safe UTF-8 YAML") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError("configuration root must be an object")
    return loaded


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ConfigurationError("contract file is missing")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfigurationError("contract JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        raw = candidate.read_bytes()
        if len(raw) > 2_097_152 or b"\x00" in raw:
            raise ConfigurationError("contract file exceeds safe limits")
        loaded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ConfigurationError("contract JSON contains a non-finite value")
            ),
        )
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("contract file is not strict UTF-8 JSON") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError("contract root must be an object")
    return loaded


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], path: str = "protocol") -> "ProtocolConfig":
        raw = _mapping(value, path)
        if set(raw) != {"name", "version"}:
            raise ConfigurationError(f"{path} must contain exactly name and version")
        return cls(
            name=_string(raw.get("name"), f"{path}.name", maximum=128),
            version=_string(raw.get("version"), f"{path}.version", maximum=32),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetConfig:
    max_tool_calls: int
    min_successful_tool_calls: int
    max_agent_turns: int
    max_execution_attempts: int
    max_provider_retries: int
    total_timeout_seconds: float
    single_xtb_timeout_seconds: float
    max_text_read_lines_per_call: int
    max_text_read_bytes_per_call: int
    require_read_text_evidence: bool
    require_native_xtb_execution: bool
    require_geometry_optimization: bool
    require_hessian: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetConfig":
        raw = _mapping(value, "budget")
        expected = {
            "max_tool_calls",
            "min_successful_tool_calls",
            "max_agent_turns",
            "max_execution_attempts",
            "max_provider_retries",
            "total_timeout_seconds",
            "single_xtb_timeout_seconds",
            "max_text_read_lines_per_call",
            "max_text_read_bytes_per_call",
            "require_read_text_evidence",
            "require_native_xtb_execution",
            "require_geometry_optimization",
            "require_hessian",
        }
        if set(raw) != expected:
            raise ConfigurationError("budget must contain exactly the published fields")
        result = cls(
            max_tool_calls=_integer(raw.get("max_tool_calls"), "budget.max_tool_calls", minimum=1),
            min_successful_tool_calls=_integer(
                raw.get("min_successful_tool_calls"),
                "budget.min_successful_tool_calls",
                minimum=0,
            ),
            max_agent_turns=_integer(raw.get("max_agent_turns"), "budget.max_agent_turns", minimum=1, maximum=10_000),
            max_execution_attempts=_integer(
                raw.get("max_execution_attempts"),
                "budget.max_execution_attempts",
                minimum=0,
                maximum=10_000,
            ),
            max_provider_retries=_integer(
                raw.get("max_provider_retries"),
                "budget.max_provider_retries",
                minimum=0,
                maximum=10,
            ),
            total_timeout_seconds=_number(
                raw.get("total_timeout_seconds"),
                "budget.total_timeout_seconds",
                minimum=0.001,
                maximum=86_400,
            ),
            single_xtb_timeout_seconds=_number(
                raw.get("single_xtb_timeout_seconds"),
                "budget.single_xtb_timeout_seconds",
                minimum=0.001,
                maximum=86_400,
            ),
            max_text_read_lines_per_call=_integer(
                raw.get("max_text_read_lines_per_call"),
                "budget.max_text_read_lines_per_call",
                minimum=1,
                maximum=100_000,
            ),
            max_text_read_bytes_per_call=_integer(
                raw.get("max_text_read_bytes_per_call"),
                "budget.max_text_read_bytes_per_call",
                minimum=1,
                maximum=16 * 1024 * 1024,
            ),
            require_read_text_evidence=_boolean(
                raw.get("require_read_text_evidence"),
                "budget.require_read_text_evidence",
            ),
            require_native_xtb_execution=_boolean(
                raw.get("require_native_xtb_execution"),
                "budget.require_native_xtb_execution",
            ),
            require_geometry_optimization=_boolean(
                raw.get("require_geometry_optimization"),
                "budget.require_geometry_optimization",
            ),
            require_hessian=_boolean(
                raw.get("require_hessian"),
                "budget.require_hessian",
            ),
        )
        if result.min_successful_tool_calls > result.max_tool_calls:
            raise ConfigurationError("minimum successful tool calls exceed the tool budget")
        if result.single_xtb_timeout_seconds > result.total_timeout_seconds:
            raise ConfigurationError("single xTB timeout exceeds the total run timeout")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskConfig:
    schema_version: str
    task_id: str
    case_id: str
    protocol: ProtocolConfig
    budget: BudgetConfig
    allowed_tools: tuple[str, ...]
    genesis_hash: str = GENESIS_HASH
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskConfig":
        root = _mapping(value, "task_config")
        schema_version = _string(root.get("schema_version"), "schema_version", maximum=32)
        task = _mapping(root.get("task"), "task")
        protocol = ProtocolConfig.from_mapping(_mapping(task.get("protocol"), "task.protocol"), "task.protocol")
        tools = _mapping(root.get("tools"), "tools")
        allowed_value = tools.get("allowed")
        if not isinstance(allowed_value, list) or not allowed_value:
            raise ConfigurationError("tools.allowed must be a non-empty list")
        allowed: list[str] = []
        for index, item in enumerate(allowed_value):
            name = _string(item, f"tools.allowed[{index}]", maximum=128)
            if name in allowed:
                raise ConfigurationError("tools.allowed contains duplicates")
            allowed.append(name)

        models = _mapping(root.get("models"), "models")
        if models.get("allow_fallback") is not False:
            raise ConfigurationError("models.allow_fallback must be false")
        if models.get("allow_environment_model_override") is not False:
            raise ConfigurationError("environment model override must be disabled")
        if models.get("require_exact_response_model") is not True:
            raise ConfigurationError("exact response-model verification must be enabled")

        deterministic = _mapping(root.get("determinism"), "determinism")
        genesis = deterministic.get("trajectory_genesis_hash", GENESIS_HASH)
        if not isinstance(genesis, str) or not _HEX_64.fullmatch(genesis):
            raise ConfigurationError("determinism.trajectory_genesis_hash must be SHA-256 text")
        return cls(
            schema_version=schema_version,
            task_id=_string(task.get("id"), "task.id", maximum=256),
            case_id=_string(task.get("case_id"), "task.case_id", maximum=256),
            protocol=protocol,
            budget=BudgetConfig.from_mapping(_mapping(root.get("budget"), "budget")),
            allowed_tools=tuple(allowed),
            genesis_hash=genesis.lower(),
            raw=copy.deepcopy(dict(root)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        return cls.from_mapping(_read_yaml(path))


_MODEL_FIELDS = {
    "provider",
    "transport",
    "base_url",
    "api_key_env",
    "temperature",
    "max_tokens",
    "timeout_seconds",
    "max_retries",
    "sdk_max_retries",
    "model",
    "family",
    "logical_name",
}


@dataclass(frozen=True)
class ModelConfig:
    logical_name: str
    family: str
    provider: str
    transport: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    sdk_max_retries: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, logical_name: str | None = None) -> "ModelConfig":
        raw = dict(_mapping(value, "model_config"))
        unknown = sorted(set(raw) - _MODEL_FIELDS)
        if unknown:
            raise ConfigurationError("model configuration contains unknown fields")
        secret_fields = sorted(
            key for key in raw if key.casefold() in {"api_key", "token", "secret", "password"}
        )
        if secret_fields:
            raise ConfigurationError("inline model credentials are forbidden")
        logical = logical_name or raw.get("logical_name")
        provider = _string(raw.get("provider"), "model.provider", maximum=64).lower()
        transport = _string(raw.get("transport"), "model.transport", maximum=64).lower()
        base_url = _string(raw.get("base_url"), "model.base_url", maximum=512)
        api_key_env = _string(raw.get("api_key_env"), "model.api_key_env", maximum=128)
        if provider != "siliconflow" or transport != "openai_compatible":
            raise ConfigurationError("formal models require SiliconFlow OpenAI-compatible transport")
        if base_url != SILICONFLOW_BASE_URL:
            raise ConfigurationError("formal models require the exact official endpoint")
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigurationError("model endpoint is unsafe")
        if api_key_env != SILICONFLOW_API_KEY_ENV or not _ENV_NAME.fullmatch(api_key_env):
            raise ConfigurationError("formal models require the exact credential environment name")
        sdk_retries = _integer(raw.get("sdk_max_retries", 0), "model.sdk_max_retries", maximum=0)
        if sdk_retries != 0:
            raise ConfigurationError("SDK retries must be disabled")
        return cls(
            logical_name=_string(logical, "model.logical_name", maximum=64),
            family=_string(raw.get("family"), "model.family", maximum=128),
            provider=provider,
            transport=transport,
            model=_string(raw.get("model"), "model.model", maximum=256),
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=_number(raw.get("temperature", 0.2), "model.temperature", minimum=0, maximum=2),
            max_tokens=_integer(raw.get("max_tokens", 8192), "model.max_tokens", minimum=1),
            timeout_seconds=_number(raw.get("timeout_seconds", 120), "model.timeout_seconds", minimum=0.001, maximum=86_400),
            max_retries=_integer(raw.get("max_retries", 2), "model.max_retries", minimum=0, maximum=10),
            sdk_max_retries=sdk_retries,
        )

    @classmethod
    def from_registry(cls, document: Mapping[str, Any], logical_name: str) -> "ModelConfig":
        root = _mapping(document, "model_registry")
        if root.get("schema_version") != "1.0":
            raise ConfigurationError("unsupported model registry schema")
        defaults = dict(_mapping(root.get("provider_defaults"), "provider_defaults"))
        models = _mapping(root.get("models"), "models")
        selected = _mapping(models.get(logical_name), f"models.{logical_name}")
        merged = {**defaults, **dict(selected), "logical_name": logical_name}
        return cls.from_mapping(merged, logical_name=logical_name)

    @classmethod
    def load_registry(cls, path: str | Path, logical_name: str) -> "ModelConfig":
        return cls.from_registry(_read_yaml(path), logical_name)

    def sanitized_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> None:
    """Validate the strict JSON-Schema subset used by tool_contracts.json."""

    if "anyOf" in schema:
        alternatives = schema["anyOf"]
        if not isinstance(alternatives, list) or not alternatives:
            raise ConfigurationError(f"{path} contract anyOf is invalid")
        for alternative in alternatives:
            try:
                validate_json_schema(value, _mapping(alternative, f"{path}.anyOf"), path)
                return
            except ToolContractError:
                continue
        raise ToolContractError(f"{path} does not match any allowed schema")

    expected = schema.get("type")
    if expected is not None:
        if not isinstance(expected, str) or expected not in _SAFE_SCHEMA_TYPES:
            raise ConfigurationError(f"{path} contract type is invalid")
        if not _json_type_matches(value, expected):
            raise ToolContractError(f"{path} must be {expected}")
    if "const" in schema and value != schema["const"]:
        raise ToolContractError(f"{path} does not equal the required constant")
    if "enum" in schema:
        allowed = schema["enum"]
        if not isinstance(allowed, list) or value not in allowed:
            raise ToolContractError(f"{path} is not an allowed value")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ConfigurationError(f"{path}.properties contract is invalid")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ConfigurationError(f"{path}.required contract is invalid")
        missing = [name for name in required if name not in value]
        if missing:
            raise ToolContractError(f"{path} is missing required fields")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ToolContractError(f"{path} contains unexpected fields")
        for key, item in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                validate_json_schema(item, _mapping(child_schema, f"{path}.{key}"), f"{path}.{key}")

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ToolContractError(f"{path} contains too few items")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ToolContractError(f"{path} contains too many items")
        if schema.get("uniqueItems") is True:
            normalized = [json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for item in value]
            if len(normalized) != len(set(normalized)):
                raise ToolContractError(f"{path} must contain unique items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                validate_json_schema(item, _mapping(item_schema, f"{path}.items"), f"{path}[{index}]")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise ToolContractError(f"{path} is too short")
        if isinstance(maximum, int) and len(value) > maximum:
            raise ToolContractError(f"{path} is too long")
        pattern = schema.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise ConfigurationError(f"{path}.pattern contract is invalid")
            try:
                matches = re.fullmatch(pattern, value) is not None
            except re.error as exc:
                raise ConfigurationError(f"{path}.pattern contract is invalid") from exc
            if not matches:
                raise ToolContractError(f"{path} does not match the required pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise ToolContractError(f"{path} is below the minimum")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise ToolContractError(f"{path} exceeds the maximum")


@dataclass(frozen=True)
class ToolContractSet:
    schema_version: str
    protocol: ProtocolConfig
    definitions: tuple[dict[str, Any], ...]
    parameters_by_name: Mapping[str, Mapping[str, Any]] = field(repr=False)
    final_answer_artifact_schema: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToolContractSet":
        root = _mapping(value, "tool_contracts")
        schema_version = _string(root.get("schema_version"), "tool_contracts.schema_version", maximum=32)
        protocol = ProtocolConfig.from_mapping(_mapping(root.get("protocol"), "tool_contracts.protocol"), "tool_contracts.protocol")
        definitions = root.get("tools")
        if not isinstance(definitions, list) or not definitions:
            raise ConfigurationError("tool_contracts.tools must be a non-empty list")
        copied: list[dict[str, Any]] = []
        parameters: dict[str, Mapping[str, Any]] = {}
        artifact_schema = _mapping(
            root.get("final_answer_artifact_schema"),
            "tool_contracts.final_answer_artifact_schema",
        )
        if (
            artifact_schema.get("type") != "object"
            or artifact_schema.get("additionalProperties") is not False
        ):
            raise ConfigurationError("final-answer artifact schema must be a closed object")
        for index, definition in enumerate(definitions):
            item = _mapping(definition, f"tools[{index}]")
            if item.get("type") != "function":
                raise ConfigurationError("every tool contract must have type=function")
            function = _mapping(item.get("function"), f"tools[{index}].function")
            name = _string(function.get("name"), f"tools[{index}].function.name", maximum=128)
            if name in parameters:
                raise ConfigurationError("tool contract names must be unique")
            if function.get("strict") is not True:
                raise ConfigurationError("tool contracts must be strict")
            schema = _mapping(function.get("parameters"), f"tools[{index}].function.parameters")
            if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
                raise ConfigurationError("tool argument schemas must be closed objects")
            parameters[name] = copy.deepcopy(dict(schema))
            copied.append(copy.deepcopy(dict(item)))
        return cls(
            schema_version=schema_version,
            protocol=protocol,
            definitions=tuple(copied),
            parameters_by_name=parameters,
            final_answer_artifact_schema=copy.deepcopy(dict(artifact_schema)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ToolContractSet":
        return cls.from_mapping(_read_json(path))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.parameters_by_name)

    def validate_arguments(self, name: str, arguments: Any) -> dict[str, Any]:
        schema = self.parameters_by_name.get(name)
        if schema is None:
            raise ToolContractError("unknown tool")
        if not isinstance(arguments, dict):
            raise ToolContractError("tool arguments must be a JSON object")
        validate_json_schema(arguments, schema)
        return copy.deepcopy(arguments)

    def validate_final_answer_artifact(
        self,
        value: Any,
        *,
        require_valid_submission: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ToolContractError("final-answer artifact must be a JSON object")
        validate_json_schema(
            value,
            self.final_answer_artifact_schema,
            "final_answer",
        )
        if require_valid_submission:
            self.validate_arguments("submit_final", value.get("model_submission"))
        return copy.deepcopy(value)

    def provider_definitions(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self.definitions))


@dataclass(frozen=True)
class RunResult:
    status: str
    run_id: str
    final_answer: dict[str, Any]
    metadata: dict[str, Any]
    artifacts: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETE"


__all__ = [
    "BudgetConfig",
    "BudgetExceededError",
    "ConfigurationError",
    "FrameworkError",
    "GENESIS_HASH",
    "LifecycleError",
    "MissingCredentialError",
    "ModelConfig",
    "ProtocolConfig",
    "ProtocolViolationError",
    "ProviderError",
    "RunResult",
    "SILICONFLOW_API_KEY_ENV",
    "SILICONFLOW_BASE_URL",
    "TaskConfig",
    "ToolContractError",
    "ToolContractSet",
    "validate_json_schema",
]

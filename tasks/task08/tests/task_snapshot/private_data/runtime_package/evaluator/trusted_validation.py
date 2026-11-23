"""Independent, deterministic validation for xTB recovery trajectories.

This module deliberately does not import the runner, the public tools, or a
tool registry.  Everything used for scoring is re-derived from the frozen
configuration, the JSONL trace, and the artifact workspace supplied to the
evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any, Iterable, Mapping, Sequence

import yaml


EVALUATOR_SCHEMA_VERSION = "1.0"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024
MAX_TRAJECTORY_LINES = 100_000
MAX_TRAJECTORY_LINE_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_FILES = 20_000
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_SCANNED_TEXT_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 40
MAX_JSON_NODES = 250_000

HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ATTEMPT_RE = re.compile(r"^attempt_((?!000)[0-9]{3})$")
INITIAL_ATTEMPT = "attempt_000_initial_failure"
TERMINAL_EVENTS = frozenset({"run_complete", "run_failure"})
ALLOWED_EVENTS = frozenset(
    {
        "run_start",
        "provider_request",
        "provider_retry",
        "provider_error",
        "provider_response_metadata",
        "assistant_message",
        "tool_call",
        "tool_result",
        "budget_state",
        "attempt_prepared",
        "native_xtb_execution",
        "attempt_validated",
        "final_submitted",
        "run_complete",
        "run_failure",
    }
)
ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "run_id",
        "sequence",
        "timestamp_utc",
        "event",
        "payload",
        "prev_event_hash",
        "event_hash",
    }
)
READ_TOOLS = frozenset(
    {
        "read_text_range",
        "tail_text",
        "parse_xtb_log",
        "inspect_scheduler_record",
        "validate_xyz",
        "inspect_artifact_provenance",
    }
)
PATH_ARGUMENTS = frozenset({"path"})
HIDDEN_REASONING_KEYS = frozenset(
    {"reasoning_content", "chain_of_thought", "thinking_content"}
)
SHELL_META_RE = re.compile(r"[;&|><`\r\n\x00]")
WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?:^|[\s\"'])(?:[a-z]:[\\/]|\\\\[^\\\s]+\\[^\\\s]+)")
POSIX_ABSOLUTE_RE = re.compile(r"(?:^|[\s\"'])/(?:home|root|tmp|var|etc|opt|usr|mnt|Users)/")
OPT_CONVERGED_RE = re.compile(
    r"geometry\s+optimization\s+converged|optimization\s+converged",
    re.IGNORECASE,
)
OPT_FAILURE_RE = re.compile(
    r"(?:maximum|max\.?)[ -]?(?:number\s+of\s+)?(?:optimization\s+)?cycles|"
    r"failed\s+to\s+converge|not\s+converged|optimization\s+failed",
    re.IGNORECASE,
)
NORMAL_TERMINATION_RE = re.compile(
    r"normal\s+termination|finished\s+run\s+on", re.IGNORECASE
)
HESSIAN_RE = re.compile(
    r"vibrational\s+frequenc|frequency\s+printout|hessian", re.IGNORECASE
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|bearer)\b\s*[:=]\s*[\"']?[A-Za-z0-9_./+\-=]{12,}"),
    re.compile(r"(?i)\bGJLD_API_KEY\s*=\s*[^\s\"']{8,}"),
)
REQUIRED_ELIGIBILITY_GATES = frozenset(
    {
        "complete_lifecycle",
        "formal_mode_truthful",
        "native_xtb_executed",
        "decisive_log_evidence_read",
        "optimization_converged",
        "hessian_completed",
        "final_artifacts_current",
        "legacy_not_used_as_current",
        "scientific_invariants_preserved",
        "oracle_not_accessed",
        "path_and_argv_safe",
        "budgets_respected",
        "trace_hash_chain_valid",
        "final_answer_schema_valid",
        "trusted_runtime_verified",
        "exact_model_identity_confirmed",
        "no_fallback_or_manual_rewrite",
        "no_secret_finding",
    }
)
FROZEN_MODELS = {
    "kimi": "Pro/moonshotai/Kimi-K2.6",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "glm": "zai-org/GLM-5.2",
}


class TrustedValidationError(ValueError):
    """A bounded, user-actionable evaluator input error."""


class DuplicateKeyError(ValueError):
    """Raised when a JSON/YAML mapping repeats a key."""


@dataclass(frozen=True, order=True)
class Issue:
    """Stable issue record; messages never contain host absolute paths."""

    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass
class TrustedConfiguration:
    config: dict[str, Any]
    project_root: Path
    config_path: Path
    job_spec: dict[str, Any]
    model_registry: dict[str, Any]
    tool_contracts: dict[str, Any]
    submission_schema: dict[str, Any]
    final_answer_artifact_schema: dict[str, Any]
    issues: list[Issue] = field(default_factory=list)

    @property
    def final_schema(self) -> dict[str, Any]:
        """Compatibility alias for the model-facing submit_final schema."""

        return self.submission_schema

    @property
    def protocol(self) -> tuple[str, str]:
        protocol = _mapping(_mapping(self.config.get("task")).get("protocol"))
        return str(protocol.get("name", "")), str(protocol.get("version", ""))

    @property
    def task_id(self) -> str:
        return str(_mapping(self.config.get("task")).get("id", ""))

    @property
    def case_id(self) -> str:
        return str(_mapping(self.config.get("task")).get("case_id", ""))

    @property
    def budget(self) -> dict[str, Any]:
        return dict(_mapping(self.config.get("budget")))

    @property
    def allowed_tools(self) -> frozenset[str]:
        values = _mapping(self.config.get("tools")).get("allowed", [])
        if not isinstance(values, list):
            return frozenset()
        return frozenset(value for value in values if type(value) is str)


@dataclass
class ToolCallFact:
    sequence: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    result_sequence: int | None = None
    ok: bool = False
    result: Any = None


@dataclass
class TraceFacts:
    events: list[dict[str, Any]]
    issues: list[Issue]
    trace_sha256: str
    final_event_hash: str
    lifecycle_valid: bool
    hash_chain_valid: bool
    protocol_valid: bool
    terminal_event: str | None
    run_id: str | None
    run_type: str
    execution_mode: str
    native_mode: str
    logical_model: str | None
    requested_model: str | None
    actual_models: list[str]
    provider_request_ids: list[str]
    exact_model_identity: bool
    identity_exempt: bool
    mode_truthful: bool
    tool_calls: list[ToolCallFact]
    successful_tool_calls: int
    agent_turns: int
    provider_retries: int
    provider_retry_event_ceiling: int
    provider_retry_budget_scope: str
    infra_degraded: bool
    attempt_prepared: list[dict[str, Any]]
    native_executions: list[dict[str, Any]]
    attempt_validations: list[dict[str, Any]]
    attempt_event_contracts_valid: bool
    native_event_contracts_valid: bool
    validation_event_contracts_valid: bool
    final_submitted: dict[str, Any] | None
    budgets_respected: bool
    budget_state_valid: bool
    direct_log_reads: list[ToolCallFact]
    decisive_log_evidence_read: bool
    scheduler_completed_observed: bool
    oracle_accessed: bool
    unsafe_path_or_argv: bool
    fallback_or_manual_marker: bool
    secret_finding: bool
    hidden_reasoning_present: bool


@dataclass
class ArtifactFacts:
    issues: list[Issue]
    workspace_sha256: str
    files: list[dict[str, Any]]
    case_manifest_valid: bool
    manifests_valid: bool
    canonical_attempt_files_valid: bool
    prepared_attempts: dict[str, dict[str, Any]]
    provenance_records: dict[str, dict[str, Any]]
    run_records: dict[str, dict[str, Any]]
    validations: dict[str, dict[str, Any]]
    actual_argv_safe: bool
    native_xtb_executed: bool
    optimization_converged: bool
    hessian_completed: bool
    final_artifacts_current: bool
    legacy_used_as_current: bool
    scientific_invariants_preserved: bool
    provenance_valid: bool
    final_xyz_valid: bool
    oracle_present: bool
    secret_finding: bool
    absolute_path_finding: bool
    filesystem_safe: bool
    attempt_isolation_valid: bool
    restart_from_last_current_frame: bool
    restart_from_original_input: bool
    targeted_cycle_repair: bool
    initial_scheduler_completed: bool
    initial_failure_max_cycles: bool
    legacy_artifact_present: bool


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate key: {key[:80]}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[str, Any]:
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node, deep=deep)
    return _duplicate_object(pairs)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _bounded_read(path: Path, maximum: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise TrustedValidationError(f"cannot stat {label}: {type(exc).__name__}") from exc
    if size > maximum:
        raise TrustedValidationError(f"{label} exceeds {maximum} bytes")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise TrustedValidationError(f"cannot read {label}: {type(exc).__name__}") from exc


def load_json(path: Path, *, maximum: int = MAX_JSON_BYTES, label: str = "JSON") -> Any:
    raw = _bounded_read(path, maximum, label)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TrustedValidationError(f"{label} must be UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_object,
            parse_constant=_reject_constant,
            parse_float=Decimal,
        )
    except (ValueError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise TrustedValidationError(f"invalid {label}: {type(exc).__name__}") from exc
    _validate_json_complexity(value, label)
    return value


def load_yaml(path: Path, *, maximum: int = MAX_CONFIG_BYTES) -> dict[str, Any]:
    raw = _bounded_read(path, maximum, "task config")
    try:
        value = yaml.load(raw.decode("utf-8-sig"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError, DuplicateKeyError) as exc:
        raise TrustedValidationError(f"invalid task config: {type(exc).__name__}") from exc
    if type(value) is not dict:
        raise TrustedValidationError("task config must be a mapping")
    _validate_json_complexity(value, "task config")
    return value


def _validate_json_complexity(value: Any, label: str) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise TrustedValidationError(f"{label} contains too many JSON nodes")
        if depth > MAX_JSON_DEPTH:
            raise TrustedValidationError(f"{label} exceeds JSON nesting limit")
        if isinstance(item, Mapping):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def canonical_json_bytes(value: Any, *, newline: bool = False, indent: int | None = None) -> bytes:
    """Return the frozen Python canonical JSON representation.

    The event protocol uses compact JSON.  Evaluation output uses the same
    key and number rules with a configured indentation and exactly one LF.
    """

    normalized = _json_compatible(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )
    if newline:
        text += "\n"
    return text.encode("utf-8")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TrustedValidationError("non-finite Decimal is forbidden")
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise TrustedValidationError("non-finite float is forbidden")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _issue(code: str, message: str) -> Issue:
    return Issue(code=code, message=message[:500])


def _safe_relative_path(value: Any) -> str | None:
    if type(value) is not str or not value or len(value) > 500 or "\x00" in value:
        return None
    if PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive:
        return None
    if value.startswith(("/", "\\", "//")):
        return None
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return "/".join(parts)


def _resolve_inside(root: Path, relative: str) -> Path | None:
    safe = _safe_relative_path(relative)
    if safe is None:
        return None
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    try:
        root_resolved = root.resolve(strict=True)
        cursor = root
        for part in PurePosixPath(safe).parts:
            cursor = cursor / part
            if cursor.exists() and _is_reparse_point(cursor):
                return None
        resolved = candidate.resolve(strict=False)
        if os.path.commonpath((str(root_resolved), str(resolved))) != str(root_resolved):
            return None
    except (OSError, ValueError):
        return None
    return candidate


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _load_relative_json(root: Path, relative: str, label: str) -> dict[str, Any]:
    path = _resolve_inside(root, relative)
    if path is None or not path.is_file():
        raise TrustedValidationError(f"{label} is missing or outside project")
    value = load_json(path, label=label)
    if type(value) is not dict:
        raise TrustedValidationError(f"{label} must be an object")
    return value


def _check_declared_hash(
    root: Path,
    descriptor: Mapping[str, Any],
    label: str,
    issues: list[Issue],
) -> Path | None:
    relative = descriptor.get("path")
    safe = _safe_relative_path(relative)
    if safe is None:
        issues.append(_issue("CONFIG_PATH_UNSAFE", f"{label} path is unsafe"))
        return None
    path = _resolve_inside(root, safe)
    if path is None or not path.is_file():
        issues.append(_issue("CONFIG_FILE_MISSING", f"{label} file is missing"))
        return None
    expected = descriptor.get("sha256")
    if type(expected) is not str or not HEX64_RE.fullmatch(expected):
        issues.append(_issue("CONFIG_HASH_UNFROZEN", f"{label} SHA256 is not frozen"))
    elif sha256_file(path).upper() != expected.upper():
        issues.append(_issue("CONFIG_HASH_MISMATCH", f"{label} SHA256 mismatch"))
    return path


def load_trusted_configuration(config_path: Path) -> TrustedConfiguration:
    config_path = config_path.resolve(strict=True)
    config = load_yaml(config_path)
    project_root = config_path.parent.parent.resolve(strict=True)
    issues: list[Issue] = []

    if str(config.get("schema_version", "")) != "2.0":
        issues.append(_issue("CONFIG_SCHEMA", "task config schema_version must be 2.0"))
    task = _mapping(config.get("task"))
    protocol = _mapping(task.get("protocol"))
    if not task.get("id") or not task.get("case_id"):
        issues.append(_issue("CONFIG_TASK_ID", "task id and case id are required"))
    if not protocol.get("name") or not protocol.get("version"):
        issues.append(_issue("CONFIG_PROTOCOL", "protocol name and version are required"))

    scoring = _mapping(config.get("scoring"))
    categories = _mapping(scoring.get("categories"))
    try:
        category_total = sum(Decimal(str(value)) for value in categories.values())
        declared_total = Decimal(str(scoring.get("total_points")))
    except (InvalidOperation, TypeError, ValueError):
        category_total = Decimal("-1")
        declared_total = Decimal("100")
    if declared_total != Decimal("100") or category_total != Decimal("100"):
        issues.append(_issue("CONFIG_SCORING", "scoring categories must total exactly 100"))
    gates = scoring.get("eligibility_gates")
    if type(gates) is not list or set(gates) != REQUIRED_ELIGIBILITY_GATES or len(gates) != len(REQUIRED_ELIGIBILITY_GATES):
        issues.append(_issue("CONFIG_GATES", "eligibility gate set is incomplete or duplicated"))

    job_spec_path = _check_declared_hash(
        project_root, _mapping(task.get("job_spec")), "job spec", issues
    )
    job_spec: dict[str, Any] = {}
    if job_spec_path is not None:
        value = load_json(job_spec_path, label="job spec")
        if type(value) is dict:
            job_spec = value
        else:
            issues.append(_issue("JOB_SPEC_SCHEMA", "job spec must be an object"))
    if job_spec and str(job_spec.get("case_id", "")) != str(task.get("case_id", "")):
        issues.append(_issue("JOB_SPEC_CONFLICT", "job spec case id conflicts with task config"))

    _check_declared_hash(
        project_root, _mapping(task.get("case_manifest")), "case manifest", issues
    )

    registry_rel = _mapping(config.get("models")).get("registry_path")
    registry_path = _resolve_inside(project_root, registry_rel) if type(registry_rel) is str else None
    model_registry: dict[str, Any] = {}
    if registry_path is None or not registry_path.is_file():
        issues.append(_issue("MODEL_REGISTRY_MISSING", "model registry is missing"))
    else:
        model_registry = load_yaml(registry_path)
        registry_models = _mapping(model_registry.get("models"))
        actual_models = {
            name: _mapping(registry_models.get(name)).get("model")
            for name in FROZEN_MODELS
        }
        defaults = _mapping(model_registry.get("provider_defaults"))
        configured_model_policy = _mapping(config.get("models"))
        if (
            actual_models != FROZEN_MODELS
            or set(registry_models) != set(FROZEN_MODELS)
            or configured_model_policy.get("formal_order") != list(FROZEN_MODELS)
            or configured_model_policy.get("formal_execution_mode") != "REAL_API"
            or configured_model_policy.get("formal_native_mode") != "LIVE_XTB"
            or configured_model_policy.get("require_exact_response_model") is not True
            or configured_model_policy.get("allow_fallback") is not False
            or configured_model_policy.get("allow_environment_model_override") is not False
            or defaults.get("provider") != "siliconflow"
            or defaults.get("transport") != "openai_compatible"
            or defaults.get("base_url") != "https://api.siliconflow.cn/v1"
            or defaults.get("sdk_max_retries") != 0
        ):
            issues.append(_issue("MODEL_POLICY", "model registry or formal model policy differs from the frozen release"))

    contracts_rel = _mapping(config.get("tools")).get("contract_path")
    contracts_path = _resolve_inside(project_root, contracts_rel) if type(contracts_rel) is str else None
    tool_contracts: dict[str, Any] = {}
    submission_schema: dict[str, Any] = {}
    final_answer_artifact_schema: dict[str, Any] = {}
    if contracts_path is None or not contracts_path.is_file():
        issues.append(_issue("TOOL_CONTRACTS_MISSING", "tool contracts are missing"))
    else:
        value = load_json(contracts_path, label="tool contracts")
        if type(value) is dict:
            tool_contracts = value
            for item in value.get("tools", []):
                function = _mapping(_mapping(item).get("function"))
                if function.get("name") == "submit_final":
                    schema = function.get("parameters")
                    if type(schema) is dict:
                        submission_schema = schema
                    break
            artifact_schema = value.get("final_answer_artifact_schema")
            if type(artifact_schema) is dict:
                final_answer_artifact_schema = artifact_schema
        if not submission_schema:
            issues.append(_issue("FINAL_SCHEMA_MISSING", "submit_final schema is missing"))
        if (
            not final_answer_artifact_schema
            or final_answer_artifact_schema.get("type") != "object"
            or final_answer_artifact_schema.get("additionalProperties") is not False
        ):
            issues.append(
                _issue(
                    "FINAL_ARTIFACT_SCHEMA_MISSING",
                    "closed final-answer artifact schema is missing",
                )
            )

    configured_tools = set(
        value for value in _mapping(config.get("tools")).get("allowed", []) if type(value) is str
    )
    contracted_tools = {
        str(_mapping(_mapping(item).get("function")).get("name"))
        for item in tool_contracts.get("tools", [])
        if isinstance(item, Mapping)
    }
    if configured_tools != contracted_tools:
        issues.append(_issue("TOOL_CONTRACT_CONFLICT", "configured and contracted tool sets differ"))

    return TrustedConfiguration(
        config=config,
        project_root=project_root,
        config_path=config_path,
        job_spec=job_spec,
        model_registry=model_registry,
        tool_contracts=tool_contracts,
        submission_schema=submission_schema,
        final_answer_artifact_schema=final_answer_artifact_schema,
        issues=issues,
    )


def validate_json_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[Issue]:
    """Validate the JSON-Schema subset used by tool_contracts.json."""

    issues: list[Issue] = []
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        if not any(not validate_json_schema(value, _mapping(option), path) for option in any_of):
            issues.append(_issue("FINAL_SCHEMA", f"{path} matches no allowed schema"))
        return issues

    expected_type = schema.get("type")
    type_ok = True
    if expected_type == "object":
        type_ok = type(value) is dict
    elif expected_type == "array":
        type_ok = type(value) is list
    elif expected_type == "string":
        type_ok = type(value) is str
    elif expected_type == "integer":
        type_ok = type(value) is int
    elif expected_type == "number":
        type_ok = type(value) in {int, float, Decimal}
    elif expected_type == "boolean":
        type_ok = type(value) is bool
    elif expected_type == "null":
        type_ok = value is None
    if not type_ok:
        return [_issue("FINAL_SCHEMA", f"{path} has the wrong type")]

    if "const" in schema and value != schema.get("const"):
        issues.append(_issue("FINAL_SCHEMA", f"{path} does not equal the required constant"))
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        issues.append(_issue("FINAL_SCHEMA", f"{path} is not an allowed value"))

    if type(value) is dict:
        properties = _mapping(schema.get("properties"))
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    issues.append(_issue("FINAL_SCHEMA", f"{path}.{key} is required"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    issues.append(_issue("FINAL_SCHEMA", f"{path} has an unexpected field"))
        for key, child in value.items():
            if key in properties:
                issues.extend(validate_json_schema(child, _mapping(properties[key]), f"{path}.{key}"))
    elif type(value) is list:
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if type(minimum) is int and len(value) < minimum:
            issues.append(_issue("FINAL_SCHEMA", f"{path} has too few items"))
        if type(maximum) is int and len(value) > maximum:
            issues.append(_issue("FINAL_SCHEMA", f"{path} has too many items"))
        if schema.get("uniqueItems") is True:
            encoded = [canonical_json_bytes(item) for item in value]
            if len(set(encoded)) != len(encoded):
                issues.append(_issue("FINAL_SCHEMA", f"{path} items must be unique"))
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, child in enumerate(value):
                issues.extend(validate_json_schema(child, item_schema, f"{path}[{index}]"))
    elif type(value) is str:
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if type(minimum) is int and len(value) < minimum:
            issues.append(_issue("FINAL_SCHEMA", f"{path} is too short"))
        if type(maximum) is int and len(value) > maximum:
            issues.append(_issue("FINAL_SCHEMA", f"{path} is too long"))
        pattern = schema.get("pattern")
        if type(pattern) is str and re.fullmatch(pattern, value) is None:
            issues.append(_issue("FINAL_SCHEMA", f"{path} does not match the required pattern"))
    elif type(value) in {int, float, Decimal} and type(value) is not bool:
        number = Decimal(str(value))
        if "minimum" in schema and number < Decimal(str(schema["minimum"])):
            issues.append(_issue("FINAL_SCHEMA", f"{path} is below the minimum"))
        if "maximum" in schema and number > Decimal(str(schema["maximum"])):
            issues.append(_issue("FINAL_SCHEMA", f"{path} is above the maximum"))
    return issues


def load_trajectory(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = _bounded_read(path, MAX_TRAJECTORY_BYTES, "trajectory")
    if b"\x00" in raw:
        raise TrustedValidationError("trajectory contains NUL bytes")
    lines = raw.splitlines()
    if not lines:
        raise TrustedValidationError("trajectory is empty")
    if len(lines) > MAX_TRAJECTORY_LINES:
        raise TrustedValidationError("trajectory contains too many events")
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        if not line.strip():
            raise TrustedValidationError(f"trajectory line {index} is blank")
        if len(line) > MAX_TRAJECTORY_LINE_BYTES:
            raise TrustedValidationError(f"trajectory line {index} is too large")
        try:
            event = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_duplicate_object,
                parse_constant=_reject_constant,
                parse_float=Decimal,
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError, DuplicateKeyError) as exc:
            raise TrustedValidationError(f"invalid trajectory event at line {index}") from exc
        if type(event) is not dict:
            raise TrustedValidationError(f"trajectory event {index} must be an object")
        _validate_json_complexity(event, f"trajectory event {index}")
        events.append(event)
    return events, raw


def _parse_timestamp(value: Any) -> datetime | None:
    if type(value) is not str or len(value) > 64:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _contains_forbidden_key(value: Any) -> bool:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            if any(str(key) in HIDDEN_REASONING_KEYS for key in item):
                return True
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _serialized_has_secret(value: Any) -> bool:
    try:
        text = canonical_json_bytes(value).decode("utf-8")
    except (UnicodeError, TrustedValidationError):
        return True
    return any(pattern.search(text) is not None for pattern in SECRET_PATTERNS)


def _serialized_has_absolute_path(value: Any) -> bool:
    try:
        text = canonical_json_bytes(value).decode("utf-8")
    except (UnicodeError, TrustedValidationError):
        return True
    return WINDOWS_ABSOLUTE_RE.search(text) is not None or POSIX_ABSOLUTE_RE.search(text) is not None


def _event_hash(event: Mapping[str, Any]) -> str:
    material = dict(event)
    material.pop("event_hash", None)
    return sha256_bytes(canonical_json_bytes(material))


def _run_start_modes(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    run_type = str(payload.get("run_type", "FORMAL_MODEL"))
    execution_mode = str(payload.get("execution_mode", ""))
    native_mode = str(payload.get("native_mode", ""))
    return run_type, execution_mode, native_mode


def _identity_expected_model(config: TrustedConfiguration, logical_model: str | None) -> str | None:
    if not logical_model:
        return None
    models = _mapping(config.model_registry.get("models"))
    entry = _mapping(models.get(logical_model))
    model = entry.get("model")
    return model if type(model) is str else None


def validate_trajectory(
    trajectory_path: Path,
    config: TrustedConfiguration,
    final_answer: Mapping[str, Any] | None = None,
) -> TraceFacts:
    events, raw = load_trajectory(trajectory_path)
    issues: list[Issue] = []
    expected_protocol = config.protocol
    genesis = str(_mapping(config.config.get("determinism")).get("trajectory_genesis_hash", ""))
    if not HEX64_RE.fullmatch(genesis):
        genesis = "0" * 64
        issues.append(_issue("CONFIG_GENESIS", "trajectory genesis hash is invalid"))

    hash_chain_valid = True
    protocol_valid = True
    previous = genesis.upper()
    run_ids: set[str] = set()
    timestamps: list[datetime] = []
    terminal_positions: list[int] = []
    final_positions: list[int] = []
    hidden_reasoning = False
    secret_finding = False
    trace_absolute_path = False
    unknown_event = False

    for index, event in enumerate(events, 1):
        extra = set(event) - ENVELOPE_KEYS
        missing = ENVELOPE_KEYS - set(event)
        if extra or missing:
            issues.append(_issue("TRACE_ENVELOPE", f"event {index} has a non-canonical envelope"))
            hash_chain_valid = False
        if event.get("schema_version") != "1.0":
            issues.append(_issue("TRACE_SCHEMA", f"event {index} has the wrong schema version"))
            protocol_valid = False
        protocol = _mapping(event.get("protocol"))
        if set(protocol) != {"name", "version"} or (
            str(protocol.get("name", "")), str(protocol.get("version", ""))
        ) != expected_protocol:
            issues.append(_issue("TRACE_PROTOCOL", f"event {index} does not match the configured protocol"))
            protocol_valid = False
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence != index:
            issues.append(_issue("TRACE_SEQUENCE", f"event {index} has a non-contiguous sequence"))
            hash_chain_valid = False
        run_id = event.get("run_id")
        if type(run_id) is not str or not run_id or len(run_id) > 128:
            issues.append(_issue("TRACE_RUN_ID", f"event {index} has an invalid run id"))
        else:
            run_ids.add(run_id)
        timestamp = _parse_timestamp(event.get("timestamp_utc"))
        if timestamp is None:
            issues.append(_issue("TRACE_TIMESTAMP", f"event {index} has an invalid UTC timestamp"))
        else:
            timestamps.append(timestamp)
            if len(timestamps) > 1 and timestamps[-1] < timestamps[-2]:
                issues.append(_issue("TRACE_TIMESTAMP_ORDER", "trajectory timestamps move backwards"))
        event_name = event.get("event")
        if event_name not in ALLOWED_EVENTS:
            issues.append(_issue("TRACE_EVENT_TYPE", f"event {index} has an unsupported type"))
            unknown_event = True
        if event_name in TERMINAL_EVENTS:
            terminal_positions.append(index)
        if event_name == "final_submitted":
            final_positions.append(index)
        if type(event.get("payload")) is not dict:
            issues.append(_issue("TRACE_PAYLOAD", f"event {index} payload must be an object"))
        if str(event.get("prev_event_hash", "")).upper() != previous:
            issues.append(_issue("TRACE_PREV_HASH", f"event {index} previous hash mismatch"))
            hash_chain_valid = False
        declared_hash = event.get("event_hash")
        computed_hash = _event_hash(event)
        if type(declared_hash) is not str or declared_hash.upper() != computed_hash:
            issues.append(_issue("TRACE_EVENT_HASH", f"event {index} hash mismatch"))
            hash_chain_valid = False
        previous = str(declared_hash).upper() if type(declared_hash) is str else computed_hash
        hidden_reasoning = hidden_reasoning or _contains_forbidden_key(event)
        secret_finding = secret_finding or _serialized_has_secret(event)
        # Free-form assistant prose may mention a path as documentation.  Path
        # safety is enforced below only for tool path arguments, source
        # geometry objects, prepared argv, and persisted artifacts.

    lifecycle_valid = (
        bool(events)
        and events[0].get("event") == "run_start"
        and len(terminal_positions) == 1
        and terminal_positions[0] == len(events)
        and len(run_ids) == 1
        and not unknown_event
        and (
            events[-1].get("event") == "run_failure"
            or (
                events[-1].get("event") == "run_complete"
                and len(final_positions) == 1
                and final_positions[0] == len(events) - 1
            )
        )
    )
    if not lifecycle_valid:
        issues.append(_issue("TRACE_LIFECYCLE", "run_start to single terminal lifecycle is invalid"))
    if hidden_reasoning:
        issues.append(_issue("HIDDEN_REASONING", "trajectory contains a forbidden hidden-reasoning field"))
    if secret_finding:
        issues.append(_issue("TRACE_SECRET", "trajectory contains a key-like secret"))

    start_payload = _mapping(events[0].get("payload")) if events else {}
    run_type, execution_mode, native_mode = _run_start_modes(start_payload)
    identity_exempt = run_type in {
        "BASELINE",
        "BASELINE_REPLAY",
        "REPLAY",
        "FORMAL_REPLAY",
        "HARBOR_PROTOCOL",
    }
    logical_model = start_payload.get("logical_model")
    logical_model = logical_model if type(logical_model) is str else None
    requested_model = start_payload.get("requested_model")
    requested_model = requested_model if type(requested_model) is str else None
    configured_model = _identity_expected_model(config, logical_model)

    if identity_exempt:
        mode_truthful = execution_mode in {"BASELINE", "REPLAY", "HARBOR"} and native_mode == "LIVE_XTB"
    else:
        models_cfg = _mapping(config.config.get("models"))
        mode_truthful = (
            execution_mode == str(models_cfg.get("formal_execution_mode", "REAL_API"))
            and native_mode == str(models_cfg.get("formal_native_mode", "LIVE_XTB"))
            and start_payload.get("provider") == "siliconflow"
            and start_payload.get("transport") == "openai_compatible"
        )

    tool_calls_by_id: dict[str, ToolCallFact] = {}
    tool_calls: list[ToolCallFact] = []
    orphan_results = False
    actual_models: list[str] = []
    request_ids: list[str] = []
    provider_retries = 0
    agent_turns = 0
    prepared: list[dict[str, Any]] = []
    native_events: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    prepared_contracts_valid = True
    native_contracts_valid = True
    validation_contracts_valid = True
    final_submitted: dict[str, Any] | None = None
    budget_states: list[dict[str, Any]] = []
    oracle_accessed = False
    unsafe = False
    fallback_or_manual = False
    scheduler_completed = False

    forbidden_roots = {
        str(item).casefold()
        for item in _mapping(config.config.get("security")).get("forbidden_read_roots", [])
        if type(item) is str
    }

    for index, event in enumerate(events, 1):
        name = event.get("event")
        payload = _mapping(event.get("payload"))
        if name == "provider_retry":
            provider_retries += 1
        elif name == "provider_response_metadata":
            agent_turns += 1
            actual = payload.get("actual_model")
            request_id = payload.get("request_id")
            if type(actual) is str and actual:
                actual_models.append(actual)
            else:
                issues.append(_issue("MODEL_ID_MISSING", f"provider response at event {index} lacks actual_model"))
            if type(request_id) is str and request_id:
                request_ids.append(request_id)
            else:
                issues.append(_issue("REQUEST_ID_MISSING", f"provider response at event {index} lacks request_id"))
            transport_marker = str(payload.get("transport", "")).casefold()
            if (
                payload.get("fallback") is True
                or "mock" in transport_marker
                or "scripted" in transport_marker
                or "stub" in transport_marker
            ):
                fallback_or_manual = True
        elif name == "tool_call":
            call_id = payload.get("call_id")
            tool_name = payload.get("name")
            arguments = payload.get("arguments")
            if type(call_id) is not str or not call_id or len(call_id) > 128:
                issues.append(_issue("TOOL_CALL_ID", f"tool call at event {index} has an invalid id"))
                continue
            if call_id in tool_calls_by_id:
                issues.append(_issue("TOOL_CALL_DUPLICATE", f"tool call id is duplicated at event {index}"))
                continue
            if type(tool_name) is not str or tool_name not in config.allowed_tools:
                issues.append(_issue("TOOL_NOT_ALLOWED", f"tool call at event {index} is not allowed"))
            if type(arguments) is not dict:
                issues.append(_issue("TOOL_ARGUMENTS", f"tool call at event {index} arguments must be an object"))
                arguments = {}
            fact = ToolCallFact(index, call_id, str(tool_name), dict(arguments))
            tool_calls_by_id[call_id] = fact
            tool_calls.append(fact)
            for key, argument in arguments.items():
                if key in PATH_ARGUMENTS or key.endswith("_path"):
                    safe = _safe_relative_path(argument)
                    if safe is None:
                        unsafe = True
                    else:
                        root_part = safe.split("/", 1)[0].casefold()
                        if root_part in forbidden_roots:
                            oracle_accessed = oracle_accessed or root_part == "evaluator_private"
                            unsafe = True
                if key == "source_geometry" and isinstance(argument, Mapping):
                    source_path = argument.get("path")
                    safe = _safe_relative_path(source_path)
                    if safe is None:
                        unsafe = True
                    elif safe.split("/", 1)[0].casefold() in forbidden_roots:
                        unsafe = True
            if tool_name in READ_TOOLS:
                path_value = arguments.get("path")
                safe = _safe_relative_path(path_value)
                if safe and "evaluator_private" in {part.casefold() for part in PurePosixPath(safe).parts}:
                    oracle_accessed = True
        elif name == "tool_result":
            call_id = payload.get("call_id")
            tool_name = payload.get("name")
            if type(call_id) is not str or call_id not in tool_calls_by_id:
                issues.append(_issue("TOOL_RESULT_ORPHAN", f"tool result at event {index} has no prior call"))
                orphan_results = True
                continue
            fact = tool_calls_by_id[call_id]
            if fact.result_sequence is not None:
                issues.append(_issue("TOOL_RESULT_DUPLICATE", f"tool result for {call_id[:40]} is duplicated"))
                orphan_results = True
                continue
            if tool_name != fact.name:
                issues.append(_issue("TOOL_RESULT_NAME", f"tool result at event {index} has the wrong tool name"))
                orphan_results = True
            fact.result_sequence = index
            fact.ok = payload.get("ok") is True
            fact.result = payload.get("result")
            if fact.name == "inspect_scheduler_record" and fact.ok:
                result = _mapping(fact.result)
                scheduler_completed = str(result.get("state", result.get("status", ""))).upper() == "COMPLETED"
        elif name == "attempt_prepared":
            prepared.append(dict(payload))
            required = {
                "attempt_id",
                "phase",
                "state",
                "source_geometry",
                "prepared_command_path",
                "input_manifest_path",
            }
            source = _mapping(payload.get("source_geometry"))
            contract_ok = (
                required.issubset(payload)
                and type(payload.get("attempt_id")) is str
                and ATTEMPT_RE.fullmatch(str(payload.get("attempt_id"))) is not None
                and payload.get("phase") in {"geometry_optimization", "hessian"}
                and payload.get("state") == "PREPARED"
                and {"path", "frame_index", "sha256", "provenance_state"}.issubset(source)
                and _safe_relative_path(source.get("path")) is not None
                and type(source.get("frame_index")) in {int, type(None)}
                and type(source.get("sha256")) is str
                and HEX64_RE.fullmatch(str(source.get("sha256"))) is not None
                and type(source.get("provenance_state")) is str
                and _safe_relative_path(payload.get("prepared_command_path")) is not None
                and _safe_relative_path(payload.get("input_manifest_path")) is not None
            )
            if not contract_ok:
                issues.append(_issue("ATTEMPT_EVENT_CONTRACT", f"attempt_prepared payload is invalid at event {index}"))
                prepared_contracts_valid = False
        elif name == "native_xtb_execution":
            native_events.append(dict(payload))
            required = {
                "attempt_id",
                "phase",
                "source_type",
                "argv",
                "shell",
                "process_exit_code",
                "timed_out",
                "execution_summary_path",
                "artifact_manifest_path",
                "artifact_hashes",
            }
            hashes = payload.get("artifact_hashes")
            contract_ok = (
                required.issubset(payload)
                and type(payload.get("attempt_id")) is str
                and ATTEMPT_RE.fullmatch(str(payload.get("attempt_id"))) is not None
                and payload.get("phase") in {"geometry_optimization", "hessian"}
                and payload.get("source_type") == "NATIVE_XTB"
                and _argv_safe(payload.get("argv"))
                and payload.get("shell") is False
                and type(payload.get("process_exit_code")) in {int, type(None)}
                and type(payload.get("timed_out")) is bool
                and _safe_relative_path(payload.get("execution_summary_path")) is not None
                and _safe_relative_path(payload.get("artifact_manifest_path")) is not None
                and type(hashes) is dict
                and all(
                    type(key) is str
                    and _safe_relative_path(key) is not None
                    and type(value) is str
                    and HEX64_RE.fullmatch(value) is not None
                    for key, value in _mapping(hashes).items()
                )
            )
            if not contract_ok:
                issues.append(_issue("NATIVE_EVENT_CONTRACT", f"native_xtb_execution payload is invalid at event {index}"))
                native_contracts_valid = False
        elif name == "attempt_validated":
            validations.append(dict(payload))
            required = {
                "attempt_id",
                "phase",
                "classification",
                "valid",
                "optimization_converged",
                "hessian_completed",
                "current_artifacts",
                "scientific_invariants_preserved",
                "source_provenance",
                "validation_path",
            }
            contract_ok = (
                required.issubset(payload)
                and type(payload.get("attempt_id")) is str
                and ATTEMPT_RE.fullmatch(str(payload.get("attempt_id"))) is not None
                and payload.get("phase") in {"geometry_optimization", "hessian"}
                and type(payload.get("classification")) is str
                and all(
                    type(payload.get(key)) is bool
                    for key in (
                        "valid",
                        "optimization_converged",
                        "hessian_completed",
                        "current_artifacts",
                        "scientific_invariants_preserved",
                    )
                )
                and type(payload.get("source_provenance")) in {str, dict}
                and _safe_relative_path(payload.get("validation_path")) is not None
            )
            if not contract_ok:
                issues.append(_issue("VALIDATION_EVENT_CONTRACT", f"attempt_validated payload is invalid at event {index}"))
                validation_contracts_valid = False
        elif name == "final_submitted":
            if final_submitted is not None:
                issues.append(_issue("FINAL_DUPLICATE", "more than one final_submitted event exists"))
            final_submitted = dict(payload)
        elif name == "budget_state":
            budget_states.append(dict(payload))

        if payload.get("manual_rewrite") is True or payload.get("source_type") in {"MANUAL", "MOCK", "SCRIPTED"}:
            fallback_or_manual = True
        if payload.get("fallback") is True:
            fallback_or_manual = True

    for fact in tool_calls:
        if fact.result_sequence is None:
            issues.append(_issue("TOOL_RESULT_MISSING", f"tool call {fact.call_id[:40]} has no result"))
            orphan_results = True

    successful_calls = sum(fact.ok for fact in tool_calls)
    budget = config.budget
    max_tool_calls = _strict_int(budget.get("max_tool_calls"), 0)
    min_successful = _strict_int(budget.get("min_successful_tool_calls"), 0)
    max_turns = _strict_int(budget.get("max_agent_turns"), 0)
    max_attempts = _strict_int(budget.get("max_execution_attempts"), 0)
    max_retries = _strict_int(budget.get("max_provider_retries"), 0)
    provider_retry_event_ceiling = _provider_retry_event_ceiling(budget)
    infra_degraded = provider_retries > max_retries
    prepared_ids = [payload.get("attempt_id") for payload in prepared]
    valid_prepared_ids = {
        item for item in prepared_ids if type(item) is str and ATTEMPT_RE.fullmatch(item)
    }
    budgets_respected = (
        len(tool_calls) <= max_tool_calls
        and successful_calls >= min_successful
        and agent_turns <= max_turns
        and len(valid_prepared_ids) <= max_attempts
        and _provider_retries_within_budget(provider_retries, budget)
    )
    if len(timestamps) >= 2:
        elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
        total_timeout = _strict_int(budget.get("total_timeout_seconds"), 0)
        if elapsed < 0 or elapsed > total_timeout:
            budgets_respected = False
            issues.append(_issue("TOTAL_TIMEOUT", "trajectory elapsed time exceeds the configured timeout"))
    if not budgets_respected:
        issues.append(_issue("BUDGET_EXCEEDED", "trusted event counts violate configured budgets"))
    if len(valid_prepared_ids) != len(prepared_ids) or len(valid_prepared_ids) != len(prepared):
        issues.append(_issue("ATTEMPT_EVENT_INVALID", "prepared attempt ids are invalid or duplicated"))

    budget_state_valid = bool(budget_states)
    previous_snapshot = {
        "tool_calls": -1,
        "successful_tool_calls": -1,
        "agent_turns": -1,
        "execution_attempts": -1,
        "provider_retries": -1,
    }
    for snapshot in budget_states:
        for key in previous_snapshot:
            value = snapshot.get(key)
            if type(value) is not int or value < previous_snapshot[key]:
                budget_state_valid = False
            elif value >= 0:
                previous_snapshot[key] = value
        elapsed_value = snapshot.get("elapsed_seconds")
        if type(elapsed_value) not in {int, float, Decimal} or Decimal(str(elapsed_value)) < 0:
            budget_state_valid = False
    expected_final_budget = {
        "tool_calls": len(tool_calls),
        "successful_tool_calls": successful_calls,
        "agent_turns": agent_turns,
        "execution_attempts": len(valid_prepared_ids),
        "provider_retries": provider_retries,
    }
    if not budget_states or any(
        budget_states[-1].get(key) != value
        for key, value in expected_final_budget.items()
    ):
        budget_state_valid = False
    if not budget_state_valid:
        budgets_respected = False
        issues.append(_issue("BUDGET_STATE", "budget_state events are missing or disagree with trusted counts"))

    max_lines = _strict_int(budget.get("max_text_read_lines_per_call"), 0)
    max_bytes = _strict_int(budget.get("max_text_read_bytes_per_call"), 0)
    direct_reads: list[ToolCallFact] = []
    decisive = False
    for fact in tool_calls:
        if not fact.ok or fact.name not in {"read_text_range", "parse_xtb_log"}:
            continue
        path_value = fact.arguments.get("path")
        safe = _safe_relative_path(path_value)
        if safe is None:
            continue
        lowered = safe.casefold()
        if fact.name == "read_text_range":
            start = _strict_int(fact.arguments.get("start_line"), -1)
            end = _strict_int(fact.arguments.get("end_line"), -1)
            if start < 1 or end < start or end - start + 1 > max_lines:
                issues.append(_issue("READ_LIMIT", f"read {fact.call_id[:40]} exceeds the line limit"))
                budgets_respected = False
                continue
            result_map = _mapping(fact.result)
            content = result_map.get("content", result_map.get("text"))
            if type(content) is str and len(content.encode("utf-8")) > max_bytes:
                issues.append(_issue("READ_LIMIT", f"read {fact.call_id[:40]} exceeds the byte limit"))
                budgets_respected = False
                continue
            direct_reads.append(fact)
            if INITIAL_ATTEMPT.casefold() in lowered and lowered.endswith((".log", ".out")):
                if type(content) is str and bool(content.strip()):
                    decisive = True
        else:
            result_map = _mapping(fact.result)
            latest = _mapping(result_map.get("latest_segment"))
            if INITIAL_ATTEMPT.casefold() in lowered and (
                bool(latest.get("facts")) or latest.get("state") not in {None, "UNKNOWN"}
            ):
                decisive = True

    if bool(budget.get("require_read_text_evidence")) and not any(
        fact.name == "read_text_range" for fact in direct_reads
    ):
        decisive = False

    exact_identity = identity_exempt or (
        bool(actual_models)
        and bool(request_ids)
        and all(request_id != "UNAVAILABLE" for request_id in request_ids)
        and requested_model is not None
        and configured_model == requested_model
        and all(actual == requested_model for actual in actual_models)
        and len(actual_models) == len(request_ids)
    )
    if not exact_identity:
        issues.append(_issue("MODEL_IDENTITY", "requested and actual model identity is not exact"))
    if not mode_truthful:
        issues.append(_issue("RUN_MODE", "run type, API mode, or native mode is not truthful"))
    if orphan_results:
        lifecycle_valid = False

    if final_answer is not None:
        submitted_object = None
        if final_submitted is not None:
            candidate = final_submitted.get("final_answer")
            submitted_object = candidate if isinstance(candidate, Mapping) else final_submitted
        submit_calls = [fact for fact in tool_calls if fact.name == "submit_final" and fact.ok]
        if len(submit_calls) != 1 or submit_calls[0].arguments != dict(final_answer):
            issues.append(
                _issue(
                    "FINAL_TRACE_MISMATCH",
                    "submit_final arguments do not match final_answer.json model_submission",
                )
            )
            lifecycle_valid = False
        if submitted_object is None or dict(submitted_object) != dict(final_answer):
            issues.append(
                _issue(
                    "FINAL_EVENT_MISMATCH",
                    "final_submitted payload does not match final_answer.json model_submission",
                )
            )
            lifecycle_valid = False

    terminal = events[-1].get("event") if events and events[-1].get("event") in TERMINAL_EVENTS else None
    return TraceFacts(
        events=events,
        issues=issues,
        trace_sha256=sha256_bytes(raw),
        final_event_hash=str(events[-1].get("event_hash", "")).upper() if events else "",
        lifecycle_valid=lifecycle_valid,
        hash_chain_valid=hash_chain_valid,
        protocol_valid=protocol_valid,
        terminal_event=terminal,
        run_id=next(iter(run_ids)) if len(run_ids) == 1 else None,
        run_type=run_type,
        execution_mode=execution_mode,
        native_mode=native_mode,
        logical_model=logical_model,
        requested_model=requested_model,
        actual_models=actual_models,
        provider_request_ids=request_ids,
        exact_model_identity=exact_identity,
        identity_exempt=identity_exempt,
        mode_truthful=mode_truthful,
        tool_calls=tool_calls,
        successful_tool_calls=successful_calls,
        agent_turns=agent_turns,
        provider_retries=provider_retries,
        provider_retry_event_ceiling=provider_retry_event_ceiling,
        provider_retry_budget_scope="per_provider_request",
        infra_degraded=infra_degraded,
        attempt_prepared=prepared,
        native_executions=native_events,
        attempt_validations=validations,
        attempt_event_contracts_valid=prepared_contracts_valid,
        native_event_contracts_valid=native_contracts_valid,
        validation_event_contracts_valid=validation_contracts_valid,
        final_submitted=final_submitted,
        budgets_respected=budgets_respected,
        budget_state_valid=budget_state_valid,
        direct_log_reads=direct_reads,
        decisive_log_evidence_read=decisive,
        scheduler_completed_observed=scheduler_completed,
        oracle_accessed=oracle_accessed,
        unsafe_path_or_argv=unsafe,
        fallback_or_manual_marker=fallback_or_manual,
        secret_finding=secret_finding,
        hidden_reasoning_present=hidden_reasoning,
    )


def _strict_int(value: Any, default: int) -> int:
    return value if type(value) is int else default


def _provider_retry_event_ceiling(budget: Mapping[str, Any]) -> int:
    """Return the case-level ceiling implied by the runner's per-request retry limit."""

    max_turns = _strict_int(budget.get("max_agent_turns"), 0)
    max_retries = _strict_int(budget.get("max_provider_retries"), 0)
    return max_turns * max_retries


def _provider_retries_within_budget(
    provider_retries: int,
    budget: Mapping[str, Any],
) -> bool:
    """Validate cumulative retry events without changing their per-request semantics."""

    return 0 <= provider_retries <= _provider_retry_event_ceiling(budget)


def _scan_artifact_tree(root: Path) -> tuple[list[dict[str, Any]], list[Issue], bool, bool, bool]:
    issues: list[Issue] = []
    files: list[dict[str, Any]] = []
    total = 0
    secret = False
    absolute = False
    oracle = False
    root = root.resolve(strict=True)
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if _is_reparse_point(path):
            issues.append(_issue("ARTIFACT_REPARSE_POINT", f"reparse point is forbidden: {relative[:200]}"))
            continue
        if path.is_dir():
            if "evaluator_private" in {part.casefold() for part in PurePosixPath(relative).parts}:
                oracle = True
            continue
        if not path.is_file():
            issues.append(_issue("ARTIFACT_SPECIAL_FILE", f"non-regular artifact is forbidden: {relative[:200]}"))
            continue
        if len(files) >= MAX_ARTIFACT_FILES:
            raise TrustedValidationError("artifact workspace contains too many files")
        size = path.stat().st_size
        total += size
        if total > MAX_ARTIFACT_BYTES:
            raise TrustedValidationError("artifact workspace is too large")
        parts_folded = {part.casefold() for part in PurePosixPath(relative).parts}
        forbidden_private = (
            "evaluator_private" in parts_folded
            or path.name.casefold() == "oracle.json"
        )
        if forbidden_private:
            # Presence is a hard failure.  Do not open, hash, or parse private
            # evaluator material even when a caller places it under artifacts.
            oracle = True
            files.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": None,
                    "forbidden_private_file": True,
                }
            )
            continue
        digest = sha256_file(path)
        files.append({"path": relative, "size_bytes": size, "sha256": digest})
        if size <= MAX_SCANNED_TEXT_BYTES:
            raw = path.read_bytes()
            if b"\x00" not in raw:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = ""
                if text:
                    secret = secret or any(pattern.search(text) is not None for pattern in SECRET_PATTERNS)
                    absolute = absolute or WINDOWS_ABSOLUTE_RE.search(text) is not None or POSIX_ABSOLUTE_RE.search(text) is not None
    return files, issues, secret, absolute, oracle


def _workspace_hash(files: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(files)))


def _verify_workspace_case_manifest(
    root: Path,
    files: Sequence[Mapping[str, Any]],
    config: TrustedConfiguration,
    issues: list[Issue],
) -> bool:
    """Verify frozen case inventory while allowing generated restart attempts."""

    manifest_path = root / "case_manifest.json"
    descriptor = _mapping(_mapping(config.config.get("task")).get("case_manifest"))
    expected_manifest_hash = descriptor.get("sha256")
    valid = True
    if (
        not manifest_path.is_file()
        or type(expected_manifest_hash) is not str
        or not HEX64_RE.fullmatch(expected_manifest_hash)
        or sha256_file(manifest_path) != expected_manifest_hash.upper()
    ):
        issues.append(
            _issue(
                "WORKSPACE_CASE_MANIFEST_HASH",
                "workspace case manifest differs from the frozen configuration",
            )
        )
        return False
    try:
        manifest = load_json(manifest_path, label="workspace case manifest")
    except TrustedValidationError:
        issues.append(
            _issue("WORKSPACE_CASE_MANIFEST_SCHEMA", "workspace case manifest is invalid")
        )
        return False
    entries = manifest.get("files") if isinstance(manifest, Mapping) else None
    if type(entries) is not list:
        issues.append(
            _issue("WORKSPACE_CASE_MANIFEST_SCHEMA", "workspace case inventory is missing")
        )
        return False

    declared: set[str] = set()
    file_index = {str(entry.get("path")): entry for entry in files}
    for entry in entries:
        if type(entry) is not dict:
            valid = False
            continue
        relative = _safe_relative_path(entry.get("path"))
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if (
            relative is None
            or relative in declared
            or type(expected_hash) is not str
            or not HEX64_RE.fullmatch(expected_hash)
            or type(expected_size) is not int
            or expected_size < 0
        ):
            valid = False
            continue
        declared.add(relative)
        actual = file_index.get(relative)
        if (
            not isinstance(actual, Mapping)
            or actual.get("sha256") != expected_hash.upper()
            or actual.get("size_bytes") != expected_size
        ):
            valid = False

    actual_frozen: set[str] = set()
    for entry in files:
        relative = str(entry.get("path", ""))
        if relative == "case_manifest.json":
            continue
        parts = PurePosixPath(relative).parts
        generated_attempt = (
            len(parts) >= 2
            and parts[0] == "attempts"
            and ATTEMPT_RE.fullmatch(parts[1]) is not None
            and parts[1] != INITIAL_ATTEMPT
        )
        if not generated_attempt:
            actual_frozen.add(relative)
    if actual_frozen != declared:
        valid = False
    if not valid:
        issues.append(
            _issue(
                "WORKSPACE_CASE_INVENTORY",
                "frozen workspace inventory, size, or hash differs from case_manifest.json",
            )
        )
    return valid


def _derive_initial_case_facts(
    root: Path, files: Sequence[Mapping[str, Any]], issues: list[Issue]
) -> tuple[bool, bool, bool]:
    scheduler_completed = False
    maximum_cycles_failure = False
    legacy_present = False
    for entry in files:
        relative = str(entry["path"])
        folded = relative.casefold()
        parts = {part.casefold() for part in PurePosixPath(relative).parts}
        legacy_present = legacy_present or (
            "legacy" in parts and PurePosixPath(relative).suffix.casefold() == ".xyz"
        )
        if INITIAL_ATTEMPT.casefold() not in folded:
            continue
        path = _resolve_inside(root, relative)
        if path is None:
            continue
        if PurePosixPath(relative).name.casefold() == "scheduler.json":
            try:
                value = load_json(path, label="initial scheduler record")
            except TrustedValidationError:
                issues.append(_issue("INITIAL_SCHEDULER", "initial scheduler record is invalid"))
                continue
            if isinstance(value, Mapping):
                state = value.get("state", value.get("status"))
                scheduler_completed = str(state).upper() == "COMPLETED"
        elif path.suffix.casefold() in {".log", ".out"} and path.stat().st_size <= MAX_SCANNED_TEXT_BYTES:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if OPT_FAILURE_RE.search(text) and OPT_CONVERGED_RE.search(text) is None:
                maximum_cycles_failure = True
    return scheduler_completed, maximum_cycles_failure, legacy_present


def _attempt_id_from_path(relative: str) -> str | None:
    for part in PurePosixPath(relative).parts:
        if ATTEMPT_RE.fullmatch(part):
            return part
    return None


def _record_relative_path(
    files: Sequence[Mapping[str, Any]], attempt_id: str, filename: str
) -> str | None:
    matches = [
        str(entry["path"])
        for entry in files
        if _attempt_id_from_path(str(entry["path"])) == attempt_id
        and PurePosixPath(str(entry["path"])).name == filename
    ]
    return matches[0] if len(matches) == 1 else None


def _verify_canonical_attempt_files(
    root: Path,
    files: Sequence[Mapping[str, Any]],
    attempt_ids: Sequence[str],
    issues: list[Issue],
) -> bool:
    """Require canonical release names and byte-identical compatibility aliases."""

    groups = (
        ("attempt.json", "prepared_command.json", "prepared.json"),
        ("input_manifest.json", "provenance.json"),
        ("execution_summary.json", "run_record.json"),
        ("stdout.txt", "stdout.log"),
        ("stderr.txt", "stderr.log"),
    )
    valid = bool(attempt_ids)
    for attempt_id in attempt_ids:
        for names in groups:
            paths = [_record_relative_path(files, attempt_id, name) for name in names]
            if any(relative is None for relative in paths):
                issues.append(
                    _issue(
                        "CANONICAL_ATTEMPT_FILE",
                        f"canonical/compatibility files are incomplete for {attempt_id}",
                    )
                )
                valid = False
                continue
            resolved = [_resolve_inside(root, str(relative)) for relative in paths]
            if any(path is None or not path.is_file() for path in resolved):
                valid = False
                continue
            digests = {sha256_file(path) for path in resolved if path is not None}
            if len(digests) != 1:
                issues.append(
                    _issue(
                        "CANONICAL_ALIAS_MISMATCH",
                        f"canonical and compatibility files differ for {attempt_id}",
                    )
                )
                valid = False
    return valid


def _read_artifact_jsons(root: Path, files: Sequence[Mapping[str, Any]], issues: list[Issue]) -> dict[str, dict[str, dict[str, Any]]]:
    kinds = {
        "prepared.json": "prepared",
        "provenance.json": "provenance",
        "run_record.json": "run_record",
        "validation.json": "validation",
        "artifact_manifest.json": "manifest",
    }
    records: dict[str, dict[str, dict[str, Any]]] = {kind: {} for kind in kinds.values()}
    for entry in files:
        relative = str(entry["path"])
        name = PurePosixPath(relative).name
        if name not in kinds:
            continue
        attempt_id = _attempt_id_from_path(relative)
        if attempt_id is None:
            parts = PurePosixPath(relative).parts
            if INITIAL_ATTEMPT in parts or "legacy" in {
                part.casefold() for part in parts
            }:
                # Frozen case fixtures are not Agent-created attempts and are
                # validated through the case manifest/initial-failure facts.
                continue
            issues.append(_issue("ATTEMPT_LAYOUT", f"{name} is outside an Agent-created attempt"))
            continue
        path = _resolve_inside(root, relative)
        if path is None:
            issues.append(_issue("ARTIFACT_PATH", f"unsafe artifact path: {relative[:200]}"))
            continue
        try:
            value = load_json(path, label=name)
        except TrustedValidationError:
            issues.append(_issue("ARTIFACT_JSON", f"invalid {name} for {attempt_id}"))
            continue
        if type(value) is not dict:
            issues.append(_issue("ARTIFACT_JSON", f"{name} for {attempt_id} must be an object"))
            continue
        if attempt_id in records[kinds[name]]:
            issues.append(_issue("ATTEMPT_DUPLICATE_RECORD", f"duplicate {name} for {attempt_id}"))
            continue
        records[kinds[name]][attempt_id] = value
    return records


def _verify_file_record(root: Path, record: Mapping[str, Any], label: str, issues: list[Issue]) -> bool:
    relative = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    safe = _safe_relative_path(relative)
    if safe is None or type(expected_hash) is not str or not HEX64_RE.fullmatch(expected_hash):
        issues.append(_issue("ARTIFACT_RECORD", f"{label} has an invalid path or hash"))
        return False
    path = _resolve_inside(root, safe)
    if path is None or not path.is_file():
        issues.append(_issue("ARTIFACT_MISSING", f"{label} file is missing"))
        return False
    if sha256_file(path).upper() != expected_hash.upper():
        issues.append(_issue("ARTIFACT_HASH", f"{label} SHA256 mismatch"))
        return False
    if type(expected_size) is not int or expected_size != path.stat().st_size:
        issues.append(_issue("ARTIFACT_SIZE", f"{label} size mismatch"))
        return False
    return True


def _verify_manifests(
    root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
    issues: list[Issue],
) -> bool:
    valid = bool(manifests)
    for attempt_id, manifest in sorted(manifests.items()):
        if manifest.get("attempt_id") != attempt_id or manifest.get("schema_version") != "1.0":
            issues.append(_issue("MANIFEST_SCHEMA", f"manifest identity is invalid for {attempt_id}"))
            valid = False
        entries = manifest.get("entries")
        if type(entries) is not list or not entries:
            issues.append(_issue("MANIFEST_EMPTY", f"manifest is empty for {attempt_id}"))
            valid = False
            continue
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            if type(entry) is not dict:
                issues.append(_issue("MANIFEST_ENTRY", f"manifest entry is invalid for {attempt_id}"))
                valid = False
                continue
            path_value = entry.get("path")
            if type(path_value) is str and path_value in seen:
                issues.append(_issue("MANIFEST_DUPLICATE", f"manifest repeats a path for {attempt_id}"))
                valid = False
            if type(path_value) is str:
                seen.add(path_value)
                if PurePosixPath(path_value).name == "artifact_manifest.json":
                    issues.append(_issue("MANIFEST_SELF_REFERENCE", f"manifest includes itself for {attempt_id}"))
                    valid = False
            if not _verify_file_record(root, entry, f"{attempt_id} manifest entry {index}", issues):
                valid = False
    return valid


def _argv_safe(argv: Any) -> bool:
    if type(argv) is not list or not argv or len(argv) > 128:
        return False
    for token in argv:
        if type(token) is not str or not token or len(token) > 1024:
            return False
        if SHELL_META_RE.search(token):
            return False
        if PureWindowsPath(token).is_absolute() or PureWindowsPath(token).drive or token.startswith(("/", "\\")):
            return False
        normalized = token.replace("\\", "/")
        if ".." in PurePosixPath(normalized).parts:
            return False
    return True


def _argv_has_pair(argv: Sequence[str], flag: str, value: Any) -> bool:
    desired = str(value)
    for index, token in enumerate(argv[:-1]):
        if token.casefold() == flag.casefold() and argv[index + 1] == desired:
            return True
    return False


def _scientific_settings_match(settings: Mapping[str, Any], job_spec: Mapping[str, Any]) -> bool:
    molecule = _mapping(job_spec.get("molecule"))
    return (
        settings.get("method") == job_spec.get("method")
        and settings.get("charge") == molecule.get("charge")
        and settings.get("uhf") == molecule.get("uhf")
        and settings.get("solvation") == job_spec.get("solvation")
        and settings.get("required_phases") == job_spec.get("required_phases")
    )


def _prepared_valid(
    attempt_id: str,
    prepared: Mapping[str, Any],
    config: TrustedConfiguration,
    issues: list[Issue],
) -> tuple[bool, bool, bool]:
    valid = True
    if prepared.get("schema_version") != "1.0" or prepared.get("attempt_id") != attempt_id or prepared.get("state") != "PREPARED":
        issues.append(_issue("PREPARED_SCHEMA", f"prepared record is invalid for {attempt_id}"))
        valid = False
    phase = prepared.get("phase")
    if phase not in {"geometry_optimization", "hessian"}:
        issues.append(_issue("PREPARED_PHASE", f"prepared phase is invalid for {attempt_id}"))
        valid = False
    settings = _mapping(prepared.get("scientific_settings"))
    if not _scientific_settings_match(settings, config.job_spec):
        issues.append(_issue("SCIENTIFIC_INVARIANT", f"scientific settings changed in {attempt_id}"))
        valid = False
    execution = _mapping(prepared.get("execution"))
    argv = execution.get("argv")
    argv_ok = _argv_safe(argv) and execution.get("shell") is False
    if not argv_ok:
        issues.append(_issue("ARGV_UNSAFE", f"prepared argv is unsafe for {attempt_id}"))
        valid = False
    if type(argv) is list and argv_ok:
        xtb_cfg = _mapping(_mapping(config.config.get("runtime")).get("xtb"))
        expected_prefix = [
            "conda",
            "run",
            "-n",
            xtb_cfg.get("conda_environment"),
            "--no-capture-output",
            xtb_cfg.get("executable_alias"),
        ]
        wrapper = execution.get("runtime_wrapper_argv")
        scientific = execution.get("scientific_argv")
        split_ok = (
            type(wrapper) is list
            and type(scientific) is list
            and wrapper == expected_prefix[:5]
            and scientific[:2] == [xtb_cfg.get("executable_alias"), "input.xyz"]
            and argv == wrapper + scientific
        )
        if (
            argv[:6] != expected_prefix
            or len(argv) < 7
            or argv[6] != "input.xyz"
            or not split_ok
        ):
            issues.append(_issue("ARGV_EXECUTABLE", f"prepared conda/xTB prefix differs for {attempt_id}"))
            valid = False
        if phase == "geometry_optimization" and not any(token.casefold().startswith("--opt") for token in argv):
            issues.append(_issue("ARGV_PHASE", f"optimization flag is missing for {attempt_id}"))
            valid = False
        if phase == "hessian" and not any(token.casefold() == "--hess" for token in argv):
            issues.append(_issue("ARGV_PHASE", f"Hessian flag is missing for {attempt_id}"))
            valid = False
        molecule = _mapping(config.job_spec.get("molecule"))
        if not _argv_has_pair(argv, "--chrg", molecule.get("charge")) or not _argv_has_pair(argv, "--uhf", molecule.get("uhf")):
            issues.append(_issue("ARGV_INVARIANT", f"charge or UHF is not explicit for {attempt_id}"))
            valid = False
        if str(config.job_spec.get("method", "")).casefold() == "gfn2-xtb" and not _argv_has_pair(argv, "--gfn", 2):
            issues.append(_issue("ARGV_INVARIANT", f"GFN2 method is not explicit for {attempt_id}"))
            valid = False
        if config.job_spec.get("solvation") is None and any(token.casefold() in {"--alpb", "--gbsa", "--cosmo"} for token in argv):
            issues.append(_issue("ARGV_INVARIANT", f"solvation was introduced in {attempt_id}"))
            valid = False
    request = _mapping(prepared.get("request"))
    xtb_cfg = _mapping(_mapping(config.config.get("runtime")).get("xtb"))
    thread_count = request.get("thread_count")
    walltime = request.get("walltime_seconds")
    timeout_value = execution.get("timeout_seconds")
    resource_limits_ok = (
        type(thread_count) is int
        and type(walltime) is int
        and type(timeout_value) is int
        and xtb_cfg.get("thread_count_min", 1)
        <= thread_count
        <= xtb_cfg.get("thread_count_max", 1)
        and 1 <= walltime <= xtb_cfg.get("walltime_seconds_max", 0)
        and walltime <= _strict_int(config.budget.get("single_xtb_timeout_seconds"), 0)
        and timeout_value == walltime
        and execution.get("environment_manager") == "conda"
        and execution.get("conda_environment") == xtb_cfg.get("conda_environment")
        and execution.get("executable_alias") == xtb_cfg.get("executable_alias")
        and execution.get("expected_version") == xtb_cfg.get("version")
        and execution.get("identity_policy") == xtb_cfg.get("identity_policy")
        and execution.get("required_help_tokens") == xtb_cfg.get("required_help_tokens")
    )
    if not resource_limits_ok:
        issues.append(_issue("RESOURCE_LIMIT", f"prepared resources violate config for {attempt_id}"))
        valid = False
    preserved = set(request.get("preserved_invariants", [])) if isinstance(request.get("preserved_invariants"), list) else set()
    required_preserved = {"method", "charge", "uhf", "solvation", "atom_order", "composition", "required_phases"}
    invariants_claimed = required_preserved.issubset(preserved)
    if not invariants_claimed:
        issues.append(_issue("INVARIANT_LIST", f"preserved invariant list is incomplete for {attempt_id}"))
        valid = False
    initial_limit = _mapping(_mapping(config.config.get("runtime")).get("xtb")).get("initial_failure_cycle_limit")
    cycle_limit = request.get("optimization_cycle_limit", request.get("cycle_limit"))
    cycle_limits_ok = (
        (
            phase == "geometry_optimization"
            and type(cycle_limit) is int
            and xtb_cfg.get("restart_cycle_limit_min", 0)
            <= cycle_limit
            <= xtb_cfg.get("restart_cycle_limit_max", 0)
        )
        or (phase == "hessian" and cycle_limit is None)
    )
    if not cycle_limits_ok:
        issues.append(_issue("CYCLE_LIMIT", f"cycle limit violates config for {attempt_id}"))
        valid = False
    targeted = (
        phase == "geometry_optimization"
        and type(cycle_limit) is int
        and type(initial_limit) is int
        and cycle_limit > initial_limit
    )
    source = _mapping(prepared.get("source_geometry"))
    source_path = _safe_relative_path(source.get("path"))
    from_last_frame = (
        source_path is not None
        and INITIAL_ATTEMPT in PurePosixPath(source_path).parts
        and PurePosixPath(source_path).name.casefold()
        in {"xtbopt.log", "xtb.trj", "xtblast.xyz", "restart.xyz"}
        and type(source.get("frame_index")) is int
    )
    return valid, targeted, from_last_frame


def _record_files_valid(root: Path, record: Mapping[str, Any], label: str, issues: list[Issue]) -> bool:
    files = record.get("files")
    if type(files) is not dict or not files:
        issues.append(_issue("RUN_FILES", f"{label} has no recorded files"))
        return False
    valid = True
    for relative, details in sorted(files.items()):
        if type(details) is not dict:
            issues.append(_issue("RUN_FILES", f"{label} contains an invalid file record"))
            valid = False
            continue
        record_with_path = dict(details)
        record_with_path["path"] = relative
        if not _verify_file_record(root, record_with_path, label, issues):
            valid = False
    return valid


def _control_input_valid(
    root: Path,
    attempt_id: str,
    prepared: Mapping[str, Any],
    issues: list[Issue],
) -> bool:
    phase = prepared.get("phase")
    control = prepared.get("control_input")
    request = _mapping(prepared.get("request"))
    scientific_argv = _mapping(prepared.get("execution")).get("scientific_argv")
    if phase == "hessian":
        return control is None and isinstance(scientific_argv, list) and "--input" not in scientific_argv
    if type(control) is not dict or type(scientific_argv) is not list:
        issues.append(_issue("CONTROL_INPUT", f"optimization control input is missing for {attempt_id}"))
        return False
    relative = _safe_relative_path(control.get("path"))
    cycle_limit = request.get("optimization_cycle_limit")
    path = _resolve_inside(root, relative) if relative is not None else None
    if (
        path is None
        or not path.is_file()
        or type(cycle_limit) is not int
        or control.get("optimization_cycle_limit") != cycle_limit
        or type(control.get("sha256")) is not str
        or sha256_file(path).upper() != str(control.get("sha256")).upper()
        or scientific_argv[-2:] == []
    ):
        issues.append(_issue("CONTROL_INPUT", f"optimization control input is invalid for {attempt_id}"))
        return False
    expected = f"$opt\n  maxcycle={cycle_limit}\n$end\n".encode("utf-8")
    argv_has_input = any(
        token == "--input" and index + 1 < len(scientific_argv) and scientific_argv[index + 1] == "xcontrol.inp"
        for index, token in enumerate(scientific_argv)
    )
    if path.read_bytes() != expected or not argv_has_input:
        issues.append(_issue("CONTROL_INPUT", f"optimization cycle control differs for {attempt_id}"))
        return False
    return True


def _parse_xyz(path: Path, job_spec: Mapping[str, Any]) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    molecule = _mapping(job_spec.get("molecule"))
    expected_count = molecule.get("atom_count")
    expected_elements = molecule.get("element_sequence")
    if type(expected_count) is not int or expected_count <= 0 or not isinstance(expected_elements, list):
        return False
    index = 0
    complete: list[list[str]] = []
    while index < len(lines):
        try:
            count = int(lines[index].strip())
        except (ValueError, IndexError):
            break
        if count <= 0 or index + count + 1 >= len(lines):
            break
        rows = lines[index + 2 : index + 2 + count]
        if len(rows) != count:
            break
        elements: list[str] = []
        valid_rows = True
        for row in rows:
            tokens = row.split()
            if len(tokens) < 4:
                valid_rows = False
                break
            elements.append(tokens[0])
            try:
                coordinates = [float(token) for token in tokens[1:4]]
            except ValueError:
                valid_rows = False
                break
            if not all(math.isfinite(number) for number in coordinates):
                valid_rows = False
                break
        if valid_rows and count == expected_count and elements == expected_elements:
            complete.append(elements)
        index += count + 2
    return bool(complete)


def _log_texts_for_attempt(
    root: Path, attempt_id: str, run_record: Mapping[str, Any]
) -> tuple[str, str]:
    stdout_candidates: list[str] = []
    stderr_candidates: list[str] = []
    files = run_record.get("files")
    if isinstance(files, Mapping):
        for relative in files:
            if type(relative) is not str:
                continue
            folded = relative.casefold()
            if folded.endswith(("stdout.txt", "stdout.log", "xtb.out", ".stdout")):
                stdout_candidates.append(relative)
            elif folded.endswith(("stderr.txt", "stderr.log", ".stderr")):
                stderr_candidates.append(relative)
    stdout_candidates.extend(
        [
            f"attempts/{attempt_id}/stdout.txt",
            f"attempts/{attempt_id}/stdout.log",
            f"{attempt_id}/stdout.txt",
            f"{attempt_id}/stdout.log",
        ]
    )
    stderr_candidates.extend(
        [
            f"attempts/{attempt_id}/stderr.txt",
            f"attempts/{attempt_id}/stderr.log",
            f"{attempt_id}/stderr.txt",
            f"{attempt_id}/stderr.log",
        ]
    )

    def first_text(candidates: Sequence[str]) -> str:
        for relative in candidates:
            path = _resolve_inside(root, relative)
            if path is not None and path.is_file() and path.stat().st_size <= MAX_SCANNED_TEXT_BYTES:
                try:
                    return path.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeDecodeError):
                    continue
        return ""

    return first_text(stdout_candidates), first_text(stderr_candidates)


def _validation_artifacts_valid(
    root: Path,
    attempt_id: str,
    validation: Mapping[str, Any],
    job_spec: Mapping[str, Any],
    issues: list[Issue],
) -> tuple[bool, bool, bool]:
    artifacts = validation.get("artifacts")
    if type(artifacts) is not dict or not artifacts:
        issues.append(_issue("VALIDATION_ARTIFACTS", f"validation artifacts are missing for {attempt_id}"))
        return False, False, False
    valid = True
    xyz_valid = False
    phase_artifact = False
    for role, entry in sorted(artifacts.items()):
        if type(entry) is not dict:
            valid = False
            continue
        if not _verify_file_record(root, entry, f"{attempt_id} validation artifact {role}", issues):
            valid = False
            continue
        safe = _safe_relative_path(entry.get("path"))
        if safe is None or attempt_id not in PurePosixPath(safe).parts or "legacy" in {part.casefold() for part in PurePosixPath(safe).parts}:
            issues.append(_issue("ARTIFACT_NOT_CURRENT", f"validation artifact is not current for {attempt_id}"))
            valid = False
            continue
        path = _resolve_inside(root, safe)
        role_folded = str(role).casefold()
        if path is not None and path.suffix.casefold() == ".xyz":
            xyz_valid = xyz_valid or _parse_xyz(path, job_spec)
        if any(token in role_folded for token in ("hessian", "vibration", "spectrum", "optimized_geometry", "xtbopt")):
            phase_artifact = True
    return valid, xyz_valid, phase_artifact


def _provenance_valid(
    root: Path,
    attempt_id: str,
    prepared: Mapping[str, Any],
    provenance: Mapping[str, Any],
    issues: list[Issue],
) -> bool:
    if provenance.get("attempt_id") != attempt_id:
        issues.append(_issue("PROVENANCE_ID", f"provenance id mismatch for {attempt_id}"))
        return False
    source = _mapping(prepared.get("source_geometry"))
    input_geometry = _mapping(prepared.get("input_geometry"))
    prov_source = _mapping(provenance.get("source"))
    prov_input = _mapping(provenance.get("input"))
    valid = True
    for left, right, label in (
        (source.get("path"), prov_source.get("path"), "source path"),
        (source.get("file_sha256"), prov_source.get("file_sha256", prov_source.get("sha256")), "source hash"),
        (input_geometry.get("path"), prov_input.get("path"), "input path"),
        (input_geometry.get("sha256"), prov_input.get("sha256"), "input hash"),
    ):
        if left != right:
            issues.append(_issue("PROVENANCE_MISMATCH", f"{label} mismatch for {attempt_id}"))
            valid = False
    for record, label in ((source, "source"), (input_geometry, "input")):
        safe = _safe_relative_path(record.get("path"))
        expected = record.get("file_sha256", record.get("sha256"))
        path = _resolve_inside(root, safe) if safe is not None else None
        if path is None or not path.is_file() or type(expected) is not str or sha256_file(path).upper() != expected.upper():
            issues.append(_issue("PROVENANCE_FILE", f"{label} file mismatch for {attempt_id}"))
            valid = False
    if provenance.get("restart_allowed") is not True or provenance.get("success_evidence_allowed") is not False:
        issues.append(_issue("PROVENANCE_PERMISSION", f"provenance permissions are invalid for {attempt_id}"))
        valid = False
    return valid


def validate_artifacts(
    artifact_root: Path,
    config: TrustedConfiguration,
    trace: TraceFacts,
    final_answer: Mapping[str, Any],
) -> ArtifactFacts:
    root = artifact_root.resolve(strict=True)
    if not root.is_dir():
        raise TrustedValidationError("artifact workspace must be a directory")
    files, issues, secret, absolute, oracle = _scan_artifact_tree(root)
    case_manifest_valid = _verify_workspace_case_manifest(
        root,
        files,
        config,
        issues,
    )
    filesystem_safe = not any(
        item.code in {"ARTIFACT_REPARSE_POINT", "ARTIFACT_SPECIAL_FILE"}
        for item in issues
    )
    initial_scheduler, initial_max_cycles, legacy_present = _derive_initial_case_facts(
        root, files, issues
    )
    records = _read_artifact_jsons(root, files, issues)
    prepared = records["prepared"]
    provenance = records["provenance"]
    run_records = records["run_record"]
    validations = records["validation"]
    manifests = records["manifest"]
    manifests_valid = _verify_manifests(root, manifests, issues)

    event_prepared_ids = [payload.get("attempt_id") for payload in trace.attempt_prepared]
    artifact_ids = sorted(prepared)
    canonical_attempt_files_valid = _verify_canonical_attempt_files(
        root, files, artifact_ids, issues
    )
    attempt_isolation = (
        bool(artifact_ids)
        and canonical_attempt_files_valid
        and event_prepared_ids == artifact_ids
        and all(ATTEMPT_RE.fullmatch(item) for item in artifact_ids)
        and len(set(artifact_ids)) == len(artifact_ids)
        and artifact_ids == sorted(artifact_ids)
    )
    if not attempt_isolation:
        issues.append(_issue("ATTEMPT_ISOLATION", "attempt events and immutable attempt directories differ"))

    all_argv_safe = True
    all_invariants = True
    all_provenance = True
    native_any = False
    opt_successes: list[str] = []
    hess_successes: list[str] = []
    current_artifacts = True
    final_xyz_valid = False
    legacy_used = False
    from_last_frame = False
    from_original = False
    targeted_repair = False

    for attempt_id in artifact_ids:
        prepared_record = prepared[attempt_id]
        prepared_ok, targeted, last_frame = _prepared_valid(
            attempt_id, prepared_record, config, issues
        )
        control_ok = _control_input_valid(root, attempt_id, prepared_record, issues)
        prepared_ok = prepared_ok and control_ok
        targeted_repair = targeted_repair or targeted
        from_last_frame = from_last_frame or last_frame
        source_path = str(_mapping(prepared_record.get("source_geometry")).get("path", ""))
        normalized_source = source_path.replace("\\", "/")
        from_original = from_original or (
            normalized_source.endswith("input/start.xyz")
            or normalized_source == f"attempts/{INITIAL_ATTEMPT}/input.xyz"
        )
        legacy_used = legacy_used or "legacy" in {
            part.casefold() for part in PurePosixPath(source_path.replace("\\", "/")).parts
        }
        all_argv_safe = all_argv_safe and prepared_ok and _argv_safe(_mapping(prepared_record.get("execution")).get("argv"))
        all_invariants = all_invariants and prepared_ok

        provenance_record = provenance.get(attempt_id)
        if provenance_record is None:
            issues.append(_issue("PROVENANCE_MISSING", f"provenance is missing for {attempt_id}"))
            all_provenance = False
        else:
            all_provenance = all_provenance and _provenance_valid(
                root, attempt_id, prepared_record, provenance_record, issues
            )

        run_record = run_records.get(attempt_id)
        validation = validations.get(attempt_id)
        manifest = manifests.get(attempt_id)
        if run_record is None or validation is None or manifest is None:
            issues.append(_issue("ATTEMPT_INCOMPLETE", f"trusted records are incomplete for {attempt_id}"))
            current_artifacts = False
            continue
        argv = _mapping(prepared_record.get("execution")).get("argv")
        prepared_execution = _mapping(prepared_record.get("execution"))
        run_argv = run_record.get("argv")
        duration = run_record.get("duration_seconds")
        timeout_seconds = run_record.get("timeout_seconds")
        run_ok = (
            run_record.get("attempt_id") == attempt_id
            and run_record.get("phase") == prepared_record.get("phase")
            and run_record.get("state") == "EXECUTED"
            and run_record.get("native_xtb_executed") is True
            and run_record.get("shell") is False
            and run_argv == argv
            and run_record.get("scientific_argv")
            == prepared_execution.get("scientific_argv")
            and run_record.get("runtime_wrapper_argv")
            == prepared_execution.get("runtime_wrapper_argv")
            and _argv_safe(run_argv)
            and run_record.get("timed_out") is False
            and run_record.get("exit_code") == 0
            and type(timeout_seconds) is int
            and timeout_seconds == prepared_execution.get("timeout_seconds")
            and timeout_seconds
            <= _strict_int(config.budget.get("single_xtb_timeout_seconds"), 0)
            and type(duration) in {int, float, Decimal}
            and Decimal(str(duration)).is_finite()
            and Decimal("0") <= Decimal(str(duration)) <= Decimal(str(timeout_seconds))
            and _record_files_valid(root, run_record, f"run record {attempt_id}", issues)
        )
        native_any = native_any or run_record.get("native_xtb_executed") is True
        if not run_ok:
            issues.append(_issue("NATIVE_RECORD", f"native execution record is invalid for {attempt_id}"))
        all_argv_safe = all_argv_safe and run_ok

        validation_ok = (
            validation.get("attempt_id") == attempt_id
            and validation.get("phase") == prepared_record.get("phase")
            and validation.get("state") == "VALIDATED"
            and validation.get("validation_passed") is True
            and validation.get("process_valid") is True
            and validation.get("native_xtb_executed") is True
            and validation.get("normal_termination_observed") is True
            and validation.get("invariants_preserved") is True
            and validation.get("source_current") is True
            and validation.get("artifacts_current") is True
        )
        artifacts_ok, xyz_ok, phase_artifact = _validation_artifacts_valid(
            root, attempt_id, validation, config.job_spec, issues
        )
        current_artifacts = current_artifacts and artifacts_ok and validation_ok
        final_xyz_valid = final_xyz_valid or xyz_ok
        stdout_text, stderr_text = _log_texts_for_attempt(root, attempt_id, run_record)
        normal_in_log = (
            NORMAL_TERMINATION_RE.search(stdout_text) is not None
            or NORMAL_TERMINATION_RE.search(stderr_text) is not None
        )
        phase = prepared_record.get("phase")
        if phase == "geometry_optimization":
            scientific = (
                validation.get("optimization_completed") is True
                and validation.get("hessian_completed") is False
                and OPT_CONVERGED_RE.search(stdout_text) is not None
                and normal_in_log
                and xyz_ok
                and phase_artifact
            )
            if prepared_ok and run_ok and validation_ok and artifacts_ok and scientific:
                opt_successes.append(attempt_id)
        elif phase == "hessian":
            scientific = (
                validation.get("optimization_completed") is False
                and validation.get("hessian_completed") is True
                and HESSIAN_RE.search(stdout_text) is not None
                and normal_in_log
                and phase_artifact
            )
            parent = _mapping(provenance_record or {}).get("parent_attempt_id")
            if type(parent) is not str or parent not in opt_successes:
                scientific = False
                issues.append(_issue("HESSIAN_PARENT", f"Hessian source is not a validated optimization for {attempt_id}"))
            if prepared_ok and run_ok and validation_ok and artifacts_ok and scientific:
                hess_successes.append(attempt_id)

    # Events must agree with files, but event assertions alone never create a pass.
    native_event_ids = [payload.get("attempt_id") for payload in trace.native_executions]
    validation_event_ids = [payload.get("attempt_id") for payload in trace.attempt_validations]
    if native_event_ids != sorted(run_records) or validation_event_ids != sorted(validations):
        issues.append(_issue("EVENT_ARTIFACT_MISMATCH", "native or validation event ids differ from artifact records"))
        native_any = False
    prepared_events = {
        str(payload.get("attempt_id")): payload for payload in trace.attempt_prepared
    }
    native_event_map = {
        str(payload.get("attempt_id")): payload for payload in trace.native_executions
    }
    validation_event_map = {
        str(payload.get("attempt_id")): payload for payload in trace.attempt_validations
    }
    for attempt_id in artifact_ids:
        prepared_record = prepared[attempt_id]
        prepared_event = prepared_events.get(attempt_id)
        prepared_path = _record_relative_path(files, attempt_id, "prepared_command.json")
        input_manifest_path = _record_relative_path(files, attempt_id, "input_manifest.json")
        if (
            prepared_event is None
            or prepared_event.get("phase") != prepared_record.get("phase")
            or prepared_event.get("state") != prepared_record.get("state")
            or _safe_relative_path(prepared_event.get("prepared_command_path"))
            != prepared_path
            or _safe_relative_path(prepared_event.get("input_manifest_path"))
            != input_manifest_path
            or _mapping(prepared_event.get("source_geometry")).get("sha256")
            not in {
                _mapping(prepared_record.get("source_geometry")).get("file_sha256"),
                _mapping(prepared_record.get("source_geometry")).get("frame_sha256"),
            }
        ):
            issues.append(_issue("PREPARED_EVENT_MISMATCH", f"prepared event differs from files for {attempt_id}"))
            attempt_isolation = False

        run_record = run_records.get(attempt_id)
        native_event = native_event_map.get(attempt_id)
        run_path = _record_relative_path(files, attempt_id, "execution_summary.json")
        manifest_path = _record_relative_path(files, attempt_id, "artifact_manifest.json")
        if run_record is not None:
            hashes = _mapping(native_event.get("artifact_hashes")) if native_event else {}
            recorded_files = _mapping(run_record.get("files"))
            hash_match = (
                bool(hashes)
                and set(hashes) == set(recorded_files)
                and all(
                type(recorded_files.get(path_value)) is dict
                and _mapping(recorded_files.get(path_value)).get("sha256") == digest
                for path_value, digest in hashes.items()
                )
            )
            if (
                native_event is None
                or native_event.get("phase") != run_record.get("phase")
                or native_event.get("source_type") != "NATIVE_XTB"
                or native_event.get("argv") != run_record.get("argv")
                or native_event.get("shell") is not False
                or native_event.get("process_exit_code") != run_record.get("exit_code")
                or native_event.get("timed_out") != run_record.get("timed_out")
                or _safe_relative_path(native_event.get("execution_summary_path")) != run_path
                or _safe_relative_path(native_event.get("artifact_manifest_path")) != manifest_path
                or not hash_match
            ):
                issues.append(_issue("NATIVE_EVENT_MISMATCH", f"native event differs from files for {attempt_id}"))
                native_any = False

        validation_record = validations.get(attempt_id)
        validation_event = validation_event_map.get(attempt_id)
        validation_path = _record_relative_path(files, attempt_id, "validation.json")
        if validation_record is not None and (
            validation_event is None
            or validation_event.get("phase") != validation_record.get("phase")
            or validation_event.get("valid") != validation_record.get("validation_passed")
            or validation_event.get("optimization_converged")
            != validation_record.get("optimization_completed")
            or validation_event.get("hessian_completed")
            != validation_record.get("hessian_completed")
            or validation_event.get("current_artifacts")
            != validation_record.get("artifacts_current")
            or validation_event.get("scientific_invariants_preserved")
            != validation_record.get("invariants_preserved")
            or _safe_relative_path(validation_event.get("validation_path"))
            != validation_path
        ):
            issues.append(_issue("VALIDATION_EVENT_MISMATCH", f"validation event differs from files for {attempt_id}"))
            current_artifacts = False
            all_invariants = False

    result_summary = _mapping(final_answer.get("result_summary"))
    final_attempt = result_summary.get("final_attempt_id")
    final_matches = (
        type(final_attempt) is str
        and final_attempt in hess_successes
        and result_summary.get("optimization_completed") is True
        and result_summary.get("hessian_completed") is True
    )
    optimization_converged = bool(opt_successes)
    hessian_completed = bool(hess_successes) and final_matches
    final_artifacts_current = current_artifacts and hessian_completed and manifests_valid

    return ArtifactFacts(
        issues=issues,
        workspace_sha256=_workspace_hash(files),
        files=files,
        case_manifest_valid=case_manifest_valid,
        manifests_valid=manifests_valid,
        canonical_attempt_files_valid=canonical_attempt_files_valid,
        prepared_attempts=prepared,
        provenance_records=provenance,
        run_records=run_records,
        validations=validations,
        actual_argv_safe=all_argv_safe,
        native_xtb_executed=native_any,
        optimization_converged=optimization_converged,
        hessian_completed=hessian_completed,
        final_artifacts_current=final_artifacts_current and case_manifest_valid,
        legacy_used_as_current=legacy_used,
        scientific_invariants_preserved=all_invariants and bool(prepared),
        provenance_valid=all_provenance and bool(prepared),
        final_xyz_valid=final_xyz_valid,
        oracle_present=oracle,
        secret_finding=secret,
        absolute_path_finding=absolute,
        filesystem_safe=filesystem_safe,
        attempt_isolation_valid=attempt_isolation,
        restart_from_last_current_frame=from_last_frame,
        restart_from_original_input=from_original,
        targeted_cycle_repair=targeted_repair,
        initial_scheduler_completed=initial_scheduler,
        initial_failure_max_cycles=initial_max_cycles,
        legacy_artifact_present=legacy_present,
    )


def verify_final_evidence(
    final_answer: Mapping[str, Any],
    trace: TraceFacts,
    artifact_root: Path,
) -> tuple[bool, list[Issue]]:
    """Tie final citations to prior successful bounded reads and file bytes."""

    issues: list[Issue] = []
    reads = {fact.call_id: fact for fact in trace.direct_log_reads if fact.ok}
    evidence = _mapping(final_answer.get("diagnosis")).get("evidence")
    if type(evidence) is not list or not evidence:
        return False, [_issue("EVIDENCE_MISSING", "final answer has no evidence")]
    valid = True
    for index, citation in enumerate(evidence):
        if type(citation) is not dict:
            valid = False
            continue
        call_id = citation.get("tool_call_id")
        fact = reads.get(call_id)
        if fact is None:
            issues.append(_issue("EVIDENCE_UNREAD", f"citation {index} is not tied to a successful direct read"))
            valid = False
            continue
        path_value = _safe_relative_path(citation.get("path"))
        if path_value is None or path_value != _safe_relative_path(fact.arguments.get("path")):
            issues.append(_issue("EVIDENCE_PATH", f"citation {index} path does not match its read"))
            valid = False
            continue
        start = citation.get("line_start")
        end = citation.get("line_end")
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            valid = False
            continue
        read_start = fact.arguments.get("start_line")
        read_end = fact.arguments.get("end_line")
        if type(read_start) is not int or type(read_end) is not int or start < read_start or end > read_end:
            issues.append(_issue("EVIDENCE_RANGE", f"citation {index} was outside the read range"))
            valid = False
            continue
        file_path = _resolve_inside(artifact_root.resolve(strict=True), path_value)
        if file_path is None or not file_path.is_file() or file_path.stat().st_size > MAX_SCANNED_TEXT_BYTES:
            issues.append(_issue("EVIDENCE_FILE", f"citation {index} file is unavailable"))
            valid = False
            continue
        try:
            content_hash = selected_line_range_sha256(file_path, start, end)
        except TrustedValidationError:
            issues.append(_issue("EVIDENCE_RANGE", f"citation {index} exceeds or cannot read the file"))
            valid = False
            continue
        if str(citation.get("content_sha256", "")).upper() != content_hash:
            issues.append(_issue("EVIDENCE_HASH", f"citation {index} content SHA256 mismatch"))
            valid = False
    return valid, issues


def _independent_attempt_history(trace: TraceFacts) -> list[dict[str, Any]]:
    ordered_ids: list[str] = []
    for payload in (
        *trace.attempt_prepared,
        *trace.native_executions,
        *trace.attempt_validations,
    ):
        attempt_id = payload.get("attempt_id")
        if type(attempt_id) is str and attempt_id not in ordered_ids:
            ordered_ids.append(attempt_id)

    def first_by_id(values: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for payload in values:
            attempt_id = payload.get("attempt_id")
            if type(attempt_id) is str and attempt_id not in result:
                result[attempt_id] = payload
        return result

    prepared = first_by_id(trace.attempt_prepared)
    executed = first_by_id(trace.native_executions)
    validated = first_by_id(trace.attempt_validations)
    history: list[dict[str, Any]] = []
    for attempt_id in ordered_ids:
        prep = prepared.get(attempt_id, {})
        run = executed.get(attempt_id, {})
        validation = validated.get(attempt_id, {})
        source = _mapping(prep.get("source_geometry"))
        phase = prep.get("phase", run.get("phase", validation.get("phase", "")))
        source_hash = source.get("sha256")
        exit_code = run.get("process_exit_code")
        history.append(
            {
                "attempt_id": attempt_id,
                "phase": phase if type(phase) is str else "",
                "source_geometry": {
                    "path": source.get("path") if type(source.get("path")) is str else "",
                    "frame_index": (
                        source.get("frame_index")
                        if type(source.get("frame_index")) is int
                        else None
                    ),
                    "sha256": source_hash.upper() if type(source_hash) is str else None,
                },
                "prepared": bool(prep),
                "native_executed": bool(run),
                "process_exit_code": exit_code if type(exit_code) is int else None,
                "timed_out": run.get("timed_out") is True,
                "validated": bool(validation),
                "validation_passed": validation.get("valid") is True,
                "optimization_completed": validation.get("optimization_converged") is True,
                "hessian_completed": validation.get("hessian_completed") is True,
                "artifacts_current": validation.get("current_artifacts") is True,
                "invariants_preserved": (
                    validation.get("scientific_invariants_preserved") is True
                ),
            }
        )
    return history


def _independent_tool_counts(trace: TraceFacts) -> dict[str, Any]:
    by_name: dict[str, dict[str, int]] = {}
    for fact in trace.tool_calls:
        record = by_name.setdefault(
            fact.name,
            {"calls": 0, "successful": 0, "failed": 0},
        )
        record["calls"] += 1
        record["successful" if fact.ok else "failed"] += 1
    event_names = [event.get("event") for event in trace.events]
    return {
        "tool_calls": len(trace.tool_calls),
        "successful_tool_calls": trace.successful_tool_calls,
        "failed_tool_calls": len(trace.tool_calls) - trace.successful_tool_calls,
        "agent_turns": trace.agent_turns,
        "provider_attempts": event_names.count("provider_request"),
        "provider_failures": event_names.count("provider_error"),
        "provider_retries": trace.provider_retries,
        "execution_attempts": len(trace.attempt_prepared),
        "native_executions": len(trace.native_executions),
        "by_tool": [
            {"name": name, **by_name[name]}
            for name in sorted(by_name)
        ],
    }


def derive_expected_trusted_runtime(
    model_submission: Mapping[str, Any],
    trace: TraceFacts,
    artifacts: ArtifactFacts,
) -> dict[str, Any]:
    """Independently reconstruct the runner-owned final-answer fields."""

    history = _independent_attempt_history(trace)
    result = _mapping(model_submission.get("result_summary"))
    final_attempt_id = result.get("final_attempt_id")
    final_attempt_id = final_attempt_id if type(final_attempt_id) is str else ""
    final_record = next(
        (item for item in history if item.get("attempt_id") == final_attempt_id),
        {},
    )
    final_valid = final_record.get("validation_passed") is True
    optimization_completed = any(
        item.get("validation_passed") is True
        and item.get("optimization_completed") is True
        for item in history
    )
    start = _mapping(_mapping(trace.events[0]).get("payload")) if trace.events else {}
    return {
        "attempt_history": history,
        "tool_counts": _independent_tool_counts(trace),
        "final_validation": {
            "final_attempt_id": final_attempt_id,
            "final_attempt_validated": final_valid,
            "optimization_completed": optimization_completed,
            "hessian_completed": (
                final_valid and final_record.get("hessian_completed") is True
            ),
            "artifacts_current": (
                final_valid and final_record.get("artifacts_current") is True
            ),
            "invariants_preserved": (
                final_valid and final_record.get("invariants_preserved") is True
            ),
        },
        "trace": {
            "trace_sha256": trace.trace_sha256.upper(),
            "final_event_hash": trace.final_event_hash.upper(),
            "event_count": len(trace.events),
            "hash_chain_valid": trace.hash_chain_valid,
        },
        "artifacts": {
            "workspace_sha256": artifacts.workspace_sha256.upper(),
            "file_count": len(artifacts.files),
            "files": [dict(item) for item in artifacts.files],
        },
        "model_identity": {
            "logical_model": trace.logical_model or "",
            "requested_model": trace.requested_model or "",
            "actual_models": list(trace.actual_models),
            "provider": start.get("provider") if type(start.get("provider")) is str else "",
            "transport": (
                start.get("transport") if type(start.get("transport")) is str else ""
            ),
            "exact_match": trace.exact_model_identity,
            "identity_exempt": trace.identity_exempt,
        },
        "execution_mode": {
            "run_type": trace.run_type,
            "execution_mode": trace.execution_mode,
            "native_mode": trace.native_mode,
        },
    }


def verify_trusted_runtime(
    value: Any,
    model_submission: Mapping[str, Any],
    trace: TraceFacts,
    artifacts: ArtifactFacts,
) -> tuple[bool, list[Issue]]:
    expected = derive_expected_trusted_runtime(model_submission, trace, artifacts)
    try:
        matches = type(value) is dict and canonical_json_bytes(value) == canonical_json_bytes(expected)
    except TrustedValidationError:
        matches = False
    if matches:
        return True, []
    return False, [
        _issue(
            "TRUSTED_RUNTIME_MISMATCH",
            "persisted trusted runtime differs from independently derived trace and artifact facts",
        )
    ]


def stable_issues(*groups: Iterable[Issue]) -> list[Issue]:
    unique = {(item.code, item.message): item for group in groups for item in group}
    return [unique[key] for key in sorted(unique)]


def selected_line_range_sha256(path: Path, start_line: int, end_line: int) -> str:
    """Hash exact selected UTF-8 line bytes, preserving LF/CRLF/final EOF."""

    if (
        type(start_line) is not int
        or type(end_line) is not int
        or start_line < 1
        or end_line < start_line
    ):
        raise TrustedValidationError("line range is invalid")
    raw = _bounded_read(path, MAX_SCANNED_TEXT_BYTES, "evidence file")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrustedValidationError("evidence file must be UTF-8") from exc
    lines = raw.splitlines(keepends=True)
    if end_line > len(lines):
        raise TrustedValidationError("line range exceeds evidence file")
    return sha256_bytes(b"".join(lines[start_line - 1 : end_line]))

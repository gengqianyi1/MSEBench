"""Standalone deterministic evaluator for constrained QED submissions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


# ``python evaluator/evaluate.py`` does not set a package context. Insert only
# the repository root containing this file so the documented command works
# from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from evaluator.scoring import ScoringConfigurationError, compute_score
else:
    from .scoring import ScoringConfigurationError, compute_score

from tools.rdkit_tools import TOOL_REGISTRY, evaluate_candidate


EVALUATION_SCHEMA_VERSION = "1.0"
MAX_CONFIG_BYTES = 1_048_576
ABSOLUTE_MAX_SUBMISSION_BYTES = 16 * 1_048_576
ABSOLUTE_MAX_LIST_ITEMS = 1_024
ABSOLUTE_MAX_STRING_LENGTH = 1_000_000
ABSOLUTE_MAX_TOOL_CALL_COUNT = 1_000_000
MAX_TRAJECTORY_BYTES = 16 * 1_048_576
MAX_TRAJECTORY_LINES = 50_000
MAX_TRAJECTORY_LINE_BYTES = 2 * 1_048_576
MAX_JSON_DEPTH = 16
MAX_JSON_CONTAINER_ITEMS = 4_096
MAX_JSON_NODES_PER_EVENT = 100_000
PUBLIC_TOOL_MAX_ARGUMENT_LENGTH = 512

# Kept independent of the runner so a trajectory cannot introduce an
# evaluator-recognized pseudo-tool merely by changing runner metadata.
PUBLIC_TOOL_SCHEMAS: dict[str, dict[str, frozenset[str]]] = {
    "validate_smiles": {
        "required": frozenset({"smiles"}),
        "optional": frozenset(),
    },
    "canonicalize_smiles": {
        "required": frozenset({"smiles"}),
        "optional": frozenset(),
    },
    "calculate_qed": {
        "required": frozenset({"smiles"}),
        "optional": frozenset(),
    },
    "calculate_properties": {
        "required": frozenset({"smiles"}),
        "optional": frozenset(),
    },
    "calculate_similarity": {
        "required": frozenset({"smiles_a", "smiles_b"}),
        "optional": frozenset(),
    },
    "check_scaffold": {
        "required": frozenset({"smiles"}),
        "optional": frozenset({"scaffold_smarts"}),
    },
    "check_substructure": {
        "required": frozenset({"smiles", "smarts"}),
        "optional": frozenset(),
    },
    "evaluate_candidate": {
        "required": frozenset({"smiles"}),
        "optional": frozenset(),
    },
}
SUPPORTED_PUBLIC_TOOLS = frozenset(PUBLIC_TOOL_SCHEMAS)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "task_id",
    "final_candidate",
    "candidate_history",
    "run_metadata",
}
_CANDIDATE_KEYS = {"name", "smiles", "rationale"}
_RUN_METADATA_KEYS = {
    "tool_call_count",
    "budget_exceeded",
    "unique_evaluated_candidates",
    "final_candidate_evaluated",
}


class SubmissionValidationError(ValueError):
    """A safe, user-actionable submission parsing or schema error."""


class EvaluatorConfigurationError(ValueError):
    """The trusted task configuration cannot be evaluated."""


class ProcessEvidenceValidationError(ValueError):
    """A trajectory or in-memory process-event stream is not trustworthy."""


class _DuplicateJSONKeyError(ValueError):
    pass


@dataclass(frozen=True)
class _ProcessEvidence:
    source: str
    present: bool
    valid: bool
    event_count: int
    tool_call_count: int
    successful_public_tool_call_count: int
    rejected_tool_call_count: int
    evaluate_candidate_attempt_count: int
    successful_evaluate_candidate_call_count: int
    unique_evaluated_candidates: tuple[str, ...]
    budget_exceeded: bool
    public_tool_events_valid: bool
    lifecycle_valid: bool
    run_completed: bool
    terminal_event: str | None
    issues: tuple[str, ...] = ()
    error: str | None = None


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError("JSON objects must not contain duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON numbers are not allowed")


def _plain_dict(value: Any, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SubmissionValidationError(f"{path} must be an object")
    return value


def _config_dict(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluatorConfigurationError(f"{path} must be an object")
    return value


def _config_int(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluatorConfigurationError(f"{path} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise EvaluatorConfigurationError(f"{path} must be >= {minimum}{upper}")
    return value


def _safe_key_names(keys: set[Any]) -> str:
    """Describe unexpected JSON keys without reflecting unbounded input."""

    safe_names = []
    for key in sorted(keys, key=lambda item: str(item))[:8]:
        if type(key) is str:
            safe_names.append(repr(key[:64]))
        else:
            safe_names.append(f"<{type(key).__name__}>")
    suffix = ", ..." if len(keys) > 8 else ""
    return ", ".join(safe_names) + suffix


def _check_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise SubmissionValidationError(
            f"{path} is missing fields: {_safe_key_names(missing)}"
        )
    if extra:
        raise SubmissionValidationError(
            f"{path} has unexpected fields: {_safe_key_names(extra)}"
        )


def _check_string(
    value: Any,
    path: str,
    *,
    maximum: int,
    allow_empty: bool = False,
    forbid_whitespace: bool = False,
) -> str:
    if type(value) is not str:
        raise SubmissionValidationError(f"{path} must be a string")
    if not allow_empty and not value:
        raise SubmissionValidationError(f"{path} must not be empty")
    if len(value) > maximum:
        raise SubmissionValidationError(
            f"{path} exceeds the maximum length of {maximum}"
        )
    for char in value:
        codepoint = ord(char)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise SubmissionValidationError(
                f"{path} contains an invalid Unicode surrogate"
            )
        if (
            codepoint == 0
            or codepoint == 0x7F
            or (codepoint < 0x20 and char not in "\t\n\r")
        ):
            raise SubmissionValidationError(
                f"{path} contains unsafe control characters"
            )
    if forbid_whitespace and any(char.isspace() for char in value):
        raise SubmissionValidationError(f"{path} must not contain whitespace")
    return value


def _config_limits(config: Mapping[str, Any]) -> dict[str, int]:
    output = _config_dict(config.get("output"), "output")
    budget = _config_dict(config.get("budget"), "budget")
    limits = {
        "max_submission_bytes": _config_int(
            output.get("max_submission_bytes"),
            "output.max_submission_bytes",
            minimum=1,
            maximum=ABSOLUTE_MAX_SUBMISSION_BYTES,
        ),
        "max_smiles_length": _config_int(
            output.get("max_smiles_length"),
            "output.max_smiles_length",
            minimum=1,
            maximum=ABSOLUTE_MAX_STRING_LENGTH,
        ),
        "max_name_length": _config_int(
            output.get("max_name_length"),
            "output.max_name_length",
            minimum=1,
            maximum=ABSOLUTE_MAX_STRING_LENGTH,
        ),
        "max_rationale_length": _config_int(
            output.get("max_rationale_length"),
            "output.max_rationale_length",
            minimum=0,
            maximum=ABSOLUTE_MAX_STRING_LENGTH,
        ),
        "max_tool_calls": _config_int(
            budget.get("max_tool_calls"),
            "budget.max_tool_calls",
            minimum=0,
            maximum=ABSOLUTE_MAX_TOOL_CALL_COUNT,
        ),
        "min_tool_calls": _config_int(
            budget.get("min_tool_calls_for_eligibility"),
            "budget.min_tool_calls_for_eligibility",
            minimum=0,
            maximum=ABSOLUTE_MAX_TOOL_CALL_COUNT,
        ),
        "min_unique_candidates": _config_int(
            budget.get("min_unique_evaluated_candidates_for_eligibility"),
            "budget.min_unique_evaluated_candidates_for_eligibility",
            minimum=1,
            maximum=ABSOLUTE_MAX_LIST_ITEMS,
        ),
        "max_unique_candidates": _config_int(
            budget.get("max_unique_evaluated_candidates"),
            "budget.max_unique_evaluated_candidates",
            minimum=1,
            maximum=ABSOLUTE_MAX_LIST_ITEMS,
        ),
    }
    if limits["min_tool_calls"] > limits["max_tool_calls"]:
        raise EvaluatorConfigurationError(
            "budget.min_tool_calls_for_eligibility must not exceed "
            "budget.max_tool_calls"
        )
    if limits["min_unique_candidates"] > limits["max_unique_candidates"]:
        raise EvaluatorConfigurationError(
            "budget.min_unique_evaluated_candidates_for_eligibility must not "
            "exceed budget.max_unique_evaluated_candidates"
        )
    final_required = budget.get("final_candidate_must_be_tool_evaluated")
    if type(final_required) is not bool:
        raise EvaluatorConfigurationError(
            "budget.final_candidate_must_be_tool_evaluated must be a boolean"
        )
    limits["final_required"] = int(final_required)
    return limits


def load_config(
    config: str | os.PathLike[str] | Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load a trusted YAML config or validate a supplied config mapping."""

    if isinstance(config, Mapping):
        loaded: Any = config
    elif isinstance(config, (str, os.PathLike)):
        try:
            resolved = Path(config).expanduser().resolve(strict=True)
            resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as exc:
            raise EvaluatorConfigurationError(
                "config path must resolve inside the benchmark project"
            ) from exc
        try:
            if not resolved.is_file():
                raise EvaluatorConfigurationError(
                    "config path must be a regular file"
                )
            before_size = resolved.stat().st_size
            if before_size > MAX_CONFIG_BYTES:
                raise EvaluatorConfigurationError(
                    "config file exceeds the safe size limit"
                )
            raw = resolved.read_bytes()
            after_size = resolved.stat().st_size
            if len(raw) != before_size or len(raw) != after_size:
                raise EvaluatorConfigurationError(
                    "config file changed while it was being read"
                )
            if len(raw) > MAX_CONFIG_BYTES:
                raise EvaluatorConfigurationError(
                    "config file exceeds the safe size limit"
                )
            text = raw.decode("utf-8")
            loaded = yaml.safe_load(text)
        except EvaluatorConfigurationError:
            raise
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise EvaluatorConfigurationError(
                "config file could not be read as safe UTF-8 YAML"
            ) from exc
    else:
        raise EvaluatorConfigurationError("config must be a path or object")

    root = _config_dict(loaded, "config")
    task = _config_dict(root.get("task"), "task")
    output = _config_dict(root.get("output"), "output")
    task_id = task.get("id")
    schema_version = output.get("schema_version")
    if type(task_id) is not str or not task_id or len(task_id) > 256:
        raise EvaluatorConfigurationError(
            "task.id must be a non-empty bounded string"
        )
    if (
        type(schema_version) is not str
        or not schema_version
        or len(schema_version) > 32
    ):
        raise EvaluatorConfigurationError(
            "output.schema_version must be a non-empty bounded string"
        )
    start_smiles = task.get("start_smiles")
    if type(start_smiles) is not str or not start_smiles:
        raise EvaluatorConfigurationError(
            "task.start_smiles must be a non-empty string"
        )
    objective = _config_dict(task.get("objective"), "task.objective")
    if objective.get("property") != "qed":
        raise EvaluatorConfigurationError(
            "task.objective.property must be 'qed'"
        )
    if objective.get("direction") != "maximize":
        raise EvaluatorConfigurationError(
            "task.objective.direction must be 'maximize'"
        )
    if type(objective.get("require_strict_improvement")) is not bool:
        raise EvaluatorConfigurationError(
            "task.objective.require_strict_improvement must be a boolean"
        )
    _config_limits(root)
    return root


def _load_submission_file(
    path_value: str | os.PathLike[str], max_bytes: int
) -> dict[str, Any]:
    path = Path(path_value)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SubmissionValidationError(
            "submission file could not be read"
        ) from exc
    if size > max_bytes:
        raise SubmissionValidationError(
            f"submission file exceeds the maximum size of {max_bytes} bytes"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SubmissionValidationError(
            "submission file could not be read"
        ) from exc
    if len(raw) > max_bytes:
        raise SubmissionValidationError(
            f"submission file exceeds the maximum size of {max_bytes} bytes"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise SubmissionValidationError(
            "submission must be valid UTF-8 JSON"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJSONKeyError as exc:
        raise SubmissionValidationError(str(exc)) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, json.JSONDecodeError):
            message = (
                f"submission is malformed JSON at line {exc.lineno}, "
                f"column {exc.colno}"
            )
        else:
            message = "submission contains an invalid JSON value"
        raise SubmissionValidationError(message) from exc
    return _plain_dict(value, "submission")


def _validate_candidate_record(
    record: Any,
    path: str,
    *,
    max_name_length: int,
    max_smiles_length: int,
    max_rationale_length: int,
) -> dict[str, Any]:
    candidate = _plain_dict(record, path)
    _check_exact_keys(candidate, _CANDIDATE_KEYS, path)
    _check_string(
        candidate["name"], path + ".name", maximum=max_name_length
    )
    _check_string(
        candidate["smiles"],
        path + ".smiles",
        maximum=max_smiles_length,
        forbid_whitespace=True,
    )
    _check_string(
        candidate["rationale"],
        path + ".rationale",
        maximum=max_rationale_length,
        allow_empty=True,
    )
    return candidate


def validate_submission_schema(
    submission: Any, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact types, keys, lengths, and bounded collection sizes."""

    root = _plain_dict(submission, "submission")
    _check_exact_keys(root, _TOP_LEVEL_KEYS, "submission")
    limits = _config_limits(config)
    task = _config_dict(config.get("task"), "task")
    output = _config_dict(config.get("output"), "output")

    schema_version = _check_string(
        root["schema_version"], "submission.schema_version", maximum=32
    )
    if schema_version != output.get("schema_version"):
        raise SubmissionValidationError(
            "submission.schema_version does not match the task config"
        )
    task_id = _check_string(
        root["task_id"], "submission.task_id", maximum=256
    )
    if task_id != task.get("id"):
        raise SubmissionValidationError(
            "submission.task_id does not match the task config"
        )

    _validate_candidate_record(
        root["final_candidate"],
        "submission.final_candidate",
        max_name_length=limits["max_name_length"],
        max_smiles_length=limits["max_smiles_length"],
        max_rationale_length=limits["max_rationale_length"],
    )

    history = root["candidate_history"]
    if type(history) is not list:
        raise SubmissionValidationError(
            "submission.candidate_history must be a list"
        )
    if len(history) > limits["max_unique_candidates"]:
        raise SubmissionValidationError(
            "submission.candidate_history exceeds "
            "budget.max_unique_evaluated_candidates"
        )
    for index, record in enumerate(history):
        _validate_candidate_record(
            record,
            f"submission.candidate_history[{index}]",
            max_name_length=limits["max_name_length"],
            max_smiles_length=limits["max_smiles_length"],
            max_rationale_length=limits["max_rationale_length"],
        )

    metadata = _plain_dict(
        root["run_metadata"], "submission.run_metadata"
    )
    _check_exact_keys(
        metadata, _RUN_METADATA_KEYS, "submission.run_metadata"
    )
    call_count = metadata["tool_call_count"]
    if type(call_count) is not int:
        raise SubmissionValidationError(
            "submission.run_metadata.tool_call_count must be an integer"
        )
    if call_count < 0 or call_count > ABSOLUTE_MAX_TOOL_CALL_COUNT:
        raise SubmissionValidationError(
            "submission.run_metadata.tool_call_count must be between 0 and "
            f"{ABSOLUTE_MAX_TOOL_CALL_COUNT}"
        )
    if type(metadata["budget_exceeded"]) is not bool:
        raise SubmissionValidationError(
            "submission.run_metadata.budget_exceeded must be a boolean"
        )
    if type(metadata["final_candidate_evaluated"]) is not bool:
        raise SubmissionValidationError(
            "submission.run_metadata.final_candidate_evaluated must be a boolean"
        )
    listed_unique = metadata["unique_evaluated_candidates"]
    if type(listed_unique) is not list:
        raise SubmissionValidationError(
            "submission.run_metadata.unique_evaluated_candidates must be a list"
        )
    if len(listed_unique) > limits["max_unique_candidates"]:
        raise SubmissionValidationError(
            "submission.run_metadata.unique_evaluated_candidates exceeds "
            "the configured maximum"
        )
    for index, smiles in enumerate(listed_unique):
        _check_string(
            smiles,
            "submission.run_metadata.unique_evaluated_candidates"
            f"[{index}]",
            maximum=limits["max_smiles_length"],
            forbid_whitespace=True,
        )
    return root


def _round_observation(value: float | None) -> float | None:
    if value is None:
        return None
    return float(f"{value:.6f}")


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Bound and normalize trusted tool output before serializing it."""

    if depth > 8:
        return "<maximum depth reached>"
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in list(value.items())[:128]:
            if type(key) is str:
                result[key[:128]] = _json_safe(child, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(child, depth=depth + 1) for child in value[:128]
        ]
    return str(value)[:256]


def _candidate_error_view(smiles: str, error_kind: str) -> dict[str, Any]:
    return {
        "input_smiles": smiles,
        "valid": False,
        "canonical_smiles": None,
        "qed": None,
        "feasible": False,
        "properties": {},
        "constraint_flags": {},
        "similarity_to_start": None,
        "errors": [error_kind],
    }


def _evaluate_smiles(
    smiles: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        raw = evaluate_candidate(smiles, config)
    except Exception:
        # A malformed molecule must never crash the evaluator, even if a custom
        # RDKit build raises unexpectedly.
        return _candidate_error_view(
            smiles, "candidate evaluation failed safely"
        )
    if type(raw) is not dict:
        return _candidate_error_view(
            smiles, "candidate evaluator returned an invalid result"
        )

    canonical = raw.get("canonical_smiles")
    if type(canonical) is not str or not canonical:
        canonical = None
    qed_value = raw.get("qed")
    if isinstance(qed_value, bool) or not isinstance(qed_value, (int, float)):
        qed: float | None = None
    else:
        numeric_qed = float(qed_value)
        qed = (
            numeric_qed
            if math.isfinite(numeric_qed) and 0.0 <= numeric_qed <= 1.0
            else None
        )
    valid_raw = raw.get("valid")
    valid = (
        valid_raw is True
        if type(valid_raw) is bool
        else canonical is not None
    )
    valid = bool(valid and canonical is not None and qed is not None)
    feasible = raw.get("feasible") is True and valid
    flags = raw.get("constraint_flags")
    if not isinstance(flags, Mapping):
        flags = raw.get("constraint_checks")
    if not isinstance(flags, Mapping):
        flags = {}
    properties = (
        raw.get("properties")
        if isinstance(raw.get("properties"), Mapping)
        else {}
    )
    similarity = raw.get("similarity_to_start")
    if (
        isinstance(similarity, bool)
        or not isinstance(similarity, (int, float))
        or not math.isfinite(float(similarity))
    ):
        similarity = None
    else:
        similarity = _round_observation(float(similarity))
    errors = raw.get("errors")
    if not isinstance(errors, list):
        single_error = raw.get("error")
        errors = (
            [single_error]
            if type(single_error) is str and single_error
            else []
        )

    return {
        "input_smiles": smiles,
        "valid": valid,
        "canonical_smiles": canonical,
        "qed": _round_observation(qed),
        "feasible": feasible,
        "properties": _json_safe(properties),
        "constraint_flags": _json_safe(flags),
        "similarity_to_start": similarity,
        "errors": _json_safe(errors),
    }


def _invalid_process_evidence(
    source: str, message: str, *, present: bool = True
) -> _ProcessEvidence:
    return _ProcessEvidence(
        source=source,
        present=present,
        valid=False,
        event_count=0,
        tool_call_count=0,
        successful_public_tool_call_count=0,
        rejected_tool_call_count=0,
        evaluate_candidate_attempt_count=0,
        successful_evaluate_candidate_call_count=0,
        unique_evaluated_candidates=(),
        budget_exceeded=False,
        public_tool_events_valid=False,
        lifecycle_valid=False,
        run_completed=False,
        terminal_event=None,
        error=message,
    )


def _validate_json_tree(
    value: Any,
    *,
    depth: int = 0,
    node_counter: list[int] | None = None,
) -> None:
    """Reject non-JSON and excessively nested in-memory event values."""

    if node_counter is None:
        node_counter = [0]
    node_counter[0] += 1
    if node_counter[0] > MAX_JSON_NODES_PER_EVENT:
        raise ProcessEvidenceValidationError(
            "process event exceeds the JSON node limit"
        )
    if depth > MAX_JSON_DEPTH:
        raise ProcessEvidenceValidationError(
            "process event exceeds the JSON nesting limit"
        )
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ProcessEvidenceValidationError(
                "process event contains a non-finite number"
            )
        return
    if type(value) is str:
        if len(value) > ABSOLUTE_MAX_STRING_LENGTH:
            raise ProcessEvidenceValidationError(
                "process event contains an oversized string"
            )
        for char in value:
            codepoint = ord(char)
            if 0xD800 <= codepoint <= 0xDFFF:
                raise ProcessEvidenceValidationError(
                    "process event contains an invalid Unicode surrogate"
                )
            if (
                codepoint == 0
                or codepoint == 0x7F
                or (codepoint < 0x20 and char not in "\t\n\r")
            ):
                raise ProcessEvidenceValidationError(
                    "process event contains an unsafe control character"
                )
        return
    if type(value) is list:
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ProcessEvidenceValidationError(
                "process event contains an oversized array"
            )
        for child in value:
            _validate_json_tree(
                child, depth=depth + 1, node_counter=node_counter
            )
        return
    if type(value) is dict:
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ProcessEvidenceValidationError(
                "process event contains an oversized object"
            )
        for key, child in value.items():
            if type(key) is not str or len(key) > 256:
                raise ProcessEvidenceValidationError(
                    "process event contains an invalid object key"
                )
            _validate_json_tree(
                child, depth=depth + 1, node_counter=node_counter
            )
        return
    raise ProcessEvidenceValidationError(
        "process event contains a non-JSON value"
    )


def _validate_in_memory_events(process_events: Any) -> list[dict[str, Any]]:
    if type(process_events) is not list:
        raise ProcessEvidenceValidationError(
            "process_events must be a list of JSON objects"
        )
    if not process_events:
        raise ProcessEvidenceValidationError("process evidence contains no events")
    if len(process_events) > MAX_TRAJECTORY_LINES:
        raise ProcessEvidenceValidationError(
            "process evidence exceeds the event-count limit"
        )

    total_bytes = 0
    events: list[dict[str, Any]] = []
    for value in process_events:
        if type(value) is not dict:
            raise ProcessEvidenceValidationError(
                "each process event must be a JSON object"
            )
        _validate_json_tree(value)
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ProcessEvidenceValidationError(
                "process event cannot be represented as safe UTF-8 JSON"
            ) from exc
        if len(encoded) > MAX_TRAJECTORY_LINE_BYTES:
            raise ProcessEvidenceValidationError(
                "process event exceeds the per-event byte limit"
            )
        total_bytes += len(encoded) + 1
        if total_bytes > MAX_TRAJECTORY_BYTES:
            raise ProcessEvidenceValidationError(
                "process evidence exceeds the byte limit"
            )
        events.append(value)
    return events


def _load_trajectory_jsonl(
    trajectory_path: str | os.PathLike[str],
    submission_path: str | os.PathLike[str] | None,
) -> list[dict[str, Any]]:
    """Read only a sibling trajectory, with bounded strict JSONL parsing."""

    if submission_path is None:
        raise ProcessEvidenceValidationError(
            "trajectory_path requires a file-based submission"
        )
    try:
        submission_lexical = Path(submission_path).absolute()
        submission_resolved = Path(submission_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProcessEvidenceValidationError(
            "submission path cannot anchor trajectory access"
        ) from exc

    requested = Path(trajectory_path)
    try:
        # Explicit paths follow normal CLI/Python semantics (relative to the
        # current working directory); they must nevertheless resolve beside
        # the submission before any trajectory bytes are read.
        requested_lexical = requested.absolute()
    except (OSError, RuntimeError) as exc:
        raise ProcessEvidenceValidationError(
            "trajectory path is invalid"
        ) from exc
    if requested_lexical.parent != submission_lexical.parent:
        raise ProcessEvidenceValidationError(
            "trajectory path must stay in the submission directory"
        )

    try:
        resolved = requested_lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProcessEvidenceValidationError(
            "trajectory file could not be read"
        ) from exc
    if resolved.parent != submission_resolved.parent:
        raise ProcessEvidenceValidationError(
            "trajectory path resolves outside the submission directory"
        )
    try:
        if not resolved.is_file():
            raise ProcessEvidenceValidationError(
                "trajectory path is not a regular file"
            )
        size = resolved.stat().st_size
        if size > MAX_TRAJECTORY_BYTES:
            raise ProcessEvidenceValidationError(
                "trajectory file exceeds the byte limit"
            )
        raw = resolved.read_bytes()
    except ProcessEvidenceValidationError:
        raise
    except OSError as exc:
        raise ProcessEvidenceValidationError(
            "trajectory file could not be read"
        ) from exc
    if len(raw) > MAX_TRAJECTORY_BYTES:
        raise ProcessEvidenceValidationError(
            "trajectory file exceeds the byte limit"
        )

    raw_lines = raw.split(b"\n")
    if raw.endswith(b"\n"):
        raw_lines.pop()
    if not raw_lines:
        raise ProcessEvidenceValidationError("trajectory contains no events")
    if len(raw_lines) > MAX_TRAJECTORY_LINES:
        raise ProcessEvidenceValidationError(
            "trajectory exceeds the line-count limit"
        )

    events: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        if not raw_line:
            raise ProcessEvidenceValidationError(
                "trajectory contains a blank JSONL line"
            )
        if len(raw_line) > MAX_TRAJECTORY_LINE_BYTES:
            raise ProcessEvidenceValidationError(
                "trajectory line exceeds the byte limit"
            )
        try:
            text = raw_line.decode("utf-8")
            value = json.loads(
                text,
                object_pairs_hook=_duplicate_rejecting_object,
                parse_constant=_reject_json_constant,
            )
        except _DuplicateJSONKeyError as exc:
            raise ProcessEvidenceValidationError(
                "trajectory JSON objects must not contain duplicate keys"
            ) from exc
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProcessEvidenceValidationError(
                "trajectory must contain strict UTF-8 JSONL"
            ) from exc
        if type(value) is not dict:
            raise ProcessEvidenceValidationError(
                "each trajectory line must be a JSON object"
            )
        _validate_json_tree(value)
        events.append(value)
    return events


def _normalized_json_text(value: Any) -> str:
    """Return a type-sensitive, key-order-independent JSON representation."""

    _validate_json_tree(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProcessEvidenceValidationError(
            "public tool result is not normalized JSON"
        ) from exc


def _validated_public_tool_arguments(
    tool: str, arguments: Any
) -> dict[str, str]:
    schema = PUBLIC_TOOL_SCHEMAS[tool]
    if type(arguments) is not dict:
        raise ProcessEvidenceValidationError(
            "successful public tool event has no arguments object"
        )
    actual = set(arguments)
    required = set(schema["required"])
    allowed = required | set(schema["optional"])
    if not required.issubset(actual) or not actual.issubset(allowed):
        raise ProcessEvidenceValidationError(
            "successful public tool event has non-schema arguments"
        )
    validated: dict[str, str] = {}
    for key in actual:
        value = arguments[key]
        if (
            type(value) is not str
            or not value.strip()
            or len(value) > PUBLIC_TOOL_MAX_ARGUMENT_LENGTH
        ):
            raise ProcessEvidenceValidationError(
                "successful public tool event has invalid argument values"
            )
        validated[key] = value
    return validated


def _matching_recomputed_public_tool_result(
    tool: str,
    arguments: dict[str, str],
    observed_result: Any,
    config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the recomputed structured result only when replay matches.

    Matching an observed payload proves deterministic replay, but it does not
    by itself make the tool call successful: callers must separately require
    the recomputed result's public ``ok`` field to be exactly ``True``.
    """

    if not isinstance(observed_result, Mapping):
        return None
    function = TOOL_REGISTRY.get(tool)
    if not callable(function):
        return None
    try:
        recomputed = function(config=config, **arguments)
    except Exception:
        return None
    if not isinstance(recomputed, Mapping):
        return None
    try:
        observed_text = _normalized_json_text(dict(observed_result))
        recomputed_text = _normalized_json_text(dict(recomputed))
        matches = observed_text == recomputed_text
    except ProcessEvidenceValidationError:
        return None
    return recomputed if matches else None


def _derive_process_evidence(
    events: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    source: str,
) -> _ProcessEvidence:
    limits = _config_limits(config)
    call_ids: set[str] = set()
    unique_candidates: list[str] = []
    unique_set: set[str] = set()
    tool_call_count = 0
    successful_public_tool_count = 0
    rejected_count = 0
    evaluate_attempt_count = 0
    successful_evaluate_count = 0
    budget_exceeded = False
    tool_issues: list[str] = []
    lifecycle_issues: list[str] = []
    run_start_positions: list[int] = []
    terminal_positions: list[tuple[int, str]] = []

    def add_tool_issue(message: str) -> None:
        if message not in tool_issues:
            tool_issues.append(message)

    for expected_sequence, event in enumerate(events, 1):
        sequence = event.get("sequence")
        event_type = event.get("event")
        if type(sequence) is not int or sequence != expected_sequence:
            raise ProcessEvidenceValidationError(
                "process event sequence must be contiguous and start at one"
            )
        if type(event_type) is not str or not event_type or len(event_type) > 128:
            raise ProcessEvidenceValidationError(
                "process event has an invalid event type"
            )
        if event_type == "run_start":
            run_start_positions.append(expected_sequence)
        if event_type in {"run_complete", "run_failure"}:
            terminal_positions.append((expected_sequence, event_type))
        if event_type == "run_failure":
            failure = event.get("error")
            if type(failure) is dict and failure.get("type") == "BudgetExceededError":
                budget_exceeded = True
        if event_type != "tool_call":
            continue

        call_id = event.get("call_id")
        tool = event.get("tool")
        status = event.get("status")
        call_id_valid = bool(
            type(call_id) is str and call_id and len(call_id) <= 256
        )
        if not call_id_valid:
            add_tool_issue("tool_call event has an invalid call_id")
        elif call_id in call_ids:
            call_id_valid = False
            add_tool_issue("tool_call event reuses a call_id")
        else:
            call_ids.add(call_id)
        tool_valid = bool(
            type(tool) is str and tool and len(tool) <= 128
        )
        if not tool_valid:
            add_tool_issue("tool_call event has an invalid tool name")
        status_valid = status in {"SUCCESS", "FAILED", "REJECTED"}
        if not status_valid:
            add_tool_issue("tool_call event has an invalid status")

        # Mirror AgentRunner accounting: accepted calls (including malformed
        # arguments and tool failures) consume budget. The outer max-call guard
        # rejection is the sole event that does not increment run_metadata.
        counted = True
        if status == "REJECTED":
            rejected_count += 1
            error = event.get("error")
            error_type = error.get("type") if type(error) is dict else None
            if error_type == "ToolBudgetExceeded":
                counted = False
            elif error_type == "CandidateBudgetExceeded":
                counted = True
            else:
                add_tool_issue(
                    "rejected tool_call has an unknown rejection reason"
                )
            budget_exceeded = True
        if counted:
            tool_call_count += 1
            if tool_call_count > ABSOLUTE_MAX_TOOL_CALL_COUNT:
                raise ProcessEvidenceValidationError(
                    "trajectory tool-call count exceeds the absolute limit"
                )

        if tool != "evaluate_candidate":
            is_evaluate_candidate = False
        else:
            is_evaluate_candidate = True
            evaluate_attempt_count += 1
        if status != "SUCCESS":
            continue

        if not tool_valid or tool not in SUPPORTED_PUBLIC_TOOLS:
            add_tool_issue("SUCCESS event names an unsupported public tool")
            continue
        if not call_id_valid:
            continue
        try:
            arguments = _validated_public_tool_arguments(
                tool, event.get("arguments")
            )
        except ProcessEvidenceValidationError as exc:
            add_tool_issue(str(exc))
            continue
        result = event.get("result")
        recomputed_result = _matching_recomputed_public_tool_result(
            tool, arguments, result, config
        )
        if recomputed_result is None:
            add_tool_issue(
                "successful public tool result does not match deterministic recomputation"
            )
            continue

        # Runner SUCCESS means the invocation completed. The benchmark's
        # minimum-success gate is stricter: only a deterministically replayed
        # structured public-tool result with an explicit ``ok: true`` counts.
        # A replayed ``ok: false`` remains valid evidence and consumes budget.
        if recomputed_result.get("ok") is not True:
            continue
        successful_public_tool_count += 1
        if not is_evaluate_candidate:
            continue
        # Candidate evidence additionally requires the tool's explicit
        # validity assertion before it can affect history, exploration, or the
        # final-candidate gate.
        if recomputed_result.get("valid") is not True:
            continue
        canonical = recomputed_result.get("canonical_smiles")
        if (
            type(canonical) is not str
            or not canonical
            or len(canonical) > limits["max_smiles_length"]
            or any(char.isspace() for char in canonical)
        ):
            add_tool_issue(
                "successful evaluate_candidate result has invalid canonical_smiles"
            )
            continue
        successful_evaluate_count += 1
        if canonical not in unique_set:
            unique_set.add(canonical)
            unique_candidates.append(canonical)

    if run_start_positions != [1]:
        lifecycle_issues.append(
            "process lifecycle requires exactly one first run_start"
        )
    if len(terminal_positions) != 1:
        lifecycle_issues.append(
            "process lifecycle requires exactly one terminal event"
        )
    elif terminal_positions[0][0] != len(events):
        lifecycle_issues.append("process lifecycle has events after terminal")
    lifecycle_valid = not lifecycle_issues
    terminal_event = (
        terminal_positions[0][1] if len(terminal_positions) == 1 else None
    )
    run_completed = bool(
        lifecycle_valid and terminal_event == "run_complete"
    )
    issues = tuple(lifecycle_issues + tool_issues)

    return _ProcessEvidence(
        source=source,
        present=True,
        valid=True,
        event_count=len(events),
        tool_call_count=tool_call_count,
        successful_public_tool_call_count=successful_public_tool_count,
        rejected_tool_call_count=rejected_count,
        evaluate_candidate_attempt_count=evaluate_attempt_count,
        successful_evaluate_candidate_call_count=successful_evaluate_count,
        unique_evaluated_candidates=tuple(unique_candidates),
        budget_exceeded=budget_exceeded,
        public_tool_events_valid=not tool_issues,
        lifecycle_valid=lifecycle_valid,
        run_completed=run_completed,
        terminal_event=terminal_event,
        issues=issues,
    )


def _obtain_process_evidence(
    *,
    process_events: Any,
    trajectory_path: str | os.PathLike[str] | None,
    submission_path: str | os.PathLike[str] | None,
    config: Mapping[str, Any],
) -> _ProcessEvidence:
    if process_events is not None and trajectory_path is not None:
        return _invalid_process_evidence(
            "conflicting_inputs",
            "process_events and trajectory_path are mutually exclusive",
        )

    source = "none"
    try:
        if process_events is not None:
            source = "process_events"
            events = _validate_in_memory_events(process_events)
        else:
            selected_path = trajectory_path
            if selected_path is None and submission_path is not None:
                auto_path = Path(submission_path).parent / "trajectory.jsonl"
                try:
                    if not auto_path.exists():
                        return _invalid_process_evidence(
                            "none",
                            "verifiable process evidence was not supplied",
                            present=False,
                        )
                except OSError as exc:
                    raise ProcessEvidenceValidationError(
                        "trajectory auto-discovery failed safely"
                    ) from exc
                selected_path = auto_path
                source = "auto_trajectory"
            elif selected_path is not None:
                source = "trajectory"
            else:
                return _invalid_process_evidence(
                    "none",
                    "verifiable process evidence was not supplied",
                    present=False,
                )
            events = _load_trajectory_jsonl(selected_path, submission_path)
        # Normalize the public source for deterministic re-evaluation: the
        # same events supplied in memory by AgentRunner and later read from its
        # JSONL artifact must produce the same evaluation object.
        return _derive_process_evidence(
            events, config, source="validated_process_events"
        )
    except ProcessEvidenceValidationError as exc:
        return _invalid_process_evidence(source, str(exc))
    except Exception:
        return _invalid_process_evidence(
            source, "process evidence validation failed safely"
        )


def _process_evidence_view(
    evidence: _ProcessEvidence, *, required: bool
) -> dict[str, Any]:
    return {
        "required": required,
        "source": evidence.source,
        "present": evidence.present,
        "valid": evidence.valid,
        "event_count": evidence.event_count,
        "tool_call_count": evidence.tool_call_count,
        "successful_public_tool_call_count": (
            evidence.successful_public_tool_call_count
        ),
        "rejected_tool_call_count": evidence.rejected_tool_call_count,
        "evaluate_candidate_attempt_count": (
            evidence.evaluate_candidate_attempt_count
        ),
        "successful_evaluate_candidate_call_count": (
            evidence.successful_evaluate_candidate_call_count
        ),
        "unique_evaluated_candidates": list(
            evidence.unique_evaluated_candidates
        ),
        "budget_exceeded": evidence.budget_exceeded,
        "public_tool_events_valid": evidence.public_tool_events_valid,
        "lifecycle_valid": evidence.lifecycle_valid,
        "run_completed": evidence.run_completed,
        "terminal_event": evidence.terminal_event,
        "issues": list(evidence.issues),
        "errors": [evidence.error] if evidence.error else [],
    }


def _base_result(
    status: str, task_id: str | None, errors: list[str]
) -> dict[str, Any]:
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "status": status,
        "task_id": task_id,
        "valid_submission": False,
        "eligible": False,
        "hard_constraints_passed": False,
        "objective_passed": False,
        "process_eligible": False,
        "score": 0.0,
        "total_score": 0.0,
        "score_breakdown": {
            "qed_absolute": 0.0,
            "qed_improvement": 0.0,
            "exploration": 0.0,
        },
        "errors": errors,
    }


def _invalid_result(
    config: Mapping[str, Any] | None, message: str
) -> dict[str, Any]:
    task_id: str | None = None
    if isinstance(config, Mapping) and isinstance(
        config.get("task"), Mapping
    ):
        configured_id = config["task"].get("id")
        task_id = configured_id if type(configured_id) is str else None
    return _base_result("invalid_submission", task_id, [message])


def _configuration_error_result(message: str) -> dict[str, Any]:
    return _base_result("evaluation_error", None, [message])


def _perform_evaluation(
    submission: dict[str, Any],
    config: Mapping[str, Any],
    evidence: _ProcessEvidence,
    *,
    require_process_evidence: bool,
) -> dict[str, Any]:
    limits = _config_limits(config)
    task = _config_dict(config.get("task"), "task")
    metadata = submission["run_metadata"]

    start_view = _evaluate_smiles(task["start_smiles"], config)
    if not start_view["valid"] or start_view["qed"] is None:
        raise EvaluatorConfigurationError(
            "the configured start molecule cannot be evaluated"
        )

    final_record = submission["final_candidate"]
    final_view = _evaluate_smiles(final_record["smiles"], config)
    final_view["name"] = final_record["name"]

    history_views: list[dict[str, Any]] = []
    canonical_first_index: dict[str, int] = {}
    actual_unique_canonical: list[str] = []
    duplicates: list[dict[str, Any]] = []
    history_feasible_canonical: set[str] = set()
    for index, record in enumerate(submission["candidate_history"]):
        view = _evaluate_smiles(record["smiles"], config)
        view["name"] = record["name"]
        history_views.append(view)
        canonical = view["canonical_smiles"]
        if canonical is None:
            continue
        if canonical in canonical_first_index:
            duplicates.append(
                {
                    "canonical_smiles": canonical,
                    "first_index": canonical_first_index[canonical],
                    "duplicate_index": index,
                }
            )
        else:
            canonical_first_index[canonical] = index
            actual_unique_canonical.append(canonical)
        if view["feasible"]:
            history_feasible_canonical.add(canonical)

    listed_unique = metadata["unique_evaluated_candidates"]
    evidence_usable = evidence.present and evidence.valid
    evidence_unique = list(evidence.unique_evaluated_candidates)
    tool_call_count = evidence.tool_call_count if evidence_usable else 0
    successful_public_tool_call_count = (
        evidence.successful_public_tool_call_count if evidence_usable else 0
    )
    max_calls = limits["max_tool_calls"]
    minimum_calls = limits["min_tool_calls"]
    minimum_unique = limits["min_unique_candidates"]
    history_all_valid = all(view["valid"] for view in history_views)
    metadata_matches_history = listed_unique == actual_unique_canonical
    evidence_matches_history = bool(
        evidence_usable and evidence_unique == actual_unique_canonical
    )
    evidence_unique_matches_metadata = bool(
        evidence_usable and evidence_unique == listed_unique
    )
    evidence_count_matches_metadata = bool(
        evidence_usable and tool_call_count == metadata["tool_call_count"]
    )
    evidence_budget_matches_metadata = bool(
        evidence_usable
        and evidence.budget_exceeded == metadata["budget_exceeded"]
    )
    tool_calls_cover_history = tool_call_count >= len(
        submission["candidate_history"]
    )
    unique_within_budget = (
        evidence_usable
        and len(evidence_unique) <= limits["max_unique_candidates"]
    )
    minimum_calls_met = bool(
        evidence_usable
        and successful_public_tool_call_count >= minimum_calls
    )
    minimum_unique_candidates_met = bool(
        evidence_usable and len(evidence_unique) >= minimum_unique
    )
    count_within_budget = tool_call_count <= max_calls
    within_tool_budget = (
        evidence_usable and count_within_budget and not evidence.budget_exceeded
    )
    budget_flag_consistent = bool(
        evidence_usable
        and not (tool_call_count > max_calls and not evidence.budget_exceeded)
    )

    final_canonical = final_view["canonical_smiles"]
    final_verified = bool(
        evidence_usable
        and final_canonical is not None
        and final_canonical in evidence_unique
    )
    evidence_final_matches_metadata = bool(
        evidence_usable
        and final_verified == metadata["final_candidate_evaluated"]
    )
    final_required = bool(limits["final_required"])
    final_gate_passed = final_verified or not final_required

    process_checks = {
        "process_evidence_present": evidence.present,
        "process_evidence_valid": evidence.valid,
        "process_lifecycle_valid": bool(
            evidence_usable and evidence.lifecycle_valid
        ),
        "process_run_completed": bool(
            evidence_usable and evidence.run_completed
        ),
        "public_tool_events_valid": bool(
            evidence_usable and evidence.public_tool_events_valid
        ),
        "minimum_calls_met": minimum_calls_met,
        "minimum_unique_evaluated_candidates_met": (
            minimum_unique_candidates_met
        ),
        "within_tool_budget": within_tool_budget,
        "budget_flag_consistent": budget_flag_consistent,
        "tool_calls_cover_history": tool_calls_cover_history,
        "history_candidates_valid": history_all_valid,
        "history_candidates_canonically_unique": not duplicates,
        "unique_candidates_within_budget": unique_within_budget,
        "metadata_matches_canonical_history": metadata_matches_history,
        "process_candidates_match_canonical_history": evidence_matches_history,
        "process_tool_call_count_matches_metadata": (
            evidence_count_matches_metadata
        ),
        "process_candidates_match_metadata": evidence_unique_matches_metadata,
        "process_budget_status_matches_metadata": (
            evidence_budget_matches_metadata
        ),
        "process_final_evaluation_matches_metadata": (
            evidence_final_matches_metadata
        ),
        "final_candidate_evaluated": final_verified,
    }
    process_eligible = all(
        (
            evidence_usable,
            evidence.lifecycle_valid,
            evidence.run_completed,
            evidence.public_tool_events_valid,
            minimum_calls_met,
            minimum_unique_candidates_met,
            within_tool_budget,
            budget_flag_consistent,
            tool_calls_cover_history,
            history_all_valid,
            not duplicates,
            unique_within_budget,
            metadata_matches_history,
            evidence_matches_history,
            evidence_count_matches_metadata,
            evidence_unique_matches_metadata,
            evidence_budget_matches_metadata,
            evidence_final_matches_metadata,
            final_gate_passed,
        )
    )
    hard_constraints_passed = bool(
        final_view["valid"] and final_view["feasible"]
    )

    objective = _config_dict(task.get("objective"), "task.objective")
    strict_improvement_required = objective["require_strict_improvement"]
    qed_strictly_improved = bool(
        final_view["qed"] is not None
        and final_view["qed"] > start_view["qed"]
    )
    objective_passed = bool(
        qed_strictly_improved or not strict_improvement_required
    )

    final_qed_for_scoring = (
        final_view["qed"] if final_view["qed"] is not None else 0.0
    )
    scoring_cfg = _config_dict(config.get("scoring"), "scoring")
    revision_cfg = scoring_cfg.get("feedback_revision", {})
    if not isinstance(revision_cfg, Mapping):
        revision_cfg = {}
    minimum_revision_gain = revision_cfg.get("minimum_qed_gain_over_prior", 0.0)
    if isinstance(minimum_revision_gain, bool) or not isinstance(
        minimum_revision_gain, (int, float)
    ):
        raise EvaluatorConfigurationError(
            "scoring.feedback_revision.minimum_qed_gain_over_prior must be numeric"
        )
    feedback_revision_quality = False
    if final_canonical in actual_unique_canonical:
        final_index = actual_unique_canonical.index(final_canonical)
        prior_views = history_views[:final_index]
        if final_view["qed"] is not None:
            feedback_revision_quality = any(
                view.get("qed") is not None
                and final_view["qed"]
                >= float(view["qed"]) + float(minimum_revision_gain)
                for view in prior_views
            )
    evidence_feasible_canonical: set[str] = set()
    if evidence_usable:
        for canonical in evidence_unique:
            if _evaluate_smiles(canonical, config)["feasible"]:
                evidence_feasible_canonical.add(canonical)

    score = compute_score(
        final_qed=final_qed_for_scoring,
        start_qed=start_view["qed"],
        unique_feasible_candidates=len(evidence_feasible_canonical),
        process_checks=process_checks,
        hard_constraints_passed=hard_constraints_passed,
        process_eligible=process_eligible,
        config=config,
        objective_passed=objective_passed,
        feedback_revision_quality=feedback_revision_quality,
    )

    violations: list[str] = []
    if not hard_constraints_passed:
        violations.append(
            "final candidate failed validity or hard-constraint checks"
        )
    if not objective_passed:
        violations.append(
            "final QED must be strictly greater than initial QED"
        )
    if not evidence.present:
        mode = "required" if require_process_evidence else "diagnostic"
        violations.append(
            f"verifiable process evidence was not supplied ({mode} mode)"
        )
    elif not evidence.valid:
        violations.append(
            "process evidence is invalid: "
            + (evidence.error or "validation failed safely")
        )
    if evidence_usable and not evidence.lifecycle_valid:
        violations.append("process event lifecycle is invalid")
    if evidence_usable and evidence.lifecycle_valid and not evidence.run_completed:
        violations.append(
            "process terminated without a successful run_complete"
        )
    if evidence_usable and not evidence.public_tool_events_valid:
        violations.append(
            "one or more successful public tool events failed validation"
        )
    if not minimum_calls_met:
        violations.append("minimum tool-call requirement was not met")
    if not minimum_unique_candidates_met:
        violations.append(
            "minimum unique successful evaluate_candidate requirement was not met"
        )
    if not within_tool_budget:
        violations.append(
            "tool-call budget was exceeded or reported exceeded"
        )
    if not budget_flag_consistent:
        violations.append(
            "budget_exceeded is inconsistent with tool_call_count"
        )
    if not tool_calls_cover_history:
        violations.append(
            "tool_call_count is smaller than candidate_history"
        )
    if not history_all_valid:
        violations.append(
            "candidate_history contains an invalid molecule"
        )
    if duplicates:
        violations.append(
            "candidate_history contains canonical duplicate molecules"
        )
    if not unique_within_budget:
        violations.append("unique evaluated candidate budget was exceeded")
    if not metadata_matches_history:
        violations.append(
            "unique_evaluated_candidates does not match canonical "
            "candidate_history"
        )
    if evidence_usable and not evidence_matches_history:
        violations.append(
            "process-derived candidates do not match canonical candidate_history"
        )
    if evidence_usable and not evidence_count_matches_metadata:
        violations.append(
            "process-derived tool_call_count does not match run_metadata"
        )
    if evidence_usable and not evidence_unique_matches_metadata:
        violations.append(
            "process-derived candidates do not match run_metadata"
        )
    if evidence_usable and not evidence_budget_matches_metadata:
        violations.append(
            "process-derived budget status does not match run_metadata"
        )
    if evidence_usable and not evidence_final_matches_metadata:
        violations.append(
            "process-derived final evaluation status does not match run_metadata"
        )
    if not final_gate_passed:
        violations.append(
            "final candidate was not verifiably tool evaluated"
        )

    total_score = score["total_score"]
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "ok",
        "task_id": task["id"],
        "valid_submission": True,
        "eligible": bool(score["gate_passed"] and process_eligible),
        "hard_constraints_passed": hard_constraints_passed,
        "objective_passed": objective_passed,
        "process_eligible": process_eligible,
        "score": total_score,
        "total_score": total_score,
        "score_breakdown": score["components"],
        "ungated_score": score["ungated_score"],
        "metrics": {
            "start_qed": start_view["qed"],
            "final_qed": final_view["qed"],
            "qed_delta": score["qed_delta"],
            "unique_history_candidates": len(actual_unique_canonical),
            "unique_feasible_history_candidates": len(
                history_feasible_canonical
            ),
            "unique_process_candidates": len(evidence_unique),
            "unique_feasible_process_candidates": len(
                evidence_feasible_canonical
            ),
            "tool_call_count": tool_call_count,
            "successful_public_tool_call_count": (
                successful_public_tool_call_count
            ),
            "reported_tool_call_count": metadata["tool_call_count"],
            "max_tool_calls": max_calls,
        },
        "process_checks": process_checks,
        "process_evidence": _process_evidence_view(
            evidence, required=require_process_evidence
        ),
        "duplicate_candidates": duplicates,
        "start_candidate": start_view,
        "final_candidate": final_view,
        "candidate_history_evaluations": history_views,
        "violations": violations,
        "errors": [],
        "rounding_decimal_places": score["rounding_decimal_places"],
    }


def write_evaluation(
    result: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> None:
    """Write stable UTF-8 JSON, creating only the requested parent directory."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    path.write_text(serialized, encoding="utf-8", newline="\n")


def evaluate_submission(
    submission: str | os.PathLike[str] | Mapping[str, Any],
    config: str | os.PathLike[str] | Mapping[str, Any],
    output_path: str | os.PathLike[str] | None = None,
    *,
    process_events: list[dict[str, Any]] | None = None,
    trajectory_path: str | os.PathLike[str] | None = None,
    require_process_evidence: bool = True,
) -> dict[str, Any]:
    """Evaluate a submission mapping or JSON file and optionally write JSON.

    ``process_events`` is the preferred runner interface. File-based callers
    may pass a sibling ``trajectory_path``; otherwise ``trajectory.jsonl`` is
    auto-discovered beside a file submission. Process metadata is compared to,
    never substituted for, this evidence. Missing evidence remains ineligible
    even when ``require_process_evidence`` is false (diagnostic/legacy mode).

    Submission failures are represented as data rather than propagated so the
    CLI can still emit a diagnostic evaluation document for malformed inputs.
    """

    if type(require_process_evidence) is not bool:
        result = _configuration_error_result(
            "require_process_evidence must be a boolean"
        )
        if output_path is not None:
            write_evaluation(result, output_path)
        return result

    try:
        loaded_config = load_config(config)
    except (EvaluatorConfigurationError, ScoringConfigurationError) as exc:
        result = _configuration_error_result(str(exc))
    else:
        try:
            limits = _config_limits(loaded_config)
            if isinstance(submission, Mapping):
                loaded_submission: Any = submission
                submission_path: str | os.PathLike[str] | None = None
            elif isinstance(submission, (str, os.PathLike)):
                submission_path = submission
                loaded_submission = _load_submission_file(
                    submission, limits["max_submission_bytes"]
                )
            else:
                raise SubmissionValidationError(
                    "submission must be a JSON path or object"
                )
            validated = validate_submission_schema(
                loaded_submission, loaded_config
            )
        except SubmissionValidationError as exc:
            result = _invalid_result(loaded_config, str(exc))
        except EvaluatorConfigurationError as exc:
            result = _configuration_error_result(str(exc))
        else:
            try:
                evidence = _obtain_process_evidence(
                    process_events=process_events,
                    trajectory_path=trajectory_path,
                    submission_path=submission_path,
                    config=loaded_config,
                )
                result = _perform_evaluation(
                    validated,
                    loaded_config,
                    evidence,
                    require_process_evidence=require_process_evidence,
                )
            except (
                EvaluatorConfigurationError,
                ScoringConfigurationError,
                ValueError,
            ) as exc:
                result = _configuration_error_result(str(exc))
            except Exception:
                # Never expose stack traces, molecule contents, or environment
                # paths in an evaluation artifact.
                result = _configuration_error_result(
                    "unexpected evaluator failure"
                )

    if output_path is not None:
        write_evaluation(result, output_path)
    return result


def evaluate_submission_file(
    submission_path: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    process_events: list[dict[str, Any]] | None = None,
    trajectory_path: str | os.PathLike[str] | None = None,
    require_process_evidence: bool = True,
) -> dict[str, Any]:
    """Path-oriented compatibility wrapper used by agent runners."""

    return evaluate_submission(
        submission_path,
        config_path,
        output_path,
        process_events=process_events,
        trajectory_path=trajectory_path,
        require_process_evidence=require_process_evidence,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a constrained-QED benchmark submission"
    )
    parser.add_argument(
        "--submission", required=True, help="submission JSON file"
    )
    parser.add_argument("--config", required=True, help="task YAML file")
    parser.add_argument(
        "--output", required=True, help="evaluation JSON destination"
    )
    parser.add_argument(
        "--trajectory",
        help=(
            "sibling runner trajectory JSONL; defaults to trajectory.jsonl "
            "beside --submission"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate_submission(
            args.submission,
            args.config,
            args.output,
            trajectory_path=args.trajectory,
        )
    except OSError:
        # The requested output itself is unusable, so there is nowhere safe to
        # persist a structured result.
        return 4
    if result["status"] == "invalid_submission":
        return 2
    if result["status"] != "ok":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EvaluatorConfigurationError",
    "PUBLIC_TOOL_SCHEMAS",
    "ProcessEvidenceValidationError",
    "SUPPORTED_PUBLIC_TOOLS",
    "SubmissionValidationError",
    "evaluate_submission",
    "evaluate_submission_file",
    "load_config",
    "main",
    "validate_submission_schema",
    "write_evaluation",
]

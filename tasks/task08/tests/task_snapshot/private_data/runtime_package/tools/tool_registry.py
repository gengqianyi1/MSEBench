"""Strict model-facing tool dispatcher and trusted-state ledger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from .file_tools import ToolInputError, WorkspaceSandbox, inspect_scheduler_record
from .provenance import ProvenanceInspector, canonical_json_sha256, read_strict_json
from .xtb_log_parser import parse_xtb_log
from .xtb_runner import XtbAttemptManager, _first, _relative_case_path
from .xyz_validator import extract_xyz_frame, validate_xyz


class ContractError(ValueError):
    pass


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _schema_error(path: str, message: str) -> ContractError:
    return ContractError(f"{path}: {message}")


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> None:
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        for alternative in alternatives:
            if not isinstance(alternative, Mapping):
                continue
            try:
                _validate_schema(value, alternative, path)
                return
            except ContractError:
                pass
        raise _schema_error(path, "does not match any allowed schema")
    expected_type = schema.get("type")
    type_matches = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if isinstance(expected_type, str) and not type_matches.get(expected_type, False):
        raise _schema_error(path, f"must have type {expected_type}")
    if "const" in schema and value != schema["const"]:
        raise _schema_error(path, "does not match the required constant")
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        raise _schema_error(path, "is not an allowed value")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise _schema_error(path, "is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise _schema_error(path, "is too long")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            raise _schema_error(path, "does not match the required pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise _schema_error(path, "is below the minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise _schema_error(path, "is above the maximum")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise _schema_error(path, "has too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise _schema_error(path, "has too many items")
        if schema.get("uniqueItems") is True:
            encoded = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(encoded) != len(set(encoded)):
                raise _schema_error(path, "items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")
    if isinstance(value, Mapping):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        required = required if isinstance(required, list) else []
        missing = [name for name in required if name not in value]
        if missing:
            raise _schema_error(path, f"missing required fields: {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise _schema_error(path, f"contains additional fields: {extras}")
        for key, item in value.items():
            item_schema = properties.get(key)
            if isinstance(item_schema, Mapping):
                _validate_schema(item, item_schema, f"{path}.{key}")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("task configuration must be an object")
    return value


class ToolRegistry:
    """Dispatch strict tools while deriving a non-model-controlled state."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        task_config_path: str = "configs/task.yaml",
        contracts_path: str = "configs/tool_contracts.json",
        case_root: str = "benchmark_case",
        process_runner: Any = None,
    ) -> None:
        self.project_root = Path(project_root).resolve(strict=True)
        task_path = (self.project_root / task_config_path).resolve(strict=True)
        contract_path = (self.project_root / contracts_path).resolve(strict=True)
        self.task_config = _load_yaml(task_path)
        contracts = read_strict_json(contract_path)
        if not isinstance(contracts, Mapping) or not isinstance(contracts.get("tools"), list):
            raise ValueError("tool contract document is invalid")
        self.contract_document = dict(contracts)
        self.contracts: dict[str, dict[str, Any]] = {}
        for declaration in contracts["tools"]:
            function = declaration.get("function") if isinstance(declaration, Mapping) else None
            if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
                raise ValueError("tool contract contains an invalid function")
            self.contracts[function["name"]] = dict(function)
        allowed = self.task_config.get("tools", {}).get("allowed", [])
        if set(allowed) != set(self.contracts):
            raise ValueError("task allowed tools and tool contracts differ")
        case_path = (self.project_root / case_root).resolve(strict=True)
        security = self.task_config.get("security")
        forbidden = security.get("forbidden_read_roots", []) if isinstance(security, Mapping) else []
        budget = self.task_config.get("budget")
        budget = budget if isinstance(budget, Mapping) else {}
        self.sandbox = WorkspaceSandbox(
            case_path,
            max_lines=int(budget.get("max_text_read_lines_per_call", 160)),
            max_bytes=int(budget.get("max_text_read_bytes_per_call", 65_536)),
            forbidden_roots=tuple(str(value) for value in forbidden),
        )
        job_path = _relative_case_path(self.task_config.get("task", {}).get("job_spec", {}).get("path"))
        manifest_path = _relative_case_path(self.task_config.get("task", {}).get("case_manifest", {}).get("path"))
        if job_path is None:
            job_path = "job_spec.json"
        if manifest_path is None:
            manifest_path = "case_manifest.json"
        job_spec = self.sandbox.read_json(job_path)
        if not isinstance(job_spec, Mapping):
            raise ValueError("job specification must be an object")
        self.job_spec = dict(job_spec)
        self.expected_symbols = self._load_expected_symbols()
        self.provenance = ProvenanceInspector(
            self.sandbox,
            manifest_path=manifest_path,
            expected_symbols=self.expected_symbols,
        )
        self.attempt_manager = XtbAttemptManager(
            self.sandbox,
            task_config=self.task_config,
            job_spec=self.job_spec,
            provenance=self.provenance,
            expected_symbols=self.expected_symbols,
            process_runner=process_runner,
        )
        self._max_tool_calls = int(budget.get("max_tool_calls", 24))
        self._call_count = 0
        self._successful_call_count = 0
        self._seen_call_ids: set[str] = set()
        self._records: list[dict[str, Any]] = []
        self._reads: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, dict[str, Any]] = {}
        self._submission: dict[str, Any] | None = None
        self._frozen = False
        self._trace_hash = str(
            self.task_config.get("determinism", {}).get("trajectory_genesis_hash", "0" * 64)
        )

    @property
    def live_native_execution(self) -> bool:
        """Read-only truth signal for LIVE_XTB runner-mode enforcement."""

        return self.attempt_manager.live_native_execution

    def _load_expected_symbols(self) -> tuple[str, ...] | None:
        declared = _first(self.job_spec, "element_sequence", "atom_symbols")
        if (
            isinstance(declared, list)
            and declared
            and all(isinstance(value, str) and value for value in declared)
        ):
            return tuple(declared)
        source = _relative_case_path(_first(self.job_spec, "input_geometry", "input_xyz", "geometry_path"))
        if source is None and (self.sandbox.root / "input" / "start.xyz").is_file():
            source = "input/start.xyz"
        if source is None:
            return None
        try:
            path = self.sandbox.resolve(source)
            frame = extract_xyz_frame(path.read_text(encoding="utf-8"), 0)
            return frame.symbols
        except (OSError, UnicodeError, ToolInputError):
            return None

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def submitted_final(self) -> bool:
        return self._submission is not None

    @property
    def live_native_execution(self) -> bool:
        """Whether run_xtb is backed by the real subprocess runtime."""

        return bool(self.attempt_manager._live_runtime)

    @property
    def trusted_state(self) -> dict[str, Any]:
        return _json_copy(
            {
                "tool_call_count": self._call_count,
                "successful_tool_call_count": self._successful_call_count,
                "attempts": self._attempts,
                "reads": self._reads,
                "submission": self._submission,
                "frozen": self._frozen,
                "submitted_final": self.submitted_final,
                "trace_hash": self._trace_hash,
            }
        )

    def _record(
        self,
        name: str,
        arguments: Any,
        call_id: str,
        result: Mapping[str, Any],
    ) -> None:
        body = {
            "sequence": self._call_count,
            "tool": name,
            "call_id": call_id,
            "arguments": arguments,
            "result": result,
            "previous_hash": self._trace_hash,
        }
        record_hash = canonical_json_sha256(body)
        record = {**body, "record_hash": record_hash}
        self._records.append(record)
        self._trace_hash = record_hash

    def _success(self, name: str, call_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
        return {"ok": True, "tool": name, "call_id": call_id, "result": _json_copy(value)}

    @staticmethod
    def _failure(name: str, call_id: str, code: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": name,
            "call_id": call_id,
            "error": {"code": code, "message": message},
        }

    def _dispatch_trusted(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name == "list_case_files":
            return self.sandbox.list_case_files(self.provenance.inspect)
        if name == "read_text_range":
            return self.sandbox.read_text_range(
                arguments["path"],
                arguments["start_line"],
                arguments["end_line"],
            )
        if name == "tail_text":
            return self.sandbox.tail_text(arguments["path"], arguments["line_count"])
        if name == "parse_xtb_log":
            return parse_xtb_log(self.sandbox, arguments["path"])
        if name == "inspect_scheduler_record":
            return inspect_scheduler_record(self.sandbox, arguments["path"])
        if name == "validate_xyz":
            return validate_xyz(
                self.sandbox,
                arguments["path"],
                expected_symbols=self.expected_symbols,
                frame_index=arguments.get("frame_index"),
            )
        if name == "inspect_artifact_provenance":
            return self.provenance.inspect(arguments["path"], arguments["frame_index"])
        if name == "prepare_restart":
            return self.attempt_manager.prepare_restart(**arguments)
        if name == "run_xtb":
            return self.attempt_manager.run_xtb(arguments["attempt_id"])
        if name == "validate_attempt":
            return self.attempt_manager.validate_attempt(arguments["attempt_id"])
        if name == "submit_final":
            return self._submit_final(arguments)
        raise ToolInputError("UNKNOWN_TOOL", "tool is not registered")

    def _submit_final(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self._submission is not None:
            raise ToolInputError("FINAL_ALREADY_SUBMITTED", "final submission is immutable")
        diagnosis = arguments["diagnosis"]
        for evidence in diagnosis["evidence"]:
            read = self._reads.get(evidence["tool_call_id"])
            if not isinstance(read, Mapping):
                raise ToolInputError("UNTRUSTED_EVIDENCE", "evidence does not reference a successful text read")
            expected = {
                "path": read.get("path"),
                "line_start": read.get("line_start"),
                "line_end": read.get("line_end"),
                "content_sha256": read.get("content_sha256"),
            }
            actual = {key: evidence.get(key) for key in expected}
            if actual != expected or expected["line_start"] is None:
                raise ToolInputError("UNTRUSTED_EVIDENCE", "evidence path, lines, or hash does not match the read")
        validations = [
            value for value in self._attempts.values()
            if value.get("state") == "VALIDATED" and value.get("validation_passed") is True
        ]
        optimization_complete = any(value.get("optimization_completed") is True for value in validations)
        hessian_complete = any(value.get("hessian_completed") is True for value in validations)
        claimed = arguments["result_summary"]
        if claimed["optimization_completed"] is not optimization_complete:
            raise ToolInputError("RESULT_CLAIM_MISMATCH", "optimization completion claim differs from trusted state")
        if claimed["hessian_completed"] is not hessian_complete:
            raise ToolInputError("RESULT_CLAIM_MISMATCH", "hessian completion claim differs from trusted state")
        final_attempt = claimed["final_attempt_id"]
        if final_attempt and final_attempt not in self._attempts:
            raise ToolInputError("RESULT_CLAIM_MISMATCH", "final attempt is not in trusted state")
        if arguments["final_status"] == "RECOVERED":
            if not optimization_complete or not hessian_complete:
                raise ToolInputError("RECOVERY_NOT_PROVEN", "both validated phases are required for RECOVERED")
            final_state = self._attempts.get(final_attempt, {})
            if final_state.get("hessian_completed") is not True:
                raise ToolInputError("RECOVERY_NOT_PROVEN", "final attempt must be the validated hessian attempt")
        submission = _json_copy(arguments)
        self._submission = submission
        self._frozen = True
        return {
            "status": "SUBMITTED",
            "final_status": arguments["final_status"],
            "submission_sha256": canonical_json_sha256(submission),
            "trusted_optimization_completed": optimization_complete,
            "trusted_hessian_completed": hessian_complete,
            "frozen": True,
        }

    def dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> dict[str, Any]:
        self._call_count += 1
        safe_call_id = call_id if isinstance(call_id, str) else "<invalid>"
        failure: dict[str, Any] | None = None
        if self._call_count > self._max_tool_calls:
            failure = self._failure(name, safe_call_id, "TOOL_BUDGET_EXCEEDED", "tool-call budget is exhausted")
        elif self._frozen:
            failure = self._failure(name, safe_call_id, "RUN_FROZEN", "no tool calls are allowed after final submission")
        elif not isinstance(call_id, str) or not call_id or len(call_id) > 100 or any(ord(c) < 32 for c in call_id):
            failure = self._failure(name, safe_call_id, "INVALID_CALL_ID", "call_id must be a non-empty bounded string")
        elif call_id in self._seen_call_ids:
            failure = self._failure(name, call_id, "DUPLICATE_CALL_ID", "call_id has already been used")
        elif name not in self.contracts:
            failure = self._failure(name, call_id, "UNKNOWN_TOOL", "tool is not registered")
        elif not isinstance(arguments, Mapping):
            failure = self._failure(name, call_id, "INVALID_ARGUMENTS", "arguments must be an object")
        if isinstance(call_id, str) and call_id:
            self._seen_call_ids.add(call_id)
        safe_arguments: Any = _json_copy(arguments) if isinstance(arguments, Mapping) else None
        if failure is not None:
            self._record(name, safe_arguments, safe_call_id, failure)
            return failure
        try:
            parameters = self.contracts[name].get("parameters")
            if not isinstance(parameters, Mapping):
                raise ContractError("tool parameter schema is missing")
            _validate_schema(arguments, parameters)
            value = self._dispatch_trusted(name, arguments)
            result = self._success(name, call_id, value)
            self._successful_call_count += 1
            # Only bounded read_text_range results are admissible citations.
            # tail_text is a navigation aid and is intentionally not trusted as
            # final evidence because the evaluator uses exact requested ranges.
            if name == "read_text_range":
                self._reads[call_id] = {
                    key: value.get(key)
                    for key in ("path", "line_start", "line_end", "content_sha256")
                }
            if name in {"prepare_restart", "run_xtb", "validate_attempt"}:
                attempt_id = value.get("attempt_id")
                if isinstance(attempt_id, str):
                    previous = self._attempts.get(attempt_id, {})
                    self._attempts[attempt_id] = {**previous, **_json_copy(value)}
        except ContractError as exc:
            result = self._failure(name, call_id, "SCHEMA_VALIDATION_FAILED", str(exc))
        except ToolInputError as exc:
            result = self._failure(name, call_id, exc.code, str(exc))
        except Exception as exc:  # fail closed without leaking host paths or provider data
            result = self._failure(name, call_id, "TOOL_INTERNAL_ERROR", type(exc).__name__)
        self._record(name, safe_arguments, call_id, result)
        return result


__all__ = ["ContractError", "ToolRegistry"]

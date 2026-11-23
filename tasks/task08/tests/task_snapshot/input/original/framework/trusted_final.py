"""Runner-side construction of the persisted trusted final-answer envelope.

This module is intentionally not used by the independent evaluator.  The
evaluator re-derives the same facts from frozen bytes so a runner bug cannot
become its own oracle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from .schemas import ProtocolViolationError
from .trajectory import canonical_json, verify_hash_chain


MAX_ARTIFACT_FILES = 20_000
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
IDENTITY_EXEMPT_RUN_TYPES = frozenset(
    {"BASELINE", "BASELINE_REPLAY", "REPLAY", "FORMAL_REPLAY"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProtocolViolationError("ARTIFACT_LSTAT_FAILED") from exc
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _workspace_files(root: Path | None) -> list[dict[str, Any]]:
    if root is None:
        return []
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolViolationError("ARTIFACT_ROOT_UNAVAILABLE") from exc
    if not resolved.is_dir() or _is_reparse_point(resolved):
        raise ProtocolViolationError("ARTIFACT_ROOT_UNSAFE")

    files: list[dict[str, Any]] = []
    total_size = 0
    pending: list[tuple[Path, tuple[str, ...]]] = [(resolved, ())]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ProtocolViolationError("ARTIFACT_DIRECTORY_UNREADABLE") from exc
        child_directories: list[tuple[Path, tuple[str, ...]]] = []
        for entry in entries:
            path = Path(entry.path)
            relative_parts = (*prefix, entry.name)
            relative = "/".join(relative_parts)
            if _is_reparse_point(path):
                raise ProtocolViolationError("ARTIFACT_REPARSE_POINT")
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as exc:
                raise ProtocolViolationError("ARTIFACT_TYPE_UNREADABLE") from exc
            if is_directory:
                child_directories.append((path, relative_parts))
                continue
            if not is_file:
                raise ProtocolViolationError("ARTIFACT_SPECIAL_FILE")
            if len(files) >= MAX_ARTIFACT_FILES:
                raise ProtocolViolationError("ARTIFACT_FILE_LIMIT_EXCEEDED")
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise ProtocolViolationError("ARTIFACT_STAT_FAILED") from exc
            total_size += size
            if total_size > MAX_ARTIFACT_BYTES:
                raise ProtocolViolationError("ARTIFACT_SIZE_LIMIT_EXCEEDED")
            files.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": _sha256_file(path),
                }
            )
        # Reverse push preserves lexical traversal when the final list is
        # sorted below, while keeping recursion bounded by the filesystem.
        pending.extend(reversed(child_directories))
    return sorted(files, key=lambda item: str(item["path"]))


def _payloads(events: Sequence[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
    return [
        dict(event.get("payload", {}))
        for event in events
        if event.get("event") == name and isinstance(event.get("payload"), Mapping)
    ]


def _attempt_history(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = _payloads(events, "attempt_prepared")
    executed = _payloads(events, "native_xtb_execution")
    validated = _payloads(events, "attempt_validated")
    ordered_ids: list[str] = []
    for payload in (*prepared, *executed, *validated):
        attempt_id = payload.get("attempt_id")
        if (
            isinstance(attempt_id, str)
            and attempt_id != "attempt_000_initial_failure"
            and attempt_id not in ordered_ids
        ):
            ordered_ids.append(attempt_id)

    def indexed(values: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for payload in values:
            attempt_id = payload.get("attempt_id")
            if isinstance(attempt_id, str) and attempt_id not in result:
                result[attempt_id] = payload
        return result

    prepared_by_id = indexed(prepared)
    executed_by_id = indexed(executed)
    validated_by_id = indexed(validated)
    history: list[dict[str, Any]] = []
    for attempt_id in ordered_ids:
        prep = prepared_by_id.get(attempt_id, {})
        run = executed_by_id.get(attempt_id, {})
        validation = validated_by_id.get(attempt_id, {})
        source = prep.get("source_geometry")
        source = source if isinstance(source, Mapping) else {}
        phase = prep.get("phase", run.get("phase", validation.get("phase", "")))
        exit_code = run.get("process_exit_code")
        history.append(
            {
                "attempt_id": attempt_id,
                "phase": phase if isinstance(phase, str) else "",
                "source_geometry": {
                    "path": source.get("path") if isinstance(source.get("path"), str) else "",
                    "frame_index": (
                        source.get("frame_index")
                        if type(source.get("frame_index")) is int
                        else None
                    ),
                    "sha256": (
                        str(source.get("sha256")).upper()
                        if isinstance(source.get("sha256"), str)
                        else None
                    ),
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


def _tool_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls = _payloads(events, "tool_call")
    results = _payloads(events, "tool_result")
    successful_ids = {
        payload.get("call_id")
        for payload in results
        if payload.get("ok") is True and isinstance(payload.get("call_id"), str)
    }
    by_name: dict[str, dict[str, int]] = {}
    for payload in calls:
        name = payload.get("name")
        call_id = payload.get("call_id")
        if not isinstance(name, str):
            continue
        record = by_name.setdefault(name, {"calls": 0, "successful": 0, "failed": 0})
        record["calls"] += 1
        if isinstance(call_id, str) and call_id in successful_ids:
            record["successful"] += 1
        else:
            record["failed"] += 1
    successful = sum(payload.get("ok") is True for payload in results)
    return {
        "tool_calls": len(calls),
        "successful_tool_calls": successful,
        "failed_tool_calls": len(calls) - successful,
        "agent_turns": len(_payloads(events, "provider_response_metadata")),
        "provider_attempts": len(_payloads(events, "provider_request")),
        "provider_failures": len(_payloads(events, "provider_error")),
        "provider_retries": len(_payloads(events, "provider_retry")),
        "execution_attempts": len(_payloads(events, "attempt_prepared")),
        "native_executions": len(_payloads(events, "native_xtb_execution")),
        "by_tool": [
            {"name": name, **by_name[name]}
            for name in sorted(by_name)
        ],
    }


def _final_validation(
    model_submission: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = model_submission.get("result_summary")
    result = result if isinstance(result, Mapping) else {}
    final_attempt_id = result.get("final_attempt_id")
    final_attempt_id = final_attempt_id if isinstance(final_attempt_id, str) else ""
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
    return {
        "final_attempt_id": final_attempt_id,
        "final_attempt_validated": final_valid,
        "optimization_completed": optimization_completed,
        "hessian_completed": final_valid and final_record.get("hessian_completed") is True,
        "artifacts_current": final_valid and final_record.get("artifacts_current") is True,
        "invariants_preserved": (
            final_valid and final_record.get("invariants_preserved") is True
        ),
    }


def build_trusted_runtime(
    *,
    model_submission: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    trajectory_path: Path,
    artifact_root: Path | None,
    genesis_hash: str,
) -> dict[str, Any]:
    """Build trusted runtime facts only after the terminal trace is frozen."""

    try:
        trace_bytes = trajectory_path.read_bytes()
    except OSError as exc:
        raise ProtocolViolationError("TRAJECTORY_UNREADABLE") from exc
    if not events:
        raise ProtocolViolationError("TRAJECTORY_EMPTY")
    files = _workspace_files(artifact_root)
    history = _attempt_history(events)
    starts = _payloads(events, "run_start")
    start = starts[0] if len(starts) == 1 else {}
    responses = _payloads(events, "provider_response_metadata")
    actual_models = [
        payload["actual_model"]
        for payload in responses
        if isinstance(payload.get("actual_model"), str) and payload["actual_model"]
    ]
    request_ids = [
        payload["request_id"]
        for payload in responses
        if isinstance(payload.get("request_id"), str) and payload["request_id"]
    ]
    requested = start.get("requested_model")
    requested = requested if isinstance(requested, str) else ""
    run_type = start.get("run_type")
    run_type = run_type if isinstance(run_type, str) else ""
    identity_exempt = run_type in IDENTITY_EXEMPT_RUN_TYPES
    exact_match = identity_exempt or (
        bool(actual_models)
        and len(actual_models) == len(request_ids)
        and all(request_id != "UNAVAILABLE" for request_id in request_ids)
        and all(actual == requested for actual in actual_models)
    )
    chain_valid = verify_hash_chain(
        events,
        genesis_hash=genesis_hash,
        require_terminal=True,
    )
    workspace_material = canonical_json(files).encode("utf-8")
    return {
        "attempt_history": history,
        "tool_counts": _tool_counts(events),
        "final_validation": _final_validation(model_submission, history),
        "trace": {
            "trace_sha256": _sha256_bytes(trace_bytes),
            "final_event_hash": str(events[-1].get("event_hash", "")).upper(),
            "event_count": len(events),
            "hash_chain_valid": chain_valid,
        },
        "artifacts": {
            "workspace_sha256": _sha256_bytes(workspace_material),
            "file_count": len(files),
            "files": files,
        },
        "model_identity": {
            "logical_model": (
                start.get("logical_model")
                if isinstance(start.get("logical_model"), str)
                else ""
            ),
            "requested_model": requested,
            "actual_models": actual_models,
            "provider": start.get("provider") if isinstance(start.get("provider"), str) else "",
            "transport": (
                start.get("transport") if isinstance(start.get("transport"), str) else ""
            ),
            "exact_match": exact_match,
            "identity_exempt": identity_exempt,
        },
        "execution_mode": {
            "run_type": run_type,
            "execution_mode": (
                start.get("execution_mode")
                if isinstance(start.get("execution_mode"), str)
                else ""
            ),
            "native_mode": (
                start.get("native_mode")
                if isinstance(start.get("native_mode"), str)
                else ""
            ),
        },
    }


def build_final_answer_envelope(
    *,
    model_submission: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    trajectory_path: Path,
    artifact_root: Path | None,
    genesis_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "model_submission": dict(model_submission),
        "trusted_runtime": build_trusted_runtime(
            model_submission=model_submission,
            events=events,
            trajectory_path=trajectory_path,
            artifact_root=artifact_root,
            genesis_hash=genesis_hash,
        ),
    }


__all__ = [
    "IDENTITY_EXEMPT_RUN_TYPES",
    "MAX_ARTIFACT_BYTES",
    "MAX_ARTIFACT_FILES",
    "build_final_answer_envelope",
    "build_trusted_runtime",
]

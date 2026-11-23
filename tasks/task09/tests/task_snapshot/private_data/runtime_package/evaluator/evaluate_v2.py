"""Independent protocol-2.0 trajectory, artifact, and scientific evaluator.

This module imports no runner or tool-runtime code.  It reconstructs the
closed batch state and automatic fit from persisted evidence.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from evaluator.profile_recompute_v2 import verify_profile_artifact
from evaluator.scoring_v2 import score_profile_v2


EVALUATOR_VERSION = "adaptive-torsion-evaluator/2.0"
GENESIS = "0" * 64
PROTOCOL_OBJECT = {
    "name": "adaptive_torsion_sampling_tool_loop",
    "version": "2.0",
}
TASK_ID = "adaptive_xtb_torsion_profile_agent_v2"
CASE_ID = "2_fluorobiphenyl_central_torsion_v1"
PROTOCOL = "adaptive_torsion_sampling_tool_loop/2.0"
REFERENCE_SHA256 = (
    "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"
)
XTB_SHA256 = (
    "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
)
NATIVE_LOG_PROJECTION = "native-log-path-redaction/1.0"
HARTREE_TO_KCAL_MOL = 627.5094740631
GRID = list(range(0, 360, 5))
EXPECTED_SYMBOLS = (
    "F",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "C",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
    "H",
)
EXPECTED_BONDS = frozenset(
    {
        (0, 1),
        (1, 2),
        (1, 6),
        (2, 3),
        (2, 13),
        (3, 4),
        (3, 14),
        (4, 5),
        (4, 15),
        (5, 6),
        (5, 16),
        (6, 7),
        (7, 8),
        (7, 12),
        (8, 9),
        (8, 17),
        (9, 10),
        (9, 18),
        (10, 11),
        (10, 19),
        (11, 12),
        (11, 20),
        (12, 21),
    }
)
COVALENT_RADII = {"H": 0.31, "C": 0.76, "F": 0.57}
FORMAL_MODELS = {
    "Pro/moonshotai/Kimi-K2.6",
    "deepseek-ai/DeepSeek-V4-Pro",
    "zai-org/GLM-5.2",
    "deterministic-baseline/2.0",
}
RUN_ID_BY_MODEL = {
    "Pro/moonshotai/Kimi-K2.6": "kimi-formal-v2",
    "deepseek-ai/DeepSeek-V4-Pro": "deepseek-formal-v2",
    "zai-org/GLM-5.2": "glm-formal-v2",
    "deterministic-baseline/2.0": "baseline-live-v2",
}
TRANSIENT_PROVIDER_CATEGORIES = {
    "RATE_LIMITED",
    "PROVIDER_OVERLOADED",
    "NETWORK_CONNECTION_FAILED",
    "READ_TIMEOUT",
}
NONRETRYABLE_PROVIDER_CATEGORIES = {
    "INSUFFICIENT_BALANCE",
    "AUTHENTICATION_FAILED",
    "MODEL_PERMISSION_DENIED",
    "MODEL_NOT_FOUND",
    "IDENTITY_MISMATCH",
    "PROTOCOL_VIOLATION",
    "UNKNOWN_PROVIDER_ERROR",
}
KNOWN_PROVIDER_CATEGORIES = (
    TRANSIENT_PROVIDER_CATEGORIES | NONRETRYABLE_PROVIDER_CATEGORIES
)
BLOCKED_SAFE_STATUSES = KNOWN_PROVIDER_CATEGORIES - {
    "IDENTITY_MISMATCH",
    "PROTOCOL_VIOLATION",
}
BLOCKED_SAFE_STATUSES |= {
    "TOTAL_TIMEOUT_EXCEEDED",
    "TOTAL_NATIVE_TIMEOUT_EXCEEDED",
}
FAIL_SAFE_STATUSES = {
    "IDENTITY_MISMATCH",
    "PROTOCOL_VIOLATION",
    "FINAL_ELIGIBILITY_FAILED",
    "NATIVE_ELIGIBILITY_IMPOSSIBLE",
    "NATIVE_INVARIANT_BREACH",
    "PROVIDER_TURN_BUDGET_EXHAUSTED",
    "RUNNER_POLICY_FAILURE",
}
EXECUTION_ROOT_FILES = frozenset(
    {"TASK.md", "environment.yml", "requirements.txt"}
)
EXECUTION_TOP_LEVEL = frozenset(
    {
        "baselines",
        "benchmark_case",
        "configs",
        "evaluator",
        "evaluator_private",
        "framework",
        "prompts",
        "tools",
    }
)
EXECUTION_SCRIPT_EXCLUSIONS = frozenset(
    {
        "package_submission.py",
        "pre_freeze_package_rehearsal.py",
        "release_patch_rehearsal.py",
        "replay_trajectories.py",
        "secret_scan.py",
        "validate_package.py",
        "validate_project.py",
        "verify_old_evidence.py",
    }
)
EXECUTION_CONFIG_EXCLUSIONS = frozenset(
    {"campaign_snapshot.json", "release.json", "release_commitment.json"}
)
WORK_EXCLUSIONS = frozenset(
    {".git", ".test_tmp", "audit_tmp", "outputs", "runs", "runs_failed", "staging"}
)
CACHE_EXCLUSIONS = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
ABSOLUTE_PATH = re.compile(
    r"(?i)^(?:[A-Z]:[\\/]|\\\\|//|\\\\[.?]\\|file:/{1,3}|/)"
)
RUNTIME_ABSOLUTE_TEXT = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[A-Z]:[\\/]|\\\\|//|"
    r"/(?:home|Users|tmp|workspace|mnt)/)[^\s'\"<>|?*]+"
)
SECRET = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._~+/-]{12,}|"
    r"(?:api[_-]?key|token|secret|authorization)\s*[:=]\s*\S+)"
)
UPPER_SHA256 = re.compile(r"[0-9A-F]{64}")
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
UTC_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?Z"
)


class EvaluationError(RuntimeError):
    """A persisted run cannot be independently validated."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_pretty(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _strict_json(path: Path) -> Any:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid JSON: {path.name}") from exc
    if duplicates:
        raise EvaluationError(f"duplicate keys in {path.name}")
    return value


def _read_xyz_evidence(path: Path) -> tuple[list[str], np.ndarray, str]:
    """Parse exactly one finite XYZ frame without using runtime science code."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvaluationError("XYZ evidence is unreadable") from exc
    if len(lines) < 2:
        raise EvaluationError("XYZ evidence is truncated")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise EvaluationError("XYZ atom count is invalid") from exc
    if count != len(EXPECTED_SYMBOLS) or len(lines) != count + 2:
        raise EvaluationError("XYZ frame length changed")
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for row in lines[2:]:
        fields = row.split()
        if len(fields) != 4 or fields[0] not in COVALENT_RADII:
            raise EvaluationError("XYZ atom row is invalid")
        try:
            xyz = [float(item) for item in fields[1:]]
        except ValueError as exc:
            raise EvaluationError("XYZ coordinate is invalid") from exc
        if not all(math.isfinite(item) for item in xyz):
            raise EvaluationError("XYZ coordinate is non-finite")
        symbols.append(fields[0])
        coordinates.append(xyz)
    return symbols, np.asarray(coordinates, dtype=float), lines[1]


def _dihedral_deg(points: np.ndarray) -> float:
    p0, p1, p2, p3 = points
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = float(np.linalg.norm(b1))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise EvaluationError("target dihedral is degenerate")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if float(np.linalg.norm(v)) <= 1.0e-12 or float(np.linalg.norm(w)) <= 1.0e-12:
        raise EvaluationError("target dihedral is degenerate")
    value = math.degrees(
        math.atan2(
            float(np.dot(np.cross(b1, v), w)),
            float(np.dot(v, w)),
        )
    )
    if not math.isfinite(value):
        raise EvaluationError("target dihedral is non-finite")
    return value % 360.0


def _circular_distance(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _infer_connectivity(
    symbols: Sequence[str], coordinates: np.ndarray
) -> frozenset[tuple[int, int]]:
    bonds: set[tuple[int, int]] = set()
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            cutoff = 1.25 * (
                COVALENT_RADII[symbols[left]] + COVALENT_RADII[symbols[right]]
            )
            if float(np.linalg.norm(coordinates[left] - coordinates[right])) <= cutoff:
                bonds.add((left, right))
    return frozenset(bonds)


def _xyz_structure_facts(
    symbols: Sequence[str], coordinates: np.ndarray
) -> dict[str, bool]:
    order_ok = tuple(symbols) == EXPECTED_SYMBOLS
    return {
        "atom_count_ok": len(symbols) == len(EXPECTED_SYMBOLS),
        "atom_order_ok": order_ok,
        "composition_ok": sorted(symbols) == sorted(EXPECTED_SYMBOLS),
        # XYZ has no charge column.  For this frozen neutral, closed-shell case,
        # exact atom order plus exact connectivity and the independently checked
        # --chrg 0/--uhf 0 invocation uniquely bind the expected neutral graph.
        "formal_charge_ok": bool(
            order_ok and _infer_connectivity(symbols, coordinates) == EXPECTED_BONDS
        ),
        "connectivity_ok": (
            _infer_connectivity(symbols, coordinates) == EXPECTED_BONDS
        ),
    }


def _parse_xtb_energy(
    *,
    stdout_text: str | None,
    xyz_comment: str,
) -> tuple[float | None, float | None]:
    comment_match = re.search(r"\benergy:\s*([-+0-9.Ee]+)", xyz_comment)
    try:
        comment_energy = (
            float(comment_match.group(1)) if comment_match is not None else None
        )
    except ValueError:
        comment_energy = None
    if comment_energy is not None and not math.isfinite(comment_energy):
        comment_energy = None
    stdout_energy: float | None = None
    if stdout_text is not None:
        matches = re.findall(
            r"\|\s*TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh\s*\|",
            stdout_text,
        )
        try:
            stdout_energy = float(matches[-1]) if matches else None
        except ValueError:
            stdout_energy = None
        if stdout_energy is not None and not math.isfinite(stdout_energy):
            stdout_energy = None
    return stdout_energy, comment_energy


def _load_trajectory(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvaluationError("trajectory is unreadable") from exc
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise EvaluationError("trajectory has missing or empty lines")
    previous = GENESIS
    events: list[dict[str, Any]] = []
    run_id: str | None = None
    previous_timestamp: datetime | None = None
    for sequence, line in enumerate(lines, start=1):
        try:
            event = json.loads(
                line,
                parse_constant=lambda raw: (_ for _ in ()).throw(
                    ValueError(raw)
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise EvaluationError("trajectory contains invalid JSON") from exc
        if line.encode("utf-8") != _canonical(event):
            raise EvaluationError("trajectory JSONL is not canonical")
        expected = {
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
        if not isinstance(event, Mapping) or set(event) != expected:
            raise EvaluationError("trajectory envelope is not closed")
        if (
            event["schema_version"] != "1.0"
            or event["protocol"] != PROTOCOL_OBJECT
            or event["sequence"] != sequence
            or event["prev_event_hash"] != previous
        ):
            raise EvaluationError("trajectory sequence or protocol changed")
        timestamp = event.get("timestamp_utc")
        if not _valid_utc_timestamp(timestamp):
            raise EvaluationError("trajectory timestamp is invalid")
        parsed_timestamp = datetime.fromisoformat(
            timestamp.removesuffix("Z") + "+00:00"
        )
        if (
            previous_timestamp is not None
            and parsed_timestamp < previous_timestamp
        ):
            raise EvaluationError("trajectory timestamps are not monotonic")
        previous_timestamp = parsed_timestamp
        if run_id is None:
            run_id = event["run_id"]
            if (
                not isinstance(run_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}", run_id)
            ):
                raise EvaluationError("trajectory run id is invalid")
        elif event["run_id"] != run_id:
            raise EvaluationError("trajectory run id changed")
        without_hash = {
            key: value for key, value in event.items() if key != "event_hash"
        }
        expected_hash = hashlib.sha256(_canonical(without_hash)).hexdigest()
        if event["event_hash"] != expected_hash:
            raise EvaluationError("trajectory event hash mismatch")
        previous = event["event_hash"]
        events.append(dict(event))
    if events[0]["event"] != "run_start":
        raise EvaluationError("trajectory lacks run_start")
    if events[-1]["event"] not in {"run_complete", "run_failure"}:
        raise EvaluationError("trajectory lacks terminal event")
    if events[-1]["event"] == "run_complete":
        if len(events) < 2 or events[-2]["event"] != "final_submitted":
            raise EvaluationError("successful trajectory lacks terminal submission")
    return events, {
        "event_count": len(events),
        "head_hash": previous,
        "file_sha256": _sha_file(path),
        "run_id": run_id,
        "terminal_event": events[-1]["event"],
    }


def _is_int(value: Any, *, low: int | None = None, high: int | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def _is_finite_number(
    value: Any, *, low: float | None = None, high: float | None = None
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return bool(
        math.isfinite(number)
        and (low is None or number >= low)
        and (high is None or number <= high)
    )


def _closed_mapping(value: Any, fields: set[str], message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvaluationError(message)
    return value


def _validate_provider_usage(value: Any) -> None:
    if value is None:
        return
    usage = _closed_mapping(
        value,
        {"input_tokens", "output_tokens", "total_tokens"},
        "provider usage event is not closed",
    )
    if any(not _is_int(item, low=0) for item in usage.values()) or (
        usage["total_tokens"]
        != usage["input_tokens"] + usage["output_tokens"]
    ):
        raise EvaluationError("provider usage event is invalid")


def _validate_safe_provider_error(value: Any) -> None:
    safe = _closed_mapping(
        value,
        {
            "status",
            "error_category",
            "http_status",
            "status_code",
            "provider_error_code",
            "sanitized_provider_message",
            "exception_type",
            "classification_version",
            "retryable",
            "message_reflected",
            "raw_response_persisted",
            "response_headers_persisted",
            "request_persisted",
        },
        "safe provider error is not closed",
    )
    category = safe.get("status")
    http_status = safe.get("http_status")
    if (
        category not in KNOWN_PROVIDER_CATEGORIES
        or safe.get("error_category") != category
        or safe.get("status_code") != http_status
        or (
            http_status is not None
            and not _is_int(http_status, low=100, high=599)
        )
        or (
            safe.get("provider_error_code") is not None
            and (
                not isinstance(safe.get("provider_error_code"), str)
                or len(safe["provider_error_code"]) > 120
            )
        )
        or not isinstance(safe.get("sanitized_provider_message"), str)
        or not 1 <= len(safe["sanitized_provider_message"]) <= 240
        or not isinstance(safe.get("exception_type"), str)
        or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,79}", safe["exception_type"]
        )
        or safe.get("classification_version")
        != "provider-error-classification/2.0"
        or safe.get("retryable")
        is not (category in TRANSIENT_PROVIDER_CATEGORIES)
        or safe.get("message_reflected") is not False
        or safe.get("raw_response_persisted") is not False
        or safe.get("response_headers_persisted") is not False
        or safe.get("request_persisted") is not False
    ):
        raise EvaluationError("safe provider error is invalid")


def _validate_event_contracts(
    events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Reject inserted event types and independently validate closed payloads."""

    allowed = {
        "run_start",
        "provider_attempt_failed",
        "provider_response_accepted",
        "provider_response_rejected_identity",
        "protocol_violation",
        "public_action",
        "public_action_result",
        "baseline_decision",
        "final_submitted",
        "run_complete",
        "run_failure",
    }
    if any(event.get("event") not in allowed for event in events):
        raise EvaluationError("trajectory contains an unknown event type")
    if sum(event["event"] == "run_start" for event in events) != 1:
        raise EvaluationError("trajectory run_start cardinality changed")
    raw_start = events[0]["payload"]
    if not isinstance(raw_start, Mapping):
        raise EvaluationError("run_start payload is not an object")
    requested = raw_start.get("requested_model_id")
    baseline = requested == "deterministic-baseline/2.0"
    start_fields = {
        "task_id",
        "benchmark_version",
        "case_id",
        "protocol",
        "mode",
        "requested_model_id",
        "fallback_allowed",
        "context_mode",
        "raw_provider_response_persisted",
    }
    if not baseline or "execution_freeze_binding" in raw_start:
        start_fields.add("execution_freeze_binding")
    start = _closed_mapping(
        raw_start,
        start_fields,
        "run_start payload is not closed",
    )
    if (
        start.get("task_id") != TASK_ID
        or start.get("benchmark_version") != "2.0.0"
        or start.get("case_id") != CASE_ID
        or start.get("protocol") != PROTOCOL
        or requested not in FORMAL_MODELS
        or events[0].get("run_id") != RUN_ID_BY_MODEL.get(requested)
        or start.get("mode")
        != (
            "DETERMINISTIC_BASELINE_LIVE_XTB"
            if baseline
            else "REAL_API_LIVE_XTB"
        )
        or start.get("fallback_allowed") is not False
        or start.get("context_mode") != "canonical_state_v2"
        or start.get("raw_provider_response_persisted") is not False
        or (
            baseline
            and "execution_freeze_binding" in start
            and start.get("execution_freeze_binding") is not None
        )
        or (
            not baseline
            and not isinstance(
                start.get("execution_freeze_binding"), Mapping
            )
        )
    ):
        raise EvaluationError("run_start identity or policy changed")

    for index, event in enumerate(events):
        name = event["event"]
        payload = event["payload"]
        if not isinstance(payload, Mapping):
            raise EvaluationError("trajectory event payload is not an object")
        if name == "run_start":
            if index != 0:
                raise EvaluationError("run_start is not first")
            continue
        if name in {
            "provider_response_accepted",
            "provider_response_rejected_identity",
        }:
            response = _closed_mapping(
                payload,
                {
                    "turn",
                    "actual_model_id",
                    "exact_identity",
                    "content_chars",
                    "content_sha256",
                    "raw_content_persisted",
                    "canonical_state_sha256",
                    "usage",
                },
                "provider response event is not closed",
            )
            expected_exact = name == "provider_response_accepted"
            if (
                baseline
                or not _is_int(response.get("turn"), low=1, high=5)
                or not isinstance(response.get("actual_model_id"), str)
                or not response["actual_model_id"]
                or response.get("exact_identity") is not expected_exact
                or not _is_int(response.get("content_chars"), low=0)
                or not isinstance(response.get("content_sha256"), str)
                or not UPPER_SHA256.fullmatch(response["content_sha256"])
                or response.get("raw_content_persisted") is not False
                or not isinstance(
                    response.get("canonical_state_sha256"), str
                )
                or not UPPER_SHA256.fullmatch(
                    response["canonical_state_sha256"]
                )
            ):
                raise EvaluationError("provider response event is invalid")
            _validate_provider_usage(response.get("usage"))
        elif name == "provider_attempt_failed":
            failure = _closed_mapping(
                payload,
                {
                    "turn",
                    "attempt_number",
                    "category",
                    "retryable",
                    "will_retry",
                    "planned_delay_seconds",
                    "safe_provider_error",
                    "same_model_run_state",
                },
                "provider retry event is not closed",
            )
            _validate_safe_provider_error(
                failure.get("safe_provider_error")
            )
            category = failure.get("category")
            if (
                baseline
                or not _is_int(failure.get("turn"), low=1, high=5)
                or not _is_int(
                    failure.get("attempt_number"), low=1, high=4
                )
                or category not in KNOWN_PROVIDER_CATEGORIES
                or failure.get("safe_provider_error", {}).get("status")
                != category
                or failure.get("retryable")
                is not (category in TRANSIENT_PROVIDER_CATEGORIES)
                or not isinstance(failure.get("will_retry"), bool)
                or not _is_finite_number(
                    failure.get("planned_delay_seconds"),
                    low=0.0,
                    high=30.0,
                )
                or failure.get("same_model_run_state") is not True
            ):
                raise EvaluationError("provider retry event is invalid")
        elif name == "protocol_violation":
            violation = _closed_mapping(
                payload,
                {
                    "turn",
                    "category",
                    "same_state_rerequest_allowed",
                    "raw_content_persisted",
                },
                "protocol violation event is not closed",
            )
            if (
                baseline
                or not _is_int(violation.get("turn"), low=1, high=5)
                or violation.get("category") != "PROTOCOL_VIOLATION"
                or not isinstance(
                    violation.get("same_state_rerequest_allowed"), bool
                )
                or violation.get("raw_content_persisted") is not False
            ):
                raise EvaluationError("protocol violation event is invalid")
        elif name == "public_action":
            action_event = _closed_mapping(
                payload,
                {"turn", "action"},
                "public action event is not closed",
            )
            if (
                not _is_int(action_event.get("turn"), low=1, high=5)
                or not isinstance(action_event.get("action"), Mapping)
            ):
                raise EvaluationError("public action event is invalid")
        elif name == "public_action_result":
            result_event = _closed_mapping(
                payload,
                {"turn", "action", "result", "tool_result_chars"},
                "public action result event is not closed",
            )
            result = result_event.get("result")
            if (
                not _is_int(result_event.get("turn"), low=1, high=5)
                or result_event.get("action")
                not in {"run_scan_batch", "inspect_failed_attempt"}
                or not isinstance(result, Mapping)
                or not _is_int(
                    result_event.get("tool_result_chars"), low=1, high=6000
                )
            ):
                raise EvaluationError("public action result event is invalid")
        elif name == "baseline_decision":
            decision = _closed_mapping(
                payload,
                {"turn", "actual_model_id", "decision_summary"},
                "baseline decision event is not closed",
            )
            if (
                not baseline
                or not _is_int(decision.get("turn"), low=1, high=5)
                or decision.get("actual_model_id") != requested
                or not isinstance(decision.get("decision_summary"), str)
                or len(decision["decision_summary"]) > 400
            ):
                raise EvaluationError("baseline decision event is invalid")
        elif name == "final_submitted":
            submitted = _closed_mapping(
                payload,
                {
                    "turn",
                    "model_submission",
                    "profile_id",
                    "profile_hash",
                    "terminal_finalization",
                    "extra_provider_turn",
                },
                "final submission event is not closed",
            )
            if (
                not _is_int(submitted.get("turn"), low=1, high=5)
                or not isinstance(submitted.get("model_submission"), Mapping)
                or not isinstance(submitted.get("profile_id"), str)
                or not isinstance(submitted.get("profile_hash"), str)
                or not UPPER_SHA256.fullmatch(submitted["profile_hash"])
                or submitted.get("terminal_finalization") is not True
                or submitted.get("extra_provider_turn") is not False
            ):
                raise EvaluationError("final submission event is invalid")
        elif name == "run_complete":
            complete = _closed_mapping(
                payload,
                {"status", "exact_model_identity", "profile_id"},
                "run_complete event is not closed",
            )
            if (
                index != len(events) - 1
                or complete.get("status") != "PASS"
                or complete.get("exact_model_identity") is not True
                or not isinstance(complete.get("profile_id"), str)
            ):
                raise EvaluationError("run_complete event is invalid")
        elif name == "run_failure":
            failure = _closed_mapping(
                payload,
                {
                    "status",
                    "safe_status",
                    "detail",
                    "raw_provider_response_persisted",
                },
                "run_failure event is not closed",
            )
            if (
                index != len(events) - 1
                or failure.get("status") not in {"FAIL", "BLOCKED"}
                or not isinstance(failure.get("safe_status"), str)
                or not isinstance(failure.get("detail"), Mapping)
                or failure.get("raw_provider_response_persisted") is not False
            ):
                raise EvaluationError("run_failure event is invalid")

    if baseline and any(
        event["event"].startswith("provider_")
        or event["event"] == "protocol_violation"
        for event in events
    ):
        raise EvaluationError("baseline contains provider events")
    if not baseline and any(
        event["event"] == "baseline_decision" for event in events
    ):
        raise EvaluationError("formal run contains baseline events")
    if events[-1]["event"] == "run_complete":
        if (
            sum(event["event"] == "final_submitted" for event in events) != 1
            or events[-2]["event"] != "final_submitted"
            or any(event["event"] == "run_failure" for event in events)
        ):
            raise EvaluationError("successful terminal event grammar changed")
    elif any(
        event["event"] in {"final_submitted", "run_complete"}
        for event in events
    ):
        raise EvaluationError("failed trajectory contains success terminal events")
    return start


def _closed_action(action: Any, expected_phase: str) -> list[int]:
    if not isinstance(action, Mapping) or set(action) != {
        "action",
        "angles_deg",
        "finalize",
        "confidence",
        "rationale",
    }:
        raise EvaluationError("scan action is not closed")
    if action["action"] != "run_scan_batch":
        raise EvaluationError("unexpected public action in completed run")
    angles = action["angles_deg"]
    if (
        not isinstance(angles, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= 360
            or value % 5
            for value in angles
        )
        or len(set(angles)) != len(angles)
    ):
        raise EvaluationError("scan action angles are invalid")
    limits = {
        "phase0": (6, 6, False),
        "phase1": (2, 3, False),
        "phase2": (2, 3, True),
        "recovery": (1, 2, True),
    }
    low, high, finalize = limits[expected_phase]
    if not low <= len(angles) <= high or action["finalize"] is not finalize:
        raise EvaluationError("scan action violates its phase")
    if finalize:
        if action["confidence"] not in {"low", "medium", "high"}:
            raise EvaluationError("terminal confidence is invalid")
    elif action["confidence"] is not None:
        raise EvaluationError("nonterminal confidence is not null")
    if (
        not isinstance(action["rationale"], str)
        or not 1 <= len(action["rationale"]) <= 400
    ):
        raise EvaluationError("action rationale is invalid")
    return list(angles)


def _action_ledger(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pending: tuple[int, Mapping[str, Any]] | None = None
    ledger: list[dict[str, Any]] = []
    phases = ("phase0", "phase1", "phase2", "recovery")
    for event in events:
        name = event["event"]
        payload = event["payload"]
        if name == "public_action":
            if pending is not None:
                raise EvaluationError("public action lacks a result")
            action = payload.get("action")
            if not isinstance(action, Mapping):
                raise EvaluationError("public action payload is invalid")
            if action.get("action") == "inspect_failed_attempt":
                # Inspection is allowed but cannot change the scan ledger.
                if (
                    set(action) != {"action", "angle_deg"}
                    or not _is_int(action.get("angle_deg"), low=0, high=355)
                    or action["angle_deg"] % 5
                ):
                    raise EvaluationError(
                        "failed-attempt inspection action is invalid"
                    )
                pending = (int(payload["turn"]), action)
                continue
            if len(ledger) >= 4:
                raise EvaluationError("scan batch budget exceeded")
            angles = _closed_action(action, phases[len(ledger)])
            pending = (int(payload["turn"]), {**action, "_angles": angles})
        elif name == "public_action_result":
            if pending is None:
                raise EvaluationError("orphan public action result")
            turn, action = pending
            if int(payload.get("turn", -1)) != turn:
                raise EvaluationError("action/result turn mismatch")
            result = payload.get("result")
            if not isinstance(result, Mapping):
                raise EvaluationError("public action result is invalid")
            if action.get("action") == "inspect_failed_attempt":
                if (
                    set(result)
                    != {
                        "ok",
                        "status",
                        "requested_angle_deg",
                        "attempt_count",
                        "attempts",
                        "replacement_sampling_allowed",
                        "budget",
                    }
                    or result.get("ok") is not True
                    or result.get("status")
                    != "FAILED_ATTEMPT_INSPECTION"
                    or result.get("requested_angle_deg")
                    != action.get("angle_deg")
                    or result.get("attempt_count") != 2
                    or not isinstance(result.get("attempts"), list)
                    or len(result["attempts"]) != 2
                    or not isinstance(
                        result.get("replacement_sampling_allowed"), bool
                    )
                    or not isinstance(result.get("budget"), Mapping)
                ):
                    raise EvaluationError("failed-attempt inspection was invalid")
                pending = None
                continue
            record = result.get("batch_record")
            if not isinstance(record, Mapping):
                raise EvaluationError("scan result lacks a batch record")
            if record.get("angles_deg") != action["_angles"]:
                raise EvaluationError("batch record changed action angles")
            if (
                record.get("finalize") is not action["finalize"]
                or record.get("confidence") != action["confidence"]
                or record.get("rationale") != action["rationale"]
                or record.get("action_hash") != _sha_bytes(
                    _canonical_pretty(
                        {
                            key: value
                            for key, value in action.items()
                            if key != "_angles"
                        }
                    )
                )
            ):
                raise EvaluationError("batch record changed terminal intent")
            ledger.append(dict(record))
            pending = None
    if pending is not None:
        raise EvaluationError("final public action lacks a result")
    return ledger


def _reconstruct_state(ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique: list[int] = []
    valid: list[int] = []
    failed: list[int] = []
    profile_ids: list[str] = []
    for index, record in enumerate(ledger):
        phase = ("phase0", "phase1", "phase2", "recovery")[index]
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "schema_version",
                "batch_number",
                "phase",
                "angles_deg",
                "valid_angles_deg",
                "failed_angles_deg",
                "finalize",
                "confidence",
                "rationale",
                "action_hash",
                "profile_id",
                "profile_hash",
                "terminal_accepted",
            }
            or record.get("schema_version") != "2.0"
            or record.get("phase") != phase
            or record.get("batch_number") != index + 1
            or not isinstance(record.get("terminal_accepted"), bool)
            or not isinstance(record.get("action_hash"), str)
            or not UPPER_SHA256.fullmatch(record["action_hash"])
        ):
            raise EvaluationError("batch phase or sequence changed")
        angles = record.get("angles_deg")
        if not isinstance(angles, list) or set(angles) & set(unique):
            raise EvaluationError("batch repeats a prior unique angle")
        unique.extend(angles)
        batch_valid = record.get("valid_angles_deg")
        batch_failed = record.get("failed_angles_deg")
        if (
            not isinstance(batch_valid, list)
            or not isinstance(batch_failed, list)
            or set(batch_valid) | set(batch_failed) != set(angles)
            or set(batch_valid) & set(batch_failed)
        ):
            raise EvaluationError("batch outcomes are incomplete")
        valid.extend(batch_valid)
        failed.extend(batch_failed)
        if (
            not isinstance(record.get("profile_id"), str)
            or not re.fullmatch(
                rf"profile_{index + 1:02d}_[0-9a-f]{{16}}",
                record["profile_id"],
            )
            or not isinstance(record.get("profile_hash"), str)
            or not UPPER_SHA256.fullmatch(record["profile_hash"])
            or not record["profile_id"].endswith(
                record["profile_hash"][:16].lower()
            )
        ):
            raise EvaluationError("batch profile binding is missing")
        profile_ids.append(record["profile_id"])
    feedback = max(0, min(2, len(ledger) - 1))
    eligible = (
        len(unique) <= 12
        and len(set(valid)) >= 10
        and feedback >= 2
        and len(profile_ids) >= 3
    )
    if len(ledger) == 3:
        if ledger[-1].get("terminal_accepted") is not eligible:
            raise EvaluationError("phase2 terminal decision changed")
    elif len(ledger) == 4:
        phase2_unique = [
            angle for record in ledger[:3] for angle in record["angles_deg"]
        ]
        phase2_valid = [
            angle
            for record in ledger[:3]
            for angle in record["valid_angles_deg"]
        ]
        phase2_failed = [
            angle
            for record in ledger[:3]
            for angle in record["failed_angles_deg"]
        ]
        if (
            not phase2_failed
            or len(set(phase2_valid)) >= 10
            or len(set(phase2_unique)) >= 12
            or ledger[2].get("terminal_accepted") is not False
            or ledger[-1].get("terminal_accepted") is not eligible
        ):
            raise EvaluationError("recovery terminal decision changed")
    return {
        "batch_count": len(ledger),
        "unique_angles": unique,
        "unique_angle_count": len(unique),
        "valid_angles": valid,
        "valid_unique_angle_count": len(set(valid)),
        "failed_angles": failed,
        "feedback_rounds": feedback,
        "profile_fit_rounds": len(profile_ids),
        "profile_ids": profile_ids,
        "eligible_process_minimums": eligible,
    }


def _expected_runtime_process(
    ledger: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Mirror the frozen state projection from independently checked records."""

    batch_count = len(ledger)
    if batch_count == 0:
        phase = "phase0"
    elif batch_count == 1:
        phase = "phase1"
    elif batch_count == 2:
        phase = "phase2"
    elif batch_count == 3 and state["valid_unique_angle_count"] < 10:
        phase = "recovery"
    else:
        phase = "terminal"
    recovery_allowed = bool(
        batch_count == 3
        and state["valid_unique_angle_count"] < 10
        and state["unique_angle_count"] < 12
        and state["failed_angles"]
    )
    profile_count = int(state["profile_fit_rounds"])
    feedback_rounds = int(state["feedback_rounds"])
    eligibility = {
        "within_unique_angle_budget": state["unique_angle_count"] <= 12,
        "minimum_valid_unique_angles": state["valid_unique_angle_count"] >= 10,
        "minimum_feedback_rounds": feedback_rounds >= 2,
        "minimum_profile_fits": profile_count >= 3,
        "terminal_profile_present": profile_count >= 3,
        "protocol_violations_absent": True,
    }
    return {
        "schema_version": "2.0",
        "batch_count": batch_count,
        "current_phase": phase,
        "unique_angles": list(state["unique_angles"]),
        "unique_angle_count": int(state["unique_angle_count"]),
        "unique_angles_remaining": max(0, 12 - int(state["unique_angle_count"])),
        "valid_angles": list(state["valid_angles"]),
        "valid_unique_angle_count": int(state["valid_unique_angle_count"]),
        "failed_angles": list(state["failed_angles"]),
        "feedback_rounds": feedback_rounds,
        "profile_fit_rounds": profile_count,
        "profile_ids": list(state["profile_ids"]),
        "recovery_allowed": recovery_allowed,
        "terminal_requested": bool(ledger and ledger[-1]["finalize"]),
        "terminal_accepted": bool(
            ledger and ledger[-1]["terminal_accepted"]
        ),
        "eligible": all(eligibility.values()),
        "eligibility": eligibility,
        "eligibility_gaps": {
            "valid_angles_needed": max(
                0, 10 - int(state["valid_unique_angle_count"])
            ),
            "feedback_rounds_needed": max(0, 2 - feedback_rounds),
            "profile_fits_needed": max(0, 3 - profile_count),
        },
        "violations": [],
    }


def _verify_attempts(
    run_root: Path,
    *,
    require_native_logs: bool,
) -> tuple[dict[str, Any], dict[int, str]]:
    summaries: dict[str, Any] = {}
    eligible_by_angle: dict[int, str] = {}
    counts: defaultdict[int, int] = defaultdict(int)
    batches_by_angle: defaultdict[int, set[int]] = defaultdict(set)
    roots = sorted((run_root / "artifacts" / "scan_summaries").glob("*"))
    expected_argv = [
        "<XTB_EXECUTABLE>",
        "input.xyz",
        "--gfn",
        "2",
        "--chrg",
        "0",
        "--uhf",
        "0",
        "--opt",
        "tight",
        "--input",
        "xcontrol.inp",
    ]
    summary_fields = {
        "schema_version",
        "attempt_id",
        "requested_angle_deg",
        "attempt_number",
        "return_code",
        "timed_out",
        "converged",
        "energy_parsed",
        "total_energy_hartree",
        "final_angle_deg",
        "angle_error_deg",
        "structure_validation",
        "profile_eligible",
        "scientific_protocol",
        "native_invocation",
        "provenance",
        "safe_files",
        "raw_host_paths_released",
    }
    validation_fields = {
        "atom_count_ok",
        "atom_order_ok",
        "composition_ok",
        "formal_charge_ok",
        "connectivity_ok",
    }
    legacy_provenance_fields = {
        "reference_sha256",
        "input_sha256",
        "constraint_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "raw_attempt_manifest_sha256",
    }
    projected_provenance_fields = {
        *legacy_provenance_fields,
        "stdout_projection_sha256",
        "stderr_projection_sha256",
        "native_log_projection",
    }
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            raise EvaluationError("attempt projection contains a link")
        summary = _strict_json(root / "summary.json")
        if (
            not isinstance(summary, Mapping)
            or set(summary) != summary_fields
            or summary.get("schema_version") != "2.0"
            or summary.get("raw_host_paths_released") is not False
        ):
            raise EvaluationError("attempt summary schema changed")
        attempt_id = summary.get("attempt_id")
        angle = summary.get("requested_angle_deg")
        attempt_match = (
            re.fullmatch(
                r"batch_(\d{2})_angle_(\d{3})_attempt_(\d{2})",
                attempt_id,
            )
            if isinstance(attempt_id, str)
            else None
        )
        if (
            attempt_id != root.name
            or attempt_match is None
            or not 1 <= int(attempt_match.group(1)) <= 4
        ):
            raise EvaluationError("attempt id or directory changed")
        if (
            isinstance(angle, bool)
            or not isinstance(angle, int)
            or angle not in GRID
        ):
            raise EvaluationError("attempt angle is invalid")
        if angle in eligible_by_angle:
            raise EvaluationError("native retry continued after success")
        counts[angle] += 1
        batch_number = int(attempt_match.group(1))
        batches_by_angle[angle].add(batch_number)
        if (
            int(attempt_match.group(2)) != angle
            or int(attempt_match.group(3)) != summary.get("attempt_number")
            or summary.get("attempt_number") != counts[angle]
            or counts[angle] > 2
            or len(batches_by_angle[angle]) != 1
        ):
            raise EvaluationError("attempt sequence exceeds frozen retry policy")
        protocol = summary.get("scientific_protocol")
        invocation = summary.get("native_invocation")
        if not isinstance(protocol, Mapping) or protocol != {
            "method": "GFN2-xTB",
            "phase": "gas",
            "charge": 0,
            "uhf": 0,
            "threads": 1,
            "requested_final_angle_tolerance_deg": 2.0,
        }:
            raise EvaluationError("scientific attempt protocol changed")
        if (
            not isinstance(invocation, Mapping)
            or invocation.get("argv") != expected_argv
            or invocation.get("shell") is not False
            or invocation.get("environment")
            != {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            or invocation.get("xtb_version") != "6.7.1"
            or invocation.get("xtb_binary_sha256") != XTB_SHA256
        ):
            raise EvaluationError("native invocation identity changed")
        files = summary.get("safe_files")
        if not isinstance(files, list):
            raise EvaluationError("safe attempt inventory is invalid")
        safe_by_name: dict[str, Mapping[str, Any]] = {}
        for item in files:
            relative = item.get("path") if isinstance(item, Mapping) else None
            if (
                not isinstance(item, Mapping)
                or set(item) != {"path", "sha256", "size_bytes"}
                or not isinstance(relative, str)
                or ABSOLUTE_PATH.search(relative)
                or ".." in Path(relative).parts
            ):
                raise EvaluationError("attempt inventory path is unsafe")
            expected_prefix = f"artifacts/scan_summaries/{attempt_id}/"
            if not relative.startswith(expected_prefix):
                raise EvaluationError("attempt inventory path binding changed")
            name = relative.removeprefix(expected_prefix)
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if (
                name
                not in {
                    "input.xyz",
                    "xcontrol.inp",
                    "stdout.log",
                    "stderr.log",
                    "xtbopt.xyz",
                }
                or name in safe_by_name
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9A-F]{64}", digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise EvaluationError("safe attempt inventory is invalid")
            path = run_root / Path(relative)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != item.get("size_bytes")
                or _sha_file(path) != item.get("sha256")
            ):
                raise EvaluationError("safe attempt file binding changed")
            safe_by_name[name] = item
        required_safe_names = {
            "input.xyz",
            "xcontrol.inp",
            *(
                ("stdout.log", "stderr.log")
                if require_native_logs
                else ()
            ),
        }
        if not required_safe_names.issubset(safe_by_name):
            raise EvaluationError("prepared native inputs are not bound")
        expected_safe_order = ["input.xyz", "xcontrol.inp"]
        if "stdout.log" in safe_by_name or "stderr.log" in safe_by_name:
            if not {"stdout.log", "stderr.log"}.issubset(safe_by_name):
                raise EvaluationError("native process logs are incomplete")
            expected_safe_order.extend(["stdout.log", "stderr.log"])
        if "xtbopt.xyz" in safe_by_name:
            expected_safe_order.append("xtbopt.xyz")
        if [
            item["path"].rsplit("/", 1)[-1] for item in files
        ] != expected_safe_order:
            raise EvaluationError("safe attempt inventory order changed")
        actual_attempt_members = {
            member.name
            for member in root.iterdir()
            if member.is_file() and not member.is_symlink()
        }
        if actual_attempt_members != {"summary.json", *safe_by_name} or any(
            member.is_dir() or member.is_symlink() for member in root.iterdir()
        ):
            raise EvaluationError("attempt projection has unbound members")
        signed_angle = angle if angle <= 180 else angle - 360
        expected_constraint = (
            "$constrain\n"
            "  force constant=1.0\n"
            f"  dihedral: 6, 7, 8, 9, {float(signed_angle):.1f}\n"
            "$end\n"
        ).encode("ascii")
        constraint_path = run_root / Path(
            safe_by_name["xcontrol.inp"]["path"]
        )
        if constraint_path.read_bytes() != expected_constraint:
            raise EvaluationError("native constraint syntax changed")
        validation = summary.get("structure_validation")
        provenance = summary.get("provenance")
        if (
            not isinstance(validation, Mapping)
            or set(validation) != validation_fields
            or any(not isinstance(value, bool) for value in validation.values())
            or not isinstance(provenance, Mapping)
            or set(provenance)
            != (
                projected_provenance_fields
                if "stdout.log" in safe_by_name
                else legacy_provenance_fields
            )
            or provenance.get("reference_sha256") != REFERENCE_SHA256
            or provenance.get("input_sha256")
            != safe_by_name["input.xyz"].get("sha256")
            or provenance.get("constraint_sha256")
            != safe_by_name["xcontrol.inp"].get("sha256")
            or any(
                not isinstance(provenance.get(name), str)
                or not re.fullmatch(
                    r"[0-9A-F]{64}", str(provenance.get(name))
                )
                for name in (
                    "stdout_sha256",
                    "stderr_sha256",
                    "raw_attempt_manifest_sha256",
                )
            )
        ):
            raise EvaluationError("attempt provenance binding changed")
        if "stdout.log" in safe_by_name:
            if (
                provenance.get("native_log_projection")
                != NATIVE_LOG_PROJECTION
                or provenance.get("stdout_projection_sha256")
                != safe_by_name["stdout.log"].get("sha256")
                or provenance.get("stderr_projection_sha256")
                != safe_by_name["stderr.log"].get("sha256")
            ):
                raise EvaluationError("native log provenance changed")

        input_path = run_root / Path(safe_by_name["input.xyz"]["path"])
        input_symbols, input_coordinates, input_comment = _read_xyz_evidence(
            input_path
        )
        input_facts = _xyz_structure_facts(input_symbols, input_coordinates)
        input_angle = _dihedral_deg(input_coordinates[[5, 6, 7, 8], :])
        expected_input_comment = (
            "independent reference initialization; constrained UFF; "
            f"target={angle} deg"
        )
        if (
            not all(input_facts.values())
            or _circular_distance(input_angle, angle) > 0.1
            or input_comment != expected_input_comment
            or safe_by_name["input.xyz"].get("sha256") == REFERENCE_SHA256
        ):
            raise EvaluationError(
                "prepared geometry is not an independent constrained reference"
            )

        stdout_text: str | None = None
        stderr_text: str | None = None
        if "stdout.log" in safe_by_name:
            try:
                stdout_text = (
                    run_root / Path(safe_by_name["stdout.log"]["path"])
                ).read_text(encoding="utf-8", errors="replace")
                stderr_text = (
                    run_root / Path(safe_by_name["stderr.log"]["path"])
                ).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise EvaluationError("native process logs are unreadable") from exc
            if (
                RUNTIME_ABSOLUTE_TEXT.search(stdout_text)
                or RUNTIME_ABSOLUTE_TEXT.search(stderr_text)
            ):
                raise EvaluationError(
                    "native log projection contains an absolute path"
                )
        parsed_converged = (
            (
                "GEOMETRY OPTIMIZATION CONVERGED" in stdout_text
                and "FAILED TO CONVERGE" not in stdout_text
            )
            if stdout_text is not None
            else summary.get("converged") is True
        )
        normal_termination = (
            "normal termination of xtb" in stderr_text
            if stderr_text is not None
            else True
        )

        output_facts = {name: False for name in validation_fields}
        recomputed_final_angle: float | None = None
        stdout_energy: float | None = None
        comment_energy: float | None = None
        if "xtbopt.xyz" in safe_by_name:
            output_path = run_root / Path(safe_by_name["xtbopt.xyz"]["path"])
            try:
                output_symbols, output_coordinates, output_comment = (
                    _read_xyz_evidence(output_path)
                )
                output_facts = _xyz_structure_facts(
                    output_symbols, output_coordinates
                )
                recomputed_final_angle = _dihedral_deg(
                    output_coordinates[[5, 6, 7, 8], :]
                )
                stdout_energy, comment_energy = _parse_xtb_energy(
                    stdout_text=stdout_text,
                    xyz_comment=output_comment,
                )
            except EvaluationError:
                # A native failure may legitimately leave a partial xtbopt.xyz.
                # Such an attempt is retained but cannot be profile eligible.
                output_facts = {name: False for name in validation_fields}
                recomputed_final_angle = None
                stdout_energy = None
                comment_energy = None
        recomputed_angle_error = (
            _circular_distance(recomputed_final_angle, angle)
            if recomputed_final_angle is not None
            else None
        )
        energy_evidence_ok = bool(
            comment_energy is not None
            and (
                stdout_text is None
                or (
                    stdout_energy is not None
                    and math.isclose(
                        stdout_energy,
                        comment_energy,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                )
            )
        )
        summary_energy = summary.get("total_energy_hartree")
        energy_binding_ok = bool(
            energy_evidence_ok
            and _is_finite_number(summary_energy)
            and math.isclose(
                float(summary_energy),
                float(comment_energy),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        final_angle_binding_ok = bool(
            recomputed_final_angle is not None
            and _is_finite_number(summary.get("final_angle_deg"))
            and _is_finite_number(summary.get("angle_error_deg"), low=0.0)
            and math.isclose(
                float(summary["final_angle_deg"]) % 360.0,
                recomputed_final_angle,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            and math.isclose(
                float(summary["angle_error_deg"]),
                float(recomputed_angle_error),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        )
        if (
            dict(validation) != output_facts
            or summary.get("converged") is not parsed_converged
            or summary.get("energy_parsed") is not energy_evidence_ok
            or (
                energy_evidence_ok
                and not energy_binding_ok
            )
            or (
                not energy_evidence_ok
                and summary_energy is not None
            )
            or (
                recomputed_final_angle is None
                and (
                    summary.get("final_angle_deg") is not None
                    or summary.get("angle_error_deg") is not None
                )
            )
            or (
                recomputed_final_angle is not None
                and not final_angle_binding_ok
            )
        ):
            raise EvaluationError(
                "attempt scientific facts do not match native evidence"
            )

        independently_eligible = bool(
            summary.get("return_code") == 0
            and summary.get("timed_out") is False
            and parsed_converged
            and normal_termination
            and energy_binding_ok
            and final_angle_binding_ok
            and recomputed_angle_error is not None
            and recomputed_angle_error <= 2.0
            and "xtbopt.xyz" in safe_by_name
            and all(value is True for value in output_facts.values())
        )
        return_code = summary.get("return_code")
        energy = summary.get("total_energy_hartree")
        final_angle = summary.get("final_angle_deg")
        angle_error = summary.get("angle_error_deg")
        if (
            not _is_int(return_code)
            or not isinstance(summary.get("timed_out"), bool)
            or not isinstance(summary.get("converged"), bool)
            or not isinstance(summary.get("energy_parsed"), bool)
            or not isinstance(summary.get("profile_eligible"), bool)
            or summary.get("energy_parsed")
            is not _is_finite_number(energy)
            or (
                energy is not None
                and not _is_finite_number(energy)
            )
            or (
                final_angle is None
                and angle_error is not None
            )
            or (
                final_angle is not None
                and (
                    not _is_finite_number(final_angle)
                    or not _is_finite_number(angle_error, low=0.0)
                )
            )
        ):
            raise EvaluationError("attempt primitive facts are invalid")
        if summary.get("profile_eligible") is not independently_eligible:
            raise EvaluationError("attempt eligibility changed")
        if independently_eligible:
            if angle in eligible_by_angle:
                raise EvaluationError("native retry continued after success")
            eligible_by_angle[angle] = attempt_id
        summaries[attempt_id] = dict(summary)
    return {
        "summaries": summaries,
        "attempt_count": len(summaries),
        "attempt_counts_by_angle": dict(counts),
        "attempt_batches_by_angle": {
            angle: sorted(values)
            for angle, values in batches_by_angle.items()
        },
    }, eligible_by_angle


def _budget_matches(
    observed: Any,
    *,
    ledger: Sequence[Mapping[str, Any]],
    attempt_info: Mapping[str, Any],
    public_actions_used: int,
    terminal_failure: str | None,
    allow_legacy_without_terminal_failure: bool = False,
) -> bool:
    expected_fields = {
        "scan_batches_used",
        "scan_batches_remaining",
        "unique_angles_used",
        "unique_angles_remaining",
        "valid_unique_angles",
        "failed_unique_angles",
        "feedback_rounds",
        "profile_fits",
        "public_actions_used",
        "native_attempts",
        "terminal_failure",
        "elapsed_seconds",
    }
    if (
        allow_legacy_without_terminal_failure
        and isinstance(observed, Mapping)
        and "terminal_failure" not in observed
    ):
        expected_fields.remove("terminal_failure")
    if not isinstance(observed, Mapping) or set(observed) != expected_fields:
        return False
    try:
        partial_state = _reconstruct_state(ledger)
    except EvaluationError:
        return False
    seen_angles = {
        angle for record in ledger for angle in record["angles_deg"]
    }
    native_attempts = sum(
        count
        for angle, count in attempt_info["attempt_counts_by_angle"].items()
        if angle in seen_angles
    )
    expected = {
        "scan_batches_used": len(ledger),
        "scan_batches_remaining": max(0, 4 - len(ledger)),
        "unique_angles_used": partial_state["unique_angle_count"],
        "unique_angles_remaining": max(
            0, 12 - partial_state["unique_angle_count"]
        ),
        "valid_unique_angles": partial_state["valid_unique_angle_count"],
        "failed_unique_angles": len(partial_state["failed_angles"]),
        "feedback_rounds": partial_state["feedback_rounds"],
        "profile_fits": partial_state["profile_fit_rounds"],
        "public_actions_used": public_actions_used,
        "native_attempts": native_attempts,
        "terminal_failure": terminal_failure,
    }
    elapsed = observed.get("elapsed_seconds")
    return bool(
        all(
            (
                key == "terminal_failure"
                and allow_legacy_without_terminal_failure
            )
            or observed.get(key) == value
            for key, value in expected.items()
        )
        and _is_finite_number(elapsed, low=0.0, high=1200.0)
    )


def _expected_final_budget(
    state: Mapping[str, Any],
    *,
    public_actions_used: int,
    native_attempts: int,
    terminal_failure: str | None,
) -> dict[str, Any]:
    return {
        "scan_batches_used": int(state.get("batch_count", 0)),
        "scan_batches_remaining": max(
            0, 4 - int(state.get("batch_count", 0))
        ),
        "unique_angles_used": int(state.get("unique_angle_count", 0)),
        "unique_angles_remaining": max(
            0, 12 - int(state.get("unique_angle_count", 0))
        ),
        "valid_unique_angles": int(
            state.get("valid_unique_angle_count", 0)
        ),
        "failed_unique_angles": len(state.get("failed_angles", [])),
        "feedback_rounds": int(state.get("feedback_rounds", 0)),
        "profile_fits": int(state.get("profile_fit_rounds", 0)),
        "public_actions_used": public_actions_used,
        "native_attempts": native_attempts,
        "terminal_failure": terminal_failure,
    }


def _verify_public_action_flow(
    events: Sequence[Mapping[str, Any]],
    *,
    attempt_info: Mapping[str, Any],
    final_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pair every public action and independently reconstruct its budget."""

    pending: tuple[int, Mapping[str, Any]] | None = None
    ledger_seen: list[Mapping[str, Any]] = []
    action_turns: set[int] = set()
    action_count = 0
    inspection_count = 0
    prior_action_turn = 0
    baseline = (
        events[0]["payload"].get("requested_model_id")
        == "deterministic-baseline/2.0"
    )
    for index, event in enumerate(events):
        name = event["event"]
        payload = event["payload"]
        if name == "public_action":
            if pending is not None:
                raise EvaluationError("public action lacks an immediate result")
            action_count += 1
            if action_count > 5:
                raise EvaluationError("public action budget exceeded")
            turn = payload["turn"]
            action = payload["action"]
            if (
                turn in action_turns
                or turn <= prior_action_turn
                or index == 0
                or events[index - 1]["event"]
                not in {"provider_response_accepted", "baseline_decision"}
                or events[index - 1]["payload"].get("turn") != turn
            ):
                raise EvaluationError(
                    "public action is not bound to one fresh decision turn"
                )
            action_turns.add(turn)
            prior_action_turn = turn
            pending = (turn, action)
        elif name == "public_action_result":
            if (
                pending is None
                or index == 0
                or events[index - 1]["event"] != "public_action"
            ):
                raise EvaluationError("orphan or delayed public action result")
            turn, action = pending
            if (
                payload.get("turn") != turn
                or payload.get("action") != action.get("action")
            ):
                raise EvaluationError("public action/result binding changed")
            result = payload.get("result")
            if not isinstance(result, Mapping):
                raise EvaluationError("public action result is invalid")
            if action.get("action") == "run_scan_batch":
                record = result.get("batch_record")
                if (
                    not isinstance(record, Mapping)
                    or len(ledger_seen) >= len(final_ledger)
                    or dict(record) != dict(final_ledger[len(ledger_seen)])
                ):
                    raise EvaluationError("scan result ledger binding changed")
                ledger_seen.append(record)
                if not _budget_matches(
                    result.get("budget"),
                    ledger=ledger_seen,
                    attempt_info=attempt_info,
                    public_actions_used=action_count,
                    terminal_failure=None,
                    allow_legacy_without_terminal_failure=baseline,
                ):
                    raise EvaluationError("scan result budget changed")
            elif action.get("action") == "inspect_failed_attempt":
                inspection_count += 1
                angle = action["angle_deg"]
                failed_so_far = {
                    value
                    for record in ledger_seen
                    for value in record["failed_angles_deg"]
                }
                attempts = sorted(
                    (
                        item
                        for item in attempt_info["summaries"].values()
                        if item["requested_angle_deg"] == angle
                    ),
                    key=lambda item: item["attempt_number"],
                )
                expected_attempts = [
                    {
                        "attempt_id": item["attempt_id"],
                        "return_code": item["return_code"],
                        "timed_out": item["timed_out"],
                        "converged": item["converged"],
                        "energy_parsed": item["energy_parsed"],
                        "angle_error_deg": item["angle_error_deg"],
                        "structure_validation": item["structure_validation"],
                    }
                    for item in attempts
                ]
                partial_state = _expected_runtime_process(
                    ledger_seen, _reconstruct_state(ledger_seen)
                )
                if (
                    angle not in failed_so_far
                    or len(attempts) != 2
                    or any(item["profile_eligible"] for item in attempts)
                    or result.get("attempts") != expected_attempts
                    or result.get("replacement_sampling_allowed")
                    is not partial_state["recovery_allowed"]
                    or not _budget_matches(
                        result.get("budget"),
                        ledger=ledger_seen,
                        attempt_info=attempt_info,
                        public_actions_used=action_count,
                        terminal_failure=None,
                        allow_legacy_without_terminal_failure=baseline,
                    )
                ):
                    raise EvaluationError(
                        "failed-attempt inspection evidence changed"
                    )
            else:
                raise EvaluationError("unexpected public action")
            pending = None
    if pending is not None or list(ledger_seen) != list(final_ledger):
        raise EvaluationError("public action flow is incomplete")
    return {
        "public_action_count": action_count,
        "inspection_count": inspection_count,
        "action_turns": sorted(action_turns),
    }


def _verify_profiles(
    run_root: Path,
    ledger: Sequence[Mapping[str, Any]],
    attempts: Mapping[str, Any],
) -> dict[str, Any]:
    validated: dict[str, dict[str, Any]] = {}
    valid_energies: dict[int, float] = {}
    for record in ledger:
        for angle in record["valid_angles_deg"]:
            candidates = [
                item
                for item in attempts.values()
                if item["requested_angle_deg"] == angle
                and item["profile_eligible"] is True
            ]
            if len(candidates) != 1:
                raise EvaluationError("valid angle lacks exactly one native source")
            valid_energies[angle] = float(candidates[0]["total_energy_hartree"])
        profile_id = record["profile_id"]
        path = run_root / "artifacts" / "profiles" / f"{profile_id}.json"
        profile = _strict_json(path)
        if not isinstance(profile, Mapping) or profile.get("profile_id") != profile_id:
            raise EvaluationError("profile id binding changed")
        verify_profile_artifact(profile)
        if profile.get("profile_hash") != record["profile_hash"]:
            raise EvaluationError("batch profile hash binding changed")
        angles = sorted(valid_energies)
        minimum = min(valid_energies.values())
        energies = [
            (valid_energies[angle] - minimum) * HARTREE_TO_KCAL_MOL
            for angle in angles
        ]
        if profile["source_angles_deg"] != angles or not np.allclose(
            profile["source_energies"], energies, rtol=0.0, atol=1.0e-9
        ):
            raise EvaluationError("profile uses data outside current native trace")
        expected_attempt_ids = [
            next(
                item["attempt_id"]
                for item in attempts.values()
                if item["requested_angle_deg"] == angle
                and item["profile_eligible"] is True
            )
            for angle in angles
        ]
        if profile["source_attempt_ids"] != expected_attempt_ids:
            raise EvaluationError("profile attempt provenance changed")
        validated[profile_id] = dict(profile)
    return {
        "profiles": validated,
        "profile_count": len(validated),
        "terminal_profile": validated[ledger[-1]["profile_id"]],
    }


def _strict_deep_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like values without Python's bool/int equivalence."""

    if type(left) is not type(right):
        if (
            not isinstance(left, bool)
            and not isinstance(right, bool)
            and isinstance(left, (int, float))
            and isinstance(right, (int, float))
        ):
            return bool(float(left) == float(right))
        return False
    if isinstance(left, Mapping):
        return bool(
            set(left) == set(right)
            and all(
                _strict_deep_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, list):
        return bool(
            len(left) == len(right)
            and all(
                _strict_deep_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return bool(left == right)


def _compact_profile_projection(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently reproduce the bounded model-visible feedback."""

    analysis = profile["analysis"]
    signals = profile["refinement_signals"]
    source_energies = [float(value) for value in profile["source_energies"]]
    minimum = min(source_energies)
    selected_candidate = next(
        candidate
        for candidate in profile["selector"]["candidates"]
        if candidate["method"] == profile["method"]
        and candidate["order"] == profile["order"]
    )
    return {
        "selected_fit": {
            "family": profile["method"],
            "fourier_order": profile["order"],
            "loo_rmse_kcal_mol": selected_candidate[
                "loo_rmse_kcal_mol"
            ],
        },
        "sampled_angles_deg": list(profile["source_angles_deg"]),
        "relative_energies_kcal_mol": [
            round(value - minimum, 8) for value in source_energies
        ],
        "global_minimum_basin_candidates_deg": list(
            analysis["global_minimum"]["degenerate_angles_deg"]
        ),
        "local_minima_candidates_deg": [
            item["angle_deg"] for item in analysis["local_minima"]
        ],
        "local_maxima_candidates_deg": [
            item["angle_deg"] for item in analysis["local_maxima"]
        ],
        "largest_unsampled_gaps": [
            {
                "start_deg": item["start_angle_deg"],
                "end_deg": item["end_angle_deg"],
                "width_deg": item["width_deg"],
                "recommended_deg": item["recommended_angle_deg"],
            }
            for item in analysis["largest_gaps"][:3]
        ],
        "high_curvature_intervals": [
            {
                "center_deg": item["center_angle_deg"],
                "strength": round(
                    float(item["abs_second_derivative_per_rad2"]), 6
                ),
            }
            for item in analysis["high_curvature_intervals"][:3]
        ],
        "high_model_disagreement_intervals": [
            {
                "center_deg": item["center_angle_deg"],
                "abs_difference": round(
                    float(item["abs_difference"]), 6
                ),
            }
            for item in analysis["model_disagreement"]["intervals"][:3]
        ],
        "recommended_refinement": [
            {
                "angle_deg": item["angle_deg"],
                "reason": item["reason"],
                "priority_rank": item["priority_rank"],
            }
            for item in signals["recommendations"][:4]
        ],
    }


def _verify_scan_result_feedback(
    events: Sequence[Mapping[str, Any]],
    *,
    ledger: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Any]],
    attempts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind every model-visible scan result to independently verified facts."""

    batch_index = 0
    for event in events:
        if event["event"] != "public_action_result":
            continue
        payload = event["payload"]
        if payload.get("action") != "run_scan_batch":
            continue
        if batch_index >= len(ledger):
            # A partial native timeout has no completed batch or profile
            # feedback. Its closed failure result is handled separately.
            continue
        record = ledger[batch_index]
        result = payload["result"]
        if result.get("batch_record") != record:
            raise EvaluationError("scan feedback batch binding changed")
        profile_id = record.get("profile_id")
        if profile_id is None:
            # Phase-0 native impossibility has no scientific fit to expose.
            batch_index += 1
            continue
        profile = profiles.get(profile_id)
        if not isinstance(profile, Mapping):
            raise EvaluationError("scan feedback profile is unverified")
        prefix = list(ledger[: batch_index + 1])
        state = _reconstruct_state(prefix)
        process = _expected_runtime_process(prefix, state)
        phase = record["phase"]
        if record["terminal_accepted"]:
            status = "FINAL_ACCEPTED"
        elif (
            phase == "phase2"
            and record["finalize"]
            and process["recovery_allowed"]
        ):
            status = "RECOVERY_REQUIRED"
        elif phase in {"phase2", "recovery"} and record["finalize"]:
            status = "FINAL_ELIGIBILITY_FAILED"
        else:
            status = "BATCH_COMPLETE"
        point_results = []
        for angle in record["angles_deg"]:
            angle_attempts = sorted(
                (
                    item
                    for item in attempts.values()
                    if item["requested_angle_deg"] == angle
                ),
                key=lambda item: item["attempt_number"],
            )
            if not angle_attempts:
                raise EvaluationError("scan feedback lacks native attempts")
            latest = angle_attempts[-1]
            point_results.append(
                {
                    "angle_deg": angle,
                    "status": (
                        "valid"
                        if angle in record["valid_angles_deg"]
                        else "failed"
                    ),
                    "attempt_count": len(angle_attempts),
                    "final_angle_deg": latest["final_angle_deg"],
                    "angle_error_deg": latest["angle_error_deg"],
                }
            )
        expected_without_budget = {
            "ok": status
            in {"BATCH_COMPLETE", "FINAL_ACCEPTED", "RECOVERY_REQUIRED"},
            "status": status,
            "batch_number": record["batch_number"],
            "phase": phase,
            "point_results": point_results,
            "profile": {
                "profile_id": profile_id,
                "profile_hash": profile["profile_hash"],
                "grid_size": 72,
                "all_finite": True,
                "analysis": _compact_profile_projection(profile),
            },
            "state": {
                "phase_completed": phase,
                "next_phase": process["current_phase"],
                "sampled_angles_deg": process["unique_angles"],
                "valid_angles_deg": process["valid_angles"],
                "failed_angles_deg": process["failed_angles"],
                "unique_angles_used": process["unique_angle_count"],
                "unique_angles_remaining": process[
                    "unique_angles_remaining"
                ],
                "valid_unique_angles": process[
                    "valid_unique_angle_count"
                ],
                "feedback_rounds": process["feedback_rounds"],
                "profile_fit_rounds": process["profile_fit_rounds"],
                "recovery_allowed": process["recovery_allowed"],
                "finalize_allowed": process["current_phase"]
                in {"phase2", "recovery"},
                "eligible": process["eligible"],
                "eligibility_gaps": process["eligibility_gaps"],
            },
            "terminal": {
                "requested": record["finalize"],
                "accepted": record["terminal_accepted"],
                "provider_must_stop": bool(
                    record["terminal_accepted"]
                    or status == "FINAL_ELIGIBILITY_FAILED"
                ),
            },
            "batch_record": dict(record),
        }
        observed_without_budget = {
            key: value for key, value in result.items() if key != "budget"
        }
        expected_for_character_count = {
            **expected_without_budget,
            "budget": result.get("budget"),
        }
        if (
            set(result) != {*expected_without_budget, "budget"}
            or not _strict_deep_equal(
                observed_without_budget, expected_without_budget
            )
            or payload.get("tool_result_chars")
            != len(
                _canonical_pretty(expected_for_character_count).decode(
                    "utf-8"
                )
            )
        ):
            raise EvaluationError(
                "model-visible profile feedback changed"
            )
        batch_index += 1
    if batch_index != len(ledger):
        raise EvaluationError("scan feedback sequence is incomplete")


def _path_or_secret_findings(root: Path) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    secrets: list[str] = []

    def secret_like(value: Any, *, depth: int = 0) -> bool:
        if depth > 4:
            return False
        if isinstance(value, str):
            if SECRET.search(value) or re.search(
                r"(?i)[\"']?(?:[A-Za-z0-9_-]*api[_-]?key|token|secret|"
                r"authorization|credential|password)[\"']?\s*[:=]\s*"
                r"[\"']?\S+",
                value,
            ):
                return True
            stripped = value.strip()
            if stripped.startswith(("{", "[")) and stripped.endswith(
                ("}", "]")
            ):
                try:
                    decoded = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    return False
                if isinstance(decoded, (Mapping, list)):
                    return secret_like(decoded, depth=depth + 1)
            return False
        if isinstance(value, Mapping):
            for key, child in value.items():
                sensitive_key = bool(
                    re.search(
                        r"(?i)(?:api[_-]?key|token|secret|authorization|"
                        r"credential|password)",
                        str(key),
                    )
                )
                if (
                    sensitive_key
                    and isinstance(child, str)
                    and child.strip()
                    and child.strip().casefold()
                    not in {
                        "none",
                        "null",
                        "false",
                        "unavailable",
                        "redacted",
                        "<redacted>",
                    }
                ):
                    return True
                if secret_like(child, depth=depth + 1):
                    return True
            return False
        if isinstance(value, list):
            return any(secret_like(item, depth=depth + 1) for item in value)
        return False

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if _link_or_reparse(path):
            paths.append(f"{relative}:link_or_reparse")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".txt", ".log", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if SECRET.search(text):
            secrets.append(relative)
        if RUNTIME_ABSOLUTE_TEXT.search(text):
            paths.append(f"{relative}:embedded_absolute_path")
        if path.suffix.lower() in {".json", ".jsonl"}:
            values: list[Any]
            if path.suffix.lower() == ".jsonl":
                try:
                    values = [json.loads(line) for line in text.splitlines()]
                except json.JSONDecodeError:
                    continue
            else:
                try:
                    values = [json.loads(text)]
                except json.JSONDecodeError:
                    continue

            def visit(value: Any, key: str = "") -> None:
                if isinstance(value, Mapping):
                    for child_key, child in value.items():
                        visit(child, str(child_key))
                elif isinstance(value, list):
                    for child in value:
                        visit(child, key)
                elif isinstance(value, str):
                    path_field = (
                        key.lower().endswith(("path", "file", "root", "executable"))
                        or key.lower() in {"cwd", "workdir"}
                    )
                    if path_field and ABSOLUTE_PATH.search(value.strip()):
                        paths.append(f"{relative}:{key}")

            for value in values:
                if secret_like(value):
                    secrets.append(relative)
                visit(value)
    return sorted(set(paths)), sorted(set(secrets))


def _link_or_reparse(path: Path) -> bool:
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError as exc:
        raise EvaluationError("execution inventory entry is unreadable") from exc
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or (callable(is_junction) and is_junction())
    )


def _execution_excluded(relative: PurePosixPath) -> bool:
    folded = tuple(part.casefold() for part in relative.parts)
    return bool(
        folded
        and (
            folded[0] in {item.casefold() for item in WORK_EXCLUSIONS}
            or set(folded) & {item.casefold() for item in CACHE_EXCLUSIONS}
        )
    )


def _independent_execution_entries(project: Path) -> list[dict[str, Any]]:
    """Rebuild the Gate-8 inventory without importing runtime/package code."""

    root = project.resolve(strict=True)
    candidates: list[Path] = []
    for name in sorted(EXECUTION_ROOT_FILES):
        path = root / name
        if path.is_file():
            candidates.append(path)
    for name in sorted(EXECUTION_TOP_LEVEL | {"scripts"}):
        base = root / name
        if not base.exists():
            continue
        if _link_or_reparse(base):
            raise EvaluationError("execution inventory contains a link")
        if base.is_file():
            descendants = [base]
        elif base.is_dir():
            descendants = list(base.rglob("*"))
        else:
            raise EvaluationError("execution inventory has a special file")
        for path in descendants:
            try:
                relative = PurePosixPath(path.relative_to(root).as_posix())
            except ValueError as exc:
                raise EvaluationError("execution inventory escaped") from exc
            if _link_or_reparse(path):
                raise EvaluationError("execution inventory contains a link")
            if _execution_excluded(relative):
                continue
            if path.is_dir():
                continue
            if not path.is_file():
                raise EvaluationError("execution inventory has a special file")
            if (
                relative.parts[0] == "scripts"
                and path.name in EXECUTION_SCRIPT_EXCLUSIONS
            ):
                continue
            if (
                relative.parts[0] == "configs"
                and "/".join(relative.parts[1:])
                in EXECUTION_CONFIG_EXCLUSIONS
            ):
                continue
            candidates.append(path)
    by_name: dict[str, Path] = {}
    folded: set[str] = set()
    for path in candidates:
        relative = path.resolve(strict=True).relative_to(root).as_posix()
        pure = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
            or relative in by_name
            or relative.casefold() in folded
        ):
            raise EvaluationError("execution inventory path is unsafe")
        by_name[relative] = path
        folded.add(relative.casefold())
    return [
        {
            "path": name,
            "size_bytes": by_name[name].stat().st_size,
            "sha256": _sha_file(by_name[name]),
        }
        for name in sorted(by_name)
    ]


def _adaptive_facts(ledger: Sequence[Mapping[str, Any]], attempt_count: int) -> dict[str, float]:
    all_angles = sorted(
        angle for record in ledger for angle in record["angles_deg"]
    )
    gaps = [
        (all_angles[(index + 1) % len(all_angles)] - angle) % 360
        for index, angle in enumerate(all_angles)
    ]
    coverage = min(1.0, 60.0 / max(gaps)) if gaps and max(gaps) else 1.0
    hits = 0
    total = 0
    previous = list(ledger[0]["angles_deg"]) if ledger else []
    for record in ledger[1:3]:
        ordered = sorted(previous)
        intervals = []
        for index, start in enumerate(ordered):
            end = ordered[(index + 1) % len(ordered)]
            width = (end - start) % 360
            intervals.append((start, width))
        maximum = max(width for _start, width in intervals)
        for angle in record["angles_deg"]:
            total += 1
            containing = next(
                (item for item in intervals if (angle - item[0]) % 360 < item[1]),
                None,
            )
            if containing is None:
                continue
            start, width = containing
            midpoint = (start + width / 2.0) % 360
            distance = abs((angle - midpoint + 180.0) % 360 - 180.0)
            if width >= maximum - 1.0e-12 and distance <= 15.0:
                hits += 1
        previous.extend(record["angles_deg"])
    unique_count = len(all_angles)
    efficiency = (
        min(1.0, 3.0 / len(ledger))
        * min(1.0, unique_count / max(attempt_count, 1))
        if ledger
        else 0.0
    )
    return {
        "coverage_fraction": coverage,
        "feedback_fraction": min(1.0, max(0, len(ledger) - 1) / 2.0),
        "refinement_fraction": hits / total if total else 0.0,
        "efficiency_fraction": efficiency,
    }


def _verify_manifest(run_root: Path) -> bool:
    manifest = _strict_json(run_root / "run_manifest.json")
    if not isinstance(manifest, Mapping):
        return False
    expected_files = []
    for path in sorted(
        run_root.rglob("*"),
        key=lambda item: item.relative_to(run_root).as_posix(),
    ):
        if path.is_file() and path.name not in {
            "run_manifest.json",
            "evaluation.json",
        }:
            expected_files.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha_file(path),
                }
            )
    return manifest.get("files") == expected_files


def _verify_formal_run_manifest(
    run_root: Path,
    events: Sequence[Mapping[str, Any]],
    *,
    requested_model: Any,
    actual_model: Any,
    run_status: str,
    blocker: Any,
    execution_freeze: Mapping[str, Any],
    attempt_count: int,
    expected_budget: Mapping[str, Any],
) -> bool:
    """Validate the closed formal manifest, not only its file inventory."""

    try:
        manifest = _strict_json(run_root / "run_manifest.json")
    except EvaluationError:
        return False
    expected_fields = {
        "schema_version",
        "run_id",
        "status",
        "blocker",
        "requested_model_id",
        "actual_model_id",
        "exact_identity",
        "fallback_used",
        "model_override_used",
        "task_id",
        "benchmark_version",
        "case_id",
        "protocol",
        "execution_freeze_binding",
        "trajectory_commitment",
        "budget_at_end",
        "files",
    }
    trajectory_commitment = {
        "algorithm": "sha256",
        "event_count": len(events),
        "head_hash": events[-1]["event_hash"],
    }
    budget = manifest.get("budget_at_end") if isinstance(manifest, Mapping) else None
    if not isinstance(budget, Mapping):
        return False
    if set(budget) != {*expected_budget, "elapsed_seconds"}:
        return False
    elapsed = budget.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
        or float(elapsed) > 1200.0
        or any(
            budget.get(name) != value
            for name, value in expected_budget.items()
        )
        or budget.get("native_attempts") != attempt_count
    ):
        return False
    return bool(
        isinstance(manifest, Mapping)
        and set(manifest) == expected_fields
        and manifest.get("schema_version") == "2.0"
        and manifest.get("run_id") == events[0]["run_id"]
        and manifest.get("status") == run_status
        and manifest.get("blocker") == blocker
        and manifest.get("requested_model_id") == requested_model
        and manifest.get("actual_model_id") == actual_model
        and manifest.get("exact_identity")
        is (actual_model == requested_model)
        and manifest.get("fallback_used") is False
        and manifest.get("model_override_used") is False
        and manifest.get("task_id") == TASK_ID
        and manifest.get("benchmark_version") == "2.0.0"
        and manifest.get("case_id") == CASE_ID
        and manifest.get("protocol") == PROTOCOL
        and manifest.get("execution_freeze_binding")
        == dict(execution_freeze)
        and manifest.get("trajectory_commitment") == trajectory_commitment
        and _verify_manifest(run_root)
    )


def _verify_usage_report(
    run_root: Path,
    events: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    attempt_count: int,
) -> dict[str, Any]:
    """Recompute formal resource accounting from trajectory and artifacts."""

    usage = _strict_json(run_root / "usage_report.json")
    fields = {
        "schema_version",
        "run_id",
        "mode",
        "provider_turns",
        "successful_provider_turns",
        "retried_attempts",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "provider_usage_unavailable",
        "turns",
        "native_xtb_attempts",
        "successful_unique_angles",
        "failed_unique_angles",
        "wall_clock_seconds",
    }
    if not isinstance(usage, Mapping) or set(usage) != fields:
        return {"passed": False, "reason": "USAGE_REPORT_SCHEMA"}
    observed_turns = usage.get("turns")
    if not isinstance(observed_turns, list):
        return {"passed": False, "reason": "USAGE_TURNS"}
    provider_events = [
        event
        for event in events
        if event["event"]
        in {
            "provider_response_accepted",
            "provider_response_rejected_identity",
        }
    ]
    failed_events = [
        event for event in events if event["event"] == "provider_attempt_failed"
    ]
    responses_by_turn: dict[int, Mapping[str, Any]] = {}
    for event in provider_events:
        turn = event["payload"].get("turn")
        if not _is_int(turn, low=1, high=5) or turn in responses_by_turn:
            return {"passed": False, "reason": "USAGE_RESPONSE_CARDINALITY"}
        responses_by_turn[int(turn)] = event["payload"]
    failed_by_turn: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in failed_events:
        turn = event["payload"].get("turn")
        if not _is_int(turn, low=1, high=5):
            return {"passed": False, "reason": "USAGE_RETRY_TURN"}
        failed_by_turn[int(turn)].append(event["payload"])

    provider_turn_count = int(usage.get("provider_turns", -1))
    if (
        not _is_int(provider_turn_count, low=0, high=5)
        or len(observed_turns) != provider_turn_count
        or {
            *responses_by_turn,
            *failed_by_turn,
        }
        - set(range(1, provider_turn_count + 1))
    ):
        return {"passed": False, "reason": "USAGE_TURN_COUNT"}
    action_results: dict[int, Mapping[str, Any]] = {}
    for event in events:
        if event["event"] != "public_action_result":
            continue
        turn = int(event["payload"]["turn"])
        if turn in action_results:
            return {"passed": False, "reason": "USAGE_MULTIPLE_ACTION_RESULTS"}
        action_results[turn] = event["payload"]

    expected_turns: list[dict[str, Any]] = []
    token_available = True
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    retry_count = 0
    prior_protocol_rerequest_turn: int | None = None
    violations_by_turn = {
        int(event["payload"]["turn"]): event["payload"]
        for event in events
        if event["event"] == "protocol_violation"
    }
    for turn in range(1, provider_turn_count + 1):
        observed = observed_turns[turn - 1]
        if (
            not isinstance(observed, Mapping)
            or set(observed)
            != {
                "turn",
                "canonical_state_chars",
                "canonical_state_sha256",
                "output_chars",
                "usage",
                "tool_result_chars",
            }
            or observed.get("turn") != turn
            or not _is_int(
                observed.get("canonical_state_chars"), low=1, high=7000
            )
            or not isinstance(observed.get("canonical_state_sha256"), str)
            or not UPPER_SHA256.fullmatch(
                observed["canonical_state_sha256"]
            )
        ):
            return {"passed": False, "reason": "USAGE_TURN_SKELETON"}
        if prior_protocol_rerequest_turn is not None:
            previous = observed_turns[prior_protocol_rerequest_turn - 1]
            if (
                turn != prior_protocol_rerequest_turn + 1
                or observed["canonical_state_sha256"]
                != previous["canonical_state_sha256"]
                or observed["canonical_state_chars"]
                != previous["canonical_state_chars"]
            ):
                return {"passed": False, "reason": "FORMAT_REREQUEST_STATE_CHANGED"}
            prior_protocol_rerequest_turn = None
        violation = violations_by_turn.get(turn)
        if (
            violation is not None
            and violation.get("same_state_rerequest_allowed") is True
        ):
            prior_protocol_rerequest_turn = turn

        failures = failed_by_turn.get(turn, [])
        cumulative_delay = 0.0
        for position, failure in enumerate(failures, start=1):
            category = failure.get("category")
            expected_retry = bool(
                category in TRANSIENT_PROVIDER_CATEGORIES and position < 4
            )
            delay = failure.get("planned_delay_seconds")
            if (
                failure.get("attempt_number") != position
                or failure.get("retryable")
                is not (category in TRANSIENT_PROVIDER_CATEGORIES)
                or failure.get("will_retry") is not expected_retry
                or not _is_finite_number(delay, low=0.0, high=30.0)
                or (
                    expected_retry
                    and float(delay) <= 0.0
                )
                or (
                    not expected_retry
                    and float(delay) != 0.0
                )
            ):
                return {"passed": False, "reason": "PROVIDER_RETRY_POLICY"}
            if expected_retry:
                retry_count += 1
                cumulative_delay = round(cumulative_delay + float(delay), 6)
                if cumulative_delay > 60.0:
                    return {
                        "passed": False,
                        "reason": "PROVIDER_RETRY_DELAY_BUDGET",
                    }
        response = responses_by_turn.get(turn)
        if failures:
            last_retry = failures[-1].get("will_retry") is True
            if (response is not None) is not last_retry:
                return {"passed": False, "reason": "PROVIDER_RETRY_TERMINATION"}

        if response is None:
            if (
                observed.get("output_chars") is not None
                or observed.get("usage") is not None
                or observed.get("tool_result_chars") is not None
            ):
                return {"passed": False, "reason": "FAILED_TURN_NOT_EMPTY"}
            token_available = False
            expected_turns.append(dict(observed))
            continue
        event_usage = response.get("usage")
        if event_usage is None:
            token_available = False
        else:
            _validate_provider_usage(event_usage)
            input_tokens += int(event_usage["input_tokens"])
            output_tokens += int(event_usage["output_tokens"])
            total_tokens += int(event_usage["total_tokens"])
        expected = {
            "turn": turn,
            "canonical_state_chars": observed["canonical_state_chars"],
            "canonical_state_sha256": response[
                "canonical_state_sha256"
            ],
            "output_chars": response["content_chars"],
            "usage": dict(event_usage) if event_usage is not None else None,
            "tool_result_chars": (
                action_results[turn]["tool_result_chars"]
                if turn in action_results
                else None
            ),
        }
        if dict(observed) != expected:
            return {"passed": False, "reason": "USAGE_TURN_BINDING"}
        expected_turns.append(expected)

    if prior_protocol_rerequest_turn is not None:
        return {"passed": False, "reason": "FORMAT_REREQUEST_NOT_PERFORMED"}
    if sum(
        event["event"] == "protocol_violation" for event in events
    ) > 1:
        return {"passed": False, "reason": "FORMAT_REREQUEST_BUDGET"}
    unavailable = (
        not token_available or bool(failed_events) or provider_turn_count == 0
    )
    expected_tokens = (
        (None, None, None)
        if unavailable
        else (input_tokens, output_tokens, total_tokens)
    )
    wall = usage.get("wall_clock_seconds")
    passed = bool(
        usage.get("schema_version") == "2.0"
        and usage.get("run_id") == events[0].get("run_id")
        and usage.get("mode") == "REAL_API_LIVE_XTB"
        and usage.get("successful_provider_turns") == len(provider_events)
        and usage.get("retried_attempts") == retry_count
        and usage.get("provider_usage_unavailable") is unavailable
        and (
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )
        == expected_tokens
        and usage.get("turns") == expected_turns
        and usage.get("native_xtb_attempts") == attempt_count
        and usage.get("successful_unique_angles")
        == state.get("valid_unique_angle_count")
        and usage.get("failed_unique_angles")
        == len(state.get("failed_angles", []))
        and _is_finite_number(wall, low=0.0, high=1200.0)
    )
    return {
        "passed": passed,
        "reason": None if passed else "USAGE_TOTALS_OR_COUNTS",
        "provider_turns": provider_turn_count,
        "successful_provider_turns": len(provider_events),
        "retried_attempts": retry_count,
        "native_xtb_attempts": attempt_count,
    }


def _verify_formal_decision_flow(
    events: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct that each refinement decision followed fresh feedback."""

    accepted: dict[int, int] = {}
    action_indices: list[tuple[int, int]] = []
    result_indices: list[tuple[int, int]] = []
    protocol_violations = 0
    for index, event in enumerate(events):
        name = event["event"]
        payload = event["payload"]
        if name == "provider_response_accepted":
            turn = payload.get("turn")
            if (
                isinstance(turn, bool)
                or not isinstance(turn, int)
                or turn in accepted
                or payload.get("exact_identity") is not True
                or payload.get("raw_content_persisted") is not False
            ):
                return {"passed": False, "reason": "PROVIDER_DECISION_EVENT"}
            accepted[turn] = index
        elif name == "public_action":
            action = payload.get("action")
            if isinstance(action, Mapping) and action.get("action") == "run_scan_batch":
                action_indices.append((int(payload.get("turn", -1)), index))
        elif name == "public_action_result":
            if payload.get("action") == "run_scan_batch":
                result_indices.append((int(payload.get("turn", -1)), index))
        elif name == "protocol_violation":
            protocol_violations += 1
            if (
                payload.get("category") != "PROTOCOL_VIOLATION"
                or payload.get("raw_content_persisted") is not False
            ):
                return {"passed": False, "reason": "PROTOCOL_REREQUEST_EVENT"}
    if (
        len(action_indices) != int(state.get("batch_count", -1))
        or len(result_indices) != len(action_indices)
        or protocol_violations > 1
    ):
        return {"passed": False, "reason": "DECISION_FLOW_COUNTS"}
    prior_result_index = -1
    prior_turn = 0
    for position, ((turn, action_index), (result_turn, result_index)) in enumerate(
        zip(action_indices, result_indices)
    ):
        provider_index = accepted.get(turn)
        if (
            turn != result_turn
            or turn <= prior_turn
            or provider_index is None
            or not prior_result_index < provider_index < action_index < result_index
        ):
            return {"passed": False, "reason": "FEEDBACK_EVENT_ORDER"}
        if position > 0 and provider_index <= prior_result_index:
            return {"passed": False, "reason": "FEEDBACK_NOT_OBSERVED"}
        prior_result_index = result_index
        prior_turn = turn
    reconstructed_feedback = max(0, min(2, len(action_indices) - 1))
    passed = reconstructed_feedback == state.get("feedback_rounds") == 2
    return {
        "passed": passed,
        "reason": None if passed else "FEEDBACK_ROUND_RECONSTRUCTION",
        "scan_decision_turns": [turn for turn, _ in action_indices],
        "feedback_rounds": reconstructed_feedback,
        "protocol_rerequests": protocol_violations,
    }


def _verify_execution_freeze_binding(
    project: Path,
    run_start: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently bind a formal trajectory to the root Gate-8 snapshot."""

    start_binding = run_start.get("execution_freeze_binding")
    runtime_binding = runtime.get("execution_freeze_binding")
    fields = {
        "schema_version",
        "content_sha256",
        "commitment_file_sha256",
        "file_count",
        "total_bytes",
    }
    if (
        not isinstance(start_binding, Mapping)
        or not isinstance(runtime_binding, Mapping)
        or dict(start_binding) != dict(runtime_binding)
        or set(start_binding) != fields
        or start_binding.get("schema_version")
        != "execution-freeze-binding/2.0"
    ):
        return {"passed": False, "reason": "BINDING_SCHEMA_OR_EVENT_MISMATCH"}
    for field in ("content_sha256", "commitment_file_sha256"):
        digest = start_binding.get(field)
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9A-F]{64}", digest)
        ):
            return {"passed": False, "reason": "BINDING_DIGEST_INVALID"}
    for field in ("file_count", "total_bytes"):
        value = start_binding.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return {"passed": False, "reason": "BINDING_COUNT_INVALID"}

    commitment_path = project / "execution_commitment.json"
    if not commitment_path.is_file():
        return {"passed": False, "reason": "ROOT_COMMITMENT_MISSING"}
    try:
        commitment = _strict_json(commitment_path)
    except EvaluationError:
        return {"passed": False, "reason": "ROOT_COMMITMENT_INVALID"}
    try:
        if commitment_path.read_bytes() != _canonical_pretty(commitment):
            return {
                "passed": False,
                "reason": "ROOT_COMMITMENT_NONCANONICAL",
            }
    except OSError:
        return {"passed": False, "reason": "ROOT_COMMITMENT_INVALID"}
    expected_keys = {
        "commitment_kind",
        "content_sha256",
        "file_count",
        "files",
        "identity",
        "schema_version",
        "total_bytes",
    }
    identity = {
        "benchmark_version": "2.0.0",
        "case_id": CASE_ID,
        "protocol": PROTOCOL,
        "task_id": TASK_ID,
    }
    if (
        not isinstance(commitment, Mapping)
        or set(commitment) != expected_keys
        or commitment.get("schema_version") != "execution-commitment/2.0"
        or commitment.get("commitment_kind") != "execution"
        or commitment.get("identity") != identity
    ):
        return {"passed": False, "reason": "ROOT_COMMITMENT_SCHEMA"}
    files = commitment.get("files")
    if not isinstance(files, list):
        return {"passed": False, "reason": "ROOT_COMMITMENT_FILES"}
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    folded_names: set[str] = set()
    for item in files:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256", "size_bytes"}
        ):
            return {"passed": False, "reason": "ROOT_COMMITMENT_FILES"}
        relative = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or relative.startswith("/")
            or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or relative in names
            or relative.casefold() in folded_names
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9A-F]{64}", digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            return {"passed": False, "reason": "ROOT_COMMITMENT_FILES"}
        candidate = project.joinpath(*relative.split("/"))
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(project.resolve(strict=True))
        except (OSError, ValueError):
            return {"passed": False, "reason": "ROOT_COMMITMENT_FILE_MISSING"}
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or resolved.stat().st_size != size
            or _sha_file(resolved) != digest
        ):
            return {"passed": False, "reason": "ROOT_COMMITMENT_FILE_MISMATCH"}
        names.add(relative)
        folded_names.add(relative.casefold())
        normalized.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        return {"passed": False, "reason": "ROOT_COMMITMENT_ORDER"}
    try:
        expected_inventory = _independent_execution_entries(project)
    except EvaluationError:
        return {"passed": False, "reason": "ROOT_EXECUTION_INVENTORY_INVALID"}
    if normalized != expected_inventory:
        return {
            "passed": False,
            "reason": "ROOT_EXECUTION_INVENTORY_MISMATCH",
        }
    content_value = {
        "commitment_kind": "execution",
        "identity": identity,
        "files": normalized,
    }
    content_sha = _sha_bytes(_canonical_pretty(content_value))
    file_sha = _sha_file(commitment_path)
    if (
        commitment.get("content_sha256") != content_sha
        or commitment.get("file_count") != len(normalized)
        or commitment.get("total_bytes")
        != sum(int(item["size_bytes"]) for item in normalized)
        or start_binding.get("content_sha256") != content_sha
        or start_binding.get("commitment_file_sha256") != file_sha
        or start_binding.get("file_count") != len(normalized)
        or start_binding.get("total_bytes")
        != sum(int(item["size_bytes"]) for item in normalized)
    ):
        return {"passed": False, "reason": "ROOT_COMMITMENT_BINDING_MISMATCH"}
    return {
        "passed": True,
        "reason": None,
        "content_sha256": content_sha,
        "commitment_file_sha256": file_sha,
    }


def _valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not UTC_TIMESTAMP.fullmatch(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() is not None


def _expected_attempt_validity(
    attempt_info: Mapping[str, Any],
) -> dict[str, Any]:
    attempts = sorted(
        attempt_info["summaries"].values(),
        key=lambda item: (
            int(item["requested_angle_deg"]),
            int(item["attempt_number"]),
        ),
    )
    projected = [
        {
            "attempt_id": item["attempt_id"],
            "angle_deg": item["requested_angle_deg"],
            "attempt_number": item["attempt_number"],
            "profile_eligible": item["profile_eligible"],
            "raw_attempt_manifest_sha256": item["provenance"][
                "raw_attempt_manifest_sha256"
            ],
        }
        for item in attempts
    ]
    return {
        "attempt_count": len(projected),
        "eligible_attempt_count": sum(
            item["profile_eligible"] is True for item in projected
        ),
        "attempts": projected,
    }


def _verify_success_envelope(
    *,
    run_root: Path,
    events: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    final: Mapping[str, Any],
    start: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    attempt_info: Mapping[str, Any],
    terminal_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every trusted success field against independent evidence."""

    if set(final) != {"model_submission", "trusted_runtime"}:
        raise EvaluationError("final envelope is not closed")
    submission = final.get("model_submission")
    runtime = final.get("trusted_runtime")
    if not isinstance(submission, Mapping) or not isinstance(runtime, Mapping):
        raise EvaluationError("final envelope members are invalid")
    runtime_fields = {
        "task_id",
        "benchmark_version",
        "case_id",
        "protocol",
        "status",
        "actual_model_identity",
        "model_identity",
        "profile_id",
        "profile_hash",
        "profile",
        "trajectory_hash",
        "trajectory",
        "unique_angles",
        "valid_unique_angles",
        "feedback_rounds",
        "profile_fit_rounds",
        "batch_count",
        "process_recomputed",
        "attempt_validity",
        "scientific_invariants",
        "execution_commitment",
        "execution_commitment_payload",
        "execution_freeze_binding",
        "evaluator_version",
        "timestamp",
    }
    requested = start["requested_model_id"]
    baseline = requested == "deterministic-baseline/2.0"
    baseline_has_freeze_field = baseline and "execution_freeze_binding" in runtime
    if baseline and not baseline_has_freeze_field:
        # The byte-frozen baseline predates Gate 8 and therefore has no root
        # execution-freeze member.  Formal runs must carry it.
        runtime_fields.remove("execution_freeze_binding")
    if set(runtime) != runtime_fields:
        raise EvaluationError("trusted success runtime is not closed")
    response_events = [
        event
        for event in events
        if event["event"]
        in {
            "provider_response_accepted",
            "provider_response_rejected_identity",
        }
    ]
    if baseline:
        decisions = [
            event for event in events if event["event"] == "baseline_decision"
        ]
        actual = (
            decisions[-1]["payload"]["actual_model_id"] if decisions else None
        )
        identity_events_ok = bool(
            len(decisions) == state["batch_count"]
            and all(
                event["payload"]["actual_model_id"] == requested
                for event in decisions
            )
            and not response_events
        )
    else:
        actual = (
            response_events[-1]["payload"]["actual_model_id"]
            if response_events
            else None
        )
        identity_events_ok = bool(
            response_events
            and not any(
                event["event"] == "provider_response_rejected_identity"
                for event in response_events
            )
            and all(
                event["payload"]["actual_model_id"] == requested
                and event["payload"]["exact_identity"] is True
                for event in response_events
            )
        )
    identity = runtime.get("model_identity")
    expected_identity = {
        "requested": requested,
        "actual": requested,
        "exact_match": True,
        "fallback_used": False,
        "model_override_used": False,
    }
    if (
        not identity_events_ok
        or actual != requested
        or runtime.get("actual_model_identity") != requested
        or identity != expected_identity
    ):
        raise EvaluationError("trusted exact model identity changed")

    profile_path = (
        run_root
        / "artifacts"
        / "profiles"
        / f"{terminal_profile['profile_id']}.json"
    )
    profile_artifact_sha = _sha_file(profile_path)
    expected_profile = {
        "profile_id": terminal_profile["profile_id"],
        "profile_hash": terminal_profile["profile_hash"],
        "artifact_sha256": profile_artifact_sha,
        "grid_angles_deg": GRID,
        "predicted_energy_kcal_mol": terminal_profile[
            "predicted_energies"
        ],
        "all_finite": True,
    }
    expected_trajectory = {
        "head_hash": trajectory["head_hash"],
        "file_sha256": trajectory["file_sha256"],
        "event_count": trajectory["event_count"],
    }
    expected_attempts = _expected_attempt_validity(attempt_info)
    expected_process = _expected_runtime_process(ledger, state)
    expected_science = {
        "method": "GFN2-xTB",
        "xtb_version": "6.7.1",
        "phase": "gas",
        "charge": 0,
        "uhf": 0,
        "threads": 1,
        "angle_grid_deg": 5,
        "reference_seed": 20260720,
        "reference_geometry_sha256": REFERENCE_SHA256,
    }
    freeze_binding = (
        None if baseline else dict(start["execution_freeze_binding"])
    )
    expected_payload = {
        "schema_version": "2.0",
        "task_id": TASK_ID,
        "case_id": CASE_ID,
        "protocol": PROTOCOL,
        "requested_model_id": requested,
        "actual_model_id": requested,
        "trajectory_head_hash": trajectory["head_hash"],
        "trajectory_sha256": trajectory["file_sha256"],
        "profile_id": terminal_profile["profile_id"],
        "profile_hash": terminal_profile["profile_hash"],
        "profile_artifact_sha256": profile_artifact_sha,
        "batch_ledger": list(ledger),
        "attempt_manifest_hashes": [
            item["raw_attempt_manifest_sha256"]
            for item in expected_attempts["attempts"]
        ],
    }
    if not baseline or baseline_has_freeze_field:
        expected_payload["execution_freeze_binding"] = freeze_binding
    execution_digest = _sha_bytes(_canonical_pretty(expected_payload))
    if (
        runtime.get("task_id") != TASK_ID
        or runtime.get("benchmark_version") != "2.0.0"
        or runtime.get("case_id") != CASE_ID
        or runtime.get("protocol") != PROTOCOL
        or runtime.get("status") != "PASS"
        or runtime.get("evaluator_version") != EVALUATOR_VERSION
        or not _valid_utc_timestamp(runtime.get("timestamp"))
        or runtime.get("profile_id") != terminal_profile["profile_id"]
        or runtime.get("profile_hash") != terminal_profile["profile_hash"]
        or runtime.get("profile") != expected_profile
        or runtime.get("trajectory_hash") != trajectory["head_hash"]
        or runtime.get("trajectory") != expected_trajectory
        or runtime.get("unique_angles") != state["unique_angle_count"]
        or runtime.get("valid_unique_angles")
        != state["valid_unique_angle_count"]
        or runtime.get("feedback_rounds") != state["feedback_rounds"]
        or runtime.get("profile_fit_rounds") != state["profile_fit_rounds"]
        or runtime.get("batch_count") != state["batch_count"]
        or runtime.get("process_recomputed") != expected_process
        or runtime.get("attempt_validity") != expected_attempts
        or runtime.get("scientific_invariants") != expected_science
        or (
            (not baseline or baseline_has_freeze_field)
            and runtime.get("execution_freeze_binding") != freeze_binding
        )
        or runtime.get("execution_commitment_payload") != expected_payload
        or runtime.get("execution_commitment") != execution_digest
    ):
        raise EvaluationError("trusted success envelope binding changed")

    submitted_event = events[-2]["payload"]
    complete_event = events[-1]["payload"]
    if (
        submitted_event.get("model_submission") != submission
        or submitted_event.get("profile_id") != terminal_profile["profile_id"]
        or submitted_event.get("profile_hash") != terminal_profile["profile_hash"]
        or submitted_event.get("turn")
        != next(
            (
                event["payload"]["turn"]
                for event in reversed(events)
                if event["event"] in {"public_action", "public_action_result"}
            ),
            None,
        )
        or complete_event
        != {
            "status": "PASS",
            "exact_model_identity": True,
            "profile_id": terminal_profile["profile_id"],
        }
    ):
        raise EvaluationError("terminal success events changed")

    commitment_path = run_root / "execution_commitment.json"
    commitment = _strict_json(commitment_path)
    expected_commitment = {
        "schema_version": "2.0",
        "algorithm": "SHA-256",
        "execution_commitment": execution_digest,
        "payload": expected_payload,
    }
    if (
        commitment != expected_commitment
        or commitment_path.read_bytes() != _canonical_pretty(expected_commitment)
    ):
        raise EvaluationError("run execution commitment changed")
    return {
        "submission": submission,
        "runtime": runtime,
        "actual_model": actual,
        "profile_artifact_sha256": profile_artifact_sha,
        "execution_commitment": execution_digest,
    }


def _failure_process_state(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unique = [angle for record in records for angle in record["angles_deg"]]
    valid = [angle for record in records for angle in record["valid_angles_deg"]]
    failed = [
        angle for record in records for angle in record["failed_angles_deg"]
    ]
    profiles = [
        record["profile_id"]
        for record in records
        if isinstance(record.get("profile_id"), str)
    ]
    return {
        "batch_count": len(records),
        "unique_angles": unique,
        "unique_angle_count": len(unique),
        "valid_angles": valid,
        "valid_unique_angle_count": len(set(valid)),
        "failed_angles": failed,
        "feedback_rounds": max(0, min(2, len(records) - 1)),
        "profile_fit_rounds": len(profiles),
        "profile_ids": profiles,
    }


def _verify_failure_action_flow(
    events: Sequence[Mapping[str, Any]],
    *,
    attempt_info: Mapping[str, Any],
    safe_status: str,
) -> dict[str, Any]:
    pending: tuple[int, Mapping[str, Any]] | None = None
    records: list[Mapping[str, Any]] = []
    profile_records: list[Mapping[str, Any]] = []
    action_count = 0
    action_turns: set[int] = set()
    prior_turn = 0
    last_result: Mapping[str, Any] | None = None
    phases = ("phase0", "phase1", "phase2", "recovery")
    for index, event in enumerate(events):
        if event["event"] == "public_action":
            if pending is not None:
                raise EvaluationError("failure action lacks a result")
            action_count += 1
            turn = event["payload"]["turn"]
            action = event["payload"]["action"]
            if (
                action_count > 5
                or turn in action_turns
                or turn <= prior_turn
                or index == 0
                or events[index - 1]["event"]
                not in {"provider_response_accepted", "baseline_decision"}
                or events[index - 1]["payload"].get("turn") != turn
            ):
                raise EvaluationError("failure public action flow changed")
            action_turns.add(turn)
            prior_turn = turn
            pending = (turn, action)
        elif event["event"] == "public_action_result":
            if (
                pending is None
                or index == 0
                or events[index - 1]["event"] != "public_action"
            ):
                raise EvaluationError("failure has an orphan action result")
            turn, action = pending
            payload = event["payload"]
            result = payload.get("result")
            if (
                payload.get("turn") != turn
                or payload.get("action") != action.get("action")
                or not isinstance(result, Mapping)
            ):
                raise EvaluationError("failure action/result binding changed")
            if action.get("action") == "run_scan_batch":
                record = result.get("batch_record")
                if record is None:
                    if (
                        result.get("ok") is not False
                        or not isinstance(result.get("status"), str)
                        or result.get("status")
                        not in {
                            "TOTAL_NATIVE_TIMEOUT_EXCEEDED",
                            "NATIVE_INVARIANT_BREACH",
                            "PHASE0_BATCH_SIZE_INVALID",
                            "PHASE1_BATCH_SIZE_INVALID",
                            "PHASE2_BATCH_SIZE_INVALID",
                            "RECOVERY_BATCH_SIZE_INVALID",
                            "PRIOR_ANGLE_DUPLICATE",
                            "UNIQUE_ANGLE_BUDGET_EXCEEDED",
                            "EARLY_FINALIZE_FORBIDDEN",
                            "PHASE2_REQUIRES_TERMINAL_FINALIZE",
                            "RECOVERY_NOT_ALLOWED",
                            "RECOVERY_REQUIRES_TERMINAL_FINALIZE",
                            "TERMINAL_FINALIZATION_ALREADY_ACCEPTED",
                            "TERMINAL_FAILURE_ALREADY_RECORDED",
                        }
                    ):
                        raise EvaluationError(
                            "failure result has no legal batch record"
                        )
                else:
                    if not isinstance(record, Mapping) or len(records) >= 4:
                        raise EvaluationError("failure batch record is invalid")
                    phase = phases[len(records)]
                    angles = _closed_action(action, phase)
                    expected_fields = {
                        "schema_version",
                        "batch_number",
                        "phase",
                        "angles_deg",
                        "valid_angles_deg",
                        "failed_angles_deg",
                        "finalize",
                        "confidence",
                        "rationale",
                        "action_hash",
                        "profile_id",
                        "profile_hash",
                        "terminal_accepted",
                    }
                    if (
                        set(record) != expected_fields
                        or record.get("schema_version") != "2.0"
                        or record.get("batch_number") != len(records) + 1
                        or record.get("phase") != phase
                        or record.get("angles_deg") != angles
                        or record.get("finalize") is not action["finalize"]
                        or record.get("confidence") != action["confidence"]
                        or record.get("rationale") != action["rationale"]
                        or record.get("terminal_accepted") is not False
                        or record.get("action_hash")
                        != _sha_bytes(_canonical_pretty(action))
                    ):
                        raise EvaluationError("failure batch record changed")
                    valid = record.get("valid_angles_deg")
                    failed = record.get("failed_angles_deg")
                    prior_angles = {
                        angle for prior in records for angle in prior["angles_deg"]
                    }
                    if (
                        not isinstance(valid, list)
                        or not isinstance(failed, list)
                        or set(valid) | set(failed) != set(angles)
                        or set(valid) & set(failed)
                        or set(angles) & prior_angles
                    ):
                        raise EvaluationError("failure batch outcomes changed")
                    for angle in angles:
                        summaries = [
                            item
                            for item in attempt_info["summaries"].values()
                            if item["requested_angle_deg"] == angle
                        ]
                        eligible = [
                            item
                            for item in summaries
                            if item["profile_eligible"] is True
                        ]
                        if (
                            angle in valid
                            and len(eligible) != 1
                        ) or (
                            angle in failed
                            and (
                                len(summaries) != 2
                                or eligible
                            )
                        ):
                            raise EvaluationError(
                                "failure native outcomes changed"
                            )
                    profile_id = record.get("profile_id")
                    if profile_id is None:
                        if (
                            record.get("profile_hash") is not None
                            or result.get("status")
                            != "NATIVE_ELIGIBILITY_IMPOSSIBLE"
                            or result.get("profile") is not None
                        ):
                            raise EvaluationError(
                                "native impossibility record changed"
                            )
                    else:
                        if (
                            not isinstance(profile_id, str)
                            or not isinstance(record.get("profile_hash"), str)
                        ):
                            raise EvaluationError(
                                "failure profile binding changed"
                            )
                        profile_records.append(record)
                    records.append(record)
            elif action.get("action") == "inspect_failed_attempt":
                angle = action.get("angle_deg")
                attempts = sorted(
                    (
                        item
                        for item in attempt_info["summaries"].values()
                        if item["requested_angle_deg"] == angle
                    ),
                    key=lambda item: item["attempt_number"],
                )
                failed_so_far = {
                    value
                    for record in records
                    for value in record["failed_angles_deg"]
                }
                expected_attempts = [
                    {
                        "attempt_id": item["attempt_id"],
                        "return_code": item["return_code"],
                        "timed_out": item["timed_out"],
                        "converged": item["converged"],
                        "energy_parsed": item["energy_parsed"],
                        "angle_error_deg": item["angle_error_deg"],
                        "structure_validation": item["structure_validation"],
                    }
                    for item in attempts
                ]
                if (
                    angle not in failed_so_far
                    or len(attempts) != 2
                    or any(item["profile_eligible"] for item in attempts)
                    or result.get("attempts") != expected_attempts
                ):
                    raise EvaluationError(
                        "failure inspection evidence changed"
                    )
            else:
                raise EvaluationError("failure contains an unknown action")
            pending = None
            last_result = result
    if pending is not None:
        raise EvaluationError("failure action lacks a terminal result")
    state = _failure_process_state(records)
    referenced_angles = set(state["unique_angles"])
    attempt_angles = set(attempt_info["attempt_counts_by_angle"])
    if safe_status not in {
        "TOTAL_NATIVE_TIMEOUT_EXCEEDED",
        "NATIVE_INVARIANT_BREACH",
    }:
        if attempt_angles != referenced_angles:
            raise EvaluationError("failure has unbound native attempts")
    elif not attempt_angles.issubset(
        referenced_angles
        | {
            angle
            for event in events
            if event["event"] == "public_action"
            for angle in event["payload"]["action"].get("angles_deg", [])
        }
    ):
        raise EvaluationError("timeout has unrelated native attempts")
    return {
        "state": state,
        "records": records,
        "profile_records": profile_records,
        "public_action_count": action_count,
        "last_result": last_result,
    }


def _verify_failure_envelope(
    *,
    final: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    trajectory: Mapping[str, Any],
    start: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(final) != {"model_submission", "trusted_runtime"}
        or final.get("model_submission") != {}
        or not isinstance(final.get("trusted_runtime"), Mapping)
    ):
        raise EvaluationError("failure envelope is not closed")
    runtime = final["trusted_runtime"]
    expected_fields = {
        "task_id",
        "benchmark_version",
        "case_id",
        "protocol",
        "status",
        "safe_status",
        "requested_model_id",
        "actual_model_id",
        "trajectory_hash",
        "trajectory_sha256",
        "execution_freeze_binding",
        "fallback_used",
        "model_override_used",
        "raw_provider_response_persisted",
        "timestamp",
    }
    if set(runtime) != expected_fields:
        raise EvaluationError("trusted failure runtime is not closed")
    terminal = events[-1]["payload"]
    status = runtime.get("status")
    safe_status = runtime.get("safe_status")
    legal_status = bool(
        (
            status == "BLOCKED"
            and safe_status in BLOCKED_SAFE_STATUSES
        )
        or (
            status == "FAIL"
            and safe_status in FAIL_SAFE_STATUSES
        )
    )
    if (
        not legal_status
        or terminal.get("status") != status
        or terminal.get("safe_status") != safe_status
        or runtime.get("task_id") != TASK_ID
        or runtime.get("benchmark_version") != "2.0.0"
        or runtime.get("case_id") != CASE_ID
        or runtime.get("protocol") != PROTOCOL
        or runtime.get("requested_model_id")
        != start.get("requested_model_id")
        or runtime.get("trajectory_hash") != trajectory["head_hash"]
        or runtime.get("trajectory_sha256") != trajectory["file_sha256"]
        or runtime.get("execution_freeze_binding")
        != start.get("execution_freeze_binding")
        or runtime.get("fallback_used") is not False
        or runtime.get("model_override_used") is not False
        or runtime.get("raw_provider_response_persisted") is not False
        or not _valid_utc_timestamp(runtime.get("timestamp"))
    ):
        raise EvaluationError("trusted failure binding changed")
    detail = terminal.get("detail")
    if not isinstance(detail, Mapping) or not set(detail).issubset(
        {"provider_turn", "error_category", "action_status", "exception_type"}
    ):
        raise EvaluationError("failure detail is not closed")
    if "provider_turn" in detail and not _is_int(
        detail["provider_turn"], low=0, high=5
    ):
        raise EvaluationError("failure provider turn is invalid")
    if (
        "error_category" in detail
        and detail["error_category"] != safe_status
    ) or (
        "action_status" in detail
        and detail["action_status"] != safe_status
    ) or (
        "exception_type" in detail
        and (
            not isinstance(detail["exception_type"], str)
            or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]{0,79}",
                detail["exception_type"],
            )
        )
    ):
        raise EvaluationError("failure detail changed")

    identity_events = [
        event
        for event in events
        if event["event"]
        in {
            "provider_response_accepted",
            "provider_response_rejected_identity",
        }
    ]
    rejected = [
        event
        for event in identity_events
        if event["event"] == "provider_response_rejected_identity"
    ]
    requested = start["requested_model_id"]
    actual = (
        identity_events[-1]["payload"]["actual_model_id"]
        if identity_events
        else None
    )
    if rejected:
        if (
            len(rejected) != 1
            or rejected[-1] is not identity_events[-1]
            or safe_status != "IDENTITY_MISMATCH"
            or status != "FAIL"
        ):
            raise EvaluationError("identity rejection terminal semantics changed")
    elif any(
        event["payload"]["actual_model_id"] != requested
        or event["payload"]["exact_identity"] is not True
        for event in identity_events
    ):
        raise EvaluationError("failure exact identity history changed")
    if (
        runtime.get("actual_model_id") != actual
        or (
            actual is not None
            and (
                not isinstance(actual, str)
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}", actual
                )
                or SECRET.search(actual)
            )
        )
    ):
        raise EvaluationError("failure actual model identity changed")
    return {
        "runtime": runtime,
        "status": status,
        "safe_status": safe_status,
        "requested_model": requested,
        "actual_model": actual,
        "exact_identity": bool(
            identity_events
            and not rejected
            and all(
                event["payload"]["actual_model_id"] == requested
                for event in identity_events
            )
        ),
    }


def evaluate_run(
    run_root: Path,
    *,
    project_root: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    project = (
        project_root.resolve(strict=True)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    events, trajectory = _load_trajectory(run_root / "trajectory.jsonl")
    validated_start = _validate_event_contracts(events)
    final = _strict_json(run_root / "final_answer.json")
    if trajectory["terminal_event"] == "run_failure":
        start = validated_start
        runtime = (
            final.get("trusted_runtime", {})
            if isinstance(final, Mapping)
            else {}
        )
        legal_failure_envelope = False
        usage_audit: dict[str, Any] = {
            "passed": False,
            "reason": "FAILURE_AUDIT_NOT_COMPLETED",
        }
        manifest_ok = False
        actual_model_identity: Any = None
        exact_model_identity = False
        execution_freeze: dict[str, Any] = {
            "passed": False,
            "reason": "FAILURE_ENVELOPE_INVALID",
        }
        failure_reason: str | None = None
        try:
            if not isinstance(final, Mapping):
                raise EvaluationError("failure final is not an object")
            failure_binding = _verify_failure_envelope(
                final=final,
                events=events,
                trajectory=trajectory,
                start=start,
            )
            runtime = failure_binding["runtime"]
            requested_model = failure_binding["requested_model"]
            actual_model_identity = failure_binding["actual_model"]
            exact_model_identity = failure_binding["exact_identity"]
            execution_freeze = (
                {"passed": True, "reason": "BASELINE_PRECEDES_GATE_8"}
                if requested_model == "deterministic-baseline/2.0"
                else _verify_execution_freeze_binding(project, start, runtime)
            )
            failure_attempt_info, _eligible = _verify_attempts(
                run_root,
                require_native_logs=(
                    requested_model != "deterministic-baseline/2.0"
                ),
            )
            failure_flow = _verify_failure_action_flow(
                events,
                attempt_info=failure_attempt_info,
                safe_status=failure_binding["safe_status"],
            )
            failure_state = failure_flow["state"]
            failure_profiles: dict[str, Any] = {"profiles": {}}
            if failure_flow["profile_records"]:
                failure_profiles = _verify_profiles(
                    run_root,
                    failure_flow["profile_records"],
                    failure_attempt_info["summaries"],
                )
            _verify_scan_result_feedback(
                events,
                ledger=failure_flow["records"],
                profiles=failure_profiles["profiles"],
                attempts=failure_attempt_info["summaries"],
            )
            usage_audit = _verify_usage_report(
                run_root,
                events,
                failure_state,
                int(failure_attempt_info["attempt_count"]),
            )
            terminal_failure = (
                failure_binding["safe_status"]
                if failure_binding["safe_status"]
                in {
                    "NATIVE_ELIGIBILITY_IMPOSSIBLE",
                    "FINAL_ELIGIBILITY_FAILED",
                    "NATIVE_INVARIANT_BREACH",
                }
                else None
            )
            manifest_ok = _verify_formal_run_manifest(
                run_root,
                events,
                requested_model=requested_model,
                actual_model=actual_model_identity,
                run_status=failure_binding["status"],
                blocker=failure_binding["safe_status"],
                execution_freeze=start.get("execution_freeze_binding", {}),
                attempt_count=int(failure_attempt_info["attempt_count"]),
                expected_budget=_expected_final_budget(
                    failure_state,
                    public_actions_used=failure_flow[
                        "public_action_count"
                    ],
                    native_attempts=int(
                        failure_attempt_info["attempt_count"]
                    ),
                    terminal_failure=terminal_failure,
                ),
            )
            path_findings, secret_findings = _path_or_secret_findings(run_root)
            legal_failure_envelope = bool(
                execution_freeze["passed"]
                and usage_audit["passed"]
                and manifest_ok
                and not path_findings
                and not secret_findings
            )
            if not legal_failure_envelope:
                failure_reason = "FAILURE_GATE_RECOMPUTATION"
        except (
            EvaluationError,
            OSError,
            UnicodeError,
            ValueError,
            TypeError,
        ) as exc:
            failure_reason = str(exc)[:240]
            legal_failure_envelope = False
        result = {
            "schema_version": "2.0",
            "evaluator_version": EVALUATOR_VERSION,
            "run_status": (
                str(runtime.get("status"))
                if legal_failure_envelope
                else "FAIL"
            ),
            "eligible": False,
            "total_score": 0.0,
            "blocker": runtime.get("safe_status"),
            "trajectory": trajectory,
            "legal_failure_envelope": legal_failure_envelope,
            "execution_freeze_binding": execution_freeze,
            "actual_model_identity": actual_model_identity,
            "exact_model_identity": exact_model_identity,
            "usage_accounting": usage_audit,
            "artifact_manifest": manifest_ok,
            "failure_audit_reason": failure_reason,
        }
        if write:
            with (run_root / "evaluation.json").open("xb") as handle:
                handle.write(_canonical(result))
        return result
    ledger = _action_ledger(events)
    state = _reconstruct_state(ledger)
    attempt_info, eligible_by_angle = _verify_attempts(
        run_root,
        require_native_logs=(
            validated_start.get("requested_model_id")
            != "deterministic-baseline/2.0"
        ),
    )
    profiles = _verify_profiles(
        run_root, ledger, attempt_info["summaries"]
    )
    _verify_scan_result_feedback(
        events,
        ledger=ledger,
        profiles=profiles["profiles"],
        attempts=attempt_info["summaries"],
    )
    public_action_audit = _verify_public_action_flow(
        events,
        attempt_info=attempt_info,
        final_ledger=ledger,
    )
    terminal_profile = profiles["terminal_profile"]

    if not isinstance(final, Mapping) or set(final) != {
        "model_submission",
        "trusted_runtime",
    }:
        raise EvaluationError("final envelope is not closed")
    submission = final["model_submission"]
    runtime = final["trusted_runtime"]
    if not isinstance(submission, Mapping) or not isinstance(runtime, Mapping):
        raise EvaluationError("final envelope members are invalid")
    final_record = ledger[-1]
    start = validated_start
    requested_model = start.get("requested_model_id")
    success_binding = _verify_success_envelope(
        run_root=run_root,
        events=events,
        trajectory=trajectory,
        final=final,
        start=start,
        ledger=ledger,
        state=state,
        attempt_info=attempt_info,
        terminal_profile=terminal_profile,
    )
    submission_bound = bool(
        set(submission)
        == {
            "case_id",
            "final_status",
            "confidence",
            "rationale",
            "finalize_action_hash",
        }
        and submission.get("case_id") == CASE_ID
        and submission.get("final_status") == "completed"
        and submission.get("confidence") == final_record.get("confidence")
        and submission.get("rationale") == final_record.get("rationale")
        and submission.get("finalize_action_hash")
        == final_record.get("action_hash")
    )
    actual_models = {success_binding["actual_model"]}
    exact_identity = True
    if requested_model == "deterministic-baseline/2.0":
        execution_freeze = {
            "passed": True,
            "reason": "BASELINE_PRECEDES_GATE_8",
        }
    else:
        execution_freeze = _verify_execution_freeze_binding(
            project, start, runtime
        )
    profile_curve = terminal_profile["predicted_energies"]
    runtime_profile = runtime.get("profile", {})
    envelope_bound = runtime_profile.get("predicted_energy_kcal_mol") == profile_curve
    execution_bound = True
    reference_ok = _sha_file(project / "benchmark_case" / "reference.xyz") == REFERENCE_SHA256
    scientific_invariants = bool(
        reference_ok
        and all(
            item["native_invocation"]["xtb_version"] == "6.7.1"
            and item["native_invocation"]["xtb_binary_sha256"] == XTB_SHA256
            for item in attempt_info["summaries"].values()
        )
    )
    native_points_legal = bool(
        set(attempt_info["attempt_counts_by_angle"])
        == set(state["unique_angles"])
        and set(state["valid_angles"]) == set(eligible_by_angle)
        and all(
            attempt_info["attempt_counts_by_angle"].get(angle) == 2
            for angle in state["failed_angles"]
        )
        and all(
            attempt_info["attempt_counts_by_angle"].get(angle) in {1, 2}
            for angle in state["valid_angles"]
        )
        and all(
            attempt_info["attempt_batches_by_angle"].get(angle)
            == [
                next(
                    record["batch_number"]
                    for record in ledger
                    if angle in record["angles_deg"]
                )
            ]
            for angle in state["unique_angles"]
        )
    )
    path_findings, secret_findings = _path_or_secret_findings(run_root)
    trajectory_ok = True
    manifest_ok = (
        _verify_manifest(run_root)
        if requested_model == "deterministic-baseline/2.0"
        else _verify_formal_run_manifest(
            run_root,
            events,
            requested_model=requested_model,
            actual_model=next(iter(actual_models), None),
            run_status="PASS",
            blocker=None,
            execution_freeze=start.get("execution_freeze_binding", {}),
            attempt_count=attempt_info["attempt_count"],
            expected_budget=_expected_final_budget(
                state,
                public_actions_used=public_action_audit[
                    "public_action_count"
                ],
                native_attempts=attempt_info["attempt_count"],
                terminal_failure=None,
            ),
        )
    )
    artifact_ok = bool(
        profiles["profile_count"] == state["profile_fit_rounds"]
        and manifest_ok
    )
    usage_audit = (
        {
            "passed": True,
            "reason": "BASELINE_HAS_NO_PROVIDER_USAGE",
        }
        if requested_model == "deterministic-baseline/2.0"
        else _verify_usage_report(
            run_root, events, state, attempt_info["attempt_count"]
        )
    )
    decision_flow = (
        {
            "passed": True,
            "reason": "BASELINE_DETERMINISTIC_DECISIONS",
        }
        if requested_model == "deterministic-baseline/2.0"
        else _verify_formal_decision_flow(events, state)
    )
    gates = {
        "unique_angles_within_budget": state["unique_angle_count"] <= 12,
        "minimum_valid_unique_angles": state["valid_unique_angle_count"] >= 10,
        "minimum_feedback_rounds": state["feedback_rounds"] >= 2,
        "minimum_automatic_profile_fits": state["profile_fit_rounds"] >= 3,
        "terminal_profile_72_finite": (
            len(profile_curve) == 72
            and all(math.isfinite(float(value)) for value in profile_curve)
        ),
        "all_points_from_legal_native_attempts": native_points_legal,
        "scientific_invariants_unchanged": scientific_invariants,
        "exact_model_identity": exact_identity,
        "trajectory_integrity": trajectory_ok,
        "final_envelope_binding": submission_bound
        and envelope_bound
        and execution_bound,
        "no_forbidden_access": not path_findings,
        "no_secret_leakage": not secret_findings,
    }
    oracle_path = project / "evaluator_private" / "oracle_profile.json"
    commitment = _strict_json(
        project / "evaluator_private" / "oracle_commitment.json"
    )
    if (
        commitment.get("oracle_profile_sha256") != _sha_file(oracle_path)
        or commitment.get("point_count") != 72
        or commitment.get("reference_xyz_sha256") != REFERENCE_SHA256
    ):
        raise EvaluationError("hidden oracle commitment changed")
    oracle = _strict_json(oracle_path)
    facts = {
        "adaptive_sampling": _adaptive_facts(
            ledger, attempt_info["attempt_count"]
        ),
        "process_and_reproducibility": {
            "trajectory_integrity": trajectory_ok,
            "artifact_provenance": artifact_ok,
            "exact_identity": exact_identity,
            "scientific_invariants": scientific_invariants,
            "replay_consistency": True,
        },
    }
    scoring = score_profile_v2(
        predicted_energies_kcal_mol=profile_curve,
        oracle_energies_kcal_mol=oracle["relative_energies_kcal_mol"],
        facts=facts,
        eligibility_gates=gates,
    )
    result = {
        "schema_version": "2.0",
        "evaluator_version": EVALUATOR_VERSION,
        "run_status": "PASS" if scoring["eligible"] else "FAIL",
        "eligible": scoring["eligible"],
        "total_score": scoring["total_score"],
        "ungated_total_score": scoring["ungated_total_score"],
        "metrics": scoring["metrics"],
        "categories": scoring["categories"],
        "eligibility": scoring["eligibility"],
        "reconstructed": {
            "state": state,
            "attempt_count": attempt_info["attempt_count"],
            "trajectory": trajectory,
            "profile_id": terminal_profile["profile_id"],
            "profile_hash": terminal_profile["profile_hash"],
            "actual_model_identity": next(iter(actual_models), None),
            "path_findings": path_findings,
            "secret_findings": secret_findings,
            "artifact_provenance": artifact_ok,
            "execution_commitment": execution_bound,
        },
        "scoring": scoring,
    }
    if requested_model != "deterministic-baseline/2.0":
        gates["gate8_execution_freeze_binding"] = execution_freeze["passed"]
        gates["usage_accounting"] = usage_audit["passed"]
        gates["feedback_event_order"] = decision_flow["passed"]
        gates["artifact_manifest_and_profile_provenance"] = artifact_ok
        # Re-score after adding the formal-only Gate-8 eligibility fact.
        scoring = score_profile_v2(
            predicted_energies_kcal_mol=profile_curve,
            oracle_energies_kcal_mol=oracle["relative_energies_kcal_mol"],
            facts=facts,
            eligibility_gates=gates,
        )
        result["run_status"] = "PASS" if scoring["eligible"] else "FAIL"
        result["eligible"] = scoring["eligible"]
        result["total_score"] = scoring["total_score"]
        result["ungated_total_score"] = scoring["ungated_total_score"]
        result["metrics"] = scoring["metrics"]
        result["categories"] = scoring["categories"]
        result["eligibility"] = scoring["eligibility"]
        result["scoring"] = scoring
        result["reconstructed"]["execution_freeze_binding"] = execution_freeze
        result["reconstructed"]["usage_accounting"] = usage_audit
        result["reconstructed"]["decision_flow"] = decision_flow
    if write:
        with (run_root / "evaluation.json").open("xb") as handle:
            handle.write(_canonical(result))
    return result


__all__ = ["EVALUATOR_VERSION", "EvaluationError", "evaluate_run"]

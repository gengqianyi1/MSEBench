"""Canonical, append-only, SHA-256 chained benchmark trajectories.

The chain is deliberately small and dependency free so that the runner and the
independent evaluator can use the same byte-level rules without trusting one
another.  A chain proves internal ordering.  Persisting ``head_hash`` and
``event_count`` in the trusted run manifest additionally commits the run against
an attacker who tries to rewrite and re-hash the whole file.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence


EVENT_SCHEMA_VERSION = "1.0"
HASH_ALGORITHM = "sha256"
GENESIS_HASH = "0" * 64
DEFAULT_PROTOCOL = {"name": "adaptive_torsion_sampling_tool_loop", "version": "2.0"}
TERMINAL_EVENTS = frozenset({"run_complete", "run_failure"})

_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_HASH_TEXT = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_KEYS = frozenset(
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


class TrajectoryError(ValueError):
    """The trajectory is malformed or violates its append-only lifecycle."""


def _canonical_value(value: Any) -> Any:
    """Return a strict JSON value with deterministic numeric normalization."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrajectoryError("trajectory contains a non-finite number")
        # This removes a Python-specific 1 versus 1.0 hash ambiguity.
        return int(value) if value.is_integer() else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TrajectoryError("trajectory object keys must be strings")
            if key in normalized:
                raise TrajectoryError("trajectory object contains duplicate keys")
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TrajectoryError(f"trajectory contains non-JSON type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Encode a value as the benchmark's canonical JSON representation."""

    try:
        return json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        if isinstance(exc, TrajectoryError):
            raise
        raise TrajectoryError("trajectory is not canonical JSON data") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _normalize_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise TrajectoryError("trajectory timestamp must be timezone-aware UTC")
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TrajectoryError("trajectory timestamp is invalid") from exc
    else:
        raise TrajectoryError("trajectory timestamp is invalid")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise TrajectoryError("trajectory timestamp must be UTC")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _normalize_protocol(value: Mapping[str, Any] | str) -> dict[str, str]:
    if isinstance(value, str):
        name, separator, version = value.rpartition("/")
        if not separator or not name or not version:
            raise TrajectoryError("trajectory protocol must be name/version")
        normalized = {"name": name, "version": version}
    elif isinstance(value, Mapping) and set(value) == {"name", "version"}:
        normalized = {"name": str(value["name"]), "version": str(value["version"])}
    else:
        raise TrajectoryError("trajectory protocol is invalid")
    if not normalized["name"] or not normalized["version"]:
        raise TrajectoryError("trajectory protocol is invalid")
    return normalized


def _without_event_hash(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key != "event_hash"
    }


def trajectory_commitment(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the count/head pair that a trusted run manifest must persist."""

    if not events:
        raise TrajectoryError("cannot commit an empty trajectory")
    head = events[-1].get("event_hash") if isinstance(events[-1], Mapping) else None
    if not isinstance(head, str) or not _HASH_TEXT.fullmatch(head):
        raise TrajectoryError("trajectory head hash is invalid")
    return {
        "algorithm": HASH_ALGORITHM,
        "event_count": len(events),
        "head_hash": head,
    }


def validate_hash_chain(
    events: Sequence[Mapping[str, Any]],
    *,
    genesis_hash: str = GENESIS_HASH,
    require_terminal: bool = True,
    expected_head_hash: str | None = None,
    expected_event_count: int | None = None,
) -> None:
    """Validate shape, order, hashes, lifecycle, and optional external commitment."""

    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence) or not events:
        raise TrajectoryError("trajectory must contain events")
    if not isinstance(genesis_hash, str) or not _HASH_TEXT.fullmatch(genesis_hash):
        raise TrajectoryError("trajectory genesis hash is invalid")
    if expected_head_hash is not None and (
        not isinstance(expected_head_hash, str)
        or not _HASH_TEXT.fullmatch(expected_head_hash)
    ):
        raise TrajectoryError("expected trajectory head hash is invalid")
    if expected_event_count is not None and (
        isinstance(expected_event_count, bool)
        or not isinstance(expected_event_count, int)
        or expected_event_count < 1
    ):
        raise TrajectoryError("expected trajectory event count is invalid")
    if expected_event_count is not None and len(events) != expected_event_count:
        raise TrajectoryError("trajectory event count does not match commitment")

    previous_hash = genesis_hash
    previous_timestamp: datetime | None = None
    run_id: str | None = None
    protocol: dict[str, str] | None = None
    terminal_index: int | None = None
    final_index: int | None = None

    for sequence, raw in enumerate(events, start=1):
        if not isinstance(raw, Mapping) or set(raw) != _EXPECTED_KEYS:
            raise TrajectoryError("trajectory event envelope is invalid")
        event = dict(raw)
        if event["schema_version"] != EVENT_SCHEMA_VERSION:
            raise TrajectoryError("trajectory event schema version is invalid")
        if isinstance(event["sequence"], bool) or event["sequence"] != sequence:
            raise TrajectoryError("trajectory sequence is not contiguous")
        name = event["event"]
        if not isinstance(name, str) or not _EVENT_NAME.fullmatch(name):
            raise TrajectoryError("trajectory event name is invalid")
        if not isinstance(event["payload"], Mapping):
            raise TrajectoryError("trajectory event payload must be an object")

        timestamp_text = _normalize_timestamp(event["timestamp_utc"])
        if timestamp_text != event["timestamp_utc"]:
            raise TrajectoryError("trajectory timestamp is not canonical")
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise TrajectoryError("trajectory timestamps are not monotonic")
        previous_timestamp = timestamp

        current_run_id = event["run_id"]
        if not isinstance(current_run_id, str) or not current_run_id or len(current_run_id) > 256:
            raise TrajectoryError("trajectory run_id is invalid")
        if run_id is None:
            run_id = current_run_id
        elif current_run_id != run_id:
            raise TrajectoryError("trajectory run_id changed")

        current_protocol = _normalize_protocol(event["protocol"])
        if current_protocol != event["protocol"]:
            raise TrajectoryError("trajectory protocol is not canonical")
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise TrajectoryError("trajectory protocol changed")

        if event["prev_event_hash"] != previous_hash:
            raise TrajectoryError("trajectory previous hash is invalid")
        event_hash = event["event_hash"]
        if not isinstance(event_hash, str) or not _HASH_TEXT.fullmatch(event_hash):
            raise TrajectoryError("trajectory event hash is invalid")
        if event_hash != canonical_sha256(_without_event_hash(event)):
            raise TrajectoryError("trajectory event hash does not match its content")
        previous_hash = event_hash

        if sequence == 1 and name != "run_start":
            raise TrajectoryError("trajectory must start with run_start")
        if sequence > 1 and name == "run_start":
            raise TrajectoryError("trajectory contains more than one run_start")
        if terminal_index is not None:
            raise TrajectoryError("trajectory contains events after its terminal event")
        if name in TERMINAL_EVENTS:
            terminal_index = sequence
        if name == "final_submitted":
            if final_index is not None:
                raise TrajectoryError("trajectory contains more than one final submission")
            final_index = sequence
        elif final_index is not None and name != "run_complete":
            raise TrajectoryError("final submission freezes the trajectory")
        if name == "run_complete" and final_index is None:
            raise TrajectoryError("successful trajectory lacks final_submitted")

    if require_terminal and terminal_index != len(events):
        raise TrajectoryError("trajectory lacks exactly one final terminal event")
    if final_index is not None and terminal_index != final_index + 1:
        raise TrajectoryError("final_submitted must be followed immediately by run_complete")
    if expected_head_hash is not None and previous_hash != expected_head_hash:
        raise TrajectoryError("trajectory head hash does not match commitment")


def verify_hash_chain(
    events: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> bool:
    try:
        validate_hash_chain(events, **kwargs)
    except TrajectoryError:
        return False
    return True


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TrajectoryError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def load_jsonl(
    path: str | Path,
    *,
    require_canonical: bool = True,
    validate: bool = True,
    **validation_kwargs: Any,
) -> list[dict[str, Any]]:
    """Load JSONL without accepting duplicate keys, blank lines, or noncanonical bytes."""

    source = Path(path)
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrajectoryError("trajectory JSONL is unreadable UTF-8") from exc
    if text.startswith("\ufeff"):
        raise TrajectoryError("trajectory JSONL must not contain a BOM")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise TrajectoryError("trajectory JSONL contains an empty line")
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except (json.JSONDecodeError, TrajectoryError) as exc:
            raise TrajectoryError("trajectory JSONL contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise TrajectoryError("trajectory JSONL line must contain an object")
        if require_canonical and line != canonical_json(value):
            raise TrajectoryError("trajectory JSONL line is not canonical JSON")
        events.append(value)
    if validate:
        validate_hash_chain(events, **validation_kwargs)
    return events


class TrajectoryRecorder:
    """In-memory append-only recorder with an exclusive-create persistence step."""

    def __init__(
        self,
        *,
        run_id: str,
        protocol: Mapping[str, Any] | str = DEFAULT_PROTOCOL,
        genesis_hash: str = GENESIS_HASH,
        timestamp_fn: Callable[[], Any] = _utc_now,
    ) -> None:
        if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
            raise TrajectoryError("trajectory run_id is invalid")
        if any(character in run_id for character in ("/", "\\", ":", "\x00")):
            raise TrajectoryError("trajectory run_id contains path syntax")
        if not isinstance(genesis_hash, str) or not _HASH_TEXT.fullmatch(genesis_hash):
            raise TrajectoryError("trajectory genesis hash is invalid")
        self.run_id = run_id
        self.protocol = _normalize_protocol(protocol)
        self.genesis_hash = genesis_hash
        self._timestamp_fn = timestamp_fn
        self._events: list[dict[str, Any]] = []
        self._terminal = False
        self._final_submitted = False

    @property
    def events(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events)

    @property
    def head_hash(self) -> str:
        return self._events[-1]["event_hash"] if self._events else self.genesis_hash

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def commitment(self) -> dict[str, Any]:
        return trajectory_commitment(self._events)

    def append(
        self,
        event: str,
        payload: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        if not isinstance(event, str) or not _EVENT_NAME.fullmatch(event):
            raise TrajectoryError("trajectory event name is invalid")
        if self._terminal:
            raise TrajectoryError("trajectory is frozen after its terminal event")
        if not self._events and event != "run_start":
            raise TrajectoryError("first trajectory event must be run_start")
        if self._events and event == "run_start":
            raise TrajectoryError("run_start may be appended only once")
        if self._final_submitted and event != "run_complete":
            raise TrajectoryError("final submission freezes the trajectory")
        if event == "run_complete" and not self._final_submitted:
            raise TrajectoryError("run_complete requires final_submitted")
        if event == "final_submitted" and self._final_submitted:
            raise TrajectoryError("submit_final may succeed only once")
        if event in TERMINAL_EVENTS and not self._events:
            raise TrajectoryError("terminal event cannot be first")

        combined = dict(payload or {})
        duplicates = set(combined) & set(fields)
        if duplicates:
            raise TrajectoryError("duplicate trajectory payload fields")
        combined.update(fields)
        safe_payload = _canonical_value(combined)
        if not isinstance(safe_payload, dict):  # defensive; combined is a dict
            raise TrajectoryError("trajectory payload must be an object")
        envelope: dict[str, Any] = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "protocol": dict(self.protocol),
            "run_id": self.run_id,
            "sequence": len(self._events) + 1,
            "timestamp_utc": _normalize_timestamp(self._timestamp_fn()),
            "event": event,
            "payload": safe_payload,
            "prev_event_hash": self.head_hash,
        }
        envelope["event_hash"] = canonical_sha256(envelope)
        self._events.append(envelope)
        if event == "final_submitted":
            self._final_submitted = True
        if event in TERMINAL_EVENTS:
            self._terminal = True
        return copy.deepcopy(envelope)

    def validate(self, *, require_terminal: bool = True) -> None:
        validate_hash_chain(
            self._events,
            genesis_hash=self.genesis_hash,
            require_terminal=require_terminal,
        )

    def to_jsonl(self, *, require_terminal: bool = True) -> str:
        self.validate(require_terminal=require_terminal)
        return "".join(canonical_json(event) + "\n" for event in self._events)

    def write(self, path: str | Path, *, require_terminal: bool = True) -> None:
        """Persist exactly once; an existing path is never overwritten."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_jsonl(require_terminal=require_terminal)
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


__all__ = [
    "DEFAULT_PROTOCOL",
    "EVENT_SCHEMA_VERSION",
    "GENESIS_HASH",
    "HASH_ALGORITHM",
    "TERMINAL_EVENTS",
    "TrajectoryError",
    "TrajectoryRecorder",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_jsonl",
    "trajectory_commitment",
    "validate_hash_chain",
    "verify_hash_chain",
]

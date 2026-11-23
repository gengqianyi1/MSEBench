"""Canonical, append-only, SHA-256 chained trajectory events."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from .redaction import sanitize_for_persistence
from .schemas import GENESIS_HASH, LifecycleError, ProtocolConfig


EVENT_SCHEMA_VERSION = "1.0"
HASH_ALGORITHM = "sha256"
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
_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_HASH_TEXT = re.compile(r"^[0-9a-f]{64}$")


def _canonical_value(value: Any) -> Any:
    """Normalize JSON numbers exactly as the independent evaluator does."""

    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise LifecycleError("event contains a non-finite number")
        return int(value) if value.is_integer() else value
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Canonical benchmark JSON: sorted, compact, UTF-8 compatible, no NaN."""

    try:
        return json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LifecycleError("event is not canonical JSON data") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LifecycleError("timestamp source returned an invalid value") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise LifecycleError("timestamp source must return a UTC value")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    raise LifecycleError("timestamp source returned an invalid value")


def _event_without_hash(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in event.items() if key != "event_hash"}


def validate_hash_chain(
    events: Sequence[Mapping[str, Any]],
    *,
    genesis_hash: str = GENESIS_HASH,
    require_terminal: bool = True,
) -> None:
    """Validate envelope shape, chain hashes, continuity, and lifecycle."""

    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)) or not events:
        raise LifecycleError("trajectory must contain events")
    if not isinstance(genesis_hash, str) or not _HASH_TEXT.fullmatch(genesis_hash):
        raise LifecycleError("genesis hash is invalid")
    expected_keys = {
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
    previous = genesis_hash
    run_id: str | None = None
    protocol: Mapping[str, Any] | None = None
    terminal_index: int | None = None
    final_index: int | None = None
    previous_timestamp: datetime | None = None
    for index, raw in enumerate(events, start=1):
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise LifecycleError("trajectory event envelope is invalid")
        event = dict(raw)
        if event["schema_version"] != EVENT_SCHEMA_VERSION:
            raise LifecycleError("trajectory schema version is invalid")
        if event["sequence"] != index:
            raise LifecycleError("trajectory sequence is not contiguous")
        if (
            not isinstance(event["event"], str)
            or not _EVENT_NAME.fullmatch(event["event"])
            or event["event"] not in ALLOWED_EVENTS
        ):
            raise LifecycleError("trajectory event name is invalid")
        if not isinstance(event["payload"], Mapping):
            raise LifecycleError("trajectory event payload must be an object")
        try:
            timestamp = datetime.fromisoformat(str(event["timestamp_utc"]).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise LifecycleError("trajectory timestamp is invalid") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise LifecycleError("trajectory timestamp is not UTC")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise LifecycleError("trajectory timestamps are not monotonic")
        previous_timestamp = timestamp
        if not isinstance(event["run_id"], str) or not event["run_id"]:
            raise LifecycleError("trajectory run_id is invalid")
        if run_id is None:
            run_id = event["run_id"]
        elif event["run_id"] != run_id:
            raise LifecycleError("trajectory run_id changed")
        if not isinstance(event["protocol"], Mapping) or set(event["protocol"]) != {"name", "version"}:
            raise LifecycleError("trajectory protocol is invalid")
        if protocol is None:
            protocol = dict(event["protocol"])
        elif event["protocol"] != protocol:
            raise LifecycleError("trajectory protocol changed")
        if event["prev_event_hash"] != previous:
            raise LifecycleError("trajectory previous hash is invalid")
        expected_hash = canonical_sha256(_event_without_hash(event))
        if event["event_hash"] != expected_hash:
            raise LifecycleError("trajectory event hash is invalid")
        previous = event["event_hash"]

        name = event["event"]
        if index == 1 and name != "run_start":
            raise LifecycleError("trajectory must start with run_start")
        if index > 1 and name == "run_start":
            raise LifecycleError("trajectory contains more than one run_start")
        if terminal_index is not None:
            raise LifecycleError("trajectory contains events after its terminal event")
        if name in TERMINAL_EVENTS:
            terminal_index = index
        if name == "final_submitted":
            if final_index is not None:
                raise LifecycleError("trajectory contains more than one final submission")
            final_index = index
        elif final_index is not None and name != "run_complete":
            raise LifecycleError("submission freeze permits only run_complete")
        if name == "run_complete" and final_index is None:
            raise LifecycleError("successful lifecycle has no final submission")

    if require_terminal:
        if terminal_index is None or terminal_index != len(events):
            raise LifecycleError("trajectory lacks one final terminal event")
    if final_index is not None and terminal_index != final_index + 1:
        raise LifecycleError("final submission must be followed immediately by run_complete")


def verify_hash_chain(
    events: Sequence[Mapping[str, Any]],
    *,
    genesis_hash: str = GENESIS_HASH,
    require_terminal: bool = True,
) -> bool:
    try:
        validate_hash_chain(events, genesis_hash=genesis_hash, require_terminal=require_terminal)
    except LifecycleError:
        return False
    return True


class TrajectoryRecorder:
    """Append-only event recorder enforcing the formal run lifecycle."""

    def __init__(
        self,
        *,
        protocol: ProtocolConfig | Mapping[str, Any],
        run_id: str,
        genesis_hash: str = GENESIS_HASH,
        schema_version: str = EVENT_SCHEMA_VERSION,
        timestamp_fn: Callable[[], Any] = _utc_now,
        secrets: Sequence[str] = (),
    ) -> None:
        if isinstance(protocol, ProtocolConfig):
            protocol_value = protocol.to_dict()
        elif isinstance(protocol, Mapping) and set(protocol) == {"name", "version"}:
            protocol_value = {"name": str(protocol["name"]), "version": str(protocol["version"])}
        else:
            raise LifecycleError("trajectory protocol is invalid")
        if schema_version != EVENT_SCHEMA_VERSION:
            raise LifecycleError("unsupported trajectory event schema")
        if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
            raise LifecycleError("run_id is invalid")
        if not isinstance(genesis_hash, str) or not _HASH_TEXT.fullmatch(genesis_hash):
            raise LifecycleError("genesis hash is invalid")
        self.protocol = protocol_value
        self.run_id = run_id
        self.genesis_hash = genesis_hash.lower()
        self.schema_version = schema_version
        self._timestamp_fn = timestamp_fn
        self._secrets = tuple(secret for secret in secrets if isinstance(secret, str) and secret)
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
    def terminal(self) -> bool:
        return self._terminal

    @property
    def final_submitted(self) -> bool:
        return self._final_submitted

    def register_secret(self, secret: str) -> None:
        """Add a runtime-only secret to the redaction set.

        Credentials are intentionally registered only after environment loading and
        are never represented by an event or by any public recorder property.
        """

        if self._terminal:
            raise LifecycleError("cannot register runtime data after termination")
        if not isinstance(secret, str) or not secret:
            raise LifecycleError("runtime secret must be a non-empty string")
        if secret not in self._secrets:
            self._secrets = (*self._secrets, secret)

    def append(self, event: str, payload: Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
        if (
            not isinstance(event, str)
            or not _EVENT_NAME.fullmatch(event)
            or event not in ALLOWED_EVENTS
        ):
            raise LifecycleError("event name is invalid")
        if self._terminal:
            raise LifecycleError("trajectory is frozen after its terminal event")
        if not self._events and event != "run_start":
            raise LifecycleError("first trajectory event must be run_start")
        if self._events and event == "run_start":
            raise LifecycleError("run_start may be appended only once")
        if self._final_submitted and event != "run_complete":
            raise LifecycleError("submission freeze permits only run_complete")
        if event == "run_complete" and not self._final_submitted:
            raise LifecycleError("run_complete requires final_submitted")
        if event == "final_submitted" and self._final_submitted:
            raise LifecycleError("submit_final may succeed only once")
        if event in TERMINAL_EVENTS and not self._events:
            raise LifecycleError("terminal event cannot be first")

        combined = dict(payload or {})
        if fields:
            duplicate = set(combined) & set(fields)
            if duplicate:
                raise LifecycleError("duplicate event payload fields")
            combined.update(fields)
        safe_payload = sanitize_for_persistence(combined, secrets=self._secrets)
        if not isinstance(safe_payload, dict):
            raise LifecycleError("event payload must sanitize to an object")
        envelope: dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol": dict(self.protocol),
            "run_id": self.run_id,
            "sequence": len(self._events) + 1,
            "timestamp_utc": _timestamp(self._timestamp_fn()),
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

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_jsonl(require_terminal=True)
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def render_dialogue(events: Sequence[Mapping[str, Any]]) -> str:
    """Generate a human view solely from already-sanitized trajectory events."""

    lines = ["# Agent dialogue", "", "Generated from the canonical hash-chained trajectory.", ""]
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        name = raw.get("event")
        payload = raw.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if name == "assistant_message":
            message = payload.get("message")
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "unknown")).upper()
            source = str(payload.get("source", "unspecified"))
            lines.extend([f"## {role} ({source})", ""])
            content = message.get("content")
            if isinstance(content, str) and content:
                lines.extend([content, ""])
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                lines.extend(["```json", json.dumps(tool_calls, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
        elif name in {"tool_call", "tool_result", "attempt_prepared", "native_xtb_execution", "attempt_validated"}:
            lines.extend(
                [
                    f"## {str(name).replace('_', ' ').title()}",
                    "",
                    "```json",
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        elif name == "final_submitted":
            lines.extend(
                [
                    "## Final submitted",
                    "",
                    "```json",
                    json.dumps(payload.get("final_answer", {}), ensure_ascii=False, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ALLOWED_EVENTS",
    "EVENT_SCHEMA_VERSION",
    "HASH_ALGORITHM",
    "TERMINAL_EVENTS",
    "TrajectoryRecorder",
    "canonical_json",
    "canonical_sha256",
    "render_dialogue",
    "validate_hash_chain",
    "verify_hash_chain",
]

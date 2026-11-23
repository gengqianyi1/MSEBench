"""Canonical-state v2 runner with terminal batch finalization."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Mapping, Sequence

from framework.canonical_state import (
    build_canonical_state,
    canonical_state_text,
)
from framework.execution_freeze import (
    load_execution_freeze_binding,
    validate_execution_freeze_binding,
)
from framework.provider_retry import (
    DEFAULT_PROVIDER_RETRY_POLICY,
    KNOWN_PROVIDER_CATEGORIES,
    ProviderRetryPolicy,
    ProviderTurnRetryBinding,
    TRANSIENT_PROVIDER_CATEGORIES,
)
from framework.redaction import contains_key_like, contains_key_like_recursive
from framework.schemas import (
    FORMAL_MODELS,
    MissingCredentialError,
    ModelConfig,
    SchemaError,
    validate_scan_batch_action,
)
from framework.siliconflow_adapter import (
    PROVIDER_ERROR_CLASSIFICATION_VERSION,
    ProviderCallError,
    SiliconFlowChatTransport,
    classify_provider_exception,
    load_runtime_credential,
    sanitize_provider_message,
)
from framework.trajectory import TrajectoryRecorder
from framework.trusted_final import (
    BENCHMARK_VERSION,
    CASE_ID,
    PROTOCOL,
    TASK_ID,
    build_final_envelope,
    write_final_envelope,
)
from tools.science import canonical_json_bytes, sha256_file
from tools.tool_registry import ToolSession


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "You are solving a frozen torsion-profile benchmark. Return exactly one "
    "JSON action matching the supplied closed schema. Select scan angles using "
    "only the canonical state and latest compact analysis. Never request paths, "
    "shell, Python, network, oracle, evaluator, source, credentials, or manual "
    "profile injection. Phase 2 or recovery must finalize terminally."
)
FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*\Z", re.I)
SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}")
REDACTED_MODEL_ID = "REDACTED_INVALID_MODEL_ID"


class RunnerError(RuntimeError):
    """Formal runner policy was violated."""


class ExactModelIdentityError(RunnerError):
    """Provider returned a different model identity."""


class ProviderOutputProtocolError(RunnerError):
    """A provider response had identity but no usable message content."""

    def __init__(
        self,
        *,
        actual_model_id: str,
        usage: Mapping[str, int] | None,
        reason: str,
    ) -> None:
        self.actual_model_id = actual_model_id
        self.usage = dict(usage) if usage is not None else None
        super().__init__(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_parts(response: Any) -> tuple[str, str, dict[str, int] | None]:
    raw_actual = _object_value(response, "model")
    actual = raw_actual
    if (
        not isinstance(actual, str)
        or not actual
        or not SAFE_MODEL_ID.fullmatch(actual)
        or contains_key_like(actual)
    ):
        actual = REDACTED_MODEL_ID
    usage_raw = _object_value(response, "usage")
    usage: dict[str, int] | None = None
    if usage_raw is not None:
        prompt = _object_value(usage_raw, "prompt_tokens")
        completion = _object_value(usage_raw, "completion_tokens")
        total = _object_value(usage_raw, "total_tokens")
        if all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in (prompt, completion, total)
        ) and total == prompt + completion:
            usage = {
                "input_tokens": prompt,
                "output_tokens": completion,
                "total_tokens": total,
            }
    # Missing or unsafe model identity is always an identity failure.  Do not
    # inspect or persist any other provider-controlled field before the safe
    # sentinel has been bound to the trajectory.
    if actual == REDACTED_MODEL_ID:
        return actual, "", usage
    choices = _object_value(response, "choices")
    if (
        isinstance(choices, (str, bytes))
        or not isinstance(choices, Sequence)
        or not choices
    ):
        raise ProviderOutputProtocolError(
            actual_model_id=actual,
            usage=usage,
            reason="provider response lacks choices",
        )
    message = _object_value(choices[0], "message")
    if message is None:
        raise ProviderOutputProtocolError(
            actual_model_id=actual,
            usage=usage,
            reason="provider response lacks a message",
        )
    content = _object_value(message, "content")
    if not isinstance(content, str):
        raise ProviderOutputProtocolError(
            actual_model_id=actual,
            usage=usage,
            reason="provider response lacks text content",
        )
    return actual, content, usage


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_single_json_action(content: Any) -> dict[str, Any]:
    """Apply only whitespace/code-fence normalization, never semantic repair."""

    if not isinstance(content, str) or contains_key_like(content):
        raise RunnerError("provider content is absent or key-like")
    text = content.strip()
    match = FENCE.fullmatch(text)
    if match:
        text = match.group(1).strip()
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RunnerError("provider output is not one strict JSON action") from exc
    if not isinstance(value, dict):
        raise RunnerError("provider action must be a JSON object")
    if contains_key_like_recursive(value):
        raise RunnerError("decoded provider action contains key-like content")
    return value


def _recordable_action(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a provider action before any provider-controlled value is logged."""

    action = dict(value)
    discriminator = action.get("action")
    if discriminator == "run_scan_batch":
        try:
            validated = validate_scan_batch_action(action)
        except SchemaError as exc:
            raise RunnerError(
                "provider action does not match the closed schema"
            ) from exc
        if validated.get("rationale") != action.get("rationale"):
            raise RunnerError(
                "provider action would require semantic normalization"
            )
        action = validated
    elif discriminator == "inspect_failed_attempt":
        if (
            set(action) != {"action", "angle_deg"}
            or isinstance(action.get("angle_deg"), bool)
            or not isinstance(action.get("angle_deg"), int)
            or action["angle_deg"] < 0
            or action["angle_deg"] >= 360
            or action["angle_deg"] % 5
        ):
            raise RunnerError("inspection action is not recordable")
    else:
        raise RunnerError("provider action is not allowlisted")
    if contains_key_like_recursive(action):
        raise RunnerError("provider action contains key-like content")
    return action


def _project_provider_error(safe: Any) -> dict[str, Any]:
    """Re-whitelist every provider-controlled error before persistence."""

    value = dict(safe) if isinstance(safe, Mapping) else {}
    category = value.get("status", value.get("error_category"))
    if category not in KNOWN_PROVIDER_CATEGORIES:
        category = "UNKNOWN_PROVIDER_ERROR"
    raw_status = value.get("http_status", value.get("status_code"))
    http_status = (
        raw_status
        if isinstance(raw_status, int)
        and not isinstance(raw_status, bool)
        and 100 <= raw_status <= 599
        else None
    )
    raw_code = value.get("provider_error_code")
    provider_code = (
        raw_code.strip()
        if isinstance(raw_code, str)
        and re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,120}", raw_code.strip())
        and not contains_key_like(raw_code)
        else None
    )
    raw_type = value.get("exception_type")
    exception_type = (
        raw_type
        if isinstance(raw_type, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", raw_type)
        else "ProviderException"
    )
    return {
        "status": category,
        "error_category": category,
        "http_status": http_status,
        "status_code": http_status,
        "provider_error_code": provider_code,
        "sanitized_provider_message": sanitize_provider_message(
            value.get("sanitized_provider_message")
        ),
        "exception_type": exception_type,
        "classification_version": PROVIDER_ERROR_CLASSIFICATION_VERSION,
        "retryable": category in TRANSIENT_PROVIDER_CATEGORIES,
        "message_reflected": False,
        "raw_response_persisted": False,
        "response_headers_persisted": False,
        "request_persisted": False,
    }


def _prepare_run_root(path: Path) -> Path:
    root = path.resolve()
    project = ROOT.resolve()
    if root != project and project not in root.parents:
        raise RunnerError("run destination must remain inside the project")
    root.mkdir(parents=True, exist_ok=True)
    existing_files = [item for item in root.rglob("*") if item.is_file() or item.is_symlink()]
    if existing_files:
        raise RunnerError("formal run destination is not empty")
    return root


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))


class RunContext:
    """Shared live context used by the formal and deterministic baseline agents."""

    def __init__(
        self,
        *,
        run_root: Path,
        run_id: str,
        requested_model_id: str,
        mode: str,
        raw_provenance_root: Path | None = None,
        native_runner: Callable[..., dict[str, Any]] | None = None,
        execution_freeze_binding: Mapping[str, Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_root = _prepare_run_root(run_root)
        self.run_id = run_id
        self.requested_model_id = requested_model_id
        self.actual_model_id: str | None = None
        self.mode = mode
        self.execution_freeze_binding = (
            validate_execution_freeze_binding(execution_freeze_binding)
            if execution_freeze_binding is not None
            else None
        )
        self.recorder = TrajectoryRecorder(run_id=run_id)
        session_options: dict[str, Any] = {
            "run_id": run_id,
            "raw_provenance_root": raw_provenance_root,
            "monotonic": monotonic,
        }
        if native_runner is not None:
            session_options["native_runner"] = native_runner
        self.session = ToolSession(self.run_root, **session_options)
        self._started = monotonic()
        self._monotonic = monotonic
        self.provider_turns = 0
        self.successful_provider_turns = 0
        self.retried_attempts = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.provider_usage_unavailable = False
        self.turn_usage: list[dict[str, Any]] = []
        self.recorder.append(
            "run_start",
            {
                "task_id": TASK_ID,
                "benchmark_version": BENCHMARK_VERSION,
                "case_id": CASE_ID,
                "protocol": PROTOCOL,
                "mode": mode,
                "requested_model_id": requested_model_id,
                "fallback_allowed": False,
                "context_mode": "canonical_state_v2",
                "execution_freeze_binding": deepcopy(
                    self.execution_freeze_binding
                ),
                "raw_provider_response_persisted": False,
            },
        )

    @classmethod
    def create(
        cls,
        destination: Path,
        *,
        run_id: str,
        requested_model_id: str,
        mode: str,
        **kwargs: Any,
    ) -> "RunContext":
        return cls(
            run_root=destination,
            run_id=run_id,
            requested_model_id=requested_model_id,
            mode=mode,
            **kwargs,
        )

    def record_provider_turn(
        self,
        *,
        turn: int,
        actual_model_id: str,
        content: str,
        canonical_state_sha256: str,
        canonical_state_chars: int,
        usage: Mapping[str, int] | None,
    ) -> None:
        matching = [
            item for item in self.turn_usage if item.get("turn") == turn
        ]
        if (
            len(matching) != 1
            or matching[0].get("canonical_state_sha256")
            != canonical_state_sha256
            or matching[0].get("canonical_state_chars")
            != canonical_state_chars
            or matching[0].get("output_chars") is not None
        ):
            raise RunnerError("provider turn lacks its logical usage skeleton")
        exact_identity = actual_model_id == self.requested_model_id
        prior_identity_consistent = self.actual_model_id in {
            None,
            actual_model_id,
        }
        self.actual_model_id = actual_model_id
        self.successful_provider_turns += 1
        if usage is None:
            self.provider_usage_unavailable = True
        else:
            self.input_tokens += int(usage["input_tokens"])
            self.output_tokens += int(usage["output_tokens"])
            self.total_tokens += int(usage["total_tokens"])
        item = matching[0]
        item["output_chars"] = len(content)
        item["usage"] = dict(usage) if usage is not None else None
        self.recorder.append(
            (
                "provider_response_accepted"
                if exact_identity and prior_identity_consistent
                else "provider_response_rejected_identity"
            ),
            {
                "turn": turn,
                "actual_model_id": actual_model_id,
                "exact_identity": exact_identity and prior_identity_consistent,
                "content_chars": len(content),
                "content_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest().upper(),
                "raw_content_persisted": False,
                "canonical_state_sha256": canonical_state_sha256,
                "usage": dict(usage) if usage is not None else None,
            },
        )
        if not exact_identity or not prior_identity_consistent:
            raise ExactModelIdentityError("IDENTITY_MISMATCH")

    def begin_provider_turn(
        self,
        *,
        turn: int,
        canonical_state_sha256: str,
        canonical_state_chars: int,
    ) -> None:
        """Create one immutable logical-turn accounting skeleton pre-request."""

        if (
            isinstance(turn, bool)
            or not isinstance(turn, int)
            or turn < 1
            or any(item.get("turn") == turn for item in self.turn_usage)
            or not re.fullmatch(r"[0-9A-F]{64}", canonical_state_sha256)
            or isinstance(canonical_state_chars, bool)
            or not isinstance(canonical_state_chars, int)
            or canonical_state_chars < 1
        ):
            raise RunnerError("provider logical turn skeleton is invalid")
        self.turn_usage.append(
            {
                "turn": turn,
                "canonical_state_chars": canonical_state_chars,
                "canonical_state_sha256": canonical_state_sha256,
                "output_chars": None,
                "usage": None,
                "tool_result_chars": None,
            }
        )

    def mark_provider_usage_unavailable(self) -> None:
        self.provider_usage_unavailable = True

    def call_action(
        self,
        *,
        turn: int,
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        safe_action = _recordable_action(action)
        self.recorder.append(
            "public_action",
            {
                "turn": turn,
                "action": deepcopy(safe_action),
            },
        )
        result = self.session.dispatch(safe_action, assistant_turn=turn)
        for usage_item in reversed(self.turn_usage):
            if usage_item.get("turn") == turn:
                usage_item["tool_result_chars"] = self.session.last_tool_result_chars
                break
        self.recorder.append(
            "public_action_result",
            {
                "turn": turn,
                "action": safe_action.get("action"),
                "result": result,
                "tool_result_chars": self.session.last_tool_result_chars,
            },
        )
        return result

    # Compatibility methods for old test helpers. They do not re-expose old tools.
    def record_assistant(
        self,
        *,
        turn: int,
        actual_model_id: str,
        content: str,
        tool_count: int,
    ) -> None:
        del tool_count
        self.actual_model_id = actual_model_id
        self.recorder.append(
            "baseline_decision",
            {
                "turn": turn,
                "actual_model_id": actual_model_id,
                "decision_summary": content[:400],
            },
        )

    def call_tool(
        self,
        *,
        turn: int,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        format_error: str | None = None,
    ) -> dict[str, Any]:
        del call_id, format_error
        payload = dict(arguments)
        payload.setdefault("action", name)
        return self.call_action(turn=turn, action=payload)

    def _usage_report(self) -> dict[str, Any]:
        state = self.session.sampling_state()
        usage_unavailable = bool(
            self.provider_usage_unavailable
            or (
                self.mode == "REAL_API_LIVE_XTB"
                and self.provider_turns == 0
            )
        )
        return {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "mode": self.mode,
            "provider_turns": self.provider_turns,
            "successful_provider_turns": self.successful_provider_turns,
            "retried_attempts": self.retried_attempts,
            "input_tokens": (
                None if usage_unavailable else self.input_tokens
            ),
            "output_tokens": (
                None if usage_unavailable else self.output_tokens
            ),
            "total_tokens": (
                None if usage_unavailable else self.total_tokens
            ),
            "provider_usage_unavailable": usage_unavailable,
            "turns": deepcopy(self.turn_usage),
            "native_xtb_attempts": self.session.budget_snapshot()["native_attempts"],
            "successful_unique_angles": state["valid_unique_angle_count"],
            "failed_unique_angles": len(state["failed_angles"]),
            "wall_clock_seconds": round(self._monotonic() - self._started, 6),
        }

    def _run_manifest(self, *, status: str, blocker: str | None = None) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(
            self.run_root.rglob("*"),
            key=lambda item: item.relative_to(self.run_root).as_posix(),
        ):
            if path.is_file() and path.name != "run_manifest.json":
                files.append(
                    {
                        "path": path.relative_to(self.run_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        return {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "status": status,
            "blocker": blocker,
            "requested_model_id": self.requested_model_id,
            "actual_model_id": self.actual_model_id,
            "exact_identity": self.actual_model_id == self.requested_model_id,
            "fallback_used": False,
            "model_override_used": False,
            "task_id": TASK_ID,
            "benchmark_version": BENCHMARK_VERSION,
            "case_id": CASE_ID,
            "protocol": PROTOCOL,
            "execution_freeze_binding": deepcopy(
                self.execution_freeze_binding
            ),
            "trajectory_commitment": self.recorder.commitment,
            "budget_at_end": self.session.budget_snapshot(),
            "files": files,
        }

    def finalize_success(self, *, turn: int) -> dict[str, Any]:
        if self.session.final_submission is None:
            raise RunnerError("terminal finalization was not accepted")
        if self.actual_model_id is None:
            # Deterministic baseline has no provider response.
            self.actual_model_id = self.requested_model_id
        state = self.session.sampling_state()
        profile_id = state["profile_ids"][-1]
        profile = self.session.profile(profile_id)
        self.recorder.append(
            "final_submitted",
            {
                "turn": turn,
                "model_submission": self.session.final_submission,
                "profile_id": profile_id,
                "profile_hash": profile["profile_hash"],
                "terminal_finalization": True,
                "extra_provider_turn": False,
            },
        )
        self.recorder.append(
            "run_complete",
            {
                "status": "PASS",
                "exact_model_identity": True,
                "profile_id": profile_id,
            },
        )
        trajectory_path = self.run_root / "trajectory.jsonl"
        self.recorder.write(trajectory_path)
        envelope = build_final_envelope(
            session=self.session,
            requested_model_id=self.requested_model_id,
            actual_model_id=self.actual_model_id,
            trajectory_path=trajectory_path,
            trajectory_final_event_hash=self.recorder.head_hash,
            execution_freeze_binding=self.execution_freeze_binding,
        )
        write_final_envelope(self.run_root / "final_answer.json", envelope)
        _write_exclusive(
            self.run_root / "execution_commitment.json",
            {
                "schema_version": "2.0",
                "algorithm": "SHA-256",
                "execution_commitment": envelope["trusted_runtime"][
                    "execution_commitment"
                ],
                "payload": envelope["trusted_runtime"][
                    "execution_commitment_payload"
                ],
            },
        )
        _write_exclusive(self.run_root / "usage_report.json", self._usage_report())
        _write_exclusive(
            self.run_root / "run_manifest.json",
            self._run_manifest(status="PASS"),
        )
        return envelope

    def finalize_failure(
        self,
        *,
        safe_status: str,
        run_status: str = "BLOCKED",
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if run_status not in {"FAIL", "BLOCKED"}:
            raise RunnerError("terminal unsuccessful status must be FAIL or BLOCKED")
        if not self.recorder.terminal:
            self.recorder.append(
                "run_failure",
                {
                    "status": run_status,
                    "safe_status": safe_status,
                    "detail": dict(detail or {}),
                    "raw_provider_response_persisted": False,
                },
            )
        trajectory_path = self.run_root / "trajectory.jsonl"
        self.recorder.write(trajectory_path)
        blocked = {
            "model_submission": {},
            "trusted_runtime": {
                "task_id": TASK_ID,
                "benchmark_version": BENCHMARK_VERSION,
                "case_id": CASE_ID,
                "protocol": PROTOCOL,
                "status": run_status,
                "safe_status": safe_status,
                "requested_model_id": self.requested_model_id,
                "actual_model_id": self.actual_model_id,
                "trajectory_hash": self.recorder.head_hash,
                "trajectory_sha256": sha256_file(trajectory_path),
                "execution_freeze_binding": deepcopy(
                    self.execution_freeze_binding
                ),
                "fallback_used": False,
                "model_override_used": False,
                "raw_provider_response_persisted": False,
                "timestamp": _utc_now(),
            },
        }
        _write_exclusive(self.run_root / "final_answer.json", blocked)
        _write_exclusive(self.run_root / "usage_report.json", self._usage_report())
        _write_exclusive(
            self.run_root / "run_manifest.json",
            self._run_manifest(status=run_status, blocker=safe_status),
        )
        return blocked


class FormalAgentRunner:
    def __init__(
        self,
        *,
        label: str,
        run_root: Path,
        transport: Any | None = None,
        retry_policy: ProviderRetryPolicy | None = None,
        retry_sleep: Callable[[float], None] = time.sleep,
        retry_jitter: Callable[[], float] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        native_runner: Callable[..., dict[str, Any]] | None = None,
        raw_provenance_root: Path | None = None,
        execution_freeze_binding: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = ModelConfig.for_label(label)
        self.label = label
        self.run_root = run_root
        self.transport = transport
        self.retry_policy = retry_policy or DEFAULT_PROVIDER_RETRY_POLICY
        self._sleep = retry_sleep
        self._jitter = retry_jitter or random.SystemRandom().random
        self._monotonic = monotonic
        self._native_runner = native_runner
        self._raw_provenance_root = raw_provenance_root
        self._execution_freeze_binding = (
            validate_execution_freeze_binding(execution_freeze_binding)
            if execution_freeze_binding is not None
            else None
        )

    def _complete_turn(
        self,
        *,
        context: RunContext,
        turn: int,
        state_text: str,
        remaining_seconds: float,
    ) -> tuple[str, str, dict[str, int] | None]:
        digest = hashlib.sha256(state_text.encode("utf-8")).hexdigest().upper()
        binding = ProviderTurnRetryBinding(
            requested_model_id=self.config.model,
            run_id=context.run_id,
            canonical_state_sha256=digest,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state_text},
        ]
        cumulative_delay = 0.0
        for attempt in range(1, self.retry_policy.maximum_total_attempts + 1):
            binding.verify_attempt(
                requested_model_id=self.config.model,
                run_id=context.run_id,
                canonical_state_sha256=digest,
            )
            try:
                response = self.transport.complete(
                    messages=messages,
                    tools=[],
                    timeout_seconds=min(
                        remaining_seconds, self.config.timeout_seconds
                    ),
                    response_format={"type": "json_object"},
                )
                return _response_parts(response)
            except ProviderCallError as exc:
                safe = _project_provider_error(exc.safe)
                context.mark_provider_usage_unavailable()
                category = str(safe.get("status", "UNKNOWN_PROVIDER_ERROR"))
                will_retry = self.retry_policy.should_retry(
                    category=category, attempt_number=attempt
                )
                delay = 0.0
                if will_retry:
                    delay = self.retry_policy.plan_delay(
                        attempt_number=attempt,
                        cumulative_delay_seconds=cumulative_delay,
                        jitter_unit=self._jitter(),
                    )
                context.recorder.append(
                    "provider_attempt_failed",
                    {
                        "turn": turn,
                        "attempt_number": attempt,
                        "category": category,
                        "retryable": safe.get("retryable") is True,
                        "will_retry": will_retry,
                        "planned_delay_seconds": delay,
                        "safe_provider_error": safe,
                        "same_model_run_state": True,
                    },
                )
                if not will_retry:
                    raise ProviderCallError(safe) from None
                context.retried_attempts += 1
                cumulative_delay = round(cumulative_delay + delay, 6)
                self._sleep(delay)
            except ProviderOutputProtocolError:
                raise
            except Exception:
                # A request was made, but no trustworthy provider usage could
                # be recovered from an unexpected transport/projection fault.
                context.mark_provider_usage_unavailable()
                raise
        raise RunnerError("provider retry loop exhausted")

    def run(self) -> dict[str, Any]:
        run_id = f"{self.label}-formal-v2"
        execution_freeze_binding = (
            deepcopy(self._execution_freeze_binding)
            if self._execution_freeze_binding is not None
            else load_execution_freeze_binding(ROOT)
        )
        context = RunContext.create(
            self.run_root,
            run_id=run_id,
            requested_model_id=self.config.model,
            mode="REAL_API_LIVE_XTB",
            monotonic=self._monotonic,
            native_runner=self._native_runner,
            raw_provenance_root=self._raw_provenance_root,
            execution_freeze_binding=execution_freeze_binding,
        )
        if self.transport is None:
            try:
                self.transport = SiliconFlowChatTransport(
                    self.config, load_runtime_credential(ROOT)
                )
            except MissingCredentialError:
                return context.finalize_failure(
                    safe_status="AUTHENTICATION_FAILED",
                    detail={
                        "provider_turn": 0,
                        "error_category": "AUTHENTICATION_FAILED",
                    },
                )
            except ProviderCallError as exc:
                safe = _project_provider_error(exc.safe)
                return context.finalize_failure(
                    safe_status=str(safe["status"]),
                    detail={
                        "provider_turn": 0,
                        "error_category": safe["status"],
                    },
                )
            except Exception as exc:
                safe = _project_provider_error(
                    classify_provider_exception(exc)
                )
                return context.finalize_failure(
                    safe_status=str(safe["status"]),
                    detail={
                        "provider_turn": 0,
                        "error_category": safe["status"],
                    },
                )
        latest_result: dict[str, Any] | None = None
        retained_profile_analysis: dict[str, Any] | None = None
        protocol_rerequest_available = True
        protocol_rerequest_pending = False
        started = self._monotonic()
        try:
            for turn in range(1, self.config.max_provider_turns + 1):
                remaining = 1200.0 - (self._monotonic() - started)
                if remaining <= 0:
                    return context.finalize_failure(
                        safe_status="TOTAL_TIMEOUT_EXCEEDED"
                    )
                state = context.session.sampling_state()
                canonical = build_canonical_state(
                    runtime_state=state,
                    latest_tool_result=latest_result,
                    retained_profile_analysis=retained_profile_analysis,
                    # The one permitted format correction must replay the
                    # byte-identical scientific state.  Its existence is
                    # recorded in the trajectory, never injected into the next
                    # provider payload.
                    protocol_rerequest=False,
                )
                protocol_rerequest_pending = False
                state_text = canonical_state_text(canonical)
                state_hash = hashlib.sha256(
                    state_text.encode("utf-8")
                ).hexdigest().upper()
                context.provider_turns += 1
                context.begin_provider_turn(
                    turn=turn,
                    canonical_state_sha256=state_hash,
                    canonical_state_chars=len(state_text),
                )
                try:
                    actual, content, usage = self._complete_turn(
                        context=context,
                        turn=turn,
                        state_text=state_text,
                        remaining_seconds=remaining,
                    )
                except ProviderOutputProtocolError as exc:
                    context.record_provider_turn(
                        turn=turn,
                        actual_model_id=exc.actual_model_id,
                        content="",
                        canonical_state_sha256=state_hash,
                        canonical_state_chars=len(state_text),
                        usage=exc.usage,
                    )
                    context.recorder.append(
                        "protocol_violation",
                        {
                            "turn": turn,
                            "category": "PROTOCOL_VIOLATION",
                            "same_state_rerequest_allowed": (
                                protocol_rerequest_available
                            ),
                            "raw_content_persisted": False,
                        },
                    )
                    if protocol_rerequest_available:
                        protocol_rerequest_available = False
                        protocol_rerequest_pending = True
                        continue
                    return context.finalize_failure(
                        safe_status="PROTOCOL_VIOLATION",
                        run_status="FAIL",
                        detail={"provider_turn": turn},
                    )
                except ProviderCallError as exc:
                    return context.finalize_failure(
                        safe_status=str(exc.safe.get("status", "UNKNOWN_PROVIDER_ERROR")),
                        detail={
                            "provider_turn": turn,
                            "error_category": exc.safe.get("status"),
                        },
                    )
                context.record_provider_turn(
                    turn=turn,
                    actual_model_id=actual,
                    content=content,
                    canonical_state_sha256=state_hash,
                    canonical_state_chars=len(state_text),
                    usage=usage,
                )
                try:
                    action = parse_single_json_action(content)
                except RunnerError:
                    context.recorder.append(
                        "protocol_violation",
                        {
                            "turn": turn,
                            "category": "PROTOCOL_VIOLATION",
                            "same_state_rerequest_allowed": protocol_rerequest_available,
                            "raw_content_persisted": False,
                        },
                    )
                    if protocol_rerequest_available:
                        protocol_rerequest_available = False
                        protocol_rerequest_pending = True
                        continue
                    return context.finalize_failure(
                        safe_status="PROTOCOL_VIOLATION",
                        run_status="FAIL",
                        detail={"provider_turn": turn},
                    )
                try:
                    latest_result = context.call_action(
                        turn=turn, action=action
                    )
                except RunnerError:
                    return context.finalize_failure(
                        safe_status="PROTOCOL_VIOLATION",
                        run_status="FAIL",
                        detail={"provider_turn": turn},
                    )
                latest_profile = latest_result.get("profile")
                if (
                    isinstance(latest_profile, Mapping)
                    and isinstance(latest_profile.get("analysis"), Mapping)
                ):
                    retained_profile_analysis = deepcopy(
                        dict(latest_profile["analysis"])
                    )
                if (
                    action.get("action") == "run_scan_batch"
                    and latest_result.get("terminal", {}).get("accepted") is True
                ):
                    return context.finalize_success(turn=turn)
                if latest_result.get("ok") is not True:
                    action_status = str(
                        latest_result.get("status", "PROTOCOL_VIOLATION")
                    )
                    if action_status == "TOTAL_NATIVE_TIMEOUT_EXCEEDED":
                        # Native execution capacity is external to the model's
                        # sampling behavior.  Preserve a legal terminal
                        # failure envelope, but do not mis-score it as a model
                        # protocol/eligibility FAIL.
                        return context.finalize_failure(
                            safe_status=action_status,
                            run_status="BLOCKED",
                            detail={
                                "provider_turn": turn,
                                "action_status": action_status,
                            },
                        )
                    return context.finalize_failure(
                        safe_status=(
                            action_status
                            if action_status
                            in {
                                "FINAL_ELIGIBILITY_FAILED",
                                "NATIVE_ELIGIBILITY_IMPOSSIBLE",
                                "NATIVE_INVARIANT_BREACH",
                            }
                            else "PROTOCOL_VIOLATION"
                        ),
                        run_status="FAIL",
                        detail={
                            "provider_turn": turn,
                            "action_status": action_status,
                        },
                    )
            return context.finalize_failure(
                safe_status="PROVIDER_TURN_BUDGET_EXHAUSTED",
                run_status="FAIL",
            )
        except ExactModelIdentityError:
            return context.finalize_failure(
                safe_status="IDENTITY_MISMATCH",
                run_status="FAIL",
            )
        except Exception as exc:
            return context.finalize_failure(
                safe_status="RUNNER_POLICY_FAILURE",
                run_status="FAIL",
                detail={"exception_type": type(exc).__name__},
            )


def run_formal_model(label: str, destination: Path) -> dict[str, Any]:
    if label not in FORMAL_MODELS:
        raise RunnerError("unknown formal model label")
    return FormalAgentRunner(label=label, run_root=destination).run()


__all__ = [
    "ExactModelIdentityError",
    "FormalAgentRunner",
    "ProviderOutputProtocolError",
    "REDACTED_MODEL_ID",
    "RunContext",
    "RunnerError",
    "parse_single_json_action",
    "run_formal_model",
]

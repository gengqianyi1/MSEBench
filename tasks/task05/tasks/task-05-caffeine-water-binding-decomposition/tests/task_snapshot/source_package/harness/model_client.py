"""Auditable OpenAI-compatible client with frozen recovery semantics."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .api_policy import (
    RETRIABLE_HTTP_STATUSES,
    extract_error_code,
    is_immediate_failure,
    sanitize_text,
    should_switch_siliconflow_key,
)
from .budget import BudgetLedger


class ApiPolicyError(RuntimeError):
    """A classified provider failure that must not be silently bypassed."""


class RequestsTransport:
    """Small stdlib OpenAI-compatible transport; name kept for API stability."""

    class Response:
        def __init__(self, status_code: int, body: bytes) -> None:
            self.status_code = int(status_code)
            self.text = body.decode("utf-8", errors="replace")

        def json(self) -> dict:
            value = json.loads(self.text)
            if not isinstance(value, dict):
                raise ValueError("provider response JSON is not an object")
            return value

    def post(self, url: str, headers: dict, json: dict, timeout: int):
        body = globals()["json"].dumps(json, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return self.Response(int(response.status), response.read())
        except urllib.error.HTTPError as exc:
            return self.Response(int(exc.code), exc.read())


class RobustModelClient:
    """Apply transport retries, one semantic retry and voucher-only fallback."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        keys: Mapping[str, str],
        ledger: BudgetLedger,
        transport: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = dict(config)
        self.keys = {str(name): str(value) for name, value in keys.items() if value}
        self.ledger = ledger
        self.transport = transport or RequestsTransport()
        self.sleep = sleep
        self.events: list[dict[str, Any]] = []
        self._resumed = False
        provider = self.config["provider"]
        if provider == "siliconflow":
            self.key_slot = "voucher" if self.keys.get("voucher") else "standard"
        else:
            self.key_slot = "primary"
        if not self.keys.get(self.key_slot):
            raise ApiPolicyError(f"missing API key slot: {self.key_slot}")

    def _attempt_class(self, semantic_retry: bool, transport_retry: int, key_fallback: bool) -> str:
        if key_fallback:
            base = "key_fallback"
        elif transport_retry:
            base = "transport_retry"
        elif semantic_retry:
            base = "semantic_retry"
        else:
            base = "initial"
        return f"resumed_{base}" if self._resumed else base

    def _payload(self, messages: list[dict], max_tokens: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config["requested_model_id"],
            "messages": messages,
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
            "seed": self.config["seed"],
            "max_tokens": max_tokens,
        }
        if self.config.get("thinking") is not None:
            payload["thinking"] = self.config["thinking"]
        if self.config.get("reasoning_effort") is not None:
            payload["reasoning_effort"] = self.config["reasoning_effort"]
        if self.config.get("tools") is not None:
            payload["tools"] = self.config["tools"]
        return payload

    def _record_event(
        self,
        *,
        logical_turn: int,
        attempt_class: str,
        key_slot: str,
        max_tokens: int,
        transport_attempt_number: int,
        semantic_retry_number: int,
        status: int | None,
        returned_model_id: str | None,
        finish_reason: str | None,
        outcome: str,
        error_code: str | None = None,
        error_text: str | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        parsed_url = urlsplit(str(self.config["url"]))
        endpoint_origin = f"{parsed_url.scheme}://{parsed_url.netloc}"
        raw_usage = dict(usage or {})
        details = raw_usage.get("completion_tokens_details")
        details = details if isinstance(details, Mapping) else {}
        normalized_usage: dict[str, int] = {}
        for name, value in (
            ("prompt_tokens", raw_usage.get("prompt_tokens")),
            ("completion_tokens", raw_usage.get("completion_tokens")),
            ("reasoning_tokens", raw_usage.get("reasoning_tokens", details.get("reasoning_tokens"))),
            ("total_tokens", raw_usage.get("total_tokens")),
            ("prompt_cache_hit_tokens", raw_usage.get("prompt_cache_hit_tokens")),
            ("prompt_cache_miss_tokens", raw_usage.get("prompt_cache_miss_tokens")),
        ):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                normalized_usage[name] = value
        limits = self.ledger.limits
        self.events.append(
            {
                "schema_version": "1.0.0",
                "sequence": len(self.events) + 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "provider": self.config["provider"],
                "endpoint_origin": endpoint_origin,
                "requested_model_id": self.config["requested_model_id"],
                "returned_model_id": returned_model_id,
                "seed": int(self.config["seed"]),
                "temperature": float(self.config["temperature"]),
                "top_p": float(self.config["top_p"]),
                "thinking": self.config.get("thinking"),
                "reasoning_effort": self.config.get("reasoning_effort"),
                "request_timeout_seconds": int(self.config["request_timeout_seconds"]),
                "logical_turn": int(logical_turn),
                "api_attempt": int(self.ledger.totals["api_attempts"]),
                "attempt_class": attempt_class,
                "transport_attempt_number": int(transport_attempt_number),
                "semantic_retry_number": int(semantic_retry_number),
                "http_status": status,
                "finish_reason": finish_reason,
                "outcome": outcome,
                "key_slot": key_slot,
                "error_code": error_code,
                "sanitized_error": (
                    sanitize_text(error_text, self.keys.values()) if error_text is not None else None
                ),
                "max_tokens": int(max_tokens),
                "usage": normalized_usage,
                "tool_budget": {
                    name: int(limits[name])
                    for name in ("agent_turns", "tool_calls", "xtb_jobs", "xtb_scan_points")
                },
            }
        )

    def complete(self, messages: list[dict], *, logical_turn: int, resumed: bool = False) -> dict:
        self._resumed = bool(resumed)
        self.ledger.begin_turn(logical_turn)
        maximum_semantic_retries = int(self.config.get("max_semantic_retries", 1))
        for semantic_index in range(maximum_semantic_retries + 1):
            semantic_retry = semantic_index > 0
            max_tokens = int(
                self.config["semantic_retry_max_tokens"]
                if semantic_retry
                else self.config["initial_max_tokens"]
            )
            response, issue = self._complete_semantic_stage(
                messages,
                logical_turn=logical_turn,
                semantic_retry=semantic_retry,
                max_tokens=max_tokens,
            )
            if issue is None:
                return response
            if issue == "empty":
                self.ledger.record_empty_response_retry(logical_turn)
            if semantic_index >= maximum_semantic_retries:
                raise ApiPolicyError(f"semantic response remained invalid after retry: {issue}")
        raise ApiPolicyError("unreachable semantic retry state")

    def _complete_semantic_stage(
        self,
        messages: list[dict],
        *,
        logical_turn: int,
        semantic_retry: bool,
        max_tokens: int,
    ) -> tuple[dict, str | None]:
        transport_retry = 0
        key_fallback = False
        while True:
            attempt_class = self._attempt_class(semantic_retry, transport_retry, key_fallback)
            reason = None
            if transport_retry:
                reason = "transport retry"
            elif semantic_retry:
                reason = "empty or truncated response"
            elif key_fallback:
                reason = "voucher balance fallback"
            self.ledger.record_api_attempt(logical_turn, max_tokens, attempt_class, reason)
            slot = self.key_slot
            headers = {
                "Authorization": f"Bearer {self.keys[slot]}",
                "Content-Type": "application/json",
            }
            try:
                raw_response = self.transport.post(
                    self.config["url"],
                    headers=headers,
                    json=self._payload(messages, max_tokens),
                    timeout=int(self.config["request_timeout_seconds"]),
                )
            except (TimeoutError, OSError, http.client.HTTPException) as exc:
                self._record_event(
                    logical_turn=logical_turn,
                    attempt_class=attempt_class,
                    key_slot=slot,
                    max_tokens=max_tokens,
                    transport_attempt_number=transport_retry + 1,
                    semantic_retry_number=int(semantic_retry),
                    status=None,
                    returned_model_id=None,
                    finish_reason=None,
                    outcome="timeout" if isinstance(exc, TimeoutError) else "transport_error",
                    error_text=f"{type(exc).__name__}: {exc}",
                )
                if transport_retry >= int(self.config["max_transport_retries"]):
                    raise ApiPolicyError("transport retries exhausted") from exc
                self.sleep(min(2**transport_retry, 8))
                transport_retry += 1
                key_fallback = False
                continue

            status = int(raw_response.status_code)
            if status != 200:
                text = str(getattr(raw_response, "text", ""))
                error_code = extract_error_code(text)
                self._record_event(
                    logical_turn=logical_turn,
                    attempt_class=attempt_class,
                    key_slot=slot,
                    max_tokens=max_tokens,
                    transport_attempt_number=transport_retry + 1,
                    semantic_retry_number=int(semantic_retry),
                    status=status,
                    returned_model_id=None,
                    finish_reason=None,
                    outcome="http_error",
                    error_code=error_code,
                    error_text=text,
                )
                if (
                    self.config["provider"] == "siliconflow"
                    and should_switch_siliconflow_key(status, text, slot)
                    and self.keys.get("standard")
                ):
                    self.key_slot = "standard"
                    key_fallback = True
                    transport_retry = 0
                    continue
                if is_immediate_failure(status, text) or status == 403:
                    raise ApiPolicyError(f"non-retriable provider response: HTTP {status}")
                if status in RETRIABLE_HTTP_STATUSES and transport_retry < int(self.config["max_transport_retries"]):
                    self.sleep(min(2**transport_retry, 8))
                    transport_retry += 1
                    key_fallback = False
                    continue
                raise ApiPolicyError(f"provider response is not recoverable: HTTP {status}")

            try:
                payload = raw_response.json()
                choice = payload["choices"][0]
                message = choice["message"]
                finish_reason = choice.get("finish_reason")
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                self._record_event(
                    logical_turn=logical_turn,
                    attempt_class=attempt_class,
                    key_slot=slot,
                    max_tokens=max_tokens,
                    transport_attempt_number=transport_retry + 1,
                    semantic_retry_number=int(semantic_retry),
                    status=200,
                    returned_model_id=None,
                    finish_reason=None,
                    outcome="invalid_response",
                    error_text=f"{type(exc).__name__}: malformed response",
                )
                return {}, "invalid_response"

            content = message.get("content")
            tool_calls = message.get("tool_calls") or []
            empty = (content is None or not str(content).strip()) and not tool_calls
            truncated = finish_reason == "length"
            outcome = "empty" if empty else "truncated" if truncated else "success"
            returned_model_id = str(payload.get("model")) if payload.get("model") is not None else None
            self._record_event(
                logical_turn=logical_turn,
                attempt_class=attempt_class,
                key_slot=slot,
                max_tokens=max_tokens,
                transport_attempt_number=transport_retry + 1,
                semantic_retry_number=int(semantic_retry),
                status=200,
                returned_model_id=returned_model_id,
                finish_reason=str(finish_reason) if finish_reason is not None else None,
                outcome=outcome,
                usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
            )
            return payload, outcome if outcome != "success" else None

    def record_process_resume(self, *, logical_turn: int, reason: str) -> None:
        self.ledger.record_process_resume(logical_turn, reason)
        self._resumed = True

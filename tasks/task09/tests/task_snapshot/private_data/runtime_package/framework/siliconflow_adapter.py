"""Strict SiliconFlow transport and non-reflective provider classification.

The adapter fixes the requested model in configuration, disables SDK retries,
never accepts a fallback/model override, and never serializes a raw provider
response.  Error projection retains only an HTTP status, a small allowlisted
provider code, a sanitized message, and a frozen category.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
from typing import Any, Callable

from dotenv import load_dotenv
import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from .redaction import contains_key_like
from .provider_retry import TRANSIENT_PROVIDER_CATEGORIES
from .schemas import (
    MissingCredentialError,
    ModelConfig,
    ProviderError,
    ProviderPreflightConfig,
    SILICONFLOW_API_KEY_ENV,
)


PROVIDER_ERROR_CLASSIFICATION_VERSION = "provider-error-classification/2.0"

_SAFE_PROVIDER_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")

_BALANCE_CODES = {
    "insufficient_balance",
    "insufficient_credit",
    "insufficient_credits",
    "insufficient_quota",
    "balance_insufficient",
    "account_balance_insufficient",
}
_AUTH_CODES = {
    "authentication_failed",
    "invalid_api_key",
    "invalid_token",
    "unauthorized",
    "account_not_verified",
    "identity_verification_required",
}
_PERMISSION_CODES = {
    "forbidden",
    "permission_denied",
    "model_permission_denied",
    "model_access_denied",
}
_MODEL_NOT_FOUND_CODES = {
    "model_not_found",
    "unknown_model",
    "model_does_not_exist",
}

_BALANCE_PHRASES = (
    "insufficient balance",
    "balance is insufficient",
    "balance not enough",
    "insufficient account balance",
    "account balance is insufficient",
    "insufficient credit",
    "credits exhausted",
    "please recharge",
    "余额不足",
    "余额不够",
    "账户余额不足",
    "账户余额不够",
    "请充值",
)
_AUTH_PHRASES = (
    "authentication failed",
    "invalid api key",
    "invalid token",
    "invalid credential",
    "unauthorized",
    "signature verification",
    "identity verification",
    "real-name verification",
    "实名认证",
    "身份验证",
)
_PERMISSION_PHRASES = (
    "permission denied",
    "no permission",
    "access denied",
    "not authorized to access",
    "model permission",
    "无权限",
    "没有权限",
    "模型权限",
)
_MODEL_NOT_FOUND_PHRASES = (
    "model not found",
    "unknown model",
    "model does not exist",
    "模型不存在",
)


class ProviderCallError(ProviderError):
    """Carries only a safe, fixed-schema provider error projection."""

    def __init__(self, safe: Mapping[str, Any]) -> None:
        self.safe = dict(safe)
        self.retryable = self.safe.get("retryable") is True
        super().__init__(str(self.safe.get("status", "UNKNOWN_PROVIDER_ERROR")))


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _http_status(exc: BaseException) -> int | None:
    for item in _exception_chain(exc):
        raw = getattr(item, "status_code", None)
        if (
            isinstance(raw, int)
            and not isinstance(raw, bool)
            and 100 <= raw <= 599
        ):
            return raw
        response = getattr(item, "response", None)
        raw = getattr(response, "status_code", None)
        if (
            isinstance(raw, int)
            and not isinstance(raw, bool)
            and 100 <= raw <= 599
        ):
            return raw
    return None


def _safe_provider_code(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if contains_key_like(text):
        return None
    return text if _SAFE_PROVIDER_CODE.fullmatch(text) else None


def sanitize_provider_message(value: Any, *, max_chars: int = 240) -> str:
    """Return a bounded message without keys, controls, or raw response data."""

    if not isinstance(value, str):
        return "Provider request failed."
    text = _WHITESPACE.sub(" ", _CONTROL.sub(" ", value)).strip()
    if not text:
        return "Provider request failed."
    if contains_key_like(text):
        return "[REDACTED_PROVIDER_MESSAGE]"
    return text[:max_chars]


def _classification_text(value: Any) -> str:
    """Normalize provider prose for in-memory classification only."""

    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", _CONTROL.sub(" ", value)).strip().casefold()[:1000]


def _provider_code_and_message(
    exc: BaseException,
) -> tuple[str | None, str, str]:
    """Extract only two scalar fields from an SDK error body.

    The body itself, response headers, request, and exception string are never
    returned or retained.
    """

    for item in _exception_chain(exc):
        direct_code = _safe_provider_code(getattr(item, "provider_code", None))
        direct_message = getattr(item, "provider_message", None)
        if direct_code is not None or isinstance(direct_message, str):
            return (
                direct_code,
                sanitize_provider_message(direct_message),
                _classification_text(direct_message),
            )

        body = getattr(item, "body", None)
        if not isinstance(body, Mapping):
            continue
        candidate: Any = body.get("error", body)
        if isinstance(candidate, Mapping):
            code = None
            for name in ("code", "error_code", "type"):
                code = _safe_provider_code(candidate.get(name))
                if code is not None:
                    break
            message: Any = None
            for name in ("message", "msg", "detail"):
                if isinstance(candidate.get(name), str):
                    message = candidate[name]
                    break
            return (
                code,
                sanitize_provider_message(message),
                _classification_text(message),
            )
        if isinstance(candidate, str):
            return (
                None,
                sanitize_provider_message(candidate),
                _classification_text(candidate),
            )
    return None, "Provider request failed.", ""


def _contains_any(value: str, phrases: Sequence[str]) -> bool:
    return any(phrase in value for phrase in phrases)


def _classify_http_error(
    *,
    status_code: int,
    provider_code: str | None,
    provider_classification_text: str,
) -> str:
    code = (provider_code or "").casefold()
    message = provider_classification_text

    if status_code == 429:
        return "RATE_LIMITED"
    if status_code in {500, 502, 503, 504}:
        return "PROVIDER_OVERLOADED"
    if status_code == 408:
        return "READ_TIMEOUT"
    if status_code == 401:
        return "AUTHENTICATION_FAILED"
    if status_code == 402:
        # SiliconFlow uses HTTP 402 with provider code 30001 for an exhausted
        # account balance.  A 402 is a permanent campaign blocker, never a
        # transient overload and never eligible for retry or model fallback.
        return "INSUFFICIENT_BALANCE"
    if status_code == 404:
        return "MODEL_NOT_FOUND"
    if status_code == 403:
        if code in _BALANCE_CODES or _contains_any(message, _BALANCE_PHRASES):
            return "INSUFFICIENT_BALANCE"
        if code in _AUTH_CODES or _contains_any(message, _AUTH_PHRASES):
            return "AUTHENTICATION_FAILED"
        if code in _MODEL_NOT_FOUND_CODES or _contains_any(
            message, _MODEL_NOT_FOUND_PHRASES
        ):
            return "MODEL_NOT_FOUND"
        if code in _PERMISSION_CODES or _contains_any(
            message, _PERMISSION_PHRASES
        ):
            return "MODEL_PERMISSION_DENIED"
        # An opaque 403 must never be guessed to mean insufficient balance.
        return "MODEL_PERMISSION_DENIED"
    return "UNKNOWN_PROVIDER_ERROR"


def classify_provider_exception(exc: BaseException) -> dict[str, Any]:
    """Classify a provider failure without persisting provider-controlled data."""

    chain = _exception_chain(exc)
    status_code = _http_status(exc)
    (
        provider_code,
        provider_message,
        provider_classification_text,
    ) = _provider_code_and_message(exc)

    if any(isinstance(item, httpx.ReadTimeout) for item in chain):
        category = "READ_TIMEOUT"
    elif any(isinstance(item, httpx.ConnectTimeout) for item in chain):
        category = "NETWORK_CONNECTION_FAILED"
    elif any(isinstance(item, APITimeoutError) for item in chain):
        category = "READ_TIMEOUT"
    elif any(
        isinstance(
            item,
            (
                APIConnectionError,
                httpx.ConnectError,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        )
        for item in chain
    ):
        category = "NETWORK_CONNECTION_FAILED"
    elif any(isinstance(item, httpx.TimeoutException) for item in chain):
        category = "READ_TIMEOUT"
    elif any(isinstance(item, TimeoutError) for item in chain):
        category = "READ_TIMEOUT"
    elif any(
        isinstance(item, (ConnectionError, OSError))
        for item in chain
    ):
        category = "NETWORK_CONNECTION_FAILED"
    elif status_code is not None:
        category = _classify_http_error(
            status_code=status_code,
            provider_code=provider_code,
            provider_classification_text=provider_classification_text,
        )
    else:
        category = "UNKNOWN_PROVIDER_ERROR"

    retryable = category in TRANSIENT_PROVIDER_CATEGORIES
    return {
        "status": category,
        "error_category": category,
        "http_status": status_code,
        # Compatibility alias for consumers being migrated to v2.
        "status_code": status_code,
        "provider_error_code": provider_code,
        "sanitized_provider_message": provider_message,
        "exception_type": type(exc).__name__ or "ProviderException",
        "classification_version": PROVIDER_ERROR_CLASSIFICATION_VERSION,
        "retryable": retryable,
        "message_reflected": False,
        "raw_response_persisted": False,
        "response_headers_persisted": False,
        "request_persisted": False,
    }


def load_runtime_credential(project_root: str | Path | None = None) -> str:
    """Read the nearest trusted ancestor ``.env`` without exposing its values."""

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    explicit = os.environ.get("ANT_BENCHMARK_ENV_FILE", "").strip()
    if explicit:
        env_path = Path(explicit).expanduser().resolve()
    else:
        env_path = next(
            (
                candidate / ".env"
                for candidate in (root, *root.parents)
                if (candidate / ".env").is_file()
            ),
            root.parent / ".env",
        )
    try:
        # An explicitly pinned credential file is an operator decision and must
        # win over a possibly stale inherited process variable.  Automatic
        # ancestor discovery retains the conservative non-overriding behavior.
        load_dotenv(dotenv_path=env_path, override=bool(explicit))
    except Exception:
        raise MissingCredentialError("MISSING_GJLD_API_KEY") from None
    value = os.environ.get(SILICONFLOW_API_KEY_ENV, "")
    if not isinstance(value, str) or not value.strip():
        raise MissingCredentialError("MISSING_GJLD_API_KEY")
    return value


class SiliconFlowChatTransport:
    """A fixed-model OpenAI-compatible transport with no implicit retries."""

    requires_api_key = True
    is_mock = False
    transport_name = "siliconflow_openai_compatible"

    def __init__(
        self,
        config: ModelConfig | ProviderPreflightConfig,
        api_key: str,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        if not isinstance(api_key, str) or not api_key.strip():
            raise MissingCredentialError("MISSING_GJLD_API_KEY")
        canonical = client_factory is None
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderCallError(
                    _client_init_error("OpenAIClientUnavailable")
                ) from None
            client_factory = OpenAI
        try:
            self._client = client_factory(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        except Exception as exc:
            raise ProviderCallError(
                _client_init_error(type(exc).__name__)
            ) from None
        self.uses_canonical_client = canonical

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
        response_format: Mapping[str, Any] | None = None,
    ) -> Any:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("provider timeout must be positive")
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": min(float(timeout_seconds), self.config.timeout_seconds),
            "stream": False,
        }
        if tools:
            request["tools"] = [dict(item) for item in tools]
            request["tool_choice"] = "auto"
        if response_format is not None:
            if dict(response_format) != {"type": "json_object"}:
                raise ValueError("only JSON-object response format is allowed")
            request["response_format"] = {"type": "json_object"}
        try:
            # The response is returned only to the caller.  This class has no
            # response cache, debug log, or raw persistence path.
            return self._client.chat.completions.create(**request)
        except ProviderCallError:
            raise
        except Exception as exc:
            raise ProviderCallError(classify_provider_exception(exc)) from None


def _client_init_error(exception_type: str) -> dict[str, Any]:
    safe_type = (
        exception_type
        if isinstance(exception_type, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", exception_type)
        else "ProviderException"
    )
    category = "UNKNOWN_PROVIDER_ERROR"
    return {
        "status": category,
        "error_category": category,
        "http_status": None,
        "status_code": None,
        "provider_error_code": None,
        "sanitized_provider_message": "Provider client initialization failed.",
        "exception_type": safe_type,
        "classification_version": PROVIDER_ERROR_CLASSIFICATION_VERSION,
        "retryable": False,
        "message_reflected": False,
        "raw_response_persisted": False,
        "response_headers_persisted": False,
        "request_persisted": False,
    }


__all__ = [
    "PROVIDER_ERROR_CLASSIFICATION_VERSION",
    "ProviderCallError",
    "SiliconFlowChatTransport",
    "classify_provider_exception",
    "load_runtime_credential",
    "sanitize_provider_message",
]

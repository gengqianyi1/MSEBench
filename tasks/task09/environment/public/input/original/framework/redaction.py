"""Provider-boundary redaction helpers.

Provider exceptions are deliberately reduced to stable public status codes;
exception messages, args, headers and response bodies are never serialized.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


KEY_LIKE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._~+/-]{12,}|"
    r"[\"']?(?:[A-Za-z0-9_-]*api[_-]?key|token|secret|authorization)"
    r"[\"']?\s*[:=]\s*[\"']?\S+)"
)


def contains_key_like(value: str) -> bool:
    return bool(KEY_LIKE.search(value or ""))


def contains_key_like_recursive(value: Any) -> bool:
    """Inspect decoded JSON values, not only their escaped wire encoding."""

    if isinstance(value, str):
        return contains_key_like(value)
    if isinstance(value, Mapping):
        return any(
            contains_key_like(str(key))
            or contains_key_like_recursive(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_key_like_recursive(item) for item in value)
    return False


def safe_provider_error(exc: BaseException, status: str = "PROVIDER_REQUEST_FAILED") -> dict[str, Any]:
    """Return non-reflective metadata without retaining ``str(exc)``."""

    return {
        "status": status,
        "exception_type": type(exc).__name__ or "ProviderException",
        "message_reflected": False,
        "raw_response_persisted": False,
    }


def sanitize_model_text(value: Any, *, max_chars: int = 8000) -> str:
    if not isinstance(value, str):
        return ""
    text = value[:max_chars]
    if contains_key_like(text):
        return "[REDACTED_KEY_LIKE_MODEL_CONTENT]"
    return text


__all__ = [
    "contains_key_like",
    "contains_key_like_recursive",
    "safe_provider_error",
    "sanitize_model_text",
]

"""Persistence-boundary sanitization for trajectories and run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REDACTED = "[REDACTED]"
LOCAL_PATH = "<LOCAL_PATH>"

_HIDDEN_REASONING_KEYS = {
    "analysis",
    "chain_of_thought",
    "hidden_reasoning",
    "internal_reasoning",
    "reasoning",
    "reasoning_content",
    "thought",
    "thoughts",
    "thinking_content",
}
_HIDDEN_BLOCK_RE = re.compile(
    r"(?is)<(?:think|thinking|reasoning|analysis)>.*?</(?:think|thinking|reasoning|analysis)>"
)
_WINDOWS_ABSOLUTE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])[^\s\"'<>|,;)}\]]*"
)
_UNC_ABSOLUTE_RE = re.compile(
    r"(?<![\\A-Za-z0-9])\\\\[^\s\\/]+[\\/][^\s\"'<>|,;)}\]]*"
)
_POSIX_ABSOLUTE_RE = re.compile(
    r"(?<![:A-Za-z0-9_])/(?:home|Users|tmp|opt|srv|root|data|var/(?:tmp|folders)|workspace|mnt/[A-Za-z0-9._-]+)(?:/[^\s\"'`<>|,;)}\]]*)?"
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\bauthorization\s*[:=]\s*['\"]?bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(?<![A-Za-z0-9_])GJLD_API_KEY\s*[:=]\s*['\"]?[^\s'\",;}{]{8,}"),
)


def to_plain(value: Any, *, _depth: int = 0) -> Any:
    """Convert SDK/dataclass-like values to bounded JSON-compatible objects."""

    if _depth > 32:
        return "<MAX_DEPTH>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "<NON_FINITE>"
    if isinstance(value, datetime):
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_plain(item, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(item, _depth=_depth + 1) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_plain(model_dump(mode="json"), _depth=_depth + 1)
        except TypeError:
            return to_plain(model_dump(), _depth=_depth + 1)
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return to_plain(as_dict(), _depth=_depth + 1)
    public = getattr(value, "__dict__", None)
    if isinstance(public, dict):
        return to_plain(
            {key: item for key, item in public.items() if not str(key).startswith("_")},
            _depth=_depth + 1,
        )
    return f"<{type(value).__name__}>"


def _sanitize_text(text: str, secrets: Sequence[str]) -> str:
    result = _HIDDEN_BLOCK_RE.sub("<HIDDEN_REASONING_REMOVED>", text)
    for secret in secrets:
        if isinstance(secret, str) and secret:
            result = result.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(REDACTED, result)
    result = _UNC_ABSOLUTE_RE.sub(LOCAL_PATH, result)
    result = _WINDOWS_ABSOLUTE_RE.sub(LOCAL_PATH, result)
    result = _POSIX_ABSOLUTE_RE.sub(LOCAL_PATH, result)
    return result


def sanitize_for_persistence(
    value: Any,
    *,
    secrets: Sequence[str] = (),
    drop_hidden_reasoning: bool = True,
) -> Any:
    """Return a JSON-safe object with secrets, local paths, and reasoning removed."""

    plain = to_plain(value)

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            return _sanitize_text(item, secrets)
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in item.items():
                lowered = str(key).casefold()
                if drop_hidden_reasoning and lowered in _HIDDEN_REASONING_KEYS:
                    continue
                result[str(key)] = visit(child)
            return result
        return item

    return visit(plain)


def safe_exception_record(exc: BaseException, *, status: str = "FAILED") -> dict[str, str]:
    """Represent an exception without persisting its message, args, or traceback."""

    original_type = getattr(exc, "original_type", None)
    safe_status = getattr(exc, "safe_status", None)
    return {
        "type": str(original_type) if isinstance(original_type, str) and original_type else type(exc).__name__,
        "status": str(safe_status) if isinstance(safe_status, str) and safe_status else status,
    }


def has_hidden_reasoning(value: Any) -> bool:
    """Conservative diagnostic check used by tests and release scanners."""

    plain = to_plain(value)
    if isinstance(plain, dict):
        for key, child in plain.items():
            if str(key).casefold() in _HIDDEN_REASONING_KEYS or has_hidden_reasoning(child):
                return True
    elif isinstance(plain, list):
        return any(has_hidden_reasoning(child) for child in plain)
    elif isinstance(plain, str):
        return bool(_HIDDEN_BLOCK_RE.search(plain))
    return False


def has_absolute_local_path(value: Any) -> bool:
    text = str(to_plain(value))
    return any(pattern.search(text) for pattern in (_WINDOWS_ABSOLUTE_RE, _UNC_ABSOLUTE_RE, _POSIX_ABSOLUTE_RE))


__all__ = [
    "LOCAL_PATH",
    "REDACTED",
    "has_absolute_local_path",
    "has_hidden_reasoning",
    "safe_exception_record",
    "sanitize_for_persistence",
    "to_plain",
]

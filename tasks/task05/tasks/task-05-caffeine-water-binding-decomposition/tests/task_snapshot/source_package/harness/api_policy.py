"""Provider-independent classification rules for retry and key fallback."""

from __future__ import annotations

import json
import re
from typing import Iterable


RETRIABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
BALANCE_MARKERS = (
    "余额不足",
    "余额耗尽",
    "insufficient balance",
    "balance insufficient",
    "balance exhausted",
    "quota exhausted",
    "insufficient quota",
    "out of credits",
    "credit balance",
)


def extract_error_code(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    return str(code) if code is not None else None


def is_explicit_balance_error(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text)).strip().lower()
    return any(marker in normalized for marker in BALANCE_MARKERS)


def should_switch_siliconflow_key(status: int, text: str, key_slot: str) -> bool:
    return key_slot == "voucher" and (status == 402 or (status == 403 and is_explicit_balance_error(text)))


def is_immediate_failure(status: int, text: str) -> bool:
    code = (extract_error_code(text) or "").lower()
    lowered = str(text).lower()
    return status == 401 or code == "model_not_found" or "model_not_found" in lowered or "model does not exist" in lowered


def sanitize_text(text: str, secrets: Iterable[str], limit: int = 1000) -> str:
    sanitized = str(text)
    for value in secrets:
        if value:
            sanitized = sanitized.replace(value, "[REDACTED]")
    return sanitized[:limit]

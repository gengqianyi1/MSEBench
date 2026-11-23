"""Run the frozen one-turn SiliconFlow preflight for all formal model IDs.

This script performs no xTB work and creates no formal trajectory.  It stores
only a safe projection of each response/error; raw provider responses, request
messages, credentials, headers, and exception text are never written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from framework.schemas import (
    ACTIVE_PREFLIGHT_REPORT,
    BENCHMARK_VERSION,
    FORMAL_MODELS,
    PREFLIGHT_MAX_OUTPUT_TOKENS,
    PREFLIGHT_NATIVE_XTB_CALLS,
    PREFLIGHT_PROVIDER_TURNS,
    PROTOCOL,
    TASK_ID,
    MissingCredentialError,
    ProviderPreflightConfig,
)
from framework.redaction import contains_key_like
from framework.provider_retry import KNOWN_PROVIDER_CATEGORIES
from framework.siliconflow_adapter import (
    PROVIDER_ERROR_CLASSIFICATION_VERSION,
    ProviderCallError,
    SiliconFlowChatTransport,
    classify_provider_exception,
    load_runtime_credential,
    sanitize_provider_message,
)


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SCHEMA_VERSION = "provider-preflight/2.0"
_SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]{0,255}$")
_SAFE_PROVIDER_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")

_SYSTEM_PROMPT = (
    'Return exactly one JSON object and no prose: {"ok":true}. '
    "Do not call tools and do not include credentials."
)
_USER_PROMPT = 'Connectivity and identity preflight. Return {"ok":true}.'


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _safe_model_id(value: Any) -> str | None:
    return (
        value
        if (
            isinstance(value, str)
            and _SAFE_MODEL_ID.fullmatch(value)
            and not contains_key_like(value)
        )
        else None
    )


def _strict_json_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("provider content is not text")
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().casefold() not in {
            "```",
            "```json",
        }:
            raise ValueError("unsupported code fence")
        text = "\n".join(lines[1:-1]).strip()
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    value = json.loads(
        text,
        object_pairs_hook=hook,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if duplicates or not isinstance(value, dict):
        raise ValueError("provider content is not a duplicate-free JSON object")
    return value


def _response_projection(
    response: Any, *, requested_model: str, timestamp_utc: str
) -> dict[str, Any]:
    actual_model = _safe_model_id(_field(response, "model"))
    choices = _field(response, "choices")
    message = None
    if isinstance(choices, (list, tuple)) and choices:
        message = _field(choices[0], "message")
    content = _field(message, "content")

    structured = False
    response_hash: str | None = None
    if isinstance(content, str):
        try:
            parsed = _strict_json_object(content)
        except (ValueError, json.JSONDecodeError, UnicodeError):
            parsed = None
        if parsed == {"ok": True}:
            structured = True
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            response_hash = hashlib.sha256(canonical).hexdigest().upper()

    exact_identity = actual_model == requested_model
    if actual_model is None:
        status = "FAIL"
        category = "IDENTITY_MISMATCH"
    elif not structured:
        status = "FAIL"
        category = "PROTOCOL_VIOLATION"
    elif not exact_identity:
        status = "FAIL"
        category = "IDENTITY_MISMATCH"
    else:
        status = "PASS"
        category = None

    usage = _field(response, "usage")
    usage_projection: dict[str, int | None] = {}
    for public_name, provider_name in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        raw = _field(usage, provider_name)
        usage_projection[public_name] = (
            raw
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0
            else None
        )
    usage_projection["provider_usage_unavailable"] = all(
        value is None for value in usage_projection.values()
    )

    return {
        "status": status,
        "error_category": category,
        "http_status": 200,
        "provider_error_code": None,
        "sanitized_provider_message": (
            "Preflight response accepted."
            if status == "PASS"
            else "Provider response failed identity or JSON protocol validation."
        ),
        "requested_model": requested_model,
        "actual_model": actual_model,
        "exact_identity": exact_identity,
        "structured_json_valid": structured,
        "fallback_used": bool(actual_model and actual_model != requested_model),
        "model_override_used": False,
        "provider_turns": 1,
        "successful_provider_turns": 1,
        "native_xtb_calls": 0,
        "formal_trajectory": False,
        "response_json_sha256": response_hash,
        "usage": usage_projection,
        "raw_response_persisted": False,
        "timestamp_utc": timestamp_utc,
    }


def _error_projection(
    safe: Mapping[str, Any], *, requested_model: str, timestamp_utc: str
) -> dict[str, Any]:
    category = safe.get("error_category", safe.get("status"))
    if category not in KNOWN_PROVIDER_CATEGORIES:
        category = "UNKNOWN_PROVIDER_ERROR"
    raw_status = safe.get("http_status")
    http_status = (
        raw_status
        if isinstance(raw_status, int)
        and not isinstance(raw_status, bool)
        and 100 <= raw_status <= 599
        else None
    )
    raw_code = safe.get("provider_error_code")
    provider_code = (
        raw_code
        if isinstance(raw_code, str)
        and _SAFE_PROVIDER_CODE.fullmatch(raw_code)
        and not contains_key_like(raw_code)
        else None
    )
    raw_message = safe.get("sanitized_provider_message")
    message = sanitize_provider_message(raw_message)
    return {
        "status": "BLOCKED",
        "error_category": category,
        "http_status": http_status,
        "provider_error_code": provider_code,
        "sanitized_provider_message": message,
        "requested_model": requested_model,
        "actual_model": None,
        "exact_identity": False,
        "structured_json_valid": False,
        "fallback_used": False,
        "model_override_used": False,
        "provider_turns": 1,
        "successful_provider_turns": 0,
        "native_xtb_calls": 0,
        "formal_trajectory": False,
        "response_json_sha256": None,
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "provider_usage_unavailable": True,
        },
        "raw_response_persisted": False,
        "timestamp_utc": timestamp_utc,
    }


def preflight_one(
    label: str,
    *,
    api_key: str,
    transport_factory: Callable[[ProviderPreflightConfig, str], Any] | None = None,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Execute exactly one provider request and return only its safe projection."""

    config = ProviderPreflightConfig.for_label(label)
    if transport_factory is None:
        transport_factory = lambda cfg, key: SiliconFlowChatTransport(cfg, key)
    timestamp = now()
    try:
        transport = transport_factory(config, api_key)
        response = transport.complete(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT},
            ],
            tools=[],
            timeout_seconds=config.timeout_seconds,
            response_format={"type": "json_object"},
        )
    except ProviderCallError as exc:
        return _error_projection(
            exc.safe,
            requested_model=config.model,
            timestamp_utc=timestamp,
        )
    except Exception as exc:
        return _error_projection(
            classify_provider_exception(exc),
            requested_model=config.model,
            timestamp_utc=timestamp,
        )
    return _response_projection(
        response,
        requested_model=config.model,
        timestamp_utc=timestamp,
    )


def run_preflight(
    *,
    credential_loader: Callable[[Path], str] = load_runtime_credential,
    transport_factory: Callable[[ProviderPreflightConfig, str], Any] | None = None,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Probe all three exact model IDs without retries or native xTB calls."""

    started = now()
    try:
        runtime_credential = credential_loader(ROOT)
    except MissingCredentialError:
        runtime_credential = ""
        results = {}
        for label, model in FORMAL_MODELS.items():
            receipt = _error_projection(
                {
                    "error_category": "AUTHENTICATION_FAILED",
                    "http_status": None,
                    "provider_error_code": None,
                    "sanitized_provider_message": "Required credential unavailable.",
                },
                requested_model=model,
                timestamp_utc=now(),
            )
            receipt["provider_turns"] = 0
            results[label] = receipt
    else:
        results = {
            label: preflight_one(
                label,
                api_key=runtime_credential,
                transport_factory=transport_factory,
                now=now,
            )
            for label in FORMAL_MODELS
        }
    statuses = {item["status"] for item in results.values()}
    if statuses == {"PASS"}:
        overall = "PASS"
    elif "BLOCKED" in statuses:
        overall = "BLOCKED"
    else:
        overall = "FAIL"
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "protocol": PROTOCOL,
        "provider": "siliconflow",
        "formal_trajectory": False,
        "native_xtb_calls": PREFLIGHT_NATIVE_XTB_CALLS,
        "preflight_provider_turns_per_model": PREFLIGHT_PROVIDER_TURNS,
        "preflight_max_output_tokens": PREFLIGHT_MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "provider_error_classification_version": (
            PROVIDER_ERROR_CLASSIFICATION_VERSION
        ),
        "raw_provider_responses_persisted": False,
        "started_at_utc": started,
        "completed_at_utc": now(),
        "models": results,
        "status": overall,
        "formal_campaign_allowed": overall == "PASS",
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = _resolve_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)


def _resolve_output_path(value: Path) -> Path:
    candidate = value if value.is_absolute() else ROOT / value
    resolved = candidate.resolve()
    reports_root = (ROOT / "reports").resolve()
    try:
        resolved.relative_to(reports_root)
    except ValueError as exc:
        raise ValueError("preflight output must remain under project reports/") from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ACTIVE_PREFLIGHT_REPORT,
    )
    args = parser.parse_args()
    report = run_preflight()
    output = _resolve_output_path(args.output)
    _write_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "formal_campaign_allowed": report["formal_campaign_allowed"],
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if report["status"] == "PASS":
        return 0
    if report["status"] == "BLOCKED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

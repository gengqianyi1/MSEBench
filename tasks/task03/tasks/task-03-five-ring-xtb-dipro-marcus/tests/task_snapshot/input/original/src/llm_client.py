from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI

@dataclass
class LLMCallResult:
    ok: bool
    content: str
    model: Optional[str]
    attempts: int
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    message: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    exception_type: Optional[str] = None
    traceback: Optional[str] = None
    empty_content: bool = False
    max_tokens_used: Optional[int] = None
    reasoning_budget_exhausted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _extract_message_and_content(response: Any):
    raw = _to_serializable(response)
    message_dict = None
    content = ""
    finish_reason = None

    try:
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        message_dict = _to_serializable(message)
        raw_content = getattr(message, "content", None)
        content = "" if raw_content is None else str(raw_content).strip()
    except Exception:
        try:
            choice = raw.get("choices", [])[0]
            finish_reason = choice.get("finish_reason")
            message_dict = choice.get("message", {})
            content = str(message_dict.get("content") or "").strip()
        except Exception:
            pass

    try:
        usage = _to_serializable(getattr(response, "usage", None))
    except Exception:
        usage = None

    return message_dict, content, finish_reason, usage


def _is_reasoning_budget_exhausted(finish_reason, usage, content: str) -> bool:
    if content:
        return False
    if finish_reason != "length":
        return False
    if not usage:
        return True
    completion_tokens = usage.get("completion_tokens")
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    if completion_tokens is None or reasoning_tokens is None:
        return True
    return int(completion_tokens) == int(reasoning_tokens)


def make_client(llm_cfg: Dict[str, Any]) -> OpenAI:
    api_key = llm_cfg.get("api_key")
    if not api_key or api_key == "PUT_YOUR_API_KEY_HERE":
        raise RuntimeError("Missing API key. Please edit configs/api_config.json and set `api_key`.")

    return OpenAI(
        api_key=api_key,
        base_url=llm_cfg.get("base_url"),
        timeout=llm_cfg.get("timeout", 60),
        # Retry only in call_llm() so attempt accounting and wall time remain
        # visible. The SDK otherwise retries twice internally and multiplies
        # the task-level retry loop.
        max_retries=0,
    )


def _token_schedule(llm_cfg: Dict[str, Any]) -> List[int]:
    schedule = llm_cfg.get("max_tokens_schedule")
    if schedule:
        return [int(x) for x in schedule]
    base = int(llm_cfg.get("max_tokens", 4096))
    retries = int(llm_cfg.get("max_retries", 0))
    return [base * (2 ** i) for i in range(retries + 1)]


def call_llm(messages: List[Dict[str, str]], llm_cfg: Dict[str, Any]) -> LLMCallResult:
    model = llm_cfg.get("model")
    max_retries = int(llm_cfg.get("max_retries", 0))
    retry_delay = float(llm_cfg.get("retry_delay_seconds", 2))
    client = make_client(llm_cfg)
    schedule = _token_schedule(llm_cfg)
    while len(schedule) < max_retries + 1:
        schedule.append(schedule[-1] * 2)

    last = None

    for attempt in range(1, max_retries + 2):
        max_tokens_this = schedule[attempt - 1]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=llm_cfg.get("temperature", 0.7),
                top_p=llm_cfg.get("top_p", 1.0),
                max_tokens=max_tokens_this,
            )
            message_dict, content, finish_reason, usage = _extract_message_and_content(response)
            raw_response = _to_serializable(response)

            if content:
                return LLMCallResult(
                    ok=True,
                    content=content,
                    model=model,
                    attempts=attempt,
                    finish_reason=finish_reason,
                    usage=usage,
                    message=message_dict,
                    raw_response=raw_response if llm_cfg.get("save_full_api_response", True) else None,
                    max_tokens_used=max_tokens_this,
                )

            exhausted = _is_reasoning_budget_exhausted(finish_reason, usage, content)
            error = (
                "LLM returned empty message.content because completion budget was consumed by reasoning tokens."
                if exhausted else
                "LLM returned empty message.content."
            )
            last = LLMCallResult(
                ok=False,
                content="",
                model=model,
                attempts=attempt,
                finish_reason=finish_reason,
                usage=usage,
                message=message_dict,
                raw_response=raw_response if llm_cfg.get("save_full_api_response", True) else None,
                error=error,
                empty_content=True,
                max_tokens_used=max_tokens_this,
                reasoning_budget_exhausted=exhausted,
            )
        except Exception as exc:
            last = LLMCallResult(
                ok=False,
                content="",
                model=model,
                attempts=attempt,
                error=str(exc),
                exception_type=type(exc).__name__,
                traceback=traceback.format_exc(),
                max_tokens_used=max_tokens_this,
            )

        if attempt <= max_retries:
            time.sleep(retry_delay)

    assert last is not None
    return last

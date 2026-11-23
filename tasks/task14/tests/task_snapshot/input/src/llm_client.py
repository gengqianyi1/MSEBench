from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


@dataclass
class LLMCallResult:
    """Serializable result for one LLM call."""
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
    truncated: bool = False
    transport: str = "non_streaming"
    history_compressed: bool = False
    reasoning_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_serializable(obj: Any) -> Any:
    """Convert OpenAI/Pydantic response objects into JSON-serializable data."""
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


def _extract_message_and_content(response: Any) -> tuple[Optional[Dict[str, Any]], str, Optional[str], Optional[Dict[str, Any]]]:
    """Extract message, content, finish_reason, and usage from OpenAI-compatible response."""
    raw = _to_serializable(response)

    message_dict: Optional[Dict[str, Any]] = None
    content = ""
    finish_reason = None

    try:
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        message_dict = _to_serializable(message)

        raw_content = getattr(message, "content", None)

        if isinstance(raw_content, list):
            parts = []
            for block in raw_content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(str(block["text"]))
                    elif block.get("type") == "text" and "content" in block:
                        parts.append(str(block["content"]))
                else:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(str(text))
            content = "".join(parts).strip()
        elif raw_content is not None:
            content = str(raw_content).strip()
        else:
            content = ""

    except Exception:
        try:
            choices = raw.get("choices", [])
            if choices:
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                message_dict = choice.get("message", {})
                raw_content = message_dict.get("content", "")
                if isinstance(raw_content, str):
                    content = raw_content.strip()
                elif raw_content is not None:
                    content = str(raw_content).strip()
        except Exception:
            pass

    usage = None
    try:
        usage = _to_serializable(getattr(response, "usage", None))
    except Exception:
        usage = None

    return message_dict, content, finish_reason, usage


def _is_reasoning_budget_exhausted(
    finish_reason: Optional[str],
    usage: Optional[Dict[str, Any]],
    content: str,
) -> bool:
    """Detect the common reasoning-model failure: all completion tokens spent on reasoning."""
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
    """Create an OpenAI-compatible client from JSON config."""
    api_key = llm_cfg.get("api_key")
    if not api_key or str(api_key).startswith("PUT_YOUR_"):
        raise RuntimeError(
            "Missing API key. Please edit configs/api_config.json and set `api_key`."
        )

    return OpenAI(
        api_key=api_key,
        base_url=llm_cfg.get("base_url", "https://api.deepseek.com"),
        timeout=llm_cfg.get("timeout", 60),
        max_retries=0,      # retries are this module's job; the SDK must not hide an attempt
    )


def _build_max_tokens_schedule(llm_cfg: Dict[str, Any]) -> List[int]:
    """Build per-attempt max_tokens schedule.

    For reasoning models, repeating the same small max_tokens can repeatedly fail
    with finish_reason='length' and empty content. A schedule lets retries expand
    the completion budget.
    """
    schedule = llm_cfg.get("max_tokens_schedule")
    if schedule:
        return [int(x) for x in schedule]

    base = int(llm_cfg.get("max_tokens", 4096))
    max_retries = int(llm_cfg.get("max_retries", 0))
    return [base * (2 ** i) for i in range(max_retries + 1)]


def _compress_history(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], bool]:
    """Drop the middle of a multi-round conversation, keeping the contract and the latest state.

    Second-round acceptance checklist §16.5: "重试策略必须具有实际变化。使用完全相同的长 prompt、
    相同输出预算和相同模型参数重复调用，不能视为有效恢复策略." Escalating max_tokens is one such
    change and it fixes a truncated or reasoning-exhausted reply, but it does nothing for a model
    that is drowning in accumulated round feedback -- and it makes each attempt slower, so on a
    gateway that drops long requests it makes the next attempt more likely to fail, not less.

    So the last attempt keeps the system message, the first user message (which carries the task
    statement, the constraint windows and the output contract -- dropping it would change the task)
    and the most recent user message (the current round's feedback), discarding the middle rounds.
    The model loses its own history of intermediate attempts, which is a real loss, but a degraded
    answer beats a lost round, and the trajectory records that this happened.
    """
    if len(messages) <= 3:
        return messages, False
    system = [m for m in messages if m.get("role") == "system"][:1]
    users = [m for m in messages if m.get("role") == "user"]
    if len(users) < 2:
        return messages, False
    kept = system + [users[0]] + [users[-1]]
    return kept, True


def _stream_completion(client: OpenAI, create_kwargs: Dict[str, Any]) -> Tuple[Any, str, Optional[str], Optional[Dict[str, Any]], int]:
    """Run a streamed chat completion and reassemble it into the non-streaming shape.

    WHY STREAMING EXISTS HERE (measured 2026-08-09). The gateway serving kimi-k2.6 and GLM-5.2 for
    the second round drops a long-running NON-streaming request outright: with a ~24 KB question the
    connection died after 3.8 s (GLM) and 112-148 s (kimi) with an APIConnectionError, while the
    identical request as a stream completed normally (94 s and 809 s respectively). No client-side
    timeout helps, because the socket is closed from the far end. A streamed request keeps arriving
    in chunks, so nothing along the path considers it idle.

    The reassembled result is deliberately the same shape the non-streaming path returns, so every
    downstream check -- truncation, empty content, reasoning exhaustion -- is written once.
    """
    chunks: List[str] = []
    reasoning_chars = 0
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    last_chunk: Any = None

    kwargs = dict(create_kwargs)
    kwargs["stream"] = True
    # Usage on the final chunk is an extension; a gateway that rejects the option must still stream.
    try_with_usage = True
    while True:
        try:
            if try_with_usage:
                kwargs["stream_options"] = {"include_usage": True}
            else:
                kwargs.pop("stream_options", None)
            stream = client.chat.completions.create(**kwargs)
            break
        except TypeError:
            if not try_with_usage:
                raise
            try_with_usage = False
        except Exception as exc:                                    # noqa: BLE001
            if try_with_usage and "stream_options" in str(exc):
                try_with_usage = False
                continue
            raise

    with stream as events:
        for event in events:
            last_chunk = event
            if getattr(event, "usage", None):
                usage = _to_serializable(event.usage)
            for choice in (getattr(event, "choices", None) or []):
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    chunks.append(piece)
                # Reasoning models on some gateways stream their chain of thought in a separate
                # field. It is NOT content -- counting it as content would let an empty answer look
                # like a full one -- but its size is the evidence that the budget went to reasoning.
                thinking = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if thinking:
                    reasoning_chars += len(str(thinking))

    return last_chunk, "".join(chunks).strip(), finish_reason, usage, reasoning_chars


def call_llm(messages: List[Dict[str, str]], llm_cfg: Dict[str, Any]) -> LLMCallResult:
    """Call an OpenAI-compatible chat completion API with retries and full debug capture.

    Returns an LLMCallResult instead of a bare string so the caller can save API
    diagnostics such as finish_reason, usage, raw response, and empty-content errors.
    """
    model = llm_cfg.get("model")
    max_retries = int(llm_cfg.get("max_retries", 0))
    retry_delay = float(llm_cfg.get("retry_delay_seconds", 2))
    raise_on_empty = bool(llm_cfg.get("raise_on_empty_response", False))
    use_stream = bool(llm_cfg.get("stream", False))

    client = make_client(llm_cfg)
    token_schedule = _build_max_tokens_schedule(llm_cfg)

    # Ensure enough entries for attempts.
    while len(token_schedule) < max_retries + 1:
        token_schedule.append(token_schedule[-1] * 2)

    last_result: Optional[LLMCallResult] = None

    # The completion budget escalates only for failures that a bigger budget can actually fix
    # (empty content because reasoning ate the budget, or a truncated object). Transport failures
    # such as timeouts are retried at the SAME budget: raising max_tokens makes the request slower,
    # so escalating on a timeout guarantees the next attempt times out too.
    budget_index = 0

    for attempt in range(1, max_retries + 2):
        max_tokens_this_attempt = int(token_schedule[min(budget_index, len(token_schedule) - 1)])

        # Last attempt: change something the previous attempts did not change. Repeating an
        # identical long prompt at an identical budget is explicitly not a recovery strategy
        # (second-round checklist §16.5).
        is_last_attempt = attempt == max_retries + 1
        attempt_messages, compressed = (
            _compress_history(messages) if is_last_attempt and attempt > 1 else (messages, False)
        )

        try:
            # `extra_body` is passed through verbatim to the provider. It exists for
            # provider-specific switches that are not part of the OpenAI schema -- for example
            # SiliconFlow's `{"enable_thinking": false}` for Qwen3.5, a reasoning model that
            # otherwise spends its whole completion budget thinking and returns EMPTY content
            # (measured: 285 reasoning tokens vs 8 tokens of JSON with the switch on). That is the
            # same silent round-loss deepseek showed at max_tokens=8192, and a round lost to the
            # plumbing is not a measurement of the model's chemistry. A config that declares
            # extra_body while the client ignores it is worse than not declaring it, so
            # grader/scripts/check_consistency.py asserts that this passthrough exists.
            #
            # NOTE for kimi-k2.6: `{"thinking": {"type": "disabled"}}` does make it answer inside a
            # small budget, and the other two switch spellings do not. It is deliberately NOT used
            # in the delivered configs: with thinking off the model resubmits the baseline design,
            # which the task states scores zero, so its score would not be comparable with the other
            # two models'. Raise the budget instead -- see max_tokens_schedule.
            extra_body = llm_cfg.get("extra_body") or None
            create_kwargs = dict(
                model=model,
                messages=attempt_messages,
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=max_tokens_this_attempt,
            )
            if extra_body:
                create_kwargs["extra_body"] = extra_body

            reasoning_chars = 0
            if use_stream:
                response, content, finish_reason, usage, reasoning_chars = _stream_completion(
                    client, create_kwargs)
                message_dict = {"role": "assistant", "content": content}
                raw_response = _to_serializable(response)
            else:
                response = client.chat.completions.create(**create_kwargs)
                message_dict, content, finish_reason, usage = _extract_message_and_content(response)
                raw_response = _to_serializable(response)

            transport = "streaming" if use_stream else "non_streaming"
            truncated = bool(content) and finish_reason == "length"

            if content and not truncated:
                return LLMCallResult(
                    ok=True,
                    content=content,
                    model=model,
                    attempts=attempt,
                    finish_reason=finish_reason,
                    usage=usage,
                    message=message_dict,
                    raw_response=raw_response if llm_cfg.get("save_full_api_response", True) else None,
                    empty_content=False,
                    max_tokens_used=max_tokens_this_attempt,
                    reasoning_budget_exhausted=False,
                    transport=transport,
                    history_compressed=compressed,
                    reasoning_chars=reasoning_chars,
                )

            if truncated:
                # The model produced content but ran out of completion budget mid-object, so the
                # JSON is cut off and cannot parse. This is a harness limitation rather than a
                # modelling error, so retry with the next (larger) budget in the schedule instead
                # of charging the model a wasted round. Previously any non-empty content was
                # accepted immediately, which silently cost a round whenever a rationale ran long.
                last_result = LLMCallResult(
                    ok=False,
                    content=content,
                    model=model,
                    attempts=attempt,
                    finish_reason=finish_reason,
                    usage=usage,
                    message=message_dict,
                    raw_response=raw_response if llm_cfg.get("save_full_api_response", True) else None,
                    error=(
                        f"LLM response was truncated (finish_reason='length') at "
                        f"max_tokens={max_tokens_this_attempt}; the returned JSON is incomplete. "
                        f"Retrying with a larger completion budget."
                    ),
                    empty_content=False,
                    max_tokens_used=max_tokens_this_attempt,
                    truncated=True,
                    transport=transport,
                    history_compressed=compressed,
                    reasoning_chars=reasoning_chars,
                )
                budget_index += 1  # a larger budget can fix truncation
                if attempt <= max_retries:
                    time.sleep(retry_delay)
                    continue

            elif not content:
                # Empty content. Distinguish "all completion tokens went to reasoning" from a
                # generic empty response so the debug log points at the right fix.
                reasoning_budget_exhausted = _is_reasoning_budget_exhausted(
                    finish_reason=finish_reason,
                    usage=usage,
                    content=content,
                )
                # A streamed reply whose only payload was reasoning tells the same story even when
                # the gateway declines to break usage down into reasoning tokens.
                if not reasoning_budget_exhausted and finish_reason == "length" and reasoning_chars:
                    reasoning_budget_exhausted = True

                if reasoning_budget_exhausted:
                    error = (
                        "LLM returned empty message.content because finish_reason='length' and "
                        "the completion budget appears to have been consumed by reasoning tokens. "
                        "Increase max_tokens or max_tokens_schedule for reasoning models."
                    )
                else:
                    error = (
                        "LLM returned empty message.content. Check model name, provider response format, "
                        "finish_reason, content filtering, or whether the selected model returns text in a different field."
                    )

                last_result = LLMCallResult(
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
                    max_tokens_used=max_tokens_this_attempt,
                    reasoning_budget_exhausted=reasoning_budget_exhausted,
                    transport=transport,
                    history_compressed=compressed,
                    reasoning_chars=reasoning_chars,
                )
                budget_index += 1  # a larger budget can fix a reasoning-exhausted empty response

                if raise_on_empty:
                    raise RuntimeError(error)

        except Exception as exc:
            last_result = LLMCallResult(
                ok=False,
                content="",
                model=model,
                attempts=attempt,
                error=str(exc),
                exception_type=type(exc).__name__,
                traceback=traceback.format_exc(),
                empty_content=False,
                max_tokens_used=max_tokens_this_attempt,
                transport="streaming" if use_stream else "non_streaming",
                history_compressed=compressed,
            )

        if attempt <= max_retries:
            time.sleep(retry_delay)

    assert last_result is not None
    return last_result

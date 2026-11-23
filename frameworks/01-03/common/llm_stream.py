"""Small OpenAI-compatible chat-completions SSE reader.

The benchmark agents use urllib instead of an SDK.  This module keeps the
streaming transport in one place and returns the same response shape that the
non-streaming code expected.
"""
from __future__ import annotations

import json


class StreamReadError(RuntimeError):
    """A stream failed after possibly producing a partial response."""

    def __init__(self, cause: BaseException, payload: dict | None, content: str):
        super().__init__(str(cause))
        self.cause = cause
        self.payload = payload
        self.content = content


def _merge_text(parts: dict[int, list[str]], index: int, value: object) -> None:
    if isinstance(value, str):
        parts.setdefault(index, []).append(value)


def read_chat_completion(response) -> dict:
    """Read an SSE response and reconstruct an OpenAI-style JSON response.

    Providers that ignore ``stream=true`` and return ordinary JSON are also
    accepted for compatibility.
    """
    content_parts: dict[int, list[str]] = {}
    reasoning_parts: dict[int, list[str]] = {}
    typo_reasoning_parts: dict[int, list[str]] = {}
    choices: dict[int, dict] = {}
    tool_calls: dict[int, dict[int, dict]] = {}
    base: dict = {}
    saw_sse = False
    saw_done = False
    partial_content = ""
    first_nonempty = True

    def consume(chunk: object) -> None:
        nonlocal base, saw_done, partial_content
        if not isinstance(chunk, dict):
            return
        if not base:
            base = {key: value for key, value in chunk.items() if key != "choices"}
        else:
            for key, value in chunk.items():
                if key != "choices":
                    base[key] = value
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = int(choice.get("index", 0))
            state = choices.setdefault(index, {"finish_reason": None, "role": "assistant"})
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice.get("finish_reason")
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("role"), str):
                state["role"] = delta["role"]
            _merge_text(content_parts, index, delta.get("content"))
            _merge_text(reasoning_parts, index, delta.get("reasoning_content"))
            _merge_text(typo_reasoning_parts, index, delta.get("reasoning_conetent"))

            deltas = delta.get("tool_calls")
            if isinstance(deltas, list):
                for item in deltas:
                    if not isinstance(item, dict):
                        continue
                    call_index = int(item.get("index", 0))
                    call = tool_calls.setdefault(index, {}).setdefault(
                        call_index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    for key in ("id", "type"):
                        if isinstance(item.get(key), str) and item[key]:
                            call[key] = item[key]
                    function = item.get("function") or {}
                    if isinstance(function, dict):
                        if isinstance(function.get("name"), str):
                            call["function"]["name"] += function["name"]
                        if isinstance(function.get("arguments"), str):
                            call["function"]["arguments"] += function["arguments"]

            legacy_function = delta.get("function_call")
            if isinstance(legacy_function, dict):
                call = tool_calls.setdefault(index, {}).setdefault(
                    0,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if isinstance(legacy_function.get("name"), str):
                    call["function"]["name"] += legacy_function["name"]
                if isinstance(legacy_function.get("arguments"), str):
                    call["function"]["arguments"] += legacy_function["arguments"]

            partial_content = "".join(content_parts.get(0, []))
        if chunk.get("object") == "chat.completion" and chunk.get("choices"):
            saw_done = True

    try:
        while True:
            line = response.readline()
            if not line:
                break
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            line = line.strip()
            if not line or line.startswith(":") or line.startswith("event:"):
                continue
            if line.startswith("data:"):
                saw_sse = True
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    continue
                try:
                    consume(json.loads(data))
                except json.JSONDecodeError:
                    # Ignore non-JSON SSE metadata, but do not hide malformed
                    # JSON when the provider claims to be an SSE endpoint.
                    continue
            elif first_nonempty:
                first_nonempty = False
                # A provider may ignore stream=true and return ordinary JSON.
                remainder = line + "\n" + response.read().decode("utf-8", "replace")
                return json.loads(remainder)
    except Exception as exc:
        raise StreamReadError(exc, _assemble(base, choices, content_parts, reasoning_parts, typo_reasoning_parts, tool_calls), partial_content) from exc

    if not saw_sse:
        return _assemble(base, choices, content_parts, reasoning_parts, typo_reasoning_parts, tool_calls)
    return _assemble(base, choices, content_parts, reasoning_parts, typo_reasoning_parts, tool_calls)


def _assemble(base, choices, content_parts, reasoning_parts, typo_reasoning_parts, tool_calls) -> dict:
    result = dict(base)
    assembled = []
    for index in sorted(choices):
        state = choices[index]
        message = {"role": state.get("role", "assistant"), "content": "".join(content_parts.get(index, []))}
        reasoning = "".join(reasoning_parts.get(index, []))
        typo_reasoning = "".join(typo_reasoning_parts.get(index, []))
        if reasoning:
            message["reasoning_content"] = reasoning
        if typo_reasoning:
            message["reasoning_conetent"] = typo_reasoning
        if index in tool_calls:
            message["tool_calls"] = [tool_calls[index][key] for key in sorted(tool_calls[index])]
        assembled.append({"index": index, "message": message, "finish_reason": state.get("finish_reason")})
    result["choices"] = assembled
    result["stream"] = True
    return result

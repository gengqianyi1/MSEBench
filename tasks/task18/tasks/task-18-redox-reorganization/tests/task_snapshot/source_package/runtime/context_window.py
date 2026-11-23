"""Deterministic bounded request views over an immutable full conversation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

REQUEST_CONTEXT_MAX_CHARS = 80_000
LARGE_MESSAGE_CHARS = 24_000


@dataclass(frozen=True)
class ContextWindowStats:
    full_chars: int
    request_chars: int
    omitted_first_turn: int | None
    omitted_last_turn: int | None
    omitted_blocks: int
    retained_blocks: int


@dataclass(frozen=True)
class _Block:
    turn: int | None
    messages: list[dict[str, object]]


def _serialized_chars(messages: list[dict[str, object]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _placeholder(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"[{kind} omitted from rolling API request; chars={len(value)}; sha256={digest}]"


def _compact_message(message: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(message)
    if result.get("role") != "assistant":
        return result
    content = result.get("content")
    if isinstance(content, str) and len(content) > LARGE_MESSAGE_CHARS:
        result["content"] = _placeholder("assistant content", content)
    return result


def _compact_write_calls(message: dict[str, object]) -> None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != "write_file":
            continue
        raw_arguments = function.get("arguments")
        arguments: dict[str, object] | None = None
        encoded_as_json = False
        if isinstance(raw_arguments, dict):
            arguments = copy.deepcopy(raw_arguments)
        elif isinstance(raw_arguments, str):
            try:
                loaded = json.loads(raw_arguments)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                arguments = loaded
                encoded_as_json = True
        if arguments is None or not isinstance(arguments.get("content"), str):
            continue
        arguments["content"] = _placeholder(
            "write_file content", str(arguments["content"])
        )
        function["arguments"] = (
            json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            if encoded_as_json
            else arguments
        )


def _split_messages(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[_Block]]:
    if len(messages) < 2 or messages[0].get("role") != "system":
        raise ValueError("conversation must begin with a system message")
    if messages[1].get("role") != "user":
        raise ValueError("conversation must include the initial user message second")
    pinned = [_compact_message(messages[0]), _compact_message(messages[1])]
    blocks: list[_Block] = []
    turn = 0
    index = 2
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            raise ValueError("orphan tool result in conversation")
        if role != "assistant":
            blocks.append(_Block(turn=None, messages=[_compact_message(message)]))
            index += 1
            continue

        turn += 1
        block = [_compact_message(message)]
        calls = message.get("tool_calls")
        call_ids = []
        if isinstance(calls, list):
            call_ids = [
                str(call.get("id"))
                for call in calls
                if isinstance(call, dict) and call.get("id") is not None
            ]
        index += 1
        for expected_id in call_ids:
            if index >= len(messages) or messages[index].get("role") != "tool":
                raise ValueError("assistant tool call is missing its result")
            tool_message = messages[index]
            if str(tool_message.get("tool_call_id")) != expected_id:
                raise ValueError("tool result does not match assistant tool call")
            block.append(_compact_message(tool_message))
            index += 1
        while index < len(messages) and messages[index].get("role") == "user":
            block.append(_compact_message(messages[index]))
            index += 1
        blocks.append(_Block(turn=turn, messages=block))
    return pinned, blocks


def _request(
    pinned: list[dict[str, object]],
    blocks: list[_Block],
    retained: int,
) -> list[dict[str, object]]:
    selected = blocks[len(blocks) - retained :] if retained else []
    result = copy.deepcopy(pinned)
    omitted = blocks[: len(blocks) - retained] if retained else blocks
    if omitted:
        turns = [block.turn for block in omitted if block.turn is not None]
        first = min(turns) if turns else None
        last = max(turns) if turns else None
        range_text = "unnumbered"
        if first is not None and last is not None:
            range_text = str(first) if first == last else f"{first}-{last}"
        notice = (
            f"Rolling context omitted {len(omitted)} completed interaction blocks "
            f"(turns {range_text}). Full audit logs and workspace files remain authoritative."
        )
        result[0]["content"] = f"{result[0].get('content') or ''}\n\n{notice}"
    for block in selected:
        result.extend(copy.deepcopy(block.messages))
    return result


def _compact_block(block: _Block) -> _Block:
    compacted = copy.deepcopy(block.messages)
    for message in compacted:
        if message.get("role") == "assistant":
            _compact_write_calls(message)
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _placeholder("tool result", content)
    return _Block(turn=block.turn, messages=compacted)


def build_request_view(
    messages: list[dict[str, object]],
    *,
    max_chars: int = REQUEST_CONTEXT_MAX_CHARS,
) -> tuple[list[dict[str, object]], ContextWindowStats]:
    """Return a bounded API request copy while leaving the full transcript untouched."""
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    full_chars = _serialized_chars(messages)
    pinned, blocks = _split_messages(messages)

    selected_view = _request(pinned, blocks, 0)
    retained_count = 0
    if _serialized_chars(selected_view) > max_chars:
        raise ValueError("system and initial user messages exceed the request context limit")
    for retained in range(1, len(blocks) + 1):
        candidate = _request(pinned, blocks, retained)
        if _serialized_chars(candidate) <= max_chars:
            selected_view = candidate
            retained_count = retained
        else:
            break

    if retained_count == 0 and blocks:
        compacted_blocks = list(blocks)
        compacted_blocks[-1] = _compact_block(compacted_blocks[-1])
        candidate = _request(pinned, compacted_blocks, 1)
        if _serialized_chars(candidate) <= max_chars:
            selected_view = candidate
            retained_count = 1

    request_chars = _serialized_chars(selected_view)
    omitted = blocks[: len(blocks) - retained_count] if retained_count else blocks
    omitted_turns = [block.turn for block in omitted if block.turn is not None]
    stats = ContextWindowStats(
        full_chars=full_chars,
        request_chars=request_chars,
        omitted_first_turn=min(omitted_turns) if omitted_turns else None,
        omitted_last_turn=max(omitted_turns) if omitted_turns else None,
        omitted_blocks=len(omitted),
        retained_blocks=retained_count,
    )
    return selected_view, stats

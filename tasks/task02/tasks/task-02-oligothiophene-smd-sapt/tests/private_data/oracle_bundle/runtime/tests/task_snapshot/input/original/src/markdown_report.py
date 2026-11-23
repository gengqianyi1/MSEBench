from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _message_key(index: int) -> str:
    return f"message{index}"


def normalize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": msg.get("role", "")}
    if msg.get("content") is not None:
        out["content"] = msg.get("content")
    if msg.get("tool_calls") is not None:
        out["tool_calls"] = msg.get("tool_calls")
    if msg.get("tool_call_id") is not None:
        out["tool_call_id"] = msg.get("tool_call_id")
    if msg.get("name") is not None:
        out["name"] = msg.get("name")
    return out


def write_conversation_markdown(path: Path, messages: List[Dict[str, Any]]) -> None:
    data = {
        _message_key(i): normalize_message(msg)
        for i, msg in enumerate(messages, start=1)
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


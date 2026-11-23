from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _message_key(index: int) -> str:
    return f"message{index}"


def write_conversation_markdown(path: Path, messages: List[Dict[str, str]]) -> None:
    data = {
        _message_key(i): {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
        }
        for i, msg in enumerate(messages, start=1)
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize_markdown_report(path: Path, *, title: str, task_cfg: Dict[str, Any], api_cfg: Dict[str, Any]) -> None:
    write_conversation_markdown(path, [])


def append_round_to_markdown_report(
    path: Path,
    *,
    round_idx: int,
    messages_input_to_llm: List[Dict[str, str]],
    llm_result: Any,
    raw_response: str,
    parsed_candidate: Optional[Dict[str, Any]],
    parse_error: Optional[str],
    evaluation: Any,
    feedback: Dict[str, Any],
    best_so_far: Any,
    tool_interactions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    write_conversation_markdown(path, messages_input_to_llm)


def append_final_report_to_markdown(path: Path, final_report: Dict[str, Any]) -> None:
    return None

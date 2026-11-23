from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def write_conversation_markdown(path: Path, messages: List[Dict[str, str]]) -> None:
    lines = ["# Conversation", ""]
    for index, msg in enumerate(messages, start=1):
        role = str(msg.get("role", "message")).title()
        content = str(msg.get("content", ""))
        lines.append(f"## {index}. {role}")
        lines.append("")
        lines.append(content)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def initialize_markdown_report(path: Path, *, title: str, task_cfg: Dict[str, Any], api_cfg: Dict[str, Any]) -> None:
    path.write_text(
        f"# {title}\n\n"
        "## Task configuration\n\n```yaml\n"
        f"{json.dumps(task_cfg, ensure_ascii=False, indent=2)}\n```\n\n"
        "## API configuration\n\n```json\n"
        f"{json.dumps(api_cfg, ensure_ascii=False, indent=2)}\n```\n\n",
        encoding="utf-8",
    )


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
) -> None:
    lines = [
        "",
        f"## Round {round_idx}",
        "",
        "### Input messages",
        "",
    ]
    for message in messages_input_to_llm:
        lines.append(f"### {str(message.get('role', 'message')).title()}")
        lines.append("")
        lines.append(str(message.get("content", "")))
        lines.append("")
    lines.extend([
        "### Raw response",
        "",
        "```text",
        str(raw_response),
        "```",
        "",
        "### Parsed candidate",
        "",
        "```json",
        json.dumps(parsed_candidate, ensure_ascii=False, indent=2) if parsed_candidate is not None else "null",
        "```",
        "",
        "### Feedback",
        "",
        "```json",
        json.dumps(feedback, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "### Best so far",
        "",
        "```json",
        json.dumps(best_so_far, ensure_ascii=False, indent=2, default=str) if best_so_far is not None else "null",
        "```",
        "",
    ])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def append_final_report_to_markdown(path: Path, final_report: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Final report\n\n```json\n")
        handle.write(json.dumps(final_report, ensure_ascii=False, indent=2, default=str))
        handle.write("\n```\n")

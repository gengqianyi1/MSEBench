from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI


RUN_PYTHON_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a complete Python script inside the benchmark workspace. "
            "Use this tool to inspect files, calculate pymatgen/matminer/JARVIS features, "
            "train scikit-learn or installed PyTorch CPU models, analyze intermediate data, "
            "and write result.json."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Complete Python script to execute.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum runtime for this script in seconds.",
                },
            },
            "required": ["script"],
        },
    },
}

CHGNET_RELAX_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "relax_selected_candidates_chgnet",
        "description": (
            "Use the benchmark's single allowed CHGNet relaxation opportunity on exactly five "
            "distinct candidate IDs already written to candidate_library.json.gz. The protocol "
            "and optimizer settings are fixed. Additional attempts are recorded and rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_ids": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "pattern": "^CAND-[0-9]{6}$",
                    },
                }
            },
            "required": ["candidate_ids"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class LLMResult:
    ok: bool
    message: Dict[str, Any]
    raw_response: Dict[str, Any]
    attempts: int
    error: str | None = None


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    return dict(message)


def call_llm(messages: List[Dict[str, Any]], api_cfg: Dict[str, Any]) -> LLMResult:
    api_key = api_cfg.get("api_key")
    if api_cfg.get("api_key_env"):
        api_key = os.environ.get(str(api_cfg["api_key_env"]))
    if not api_key:
        raise ValueError("LLM api key is missing; set api_key or api_key_env in the API config")
    client = OpenAI(
        api_key=api_key,
        base_url=api_cfg.get("base_url"),
        timeout=api_cfg.get("timeout", 300),
    )
    max_retries = int(api_cfg.get("max_retries", 2))
    delay = float(api_cfg.get("retry_delay_seconds", 2))
    last_error: str | None = None

    for attempt in range(1, max_retries + 2):
        try:
            response = client.chat.completions.create(
                model=api_cfg["model"],
                messages=messages,
                tools=[RUN_PYTHON_TOOL, CHGNET_RELAX_TOOL],
                tool_choice="auto",
                temperature=api_cfg.get("temperature", 0.2),
                top_p=api_cfg.get("top_p", 1.0),
                max_tokens=api_cfg.get("max_tokens", 32000),
            )
            raw = response.model_dump(exclude_none=True) if hasattr(response, "model_dump") else response
            message = _message_to_dict(response.choices[0].message)
            return LLMResult(ok=True, message=message, raw_response=raw, attempts=attempt)
        except Exception as exc:
            last_error = str(exc)
            if attempt <= max_retries:
                time.sleep(delay)

    return LLMResult(ok=False, message={"role": "assistant", "content": ""}, raw_response={}, attempts=max_retries + 1, error=last_error)

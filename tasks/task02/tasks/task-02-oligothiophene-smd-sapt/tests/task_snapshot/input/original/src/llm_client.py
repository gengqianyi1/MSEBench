from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List


RUN_PYTHON_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a complete Python script inside the benchmark workspace. "
            "Use this tool to inspect files, run RDKit directly, call xTB/PySCF/Psi4 only through benchmark_tools.py, "
            "analyze evidence, validate stage JSON files, and write final outputs. "
            "Rejected scripts still consume tool budget. The runner may raise undersized requested timeouts to the fixed class floor."
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
                    "description": "Requested maximum runtime. The runner enforces a protocol-specific minimum timeout floor and the global wall-time cap.",
                },
            },
            "required": ["script"],
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
    from openai import OpenAI

    client = OpenAI(
        api_key=api_cfg["api_key"],
        base_url=api_cfg.get("base_url"),
        timeout=api_cfg.get("timeout", 300),
    )
    max_retries = int(api_cfg.get("max_retries", 2))
    delay = float(api_cfg.get("retry_delay_seconds", 2))
    last_error: str | None = None

    for attempt in range(1, max_retries + 2):
        try:
            request: Dict[str, Any] = {
                "model": api_cfg["model"],
                "messages": messages,
                "tools": [RUN_PYTHON_TOOL],
                "tool_choice": "auto",
                "temperature": api_cfg.get("temperature", 0),
                "max_tokens": api_cfg.get("max_tokens", 32000),
            }
            for optional_key in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
                if optional_key in api_cfg:
                    request[optional_key] = api_cfg[optional_key]
            response = client.chat.completions.create(
                **request,
            )
            raw = response.model_dump(exclude_none=True) if hasattr(response, "model_dump") else response
            message = _message_to_dict(response.choices[0].message)
            return LLMResult(ok=True, message=message, raw_response=raw, attempts=attempt)
        except Exception as exc:
            last_error = str(exc)
            if attempt <= max_retries:
                time.sleep(delay)

    return LLMResult(ok=False, message={"role": "assistant", "content": ""}, raw_response={}, attempts=max_retries + 1, error=last_error)

"""One-shot, no-tool agent for the versioned TextQA benchmark.

The model receives exactly the task instruction.  It is not offered shell,
file, browser, Python, MCP, sidecar, or any task-specific tool.  The sole
environment operation below is performed by the harness after the response is
received, to persist the submission for the verifier.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from harbor.agents.base import BaseAgent


class TextQAAgent(BaseAgent):
    @staticmethod
    def name():
        return "textqa-no-tools-agent"

    @staticmethod
    def version():
        return "1.0.0"

    async def setup(self, environment):
        return None

    @staticmethod
    def _endpoint() -> str:
        explicit = os.getenv("TEXTQA_API_URL")
        if explicit:
            return explicit
        base = os.getenv("TEXTQA_API_BASE") or os.getenv("CHEMBENCH_API_BASE") or os.getenv("OPENAI_BASE_URL")
        if not base:
            raise RuntimeError("TEXTQA_API_BASE (or TEXTQA_API_URL) is required")
        return base.rstrip("/") + "/chat/completions"

    @staticmethod
    def _extract_content(payload: dict) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("provider response has no choices[0].message.content") from exc
        if not isinstance(content, str):
            raise RuntimeError("provider response content is not text")
        return content

    @staticmethod
    def _parse_answer(raw: str) -> tuple[str, str | None]:
        """Accept the declared JSON contract, with a narrow labelled fallback."""
        text = raw.strip()
        candidate = text
        if not text.startswith("{"):
            left, right = text.find("{"), text.rfind("}")
            if left >= 0 and right > left:
                candidate = text[left : right + 1]
        try:
            value = json.loads(candidate)
            if isinstance(value, dict) and isinstance(value.get("answer"), str):
                reasoning = value.get("reasoning")
                return value["answer"].strip(), reasoning if isinstance(reasoning, str) else None
        except json.JSONDecodeError:
            pass
        labelled = re.search(r"(?:\[答案\]|答案\s*[:：])\s*([^\n\r]+)", text)
        if labelled:
            return labelled.group(1).strip(), None
        # Keep the full response only as an auditable invalid candidate.  The
        # private grader decides validity; do not mine answer letters from
        # chain-of-thought-like prose.
        return text, None

    @staticmethod
    def _record_provider_call(record: dict) -> None:
        target = os.getenv("TEXTQA_LLM_CALL_LOG")
        if not target:
            return
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")

    def _call(self, instruction: str) -> str:
        key = os.getenv("TEXTQA_API_KEY") or os.getenv("CHEMBENCH_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("TEXTQA_API_KEY (or CHEMBENCH_API_KEY) is required")
        model = os.getenv("TEXTQA_MODEL_ID") or os.getenv("CHEMBENCH_MODEL_ID")
        if not model:
            raise RuntimeError("TEXTQA_MODEL_ID (or CHEMBENCH_MODEL_ID) is required")
        body = {
            "model": model,
            "temperature": float(os.getenv("TEXTQA_TEMPERATURE", "0")),
            "max_tokens": int(os.getenv("TEXTQA_MAX_TOKENS", "4096")),
            "messages": [
                {
                    "role": "system",
                    "content": "You answer a closed-book multiple-choice benchmark. No tools are available. Return exactly one JSON object with string keys answer and reasoning. answer must contain only option letters in the requested format; do not include any other prose in answer.",
                },
                {"role": "user", "content": instruction},
            ],
        }
        request_id = str(uuid.uuid4())
        started = time.time()
        req = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(os.getenv("TEXTQA_REQUEST_TIMEOUT", "300"))) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self._record_provider_call({"request_id": request_id, "endpoint": self._endpoint(), "elapsed_seconds": round(time.time() - started, 6), "request": body, "http_status": 200, "response": payload})
            return self._extract_content(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            self._record_provider_call({"request_id": request_id, "endpoint": self._endpoint(), "elapsed_seconds": round(time.time() - started, 6), "request": body, "http_status": exc.code, "error": detail})
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc

    async def run(self, instruction, environment, context):
        raw = self._call(instruction)
        answer, reasoning = self._parse_answer(raw)
        submission = {"schema_version": "textqa-submission/1.0", "answer": answer, "raw_response": raw}
        if reasoning is not None:
            submission["reasoning"] = reasoning
        encoded = base64.b64encode(json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8")).decode("ascii")
        result = await environment.exec(
            command=f"python -c \"import base64; open('/workspace/final_answer.json','wb').write(base64.b64decode('{encoded}'))\"",
            cwd="/workspace",
            timeout_sec=30,
        )
        if getattr(result, "return_code", 0) != 0:
            raise RuntimeError("could not persist TextQA submission")
        if hasattr(context, "metadata"):
            context.metadata = {"tool_calls": 0, "model_calls": 1, "submission_written": True}
        return None

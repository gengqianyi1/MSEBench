"""Organizer-only deterministic Oracle Agent for Tasks 19/20 Harbor E2E checks."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from harbor.agents.base import BaseAgent


CALLS = {
    "task-19-active-impurity-elucidation": [
        ("energy_resolved_msms", {"energy_eV": 45}),
        ("correlation_window", {"window": "aromatic"}),
        ("correlation_window", {"window": "sidechain"}),
        ("condition_perturbation", {"condition": "pH4"}),
    ],
    "task-20-catalyst-deactivation-diagnosis": [
        ("tpo", {"sample": "spent"}),
        ("bet", {"sample": "spent"}),
        ("xps", {"region": "S2p"}),
        ("particle_size_test", {"mode": "tem"}),
    ],
}


class Task1920OracleAgent(BaseAgent):
    @staticmethod
    def name():
        return "task19-20-oracle"

    @staticmethod
    def version():
        return "1.0.0"

    def __init__(self, *args, extra_env=None, **kwargs):
        super().__init__(*args, extra_env=extra_env, **kwargs)
        self.settings = dict(extra_env or {})

    def _get(self, name):
        value = self.settings.get(name) or os.environ.get(name)
        if not value:
            raise RuntimeError(f"missing organizer setting: {name}")
        return value

    @staticmethod
    def _post(url, payload):
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError(f"sidecar rejected Oracle action: {result!r}")
        return result

    @staticmethod
    async def _write(environment, path, value):
        encoded = base64.b64encode(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()
        ).decode()
        parent = str(Path(path).parent).replace("\\", "/")
        for offset in range(0, len(encoded), 6000):
            chunk = encoded[offset:offset + 6000]
            mode = "wb" if offset == 0 else "ab"
            prefix = f"from pathlib import Path; Path('{parent}').mkdir(parents=True,exist_ok=True); " if offset == 0 else ""
            result = await environment.exec(
                command=f"python -c \"import base64; {prefix}open('{path}','{mode}').write(base64.b64decode('{chunk}'))\"",
                cwd="/workspace", timeout_sec=30,
            )
            if getattr(result, "return_code", 1) != 0:
                raise RuntimeError(f"failed to write {path}")

    async def setup(self, environment):
        del environment

    async def run(self, instruction, environment, context):
        del instruction
        task_id = self._get("CHEMBENCH_ORACLE_TASK_ID")
        base = self._get("CHEMBENCH_ORACLE_SIDECAR_URL").rstrip("/")
        candidate = json.loads(Path(self._get("CHEMBENCH_ORACLE_CANDIDATE")).read_text(encoding="utf-8"))
        started = self._post(base + "/session", {})
        events = []
        for tool, arguments in CALLS[task_id]:
            response = self._post(base + "/tool", {"session_id": started["session_id"], "tool": tool, "arguments": arguments})
            events.append({"tool": tool, "arguments": arguments, "response": response})
        submitted = self._post(base + "/submit", {"session_id": started["session_id"], "candidate": candidate})
        canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        artifact = {
            "schema_version": "chembench-organizer-evidence/1.0",
            "task_id": task_id,
            "case_id": started["case"]["case_id"],
            "session_id": started["session_id"],
            "evidence_id": submitted["evidence_id"],
            "attestation": submitted["attestation"],
            "candidate": candidate,
            "candidate_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "tool_events": events,
            "public_case": started["case"],
            "submission": submitted,
        }
        await self._write(environment, "/workspace/cases/case-1/case_result.json", artifact)
        await self._write(environment, "/workspace/final_answer.json", {"candidate": candidate})
        if hasattr(context, "metadata"):
            context.metadata = {"oracle": True, "task_id": task_id, "tool_calls": len(events)}

#!/usr/bin/env python3
"""Offline contract test for the restricted Task 15 Harbor protocol-Agent branch."""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agents.chembench_agent import MSEBenchAgent

TASK = ROOT / "tasks/task-15-lipophilic-sulfonamide-edit-budget"


class Result:
    def __init__(self, stdout="", stderr="", return_code=0):
        self.stdout, self.stderr, self.return_code = stdout, stderr, return_code


class FakeEnvironment:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.commands = []

    def local(self, value: str) -> Path:
        assert value.startswith("/workspace")
        return self.workspace / value.removeprefix("/workspace").lstrip("/")

    async def exec(self, command, cwd=None, timeout_sec=None):
        self.commands.append(command)
        if command == "cat /workspace/.chembench_agent.json":
            return Result((self.workspace / ".chembench_agent.json").read_text(encoding="utf-8"))
        if command.startswith("python /workspace/chembench_protocol/run_tools.py") or command.startswith("python /workspace/chembench_protocol/run_public_tools.py"):
            local_command = command.replace("/workspace/chembench_protocol/run_tools.py",str(self.workspace / "chembench_protocol/run_tools.py"))
            local_command = local_command.replace("/workspace/chembench_protocol/run_public_tools.py",str(self.workspace / "chembench_protocol/run_public_tools.py"))
            completed = subprocess.run(local_command, shell=True, text=True, capture_output=True)
            return Result(completed.stdout, completed.stderr, completed.returncode)
        match = re.search(r"base64\.b64decode\('([^']+)'\)", command)
        if not match:
            return Result(stderr="unexpected command", return_code=1)
        decoded = base64.b64decode(match.group(1))
        source = decoded.decode("utf-8", errors="ignore")
        path_match = re.search(r"Path\('(/workspace[^']+)'\)", source)
        if path_match:
            path = self.local(path_match.group(1))
            if not path.is_file(): return Result(stderr="missing", return_code=1)
            return Result(path.read_text(encoding="utf-8"))
        open_match = re.search(r"open\('(/workspace[^']+)'", command)
        if open_match:
            path = self.local(open_match.group(1)); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(decoded); return Result()
        return Result(stderr="unrecognized fixed command", return_code=1)


class Task15ProtocolEnvironment(FakeEnvironment):
    """Deterministic public-tool bridge for the Agent contract regression.

    The actual task executes ``run_tools.py`` inside its task Docker image,
    which contains RDKit.  This offline Agent test is intentionally host-side
    and must test quota/accounting without making the host Python installation
    a hidden dependency.  It therefore emulates the *public bridge envelope*,
    not the scientific tool implementation.
    """

    async def exec(self, command, cwd=None, timeout_sec=None):
        if command.startswith("python /workspace/chembench_protocol/run_tools.py"):
            self.commands.append(command)
            payload_match = re.search(r"--payload\s+([^\s]+)", command)
            used_match = re.search(r"--used\s+(\d+)", command)
            if not payload_match or not used_match:
                return Result(stderr="malformed bridge invocation", return_code=2)
            try:
                calls = json.loads(base64.b64decode(payload_match.group(1)).decode("utf-8"))
            except Exception:
                return Result(stderr="malformed bridge payload", return_code=2)
            used = int(used_match.group(1))
            accepted = max(0, min(len(calls), 6 - used))
            responses = [
                {"ok": True, "tool": call.get("tool"), "result": {"fixture": True}}
                for call in calls[:accepted]
                if isinstance(call, dict)
            ]
            responses.extend(
                {"ok": False, "error": {"code": "quota_exceeded"}}
                for _ in calls[accepted:]
            )
            return Result(json.dumps({"responses": responses, "calls_used": used + accepted}))
        return await super().exec(command, cwd=cwd, timeout_sec=timeout_sec)


async def check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(TASK / "environment/public", workspace)
        environment = Task15ProtocolEnvironment(workspace)
        agent = object.__new__(MSEBenchAgent)
        calls = [{"tool": "similarity_to_start", "arguments": {"smiles": "c1ccccc1"}}] * 7
        replies = iter([
            json.dumps({"tool_calls": calls}),
            json.dumps({"tool_calls": calls[:1]}),
            json.dumps({"candidate": {"name": "fixture", "smiles": "CCO", "rationale": "test"}}),
        ])
        agent._call = lambda *args, **kwargs: next(replies)
        context = SimpleNamespace(metadata={})
        await agent.run("Submit one candidate for round 1.", environment, context)
        candidate = json.loads((workspace / "rounds/round-1/candidate.json").read_text(encoding="utf-8"))
        state = json.loads((workspace / "protocol_state.json").read_text(encoding="utf-8"))
        assert candidate["candidate"]["smiles"] == "CCO"
        assert context.metadata["tool_calls"] == 6, {"metadata": context.metadata, "state": state}
        assert context.metadata["tool_messages"] == 2, {"metadata": context.metadata, "state": state}
        assert state["task15"]["calls_used"] == 6, {"metadata": context.metadata, "state": state}
        assert not any("CCO" in command and "run_tools" in command for command in environment.commands)
        print(json.dumps({"status": "PASS", "tool_calls": 6, "tool_messages": 2, "candidate_written": True}))


if __name__ == "__main__":
    asyncio.run(check())

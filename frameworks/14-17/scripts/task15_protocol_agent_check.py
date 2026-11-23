#!/usr/bin/env python3
"""Offline contract test for the restricted Task 15 Harbor protocol-Agent branch."""
from __future__ import annotations
import os as _os, sys as _sys  # noqa: E401
# Never write __pycache__ into tasks/: this script loads task code, and bytecode dropped there
# changes the task's Harbor checksum, so datasets/*/dataset.toml would stop matching the tree.
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

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
from agents.chembench_agent import ChemBenchAgent

TASK = ROOT / "tasks/task-15-lipophilic-sulfonamide-edit-budget"


class Result:
    def __init__(self, stdout="", stderr="", return_code=0):
        self.stdout, self.stderr, self.return_code = stdout, stderr, return_code


ARG_STRLEN_MAX = 131072  # Linux MAX_ARG_STRLEN: Harbor passes each exec command to `bash -c` as ONE argument


class FakeEnvironment:
    """Runs the agent's container commands for real, against a local copy of /workspace.

    It used to recognise command shapes and emulate them -- creating parent directories for every
    write and accepting any command length. That emulation is how agent defects passed this check
    while the organizer's real Harbor LLM run failed on all four tasks (2026-09-16): a write into a
    directory that did not exist, and saved state larger than the 128 KB argument limit. Commands now
    execute with /workspace mapped to the copy, under the same argument limit, and return their real
    exit code. It still shares the host network, so the third defect -- a sidecar call made from the
    runner instead of the task container -- is only caught end to end, by `./chembench llm-mock`.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.commands = []

    def local(self, value: str) -> Path:
        assert value.startswith("/workspace")
        return self.workspace / value.removeprefix("/workspace").lstrip("/")

    async def exec(self, command, cwd=None, timeout_sec=None, env=None, user=None):
        self.commands.append(command)
        if len(command.encode("utf-8")) >= ARG_STRLEN_MAX:
            return Result(stderr="bash: Argument list too long", return_code=126)
        root = str(self.workspace)
        inline = re.fullmatch(r"python -c \"import base64; exec\(base64\.b64decode\('([A-Za-z0-9+/=]+)'\)\)\"", command)
        if inline:
            source = base64.b64decode(inline.group(1)).decode("utf-8").replace("/workspace", root)
            argv = [sys.executable, "-c", source]
        else:
            argv = ["bash", "-c", command.replace("/workspace", root)]
        environ = dict(_os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PATH=str(Path(sys.executable).parent) + _os.pathsep + _os.environ.get("PATH", ""))
        done = subprocess.run(argv, capture_output=True, text=True, cwd=root, env=environ, timeout=timeout_sec)
        return Result(done.stdout, done.stderr, done.returncode)


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
        agent = object.__new__(ChemBenchAgent)
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

    # Round 2 with the REAL step instruction, which names round 1 before round 2. The agent used to
    # take the first number, overwrite round 1 and save last_round=1; the organizer's DeepSeek run then
    # stopped at round 3 because round 2 had no candidate (2026-09-16).
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(TASK / "environment/public", workspace)
        environment = Task15ProtocolEnvironment(workspace)
        agent = object.__new__(ChemBenchAgent)
        answers = {n: {"name": f"r{n}", "smiles": "CCO" if n == 1 else "CCN", "rationale": "test"} for n in (1, 2)}
        for n in (1, 2):
            agent._call = lambda *args, _n=n, **kwargs: json.dumps({"candidate": answers[_n]})
            instruction = (TASK / f"steps/round-{n}/instruction.md").read_text(encoding="utf-8")
            await agent.run(instruction, environment, SimpleNamespace(metadata={}))
        state = json.loads((workspace / "protocol_state.json").read_text(encoding="utf-8"))
        assert state["task15"]["last_round"] == 2, state["task15"]
        for n in (1, 2):
            saved = json.loads((workspace / f"rounds/round-{n}/candidate.json").read_text(encoding="utf-8"))
            assert saved["candidate"]["smiles"] == answers[n]["smiles"], (n, saved)
        print(json.dumps({"status": "PASS", "tool_calls": 6, "tool_messages": 2, "candidate_written": True,
                          "real_instruction_rounds_saved": [1, 2]}))


if __name__ == "__main__":
    asyncio.run(check())

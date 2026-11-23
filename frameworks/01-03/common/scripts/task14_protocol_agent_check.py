#!/usr/bin/env python3
"""Offline contract test for Task 14's candidate-only Harbor protocol Agent."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

TASK = ROOT / "tasks/task-14-multiobjective-mol-design"


async def check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(TASK / "environment/public", workspace)
        environment = FakeEnvironment(workspace)
        agent = object.__new__(MSEBenchAgent)
        candidate = {"name": "fixture", "smiles": "CCO", "rationale": "test"}
        agent._call = lambda *args, **kwargs: json.dumps({"candidate": candidate})
        context = SimpleNamespace(metadata={})
        await agent.run("Submit one candidate for round 1.", environment, context)
        actual = json.loads((workspace / "rounds/round-1/candidate.json").read_text(encoding="utf-8"))
        assert actual == {"candidate": candidate}
        assert context.metadata == {"round": 1, "candidate_submitted": True, "failure": None}
        model_commands = [command for command in environment.commands if "round_tools" in command]
        assert not model_commands
        print(json.dumps({"status": "PASS", "model_calls": 1, "candidate_written": True, "arbitrary_commands": 0}))


if __name__ == "__main__":
    asyncio.run(check())

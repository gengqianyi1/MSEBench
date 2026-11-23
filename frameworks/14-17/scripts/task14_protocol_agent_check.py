#!/usr/bin/env python3
"""Offline contract test for Task 14's candidate-only Harbor protocol Agent."""
from __future__ import annotations
import os as _os, sys as _sys  # noqa: E401
# Never write __pycache__ into tasks/: this script loads task code, and bytecode dropped there
# changes the task's Harbor checksum, so datasets/*/dataset.toml would stop matching the tree.
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

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
from agents.chembench_agent import ChemBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

TASK = ROOT / "tasks/task-14-multiobjective-mol-design"


async def check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(TASK / "environment/public", workspace)
        environment = FakeEnvironment(workspace)
        agent = object.__new__(ChemBenchAgent)
        candidate = {"name": "fixture", "smiles": "CCO", "rationale": "test"}
        agent._call = lambda *args, **kwargs: json.dumps({"candidate": candidate})
        context = SimpleNamespace(metadata={})
        await agent.run("Submit one candidate for round 1.", environment, context)
        actual = json.loads((workspace / "rounds/round-1/candidate.json").read_text(encoding="utf-8"))
        assert actual == {"candidate": candidate}
        assert context.metadata == {"round": 1, "candidate_submitted": True, "failure": None}
        model_commands = [command for command in environment.commands if "round_tools" in command]
        assert not model_commands

    # A long-reasoning model makes the saved conversation larger than the 128 KB limit on a single
    # exec argument. The organizer's DeepSeek run lost round state this way (2026-09-16). Five such
    # rounds must still leave every candidate and the complete state behind.
    padding = "Working notes: weighing the descriptor windows before committing. " * 2400
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(TASK / "environment/public", workspace)
        environment = FakeEnvironment(workspace)
        agent = object.__new__(ChemBenchAgent)
        agent._call = lambda *args, **kwargs: padding + json.dumps({"candidate": candidate})
        # The REAL step instructions: rounds 2-5 name the previous round first, which made the agent
        # save round N over round N-1 (organizer DeepSeek run, 2026-09-16).
        for n in range(1, 6):
            instruction = (TASK / f"steps/round-{n}/instruction.md").read_text(encoding="utf-8")
            await agent.run(instruction, environment, SimpleNamespace(metadata={}))
        state = json.loads((workspace / "protocol_state.json").read_text(encoding="utf-8"))
        assert state["candidate_only"]["last_round"] == 5, state["candidate_only"]
        assert all((workspace / f"rounds/round-{n}/candidate.json").is_file() for n in range(1, 6))
        state_bytes = (workspace / "protocol_state.json").stat().st_size
        assert state_bytes > 131072, state_bytes
        print(json.dumps({"status": "PASS", "model_calls": 1, "candidate_written": True, "arbitrary_commands": 0,
                          "large_state_bytes": state_bytes, "large_state_rounds_saved": 5}))


if __name__ == "__main__":
    asyncio.run(check())

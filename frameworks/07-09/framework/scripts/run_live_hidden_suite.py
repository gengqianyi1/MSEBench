#!/usr/bin/env python3
"""Run one hidden-suite task with the configured OpenAI-compatible model.

Environment variables:
  CHEMBENCH_TASK_ID             task-01-... or task-02-...
  CHEMBENCH_WORKSPACE           destination workspace under parity/
  CHEMBENCH_MODEL_ID            evaluated model
  CHEMBENCH_API_KEY             API key (never written to disk)
  CHEMBENCH_API_BASE            base URL
  CHEMBENCH_SIDECAR_URL         optional sidecar URL override
  CHEMBENCH_MAX_TOKENS          optional
  CHEMBENCH_TEMPERATURE         optional
"""
from __future__ import annotations
import asyncio, json, os, shutil
from pathlib import Path
from types import SimpleNamespace

from agents.chembench_agent import MSEBenchAgent
from scripts.task15_protocol_agent_check import FakeEnvironment

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    task_id = os.environ["CHEMBENCH_TASK_ID"]
    task = ROOT / "tasks" / task_id
    output = ROOT / os.environ.get("CHEMBENCH_WORKSPACE", f"parity/{task_id}_model_workspace")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    shutil.copytree(task / "environment/public", output)
    environment = FakeEnvironment(output)
    agent = object.__new__(MSEBenchAgent)
    context = SimpleNamespace(metadata={})
    instruction = os.environ.get("CHEMBENCH_INSTRUCTION", "Solve protected hidden case 1.")
    await agent.run(instruction, environment, context)
    case_result = output / "cases/case-1/case_result.json"
    result = {}
    if case_result.exists():
        result["case_result"] = json.loads(case_result.read_text(encoding="utf-8"))
    result["metadata"] = getattr(context, "metadata", {})
    result["workspace"] = str(output)
    (output / "run_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    case = result.get("case_result") or {}
    submission = case.get("submission") or {}
    print(json.dumps({
        "status": "PASS" if submission.get("ok") else "FAIL",
        "tool_calls": result.get("metadata", {}).get("tool_calls"),
        "candidate_submitted": result.get("metadata", {}).get("candidate_submitted"),
        "failure": result.get("metadata", {}).get("failure"),
        "submission_ok": submission.get("ok"),
        "submission_status": submission.get("status"),
        "result_json_seen": result.get("metadata", {}).get("result_json_seen"),
        "workspace": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

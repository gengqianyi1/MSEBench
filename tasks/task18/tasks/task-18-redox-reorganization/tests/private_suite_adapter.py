"""Private Harbor sidecar adapter for Task 18.

The model receives only generic file and xTB primitives. Scientific planning,
calculation selection, interpretation, and conclusions remain model work.
"""
from __future__ import annotations

import json
from dataclasses import replace
import shutil
import sys
import tempfile
from pathlib import Path


class Adapter:
    case_count = 1

    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot)
        self.source = self.snapshot / "source_package"
        sys.path.insert(0, str(self.source))
        from runtime.staging import stage_model_workspace
        from runtime.task_adapter import build_task_runtime

        self.stage_model_workspace = stage_model_workspace
        self.build_task_runtime = build_task_runtime
        self.public = self.snapshot / "public"
        self.task_yaml = self.public / "task.yaml"
        self.tools = json.loads((self.public / "tools.json").read_text(encoding="utf-8"))["tools"]

    def public_case(self, index: int) -> dict:
        if index != 0:
            raise IndexError(index)
        return {
            "case_id": "task18-redox-reorganization",
            "task": {
                "id": "task_18_redox_reorganization",
                "instance_id": "task-18-redox-reorganization",
                "scientific_problem_id": "task_18_redox_reorganization",
                "version": "4.1.0",
                "question_path": "question.md",
            },
            "constraints": {
                "method": "GFN2-xTB",
                "phase": "gas",
                "neutral_charge": 0,
                "neutral_uhf": 0,
                "oxidized_charge": 1,
                "oxidized_uhf": 1,
                "scan_allowed": False,
                "failure_cost_is_retained": True,
            },
            "budget": {
                "max_agent_turns": 98,
                "wall_time_seconds": 10800,
                # Audit 2026-09-27: finalization wall clock raised 600 -> 1800
                # seconds (matches question.md and the 12600-second Harbor
                # agent timeout) so the finalization-only turns stay usable
                # for long-thinking models.
                "finalization_wall_time_seconds": 1800,
                "scientific_turns": 90,
                "finalization_turns": 8,
                "max_tool_calls": 360,
                "compute_credits": 56,
                "max_xtb_jobs": 32,
            },
            "output": {
                "submission_root": "submission",
                "required_files": ["final_answer.json", "report.md"],
                "schemas": ["schema/final_answer.schema.json"],
            },
            "public_tools": self.tools,
        }

    def start_session(self, evidence_dir, token: str) -> dict:
        temp = tempfile.TemporaryDirectory(prefix="chembench-task18-")
        run_root = Path(temp.name) / "run"
        workspace = self.stage_model_workspace(self.public, run_root)
        runtime = self.build_task_runtime(
            workspace,
            self.task_yaml,
            python_executable=Path(sys.executable),
            runtime_paths=(Path(sys.prefix), Path("/usr"), Path("/lib"), Path("/lib64")),
        )
        return {
            "temp": temp,
            "run_root": run_root,
            "runtime": runtime,
            "calls": 0,
            "token": token,
            "evidence_dir": Path(evidence_dir) if evidence_dir else None,
        }

    def compute_budget_exhausted(self, state: dict) -> bool:
        budget = state["runtime"].state.budget_status()
        return budget["main_remaining"] <= 0 or budget["xtb_jobs_remaining"] <= 0

    def call_tool(self, index: int, name, arguments, state: dict) -> dict:
        if index != 0:
            raise IndexError(index)
        state["calls"] += 1
        call_id = f"task18-{state['calls']:03d}"
        allowed = {tool["name"] for tool in self.tools} - {"submit"}
        if name not in allowed or not isinstance(arguments, dict):
            return {"ok": False, "tool": name, "call_id": call_id, "error": "INVALID_TOOL_REQUEST"}
        runtime = state["runtime"]
        old_config, old_timeout = runtime.task_config, runtime.state.timeout_seconds
        remaining = state.get("remaining_wall_seconds", old_timeout)
        runtime.task_config = replace(old_config, timeout_seconds=min(old_config.timeout_seconds, remaining))
        runtime.state.timeout_seconds = min(old_timeout, remaining)
        try:
            result = runtime.dispatch(name, arguments)
            return {"ok": True, "tool": name, "call_id": call_id, "result": result}
        except Exception as exc:
            return {"ok": False, "tool": name, "call_id": call_id, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            runtime.task_config, runtime.state.timeout_seconds = old_config, old_timeout

    def submit(self, index: int, candidate, tool_events, state: dict) -> dict:
        del tool_events
        if index != 0:
            raise IndexError(index)
        partial = isinstance(candidate, dict) and candidate.get("status") == "partial"
        try:
            result = (
                state["runtime"].finalize_at_turn_limit()
                if partial
                else state["runtime"].dispatch("submit", {})
            )
        except FileNotFoundError:
            partial = True
            result = state["runtime"].finalize_at_turn_limit()
        evidence_id = f"{state['token']}/task18-redox-reorganization"
        evidence_dir = state["evidence_dir"]
        if evidence_dir is not None:
            target = evidence_dir / evidence_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(state["run_root"], target / "run")
            (target / "sidecar_submission.json").write_text(
                json.dumps(
                    {"partial": partial, "runtime_submission": result},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "ok": True,
            "case_id": "task18-redox-reorganization",
            "status": "submitted",
            "feedback": {"accepted": True, "partial": partial},
            "cases_remaining": 0,
            "evidence_id": evidence_id,
        }


def create_adapter(snapshot: Path) -> Adapter:
    return Adapter(snapshot)

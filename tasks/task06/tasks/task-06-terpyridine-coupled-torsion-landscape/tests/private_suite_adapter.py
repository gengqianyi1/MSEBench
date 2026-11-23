"""Private Harbor sidecar adapter for Task 06.

The public boundary is the frozen seven-tool minimal runtime. It contains no
coordinate recommendation, basin identity, scan schedule, path selection or
coupling analysis.
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
            "case_id": "task06-terpyridine-coupled-torsion-landscape",
            "task": {"id": "task_06", "version": "1.16.0", "question_path": "question.md"},
            "constraints": {
                "method": "GFN2-xTB",
                "charge": 0,
                "uhf": 0,
                "environment": "gas",
                "single_threaded": True,
                "scan_allowed": True,
                "failure_cost_is_retained": True,
            },
            "budget": {
                "max_agent_turns": 63,
                "scientific_turns": 55,
                "finalization_turns": 8,
                "max_tool_calls": 240,
                "compute_credits": 12,
                "max_scan_points": 72,
                "operation_costs": {"sp": 1, "opt": 1, "hess": 1, "scan": 1},
            },
            "output": {
                "submission_root": "submission",
                "required_files": ["final_answer.json", "report.md"],
                "schemas": ["schema/final_answer.schema.json"],
            },
            "public_tools": self.tools,
        }

    def start_session(self, evidence_dir, token: str) -> dict:
        temp = tempfile.TemporaryDirectory(prefix="chembench-task06-")
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
        return state["runtime"].state.budget_status()["main_remaining"] <= 0

    def call_tool(self, index: int, name, arguments, state: dict) -> dict:
        if index != 0:
            raise IndexError(index)
        state["calls"] += 1
        call_id = f"task06-{state['calls']:03d}"
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
        evidence_id = f"{state['token']}/task06-terpyridine-coupled-torsion-landscape"
        evidence_dir = state["evidence_dir"]
        if evidence_dir is not None:
            target = evidence_dir / evidence_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(state["run_root"], target / "run")
            (target / "sidecar_submission.json").write_text(
                json.dumps({"partial": partial, "runtime_submission": result}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return {
            "ok": True,
            "case_id": "task06-terpyridine-coupled-torsion-landscape",
            "status": "submitted",
            "feedback": {"accepted": True, "partial": partial},
            "cases_remaining": 0,
            "evidence_id": evidence_id,
        }


def create_adapter(snapshot: Path) -> Adapter:
    return Adapter(snapshot)

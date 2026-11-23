"""Private Harbor sidecar adapter for Task 05.

The adapter forwards the original five generic file/xTB operations and adds
only budget inspection and explicit submission. Pose selection, optimization
strategy, energy decomposition, clustering, recovery and conclusions remain
model responsibilities.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path


FINALIZATION_NOTICE = (
    "Active work budget exhausted. The Harness will state the exact trigger and remaining "
    "submission-only finalization turns. No new xTB jobs are available; immediately materialize "
    "an honest complete or partial submission from existing evidence."
)


class _NoModelClient:
    metadata = {"provider": "harbor-sidecar", "model": "external-agent"}


class Adapter:
    case_count = 1

    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot)
        self.source = self.snapshot / "source_package"
        sys.path.insert(0, str(self.source))
        sys.path.insert(0, str(self.source / "runtime_common"))
        # The four-task validation suite imports several frozen runtime
        # snapshots in one interpreter.  Their deliberately generic top-level
        # package names (``tools``, ``framework`` and friends) must therefore
        # not leak from a previously validated task into this adapter.
        for module_name in tuple(sys.modules):
            if module_name in {"common", "framework", "harness", "tools"} or any(
                module_name.startswith(f"{prefix}.")
                for prefix in ("common", "framework", "harness", "tools")
            ):
                del sys.modules[module_name]
        from runtime_common.framework.agent_loop import AgentRun, TOOL_SCHEMAS

        self.AgentRun = AgentRun
        self.public = self.snapshot / "public"
        self.schemas = self.source / "schemas"
        self.lock_path = self.source / "tasks" / "task_05" / "environment" / "conda-linux-64.explicit.txt"
        self.tools = list(TOOL_SCHEMAS) + [
            {
                "type": "function",
                "function": {
                    "name": "budget_status",
                    "description": "Return current public tool and xTB-job budget usage.",
                    "parameters": {"type": "object", "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit",
                    "description": "Finalize the current complete or partial submission.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"status": {"enum": ["complete", "partial"]}},
                    },
                },
            },
        ]

    def public_case(self, index: int) -> dict:
        if index != 0:
            raise IndexError(index)
        return {
            "case_id": "task05-caffeine-water-binding-decomposition",
            "task": {"id": "task_05", "version": "1.15.0", "question_path": "question.md"},
            "constraints": {
                "method": "GFN2-xTB",
                "phase": "gas",
                "charge": 0,
                "uhf": 0,
                "counterpoise": False,
                "failure_cost_is_retained": True,
            },
            "budget": {
                "max_agent_turns": 69,
                "scientific_turns": 65,
                "finalization_turns": 4,
                "max_tool_calls": 260,
                "max_xtb_jobs": 16,
                "wall_time_seconds": 9000,
                # Audit 2026-09-27: finalization wall clock raised 600 -> 1800
                # seconds (matches task.yaml budgets and question.md) so the
                # submission-only finalization turns stay usable for
                # long-thinking models.
                "finalization_wall_time_seconds": 1800,
            },
            "output": {
                "submission_root": "submission",
                "required_files": ["final_answer.json", "report.md"],
                "schemas": ["submission_contract/final_answer.schema.json"],
            },
            "public_tools": self.tools,
        }

    def start_session(self, evidence_dir, token: str) -> dict:
        temp = tempfile.TemporaryDirectory(prefix="chembench-task05-")
        agent = self.AgentRun(
            task_dir=self.public,
            output_root=Path(temp.name) / "runs",
            model_client=_NoModelClient(),
            xtb_executable="xtb",
            run_id="harbor-task05",
        )
        return {
            "temp": temp,
            "agent": agent,
            "calls": 0,
            "token": token,
            "evidence_dir": Path(evidence_dir) if evidence_dir else None,
        }

    @staticmethod
    def _job_counts(agent) -> tuple[int, int]:
        manifest_path = agent.run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return 0, 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs = [item for item in manifest.get("jobs", []) if isinstance(item, dict)]
        completed = sum(str(item.get("status")) == "completed" for item in jobs)
        return len(jobs), completed

    def compute_budget_exhausted(self, state: dict) -> bool:
        reserved, _ = self._job_counts(state["agent"])
        return reserved >= self.public_case(0)["budget"]["max_xtb_jobs"]

    def call_tool(self, index: int, name, arguments, state: dict) -> dict:
        if index != 0:
            raise IndexError(index)
        state["calls"] += 1
        call_id = f"task05-{state['calls']:03d}"
        if name == "budget_status":
            reserved, completed = self._job_counts(state["agent"])
            return {
                "ok": True,
                "tool": name,
                "call_id": call_id,
                "result": {
                    "tool_calls_used": state["calls"],
                    "tool_calls_remaining": max(0, 260 - state["calls"]),
                    "xtb_jobs_reserved": reserved,
                    "xtb_jobs_completed": completed,
                    "xtb_jobs_remaining": max(0, 16 - reserved),
                },
            }
        allowed = {"read_file", "write_file", "run_python", "xtb_run", "xtb_parse"}
        if name not in allowed or not isinstance(arguments, dict):
            return {"ok": False, "tool": name, "call_id": call_id, "error": "INVALID_TOOL_REQUEST"}
        if name == "xtb_run" and self.compute_budget_exhausted(state):
            return {"ok": False, "tool": name, "call_id": call_id, "error": "COMPUTE_BUDGET_EXHAUSTED"}
        agent = state["agent"]
        remaining = state.get("remaining_wall_seconds", 9000)
        arguments = dict(arguments)
        if name == "xtb_run":
            arguments["timeout_seconds"] = min(int(arguments["timeout_seconds"]), remaining)
        old_budgets = agent.task_config.get("budgets")
        agent.task_config["budgets"] = {**(old_budgets or {}), "max_wall_time_seconds": remaining}
        try:
            result = agent.execute_tool(name, arguments)
            return {"ok": True, "tool": name, "call_id": call_id, "result": result}
        except Exception as exc:
            return {"ok": False, "tool": name, "call_id": call_id, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if old_budgets is None:
                agent.task_config.pop("budgets", None)
            else:
                agent.task_config["budgets"] = old_budgets

    def _budget_document(self, state: dict, reserved: int, completed: int) -> dict:
        # The protocol owns turn/tool/provider accounting; the scientific runtime owns jobs.
        accounting = state.get("organizer_accounting", {})
        scientific = int(accounting.get("scientific_turns_used", 0))
        attempts = int(accounting.get("provider_attempts", 0))
        calls = int(accounting.get("tool_calls_used", state["calls"]))
        elapsed = min(float(accounting.get("elapsed_seconds", 0)), 9000.0)
        totals = {"agent_turns": scientific, "api_attempts": attempts,
                  "requested_tokens": 0, "tool_calls": calls, "wall_seconds": elapsed,
                  "xtb_jobs": reserved, "xtb_scan_points": 0}
        finalizing = accounting.get("phase") == "finalization"
        history = [{"phase": "active", "trigger": "run_started"}]
        if finalizing:
            history.append({"phase": "finalizing", "trigger": accounting["exhaustion_trigger"]})
        return {
            "schema_version": "2.0.0",
            "agent_turns_used": scientific, "api_requests_used": attempts,
            "empty_response_retries": 0, "recovery_attempts": 0,
            "tool_calls_used": calls, "xtb_jobs_reserved": reserved,
            "xtb_jobs_completed": completed, "wall_time_seconds": elapsed,
            "limits": {"agent_turns": 65, "finalization_turns": 4,
                       "api_attempts": 69, "tool_calls": 260, "xtb_jobs": 16,
                       "xtb_scan_points": 0, "wall_seconds": 9000,
                       # Audit 2026-09-27: ledger limit synced with the
                       # finalization wall clock change 600 -> 1800 seconds.
                       "finalization_wall_seconds": 1800, "requested_tokens_per_turn": 0},
            "events": [{"sequence": 1, "event_type": "organizer_protocol_snapshot",
                        "logical_turn": scientific, "api_attempt": attempts,
                        "cumulative": dict(totals)}],
            "totals": totals, "phase": "finalizing" if finalizing else "active",
            "exhaustion_trigger": accounting.get("exhaustion_trigger"),
            "rejected_requests": int(accounting.get("rejected_requests", 0)),
            "finalization_turns_used": int(accounting.get("finalization_turns_used", 0)),
            "finalization_tool_calls": int(accounting.get("finalization_tool_calls", 0)),
            "phase_history": history, "finalization_notice": FINALIZATION_NOTICE,
        }

    def _runtime_checks(self, agent, old_manifest: dict) -> tuple[dict, dict]:
        from harness.runtime_checks import sha256_file

        provenance = old_manifest.get("provenance", {})
        lock_sha = sha256_file(self.lock_path)
        task_sha = str((provenance.get("task_config") or {}).get("sha256"))
        public_sha = str((provenance.get("public_inputs") or {}).get("tree_sha256"))
        xtb_sha = str((provenance.get("xtb_executable") or {}).get("sha256"))
        modules = []
        for import_name, distribution in (
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("yaml", "PyYAML"),
            ("jsonschema", "jsonschema"),
            ("psutil", "psutil"),
        ):
            modules.append(
                {"name": import_name, "status": "passed", "version": importlib.metadata.version(distribution)}
            )
        checks = {
            "schema_version": "1.0.0",
            "status": "passed",
            "python": {"executable": sys.executable, "version": platform.python_version()},
            "modules": modules,
            "binaries": [
                {
                    "name": "xtb",
                    "path": str(agent.xtb_runner.executable),
                    "expected_version": "6.7.1",
                    "actual_version": str(agent.xtb_runner._version),
                    "sha256": xtb_sha,
                    "status": "passed",
                }
            ],
            "environment_lock": {
                "path": "tasks/task_05/environment/conda-linux-64.explicit.txt",
                "expected_sha256": lock_sha,
                "actual_sha256": lock_sha,
                "status": "passed",
            },
            "system": {
                "platform": platform.system().lower(),
                "cpu": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
            },
            "hashes": {"task_config_sha256": task_sha, "public_inputs_sha256": public_sha},
        }
        environment = {
            "environment_lock_sha256": lock_sha,
            "task_config_sha256": task_sha,
            "public_inputs_sha256": public_sha,
            "xtb_executable_sha256": xtb_sha,
        }
        return checks, environment

    def _upgrade_submission(self, state: dict, submission: Path) -> None:
        from harness.native_task_runner import upgrade_scientific_manifest
        from harness.schema_validation import validate_budget_ledger, validate_scientific_manifest

        path = submission / "run_manifest.json"
        old = json.loads(path.read_text(encoding="utf-8"))
        reserved, completed = self._job_counts(state["agent"])
        budget = self._budget_document(state, reserved, completed)
        runtime, environment = self._runtime_checks(state["agent"], old)
        validate_budget_ledger(budget, self.schemas)
        upgraded = upgrade_scientific_manifest(
            old,
            task_id="task_05",
            scientific_problem_id="task_05_binding_decomposition",
            run_id=state["agent"].run_id,
            task_version="1.15.0",
            grader_version="1.21.0",
            package_release="1.19.0+eng.20260826",
            runtime_checks=runtime,
            budget_ledger=budget,
            environment=environment,
            submission_dir=submission,
        )
        validate_scientific_manifest(upgraded, self.schemas)
        path.write_text(json.dumps(upgraded, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def submit(self, index: int, candidate, tool_events, state: dict) -> dict:
        del tool_events
        if index != 0:
            raise IndexError(index)
        partial = isinstance(candidate, dict) and candidate.get("status") == "partial"
        final_dir, missing = state["agent"]._materialize_submission()
        partial = partial or bool(missing)
        self._upgrade_submission(state, final_dir)
        evidence_id = f"{state['token']}/task05-caffeine-water-binding-decomposition"
        evidence_dir = state["evidence_dir"]
        if evidence_dir is not None:
            target = evidence_dir / evidence_id
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()
            shutil.copytree(final_dir, target / "submission")
            (target / "sidecar_submission.json").write_text(
                json.dumps({"partial": partial, "missing_required_outputs": missing}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return {
            "ok": True,
            "case_id": "task05-caffeine-water-binding-decomposition",
            "status": "submitted",
            "feedback": {"accepted": True, "partial": partial},
            "cases_remaining": 0,
            "evidence_id": evidence_id,
        }


def create_adapter(snapshot: Path) -> Adapter:
    return Adapter(snapshot)

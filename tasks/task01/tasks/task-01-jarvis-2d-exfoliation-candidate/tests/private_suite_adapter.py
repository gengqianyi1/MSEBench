"""Stateful private adapter for the JARVIS 2D exfoliation controlled tool loop."""
from __future__ import annotations
import json,os,shutil,sys,tempfile
from pathlib import Path
from typing import Any

PRIMARY_TOOL_CALLS = 40
MAX_AGENT_TURNS = 50
MAX_READ_FILE_CALLS = 100
OUTPUT_REPAIR_CALLS = 2
TOTAL_TOOL_CALLS = PRIMARY_TOOL_CALLS + OUTPUT_REPAIR_CALLS + MAX_READ_FILE_CALLS

def _copytree_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for root, directories, files in os.walk(source):
        relative = Path(root).relative_to(source)
        destination = target / relative
        destination.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (destination / directory).mkdir(exist_ok=True)
        for name in files:
            shutil.copyfile(Path(root) / name, destination / name)


class Adapter:
    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot)
        self.source = self.snapshot / "input" / "original"
        self.task_dir = self.source / "tasks" / "jarvis_2d_exfoliation"
        self.case_count = 1
        sys.path.insert(0, str(self.source / "src"))
        from tool_executor import run_python_tool
        from chgnet_stability import run_selected_relaxation
        from evaluator import evaluate_result
        self.run_python_tool = run_python_tool
        self.run_selected_relaxation = run_selected_relaxation
        self.evaluate_result = evaluate_result
        self.tools = json.loads((self.snapshot / "tools.json").read_text())

    def start_session(self, evidence_dir, token):
        temp = tempfile.TemporaryDirectory(prefix="chembench-task01-")
        root = Path(temp.name) / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        # The public task contract names files under /workspace/input.  Keep
        # that namespace available inside the sidecar in addition to the
        # legacy root-level data/schemas layout; only the public snapshot is
        # copied (private_data is outside snapshot/input).
        public_input = self.snapshot / "input"
        if public_input.is_dir():
            shutil.copytree(public_input, root / "input")
        (root / "data").mkdir(parents=True, exist_ok=True)
        for name in ("train.json.gz", "validation.json.gz", "test_public.json.gz", "dataset_metadata.json", "candidate_generation_constraints.json"):
            source = self.source / "data" / name
            if source.exists():
                shutil.copy2(source, root / "data" / name)
        (root / "schemas").mkdir(parents=True, exist_ok=True)
        for name in ("result_schema.json", "candidate_record_schema.json", "candidate_summary_schema.json", "generation_manifest_schema.json"):
            source = self.task_dir / name
            if source.exists():
                shutil.copy2(source, root / "schemas" / name)
        seed = os.environ.get("CHEMBENCH_ORACLE_SEED_WORKSPACE")
        if seed:
            seed_path = Path(seed).resolve()
            if seed_path.is_dir():
                shutil.copytree(seed_path, root, dirs_exist_ok=True)
        return {
            "temp": temp,
            "root": root,
            "state": {
                "tool_index": 0,
                "primary_actions_used": 0,
                "chgnet_attempts": 0,
                "repair_calls_used": 0,
                "read_file_calls_used": 0,
            },
            "evidence_dir": Path(evidence_dir) if evidence_dir else None,
            "token": token,
        }

    def public_case(self, index):
        return {
            "case_id": "task_01_jarvis_2d_exfoliation_candidate",
            "task": {
                "id": "task_01_jarvis_2d_exfoliation_candidate",
                "case_id": "task_01_jarvis_2d_exfoliation_candidate",
                "protocol": "controlled_python/1.0",
            },
            "budget": {
                "max_tool_calls": TOTAL_TOOL_CALLS,
                "max_agent_turns": MAX_AGENT_TURNS,
                "max_script_runtime_seconds": 1800,
                "max_wall_time_seconds": 21600,
                "budget_mode": "40_primary_actions_plus_2_output_repairs",
                "primary_action_limit": PRIMARY_TOOL_CALLS,
                "repair_action_limit": OUTPUT_REPAIR_CALLS,
                "read_file_call_limit": MAX_READ_FILE_CALLS,
            },
            "constraints": {"workspace_only": True, "network_forbidden": True},
            "output": {"candidate_schema": {"final_status": "completed", "rationale": "string"}},
            "public_tools": list(self.tools.get("tools", {}).values()),
        }

    @staticmethod
    def export_paths(state):
        # Export only solver-authored submission material.  Public input and
        # schemas are already present in the Harbor workspace and copying the
        # full sidecar tree made /submit fragile on Windows bind mounts.
        return [
            "final_answer.json", "result.json", "native_final_report.json",
            "llm_request_accounting.json",
            "data", "predictions.csv", "candidate_library.json.gz", "candidate_summary.json",
            "model.joblib", "feature_names.json", "training_log.json",
            "generator", "selected_chgnet", "chgnet_tool_attempts.json",
        ]

    def _export_evidence(self, state, target: Path) -> None:
        stage = target.parent / f".{target.name}.tmp"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=True)
        for raw in self.export_paths(state):
            source = state["root"] / raw
            if not source.exists():
                continue
            destination = stage / "workspace" / raw
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                # Destination can be a Docker Desktop bind mount; preserve
                # contents only because metadata restoration is unsupported.
                _copytree_contents(source, destination)
            else:
                shutil.copyfile(source, destination)
        if target.exists():
            shutil.rmtree(target)
        stage.replace(target)

    def call_tool(self, index, name, arguments, state):
        del index
        if name not in {"run_python", "relax_selected_candidates_chgnet", "read_file"}:
            return {"ok": False, "tool": name, "error": "ACTION_NOT_ALLOWLISTED"}
        if not isinstance(arguments, dict):
            return {"ok": False, "tool": name, "error": "INVALID_ARGUMENTS"}

        next_index = int(state["state"].get("tool_index", 0)) + 1
        state["state"]["tool_index"] = next_index
        if name == "read_file":
            state["state"]["read_file_calls_used"] = int(state["state"].get("read_file_calls_used", 0)) + 1
            if state["state"]["read_file_calls_used"] > MAX_READ_FILE_CALLS:
                return {"ok": False, "tool": name, "error": "READ_FILE_BUDGET_EXHAUSTED", "budget": self._budget_state(state)}
            try:
                return self._read_file_tool(arguments, state)
            except Exception as exc:
                return {
                    "ok": False,
                    "tool": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": self._budget_state(state),
                }
        budget_error = self._consume_action_budget(name, arguments, state)
        if budget_error:
            return {"ok": False, "tool": name, "error": budget_error, "budget": self._budget_state(state)}
        if name == "run_python":
            try:
                result = self.run_python_tool(
                    script=arguments.get("script", ""),
                    timeout_seconds=int(arguments.get("timeout_seconds") or 1800),
                    workspace=state["root"],
                    project_root=state["root"].parent,
                    python_executable="python",
                    max_script_runtime_seconds=1800,
                    max_script_size_bytes=250000,
                    stdout_stderr_tail_chars=12000,
                    tool_index=state["state"]["tool_index"],
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "tool": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": self._budget_state(state),
                }
            return {
                "ok": True,
                "tool": name,
                "call_id": f"harbor-{state['state']['tool_index']:03d}",
                "result": result,
                "budget": self._budget_state(state),
            }
        state["state"]["chgnet_attempts"] += 1
        if state["state"]["chgnet_attempts"] > 1:
            return {
                "ok": False,
                "tool": name,
                "error": "CHGNET_TOOL_ALREADY_USED",
                "result": {"chgnet_tool_attempts": state["state"]["chgnet_attempts"]},
                "budget": self._budget_state(state),
            }
        try:
            result = self.run_selected_relaxation(
                candidate_ids=arguments.get("candidate_ids") or [],
                candidate_path=state["root"] / "candidate_library.json.gz",
                constraints_path=state["root"] / "data" / "candidate_generation_constraints.json",
                output_dir=state["root"] / "selected_chgnet",
            )
            self._record_chgnet_tool_attempt(state["root"], state["state"]["chgnet_attempts"])
            return {
                "ok": True,
                "tool": name,
                "call_id": f"harbor-{state['state']['tool_index']:03d}",
                "result": result,
                "budget": self._budget_state(state),
            }

        except Exception as exc:
            return {
                "ok": False,
                "tool": name,
                "error": type(exc).__name__,
                "result": {"error": str(exc)},
                "budget": self._budget_state(state),
            }

    @staticmethod
    def _record_chgnet_tool_attempt(root: Path, attempts: int) -> None:
        report_path = root / "native_final_report.json"
        if not report_path.exists():
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                return
            report["chgnet_tool_attempts"] = int(attempts)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return

    @staticmethod
    def _write_chgnet_attempts(state) -> None:
        payload = {
            "chgnet_tool_attempts": int(state["state"].get("chgnet_attempts", 0)),
            "counting_authority": "task01_sidecar",
        }
        (state["root"] / "chgnet_tool_attempts.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _budget_state(state):
        return {
            "tool_index": int(state["state"].get("tool_index", 0)),
            "primary_actions_used": int(state["state"].get("primary_actions_used", 0)),
            "primary_action_limit": PRIMARY_TOOL_CALLS,
            "repair_actions_used": int(state["state"].get("repair_calls_used", 0)),
            "repair_action_limit": OUTPUT_REPAIR_CALLS,
            "read_file_calls_used": int(state["state"].get("read_file_calls_used", 0)),
            "read_file_call_limit": MAX_READ_FILE_CALLS,
            "mode": "40_primary_actions_plus_2_output_repairs",
        }

    @staticmethod
    def _consume_action_budget(name, arguments, state):
        """Count compute actions while allowing bounded read-only inspection."""
        primary_used = int(state["state"].get("primary_actions_used", 0))
        if primary_used < PRIMARY_TOOL_CALLS:
            state["state"]["primary_actions_used"] = primary_used + 1
            return None
        if name != "run_python":
            return "PRIMARY_ACTION_BUDGET_EXHAUSTED"
        repair_used = int(state["state"].get("repair_calls_used", 0))
        if repair_used >= OUTPUT_REPAIR_CALLS:
            return "REPAIR_ACTION_BUDGET_EXHAUSTED"
        script = arguments.get("script", "") if isinstance(arguments, dict) else ""
        if not isinstance(script, str):
            return "INVALID_REPAIR_SCRIPT"
        prohibited = ("candidate_library.json.gz", "model.fit(", ".fit(", "train_test_split", "relax_selected_candidates_chgnet")
        if any(token in script for token in prohibited):
            return "REPAIR_ACTION_NOT_OUTPUT_ONLY"
        state["state"]["repair_calls_used"] = repair_used + 1
        return None

    def _read_file_tool(self, arguments, state):
        root = state["root"]
        raw = arguments.get("path")
        if not isinstance(raw, str) or not raw:
            return {"ok": False, "tool": "read_file", "error": "INVALID_PATH"}
        path = (root / raw).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return {"ok": False, "tool": "read_file", "error": "PATH_OUTSIDE_WORKSPACE"}
        if not path.is_file():
            return {"ok": False, "tool": "read_file", "error": "FILE_NOT_FOUND"}
        try:
            offset = max(0, int(arguments.get("offset", 0)))
            limit = max(1, min(20000, int(arguments.get("limit", 8000))))
        except (TypeError, ValueError):
            offset, limit = 0, 8000
        # Read only the requested prefix plus one character. The previous
        # implementation loaded the entire file before slicing it, which was
        # expensive for prediction/log artifacts and encouraged repeated I/O.
        try:
            with path.open("r", encoding="utf-8") as handle:
                prefix = handle.read(offset + limit + 1)
        except UnicodeDecodeError:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                prefix = handle.read(offset + limit + 1)
        content = prefix[offset:offset + limit]
        return {
            "ok": True,
            "tool": "read_file",
            "call_id": f"harbor-{state['state']['tool_index']:03d}",
            "result": {
                "path": raw,
                "size": path.stat().st_size,
                "offset": offset,
                "limit": limit,
                "truncated": len(prefix) > offset + limit,
                "content": content,
            },
            "budget": self._budget_state(state),
        }

    def submit(self, index, candidate, tool_events, state):
        del index, tool_events
        self._write_chgnet_attempts(state)
        required = [
            "final_answer.json",
            "result.json",
            "native_final_report.json",
            "data",
            "predictions.csv",
            "candidate_library.json.gz",
            "model.joblib",
            "feature_names.json",
            "training_log.json",
            "generator",
        ]
        missing = [name for name in required if not (state["root"] / name).exists()]
        if missing:
            diagnostic_id = f'{state["token"]}/rejected_task_01'
            if state["evidence_dir"]:
                target = state["evidence_dir"] / diagnostic_id
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._export_evidence(state, target)
            return {
                "ok": False,
                "case_id": self.public_case(0)["case_id"],
                "status": "invalid",
                "feedback": {
                    "accepted": False,
                    "missing_required_artifacts": missing,
                    "present_required_artifacts": [name for name in required if (state["root"] / name).exists()],
                    "diagnostic_evidence_id": diagnostic_id if state["evidence_dir"] else None,
                },
                "cases_remaining": 0,
            }
        evidence_id = f'{state["token"]}/task_01_jarvis_2d_exfoliation_candidate'
        if state["evidence_dir"]:
            target = state["evidence_dir"] / evidence_id
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            self._export_evidence(state, target)
            (target / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
        return {
            "ok": True,
            "case_id": self.public_case(0)["case_id"],
            "status": "submitted",
            "feedback": {"accepted": True, "final_status": candidate.get("final_status") if isinstance(candidate, dict) else None},
            "cases_remaining": 0,
            "evidence_id": evidence_id,
        }

def create_adapter(snapshot):
    return Adapter(snapshot)

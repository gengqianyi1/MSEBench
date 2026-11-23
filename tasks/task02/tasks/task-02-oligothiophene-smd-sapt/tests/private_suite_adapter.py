"""Stateful private adapter for the staged SMD-SAPT run_python protocol."""
from __future__ import annotations
import hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator

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

def _sanitize_public_input(input_root: Path) -> None:
    for cfg_path in input_root.rglob("task_config.yaml"):
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and ("instance_generation" in data or "scoring_reference" in data):
            data.pop("instance_generation", None)
            data.pop("scoring_reference", None)
            cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    for generator in input_root.rglob("instance_generator.py"):
        generator.unlink()

STAGE_FILES = ("stage_A_audit.json", "stage_B_bulk.json", "stage_C_local.json", "result.json")

class Adapter:
    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot)
        self.source = self.snapshot / "input" / "original"
        self.task_dir = self.source / "tasks" / "oligothiophene_smd_sapt_v3"
        self.case_count = 1
        sys.path.insert(0, str(self.source / "src"))
        from tool_executor import run_python_tool
        from instance_generator import prepare_runtime_instance
        self.run_python_tool = run_python_tool
        self.prepare_runtime_instance = prepare_runtime_instance
        self.tools = json.loads((self.snapshot / "tools.json").read_text())

    def start_session(self, evidence_dir, token):
        temp = tempfile.TemporaryDirectory(prefix="chembench-task02-")
        root = Path(temp.name) / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        # Mirror the public /workspace/input namespace referenced by the
        # question and submission contract.  This contains no private_data.
        public_input = self.snapshot / "input"
        if public_input.is_dir():
            shutil.copytree(public_input, root / "input")
            _sanitize_public_input(root / "input")
        task_cfg = yaml.safe_load((self.task_dir / "task_config.yaml").read_text(encoding="utf-8"))
        if not isinstance(task_cfg, dict):
            raise TypeError("task_config.yaml must contain an object")
        self.prepare_runtime_instance(task_cfg, root)
        for name in ("instance.json", "protocol.yaml", "benchmark_tools.py", "stage_A_schema.json", "stage_B_schema.json", "stage_C_schema.json", "result_schema.json", "stage_validation.py"):
            source = self.task_dir / name
            if source.exists():
                shutil.copy2(source, root / name)
        seed_source = self.task_dir / "seed_evidence"
        if seed_source.exists():
            shutil.copytree(seed_source, root / "seed_evidence")
        seed = os.environ.get("CHEMBENCH_ORACLE_SEED_WORKSPACE")
        if seed:
            seed_path = Path(seed).resolve()
            if seed_path.is_dir():
                shutil.copytree(seed_path, root, dirs_exist_ok=True)
        return {
            "temp": temp,
            "root": root,
            "state": {"tool_index": 0, "tool_rows": [], "stage_rows": []},
            "evidence_dir": Path(evidence_dir) if evidence_dir else None,
            "token": token,
        }

    def public_case(self, index):
        return {
            "case_id": "task_02_oligothiophene_smd_sapt",
            "task": {
                "id": "task_02_oligothiophene_smd_sapt",
                "case_id": "task_02_oligothiophene_smd_sapt",
                "protocol": "staged_python/1.0",
            },
            "budget": {
                "max_tool_calls": None,
                "max_agent_turns": None,
                "max_script_runtime_seconds": 7200,
                "max_wall_time_seconds": 21600,
                "budget_mode": "unlimited_actions_with_wall_time_safety_cap",
            },
            "constraints": {"workspace_only": True, "network_forbidden": True},
            "output": {"candidate_schema": {"final_status": "completed", "rationale": "string"}},
            "public_tools": list(self.tools.get("tools", {}).values()),
        }

    @staticmethod
    def export_paths(state):
        # Keep only model-authored evidence needed by the staged verifier.
        # `input/` is public and already exists in the Harbor workspace.
        return [
            "final_answer.json", "result.json", "native_final_report.json",
            "llm_request_accounting.json",
            "stage_A_audit.json", "plan.json", "stage_B_bulk.json",
            "stage_C_local.json", "revisions.json", "structures",
            "logs", "seed_evidence", "tool_calls.jsonl", "stage_integrity.jsonl",
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
        if name not in {"run_python", "read_file"}:
            return {"ok": False, "tool": name, "error": "ACTION_NOT_ALLOWLISTED"}
        state["state"]["tool_index"] += 1
        if name == "read_file":
            try:
                return self._read_file_tool(arguments, state)
            except Exception as exc:
                return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}
        try:
            result = self.run_python_tool(
                script=arguments.get("script", ""),
                timeout_seconds=int(arguments.get("timeout_seconds") or 7200),
                workspace=state["root"],
                project_root=state["root"].parent,
                python_executable="python",
                max_script_runtime_seconds=7200,
                max_script_size_bytes=250000,
                stdout_stderr_tail_chars=12000,
                tool_index=state["state"]["tool_index"],
            )
        except Exception as exc:
            return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}
        self._record_tool_call(state, result)
        return {"ok": True, "tool": name, "call_id": f"harbor-{state['state']['tool_index']:03d}", "result": result}

    @staticmethod
    def _stage_now(root: Path) -> str:
        for stage, filename in (("stage_D", "result.json"), ("stage_C", "stage_C_local.json"),
                                ("stage_B", "stage_B_bulk.json"), ("stage_A", "stage_A_audit.json")):
            if (root / filename).is_file():
                return stage
        return "stage_A"

    def _record_tool_call(self, state, result) -> None:
        root = state["root"]
        index = state["state"]["tool_index"]
        stage = self._stage_now(root)
        state["state"]["tool_rows"].append({"tool_index": index, "stage": stage, "result": result})
        snapshot = {name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                    for name in STAGE_FILES if (root / name).is_file()}
        state["state"]["stage_rows"].append({"runner_stage": stage, "tool_index": index, "snapshot": snapshot})

    def _write_tool_logs(self, state) -> None:
        root = state["root"]
        for filename, key in (("tool_calls.jsonl", "tool_rows"), ("stage_integrity.jsonl", "stage_rows")):
            rows = state["state"].get(key) or []
            body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            (root / filename).write_text(body, encoding="utf-8")

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
        }

    def submit(self, index, candidate, tool_events, state):
        del index, tool_events
        self._write_tool_logs(state)
        required = [
            "final_answer.json",
            "result.json",
            "native_final_report.json",
            "stage_A_audit.json",
            "plan.json",
            "stage_B_bulk.json",
            "stage_C_local.json",
            "revisions.json",
        ]
        missing = [name for name in required if not (state["root"] / name).exists()]
        invalid = []
        wrapper_schemas = {
            "final_answer.json": "final_answer.schema.json",
            "native_final_report.json": "native_final_report.schema.json",
            "revisions.json": "revisions.schema.json",
        }
        for artifact, schema_name in wrapper_schemas.items():
            path = state["root"] / artifact
            if not path.is_file():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                schema = json.loads((state["root"] / "input" / "schemas" / schema_name).read_text(encoding="utf-8"))
                failures = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: "/".join(map(str, item.absolute_path)))
                invalid.extend(f"{artifact}: {'/'.join(map(str, item.absolute_path)) or '<root>'}: {item.message}" for item in failures)
            except Exception as exc:
                invalid.append(f"{artifact}: {type(exc).__name__}: {exc}")
        if missing or invalid:
            diagnostic_id = f'{state["token"]}/rejected_task_02'
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
                    "invalid_required_artifacts": invalid,
                    "present_required_artifacts": [name for name in required if (state["root"] / name).exists()],
                    "diagnostic_evidence_id": diagnostic_id if state["evidence_dir"] else None,
                },
                "cases_remaining": 0,
            }
        evidence_id = f'{state["token"]}/task_02_oligothiophene_smd_sapt'
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

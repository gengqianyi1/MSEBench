"""Task-agnostic Agent runner and submission collector."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import GlobalConfig, TaskConfig
from .artifact_integrity import link_and_validate_submission, submission_payload_sha256, submission_tree_sha256
from .evaluator import evaluate_submission
from .runtime_checks import run_runtime_checks
from .runtime_checks import sha256_file
from .release_contract import load_release_contract, validate_task_metadata
from .utils import (
    LLMClient,
    artifact_manifest,
    copy_artifact,
    create_workspace,
    ensure_within,
    extract_json_block,
    load_json,
    load_text,
    run_python_script,
    save_json,
    save_jsonl,
    save_text,
    validate_python_source,
)


def _native_runner_failure_detail(
    result: dict,
    native_result: dict | None,
) -> str:
    """Return the most specific observable native-runner failure message.

    The native runner writes its sanitized structured result before returning
    non-zero.  Its JSON summary is printed to stdout, while stderr can be empty;
    therefore stderr alone is not a valid failure contract.
    """

    if isinstance(native_result, dict):
        failure = native_result.get("failure")
        if isinstance(failure, str) and failure.strip():
            return failure.strip()[-2000:]
    for field in ("stderr", "stdout"):
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()[-2000:]
    return f"native runner exited with code {result.get('exit_code')} without a failure message"


class Orchestrator:
    def __init__(self, task_cfg: TaskConfig, global_cfg: GlobalConfig, task_dir: Path, run_dir: Path):
        self.task_cfg = task_cfg
        self.global_cfg = global_cfg
        self.task_dir = Path(task_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.workspace: Path | None = None
        # Native tasks own their robust provider client outside the model
        # workspace.  Avoid resolving or retaining credentials in the outer
        # Harness process configuration object.
        self.llm = None if task_cfg.execution_mode == "native_runner" else LLMClient(global_cfg.llm_api)
        self.prompt = load_text(self.task_dir / "prompt.md")
        tools_path = self.task_dir / task_cfg.tools_registry_path
        self.tools_registry = load_json(tools_path) if tools_path.exists() else {}
        rubric_path = self.task_dir / task_cfg.rubric_path
        self.rubric = load_json(rubric_path) if rubric_path.exists() else None
        self.started_monotonic = 0.0
        self.tool_call_count = 0

    def run(self) -> dict:
        self._init_run()
        try:
            if self.task_cfg.execution_mode == "native_runner":
                self._run_native_runner()
            else:
                self._agent_loop()
            self._transition("execution_status", "COMPLETED", "runner completed")
        except TimeoutError as exc:
            self._transition("execution_status", "TIMEOUT", str(exc))
            self.run_state["error"] = str(exc)
        except Exception as exc:
            self._transition("execution_status", "AGENT_ERROR", str(exc))
            self.run_state["error"] = str(exc)

        collection_errors = self._collect_submission()
        self.run_state["finished_at"] = datetime.now(timezone.utc).isoformat()

        if self.run_state.get("execution_status") == "COMPLETED":
            if collection_errors:
                self._transition("submission_status", "INVALID", "submission collection failed")
                self.run_state["error"] = "; ".join(collection_errors)
                save_json(self.run_dir / "run_manifest.json", self._manifest_view())
                return self.run_state
            self._transition("submission_status", "SUBMITTED", "submission artifacts collected")
            try:
                link_and_validate_submission(
                    self._manifest_view(),
                    self.run_dir / "submission",
                    self.schema_dir,
                )
            except Exception as exc:
                self._transition("submission_status", "INVALID", "scientific submission binding failed")
                self.run_state["error"] = f"{type(exc).__name__}: {exc}"
                save_json(self.run_dir / "run_manifest.json", self._manifest_view())
                return self.run_state

        self._score()
        self.run_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest = self._manifest_view()
        if (
            manifest.get("execution_status") == "COMPLETED"
            and manifest.get("submission_status") == "SUBMITTED"
            and manifest.get("scoring_status") == "SCORED"
        ):
            manifest = link_and_validate_submission(manifest, self.run_dir / "submission", self.schema_dir)
        save_json(self.run_dir / "run_manifest.json", manifest)
        return self.run_state

    def _transition(self, field: str, value: str, reason: str) -> None:
        previous = self.run_state.get(field)
        self.run_state[field] = value
        self.run_state.setdefault("state_transitions", []).append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "field": field,
            "from": previous,
            "to": value,
            "reason": reason or "state updated",
        })

    def _run_native_runner(self) -> None:
        """Execute a benchmark's verified native agent/evaluator loop as one harness event."""
        assert self.workspace is not None
        script = ensure_within(self.task_dir / self.task_cfg.runner_script, self.task_dir)
        if not script.is_file():
            raise FileNotFoundError(f"Native runner script not found: {self.task_cfg.runner_script}")
        timeout = self.task_cfg.runner_timeout_seconds or self.task_cfg.total_timeout_seconds
        runner_env = dict(self.task_cfg.env_overrides)
        runner_env.update({
            "CHEMBENCH_TASK_DIR": str(self.task_dir),
            "CHEMBENCH_RUN_DIR": str(self.run_dir),
            "CHEMBENCH_WORKSPACE": str(self.workspace),
            "CHEMBENCH_RUN_ID": str(self.run_state["run_id"]),
            "CHEMBENCH_LLM_URL": str(self.global_cfg.llm_api.get("url", "")),
            "CHEMBENCH_LLM_MODEL": str(self.global_cfg.llm_api.get("model", "")),
            "CHEMBENCH_MODEL_NAME": str(self.global_cfg.llm_api.get("model_name", self.global_cfg.llm_api.get("model", ""))),
            "CHEMBENCH_LLM_TEMPERATURE": str(self.global_cfg.llm_api.get("temperature", 0.0)),
        })
        if self.global_cfg.llm_api.get("model_config_file"):
            runner_env["CHEMBENCH_MODEL_CONFIG_FILE"] = str(self.global_cfg.llm_api["model_config_file"])
        result = run_python_script(
            script,
            workdir=self.task_dir,
            python_path=sys.executable,
            timeout=timeout,
            env=runner_env,
        )
        self.tool_call_count = 1
        event = {
            "iteration": 1,
            "call_index": 1,
            "tool_name": "native_runner",
            "description": self.task_cfg.runner_script,
            "exit_code": result.get("exit_code"),
            "wall_seconds": result.get("wall_seconds"),
        }
        self.run_state["tool_events"].append(event)
        self.run_state["iterations"].append({
            "iteration": 1,
            "agent_status": "complete" if result.get("exit_code") == 0 else "runner_failed",
            "commands_count": 1,
            "final_answer": None,
            "tool_results": [{**event, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", ""), "result_json": result.get("result_json")}],
            "read_results": [],
            "llm_reasoning": "Task-native verified benchmark runner executed by the unified harness.",
        })
        native_artifacts = self.run_dir / "native_run_artifacts"
        native_result: dict | None = None
        if native_artifacts.is_dir():
            copy_artifact(native_artifacts, self.run_dir / "logs" / "native")
            runtime_path = native_artifacts / "runtime_checks.json"
            budget_path = native_artifacts / "budget_ledger.json"
            result_path = native_artifacts / "native_runner_result.json"
            if runtime_path.is_file():
                self.run_state["runtime_checks"] = load_json(runtime_path)
            if budget_path.is_file():
                self.run_state["budget_ledger"] = load_json(budget_path)
            if result_path.is_file():
                native_result = load_json(result_path)
                self.run_state["returned_model_ids"] = native_result.get("returned_model_ids", [])
        runtime_environment = self.workspace / "runtime_environment.json"
        if runtime_environment.is_file():
            copy_artifact(runtime_environment, self.run_dir / "logs" / "runtime_environment.json")
        if result.get("exit_code") != 0:
            raise RuntimeError(
                f"Native runner failed: {_native_runner_failure_detail(result, native_result)}"
            )

    def _init_run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs" / "scripts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "submission").mkdir(exist_ok=True)
        self.workspace = create_workspace(self.run_dir, self.task_dir)
        self.schema_dir = self.task_dir.parents[1] / "schemas"
        self.release_contract = load_release_contract(self.task_dir.parents[1])
        scientific_config = yaml.safe_load((self.task_dir / "input" / "original" / "config" / "task.yaml").read_text(encoding="utf-8")) or {}
        validate_task_metadata(self.release_contract, task_id=self.task_cfg.task_id, version=self.task_cfg.version, status=str(scientific_config.get("status", "")))
        self.started_monotonic = time.monotonic()
        runtime_checks = run_runtime_checks(self.task_dir, self.task_cfg.runtime_checks, self.schema_dir)
        if runtime_checks["status"] != "passed":
            raise RuntimeError("task-declared runtime checks failed")
        zero_totals = {
            "agent_turns": 0,
            "api_attempts": 0,
            "tool_calls": 0,
            "xtb_jobs": 0,
            "xtb_scan_points": 0,
            "wall_seconds": 0.0,
            "requested_tokens": 0,
        }
        self.run_state = {
            "schema_name": "harness_run_manifest_v2",
            "schema_version": "2.1.0",
            "manifest_role": "harness",
            "run_id": f"{self.task_cfg.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "task_id": self.task_cfg.task_id,
            "task_version": self.task_cfg.version,
            "grader_version": self.release_contract.task_grader_version(self.task_cfg.task_id),
            "package_release": self.release_contract.task_package_release(self.task_cfg.task_id),
            "environment_lock_sha256": runtime_checks["environment_lock"]["actual_sha256"],
            "task_config_sha256": runtime_checks["hashes"]["task_config_sha256"],
            "public_inputs_sha256": runtime_checks["hashes"]["public_inputs_sha256"],
            "xtb_executable_sha256": runtime_checks["binaries"][0]["sha256"],
            "model_id": self.global_cfg.llm_api.get("model", "unknown"),
            "returned_model_ids": [],
            "runtime_checks": runtime_checks,
            "budget_ledger": {
                "schema_version": "2.0.0",
                "agent_turns_used": 0,
                "api_requests_used": 0,
                "empty_response_retries": 0,
                "recovery_attempts": 0,
                "tool_calls_used": 0,
                "xtb_jobs_reserved": 0,
                "xtb_jobs_completed": 0,
                "wall_time_seconds": 0.0,
                "limits": dict(self.task_cfg.budget_limits),
                "events": [],
                "totals": zero_totals,
            },
            "started_at": datetime.now(timezone.utc).isoformat(),
            "iterations": [],
            "tool_events": [],
            "execution_status": "RUNNING",
            "submission_status": "NOT_CHECKED",
            "scoring_status": "NOT_SCORED",
            "state_transitions": [{"timestamp_utc": datetime.now(timezone.utc).isoformat(), "field": "execution_status", "from": None, "to": "RUNNING", "reason": "run initialized"}],
            "score": None,
        }

        system = f"""You are an expert materials-science and computational-chemistry agent.

## Task
{self.prompt}

## Available tools
{self._render_tool_descriptions()}

{self._render_environment()}

## Response protocol
Return exactly one JSON object, preferably in a ```json block:
{{
  "reasoning": "concise scientific reasoning",
  "status": "running",
  "commands": [
    {{"command": "valid Python source code", "workdir": ".", "description": "purpose"}}
  ],
  "read_files": ["relative/path/to/file"],
  "final_answer": null
}}

Rules:
- Each commands[].command is Python source code, not a shell command.
- The default working directory is the workspace root. Write required artifacts there using the exact paths from the task.
- Use read_files to inspect small text outputs from previous iterations.
- Do not access paths outside the workspace.
- Network access is {'allowed only when explicitly required by the task' if self.task_cfg.network_allowed else 'disabled'}.
- When finished, set status to "complete" and provide final_answer.
"""
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Complete the task in at most {self.task_cfg.max_iterations} iterations."},
        ]

    def _check_total_timeout(self) -> None:
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed > self.task_cfg.total_timeout_seconds:
            raise TimeoutError(f"Task exceeded total timeout of {self.task_cfg.total_timeout_seconds}s")

    def _agent_loop(self) -> None:
        completed = False
        for index in range(1, self.task_cfg.max_iterations + 1):
            self._check_total_timeout()
            result = self._run_iteration(index)
            self.run_state["iterations"].append(result)
            if result.get("agent_status") == "complete" and self.task_cfg.stop_on_status_complete:
                completed = True
                break
            self.messages.append({"role": "user", "content": self._summarize_iteration(result)})

        if not completed and self.task_cfg.final_answer_required:
            self._check_total_timeout()
            self.messages.append({
                "role": "user",
                "content": "The iteration budget is exhausted. Submit the best available final_answer now; do not run additional commands.",
            })
            result = self._run_iteration(self.task_cfg.max_iterations + 1, allow_commands=False)
            self.run_state["iterations"].append(result)

    def _run_iteration(self, index: int, allow_commands: bool = True) -> dict:
        assert self.workspace is not None
        assert self.llm is not None
        response = self.llm.call(self.messages)
        try:
            llm_text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response does not contain choices[0].message.content") from exc
        self.messages.append({"role": "assistant", "content": llm_text})
        save_text(self.run_dir / "logs" / f"llm_response_{index:03d}.txt", llm_text)
        parsed = extract_json_block(llm_text)
        if parsed is None:
            return {"iteration": index, "agent_status": "invalid_response", "error": "JSON parse failed"}

        commands = parsed.get("commands", []) if allow_commands else []
        if not isinstance(commands, list):
            commands = []
        tool_results: list[dict] = []
        for command_index, command_spec in enumerate(commands):
            self._check_total_timeout()
            if self.tool_call_count >= self.task_cfg.max_tool_calls:
                tool_results.append({"error": "TOOL_BUDGET_EXHAUSTED", "exit_code": -2})
                break
            if not isinstance(command_spec, dict):
                tool_results.append({"error": "INVALID_COMMAND_SPEC", "exit_code": -2})
                continue
            source = str(command_spec.get("command", ""))
            try:
                validate_python_source(
                    source,
                    max_bytes=int(self.global_cfg.safety.get("max_script_size_bytes", 100000)),
                    block_network=bool(self.global_cfg.safety.get("block_network_access", True)) and not self.task_cfg.network_allowed,
                )
                requested_workdir = command_spec.get("workdir", ".")
                workdir = ensure_within(self.workspace / requested_workdir, self.workspace)
                workdir.mkdir(parents=True, exist_ok=True)
                script_file = self.run_dir / "logs" / "scripts" / f"iter_{index:03d}_{command_index:03d}.py"
                save_text(script_file, source)
                timeout = min(self.task_cfg.timeout_per_iteration, self.task_cfg.total_timeout_seconds)
                result = run_python_script(
                    script_file,
                    workdir=workdir,
                    python_path=self.task_cfg.python_path,
                    timeout=timeout,
                    env=self.task_cfg.env_overrides,
                )
            except Exception as exc:
                result = {"exit_code": -2, "stdout": "", "stderr": str(exc), "wall_seconds": 0, "result_json": None}
            self.tool_call_count += 1
            event = {
                "iteration": index,
                "call_index": self.tool_call_count,
                "tool_name": "python_script",
                "description": command_spec.get("description", ""),
                "exit_code": result.get("exit_code"),
                "wall_seconds": result.get("wall_seconds"),
            }
            self.run_state["tool_events"].append(event)
            tool_results.append({
                **event,
                "stdout": result.get("stdout", "")[-3000:],
                "stderr": result.get("stderr", "")[-2000:],
                "result_json": result.get("result_json"),
            })

        read_results = self._read_requested_files(parsed.get("read_files", []))
        return {
            "iteration": index,
            "agent_status": parsed.get("status", "running"),
            "commands_count": len(commands),
            "final_answer": parsed.get("final_answer"),
            "tool_results": tool_results,
            "read_results": read_results,
            "llm_reasoning": str(parsed.get("reasoning", ""))[:2000],
        }

    def _read_requested_files(self, requested: object) -> list[dict]:
        assert self.workspace is not None
        if not isinstance(requested, list):
            return []
        results = []
        for value in requested[:20]:
            try:
                path = ensure_within(self.workspace / str(value), self.workspace)
                if not path.is_file():
                    raise FileNotFoundError(str(value))
                content = load_text(path)
                results.append({"path": str(value), "content": content[-10000:]})
            except Exception as exc:
                results.append({"path": str(value), "error": str(exc)})
        return results

    def _collect_submission(self) -> list[str]:
        assert self.workspace is not None
        submission_dir = self.run_dir / "submission"
        if self.task_cfg.execution_mode == "native_runner":
            staging = self.run_dir / "native_submission"
            if not staging.is_dir():
                return ["MISSING:native_submission"]
            if any(submission_dir.iterdir()):
                raise RuntimeError("native submission destination is not empty")
            copy_artifact(staging, submission_dir)
            collection_errors = []
            for relative in self.task_cfg.required_artifacts:
                matches = sorted(submission_dir.glob(relative))
                if not matches:
                    collection_errors.append(f"MISSING:{relative}")
            save_jsonl(self.run_dir / "logs" / "tool_events.jsonl", self.run_state["tool_events"])
            save_json(self.run_dir / "logs" / "iteration_history.json", self.run_state["iterations"])
            return collection_errors

        final_answer = None
        for iteration in reversed(self.run_state["iterations"]):
            if iteration.get("final_answer") is not None:
                final_answer = iteration["final_answer"]
                break
        if final_answer is not None:
            save_json(submission_dir / self.task_cfg.final_answer_file, final_answer)
        else:
            workspace_answer = self.workspace / self.task_cfg.final_answer_file
            if workspace_answer.is_file():
                copy_artifact(workspace_answer, submission_dir / self.task_cfg.final_answer_file)

        collection_errors = []
        for relative in self.task_cfg.required_artifacts:
            if relative == self.task_cfg.final_answer_file:
                continue
            collected = self._collect_artifact_pattern(relative, submission_dir)
            if not collected:
                collection_errors.append(f"MISSING:{relative}")

        # Optional artifacts do not determine submission completeness, but must
        # still be preserved whenever an agent produced them. This is important
        # for independently scored modules and for reproducible run traces.
        for relative in self.task_cfg.optional_artifacts:
            if relative == self.task_cfg.final_answer_file:
                continue
            self._collect_artifact_pattern(relative, submission_dir)

        manifest = artifact_manifest(submission_dir)
        save_json(submission_dir / "submission.json", {
            "schema_version": "1.0",
            "instance_id": self.task_cfg.task_id,
            "task_version": self.task_cfg.version,
            "artifacts": manifest,
            "collection_errors": collection_errors,
        })
        save_jsonl(self.run_dir / "logs" / "tool_events.jsonl", self.run_state["tool_events"])
        save_json(self.run_dir / "logs" / "iteration_history.json", self.run_state["iterations"])
        return collection_errors

    def _collect_artifact_pattern(self, pattern: str, submission_dir: Path) -> list[str]:
        """Collect canonical outputs from workspace, final/, or legacy iter_N/.

        Exact paths are resolved from the workspace root first, then final/, then
        the newest iter_N/. Glob patterns collect all non-colliding matches while
        preserving paths relative to the matching base directory.
        """
        assert self.workspace is not None
        iteration_dirs = sorted(
            [path for path in self.workspace.glob("iter_*") if path.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        bases = [self.workspace]
        if (self.workspace / "final").is_dir():
            bases.append(self.workspace / "final")
        bases.extend(iteration_dirs)
        has_glob = any(char in pattern for char in "*?[")
        collected: list[str] = []
        for base in bases:
            matches = sorted(base.glob(pattern)) if has_glob else [base / pattern]
            for source in matches:
                if not source.exists():
                    continue
                destination_relative = source.relative_to(base)
                destination = submission_dir / destination_relative
                if destination.exists():
                    continue
                copy_artifact(source, destination)
                collected.append(destination_relative.as_posix())
            if collected and not has_glob:
                break
        return collected

    def _score(self) -> None:
        score = evaluate_submission(
            run_dir=self.run_dir,
            submission_dir=self.run_dir / "submission",
            task_cfg=self.task_cfg,
            rubric=self.rubric,
            task_dir=self.task_dir,
            execution_status=self.run_state["execution_status"],
            run_manifest=self._manifest_view(),
        )
        submission_status = "INVALID" if score["submission_status"] == "INVALID_SUBMISSION" else score["submission_status"]
        self._transition("submission_status", submission_status, "submission collection and contract checks completed")
        self._transition("scoring_status", score.get("grading_status", "NOT_SCORED"), "grader completed")
        self.run_state["score"] = score
        save_json(self.run_dir / "score.json", score)

    def _manifest_view(self) -> dict:
        llm = self.global_cfg.llm_api
        provider_config = {
            "temperature": float(llm.get("temperature", 0.2)),
            "top_p": float(llm.get("top_p", 0.95)),
            "seed": int(llm.get("seed", 0)),
            "request_timeout_seconds": int(llm.get("request_timeout_seconds", 600)),
            "max_transport_retries": int(llm.get("max_transport_retries", 3)),
            "max_semantic_retries": int(llm.get("max_semantic_retries", 1)),
            "max_process_resumes": int(llm.get("max_process_resumes", 1)),
        }
        score = self.run_state.get("score") or {}
        submission_status = self.run_state.get("submission_status")
        if submission_status == "INVALID_SUBMISSION":
            submission_status = "INVALID"
        return {
            "schema_name": "harness_run_manifest_v2",
            "schema_version": "2.1.0",
            "manifest_role": "harness",
            "run_id": self.run_state.get("run_id"),
            "task_id": self.task_cfg.task_id,
            "task_version": self.task_cfg.version,
            "grader_version": self.release_contract.task_grader_version(self.task_cfg.task_id),
            "package_release": self.release_contract.task_package_release(self.task_cfg.task_id),
            "environment_lock_sha256": self.run_state.get("environment_lock_sha256"),
            "task_config_sha256": self.run_state.get("task_config_sha256"),
            "public_inputs_sha256": self.run_state.get("public_inputs_sha256"),
            "xtb_executable_sha256": self.run_state.get("xtb_executable_sha256"),
            "requested_model_id": self.run_state.get("model_id"),
            "returned_model_ids": self.run_state.get("returned_model_ids", []),
            "provider": str(llm.get("provider", "fixture")),
            "provider_config": provider_config,
            "runtime_checks": self.run_state.get("runtime_checks"),
            "budget_ledger": self.run_state.get("budget_ledger"),
            "state_transitions": self.run_state.get("state_transitions", []),
            "started_at": self.run_state.get("started_at"),
            "finished_at": self.run_state.get("finished_at"),
            "execution_status": self.run_state.get("execution_status"),
            "submission_status": submission_status,
            "scoring_status": self.run_state.get("scoring_status", "NOT_SCORED"),
            "submission_sha256": submission_tree_sha256(self.run_dir / "submission"),
            "submission_payload_sha256": submission_payload_sha256(self.run_dir / "submission"),
            "submission_manifest_path": "submission/run_manifest.json",
            "submission_manifest_sha256": sha256_file(self.run_dir / "submission" / "run_manifest.json") if (self.run_dir / "submission" / "run_manifest.json").is_file() else "0" * 64,
        }

    def _render_tool_descriptions(self) -> str:
        descriptions = []
        for name, spec in self.tools_registry.get("tools", {}).items():
            if name in self.task_cfg.required_tools:
                descriptions.append(f"- {name}: {spec.get('description', '')}")
                if spec.get("python_usage"):
                    descriptions.append(f"  Usage: {spec['python_usage'][:500]}")
        return "\n".join(descriptions) or "- Python standard library only"

    def _render_environment(self) -> str:
        return f"""## Environment
- Workspace: {self.workspace}
- Python: {self.task_cfg.python_path}
- Iteration budget: {self.task_cfg.max_iterations}
- Tool-call budget: {self.task_cfg.max_tool_calls}
- Total timeout: {self.task_cfg.total_timeout_seconds}s
"""

    def _summarize_iteration(self, result: dict) -> str:
        lines = [f"Iteration {result.get('iteration')} results:"]
        for item in result.get("tool_results", []):
            lines.append(f"- tool exit={item.get('exit_code')} time={item.get('wall_seconds')}s")
            if item.get("stdout"):
                lines.append(f"stdout:\n{item['stdout'][-2000:]}")
            if item.get("stderr"):
                lines.append(f"stderr:\n{item['stderr'][-1000:]}")
            if item.get("result_json") is not None:
                lines.append("result.json: " + json.dumps(item["result_json"], ensure_ascii=False)[:3000])
        for item in result.get("read_results", []):
            lines.append(f"file {item.get('path')}:\n{item.get('content', item.get('error', ''))}")
        lines.append("Continue with the next evidence-based step, or submit status=complete.")
        return "\n\n".join(lines)


def run_single_task(task_dir: Path, global_config_path: Path, output_dir: Path | None = None) -> dict:
    task_dir = Path(task_dir).resolve()
    task_cfg = TaskConfig.from_yaml(task_dir / "task.yaml")
    global_cfg = GlobalConfig.from_yaml(Path(global_config_path).resolve())
    run_name = f"{task_cfg.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = Path(output_dir or "runs").resolve() / run_name
    result = Orchestrator(task_cfg, global_cfg, task_dir, run_dir).run()
    numeric_score = result["score"].get("score")
    score_text = "N/A" if numeric_score is None else f"{numeric_score:.1f} / {task_cfg.max_score:.1f}"
    print(f"Task: {task_cfg.task_id} - {task_cfg.title}")
    print(f"Score: {score_text}")
    print(f"Status: {result['execution_status']} | Submission: {result.get('submission_status')}")
    print(f"Output: {run_dir}")
    return result

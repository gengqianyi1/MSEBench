"""Minimal observable OpenAI-compatible Agent loop for the benchmark.

The framework deliberately exposes only five tools and contains no
chemistry-specific planning.  It records observable messages/tool calls without
requesting or retaining hidden chain-of-thought.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from harness.budget import classify_xtb_compute, task04_compute_cost

from common.io_utils import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    bubblewrap_command,
    file_hash_manifest,
    load_yaml,
    redact,
    relative_posix,
    safe_resolve,
    sha256_file,
    sha256_text,
    utc_now_iso,
)
from framework.model_client import ModelClient
from framework.model_visible_bundle import stage_model_workspace
from tools.xtb_parse import parse_xtb_output
from tools.xtb_run import XTBRunner, declared_scan_point_count


FINALIZATION_TOOL_NAMES = frozenset({"read_file", "run_python", "write_file", "xtb_parse"})


class AgentError(RuntimeError):
    pass


def render_finalization_notice(trigger: str, remaining_turns: int) -> str:
    """Explain the real cutoff and the exact submission-only allowance."""
    labels = {
        "agent_turns": "Agent-turn",
        "wall_seconds": "wall-time",
        "tool_calls": "tool-call",
        "xtb_jobs": "xTB-job",
        "xtb_scan_points": "xTB scan-point",
        "main_compute_credits": "main scientific compute-credit",
        "recovery_compute_credits": "recovery compute-credit",
    }
    label = labels.get(str(trigger), str(trigger).replace("_", " ") or "active work")
    turns = max(0, int(remaining_turns))
    return (
        f"Active {label} budget exhausted. You have exactly {turns} submission-only "
        "finalization turns remaining. No new xTB jobs are available. Immediately write and "
        "validate the required machine-readable submission from existing evidence; submit an "
        "honest partial answer rather than omitting required files."
    )


def reconstruct_observable_messages(trajectory_path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Rebuild the provider-visible conversation from an observable JSONL trace.

    Hidden reasoning is never present in the trace.  Tool results are paired in
    execution order with their preceding observable tool-call records.
    """
    path = Path(trajectory_path)
    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []
    assistant_turns = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AgentError(f"invalid trajectory JSON on line {line_number}") from exc
        event_type = event.get("type")
        if event_type == "message":
            role = event.get("role")
            if role not in {"system", "user", "assistant"}:
                continue
            message: dict[str, Any] = {"role": role, "content": str(event.get("content") or "")}
            tool_calls = event.get("tool_calls")
            if role == "assistant":
                assistant_turns += 1
                if isinstance(tool_calls, list) and tool_calls:
                    message["tool_calls"] = tool_calls
            messages.append(message)
        elif event_type == "tool_call":
            pending_calls.append(
                {
                    "id": str(event.get("tool_call_id") or ""),
                    "name": str(event.get("name") or ""),
                }
            )
        elif event_type == "tool_result":
            if not pending_calls:
                raise AgentError(f"unpaired tool result on trajectory line {line_number}")
            call = pending_calls.pop(0)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(event.get("result", {}), ensure_ascii=False, sort_keys=True),
                }
            )
    if pending_calls:
        raise AgentError("trajectory ends with an unpaired observable tool call")
    if len(messages) < 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise AgentError("trajectory does not begin with observable system and user messages")
    return messages, assistant_turns


def _trajectory_elapsed_seconds(path: Path) -> float:
    timestamps: list[dt.datetime] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            value = json.loads(raw).get("time")
            if isinstance(value, str):
                timestamps.append(dt.datetime.fromisoformat(value.replace("Z", "+00:00")))
        except (json.JSONDecodeError, ValueError):
            continue
    if len(timestamps) < 2:
        return 0.0
    return max(0.0, (max(timestamps) - min(timestamps)).total_seconds())


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file inside generated_files, using a path relative to generated_files. "
                "For example, read registered_output_paths returned by xtb_run. This tool cannot read "
                "the authoritative xtb_jobs evidence tree."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file inside the run workspace.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "content"],
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a Python script inside the workspace with network APIs disabled at language level.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["script_path", "args"],
                "properties": {
                    "script_path": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xtb_run",
            "description": "Directly run xTB with the exact supplied arguments; no defaults or retries are added.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["geometry", "arguments", "namespace", "timeout_seconds"],
                "properties": {
                    "geometry": {
                        "type": "string",
                        "description": "Workspace-relative path to a single-frame XYZ file (for example inputs/start.xyz). Do not pass XYZ text or an absolute path.",
                    },
                    "arguments": {"type": "array", "items": {"type": "string"}},
                    "detailed_input": {
                        "type": ["string", "null"],
                        "description": "Workspace-relative path to an existing xTB detailed-input file, or null. Do not pass file contents.",
                    },
                    "namespace": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                    "retry_of": {"type": ["string", "null"]},
                    "parent_job_id": {"type": ["string", "null"]},
                    "candidate_id": {"type": ["string", "null"]},
                    "condition": {"enum": ["gas", "water", None]},
                    "recovery_of_job_id": {"type": ["string", "null"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "xtb_parse",
            "description": (
                "Parse task-limited factual fields from authoritative xTB evidence. Set output_path to "
                "the exact workdir returned by xtb_run (an xtb_jobs/... path). Do not pass any value "
                "from registered_output_paths; those generated_files copies are for read_file and for "
                "chaining geometries into later xtb_run calls."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["output_path"],
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Exact xtb_jobs/... workdir returned by xtb_run.",
                    }
                },
            },
        },
    },
]


def _tool_names() -> list[str]:
    return [str(item["function"]["name"]) for item in TOOL_SCHEMAS]


def _protected(path: Path, workspace: Path) -> bool:
    relative = path.resolve().relative_to(workspace.resolve()).parts
    return bool(relative and relative[0] in {"xtb_jobs", "registered_outputs", "run_manifest.json", "manifest.json", "trajectory.jsonl", "tool_calls.jsonl", ".agent"})


def _network_disabled_wrapper() -> str:
    return textwrap.dedent(
        """
        import multiprocessing, os, runpy, socket, subprocess, sys
        def denied(*args, **kwargs):
            raise RuntimeError("network and child-process access are disabled in run_python")
        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        socket.create_connection = denied
        socket.getaddrinfo = denied
        subprocess.Popen = denied
        subprocess.run = denied
        subprocess.call = denied
        subprocess.check_call = denied
        subprocess.check_output = denied
        os.system = denied
        os.popen = denied
        for name in tuple(dir(os)):
            if name.startswith("exec") or name.startswith("spawn"):
                setattr(os, name, denied)
        multiprocessing.Process = denied
        multiprocessing.get_context = denied
        script = sys.argv[1]
        sys.argv = sys.argv[1:]
        # ``-I`` intentionally removes cwd from sys.path. Restore only the
        # directory containing the model-authored entrypoint so that local
        # helper modules work without exposing host Python packages.
        sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
        runpy.run_path(script, run_name="__main__")
        """
    )


def restricted_python_environment(workspace: str | Path) -> dict[str, str]:
    """Return the complete environment visible to model-authored Python."""

    return {
        "PATH": "",
        "PYTHONUNBUFFERED": "1",
        "HOME": str(Path(workspace).resolve()),
    }


class AgentRun:
    def __init__(
        self,
        *,
        task_dir: str | Path,
        output_root: str | Path,
        model_client: ModelClient,
        xtb_executable: str | Path = "xtb",
        seed: int | None = None,
        run_id: str | None = None,
        budget_ledger: Any | None = None,
    ) -> None:
        self.task_dir = Path(task_dir).resolve()
        self.task_config = load_yaml(self.task_dir / "config" / "task.yaml", default={}) or {}
        if not isinstance(self.task_config, dict):
            raise AgentError("task.yaml must contain an object")
        self.run_id = run_id or f"{self.task_config.get('task_id','task')}_{utc_now_iso().replace(':','').replace('-','')}_{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(output_root).resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.workspace = self.run_dir / "generated_files"
        self.workspace.mkdir()
        stage_model_workspace(self.task_dir, self.workspace)
        self.model_client = model_client
        self.budget_ledger = budget_ledger
        self.seed = seed
        self.trajectory_path = self.run_dir / "trajectory.jsonl"
        self.tool_calls_path = self.run_dir / "tool_calls.jsonl"
        self.conversation_path = self.run_dir / "conversation.md"
        # A zero-call API failure is still a complete observable trace and must
        # carry the required (empty) JSONL artifacts.
        self.trajectory_path.touch()
        self.tool_calls_path.touch()
        self.conversation_path.touch()
        self.xtb_runner = XTBRunner(task_dir=self.task_dir, run_root=self.run_dir, xtb_executable=xtb_executable)
        self._start = time.monotonic()
        self._active_elapsed_before = 0.0
        self._resume_count = 0
        self._turns = 0
        self._write_manifest(status="running")

    @classmethod
    def resume_existing(
        cls,
        *,
        task_dir: str | Path,
        run_dir: str | Path,
        model_client: ModelClient,
        xtb_executable: str | Path = "xtb",
        seed: int | None = None,
        budget_ledger: Any | None = None,
    ) -> "AgentRun":
        """Resume an API-interrupted run without repeating completed tool work."""
        self = cls.__new__(cls)
        self.task_dir = Path(task_dir).resolve()
        self.task_config = load_yaml(self.task_dir / "config" / "task.yaml", default={}) or {}
        if not isinstance(self.task_config, dict):
            raise AgentError("task.yaml must contain an object")
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.workspace = self.run_dir / "generated_files"
        self.trajectory_path = self.run_dir / "trajectory.jsonl"
        self.tool_calls_path = self.run_dir / "tool_calls.jsonl"
        self.conversation_path = self.run_dir / "conversation.md"
        manifest_path = self.run_dir / "manifest.json"
        required_paths = (self.workspace, self.trajectory_path, self.tool_calls_path, self.conversation_path, manifest_path)
        if not all(path.exists() for path in required_paths):
            raise AgentError("resume target is missing required observable artifacts")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_task = self.task_config.get("task_id")
        if previous.get("task_id") != expected_task or previous.get("seed") != seed:
            raise AgentError("resume target task or seed does not match")
        retryable_tokens = (
            "ModelClientError",
            "RemoteDisconnected",
            "IncompleteRead",
            "TimeoutError",
            "SSLError",
            "ConnectionResetError",
            "ApiPolicyError",
        )
        if previous.get("status") != "failed" or not any(
            token in str(previous.get("error") or "") for token in retryable_tokens
        ):
            raise AgentError("only an observable provider or transport failure may be resumed")
        if previous.get("task_config_sha256") != sha256_file(self.task_dir / "config" / "task.yaml"):
            raise AgentError("task configuration changed since the failed run")
        messages, turns = reconstruct_observable_messages(self.trajectory_path)
        self._resume_messages = messages
        self.model_client = model_client
        self.budget_ledger = budget_ledger
        self.seed = seed
        self.xtb_runner = XTBRunner(task_dir=self.task_dir, run_root=self.run_dir, xtb_executable=xtb_executable)
        self._start = time.monotonic()
        self._active_elapsed_before = float(previous.get("active_elapsed_seconds") or _trajectory_elapsed_seconds(self.trajectory_path))
        self._resume_count = int(previous.get("resume_count", 0)) + 1
        self._turns = turns
        self._started_at = str(previous.get("started_at") or utc_now_iso())
        append_jsonl(
            self.run_dir / "resume_history.jsonl",
            {
                "time": utc_now_iso(),
                "reason": str(previous.get("error") or ""),
                "previous_manifest_sha256": sha256_file(manifest_path),
                "observable_turns_completed": turns,
                "active_elapsed_seconds_before_resume": self._active_elapsed_before,
            },
        )
        append_jsonl(
            self.trajectory_path,
            {"type": "resume", "time": utc_now_iso(), "reason": "observable model API failure", "resume_count": self._resume_count},
        )
        self._write_manifest(status="running")
        return self

    def _write_manifest(self, *, status: str, error: str | None = None) -> None:
        budgets = self.task_config.get("budgets", {}) or {}
        question = (self.task_dir / "question.md").read_text(encoding="utf-8")
        task_config_path = self.task_dir / "config" / "task.yaml"
        payload = {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "task_id": self.task_config.get("task_id"),
            "task_config_sha256": sha256_file(task_config_path),
            "public_inputs_sha256": file_hash_manifest(self.task_dir / "data" / "inputs")
            if (self.task_dir / "data" / "inputs").is_dir()
            else {},
            "model": self.model_client.metadata,
            "seed": self.seed,
            "tool_schema_sha256": sha256_text(json.dumps(TOOL_SCHEMAS, sort_keys=True)),
            "user_prompt_sha256": sha256_text(question),
            "budgets": budgets,
            "started_at": getattr(self, "_started_at", utc_now_iso()),
            "updated_at": utc_now_iso(),
            "active_elapsed_seconds": self._active_elapsed_before + max(0.0, time.monotonic() - self._start),
            "resume_count": self._resume_count,
            "status": status,
            "error": error,
        }
        if not hasattr(self, "_started_at"):
            self._started_at = payload["started_at"]
        atomic_write_json(self.run_dir / "manifest.json", payload)

    def _log_message(self, role: str, content: str | None, *, tool_calls: Any = None) -> None:
        event = {"type": "message", "time": utc_now_iso(), "role": role, "content": content or ""}
        if tool_calls is not None:
            event["tool_calls"] = redact(tool_calls)
        append_jsonl(self.trajectory_path, event)
        with self.conversation_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## {role}\n\n{content or ''}\n")
            if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
                handle.write("\n### observable tool calls\n\n")
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
                    handle.write(
                        f"- `{function.get('name', '')}` id=`{call.get('id', '') if isinstance(call, dict) else ''}` "
                        f"arguments=`{function.get('arguments', '{}')}`\n"
                    )

    def _log_tool_result(self, tool_call_id: str, name: str, result: dict[str, Any]) -> None:
        """Render the same observable tool result already stored in JSONL."""
        with self.conversation_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## tool result: {name}\n\n")
            handle.write(f"tool_call_id: `{tool_call_id}`\n\n")
            handle.write("```json\n")
            handle.write(json.dumps(redact(result), ensure_ascii=False, sort_keys=True, indent=2))
            handle.write("\n```\n")

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = safe_resolve(self.workspace, str(args.get("path", "")), must_exist=True)
        if not path.is_file():
            raise AgentError("read_file target is not a file")
        data = path.read_bytes()
        if len(data) > 2_000_000:
            raise AgentError("read_file limit is 2 MB")
        return {"path": relative_posix(path, self.workspace), "content": data.decode("utf-8"), "sha256": sha256_file(path)}

    def _write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = safe_resolve(self.workspace, str(args.get("path", "")))
        if _protected(path, self.workspace):
            raise AgentError("write_file cannot modify protected evidence paths")
        content = args.get("content")
        if not isinstance(content, str) or len(content.encode("utf-8")) > 5_000_000:
            raise AgentError("write_file content must be UTF-8 text no larger than 5 MB")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, content)
        event = {"type": "file_write", "time": utc_now_iso(), "path": relative_posix(path, self.workspace), "sha256": sha256_file(path)}
        append_jsonl(self.trajectory_path, event)
        return {"path": event["path"], "sha256": event["sha256"], "bytes": len(content.encode("utf-8"))}

    def _run_python(self, args: dict[str, Any]) -> dict[str, Any]:
        script = safe_resolve(self.workspace, str(args.get("script_path", "")), must_exist=True)
        if not script.is_file() or script.suffix != ".py":
            raise AgentError("run_python requires a .py file")
        arguments = args.get("args")
        if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
            raise AgentError("run_python args must be an array of strings")
        timeout = min(300, int((self.task_config.get("budgets", {}) or {}).get("max_wall_time_seconds", 300)))
        raw_command = [sys.executable, "-I", "-c", _network_disabled_wrapper(), str(script), *arguments]
        masked_files: dict[Path, Path] = {}
        xtb_runner = getattr(self, "xtb_runner", None)
        xtb_executable = getattr(xtb_runner, "executable", None)
        if xtb_executable is not None:
            blocker = self.workspace / ".agent" / "blocked_executable"
            blocker.parent.mkdir(parents=True, exist_ok=True)
            if not blocker.exists():
                atomic_write_text(blocker, "")
            masked_files[Path(xtb_executable).resolve()] = blocker
        command, sandbox = bubblewrap_command(
            raw_command,
            cwd=self.workspace,
            writable_roots=[self.workspace],
            readonly_roots=[Path(sys.prefix)],
            masked_files=masked_files,
        )
        completed = subprocess.run(
            command,
            cwd=self.workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env=restricted_python_environment(self.workspace),
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-100_000:],
            "stderr": completed.stderr[-100_000:],
            "sandbox": sandbox,
        }

    def _prepare_xtb_request(self, args: dict[str, Any]) -> dict[str, Any]:
        request = dict(args)
        geometry = safe_resolve(self.workspace, str(request.get("geometry", "")), must_exist=True)
        request["geometry"] = relative_posix(geometry, self.run_dir)
        detailed = request.get("detailed_input")
        if isinstance(detailed, str):
            detailed_path = safe_resolve(self.workspace, detailed, must_exist=True)
            request["detailed_input"] = relative_posix(detailed_path, self.run_dir)
        return request

    def _preflight_xtb_run(self, args: dict[str, Any]) -> None:
        self.xtb_runner.validate_request(self._prepare_xtb_request(args))

    def _xtb_run(self, args: dict[str, Any]) -> dict[str, Any]:
        request = self._prepare_xtb_request(args)
        response = self.xtb_runner.run(request)
        # Expose byte-for-byte copies of safe registered outputs in the model
        # workspace. This is necessary for chaining an optimized geometry into
        # a Hessian/scan while preserving the SHA-256 identity required by the
        # evaluator. The authoritative evidence remains under xtb_jobs/.
        workdir = safe_resolve(self.run_dir, str(response.get("workdir", "")), must_exist=True)
        staged: dict[str, str] = {}
        stage_dir = self.workspace / "registered_outputs" / str(response.get("job_id", "unknown"))
        stage_dir.mkdir(parents=True, exist_ok=True)
        allowed_names = {
            "xtbopt.xyz",
            "xtbscan.log",
            "xtbscan.xyz",
            "vibspectrum",
            "stdout.txt",
            "stderr.txt",
            "detailed.inp",
            "request.json",
        }
        for name in sorted(allowed_names):
            source = workdir / name
            if source.is_file():
                target = stage_dir / name
                shutil.copy2(source, target)
                staged[name] = relative_posix(target, self.workspace)
        response["registered_output_paths"] = staged
        return response

    def _xtb_parse(self, args: dict[str, Any]) -> dict[str, Any]:
        output = safe_resolve(self.run_dir, str(args.get("output_path", "")), must_exist=True)
        jobs_root = (self.run_dir / "xtb_jobs").resolve()
        try:
            output.resolve().relative_to(jobs_root)
        except ValueError as exc:
            raise AgentError("xtb_parse may read only registered xtb_jobs paths") from exc
        parsed = parse_xtb_output(output)
        append_jsonl(self.trajectory_path, {"type": "feedback", "time": utc_now_iso(), "output_path": relative_posix(output, self.run_dir), "result": parsed})
        return parsed

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "run_python": self._run_python,
            "xtb_run": self._xtb_run,
            "xtb_parse": self._xtb_parse,
        }
        if set(handlers) != set(_tool_names()):
            raise AssertionError("tool implementation does not match public schema")
        if name not in handlers:
            raise AgentError(f"unknown tool: {name}")
        if getattr(self, "_in_finalization", False) and name not in FINALIZATION_TOOL_NAMES:
            result = {"error": f"AgentError: tool unavailable during finalization: {name}"}
            append_jsonl(
                self.trajectory_path,
                {
                    "type": "tool_result",
                    "time": utc_now_iso(),
                    "tool": name,
                    "status": "error",
                    "elapsed_seconds": 0.0,
                    "result": result,
                },
            )
            return result
        started = time.monotonic()
        weighted_job_id: str | None = None
        try:
            if self.budget_ledger is not None:
                # Reserve cumulative budget before any file, process, or xTB
                # side effect.  Retries and resumed processes share this same
                # ledger instance/document, so these counters never reset.
                if hasattr(self.budget_ledger, "reserve_xtb_job"):
                    phase = "finalization" if getattr(self, "_in_finalization", False) else "active"
                    tool_decision = self.budget_ledger.record_tool_call(name, phase)
                    if tool_decision is not None and not tool_decision.allowed:
                        raise AgentError(tool_decision.reason or "tool budget exceeded")
                else:
                    self.budget_ledger.record_tool_call(self._turns)
                if name == "xtb_run":
                    # Validate every path and CLI field before reserving scarce
                    # scientific compute.  Rejected requests still consume a
                    # tool call, but they did not launch xTB and therefore are
                    # not scientific jobs.
                    self._preflight_xtb_run(arguments)
                    detailed_input = arguments.get("detailed_input")
                    scan_points = 0
                    if detailed_input is not None:
                        if not isinstance(detailed_input, str) or not detailed_input.strip():
                            raise AgentError("xtb_run detailed_input must be a non-empty relative path")
                        detailed_path = safe_resolve(self.workspace, detailed_input, must_exist=True)
                        scan_points = declared_scan_point_count(detailed_path)
                    if getattr(self.budget_ledger, "compute_budget", None) is not None:
                        next_job = int(self.budget_ledger.totals["xtb_jobs"]) + 1
                        weighted_job_id = f"job_{next_job:04d}"
                        compute_class = "scan" if scan_points else classify_xtb_compute(
                            arguments.get("arguments", [])
                        )
                        job_decision = self.budget_ledger.reserve_xtb_compute(
                            weighted_job_id,
                            compute_class=compute_class,
                            credits=task04_compute_cost(
                                compute_class, self.budget_ledger.compute_budget
                            ),
                            candidate_id=arguments.get("candidate_id"),
                            condition=arguments.get("condition"),
                            parent_job_id=arguments.get("parent_job_id"),
                            recovery_of_job_id=arguments.get("recovery_of_job_id")
                            or arguments.get("retry_of"),
                        )
                        if not job_decision.allowed:
                            raise AgentError(job_decision.reason or "xTB budget exceeded")
                    elif hasattr(self.budget_ledger, "reserve_xtb_job"):
                        next_job = int(self.budget_ledger.totals["xtb_jobs"]) + 1
                        job_decision = self.budget_ledger.reserve_xtb_job(
                            f"job_{next_job:04d}", scan_points=scan_points
                        )
                        if not job_decision.allowed:
                            raise AgentError(job_decision.reason or "xTB budget exceeded")
                    else:
                        self.budget_ledger.record_xtb_job(self._turns, scan_points=scan_points)
            result = handlers[name](arguments)
            if self.budget_ledger is not None and name == "xtb_run":
                if weighted_job_id is not None:
                    outcome = str(result.get("status"))
                    if outcome not in {"completed", "failed", "timeout"}:
                        outcome = "failed"
                    self.budget_ledger.record_xtb_job_outcome(weighted_job_id, outcome)
                elif result.get("status") == "completed":
                    self.budget_ledger.record_xtb_job_completed(self._turns)
            status = "ok"
        except Exception as exc:
            if weighted_job_id is not None:
                try:
                    self.budget_ledger.record_xtb_job_outcome(weighted_job_id, "failed")
                except ValueError:
                    pass
            result = {"error": f"{type(exc).__name__}: {exc}"}
            status = "error"
        event = {
            "type": "tool_result",
            "time": utc_now_iso(),
            "tool": name,
            "status": status,
            "elapsed_seconds": time.monotonic() - started,
            "result": result,
        }
        append_jsonl(self.trajectory_path, event)
        return result

    def _system_prompt(self) -> str:
        required = self.task_config.get("required_outputs", []) or []
        budgets = self.task_config.get("budgets", {}) or {}
        active_turns = int(budgets.get("max_agent_turns", 20))
        finalization_turns = int(budgets.get("finalization_turns", 2))
        submission_paths = [f"submission/{str(name).lstrip('/')}" for name in required]
        for mandatory in ("submission/final_answer.json", "submission/report.md"):
            if mandatory not in submission_paths:
                submission_paths.append(mandatory)
        return (
            "You are executing a standalone goal-driven xTB benchmark. Use only the five supplied tools. "
            "Every file parameter is a workspace-relative path. xtb_run.geometry and detailed_input are paths, never file contents. "
            "The authoritative xTB evidence is under xtb_jobs; registered_output_paths are byte-identical workspace copies for read_file or reuse. "
            "Pass the exact xtb_run.workdir value to xtb_parse.output_path. Do not modify registered evidence or claim hidden references. "
            "Choose the scientific route yourself under the cumulative budget. Every scored claim must link to evidence from this run. "
            f"This run provides exactly {active_turns} active Agent turns followed, if needed, by exactly "
            f"{finalization_turns} submission-only finalization turns. Plan efficiently because Agent-turn "
            "efficiency is recorded and may affect evaluation; scientific correctness and evidence remain primary. "
            "API retries caused by empty, truncated, or transport-failed responses do not consume active Agent turns. "
            "Before scientific work, read submission_contract/README.md and "
            "submission_contract/final_answer.schema.json. "
            "Before completion, validate submission/final_answer.json against the exact public schema; do not invent or rename fields. "
            "Write every model-authored output under submission/ inside the workspace. "
            f"Required output paths: {', '.join(submission_paths)}. "
            "An honest partial answer is allowed. Finish with a brief observable summary; do not provide hidden reasoning."
        )

    def _materialize_submission(self) -> tuple[Path, list[str]]:
        """Snapshot every model-authored required output available at cutoff.

        A turn or wall-time budget is a hard stop on further work, not a reason
        to discard files already written inside the sandbox.  Missing mandatory
        files remain missing and are handled by the deterministic scorer.
        """
        final_dir = self.run_dir / "final_submission"
        final_dir.mkdir(exist_ok=True)
        if self.xtb_runner.jobs_dir.exists():
            shutil.copytree(self.xtb_runner.jobs_dir, final_dir / "xtb_jobs", dirs_exist_ok=True)
        if self.xtb_runner.rejections_path.exists():
            shutil.copy2(self.xtb_runner.rejections_path, final_dir / "tool_rejections.jsonl")
        missing: list[str] = []
        for name in self.task_config.get("required_outputs", []) or []:
            if name == "run_manifest.json":
                source = self.run_dir / "run_manifest.json"
            else:
                contracted_source = self.workspace / "submission" / str(name)
                legacy_source = self.workspace / str(name)
                source = contracted_source if contracted_source.exists() else legacy_source
            target = final_dir / str(name)
            if not source.exists():
                missing.append(str(name))
                continue
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        # ``run_manifest.json`` is generated by the Harness, not authored by
        # the model, so it is intentionally absent from the public required
        # output list.  It must nevertheless accompany every scientific
        # submission for provenance validation and scoring.
        manifest_source = self.run_dir / "run_manifest.json"
        if manifest_source.is_file():
            shutil.copy2(manifest_source, final_dir / "run_manifest.json")
        append_jsonl(
            self.trajectory_path,
            {
                "type": "final_submission",
                "time": utc_now_iso(),
                "missing_required_outputs": missing,
                "files_sha256": file_hash_manifest(final_dir),
            },
        )
        return final_dir, missing

    def _perform_model_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> bool:
        """Execute one observable provider response and return whether it requested tools."""
        response = self.model_client.create_chat_completion(messages=messages, tools=tools, seed=self.seed)
        message = response["message"]
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        observable: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            observable["tool_calls"] = tool_calls
        messages.append(observable)
        self._log_message("assistant", content, tool_calls=tool_calls or None)
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name", ""))
            raw_args = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must decode to an object")
            except Exception as exc:
                arguments = {}
                result = {"error": f"invalid tool arguments: {exc}"}
            else:
                append_jsonl(
                    self.tool_calls_path,
                    {"time": utc_now_iso(), "tool_call_id": call.get("id"), "name": name, "arguments": arguments},
                )
                append_jsonl(
                    self.trajectory_path,
                    {"type": "tool_call", "time": utc_now_iso(), "tool_call_id": call.get("id"), "name": name, "arguments": arguments},
                )
                result = self.execute_tool(name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id", "")),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
                }
            )
            self._log_tool_result(str(call.get("id", "")), name, result)
        return bool(tool_calls)

    def run(self) -> Path:
        question = (self.task_dir / "question.md").read_text(encoding="utf-8")
        if hasattr(self, "_resume_messages"):
            messages = list(self._resume_messages)
            start_turn = self._turns
        else:
            messages = [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": question},
            ]
            self._log_message("system", messages[0]["content"])
            self._log_message("user", question)
            start_turn = 0
        budgets = self.task_config.get("budgets", {}) or {}
        max_turns = int(budgets.get("max_agent_turns", 20))
        wall_limit = float(budgets.get("max_wall_time_seconds", 3600))
        finalization_turns = int(budgets.get("finalization_turns", 2))
        finalization_wall_limit = float(budgets.get("finalization_wall_time_seconds", 600))
        self._in_finalization = False
        try:
            exhausted = False
            for turn in range(start_turn, max_turns):
                self._turns = turn + 1
                if self.budget_ledger is not None:
                    if hasattr(self.budget_ledger, "record_agent_turn"):
                        decision = self.budget_ledger.record_agent_turn("active")
                        if not decision.allowed:
                            exhausted = True
                            break
                    else:
                        self.budget_ledger.begin_turn(self._turns)
                if self._active_elapsed_before + time.monotonic() - self._start > wall_limit:
                    if self.budget_ledger is not None and hasattr(self.budget_ledger, "enter_finalization"):
                        self.budget_ledger.enter_finalization("wall_seconds")
                    exhausted = True
                    break
                if not self._perform_model_turn(messages, TOOL_SCHEMAS):
                    break
                if self.budget_ledger is not None and getattr(self.budget_ledger, "phase", "active") == "finalizing":
                    exhausted = True
                    break
            else:
                exhausted = True
                if self.budget_ledger is not None and hasattr(self.budget_ledger, "enter_finalization"):
                    self.budget_ledger.enter_finalization("agent_turns")

            if exhausted:
                if self.budget_ledger is not None and hasattr(self.budget_ledger, "enter_finalization"):
                    self.budget_ledger.enter_finalization(
                        str(getattr(self.budget_ledger, "exhaustion_trigger", None) or "agent_turns")
                    )
                self._in_finalization = True
                trigger = str(getattr(self.budget_ledger, "exhaustion_trigger", None) or "agent_turns")
                notice_text = render_finalization_notice(trigger, finalization_turns)
                notice = {"role": "user", "content": notice_text}
                messages.append(notice)
                self._log_message("user", notice_text)
                finalization_started = time.monotonic()
                allowed_tools = [
                    schema
                    for schema in TOOL_SCHEMAS
                    if schema["function"]["name"] in FINALIZATION_TOOL_NAMES
                ]
                for _ in range(finalization_turns):
                    if time.monotonic() - finalization_started > finalization_wall_limit:
                        break
                    if self.budget_ledger is not None and hasattr(self.budget_ledger, "record_agent_turn"):
                        decision = self.budget_ledger.record_agent_turn("finalization")
                        if not decision.allowed:
                            break
                    if not self._perform_model_turn(messages, allowed_tools):
                        break
            final_dir, missing = self._materialize_submission()
            self._write_manifest(status="completed" if not missing else "incomplete", error=None if not missing else f"missing: {missing}")
            return final_dir
        except Exception as exc:
            try:
                self._materialize_submission()
            except Exception:
                # Preserve the original execution error; the campaign wrapper
                # still snapshots the registered run manifest and xTB jobs.
                pass
            self._write_manifest(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--xtb", default="xtb")
    args = parser.parse_args(argv)
    client = ModelClient(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        temperature=args.temperature,
    )
    final_dir = AgentRun(
        task_dir=args.task_dir,
        output_root=args.output_root,
        model_client=client,
        xtb_executable=args.xtb,
        seed=args.seed,
    ).run()
    print(final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Seven generic model-facing tools for one disposable run."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.runner import JobRequest, RunState, _text
from runtime.sandbox import build_bubblewrap_command
from runtime.staging import ModelWorkspace

if TYPE_CHECKING:
    from runtime.task_adapter import TaskRuntimeConfig

READ_FILE_DEFAULT_CHARS = 24_000
READ_FILE_MAX_CHARS = 32_000
RUN_PYTHON_RESPONSE_CHARS = 24_000


class ToolRuntime:
    def __init__(
        self,
        workspace: ModelWorkspace | None = None,
        state: RunState | None = None,
        task_config: TaskRuntimeConfig | None = None,
        *,
        python_executable: Path | None = None,
        runtime_paths: tuple[Path, ...] = (),
        bubblewrap_executable: Path = Path("/usr/bin/bwrap"),
    ) -> None:
        self.workspace = workspace
        self.state = state
        self.task_config = task_config
        self.python_executable = python_executable
        self.runtime_paths = runtime_paths
        self.bubblewrap_executable = bubblewrap_executable

    def tool_definitions(self) -> list[dict[str, object]]:
        path = {"type": "string", "minLength": 1}
        readable_root = {
            "type": "string",
            "enum": ["public", "workspace", "submission"],
        }
        writable_root = {"type": "string", "enum": ["workspace", "submission"]}
        empty = {"type": "object", "properties": {}, "additionalProperties": False}
        return [
            {
                "name": "list_files",
                "description": "List files in a visible directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {"root": readable_root},
                    "required": ["root"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_file",
                "description": (
                    "Read a bounded UTF-8 file chunk. Use next_offset to continue "
                    "when truncated is true."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "root": readable_root,
                        "path": path,
                        "offset": {"type": "integer", "minimum": 0},
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": READ_FILE_MAX_CHARS,
                        },
                    },
                    "required": ["root", "path"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "write_file",
                "description": "Write a UTF-8 file in a writable directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "root": writable_root,
                        "path": path,
                        "content": {"type": "string"},
                    },
                    "required": ["root", "path", "content"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_python",
                "description": "Run a Python file whose path is relative to workspace.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": path},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "xtb_run",
                "description": (
                    "Run one configured calculation; input_path and control_path are "
                    "relative to workspace, so copy public inputs there first."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["sp", "opt", "hess", "scan"],
                        },
                        "input_path": path,
                        "control_path": path,
                        "charge": {"type": "integer"},
                        "uhf": {"type": "integer", "minimum": 0},
                        "solvent": {"type": ["string", "null"]},
                    },
                    "required": ["operation", "input_path"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "budget_status",
                "description": "Return current run counters.",
                "input_schema": empty,
            },
            {
                "name": "submit",
                "description": "Finish the run with the submission directory.",
                "input_schema": empty,
            },
        ]

    def _ready(self) -> tuple[ModelWorkspace, RunState, TaskRuntimeConfig]:
        if self.workspace is None or self.state is None or self.task_config is None:
            raise RuntimeError("tool runtime is not initialized")
        return self.workspace, self.state, self.task_config

    @staticmethod
    def _path(root: Path, relative: str) -> Path:
        supplied = Path(relative)
        if supplied.is_absolute() or ".." in supplied.parts:
            raise ValueError("path must stay inside its visible directory")
        resolved_root = root.resolve()
        resolved = (resolved_root / supplied).resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValueError("path must stay inside its visible directory")
        return resolved

    def _root(self, name: str) -> Path:
        workspace, _, _ = self._ready()
        roots = {
            "public": workspace.public_root,
            "workspace": workspace.workspace_root,
            "submission": workspace.submission_root,
        }
        try:
            return roots[name]
        except KeyError as exc:
            raise ValueError(f"unknown visible root: {name}") from exc

    def _run_python(self, relative: str) -> dict[str, object]:
        workspace, state, config = self._ready()
        script = self._path(workspace.workspace_root, relative)
        if not script.is_file():
            raise FileNotFoundError(script)
        if self.python_executable is None:
            raise RuntimeError("sandbox Python is not configured")
        command = build_bubblewrap_command(
            workspace.public_root,
            workspace.workspace_root,
            workspace.submission_root,
            [str(self.python_executable), f"/model/workspace/{Path(relative).as_posix()}"],
            runtime_paths=self.runtime_paths,
            masked_paths=(state.xtb_executable,),
            bubblewrap_executable=self.bubblewrap_executable,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                check=False,
                env={
                    "PATH": os.defpath,
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                },
            )
            return self._python_result(
                relative=relative,
                status="completed" if completed.returncode == 0 else "failed",
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return self._python_result(
                relative=relative,
                status="timeout",
                return_code=None,
                stdout=_text(exc.stdout),
                stderr=_text(exc.stderr),
            )
        except OSError as exc:
            return self._python_result(
                relative=relative,
                status="failed",
                return_code=None,
                stdout="",
                stderr=str(exc),
            )

    def _python_result(
        self,
        *,
        relative: str,
        status: str,
        return_code: int | None,
        stdout: str,
        stderr: str,
    ) -> dict[str, object]:
        workspace, _, _ = self._ready()
        stdout_bytes = stdout.encode("utf-8")
        stderr_bytes = stderr.encode("utf-8")
        digest = hashlib.sha256(
            Path(relative).as_posix().encode("utf-8")
            + b"\0"
            + stdout_bytes
            + b"\0"
            + stderr_bytes
        ).hexdigest()
        output_root = workspace.workspace_root / "tool_outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        stdout_relative = Path("tool_outputs") / f"run_python_{digest}.stdout.txt"
        stderr_relative = Path("tool_outputs") / f"run_python_{digest}.stderr.txt"
        (workspace.workspace_root / stdout_relative).write_bytes(stdout_bytes)
        (workspace.workspace_root / stderr_relative).write_bytes(stderr_bytes)
        return {
            "status": status,
            "return_code": return_code,
            "stdout": stdout[:RUN_PYTHON_RESPONSE_CHARS],
            "stderr": stderr[:RUN_PYTHON_RESPONSE_CHARS],
            "stdout_total_chars": len(stdout),
            "stderr_total_chars": len(stderr),
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
            "stdout_truncated": len(stdout) > RUN_PYTHON_RESPONSE_CHARS,
            "stderr_truncated": len(stderr) > RUN_PYTHON_RESPONSE_CHARS,
            "stdout_path": stdout_relative.as_posix(),
            "stderr_path": stderr_relative.as_posix(),
        }

    def dispatch(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        _, state, config = self._ready()
        if name == "list_files":
            root = self._root(str(arguments["root"]))
            return {
                "files": sorted(
                    path.relative_to(root).as_posix()
                    for path in root.rglob("*")
                    if path.is_file()
                )
            }
        if name == "read_file":
            supported = {"root", "path", "offset", "max_chars"}
            unknown = sorted(set(arguments) - supported)
            if unknown:
                raise ValueError(f"unsupported read_file argument: {unknown[0]}")
            root = self._root(str(arguments["root"]))
            path = self._path(root, str(arguments["path"]))
            text = path.read_text(encoding="utf-8")
            offset = arguments.get("offset", 0)
            max_chars = arguments.get("max_chars", READ_FILE_DEFAULT_CHARS)
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise ValueError("read_file offset must be a non-negative integer")
            if (
                not isinstance(max_chars, int)
                or isinstance(max_chars, bool)
                or max_chars < 1
                or max_chars > READ_FILE_MAX_CHARS
            ):
                raise ValueError(
                    f"read_file max_chars must be between 1 and {READ_FILE_MAX_CHARS}"
                )
            if offset > len(text):
                raise ValueError("read_file offset is beyond end of file")
            content = text[offset : offset + max_chars]
            end = offset + len(content)
            truncated = end < len(text)
            return {
                "content": content,
                "offset": offset,
                "returned_chars": len(content),
                "total_chars": len(text),
                "next_offset": end if truncated else None,
                "truncated": truncated,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        if name == "write_file":
            root_name = str(arguments["root"])
            if root_name not in {"workspace", "submission"}:
                raise ValueError("write_file root must be writable")
            root = self._root(root_name)
            path = self._path(root, str(arguments["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(arguments["content"]), encoding="utf-8", newline="\n")
            return {"path": path.relative_to(root).as_posix()}
        if name == "run_python":
            return self._run_python(str(arguments["path"]))
        if name == "xtb_run":
            supported = {
                "operation",
                "input_path",
                "control_path",
                "charge",
                "uhf",
                "solvent",
            }
            unknown = sorted(set(arguments) - supported)
            if unknown:
                raise ValueError(f"unsupported xtb_run argument: {unknown[0]}")
            request = JobRequest(
                operation=str(arguments["operation"]),
                input_path=str(arguments["input_path"]),
                charge=int(arguments.get("charge", config.default_charge)),
                uhf=int(arguments.get("uhf", config.default_uhf)),
                solvent=(
                    None if arguments.get("solvent") is None else str(arguments["solvent"])
                ),
                control_path=(
                    None
                    if arguments.get("control_path") is None
                    else str(arguments["control_path"])
                ),
            )
            return state.run_xtb(request)
        if name == "budget_status":
            return state.budget_status()
        if name == "submit":
            return state.submit(config.required_outputs)
        raise ValueError(f"unknown tool: {name}")

    def finalize_at_turn_limit(self) -> dict[str, object]:
        """Freeze the model's current files for grading without repairing them."""
        _, state, config = self._ready()
        return state.submit(config.required_outputs, allow_incomplete=True)

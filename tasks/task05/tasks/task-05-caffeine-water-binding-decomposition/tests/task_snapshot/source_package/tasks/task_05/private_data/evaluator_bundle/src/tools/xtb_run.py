"""Direct, non-planning xTB subprocess wrapper.

The wrapper implements the public request/response contract in
``common/minimal_tool_contract.md``.  It never changes scientific arguments,
never retries automatically, and records a SHA-256 evidence chain for each
isolated job.
"""

from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import psutil

try:  # Keep parser/evaluator imports portable; execution is Linux-only.
    import resource
except ImportError:  # pragma: no cover - exercised on Windows
    resource = None  # type: ignore[assignment]

from common.io_utils import (
    FileLock,
    append_jsonl,
    atomic_write_json,
    bubblewrap_command,
    file_hash_manifest,
    load_yaml,
    relative_posix,
    require_linux_execution,
    safe_resolve,
    sanitize_environment,
    sha256_file,
    utc_now_iso,
    validate_namespace,
)


class XTBRunError(RuntimeError):
    """Base error for rejected or failed wrapper operations."""


class BudgetExceededError(XTBRunError):
    pass


class VersionMismatchError(XTBRunError):
    pass


_FORBIDDEN_ARGUMENT = re.compile(r"[\x00\r\n;|&<>`]")
_VERSION = re.compile(r"(?:xtb\s+version|version)\s*[:=]?\s*v?([0-9]+(?:\.[0-9]+){1,2})", re.I)
_SCAN_BLOCK = re.compile(r"(?ims)^\s*\$scan\s*$\n(?P<body>.*?)^\s*\$end\s*$")
_SCAN_SCHEDULE = re.compile(
    r"^\s*\d+\s*:\s*(?P<start>[-+0-9.eEdD]+)\s*,\s*(?P<stop>[-+0-9.eEdD]+)\s*,\s*(?P<points>\d+)\s*$"
)


def _evidence_tree_sha256(files_sha256: Mapping[str, str]) -> str:
    """Frozen evidence tree: sorted path + NUL + SHA-256 + LF records."""
    digest = hashlib.sha256()
    for path in sorted(files_sha256):
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(files_sha256[path]).lower().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def declared_scan_point_count(detailed_input: str | Path) -> int:
    """Return the declared one-dimensional xTB scan size, or zero.

    The budget is reserved from the official ``$scan`` schedule before xTB is
    launched.  Ambiguous or malformed schedules fail closed instead of being
    treated as free work.
    """

    path = Path(detailed_input)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise XTBRunError(f"cannot read detailed input for scan accounting: {path}") from exc
    blocks = list(_SCAN_BLOCK.finditer(text))
    if not blocks:
        return 0
    if len(blocks) != 1:
        raise XTBRunError("detailed input must contain at most one $scan block")
    schedules = [
        line.split("#", 1)[0].strip()
        for line in blocks[0].group("body").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if len(schedules) != 1:
        raise XTBRunError("only one one-dimensional $scan schedule is supported")
    match = _SCAN_SCHEDULE.fullmatch(schedules[0])
    if match is None:
        raise XTBRunError("malformed $scan schedule; expected 'index: start, stop, points'")
    points = int(match.group("points"))
    if points < 2:
        raise XTBRunError("a scan schedule must declare at least two points")
    start = float(match.group("start").replace("d", "e").replace("D", "E"))
    stop = float(match.group("stop").replace("d", "e").replace("D", "E"))
    span = abs(stop - start)
    at_branch_cut = abs(start % 360.0) <= 1e-9 and abs(stop % 360.0) <= 1e-9
    if at_branch_cut and abs(span - 360.0) <= 1e-9:
        raise XTBRunError(
            "periodic branch cut makes 0/360-equivalent scan endpoints ambiguous; "
            "use an offset unwrapped interval and verify the actual frames"
        )
    return points


def _runtime_root(executable: Path) -> Path:
    """Return the narrowest useful runtime prefix without ever selecting /."""

    parent = executable.parent
    candidate = parent.parent if parent.name == "bin" else parent
    return parent if candidate == Path(candidate.anchor) else candidate


def _load_task_config(task_dir: Path) -> dict[str, Any]:
    value = load_yaml(task_dir / "config" / "task.yaml", default={}) or {}
    if not isinstance(value, dict):
        raise XTBRunError("task.yaml must contain an object")
    return value


def _preexec(memory_mb: int) -> None:
    if resource is None:  # pragma: no cover - guarded by require_linux_execution
        raise RuntimeError("POSIX resource limits are unavailable")
    os.setsid()
    if memory_mb > 0:
        limit = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _parse_version(text: str) -> str | None:
    match = _VERSION.search(text)
    return match.group(1) if match else None


def _validate_arguments(arguments: Any) -> list[str]:
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise XTBRunError("arguments must be a JSON array of strings")
    result: list[str] = []
    for value in arguments:
        if len(value) > 4096:
            raise XTBRunError("an xTB argument exceeds 4096 characters")
        if _FORBIDDEN_ARGUMENT.search(value):
            raise XTBRunError(f"argument contains a forbidden shell/control character: {value!r}")
        result.append(value)
    return result


def _staged_arguments(arguments: list[str], staged_input_name: str) -> list[str]:
    """Bind an explicit ``--input`` argument to the sandbox-local copy."""

    positions = [
        index
        for index, value in enumerate(arguments)
        if value == "--input" or value.startswith("--input=")
    ]
    if len(positions) > 1:
        raise XTBRunError("--input may appear exactly once when detailed_input is supplied")
    normalized = list(arguments)
    if not positions:
        return [*normalized, "--input", staged_input_name]
    index = positions[0]
    if normalized[index] == "--input":
        if index + 1 >= len(normalized) or normalized[index + 1].startswith("--"):
            raise XTBRunError("--input requires one path argument")
        normalized[index + 1] = staged_input_name
    else:
        normalized[index] = f"--input={staged_input_name}"
    return normalized


def _environment_summary(environment: Mapping[str, str], executable: Path, version: str) -> dict[str, Any]:
    keys = ["PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XTB_NUM_THREADS", "HOME", "LANG"]
    return {
        "captured_at": utc_now_iso(),
        "xtb_executable": str(executable),
        "xtb_executable_sha256": sha256_file(executable) if executable.is_file() else None,
        "xtb_version": version,
        "environment": {key: environment.get(key) for key in keys if key in environment},
    }


class XTBRunner:
    """State is persisted in ``run_manifest.json`` under a run workspace."""

    def __init__(
        self,
        *,
        task_dir: str | Path,
        run_root: str | Path,
        xtb_executable: str | Path = "xtb",
        base_environment: Mapping[str, str] | None = None,
    ) -> None:
        require_linux_execution("xTB runner")
        self.task_dir = Path(task_dir).resolve()
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.task_config = _load_task_config(self.task_dir)
        self.budgets = self.task_config.get("budgets", {}) or {}
        self.protocol = self.task_config.get("xtb_protocol", {}) or {}
        resolved = shutil.which(str(xtb_executable))
        if resolved is None:
            candidate = Path(xtb_executable)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                resolved = str(candidate.resolve())
        if resolved is None:
            raise FileNotFoundError(f"xTB executable not found: {xtb_executable}")
        self.executable = Path(resolved).resolve()
        self.environment = sanitize_environment(base_environment)
        threads = str(int(self.budgets.get("cpu_threads", 1)))
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XTB_NUM_THREADS"):
            self.environment[key] = threads
        self.environment.setdefault("PYTHONUNBUFFERED", "1")
        self.jobs_dir = self.run_root / "xtb_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_root / "run_manifest.json"
        self.lock_path = self.run_root / ".run_manifest.lock"
        self.rejections_path = self.run_root / "tool_rejections.jsonl"
        # Fail before reserving any job if the mandatory sandbox is unavailable.
        _, self.sandbox_summary = bubblewrap_command(
            [str(self.executable), "--version"],
            cwd=self.run_root,
            writable_roots=[self.run_root],
            readonly_roots=[_runtime_root(self.executable), self.executable],
        )
        self._version = self._verify_version()
        self._initialize_manifest()

    def _verify_version(self) -> str:
        command, _ = bubblewrap_command(
            [str(self.executable), "--version"],
            cwd=self.run_root,
            writable_roots=[self.run_root],
            readonly_roots=[_runtime_root(self.executable), self.executable],
        )
        completed = subprocess.run(
            command,
            cwd=self.run_root,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        version = _parse_version(completed.stdout or "")
        expected = str(self.protocol.get("version", ""))
        if completed.returncode != 0 or version is None:
            raise VersionMismatchError(f"unable to determine xTB version from: {(completed.stdout or '')[:500]}")
        if expected and version != expected:
            raise VersionMismatchError(f"xTB version mismatch: expected {expected}, got {version}")
        return version

    def _initialize_manifest(self) -> None:
        with FileLock(self.lock_path):
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if manifest.get("task_id") != self.task_config.get("task_id"):
                    raise XTBRunError("existing run_manifest.json belongs to another task")
                return
            task_config_path = self.task_dir / "config" / "task.yaml"
            public_inputs = self.task_dir / "data" / "inputs"
            project_root = Path(__file__).resolve().parents[1]
            lock_candidates = [
                project_root / "environment" / "conda-linux-64.lock.json",
                project_root / "environment" / "conda-linux-64.explicit.txt",
                project_root / "environment" / "conda-linux-64.lock",
                project_root / "environment" / "environment-linux-64.lock",
                project_root / "environment" / "environment.yml",
            ]
            lock_path = next((path for path in lock_candidates if path.is_file()), None)
            public_hashes = file_hash_manifest(public_inputs) if public_inputs.is_dir() else {}
            manifest = {
                "schema_version": "2.0",
                "task_id": self.task_config.get("task_id"),
                "task_version": self.task_config.get("task_version"),
                "created_at": utc_now_iso(),
                "xtb_version": self._version,
                "environment_summary": _environment_summary(self.environment, self.executable, self._version),
                "provenance": {
                    "task_config": {
                        "path": relative_posix(task_config_path, self.task_dir),
                        "sha256": sha256_file(task_config_path),
                    },
                    "public_inputs": {
                        "files_sha256": public_hashes,
                        "tree_sha256": __import__("hashlib").sha256(
                            json.dumps(public_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                    },
                    "environment_lock": {
                        "path": relative_posix(lock_path, project_root) if lock_path else None,
                        "sha256": sha256_file(lock_path) if lock_path else None,
                    },
                    "xtb_executable": {"path": str(self.executable), "sha256": sha256_file(self.executable)},
                    "host": {"system": platform.system(), "machine": platform.machine(), "processor": platform.processor()},
                    "sandbox": self.sandbox_summary,
                },
                "budgets": self.budgets,
                "jobs": [],
            }
            atomic_write_json(self.manifest_path, manifest)

    def _reserve_job(self, request: dict[str, Any]) -> tuple[str, Path]:
        with FileLock(self.lock_path):
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            jobs = manifest.get("jobs", [])
            max_calls = int(self.budgets.get("max_xtb_jobs", self.budgets.get("max_xtb_calls", 0)))
            if max_calls and len(jobs) >= max_calls:
                append_jsonl(
                    self.rejections_path,
                    {"time": utc_now_iso(), "reason": "xTB call budget exceeded", "request": request},
                )
                raise BudgetExceededError(f"xTB call budget exceeded ({len(jobs)}/{max_calls})")
            existing_job_ids = {str(item.get("job_id")) for item in jobs}
            retry_of = request.get("retry_of")
            parent_job_id = request.get("parent_job_id") or retry_of
            if retry_of is not None and str(retry_of) not in existing_job_ids:
                raise XTBRunError(f"retry_of does not reference an existing job: {retry_of}")
            if parent_job_id is not None and str(parent_job_id) not in existing_job_ids:
                raise XTBRunError(f"parent_job_id does not reference an existing job: {parent_job_id}")
            job_id = f"job_{len(jobs) + 1:04d}"
            namespace = validate_namespace(str(request.get("namespace") or job_id))
            workdir = self.jobs_dir / f"{job_id}_{namespace}"
            workdir.mkdir(mode=0o700)
            # A reserved record is immediately durable so concurrent callers cannot
            # overrun the budget.  Final fields are replaced after termination.
            record = {
                "job_id": job_id,
                "namespace": namespace,
                "status": "reserved",
                "retry_of": retry_of,
                "parent_job_id": parent_job_id,
                "workdir": relative_posix(workdir, self.run_root),
                "request": request,
                "reserved_at": utc_now_iso(),
                "files_sha256": {},
            }
            jobs.append(record)
            manifest["jobs"] = jobs
            atomic_write_json(self.manifest_path, manifest)
            return job_id, workdir

    def _finish_job(self, job_id: str, record: dict[str, Any]) -> None:
        with FileLock(self.lock_path):
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            jobs = manifest.get("jobs", [])
            for index, value in enumerate(jobs):
                if value.get("job_id") == job_id:
                    jobs[index] = record
                    break
            else:
                raise XTBRunError(f"reserved job vanished from manifest: {job_id}")
            manifest["jobs"] = jobs
            manifest["updated_at"] = utc_now_iso()
            atomic_write_json(self.manifest_path, manifest)

    def _validated_request(
        self, request: dict[str, Any]
    ) -> tuple[dict[str, Any], Path, Path | None, dict[str, Any] | None]:
        """Validate a request without reserving a job or creating files."""
        if not isinstance(request, dict):
            raise XTBRunError("request must be a JSON object")
        allowed = {
            "geometry", "arguments", "detailed_input", "namespace", "timeout_seconds",
            "parent_job_id", "retry_of", "recovery_of_job_id", "candidate_id",
            "condition", "environment",
        }
        unknown = sorted(set(request) - allowed)
        if unknown:
            raise XTBRunError(f"unknown request fields: {unknown}")
        geometry_value = request.get("geometry")
        if not isinstance(geometry_value, str) or not geometry_value.strip():
            raise XTBRunError("geometry is required")
        arguments = _validate_arguments(request.get("arguments", []))
        timeout = int(request.get("timeout_seconds") or self.budgets.get("max_seconds_per_xtb_call", 900))
        maximum_timeout = int(self.budgets.get("max_seconds_per_xtb_call", timeout))
        if timeout <= 0 or timeout > maximum_timeout:
            raise XTBRunError(f"timeout_seconds must be in [1, {maximum_timeout}]")
        # Geometry/detailed input must live under the run workspace.  The agent
        # copies public task inputs into that workspace before exposing tools.
        geometry = safe_resolve(self.run_root, geometry_value, must_exist=True)
        if not geometry.is_file():
            raise XTBRunError("geometry is not a file")
        detailed: Path | None = None
        if request.get("detailed_input") is not None:
            if not isinstance(request["detailed_input"], str):
                raise XTBRunError("detailed_input must be a path string or null")
            detailed = safe_resolve(self.run_root, request["detailed_input"], must_exist=True)
            if not detailed.is_file():
                raise XTBRunError("detailed_input is not a file")
        overrides = request.get("environment")
        # ``environment`` is an internal testing hook, never exposed in the public
        # tool schema.  Only FAKE_XTB_* variables are accepted.
        if overrides is not None:
            if not isinstance(overrides, dict) or any(not str(key).startswith("FAKE_XTB_") for key in overrides):
                raise XTBRunError("environment overrides are restricted to FAKE_XTB_* test variables")
        candidate_id = request.get("candidate_id")
        condition = request.get("condition")
        recovery_of_job_id = request.get("recovery_of_job_id") or request.get("retry_of")
        if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id.strip()):
            raise XTBRunError("candidate_id must be a non-empty string or null")
        if condition is not None and condition not in {"gas", "water"}:
            raise XTBRunError("condition must be gas, water or null")
        if recovery_of_job_id is not None and (
            not isinstance(recovery_of_job_id, str) or not recovery_of_job_id.strip()
        ):
            raise XTBRunError("recovery_of_job_id must be a non-empty string or null")
        clean_request = {
            "geometry": relative_posix(geometry, self.run_root),
            "arguments": arguments,
            "detailed_input": relative_posix(detailed, self.run_root) if detailed else None,
            "namespace": str(request.get("namespace") or "xtb"),
            "timeout_seconds": timeout,
            "parent_job_id": request.get("parent_job_id") or recovery_of_job_id,
            "retry_of": recovery_of_job_id,
            "recovery_of_job_id": recovery_of_job_id,
            "candidate_id": candidate_id,
            "condition": condition,
        }
        return clean_request, geometry, detailed, overrides

    def validate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Public side-effect-free preflight used before cumulative reservation."""
        clean_request, _geometry, _detailed, _overrides = self._validated_request(request)
        return dict(clean_request)

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        clean_request, geometry, detailed, overrides = self._validated_request(request)
        arguments = clean_request["arguments"]
        timeout = int(clean_request["timeout_seconds"])
        job_id, workdir = self._reserve_job(clean_request)
        stdout_path = workdir / "stdout.txt"
        stderr_path = workdir / "stderr.txt"
        started_at = utc_now_iso()
        start = time.monotonic()
        timed_out = False
        peak_rss = 0
        exit_code: int | None = None
        command: list[str] = []
        sandbox: dict[str, Any] = self.sandbox_summary
        failure: str | None = None
        try:
            geometry_copy = workdir / f"input{geometry.suffix or '.xyz'}"
            shutil.copy2(geometry, geometry_copy)
            detailed_copy: Path | None = None
            if detailed:
                detailed_copy = workdir / "detailed.inp"
                shutil.copy2(detailed, detailed_copy)
            staged_arguments = (
                _staged_arguments(arguments, detailed_copy.name)
                if detailed_copy is not None
                else list(arguments)
            )
            raw_command = [str(self.executable), geometry_copy.name, *staged_arguments]
            command, sandbox = bubblewrap_command(
                raw_command,
                cwd=workdir,
                writable_roots=[workdir],
                readonly_roots=[_runtime_root(self.executable), self.executable],
            )
            environment = dict(self.environment)
            if overrides is not None:
                environment.update({str(key): str(value) for key, value in overrides.items()})
            environment["HOME"] = str(workdir)
            request_on_disk = dict(clean_request)
            request_on_disk.update(
                {
                    "geometry_original_sha256": sha256_file(geometry),
                    "geometry_job_copy": geometry_copy.name,
                    "geometry_job_copy_sha256": sha256_file(geometry_copy),
                    "parent_job_id": clean_request.get("parent_job_id"),
                }
            )
            if detailed_copy:
                request_on_disk.update(
                    {
                        "detailed_input_original_sha256": sha256_file(detailed),
                        "detailed_input_job_copy": detailed_copy.name,
                        "detailed_input_job_copy_sha256": sha256_file(detailed_copy),
                    }
                )
            atomic_write_json(workdir / "request.json", request_on_disk)
            atomic_write_json(workdir / "command.json", {"argv": raw_command, "sandbox_argv": command, "shell": False})
            summary = _environment_summary(environment, self.executable, self._version)
            summary["sandbox"] = sandbox
            atomic_write_json(workdir / "environment.json", summary)
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=workdir,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    preexec_fn=lambda: _preexec(int(self.budgets.get("memory_mb", 0))),
                )
                ps_process = psutil.Process(process.pid)
                deadline = start + timeout
                while process.poll() is None:
                    try:
                        rss = ps_process.memory_info().rss
                        for child in ps_process.children(recursive=True):
                            try:
                                rss += child.memory_info().rss
                            except psutil.Error:
                                pass
                        peak_rss = max(peak_rss, rss)
                    except psutil.Error:
                        pass
                    if time.monotonic() >= deadline:
                        timed_out = True
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        break
                    time.sleep(0.05)
                exit_code = process.wait()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            stderr_path.write_text(failure + "\n", encoding="utf-8")
        elapsed = time.monotonic() - start
        status = "timeout" if timed_out else ("completed" if exit_code == 0 and failure is None else "failed")
        output_files = [
            path.relative_to(workdir).as_posix()
            for path in sorted(workdir.rglob("*"))
            if path.is_file() and path.name not in {"request.json", "command.json", "environment.json", "stdout.txt", "stderr.txt"}
        ]
        files_sha256 = file_hash_manifest(workdir)
        record = {
            "job_id": job_id,
            "namespace": clean_request["namespace"],
            "status": status,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "retry_of": clean_request.get("retry_of"),
            "parent_job_id": clean_request.get("parent_job_id"),
            "workdir": relative_posix(workdir, self.run_root),
            "request": clean_request,
            "inputs": {
                "geometry": {
                    "source": clean_request["geometry"],
                    "source_sha256": sha256_file(geometry),
                    "job_copy": geometry_copy.name if "geometry_copy" in locals() else None,
                    "job_copy_sha256": sha256_file(geometry_copy) if "geometry_copy" in locals() and geometry_copy.is_file() else None,
                },
                "detailed_input": {
                    "source": clean_request.get("detailed_input"),
                    "source_sha256": sha256_file(detailed) if detailed and detailed.is_file() else None,
                    "job_copy": detailed_copy.name if "detailed_copy" in locals() and detailed_copy else None,
                    "job_copy_sha256": sha256_file(detailed_copy) if "detailed_copy" in locals() and detailed_copy and detailed_copy.is_file() else None,
                },
            },
            "command": command,
            "sandbox": sandbox,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
            "failure": failure,
            "stdout_file": "stdout.txt",
            "stderr_file": "stderr.txt",
            "output_files": output_files,
            "files_sha256": files_sha256,
            "evidence_tree_sha256": _evidence_tree_sha256(files_sha256),
        }
        self._finish_job(job_id, record)
        response = {
            "job_id": job_id,
            "status": status,
            "exit_code": exit_code,
            "stdout_file": relative_posix(stdout_path, self.run_root),
            "stderr_file": relative_posix(stderr_path, self.run_root),
            "workdir": relative_posix(workdir, self.run_root),
            "elapsed_seconds": elapsed,
            "output_files": [f"{relative_posix(workdir, self.run_root)}/{value}" for value in output_files],
        }
        if failure is not None:
            response["error"] = failure
        return response


def run_xtb(
    request: dict[str, Any],
    *,
    task_dir: str | Path,
    run_root: str | Path,
    xtb_executable: str | Path = "xtb",
) -> dict[str, Any]:
    """Stateless convenience wrapper around :class:`XTBRunner`."""

    return XTBRunner(task_dir=task_dir, run_root=run_root, xtb_executable=xtb_executable).run(request)

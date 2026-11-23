"""Immutable prepare -> native run -> independent validation lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
from typing import Any

from .file_tools import ToolInputError, WorkspaceSandbox, sha256_bytes, sha256_file
from .provenance import (
    ProvenanceInspector,
    build_artifact_manifest,
    canonical_json_sha256,
    read_strict_json,
    utc_now,
    write_json_exclusive,
)
from .xtb_log_parser import parse_xtb_log
from .xyz_validator import extract_xyz_frame, parse_xyz_text, validate_xyz


ATTEMPT_PATTERN = re.compile(r"^attempt_(?!000)[0-9]{3}$")
VALIDATE_ATTEMPT_PATTERN = re.compile(
    r"^attempt_(?:000_initial_failure|(?!000)[0-9]{3})$"
)
ALL_INVARIANTS = (
    "method",
    "charge",
    "uhf",
    "solvation",
    "atom_order",
    "composition",
    "required_phases",
)
_SAFE_SETTING = re.compile(r"^[A-Za-z0-9_.+-]{1,80}$")
VALIDATION_CLASSIFICATION_PRIORITY = (
    "INVALID_ARTIFACT",
    "PROCESS_FAILED",
    "OPTIMIZATION_NOT_CONVERGED",
    "PROGRAM_COMPLETED_TASK_INCOMPLETE",
    "SCIENTIFIC_WARNING",
    "VALIDATED_SUCCESS",
)


def _recursive_values(value: Any, keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in keys:
                found.append(child)
            found.extend(_recursive_values(child, keys))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.extend(_recursive_values(child, keys))
    return found


def _first(value: Any, *names: str, default: Any = None) -> Any:
    values = _recursive_values(value, {name.casefold() for name in names})
    return values[0] if values else default


def _relative_case_path(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("benchmark_case/"):
        normalized = normalized[len("benchmark_case/") :]
    elif PurePosixPath(normalized).name in {"job_spec.json", "case_manifest.json"}:
        # Organizer configs may bind a hidden case by a project-relative path;
        # the selected case workspace still exposes the authority document at
        # its canonical root filename.
        normalized = PurePosixPath(normalized).name
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        return None
    return pure.as_posix()


def _scientific_settings(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    method = _first(job_spec, "method", "xtb_method", "level_of_theory")
    charge = _first(job_spec, "charge", "molecular_charge")
    uhf = _first(job_spec, "uhf", "unpaired_electrons")
    solvation = _first(job_spec, "solvation", "solvent", default=None)
    phases = _first(job_spec, "required_phases", "phases", "workflow")
    if method is None or charge is None or uhf is None:
        raise ToolInputError("JOB_SPEC_INVALID", "job specification lacks method, charge, or UHF")
    if isinstance(phases, Mapping):
        phases = phases.get("required", phases.get("phases"))
    if not isinstance(phases, list):
        phases = ["geometry_optimization", "hessian"]
    normalized_phases = [str(value) for value in phases]
    return {
        "method": method,
        "charge": charge,
        "uhf": uhf,
        "solvation": solvation,
        "required_phases": normalized_phases,
    }


def _method_number(method: Any) -> str:
    text = str(method).casefold().replace("-", "")
    match = re.search(r"gfn\s*([012])", text)
    if not match:
        raise ToolInputError("UNSUPPORTED_METHOD", "job method is not a supported fixed GFN level")
    return match.group(1)


def _safe_number(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError("JOB_SPEC_INVALID", f"job {name} must be an integer")
    return str(value)


def _normalize_native_stream(raw_path: Path, text_path: Path) -> dict[str, Any]:
    """Preserve raw native bytes and create a deterministic UTF-8 text view."""

    raw = raw_path.read_bytes()
    try:
        decoded = raw.decode("utf-8")
        decoding = "UTF-8_STRICT"
    except UnicodeDecodeError:
        decoded = raw.decode("utf-8", errors="backslashreplace")
        decoding = "UTF-8_WITH_INVALID_BYTES_ESCAPED"
    normalized = "".join(
        character
        if character in "\t\r\n" or ord(character) >= 32
        else f"\\x{ord(character):02x}"
        for character in decoded
    )
    with text_path.open("xb") as handle:
        handle.write(normalized.encode("utf-8"))
    return {
        "raw_path": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
        "raw_size_bytes": raw_path.stat().st_size,
        "text_path": text_path.name,
        "text_sha256": sha256_file(text_path),
        "text_size_bytes": text_path.stat().st_size,
        "decoding": decoding,
        "control_bytes_escaped": True,
    }


def _classify_attempt_validation(
    *,
    phase: Any,
    process_valid: bool,
    artifact_integrity_valid: bool,
    optimization_completed: bool,
    hessian_completed: bool,
    warnings: Sequence[Any],
) -> tuple[str, str]:
    """Return fail-closed overall and phase-specific classifications."""

    if phase == "geometry_optimization":
        phase_classification = (
            "OPTIMIZATION_CONVERGED"
            if optimization_completed
            else "OPTIMIZATION_NOT_CONVERGED"
        )
    elif phase == "hessian":
        phase_classification = (
            "HESSIAN_COMPLETED"
            if hessian_completed
            else "PROGRAM_COMPLETED_TASK_INCOMPLETE"
        )
    else:
        phase_classification = "PROGRAM_COMPLETED_TASK_INCOMPLETE"
    if not artifact_integrity_valid:
        classification = "INVALID_ARTIFACT"
    elif not process_valid:
        classification = "PROCESS_FAILED"
    elif phase == "geometry_optimization" and not optimization_completed:
        classification = "OPTIMIZATION_NOT_CONVERGED"
    elif phase == "hessian" and not hessian_completed:
        classification = "PROGRAM_COMPLETED_TASK_INCOMPLETE"
    elif warnings:
        classification = "SCIENTIFIC_WARNING"
    elif optimization_completed or hessian_completed:
        classification = "VALIDATED_SUCCESS"
    else:
        classification = "PROGRAM_COMPLETED_TASK_INCOMPLETE"
    return classification, phase_classification


class XtbAttemptManager:
    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        *,
        task_config: Mapping[str, Any],
        job_spec: Mapping[str, Any],
        provenance: ProvenanceInspector,
        expected_symbols: Sequence[str] | None = None,
        process_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.task_config = dict(task_config)
        self.job_spec = dict(job_spec)
        self.provenance = provenance
        self.expected_symbols = tuple(expected_symbols) if expected_symbols else None
        self.process_runner = process_runner or subprocess.run
        self._live_runtime = process_runner is None or process_runner is subprocess.run
        self.settings = _scientific_settings(self.job_spec)
        self.attempts_relative = "attempts"

    @property
    def live_native_execution(self) -> bool:
        """Whether attempts use the real ``subprocess.run`` native runtime."""

        return self._live_runtime

    @property
    def _budget(self) -> Mapping[str, Any]:
        value = self.task_config.get("budget")
        return value if isinstance(value, Mapping) else {}

    @property
    def _xtb_config(self) -> Mapping[str, Any]:
        runtime = self.task_config.get("runtime")
        xtb = runtime.get("xtb") if isinstance(runtime, Mapping) else None
        return xtb if isinstance(xtb, Mapping) else {}

    def _attempts_root(self, *, create: bool = False) -> Path:
        root = self.sandbox.root / self.attempts_relative
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise ToolInputError("UNSAFE_ATTEMPT_ROOT", "attempt root is not a safe directory")
        if create:
            root.mkdir(exist_ok=True)
        return root

    def _attempt_path(self, attempt_id: str, *, must_exist: bool = True) -> Path:
        if not ATTEMPT_PATTERN.fullmatch(attempt_id):
            raise ToolInputError("INVALID_ATTEMPT_ID", "attempt_id is not canonical")
        return self.sandbox.resolve(
            f"{self.attempts_relative}/{attempt_id}",
            must_exist=must_exist,
            require_file=False,
        )

    def _generated_attempt_ids(self) -> list[str]:
        root = self._attempts_root()
        if not root.is_dir():
            return []
        result: list[str] = []
        for child in root.iterdir():
            if child.is_symlink():
                continue
            if child.is_dir() and ATTEMPT_PATTERN.fullmatch(child.name) and child.name != "attempt_000":
                result.append(child.name)
        return sorted(result)

    def _next_attempt_id(self) -> str:
        existing = self._generated_attempt_ids()
        maximum = int(self._budget.get("max_execution_attempts", 3))
        if len(existing) >= maximum:
            raise ToolInputError("ATTEMPT_BUDGET_EXCEEDED", "execution-attempt budget is exhausted")
        indices = [int(value.rsplit("_", 1)[1]) for value in existing]
        next_index = max(indices, default=0) + 1
        if next_index > 999:
            raise ToolInputError("ATTEMPT_BUDGET_EXCEEDED", "attempt identifier space is exhausted")
        return f"attempt_{next_index:03d}"

    def _prepared_records(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for attempt_id in self._generated_attempt_ids():
            path = self._attempts_root() / attempt_id / "prepared.json"
            if path.is_file():
                value = read_strict_json(path)
                if isinstance(value, dict):
                    result.append(value)
        return result

    def _validate_limits(
        self,
        phase: str,
        optimization_cycle_limit: int | None,
        thread_count: int,
        walltime_seconds: int,
        preserved_invariants: Sequence[str],
    ) -> None:
        if phase not in {"geometry_optimization", "hessian"}:
            raise ToolInputError("INVALID_PHASE", "phase is not supported")
        if isinstance(thread_count, bool) or not isinstance(thread_count, int):
            raise ToolInputError("INVALID_THREAD_COUNT", "thread_count must be an integer")
        thread_min = int(self._xtb_config.get("thread_count_min", 1))
        thread_max = int(self._xtb_config.get("thread_count_max", 1))
        if not thread_min <= thread_count <= thread_max:
            raise ToolInputError("INVALID_THREAD_COUNT", "thread_count is outside the fixed range")
        if isinstance(walltime_seconds, bool) or not isinstance(walltime_seconds, int):
            raise ToolInputError("INVALID_WALLTIME", "walltime_seconds must be an integer")
        maximum_walltime = min(
            int(self._xtb_config.get("walltime_seconds_max", 240)),
            int(self._budget.get("single_xtb_timeout_seconds", 240)),
        )
        if not 1 <= walltime_seconds <= maximum_walltime:
            raise ToolInputError("INVALID_WALLTIME", "walltime_seconds exceeds the configured bound")
        if phase == "geometry_optimization":
            if isinstance(optimization_cycle_limit, bool) or not isinstance(optimization_cycle_limit, int):
                raise ToolInputError("INVALID_CYCLE_LIMIT", "optimization requires an integer cycle limit")
            minimum = int(self._xtb_config.get("restart_cycle_limit_min", 2))
            maximum = int(self._xtb_config.get("restart_cycle_limit_max", 200))
            if not minimum <= optimization_cycle_limit <= maximum:
                raise ToolInputError("INVALID_CYCLE_LIMIT", "optimization cycle limit is outside the configured range")
        elif optimization_cycle_limit is not None:
            raise ToolInputError("INVALID_CYCLE_LIMIT", "hessian phase must not set an optimization cycle limit")
        values = list(preserved_invariants)
        if len(values) != len(ALL_INVARIANTS) or set(values) != set(ALL_INVARIANTS):
            raise ToolInputError(
                "INVALID_INVARIANTS",
                "preserved_invariants must contain every required invariant exactly once",
            )

    def _scientific_argv(self, phase: str, thread_count: int) -> list[str]:
        config = self._xtb_config
        executable = config.get("executable_alias")
        if not isinstance(executable, str) or not _SAFE_SETTING.fullmatch(executable):
            raise ToolInputError("RUNTIME_CONFIG_INVALID", "fixed xTB executable alias is invalid")
        argv = [
            executable,
            "input.xyz",
            "--gfn",
            _method_number(self.settings["method"]),
            "--chrg",
            _safe_number(self.settings["charge"], "charge"),
            "--uhf",
            _safe_number(self.settings["uhf"], "UHF"),
        ]
        solvation = self.settings.get("solvation")
        if isinstance(solvation, Mapping) and (
            solvation.get("enabled") is False
            or solvation.get("solvent") in (None, "", "none", "gas", "gas_phase")
            or str(solvation.get("model", "")).casefold() in {"", "none", "gas", "gas_phase"}
        ):
            solvation = None
        if solvation not in (None, "", "none", "gas", "gas_phase"):
            if isinstance(solvation, Mapping):
                model = str(solvation.get("model", "")).casefold()
                solvent = solvation.get("solvent")
            else:
                model, solvent = "alpb", solvation
            if model not in {"alpb", "gbsa"} or not isinstance(solvent, str) or not _SAFE_SETTING.fullmatch(solvent):
                raise ToolInputError("UNSUPPORTED_SOLVATION", "job solvation setting cannot be mapped safely")
            argv.extend([f"--{model}", solvent])
        if phase == "geometry_optimization":
            # xTB 6.7.1 has no supported --cycles option.  The bounded cycle
            # limit is supplied through the per-attempt xcontrol file.
            argv.extend(["--opt", "normal", "--input", "xcontrol.inp"])
        else:
            argv.append("--hess")
        argv.extend(["--parallel", str(thread_count)])
        return argv

    def _runtime_wrapper_argv(self) -> list[str]:
        config = self._xtb_config
        conda_environment = config.get("conda_environment")
        if not isinstance(conda_environment, str) or not _SAFE_SETTING.fullmatch(conda_environment):
            raise ToolInputError("RUNTIME_CONFIG_INVALID", "fixed conda environment is invalid")
        return ["conda", "run", "-n", conda_environment, "--no-capture-output"]

    def _build_argv(self, phase: str, cycle_limit: int | None, thread_count: int) -> list[str]:
        if phase == "geometry_optimization" and cycle_limit is None:
            raise ToolInputError("INVALID_CYCLE_LIMIT", "optimization requires a cycle limit")
        return self._runtime_wrapper_argv() + self._scientific_argv(phase, thread_count)

    @staticmethod
    def _xcontrol_text(cycle_limit: int) -> str:
        return f"$opt\n  maxcycle={cycle_limit}\n$end\n"

    def prepare_restart(
        self,
        *,
        phase: str,
        source_geometry: Mapping[str, Any],
        optimization_cycle_limit: int | None,
        thread_count: int,
        walltime_seconds: int,
        restart_policy: str,
        rationale: str,
        preserved_invariants: Sequence[str],
    ) -> dict[str, Any]:
        self._validate_limits(
            phase,
            optimization_cycle_limit,
            thread_count,
            walltime_seconds,
            preserved_invariants,
        )
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 500:
            raise ToolInputError("INVALID_RATIONALE", "rationale must be non-empty and bounded")
        if restart_policy not in {
            "last_complete_current_frame",
            "current_restart_artifact",
            "original_input",
            "validated_optimized_geometry",
        }:
            raise ToolInputError("INVALID_RESTART_POLICY", "restart policy is not supported")
        source_path = source_geometry.get("path") if isinstance(source_geometry, Mapping) else None
        frame_index = source_geometry.get("frame_index") if isinstance(source_geometry, Mapping) else None
        if not isinstance(source_path, str):
            raise ToolInputError("INVALID_SOURCE", "source geometry path is required")
        provenance = self.provenance.inspect(source_path, frame_index)
        source_kind = provenance.get("source_kind")
        origin = str(provenance.get("origin") or "")
        policy_allowed = {
            "original_input": source_kind == "ORIGINAL_INPUT",
            "last_complete_current_frame": (
                source_kind == "CURRENT_TRAJECTORY_FRAME"
                or (
                    source_kind == "VALIDATED_ATTEMPT_ARTIFACT"
                    and origin.rsplit(":", 1)[-1] in {"trajectory", "restart_geometry"}
                )
            ),
            "current_restart_artifact": (
                source_kind == "VALIDATED_ATTEMPT_ARTIFACT"
                and origin.rsplit(":", 1)[-1] == "restart_geometry"
            ),
            "validated_optimized_geometry": (
                source_kind == "VALIDATED_ATTEMPT_ARTIFACT"
                and origin.rsplit(":", 1)[-1] == "optimized_geometry"
            ),
        }[restart_policy]
        source_allowed = bool(provenance.get("restart_allowed") and policy_allowed)
        if not source_allowed:
            raise ToolInputError("SOURCE_NOT_AUTHORIZED", "source geometry is not hash-bound and restart-authorized")
        if phase == "hessian":
            if restart_policy != "validated_optimized_geometry":
                raise ToolInputError("INVALID_HESSIAN_SOURCE", "hessian requires a validated optimized geometry")
            if (
                provenance.get("source_kind") != "VALIDATED_ATTEMPT_ARTIFACT"
                or origin.rsplit(":", 1)[-1] != "optimized_geometry"
            ):
                raise ToolInputError("INVALID_HESSIAN_SOURCE", "hessian source is not a validated attempt artifact")
        source_resolved = self.sandbox.resolve(provenance["path"])
        try:
            source_text = source_resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolInputError("INVALID_UTF8", "source geometry is not valid UTF-8") from exc
        frame = extract_xyz_frame(source_text, frame_index)
        if self.expected_symbols and frame.symbols != self.expected_symbols:
            raise ToolInputError("INVARIANT_VIOLATION", "source atom identity/order differs from the job input")
        attempt_id = self._next_attempt_id()
        scientific_argv = self._scientific_argv(phase, thread_count)
        runtime_wrapper_argv = self._runtime_wrapper_argv()
        argv = self._build_argv(phase, optimization_cycle_limit, thread_count)
        request = {
            "phase": phase,
            "optimization_cycle_limit": optimization_cycle_limit,
            "thread_count": thread_count,
            "walltime_seconds": walltime_seconds,
            "restart_policy": restart_policy,
            "rationale": rationale.strip(),
            "preserved_invariants": list(preserved_invariants),
            "enforced_invariants": list(ALL_INVARIANTS),
        }
        fingerprint = canonical_json_sha256(
            {
                "phase": phase,
                "source_sha256": provenance["sha256"],
                "frame_sha256": frame.sha256,
                "settings": self.settings,
                "request": {
                    key: value for key, value in request.items() if key != "rationale"
                },
                "argv": argv,
            }
        )
        if any(record.get("fingerprint") == fingerprint for record in self._prepared_records()):
            raise ToolInputError("DUPLICATE_ATTEMPT", "an identical prepared execution already exists")
        attempt_root = self._attempts_root(create=True) / attempt_id
        attempt_root.mkdir(exist_ok=False)
        input_path = attempt_root / "input.xyz"
        with input_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(frame.text)
        control_path: Path | None = None
        if phase == "geometry_optimization":
            control_path = attempt_root / "xcontrol.inp"
            with control_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(self._xcontrol_text(int(optimization_cycle_limit)))
        input_relative = self.sandbox.relative(input_path)
        source_parent = None
        source_parts = PurePosixPath(provenance["path"]).parts
        if len(source_parts) >= 2 and source_parts[0] == "attempts":
            source_parent = source_parts[1]
        execution = {
            "environment_manager": "conda",
            "conda_environment": self._xtb_config["conda_environment"],
            "executable_alias": self._xtb_config["executable_alias"],
            "argv": argv,
            "scientific_argv": scientific_argv,
            "runtime_wrapper_argv": runtime_wrapper_argv,
            "shell": False,
            "timeout_seconds": walltime_seconds,
            "expected_version": self._xtb_config.get("version"),
            "identity_policy": self._xtb_config.get("identity_policy"),
            "required_help_tokens": self._xtb_config.get("required_help_tokens"),
        }
        prepared = {
            "schema_version": "1.0",
            "attempt_id": attempt_id,
            "phase": phase,
            "state": "PREPARED",
            "request": request,
            "source_geometry": {
                "path": provenance["path"],
                "frame_index": frame.index,
                "file_sha256": provenance["sha256"],
                "frame_sha256": frame.sha256,
            },
            "input_geometry": {
                "path": input_relative,
                "sha256": sha256_file(input_path),
            },
            "control_input": (
                {
                    "path": self.sandbox.relative(control_path),
                    "sha256": sha256_file(control_path),
                    "optimization_cycle_limit": optimization_cycle_limit,
                }
                if control_path is not None
                else None
            ),
            "scientific_settings": self.settings,
            "execution": execution,
            "fingerprint": fingerprint,
            "created_at": utc_now(),
        }
        provenance_record = {
            "schema_version": "1.0",
            "attempt_id": attempt_id,
            "source": prepared["source_geometry"],
            "input": prepared["input_geometry"],
            "chain": [provenance["path"], input_relative],
            "parent_attempt_id": source_parent,
            "currentness": "PREPARED_INPUT",
            "restart_allowed": True,
            "success_evidence_allowed": False,
            "created_at": prepared["created_at"],
        }
        # Canonical public records and evaluator compatibility aliases are
        # byte-identical, making their relationship independently hashable.
        for filename in ("attempt.json", "prepared_command.json", "prepared.json"):
            write_json_exclusive(attempt_root / filename, prepared)
        for filename in ("input_manifest.json", "provenance.json"):
            write_json_exclusive(attempt_root / filename, provenance_record)
        source_result = {
            **prepared["source_geometry"],
            "sha256": prepared["source_geometry"]["file_sha256"],
            "provenance_state": provenance["source_kind"],
        }
        return {
            "attempt_id": attempt_id,
            "phase": phase,
            "state": "PREPARED",
            "attempt_directory": self.sandbox.relative(attempt_root),
            "prepared_command_path": self.sandbox.relative(attempt_root / "prepared_command.json"),
            "input_manifest_path": self.sandbox.relative(attempt_root / "input_manifest.json"),
            "prepared_record": self.sandbox.relative(attempt_root / "prepared.json"),
            "provenance_record": self.sandbox.relative(attempt_root / "provenance.json"),
            "input_geometry": prepared["input_geometry"],
            "source_geometry": source_result,
            "source_provenance": provenance["source_kind"],
            "argv": argv,
            "scientific_argv": scientific_argv,
            "runtime_wrapper_argv": runtime_wrapper_argv,
            "shell": False,
            "fingerprint": fingerprint,
        }

    @staticmethod
    def _safe_environment(thread_count: int) -> dict[str, str]:
        allowed = (
            "SystemRoot",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "PATH",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
        )
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        # conda's Windows wrapper invokes chcp.exe.  Some host PATH values are
        # malformed, so prepend the trusted OS System32 directory explicitly.
        system_root = environment.get("SystemRoot", environment.get("WINDIR"))
        if isinstance(system_root, str) and system_root:
            system32 = str(Path(system_root) / "System32")
            environment["PATH"] = system32 + os.pathsep + environment.get("PATH", "")
        value = str(thread_count)
        environment.update(
            {
                "OMP_NUM_THREADS": value,
                "MKL_NUM_THREADS": value,
                "OPENBLAS_NUM_THREADS": value,
                "PYTHONNOUSERSITE": "1",
            }
        )
        return environment

    def _runtime_identity(
        self,
        *,
        conda_path: str | None,
        environment: Mapping[str, str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        config = self._xtb_config
        expected = {
            "version": str(config.get("version")),
            "identity_policy": str(config.get("identity_policy")),
            "required_help_tokens": list(config.get("required_help_tokens", [])),
        }
        if not self._live_runtime:
            return {
                **expected,
                "verified": False,
                "verification_mode": "INJECTED_TEST_RUNNER_CONFIG_EXPECTATION",
            }
        if not conda_path:
            raise ToolInputError("CONDA_NOT_FOUND", "fixed conda runtime is unavailable")
        preflight_timeout = min(60, timeout_seconds)
        try:
            env_result = subprocess.run(
                [conda_path, "env", "list", "--json"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=preflight_timeout,
                env=dict(environment),
            )
            if env_result.returncode != 0:
                raise ToolInputError("RUNTIME_IDENTITY_FAILED", "conda environment discovery failed")
            env_document = json.loads(env_result.stdout.decode("utf-8"))
            env_name = str(config.get("conda_environment"))
            prefixes = [Path(value) for value in env_document.get("envs", []) if isinstance(value, str)]
            matches = [path for path in prefixes if path.name.casefold() == env_name.casefold()]
            if len(matches) != 1:
                raise ToolInputError("RUNTIME_IDENTITY_FAILED", "fixed conda environment is not unique")
            executable_relative = Path("Library/bin/xtb.exe") if os.name == "nt" else Path("bin/xtb")
            executable = matches[0] / executable_relative
            if not executable.is_file() or executable.is_symlink():
                raise ToolInputError("RUNTIME_IDENTITY_FAILED", "fixed xTB executable is unavailable")
            binary_hash = sha256_file(executable).upper()
            prefix = [
                conda_path,
                "run",
                "-n",
                env_name,
                "--no-capture-output",
                str(config.get("executable_alias")),
            ]
            version_result = subprocess.run(
                prefix + ["--version"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=preflight_timeout,
                env=dict(environment),
            )
            help_result = subprocess.run(
                prefix + ["--help"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=preflight_timeout,
                env=dict(environment),
            )
        except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolInputError(
                "RUNTIME_IDENTITY_FAILED", f"native runtime preflight failed: {type(exc).__name__}"
            ) from exc
        version_bytes = (version_result.stdout + version_result.stderr).replace(b"\r\n", b"\n")
        help_bytes = (help_result.stdout + help_result.stderr).replace(b"\r\n", b"\n")
        match = re.search(
            rb"xtb\s+version\s+([0-9]+(?:\.[0-9]+)+)",
            version_bytes,
            flags=re.IGNORECASE,
        )
        actual = {
            "version": match.group(1).decode("ascii") if match else "UNKNOWN",
            "binary_sha256": binary_hash,
            "help_output_sha256": sha256_bytes(help_bytes).upper(),
            "identity_policy": expected["identity_policy"],
            "required_help_tokens": expected["required_help_tokens"],
        }
        help_lower = help_bytes.lower()
        help_contract_valid = all(
            isinstance(token, str) and token.encode("utf-8").lower() in help_lower
            for token in expected["required_help_tokens"]
        )
        if (
            version_result.returncode != 0
            or help_result.returncode != 0
            or actual["version"] != expected["version"]
            or expected["identity_policy"] != "xtb_portable_identity/2.0"
            or not help_contract_valid
        ):
            raise ToolInputError("RUNTIME_IDENTITY_MISMATCH", "native xTB identity differs from frozen config")
        return {
            **actual,
            "version_output_sha256": sha256_bytes(version_bytes).upper(),
            "help_contract_valid": help_contract_valid,
            "verified": True,
            "verification_mode": "LIVE_VERSION_AND_NORMALIZED_HELP_CONTRACT",
        }

    def run_xtb(self, attempt_id: str) -> dict[str, Any]:
        attempt_root = self._attempt_path(attempt_id)
        prepared_path = attempt_root / "prepared.json"
        run_record_path = attempt_root / "run_record.json"
        execution_summary_path = attempt_root / "execution_summary.json"
        manifest_path = attempt_root / "artifact_manifest.json"
        if not prepared_path.is_file():
            raise ToolInputError("ATTEMPT_NOT_PREPARED", "attempt has no prepared record")
        if run_record_path.exists() or execution_summary_path.exists() or manifest_path.exists():
            raise ToolInputError("ATTEMPT_ALREADY_RUN", "attempt execution is immutable and already recorded")
        for canonical in (attempt_root / "attempt.json", attempt_root / "prepared_command.json"):
            if not canonical.is_file() or sha256_file(canonical) != sha256_file(prepared_path):
                raise ToolInputError("ATTEMPT_TAMPERED", "canonical prepared records are missing or differ")
        input_manifest_path = attempt_root / "input_manifest.json"
        provenance_path = attempt_root / "provenance.json"
        if (
            not input_manifest_path.is_file()
            or not provenance_path.is_file()
            or sha256_file(input_manifest_path) != sha256_file(provenance_path)
        ):
            raise ToolInputError("ATTEMPT_TAMPERED", "canonical input manifest and provenance alias differ")
        prepared = read_strict_json(prepared_path)
        if not isinstance(prepared, Mapping) or prepared.get("attempt_id") != attempt_id:
            raise ToolInputError("ATTEMPT_RECORD_INVALID", "prepared record does not match attempt")
        execution = prepared.get("execution")
        request = prepared.get("request")
        if not isinstance(execution, Mapping) or not isinstance(request, Mapping):
            raise ToolInputError("ATTEMPT_RECORD_INVALID", "prepared execution record is incomplete")
        safe_argv = execution.get("argv")
        if not isinstance(safe_argv, list) or not all(isinstance(value, str) for value in safe_argv):
            raise ToolInputError("ATTEMPT_RECORD_INVALID", "prepared argv is invalid")
        thread_count = request.get("thread_count")
        expected_scientific = self._scientific_argv(prepared["phase"], thread_count)
        expected_wrapper = self._runtime_wrapper_argv()
        expected_argv = self._build_argv(
            prepared["phase"], request.get("optimization_cycle_limit"), thread_count
        )
        if (
            execution.get("shell") is not False
            or safe_argv != expected_argv
            or execution.get("scientific_argv") != expected_scientific
            or execution.get("runtime_wrapper_argv") != expected_wrapper
            or safe_argv != expected_wrapper + expected_scientific
        ):
            raise ToolInputError("ATTEMPT_TAMPERED", "prepared argv does not match trusted reconstruction")
        control_path = attempt_root / "xcontrol.inp"
        if prepared["phase"] == "geometry_optimization":
            expected_control = self._xcontrol_text(int(request["optimization_cycle_limit"]))
            control_record = prepared.get("control_input")
            if (
                not control_path.is_file()
                or control_path.read_text(encoding="utf-8") != expected_control
                or not isinstance(control_record, Mapping)
                or control_record.get("path") != self.sandbox.relative(control_path)
                or control_record.get("sha256") != sha256_file(control_path)
            ):
                raise ToolInputError("ATTEMPT_TAMPERED", "per-attempt xcontrol input is missing or changed")
        elif control_path.exists() or prepared.get("control_input") is not None:
            raise ToolInputError("ATTEMPT_TAMPERED", "hessian attempt unexpectedly contains optimization control")
        conda_path = shutil.which("conda")
        runtime_argv = list(safe_argv)
        if conda_path:
            runtime_argv[0] = str(Path(conda_path).resolve())
        timeout_seconds = int(execution.get("timeout_seconds"))
        safe_environment = self._safe_environment(int(request["thread_count"]))
        runtime_identity = self._runtime_identity(
            conda_path=runtime_argv[0] if conda_path else None,
            environment=safe_environment,
            timeout_seconds=timeout_seconds,
        )
        stdout_raw_path = attempt_root / "stdout.raw.bin"
        stderr_raw_path = attempt_root / "stderr.raw.bin"
        stdout_path = attempt_root / "stdout.txt"
        stderr_path = attempt_root / "stderr.txt"
        started = utc_now()
        start_clock = time.monotonic()
        state = "LAUNCH_FAILED"
        exit_code: int | None = None
        timed_out = False
        native_executed = False
        exception_type: str | None = None
        with stdout_raw_path.open("xb") as stdout_handle, stderr_raw_path.open("xb") as stderr_handle:
            try:
                completed = self.process_runner(
                    runtime_argv,
                    cwd=attempt_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=timeout_seconds,
                    shell=False,
                    check=False,
                    env=safe_environment,
                )
                native_executed = True
                exit_code = int(completed.returncode)
                state = "EXECUTED"
            except subprocess.TimeoutExpired:
                native_executed = True
                timed_out = True
                state = "TIMED_OUT"
                exception_type = "TimeoutExpired"
            except (OSError, ValueError) as exc:
                state = "LAUNCH_FAILED"
                exception_type = type(exc).__name__
        stream_normalization = {
            "stdout": _normalize_native_stream(stdout_raw_path, stdout_path),
            "stderr": _normalize_native_stream(stderr_raw_path, stderr_path),
        }
        for source, alias in (
            (stdout_path, attempt_root / "stdout.log"),
            (stderr_path, attempt_root / "stderr.log"),
        ):
            with source.open("rb") as source_handle, alias.open("xb") as alias_handle:
                shutil.copyfileobj(source_handle, alias_handle)
        completed_at = utc_now()
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(attempt_root.iterdir(), key=lambda value: value.name.casefold()):
            if path.is_file() and not path.is_symlink() and path.name != "run_record.json":
                relative = self.sandbox.relative(path)
                files[relative] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
        record = {
            "schema_version": "1.0",
            "attempt_id": attempt_id,
            "phase": prepared["phase"],
            "state": state,
            "native_xtb_executed": native_executed,
            "shell": False,
            "argv": safe_argv,
            "scientific_argv": expected_scientific,
            "runtime_wrapper_argv": expected_wrapper,
            "conda_environment": execution["conda_environment"],
            "runtime_identity": runtime_identity,
            "started_at": started,
            "completed_at": completed_at,
            "duration_seconds": round(time.monotonic() - start_clock, 6),
            "timeout_seconds": timeout_seconds,
            "timed_out": timed_out,
            "exit_code": exit_code,
            "exception_type": exception_type,
            "stdout_path": self.sandbox.relative(stdout_path),
            "stderr_path": self.sandbox.relative(stderr_path),
            "stream_normalization": stream_normalization,
            "files": files,
        }
        write_json_exclusive(execution_summary_path, record)
        write_json_exclusive(run_record_path, record)
        # The native-execution manifest is part of run_xtb's atomic evidence
        # boundary.  validate_attempt verifies this immutable record and writes
        # validation.json separately; it must never replace the run manifest.
        execution_roles = self._generated_roles(attempt_root, str(prepared["phase"]))
        write_json_exclusive(
            manifest_path,
            build_artifact_manifest(self.sandbox, attempt_id, execution_roles),
        )
        artifact_hashes = {
            relative: metadata["sha256"] for relative, metadata in files.items()
        }
        return {
            "attempt_id": attempt_id,
            "phase": prepared["phase"],
            "state": state,
            "native_xtb_executed": native_executed,
            "shell": False,
            "argv": safe_argv,
            "scientific_argv": expected_scientific,
            "runtime_wrapper_argv": expected_wrapper,
            "runtime_identity": runtime_identity,
            "exit_code": exit_code,
            "process_exit_code": exit_code,
            "timed_out": timed_out,
            "execution_summary_path": self.sandbox.relative(execution_summary_path),
            "artifact_manifest_path": self.sandbox.relative(manifest_path),
            "artifact_hashes": artifact_hashes,
            "run_record": self.sandbox.relative(run_record_path),
            "stdout": self.sandbox.relative(stdout_path),
            "stderr": self.sandbox.relative(stderr_path),
            "files": files,
        }

    def _verify_execution_manifest(self, attempt_root: Path, attempt_id: str) -> Path:
        manifest_path = attempt_root / "artifact_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            code = (
                "ATTEMPT_TAMPERED"
                if (attempt_root / "run_record.json").exists()
                or (attempt_root / "execution_summary.json").exists()
                else "ATTEMPT_NOT_RUN"
            )
            raise ToolInputError(code, "native execution manifest is missing")
        try:
            manifest = read_strict_json(manifest_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ToolInputError("ATTEMPT_TAMPERED", "native execution manifest is invalid") from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != "1.0"
            or manifest.get("attempt_id") != attempt_id
            or not isinstance(manifest.get("entries"), list)
            or not manifest["entries"]
        ):
            raise ToolInputError("ATTEMPT_TAMPERED", "native execution manifest identity is invalid")
        seen: set[str] = set()
        resolved_attempt = attempt_root.resolve()
        for entry in manifest["entries"]:
            if not isinstance(entry, Mapping):
                raise ToolInputError("ATTEMPT_TAMPERED", "native execution manifest entry is invalid")
            relative = entry.get("path")
            if not isinstance(relative, str) or relative in seen:
                raise ToolInputError("ATTEMPT_TAMPERED", "native execution manifest path is invalid")
            seen.add(relative)
            resolved = self.sandbox.resolve(relative)
            try:
                resolved.relative_to(resolved_attempt)
            except ValueError as exc:
                raise ToolInputError(
                    "ATTEMPT_TAMPERED", "native execution manifest escapes its attempt"
                ) from exc
            if resolved.name == "artifact_manifest.json" or not resolved.is_file():
                raise ToolInputError("ATTEMPT_TAMPERED", "native execution manifest entry is missing")
            if (
                entry.get("sha256") != sha256_file(resolved)
                or entry.get("size_bytes") != resolved.stat().st_size
            ):
                raise ToolInputError("ATTEMPT_TAMPERED", "native execution artifact changed")
        return manifest_path

    def _generated_roles(self, attempt_root: Path, phase: str) -> dict[str, str]:
        role_names = {
            "input.xyz": "input_geometry",
            "xcontrol.inp": "control_input",
            "attempt.json": "attempt_record",
            "prepared_command.json": "prepared_command",
            "input_manifest.json": "input_manifest",
            "stdout.txt": "stdout",
            "stderr.txt": "stderr",
            "stdout.raw.bin": "stdout_raw_native_bytes",
            "stderr.raw.bin": "stderr_raw_native_bytes",
            "execution_summary.json": "execution_summary",
            "prepared.json": "prepared_record",
            "provenance.json": "provenance_record",
            "stdout.log": "stdout_compatibility_alias",
            "stderr.log": "stderr_compatibility_alias",
            "run_record.json": "run_record",
            "validation.json": "validation_record",
            "xtbopt.xyz": "optimized_geometry",
            "xtbopt.log": "trajectory",
            "xtb.trj": "trajectory",
            "xtblast.xyz": "restart_geometry",
            "restart.xyz": "restart_geometry",
            "xtbrestart": "restart_state",
            ".xtboptok": "non_authoritative_xtb_marker",
            "charges": "atomic_charges",
            "wbo": "wiberg_bond_orders",
            "hessian": "hessian",
            "vibspectrum": "vibrational_spectrum",
            "g98.out": "gaussian_frequency_output",
        }
        result: dict[str, str] = {}
        for name, role in role_names.items():
            path = attempt_root / name
            if path.is_file() and not path.is_symlink():
                result[self.sandbox.relative(path)] = role
        for path in sorted(attempt_root.iterdir(), key=lambda value: value.name.casefold()):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.name != "artifact_manifest.json"
                and self.sandbox.relative(path) not in result
            ):
                result[self.sandbox.relative(path)] = "generated_artifact"
        return result

    def _validate_initial_failure(self, attempt_id: str) -> dict[str, Any]:
        candidates = [
            self.sandbox.root / "attempts" / attempt_id,
            self.sandbox.root / attempt_id,
            self.sandbox.root / "initial_failure",
        ]
        root = next((path for path in candidates if path.is_dir() and not path.is_symlink()), None)
        if root is None:
            raise ToolInputError("NOT_FOUND", "initial failure attempt is not present")
        logs = [
            path for path in sorted(root.iterdir())
            if path.is_file() and path.suffix.casefold() in {".log", ".out", ".txt"}
        ]
        stdout = root / "stdout.log"
        if stdout in logs:
            logs.remove(stdout)
            logs.insert(0, stdout)
        parsed = None
        if logs:
            parsed = parse_xtb_log(self.sandbox, self.sandbox.relative(logs[0]))
        latest = parsed.get("latest_segment") if isinstance(parsed, Mapping) else None
        return {
            "attempt_id": attempt_id,
            "phase": "geometry_optimization",
            "state": "FROZEN_INITIAL_EVIDENCE",
            "classification": "OPTIMIZATION_NOT_CONVERGED",
            "phase_classification": "OPTIMIZATION_NOT_CONVERGED",
            "classification_priority": list(VALIDATION_CLASSIFICATION_PRIORITY),
            "validation_passed": False,
            "process_valid": False,
            "native_xtb_executed": False,
            "optimization_completed": bool(
                isinstance(latest, Mapping)
                and latest.get("optimization_converged_observed")
                and latest.get("normal_termination_observed")
            ),
            "hessian_completed": False,
            "normal_termination_observed": bool(
                isinstance(latest, Mapping) and latest.get("normal_termination_observed")
            ),
            "invariants_preserved": None,
            "source_current": None,
            "artifacts_current": None,
            "artifacts": {},
            "restart_sources": [],
            "warnings": ["initial evidence is read-only and not a generated native attempt"],
            "errors": ["initial attempt is not validated as complete recovery"],
        }

    def validate_attempt(self, attempt_id: str) -> dict[str, Any]:
        if not VALIDATE_ATTEMPT_PATTERN.fullmatch(attempt_id):
            raise ToolInputError("INVALID_ATTEMPT_ID", "attempt_id is not canonical")
        if attempt_id == "attempt_000_initial_failure":
            return self._validate_initial_failure(attempt_id)
        attempt_root = self._attempt_path(attempt_id)
        manifest_path = self._verify_execution_manifest(attempt_root, attempt_id)
        validation_path = attempt_root / "validation.json"
        if validation_path.is_file():
            existing = read_strict_json(validation_path)
            if not isinstance(existing, dict):
                raise ToolInputError("VALIDATION_RECORD_INVALID", "stored validation is not an object")
            legacy_errors = existing.get("errors") if isinstance(existing.get("errors"), list) else []
            legacy_artifact_integrity = bool(
                existing.get("invariants_preserved") is not False
                and existing.get("source_current") is not False
                and existing.get("artifacts_current") is not False
                and not any(
                    any(token in str(error).casefold() for token in ("invalid", "changed", "disagree", "invariant"))
                    for error in legacy_errors
                )
            )
            classification, phase_classification = _classify_attempt_validation(
                phase=existing.get("phase"),
                process_valid=existing.get("process_valid") is True,
                artifact_integrity_valid=legacy_artifact_integrity,
                optimization_completed=existing.get("optimization_completed") is True,
                hessian_completed=existing.get("hessian_completed") is True,
                warnings=existing.get("warnings") if isinstance(existing.get("warnings"), list) else [],
            )
            stored_classification = existing.get("classification")
            if stored_classification not in VALIDATION_CLASSIFICATION_PRIORITY:
                stored_classification = classification
            result = {
                **existing,
                "validation_record": self.sandbox.relative(validation_path),
                "validation_path": self.sandbox.relative(validation_path),
                "artifact_manifest": (
                    self.sandbox.relative(manifest_path) if manifest_path.is_file() else None
                ),
                "artifact_manifest_path": (
                    self.sandbox.relative(manifest_path) if manifest_path.is_file() else None
                ),
                "reused_immutable_record": True,
                "classification": stored_classification,
                "phase_classification": existing.get("phase_classification", phase_classification),
                "classification_priority": list(VALIDATION_CLASSIFICATION_PRIORITY),
            }
            result.update(
                {
                    "valid": existing.get("validation_passed") is True,
                    "optimization_converged": existing.get("optimization_completed") is True,
                    "current_artifacts": existing.get("artifacts_current") is True,
                    "scientific_invariants_preserved": existing.get("invariants_preserved") is True,
                    "source_provenance": {"current": existing.get("source_current") is True},
                }
            )
            return result
        prepared_path = attempt_root / "prepared.json"
        run_path = attempt_root / "run_record.json"
        canonical_prepared = (attempt_root / "attempt.json", attempt_root / "prepared_command.json")
        canonical_run = attempt_root / "execution_summary.json"
        canonical_provenance = attempt_root / "input_manifest.json"
        if (
            not prepared_path.is_file()
            or not run_path.is_file()
            or not canonical_run.is_file()
            or not canonical_provenance.is_file()
        ):
            raise ToolInputError("ATTEMPT_NOT_RUN", "attempt lacks prepared or native execution evidence")
        if any(not path.is_file() or sha256_file(path) != sha256_file(prepared_path) for path in canonical_prepared):
            raise ToolInputError("ATTEMPT_TAMPERED", "canonical prepared records differ")
        if sha256_file(canonical_run) != sha256_file(run_path):
            raise ToolInputError("ATTEMPT_TAMPERED", "canonical execution summary differs from run alias")
        if sha256_file(canonical_provenance) != sha256_file(attempt_root / "provenance.json"):
            raise ToolInputError("ATTEMPT_TAMPERED", "canonical input manifest differs from provenance alias")
        if (
            sha256_file(attempt_root / "stdout.txt") != sha256_file(attempt_root / "stdout.log")
            or sha256_file(attempt_root / "stderr.txt") != sha256_file(attempt_root / "stderr.log")
        ):
            raise ToolInputError("ATTEMPT_TAMPERED", "canonical output streams differ from compatibility aliases")
        prepared = read_strict_json(prepared_path)
        run = read_strict_json(run_path)
        if not isinstance(prepared, Mapping) or not isinstance(run, Mapping):
            raise ToolInputError("ATTEMPT_RECORD_INVALID", "attempt records are not JSON objects")
        phase = prepared.get("phase")
        errors: list[str] = []
        warnings: list[str] = []
        if run.get("attempt_id") != attempt_id or run.get("phase") != phase:
            errors.append("prepared and run records disagree on attempt identity or phase")
        input_path = attempt_root / "input.xyz"
        input_result = validate_xyz(
            self.sandbox,
            self.sandbox.relative(input_path),
            expected_symbols=self.expected_symbols,
        )
        if not input_result["valid"]:
            errors.append("prepared input geometry is invalid or violates atom-order invariants")
        process_valid = bool(
            run.get("state") == "EXECUTED"
            and run.get("native_xtb_executed") is True
            and run.get("shell") is False
            and run.get("timed_out") is False
            and run.get("exit_code") == 0
        )
        stdout_relative = self.sandbox.relative(attempt_root / "stdout.txt")
        log_result = parse_xtb_log(self.sandbox, stdout_relative)
        latest = log_result.get("latest_segment")
        latest = latest if isinstance(latest, Mapping) else {}
        stderr_result = parse_xtb_log(
            self.sandbox, self.sandbox.relative(attempt_root / "stderr.txt")
        )
        stderr_latest = stderr_result.get("latest_segment")
        stderr_latest = stderr_latest if isinstance(stderr_latest, Mapping) else {}

        def observed(key: str) -> bool:
            return latest.get(key) is True or stderr_latest.get(key) is True

        normal_termination = observed("normal_termination_observed")
        roles = self._generated_roles(attempt_root, str(phase))
        artifacts: dict[str, dict[str, Any]] = {}
        restart_sources: list[dict[str, Any]] = []
        for relative, role in roles.items():
            path = self.sandbox.resolve(relative)
            artifact_key = role
            if artifact_key in artifacts:
                artifact_key = f"{role}:{Path(relative).name}"
            artifacts[artifact_key] = {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "role": role,
            }
            if role in {"trajectory", "restart_geometry", "optimized_geometry"}:
                xyz = validate_xyz(self.sandbox, relative, expected_symbols=self.expected_symbols)
                restart_geometry_valid = bool(
                    xyz.get("valid") is True
                    or (
                        role in {"trajectory", "restart_geometry"}
                        and xyz.get("safely_recoverable") is True
                    )
                )
                if restart_geometry_valid:
                    restart_sources.append(
                        {
                            "path": relative,
                            "frame_index": xyz.get("selected_frame_index"),
                            "frame_sha256": xyz.get("selected_frame_sha256"),
                            "sha256": xyz.get("file_sha256"),
                            "role": role,
                        }
                    )
        optimized = artifacts.get("optimized_geometry")
        optimized_valid = False
        if optimized:
            xyz = validate_xyz(
                self.sandbox,
                optimized["path"],
                expected_symbols=self.expected_symbols,
            )
            optimized_valid = xyz.get("valid") is True
            if not optimized_valid:
                errors.append("optimized geometry artifact is invalid")
        hessian_artifact = artifacts.get("hessian")
        hessian_nonempty = bool(hessian_artifact and hessian_artifact["size_bytes"] > 0)
        optimization_completed = bool(
            phase == "geometry_optimization"
            and process_valid
            and normal_termination
            and observed("optimization_converged_observed")
            and optimized_valid
        )
        hessian_completed = bool(
            phase == "hessian"
            and process_valid
            and normal_termination
            and (
                observed("hessian_completed_observed")
                or observed("hessian_output_observed")
            )
            and hessian_nonempty
        )
        if phase == "geometry_optimization" and not optimization_completed:
            errors.append("geometry optimization lacks converged log and current valid xtbopt.xyz evidence")
        if phase == "hessian" and not hessian_completed:
            errors.append("hessian lacks completed log and current non-empty hessian evidence")
        if phase == "hessian" and "vibrational_spectrum" not in artifacts:
            warnings.append("vibrational spectrum artifact was not produced")
        expected_execution = self._build_argv(
            str(phase),
            prepared.get("request", {}).get("optimization_cycle_limit"),
            prepared.get("request", {}).get("thread_count"),
        )
        settings_match = prepared.get("scientific_settings") == self.settings
        argv_match = (
            run.get("argv")
            == expected_execution
            == prepared.get("execution", {}).get("argv")
            and prepared.get("execution", {}).get("scientific_argv")
            == self._scientific_argv(str(phase), prepared.get("request", {}).get("thread_count"))
            and prepared.get("execution", {}).get("runtime_wrapper_argv")
            == self._runtime_wrapper_argv()
        )
        invariants_preserved = bool(settings_match and argv_match and input_result["atom_order_consistent"])
        if not invariants_preserved:
            errors.append("scientific settings, trusted argv, or atom-order invariants do not match")
        run_files = run.get("files") if isinstance(run.get("files"), Mapping) else {}
        artifacts_current = True
        for relative, metadata in run_files.items():
            if not isinstance(relative, str) or not isinstance(metadata, Mapping):
                artifacts_current = False
                break
            try:
                path = self.sandbox.resolve(relative)
            except ToolInputError:
                artifacts_current = False
                break
            if metadata.get("sha256") != sha256_file(path) or metadata.get("size_bytes") != path.stat().st_size:
                artifacts_current = False
                break
        if not artifacts_current:
            errors.append("one or more execution artifacts changed after run_record creation")
        validation_passed = bool(
            process_valid
            and invariants_preserved
            and artifacts_current
            and (optimization_completed or hessian_completed)
            and not errors
        )
        try:
            source_current = bool(
                prepared.get("source_geometry", {}).get("file_sha256")
                == sha256_file(self.sandbox.resolve(prepared["source_geometry"]["path"]))
            )
        except (KeyError, ToolInputError, OSError):
            source_current = False
        if not source_current:
            errors.append("prepared source geometry is no longer hash-current")
            validation_passed = False
        artifact_integrity_valid = bool(
            input_result["valid"]
            and invariants_preserved
            and artifacts_current
            and source_current
            and (not optimized or optimized_valid)
            and run.get("attempt_id") == attempt_id
            and run.get("phase") == phase
        )
        classification, phase_classification = _classify_attempt_validation(
            phase=phase,
            process_valid=process_valid,
            artifact_integrity_valid=artifact_integrity_valid,
            optimization_completed=optimization_completed,
            hessian_completed=hessian_completed,
            warnings=warnings,
        )
        validation = {
            "schema_version": "1.0",
            "attempt_id": attempt_id,
            "phase": phase,
            "state": "VALIDATED",
            "classification": classification,
            "phase_classification": phase_classification,
            "classification_priority": list(VALIDATION_CLASSIFICATION_PRIORITY),
            "validation_passed": validation_passed,
            "process_valid": process_valid,
            "native_xtb_executed": run.get("native_xtb_executed") is True,
            "optimization_completed": optimization_completed,
            "hessian_completed": hessian_completed,
            "normal_termination_observed": normal_termination,
            "invariants_preserved": invariants_preserved,
            "source_current": source_current,
            "artifacts_current": artifacts_current,
            "artifacts": artifacts,
            "restart_sources": restart_sources,
            "warnings": warnings,
            "errors": errors,
            "validated_at": utc_now(),
        }
        write_json_exclusive(validation_path, validation)
        result = {
            **validation,
            "validation_record": self.sandbox.relative(validation_path),
            "validation_path": self.sandbox.relative(validation_path),
            "artifact_manifest": self.sandbox.relative(manifest_path),
            "artifact_manifest_path": self.sandbox.relative(manifest_path),
        }
        result.update(
            {
                "classification": validation["classification"],
                "valid": validation["validation_passed"],
                "optimization_converged": validation["optimization_completed"],
                "current_artifacts": validation["artifacts_current"],
                "scientific_invariants_preserved": validation["invariants_preserved"],
                "source_provenance": {"current": validation["source_current"]},
            }
        )
        return result


__all__ = ["ALL_INVARIANTS", "XtbAttemptManager"]

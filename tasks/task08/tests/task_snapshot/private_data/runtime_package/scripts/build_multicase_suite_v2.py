"""Build organizer-only xTB 6.7.1 failure cases from native executions.

The script refuses to overwrite existing hidden cases.  Stdout/stderr bytes are
captured directly from the installed executable; generated JSON is explicitly
labelled as benchmark metadata rather than native program output.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = Path(os.environ.get("XTB_BENCHMARK_ENV", sys.prefix))
XTB = ENV_ROOT / ("Library/bin/xtb.exe" if os.name == "nt" else "bin/xtb")
BASE_INPUT = ROOT / "benchmark_case" / "input" / "start.xyz"
HIDDEN_ROOT = ROOT / "organizer_private" / "cases"
CONFIG_ROOT = ROOT / "organizer_configs"


CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "xtb_hidden_02_scc_nonconvergence",
        "mode": "scc_nonconvergence",
        "argv": ["input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--iterations", "1"],
        "timeout": 60.0,
        "truth": {
            "primary_category": "SCC_FAILURE",
            "primary_cause_code": "SCC_NON_CONVERGENCE",
            "scheduler_assessment": "EXECUTION_FAILED",
            "conflict_resolution": "INCOMPLETE_OUTPUT_OVERRIDES_PROCESS_STATUS",
        },
    },
    {
        "case_id": "xtb_hidden_03_malformed_coordinates",
        "mode": "malformed_coordinates",
        "argv": ["malformed.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "normal"],
        "timeout": 60.0,
        "truth": {
            "primary_category": "INPUT_FAILURE",
            "primary_cause_code": "COORDINATE_FORMAT_ERROR",
            "scheduler_assessment": "EXECUTION_FAILED",
            "conflict_resolution": "INPUT_REJECTION_PREVENTED_SCIENTIFIC_RUN",
        },
    },
    {
        "case_id": "xtb_hidden_04_unsupported_option",
        "mode": "unsupported_option",
        "argv": ["input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--not-a-real-xtb-option"],
        "timeout": 60.0,
        "truth": {
            "primary_category": "INPUT_FAILURE",
            "primary_cause_code": "UNSUPPORTED_OPTION",
            "scheduler_assessment": "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE",
            "conflict_resolution": "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION",
        },
    },
    {
        "case_id": "xtb_hidden_05_native_timeout",
        "mode": "native_timeout",
        "argv": ["input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "normal"],
        "timeout": 0.05,
        "truth": {
            "primary_category": "RESOURCE_FAILURE",
            "primary_cause_code": "TIMEOUT",
            "scheduler_assessment": "EXECUTION_FAILED",
            "conflict_resolution": "INCOMPLETE_OUTPUT_OVERRIDES_PROCESS_STATUS",
        },
    },
    {
        "case_id": "xtb_hidden_06_wrong_phase",
        "mode": "wrong_phase_or_missing_output",
        "argv": ["input.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0"],
        "timeout": 60.0,
        "truth": {
            "primary_category": "PROGRAM_FAILURE",
            "primary_cause_code": "WRONG_PHASE_OR_MISSING_OUTPUT",
            "scheduler_assessment": "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE",
            "conflict_resolution": "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION",
        },
    },
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    write_bytes(path, payload)


def file_record(root: Path, path: Path, *, role: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if role is not None:
        record["role"] = role
    return record


def environment() -> dict[str, str]:
    allowed = (
        "PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR",
        "COMSPEC", "PATHEXT", "SYSTEMDRIVE", "NUMBER_OF_PROCESSORS",
    )
    result = {key: os.environ[key] for key in allowed if key in os.environ}
    prefix = [str(ENV_ROOT / "Library/bin"), str(ENV_ROOT / "Scripts"), str(ENV_ROOT)]
    result["PATH"] = os.pathsep.join(prefix + [result.get("PATH", "")])
    result.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONNOUSERSITE="1",
    )
    return result


def native_identity(env: dict[str, str]) -> dict[str, Any]:
    version = subprocess.run(
        [str(XTB), "--version"], shell=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env,
    )
    help_result = subprocess.run(
        [str(XTB), "--help"], shell=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env,
    )
    version_bytes = (version.stdout + version.stderr).replace(b"\r\n", b"\n")
    help_bytes = (help_result.stdout + help_result.stderr).replace(b"\r\n", b"\n")
    match = re.search(rb"xtb\s+version\s+([0-9]+(?:\.[0-9]+)+)", version_bytes, re.I)
    actual_version = match.group(1).decode("ascii") if match else "UNKNOWN"
    required = ["--opt", "--hess", "--gfn", "--chrg", "--uhf"]
    if actual_version != "6.7.1" or not all(token.encode() in help_bytes.lower() for token in required):
        raise RuntimeError("XTB_PORTABLE_IDENTITY_CONTRACT_FAILED")
    return {
        "version": actual_version,
        "identity_policy": "xtb_portable_identity/2.0",
        "required_help_tokens": required,
        "binary_sha256": sha256_file(XTB),
        "version_output_sha256": sha256_bytes(version_bytes),
        "help_output_sha256": sha256_bytes(help_bytes),
        "platform": platform.system(),
        "architecture": platform.machine(),
    }


def run_native(case: dict[str, Any], attempt: Path, env: dict[str, str]) -> dict[str, Any]:
    argv = ["xtb", *case["argv"]]
    started = datetime.now(timezone.utc)
    timed_out = False
    exit_code: int | None
    try:
        result = subprocess.run(
            argv, cwd=attempt, shell=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=float(case["timeout"]), env=env,
        )
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        exit_code = None
        timed_out = True
    completed = datetime.now(timezone.utc)
    write_bytes(attempt / "stdout.log", stdout.replace(b"\r\n", b"\n"))
    write_bytes(attempt / "stderr.log", stderr.replace(b"\r\n", b"\n"))
    return {
        "schema_version": "2.0",
        "source_type": "NATIVE_XTB_EXECUTION_RECEIPT",
        "argv": ["xtb", *case["argv"]],
        "shell": False,
        "stdin": "DEVNULL",
        "timeout_seconds": case["timeout"],
        "timed_out": timed_out,
        "exit_code": exit_code,
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": completed.isoformat().replace("+00:00", "Z"),
        "duration_seconds": round((completed - started).total_seconds(), 6),
        "stdout_sha256": sha256_file(attempt / "stdout.log"),
        "stderr_sha256": sha256_file(attempt / "stderr.log"),
    }


def build_job_spec(case_id: str) -> dict[str, Any]:
    base = json.loads((ROOT / "benchmark_case" / "job_spec.json").read_text(encoding="utf-8"))
    base["case_id"] = case_id
    base["schema_version"] = "2.0"
    return base


def build_case(case: dict[str, Any], identity: dict[str, Any], env: dict[str, str]) -> tuple[Path, Path]:
    root = HIDDEN_ROOT / case["case_id"]
    if root.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE:{root.name}")
    attempt = root / "attempts" / "attempt_000_initial_failure"
    attempt.mkdir(parents=True)
    write_bytes(root / "input" / "start.xyz", BASE_INPUT.read_bytes())
    write_bytes(attempt / "input.xyz", BASE_INPUT.read_bytes())
    if case["mode"] == "malformed_coordinates":
        write_bytes(
            attempt / "malformed.xyz",
            b"21\nintentionally malformed native-input probe\nC 0.0 0.0\n",
        )
    receipt = run_native(case, attempt, env)
    scheduler_state = "TIMEOUT" if receipt["timed_out"] else (
        "COMPLETED" if receipt["exit_code"] == 0 else "FAILED"
    )
    write_json(attempt / "command.json", receipt)
    write_json(
        attempt / "scheduler.json",
        {
            "schema_version": "2.0",
            "source_type": "BENCHMARK_EXECUTION_METADATA",
            "state": scheduler_state,
            "exit_code": receipt["exit_code"],
            "timed_out": receipt["timed_out"],
            "scientific_success_inference_allowed": False,
        },
    )
    write_json(attempt / "environment.json", identity)
    attempt_entries = [
        {
            **file_record(root, path, role="initial_failure_artifact"),
            "path": path.name,
        }
        for path in sorted(attempt.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file()
    ]
    write_json(
        attempt / "artifact_manifest.json",
        {
            "schema_version": "2.0",
            "attempt_id": "attempt_000_initial_failure",
            "entries": attempt_entries,
        },
    )
    write_json(root / "job_spec.json", build_job_spec(case["case_id"]))
    files = [
        file_record(root, path)
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path.name != "case_manifest.json"
    ]
    manifest = {
        "schema_version": "2.0",
        "case_id": case["case_id"],
        "source_type": "NATIVE_XTB_FAILURE_INJECTION",
        "failure_injection": {
            "mode": case["mode"],
            "disclosure": "Inputs/options are intentionally bounded; stdout/stderr are unedited native xTB bytes.",
        },
        "initial_failure": {
            "attempt_id": "attempt_000_initial_failure",
            "verified_not_converged": False,
        },
        "legacy": {
            "provenance_state": "NONE",
            "allowed_as_current_success": False,
            "allowed_as_restart_source": False,
        },
        "xtb_generation_identity": identity,
        "files": files,
    }
    write_json(root / "case_manifest.json", manifest)
    return root / "job_spec.json", root / "case_manifest.json"


def write_hidden_config(
    base: dict[str, Any], case: dict[str, Any], job_path: Path, manifest_path: Path
) -> Path:
    config = deepcopy(base)
    case_id = case["case_id"]
    config["schema_version"] = "2.0"
    config["task"]["id"] = "xtb_failure_diagnosis_restart_agent_v2"
    config["task"]["case_id"] = case_id
    config["task"]["job_spec"] = {
        "path": job_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(job_path),
    }
    config["task"]["case_manifest"] = {
        "path": manifest_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(manifest_path),
    }
    config["scoring"]["diagnosis_reference"] = {
        **case["truth"],
        "legacy_disposition": "NOT_INSPECTED",
        "scientific_core_cap_if_wrong": 45,
    }
    path = CONFIG_ROOT / f"{case_id}.yaml"
    payload = yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8")
    write_bytes(path, payload)
    return path


def migrate_public_metadata(identity: dict[str, Any]) -> None:
    manifest_path = ROOT / "benchmark_case" / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "2.0"
    manifest.pop("xtb", None)
    manifest["xtb_generation_identity"] = identity
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def main() -> None:
    if not XTB.is_file() or not BASE_INPUT.is_file():
        raise FileNotFoundError("XTB_OR_BASE_INPUT_MISSING")
    if HIDDEN_ROOT.exists() or CONFIG_ROOT.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_EXISTING_MULTICASE_SUITE")
    env = environment()
    identity = native_identity(env)
    base = yaml.safe_load((ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    suite_cases: list[dict[str, Any]] = []
    for case in CASES:
        job, manifest = build_case(case, identity, env)
        config = write_hidden_config(base, case, job, manifest)
        suite_cases.append(
            {
                "case_id": case["case_id"],
                "config": config.relative_to(ROOT).as_posix(),
                "workspace": manifest.parent.relative_to(ROOT).as_posix(),
                "failure_mode": case["mode"],
            }
        )
    migrate_public_metadata(identity)
    write_json(
        ROOT / "organizer_private" / "suite.json",
        {
            "schema_version": "2.0",
            "aggregation": {"case_weight": "equal", "blocked_cases_excluded": True},
            "public_case": {
                "case_id": "aspirin_gfn2_opt_hess_restart_v1",
                "config": "configs/task.yaml",
                "workspace": "benchmark_case",
                "failure_mode": "optimization_max_cycles",
            },
            "hidden_cases": suite_cases,
            "native_generation_identity": identity,
        },
    )
    print(json.dumps({"status": "BUILT", "hidden_case_count": len(suite_cases)}, sort_keys=True))


if __name__ == "__main__":
    main()

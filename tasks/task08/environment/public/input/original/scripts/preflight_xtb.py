from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _canonical_output(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _run(argv: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    system_root = os.environ.get("SystemRoot")
    if system_root:
        system32 = str(Path(system_root) / "System32")
        environment["PATH"] = system32 + os.pathsep + environment.get("PATH", "")
    return subprocess.run(
        argv,
        shell=False,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("task config must be a mapping")
    return loaded


def _resolve_conda_environment(conda: str, name: str) -> tuple[Path, Path]:
    result = _run([conda, "env", "list", "--json"])
    if result.returncode != 0:
        raise RuntimeError("CONDA_ENV_LIST_FAILED")
    payload = json.loads(result.stdout.decode("utf-8"))
    prefixes = [Path(value) for value in payload.get("envs", [])]
    matches = [prefix for prefix in prefixes if prefix.name.casefold() == name.casefold()]
    if len(matches) != 1:
        raise RuntimeError("XTB_CONDA_ENV_NOT_FOUND")
    relative = Path("Library/bin/xtb.exe") if os.name == "nt" else Path("bin/xtb")
    executable = matches[0] / relative
    if not executable.is_file():
        raise RuntimeError("XTB_EXECUTABLE_NOT_FOUND")
    return matches[0], executable


def collect_preflight(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    xtb_config = config["runtime"]["xtb"]
    conda_name = str(xtb_config["conda_environment"])
    conda = shutil.which("conda")
    errors: list[str] = []
    if not conda:
        return {
            "schema_version": "1.0",
            "passed": False,
            "errors": ["CONDA_NOT_FOUND"],
            "xtb": {"status": "MISSING"},
        }

    try:
        _, executable = _resolve_conda_environment(conda, conda_name)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "1.0",
            "passed": False,
            "errors": [str(exc)],
            "xtb": {"status": "MISSING"},
        }

    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest().upper()

    prefix = [conda, "run", "-n", conda_name, "--no-capture-output", "xtb"]
    version_result = _run(prefix + ["--version"])
    help_result = _run(prefix + ["--help"])
    version_bytes = _canonical_output(version_result.stdout + version_result.stderr)
    help_bytes = _canonical_output(help_result.stdout + help_result.stderr)
    version_match = re.search(
        rb"xtb\s+version\s+([0-9]+(?:\.[0-9]+)+)",
        version_bytes,
        flags=re.IGNORECASE,
    )
    actual_version = version_match.group(1).decode("ascii") if version_match else "UNKNOWN"
    if version_result.returncode != 0:
        errors.append("XTB_VERSION_COMMAND_FAILED")
    if help_result.returncode != 0:
        errors.append("XTB_HELP_COMMAND_FAILED")
    if actual_version != str(xtb_config["version"]):
        errors.append("XTB_VERSION_MISMATCH")
    help_hash = _sha256_bytes(help_bytes)
    required_help_tokens = xtb_config.get("required_help_tokens", [])
    help_lower = help_bytes.lower()
    if not all(
        isinstance(token, str) and token.encode("utf-8").lower() in help_lower
        for token in required_help_tokens
    ):
        errors.append("XTB_HELP_CONTRACT_MISMATCH")

    thread_env = {
        key: str(value)
        for key, value in xtb_config.get("environment", {}).items()
    }
    return {
        "schema_version": "1.0",
        "passed": not errors,
        "errors": errors,
        "xtb": {
            "status": "AVAILABLE",
            "executable_alias": "conda://xtb-benchmark/xtb",
            "version": actual_version,
            "binary_sha256": executable_hash,
            "version_output_sha256": _sha256_bytes(version_bytes),
            "help_output_sha256": help_hash,
            "identity_policy": xtb_config.get("identity_policy"),
            "required_help_tokens": required_help_tokens,
            "help_line_count": len(help_bytes.splitlines()),
        },
        "runtime": {
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "thread_environment": thread_env,
            "working_directory_policy": "one isolated attempt directory per native process",
            "native_subprocess_policy": {
                "argv_array": True,
                "shell": False,
                "stdin": "DEVNULL",
                "stdout": "CAPTURED",
                "stderr": "CAPTURED",
            },
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight the frozen native xTB runtime")
    parser.add_argument("--config", default="configs/task.yaml")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    report = collect_preflight(config_path)
    if args.output:
        output = (ROOT / args.output).resolve()
        try:
            output.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("output must remain below project root") from exc
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

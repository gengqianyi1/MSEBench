from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package_tools.archive_validation import (
    extract_safely,
    preflight_archive,
    validate_extracted_static,
)
from package_tools.manifest import sha256_file


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    for name in list(environment):
        if any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
            environment.pop(name, None)
    system_root = os.environ.get("SystemRoot")
    if system_root:
        environment["PATH"] = (
            str(Path(system_root) / "System32")
            + os.pathsep
            + environment.get("PATH", "")
        )
    return environment


def _run(command: list[str], cwd: Path, timeout: int = 1200) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=_environment(),
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )
    return {
        "passed": completed.returncode == 0,
        "command": [Path(command[0]).name] + command[1:],
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def dynamic_validation(root: Path) -> dict[str, Any]:
    commands = {
        "imports": _run(
            [
                sys.executable,
                "-c",
                (
                    "import baselines.baseline_agent,evaluator.evaluate,"
                    "framework.agent_runner,tools.tool_registry,package_tools.archive;"
                    "print('IMPORT_OK')"
                ),
            ],
            root,
        ),
        "pytest": _run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
            root,
        ),
        "baseline_replay": _run(
            [sys.executable, "baselines/baseline_agent.py", "--mode", "replay"],
            root,
        ),
        "three_model_replay": _run(
            [sys.executable, "scripts/replay_trajectories.py"],
            root,
        ),
    }
    powershell = Path(os.environ.get("SystemRoot", "")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if powershell.is_file():
        commands["run_all_ps1"] = _run(
            [
                str(powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(root / "run_all.ps1"),
                "tests",
            ],
            root,
        )
    else:
        commands["run_all_ps1"] = {
            "passed": False,
            "exit_code": None,
            "error": "POWERSHELL_NOT_FOUND",
        }
    return {
        "passed": all(result.get("passed") for result in commands.values()),
        "commands": commands,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a flat-root benchmark archive")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--extract-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--execute-trusted-code", action="store_true")
    args = parser.parse_args(argv)
    archive = Path(args.archive).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_VALIDATION_REPORT")
    identity_errors = []
    actual_hash = sha256_file(archive) if archive.is_file() else None
    if args.expected_sha256 and actual_hash != args.expected_sha256.upper():
        identity_errors.append("ARCHIVE_SHA256_MISMATCH")
    preflight = (
        preflight_archive(archive)
        if not identity_errors
        else {"passed": False, "errors": identity_errors}
    )
    static: dict[str, Any] = {"passed": False, "errors": ["NOT_EXTRACTED"]}
    dynamic: dict[str, Any] = {"passed": False, "executed": False}
    if preflight["passed"]:
        extracted = extract_safely(archive, extract_dir)
        static = validate_extracted_static(extracted)
        if static["passed"] and args.execute_trusted_code:
            dynamic = dynamic_validation(extracted)
            dynamic["executed"] = True
    passed = bool(
        preflight["passed"]
        and static["passed"]
        and args.execute_trusted_code
        and dynamic["passed"]
    )
    report = {
        "schema_version": "1.0",
        "passed": passed,
        "archive_filename": archive.name,
        "archive_size_bytes": archive.stat().st_size if archive.is_file() else None,
        "archive_sha256": actual_hash,
        "expected_sha256": args.expected_sha256.upper() if args.expected_sha256 else None,
        "preflight": preflight,
        "static_extraction_validation": static,
        "dynamic_validation": dynamic,
        "extract_dir_retained": preflight["passed"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

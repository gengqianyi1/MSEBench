#!/usr/bin/env python3
"""Run post-campaign syntax, pytest, and secret-scan gates and write a receipt."""

from __future__ import annotations

import argparse
import compileall
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    output.relative_to((ROOT / "reports").resolve())
    if output.exists() or output.is_symlink():
        raise FileExistsError("POST_RECHARGE_TEST_GATE_OUTPUT_EXISTS")

    compile_ok = compileall.compile_dir(
        ROOT,
        quiet=2,
        force=True,
        rx=re.compile(r"[\\/](?:\.git|\.pytest_cache|\.test_tmp|m|validation)[\\/]"),
    )
    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        shell=False,
    )
    pytest_text = pytest_run.stdout + pytest_run.stderr
    match = re.search(r"(\d+) passed in ([0-9.]+)s", pytest_text)
    secret_run = subprocess.run(
        [sys.executable, "-m", "scripts.secret_scan", "--root", str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        shell=False,
    )
    try:
        secret = json.loads(secret_run.stdout)
    except json.JSONDecodeError:
        secret = {"status": "FAIL", "parse_error": True}
    checks = {
        "python_syntax_compile": compile_ok,
        "pytest_exit_zero": pytest_run.returncode == 0,
        "pytest_count_not_below_historical_480": bool(
            match and int(match.group(1)) >= 480
        ),
        "working_tree_secret_scan_completed": secret.get("parse_error") is not True,
        "secret_findings_zero": secret.get("secret_finding_count") == 0,
        "real_env_absent_from_project": secret.get("real_env_present") is False,
    }
    report = {
        "schema_version": "post-recharge-test-gate/1.0",
        "python_version": platform.python_version(),
        "pytest": {
            "exit_code": pytest_run.returncode,
            "passed_count": int(match.group(1)) if match else None,
            "duration_seconds": float(match.group(2)) if match else None,
            "summary": match.group(0) if match else "PYTEST_SUMMARY_UNAVAILABLE",
        },
        "secret_scan": {
            **secret,
            "scope_policy": (
                "Working tree is checked for secrets and a real .env only. "
                "Role-specific forbidden-name and boundary checks run on "
                "participant and organizer staging trees and ZIPs."
            ),
            "scanner_exit_code": secret_run.returncode,
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_bytes(report))
    print(json.dumps({"status": report["status"], "output": str(output), "pytest": report["pytest"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

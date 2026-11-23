from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    system_root = os.environ.get("SystemRoot")
    if system_root:
        environment["PATH"] = (
            str(Path(system_root) / "System32")
            + os.pathsep
            + environment.get("PATH", "")
        )
    return environment


def run(arguments: list[str], *, timeout: int = 1800) -> int:
    completed = subprocess.run(
        [sys.executable] + arguments,
        cwd=ROOT,
        env=_environment(),
        shell=False,
        check=False,
        timeout=timeout,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified xTB benchmark launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("tests")
    subparsers.add_parser("build-case")
    subparsers.add_parser("baseline-replay")
    subparsers.add_parser("baseline-live")
    model = subparsers.add_parser("model")
    model.add_argument("logical_model", choices=["kimi", "deepseek", "glm"])
    subparsers.add_parser("replay-all")
    subparsers.add_parser("validate")
    subparsers.add_parser("package")
    args = parser.parse_args(argv)

    if args.command == "preflight":
        return run(["scripts/preflight_xtb.py"])
    if args.command == "tests":
        return run(["-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])
    if args.command == "build-case":
        return run(["scripts/build_initial_case.py"])
    if args.command == "baseline-replay":
        return run(["baselines/baseline_agent.py", "--mode", "replay"])
    if args.command == "baseline-live":
        return run(["baselines/baseline_agent.py", "--mode", "live"])
    if args.command == "model":
        return run(["scripts/run_model.py", "--model", args.logical_model], timeout=2400)
    if args.command == "replay-all":
        return run(["scripts/replay_trajectories.py"])
    if args.command == "validate":
        return run(["scripts/validate_project.py"], timeout=2400)
    if args.command == "package":
        # package_submission performs staging, exact-byte clean-room validation,
        # and atomic publication; a successful build alone is not sufficient.
        return run(["scripts/package_submission.py"], timeout=4800)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

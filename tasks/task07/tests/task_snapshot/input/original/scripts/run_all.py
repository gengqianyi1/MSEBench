from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def run(command: List[str]) -> int:
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified benchmark launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tests", help="Run all offline tests")

    baseline = sub.add_parser("baseline", help="Run deterministic offline baseline")
    baseline.add_argument("--output", default="outputs/baseline")
    baseline.add_argument("--config", default="configs/task.yaml")

    evaluate = sub.add_parser("evaluate", help="Evaluate one final_answer.json")
    evaluate.add_argument("submission")
    evaluate.add_argument("output")
    evaluate.add_argument("--config", default="configs/task.yaml")
    evaluate.add_argument(
        "--trajectory",
        default=None,
        help="Optional trajectory.jsonl used for verifiable process scoring",
    )

    model = sub.add_parser("model", help="Run one real configured model")
    model.add_argument("model_name")
    model.add_argument("--models-config", default="configs/models.yaml")
    model.add_argument("--task-config", default="configs/task.yaml")
    model.add_argument("--output-root", default="trajectories")

    models = sub.add_parser("models", help="Run Kimi, DeepSeek, and GLM sequentially")
    models.add_argument("--models-config", default="configs/models.yaml")
    models.add_argument("--task-config", default="configs/task.yaml")
    models.add_argument("--output-root", default="trajectories")

    package = sub.add_parser("package", help="Build the allowlisted ZIP")
    package.add_argument("--output-dir", default=None)
    package.add_argument("--workspace-root", default=None)
    package.add_argument("--staging-dir", default=None)
    package.add_argument("--filename", default=None)
    package.add_argument(
        "--status",
        choices=["BLOCKED", "READY_FOR_REVIEW"],
        default=None,
        help="Deprecated request only; evidence always determines the manifest status",
    )
    package.add_argument("--skip-test-run", action="store_true")

    validate = sub.add_parser("validate-package", help="Validate an independently extracted ZIP")
    validate.add_argument("archive")
    validate.add_argument("--extract-dir", required=True)
    validate.add_argument("--output", default="reports/package_validation.json")
    validate.add_argument("--workspace-root", default=None)
    validate.add_argument("--expected-sha256", default=None)
    validate.add_argument("--execute-trusted-code", action="store_true")

    args = parser.parse_args()

    if args.command == "tests":
        return run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])

    if args.command == "baseline":
        return run([
            sys.executable,
            "baselines/baseline_agent.py",
            "--config",
            args.config,
            "--output",
            args.output,
        ])

    if args.command == "evaluate":
        command = [
            sys.executable,
            "evaluator/evaluate.py",
            "--submission",
            args.submission,
            "--config",
            args.config,
            "--output",
            args.output,
        ]
        if args.trajectory:
            command.extend(["--trajectory", args.trajectory])
        return run(command)

    if args.command == "model":
        return run([
            sys.executable,
            "scripts/run_model.py",
            "--model",
            args.model_name,
            "--models-config",
            args.models_config,
            "--task-config",
            args.task_config,
            "--output-root",
            args.output_root,
        ])

    if args.command == "models":
        for name in ("kimi", "deepseek", "glm"):
            code = run([
                sys.executable,
                "scripts/run_model.py",
                "--model",
                name,
                "--models-config",
                args.models_config,
                "--task-config",
                args.task_config,
                "--output-root",
                args.output_root,
            ])
            if code != 0:
                return code
        return 0

    if args.command == "package":
        command = [sys.executable, "scripts/package_submission.py"]
        if args.output_dir:
            command.extend(["--output-dir", args.output_dir])
        if args.workspace_root:
            command.extend(["--workspace-root", args.workspace_root])
        if args.staging_dir:
            command.extend(["--staging-dir", args.staging_dir])
        if args.filename:
            command.extend(["--filename", args.filename])
        if args.status:
            command.extend(["--status", args.status])
        if args.skip_test_run:
            command.append("--skip-test-run")
        return run(command)

    if args.command == "validate-package":
        command = [
            sys.executable,
            "scripts/validate_package.py",
            "--archive",
            args.archive,
            "--extract-dir",
            args.extract_dir,
            "--output",
            args.output,
        ]
        if args.workspace_root:
            command.extend(["--workspace-root", args.workspace_root])
        if args.expected_sha256:
            command.extend(["--expected-sha256", args.expected_sha256])
        if args.execute_trusted_code:
            command.append("--execute-trusted-code")
        return run(command)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

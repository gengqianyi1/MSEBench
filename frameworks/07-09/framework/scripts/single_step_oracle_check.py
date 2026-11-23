#!/usr/bin/env python3
"""Execute and independently grade the lifecycle Oracles for Tasks 10-13."""
from __future__ import annotations

import importlib.util
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def heredoc_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.index("<<'JSON'\n") + len("<<'JSON'\n")
    end = text.index("\nJSON", start)
    return json.loads(text[start:end])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", type=int, default=[10, 11, 13])
    args = parser.parse_args()
    results = {}
    for number in args.tasks:
        if number not in {10, 11, 12, 13}:
            parser.error("task numbers must be one of 10, 11, 12, 13")
        task = next((ROOT / "tasks").glob(f"task-{number:02d}-*"))
        snapshot = task / "tests/task_snapshot"
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "workspace"
            shutil.copytree(task / "environment/public", work)
            if number == 10:
                env = dict(os.environ, CHEMBENCH_WORKSPACE=str(work))
                subprocess.run([sys.executable, str(task / "solution/solve.py")], check=True, env=env)
            else:
                payload = heredoc_json(task / "solution/solve.sh")
                (work / "final_answer.json").write_text(json.dumps(payload), encoding="utf-8")
            grader = load(f"task{number}_oracle_grader", snapshot / "private_grader/grader.py")
            context = SimpleNamespace(
                submission_dir=work,
                private_data_dir=snapshot / "private_data",
                run_dir=work / "run",
                task_dir=snapshot,
            )
            result = grader.grade(context)
            results[f"task-{number:02d}"] = {
                "score": result.get("score"),
                "failure_codes": result.get("failure_codes", []),
                "simulation_error": result.get("evidence", {}).get("simulation_error"),
            }
    print(json.dumps(results, indent=2))
    return 1 if any(value["failure_codes"] for value in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

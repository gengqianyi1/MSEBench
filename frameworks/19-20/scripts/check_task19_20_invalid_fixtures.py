"""Evaluate every declared invalid fixture at the private sidecar boundary."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from task_paths import task_path, PACKAGE

ROOT = Path(__file__).parents[1]
TASKS = (
    "task-19-active-impurity-elucidation",
    "task-20-catalyst-deactivation-diagnosis",
)


def load(path):
    spec = importlib.util.spec_from_file_location("fixture_adapter_" + path.parent.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for task_name in TASKS:
        tests = task_path(task_name) / "tests"
        adapter = load(tests / "private_suite_adapter.py").create_adapter(tests / "task_snapshot")
        for fixture in sorted((tests / "fixtures" / "invalid").glob("*.json")):
            state = adapter.start_session(None, "fixture-session-" + fixture.stem)
            candidate = json.loads(fixture.read_text(encoding="utf-8"))
            response = adapter.submit(0, candidate, [], state)
            results.append({
                "task": task_name,
                "fixture": fixture.relative_to(PACKAGE).as_posix(),
                "sidecar_accepted": response.get("ok") is True,
                "sidecar_status": response.get("status"),
                "expected_harbor_reward": 0.0,
                "expected_failure": "INVALID_SUBMISSION",
            })
    if len(results) != 6 or any(row["sidecar_accepted"] for row in results):
        raise SystemExit("invalid fixture contract failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "fixtures": len(results)}))


if __name__ == "__main__":
    main()

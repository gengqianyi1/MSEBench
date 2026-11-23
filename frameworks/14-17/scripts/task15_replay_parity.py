#!/usr/bin/env python3
"""Prove Task 15 Harbor round reconstruction preserves all calibrated legacy scores."""
from __future__ import annotations
import os as _os, sys as _sys  # noqa: E401
# Never write __pycache__ into tasks/: this script loads task code, and bytecode dropped there
# changes the task's Harbor checksum, so datasets/*/dataset.toml would stop matching the tree.
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/task-15-lipophilic-sulfonamide-edit-budget"
SNAPSHOT = TASK / "tests/task_snapshot"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    import_root = str(SNAPSHOT)
    sys.path.insert(0, import_root)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(import_root)
    return module


def main() -> int:
    final_grader = load_module("task15_final_grader", TASK / "tests/final_grader.py")
    legacy_grader = load_module("task15_legacy_grader", SNAPSHOT / "private_grader/grader.py")
    errors = []
    checked = 0
    for corpus in sorted((SNAPSHOT / "private_grader/replay_corpus").iterdir()):
        if not corpus.is_dir(): continue
        expected = json.loads((corpus / "expected.json").read_text(encoding="utf-8"))["total_score"]
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            records = [json.loads(line) for line in (corpus / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for record in records:
                candidate = record.get("submitted_candidate")
                number = record.get("round")
                if candidate and number:
                    path = workspace / "rounds" / f"round-{number}" / "candidate.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"candidate": candidate}), encoding="utf-8")
            final_grader.WORK = workspace
            final_grader.TASK = SNAPSHOT
            final_grader.main()
            context = SimpleNamespace(
                task_dir=SNAPSHOT,
                submission_dir=workspace,
                run_dir=workspace / "run",
                private_data_dir=SNAPSHOT / "private_data",
            )
            result = legacy_grader.grade(context)
            actual = result["score"]
            checked += 1
            if float(actual) != float(expected):
                errors.append({"corpus": corpus.name, "expected": expected, "actual": actual})
    print(json.dumps({"task": "task-15", "corpora_checked": checked, "errors": errors, "status": "PASS" if not errors else "FAIL"}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

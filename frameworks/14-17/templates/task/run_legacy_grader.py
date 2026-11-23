#!/usr/bin/env python3
"""Framework-owned compatibility boundary for a task-local legacy grader."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace

import yaml

TASK = Path("/task").resolve()
WORK = Path("/workspace").resolve()
LOG = Path("/logs/verifier").resolve()
RUN = Path("/run").resolve()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes {root}: {relative}")
    return path


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    metadata = yaml.safe_load((TASK / "task.yaml").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("task.yaml must contain an object")
    grader = _inside(TASK, str(metadata.get("grader_path", "private_grader/grader.py")))
    private_data = _inside(TASK, str(metadata.get("private_data_path", "private_data")))
    if not grader.is_file():
        raise FileNotFoundError(f"legacy grader not found: {grader}")
    minimum = float(metadata.get("min_score", 0))
    maximum = float(metadata.get("max_score", 100))
    pass_score = float(metadata.get("pass_score", minimum))
    if not all(math.isfinite(value) for value in (minimum, maximum, pass_score)) or maximum <= minimum:
        raise ValueError("invalid score bounds in task.yaml")
    specification = importlib.util.spec_from_file_location("chembench_legacy_grader", grader)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import grader: {grader}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    (RUN / "submission").mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    context = SimpleNamespace(
        task_dir=TASK,
        submission_dir=WORK,
        run_dir=RUN,
        private_data_dir=private_data,
    )
    entrypoint = str(metadata.get("grader_entrypoint", "grade"))
    function = getattr(module, entrypoint)
    legacy_result = function(context)
    if not isinstance(legacy_result, dict):
        raise TypeError("grader result must be an object")
    raw_score = legacy_result.get("score")
    if not isinstance(raw_score, (int, float)) or not math.isfinite(float(raw_score)):
        raise ValueError(f"grader returned invalid score: {raw_score!r}")
    reward = max(0.0, min(1.0, (float(raw_score) - minimum) / (maximum - minimum)))
    status_path = TASK / "HARBOR_MIGRATION_STATUS.txt"
    migration_status = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "unknown"
    _atomic_json(LOG / "details.json", {
        "schema_version": "chembench-verifier-details/1.0",
        "status": "scored",
        "legacy_result": legacy_result,
        "migration_status": migration_status,
    })
    _atomic_json(LOG / "reward.json", {
        "schema_version": "chembench-grader-result/1.0",
        "reward": reward,
        "raw_score": float(raw_score),
        "min_raw_score": minimum,
        "max_raw_score": maximum,
        "passed": bool(float(raw_score) >= pass_score),
        "status": "scored",
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        LOG.mkdir(parents=True, exist_ok=True)
        _atomic_json(LOG / "details.json", {
            "schema_version": "chembench-verifier-details/1.0",
            "status": "verifier_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        _atomic_json(LOG / "reward.json", {
            "schema_version": "chembench-grader-result/1.0",
            "reward": 0.0,
            "status": "verifier_error",
        })
        raise

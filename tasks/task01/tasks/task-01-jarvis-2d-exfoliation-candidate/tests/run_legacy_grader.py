#!/usr/bin/env python3
"""Framework-owned compatibility boundary for a task-local legacy grader."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
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


def _provision_public_replay_inputs() -> None:
    """Make fixed public scorer inputs available beside submitted artifacts.

    The sidecar evidence export is deliberately restricted to model-authored
    files.  Task 01's frozen native scorer nevertheless reads three immutable
    public files using ``workspace/data/...`` paths.  Copying them from the
    verifier's read-only task snapshot keeps the evidence boundary narrow and
    prevents a valid submission from being rejected solely because static
    public inputs were intentionally omitted from the submission snapshot.
    """
    source_dir = TASK / "input" / "original" / "data"
    target_dir = WORK / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.json.gz", "test_public.json.gz", "candidate_generation_constraints.json"):
        source = source_dir / name
        target = target_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"public replay input missing from task snapshot: {source}")
        if not target.is_file():
            shutil.copyfile(source, target)


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    _provision_public_replay_inputs()
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
    scientific_score = float(raw_score)
    efficiency_metadata = RUN / "llm_request_accounting.json"
    if not efficiency_metadata.is_file():
        # Harbor's verifier container may not share the controller's /run
        # bind in older Harbor 0.21 runners.  The sidecar therefore exports
        # the same controller-produced record inside the submission boundary.
        # Keep the fail-closed behavior when neither location exists.
        exported = WORK / "llm_request_accounting.json"
        if exported.is_file():
            efficiency_metadata = exported
        else:
            raise RuntimeError("missing Harbor control-plane LLM request accounting")
    metadata = json.loads(efficiency_metadata.read_text(encoding="utf-8"))
    actual_requests = metadata.get("actual_llm_requests")
    if not isinstance(actual_requests, int) or actual_requests < 1:
        raise ValueError("actual_llm_requests must be a positive integer")
    if metadata.get("counting_authority") != "harbor_control_plane":
        raise ValueError("LLM request accounting must be produced by Harbor control plane")
    # 2026-09-27 policy: strict request gate replaces their proportional multiplier min(1, 40/actual_llm_requests).
    LLM_REQUEST_LIMIT = 40
    llm_request_gate = "PASS" if actual_requests <= LLM_REQUEST_LIMIT else "LLM_REQUEST_LIMIT_EXCEEDED"
    efficiency = 1.0 if actual_requests <= LLM_REQUEST_LIMIT else 0.0
    final_score = scientific_score * efficiency
    reward = max(0.0, min(1.0, (final_score - minimum) / (maximum - minimum)))
    status_path = TASK / "HARBOR_MIGRATION_STATUS.txt"
    migration_status = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "unknown"
    _atomic_json(LOG / "details.json", {
        "schema_version": "chembench-verifier-details/1.0",
        "status": "scored",
        "legacy_result": legacy_result,
        "scientific_score": scientific_score,
        "actual_llm_requests": actual_requests,
        "llm_request_limit": LLM_REQUEST_LIMIT,
        "llm_request_gate": llm_request_gate,
        "efficiency": efficiency,
        "final_score": final_score,
        "migration_status": migration_status,
    })
    _atomic_json(LOG / "reward.json", {
        "reward": reward,
        "raw_score": scientific_score,
        "final_score": final_score,
        "efficiency": efficiency,
        "passed": 1.0 if final_score >= pass_score else 0.0,
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
            "reward": 0.0,
            "raw_score": 0.0,
            "passed": 0.0,
        })
        raise

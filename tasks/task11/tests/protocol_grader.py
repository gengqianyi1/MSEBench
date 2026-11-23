#!/usr/bin/env python3
"""Independent organizer-evidence replay grader for single-step Harbor tasks."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import traceback

import yaml


WORKSPACE = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace")).resolve()
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier")).resolve()
RUN = Path(os.environ.get("CHEMBENCH_RUN", "/run")).resolve()
SNAPSHOT = Path(os.environ.get("CHEMBENCH_TASK_SNAPSHOT", "/task")).resolve()
if not (SNAPSHOT / "task.yaml").is_file():
    SNAPSHOT = (Path(__file__).parent / "task_snapshot").resolve()


def _load_adapter():
    path = Path(__file__).parent / "private_suite_adapter.py"
    spec = importlib.util.spec_from_file_location("chembench_single_step_private_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load private evidence adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_adapter(SNAPSHOT)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    metadata = yaml.safe_load((SNAPSHOT / "task.yaml").read_text(encoding="utf-8"))
    adapter = _load_adapter()
    replay = os.environ.get("CHEMBENCH_REPLAY_EVIDENCE")
    evidence_dir = Path(replay).resolve() if replay else (RUN / "organizer_evidence").resolve()
    if replay:
        manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    else:
        manifest = adapter.capture_evidence(WORKSPACE, evidence_dir)
    result = adapter.grade_evidence(evidence_dir, RUN / "private_replay")
    raw_score = float(result["score"])
    minimum = float(metadata.get("min_score", 0.0))
    maximum = float(metadata.get("max_score", 100.0))
    pass_score = float(metadata.get("pass_score", minimum))
    if not all(math.isfinite(value) for value in (raw_score, minimum, maximum, pass_score)):
        raise ValueError("non-finite score contract")
    if maximum <= minimum:
        raise ValueError("invalid score bounds")
    reward = max(0.0, min(1.0, (raw_score - minimum) / (maximum - minimum)))
    details = {
        "schema_version": "chembench-verifier-details/1.0",
        "status": "scored",
        "task_id": metadata["task_id"],
        "protocol": "single_step_organizer_evidence_replay/1.0",
        "evidence_id": manifest["evidence_id"],
        "artifact_digests": manifest["artifacts"],
        "private_result": result,
    }
    reward_row = {
        "schema_version": "chembench-grader-result/1.0",
        "reward": reward,
        "raw_score": raw_score,
        "min_raw_score": minimum,
        "max_raw_score": maximum,
        "passed": raw_score >= pass_score,
        "status": "scored",
    }
    _atomic_json(LOGS / "details.json", details)
    _atomic_json(LOGS / "reward.json", {"reward": reward})
    _atomic_json(LOGS / "grader-result.json", {**reward_row, "evidence_id": manifest["evidence_id"]})
    (LOGS / "reward.txt").write_text(str(reward), encoding="utf-8")
    print(json.dumps({"task_id": metadata["task_id"], "score": raw_score, "evidence_id": manifest["evidence_id"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        LOGS.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            LOGS / "details.json",
            {
                "schema_version": "chembench-verifier-details/1.0",
                "status": "verifier_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        _atomic_json(
            LOGS / "reward.json",
            {"reward": 0.0},
        )
        raise

#!/usr/bin/env python3
"""Replay Task 05 solely from organizer evidence and the unchanged private evaluator."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


WORK = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
EVIDENCE = Path(os.environ.get("CHEMBENCH_ORGANIZER_EVIDENCE", "/organizer_evidence"))
SNAPSHOT = Path("/task")
if not SNAPSHOT.is_dir():
    SNAPSHOT = Path(__file__).resolve().parent / "task_snapshot"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path, *, exclude_manifest: bool) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if exclude_manifest and relative == "run_manifest.json":
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metrics(value) -> dict:
    defaults = {
        "agent_turns_used": 0,
        "scientific_turns_used": 0,
        "finalization_turns_used": 0,
        "tool_calls_used": 0,
        "termination_reason": "unknown",
    }
    if not isinstance(value, dict):
        return defaults
    for key in defaults:
        item = value.get(key)
        if key == "termination_reason":
            defaults[key] = item if isinstance(item, str) else "invalid_agent_metrics"
        elif isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            defaults[key] = item
    return defaults


def _outer_manifest(submission: Path, metrics: dict) -> dict:
    inner_path = submission / "run_manifest.json"
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    jobs = [item for item in inner.get("jobs", []) if isinstance(item, dict)]
    runtime = inner.get("runtime_checks", {})
    hashes = {
        key: inner[key]
        for key in (
            "environment_lock_sha256",
            "task_config_sha256",
            "public_inputs_sha256",
            "xtb_executable_sha256",
        )
    }
    budget = dict(inner.get("budget_ledger", {}))
    # The organizer runtime seals this ledger before materializing the submission.
    # Replacing totals alone would invalidate its cumulative event and mix scientific
    # turns with the separate finalization budget.
    return {
        "schema_name": "harness_run_manifest_v2",
        "schema_version": "2.1.0",
        "manifest_role": "harness",
        "run_id": str(inner.get("run_id") or "harbor-task05"),
        "task_id": "task_05",
        "task_version": "1.15.0",
        "grader_version": "1.21.0",
        **hashes,
        "runtime_checks": runtime,
        "budget_ledger": budget,
        "provider": "harbor-agent",
        "provider_config": {"protocol": "workspace_sidecar_case"},
        "requested_model_id": "harbor-agent",
        "returned_model_ids": ["harbor-agent"],
        "execution_status": "COMPLETED",
        "submission_status": "SUBMITTED",
        "scoring_status": "SCORED",
        "state_transitions": [
            {"field": "execution_status", "from": None, "to": "COMPLETED", "reason": metrics["termination_reason"]},
            {"field": "submission_status", "from": "NOT_CHECKED", "to": "SUBMITTED", "reason": "sidecar evidence snapshot"},
            {"field": "scoring_status", "from": "NOT_SCORED", "to": "SCORED", "reason": "private evaluator replay"},
        ],
        "submission_manifest_path": "submission/run_manifest.json",
        "submission_manifest_sha256": _sha256_file(inner_path),
        "submission_payload_sha256": str(inner.get("submission_sha256") or _tree_hash(submission, exclude_manifest=True)),
        "submission_sha256": _tree_hash(submission, exclude_manifest=False),
    }


def grade_evidence(
    submission: Path,
    snapshot: Path,
    harness_manifest: Path | dict | None = None,
    agent_metrics: dict | None = None,
) -> dict:
    source = Path(snapshot) / "source_package"
    sys.path.insert(0, str(source))
    from tasks.task_05.private_grader.grader import grade

    metrics = _metrics(agent_metrics)
    if isinstance(harness_manifest, Path):
        outer = json.loads(harness_manifest.read_text(encoding="utf-8"))
    elif isinstance(harness_manifest, dict):
        outer = harness_manifest
    else:
        outer = _outer_manifest(Path(submission), metrics)
    with tempfile.TemporaryDirectory(prefix="chembench-task05-grade-") as temporary:
        run_dir = Path(temporary) / "grader"
        run_dir.mkdir()
        result = grade(SimpleNamespace(run_dir=run_dir, run_manifest=outer, submission_dir=Path(submission)))
        replay = json.loads((run_dir / "canonical_replay.json").read_text(encoding="utf-8"))
    score = float(result.get("score", 0.0))
    result.update(
        {
            "task_id": "task_05",
            "grader_version": "1.21.0",
            "task_local_score": score,
            "scientific_basis_score": score,
            "comparison_score": score,
            "canonical_replay_consistent": bool(replay.get("matched")),
        }
    )
    return result


def _case_root(artifact: dict) -> Path:
    submission = artifact.get("submission")
    evidence_id = submission.get("evidence_id") if isinstance(submission, dict) else None
    parts = Path(str(evidence_id or "")).parts
    if len(parts) != 2 or parts[1] != "task05-caffeine-water-binding-decomposition":
        raise ValueError("invalid evidence id")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid evidence id")
    case_root = (EVIDENCE / Path(*parts)).resolve()
    if EVIDENCE.resolve() not in case_root.parents:
        raise ValueError("evidence path escape")
    return case_root


def main() -> int:
    artifact = json.loads((EVIDENCE / "current.json").read_text(encoding="utf-8"))
    case_root = _case_root(artifact)
    result = grade_evidence(case_root / "submission", SNAPSHOT, agent_metrics=artifact.get("agent_metrics"))
    if not result["canonical_replay_consistent"]:
        raise RuntimeError("canonical replay mismatch")
    score = float(result["score"])
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text(str(max(0.0, min(1.0, score / 100.0))), encoding="utf-8")
    _write(LOGS / "grader-result.json", result)
    _write(WORK / "benchmark_result.json", {"task_id": "task_05", "score": score, "reward": score / 100.0})
    print(json.dumps({"score": score, "replay_consistent": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

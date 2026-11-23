#!/usr/bin/env python3
"""Replay Task 06 solely from organizer evidence and the private reference."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


WORK = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
EVIDENCE = Path(os.environ.get("CHEMBENCH_ORGANIZER_EVIDENCE", "/organizer_evidence"))
SNAPSHOT = Path("/task")
if not SNAPSHOT.is_dir():
    SNAPSHOT = Path(__file__).resolve().parent / "task_snapshot"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical(result: dict) -> dict:
    return {
        key: result.get(key)
        for key in (
            "task_id",
            "task_version",
            "grader_version",
            "hard_zero",
            "hard_zero_reason",
            "sections",
            "derived",
            "score",
            "task_local_score",
            "scientific_basis_score",
            "comparison_score",
            "contract_errors",
        )
    }


def grade_evidence(evidence_run: Path, snapshot: Path) -> dict:
    source = Path(snapshot) / "source_package"
    sys.path.insert(0, str(source))
    from tasks.task_06.private.grader import grade_task06

    root = Path(evidence_run)
    submission = root / "accepted_submission"
    if not submission.is_dir():
        submission = root / "submission"
    first = grade_task06(
        submission,
        root,
        Path(snapshot) / "private_reference",
        Path(snapshot) / "public",
    )
    second = grade_task06(
        submission,
        root,
        Path(snapshot) / "private_reference",
        Path(snapshot) / "public",
    )
    consistent = _canonical(first) == _canonical(second)
    if not consistent:
        raise RuntimeError("canonical replay mismatch")
    first["canonical_replay_consistent"] = True
    return first


def _evidence_run(artifact: dict) -> Path:
    submission = artifact.get("submission")
    evidence_id = submission.get("evidence_id") if isinstance(submission, dict) else None
    parts = Path(str(evidence_id or "")).parts
    if len(parts) != 2 or parts[1] != "task06-terpyridine-coupled-torsion-landscape":
        raise ValueError("invalid evidence id")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid evidence id")
    case_root = (EVIDENCE / Path(*parts)).resolve()
    if EVIDENCE.resolve() not in case_root.parents:
        raise ValueError("evidence path escape")
    return case_root / "run"


def main() -> int:
    artifact = json.loads((EVIDENCE / "current.json").read_text(encoding="utf-8"))
    result = grade_evidence(_evidence_run(artifact), SNAPSHOT)
    score = float(result.get("score", 0.0))
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text(str(max(0.0, min(1.0, score / 100.0))), encoding="utf-8")
    _write(LOGS / "grader-result.json", result)
    _write(LOGS / "canonical-replay.json", {"consistent": True, "canonical": _canonical(result)})
    _write(WORK / "benchmark_result.json", {"task_id": "task_06", "score": score, "reward": score / 100.0})
    print(json.dumps({"score": score, "replay_consistent": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

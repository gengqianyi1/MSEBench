#!/usr/bin/env python3
"""Replay Task 18 solely from organizer-owned xTB evidence."""
from __future__ import annotations

import hashlib
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


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    if not Path(root).is_dir():
        return digest.hexdigest()
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validated_metrics(value: object) -> dict[str, object]:
    required = {
        "agent_turns_used",
        "scientific_turns_used",
        "finalization_turns_used",
        "tool_calls_used",
        "termination_reason",
    }
    if not isinstance(value, dict) or set(value) != required:
        return {
            "agent_turns_used": 0,
            "scientific_turns_used": 0,
            "finalization_turns_used": 0,
            "tool_calls_used": 0,
            "termination_reason": "invalid_agent_metrics",
        }
    result = dict(value)
    for key in required - {"termination_reason"}:
        item = result[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            result[key] = 0
    if not isinstance(result["termination_reason"], str):
        result["termination_reason"] = "invalid_agent_metrics"
    return result


def grade_evidence(evidence_run: Path, snapshot: Path, agent_metrics: object) -> dict[str, object]:
    grader_root = Path(snapshot) / "private_grader"
    sys.path.insert(0, str(grader_root))
    from grader import grade_task18

    submission = Path(evidence_run) / "accepted_submission"
    if not submission.is_dir():
        submission = Path(evidence_run) / "model" / "submission"
    result = grade_task18(
        submission,
        Path(evidence_run),
        Path(snapshot) / "private_reference",
        Path(snapshot) / "public",
        _validated_metrics(agent_metrics),
    )
    result["accepted_submission_sha256"] = _tree_sha256(submission)
    result["organizer_evidence_sha256"] = _tree_sha256(Path(evidence_run) / "jobs")
    identity = json.loads((Path(snapshot) / "public/identity_registry.json").read_text(encoding="utf-8"))
    if result["task_id"] not in identity["aliases"]:
        raise ValueError("UNKNOWN_TASK_ALIAS")
    result["instance_id"] = identity["instance_id"]
    result["scientific_problem_id"] = identity["scientific_problem_id"]
    return result


def _evidence_run(artifact: dict[str, object]) -> Path:
    submission = artifact.get("submission")
    evidence_id = submission.get("evidence_id") if isinstance(submission, dict) else None
    parts = Path(str(evidence_id or "")).parts
    if len(parts) != 2 or parts[1] != "task18-redox-reorganization":
        raise ValueError("invalid evidence id")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid evidence id")
    case_root = (EVIDENCE / Path(*parts)).resolve()
    evidence_root = EVIDENCE.resolve()
    if evidence_root not in case_root.parents:
        raise ValueError("evidence path escape")
    return case_root / "run"


def _canonical(result: dict[str, object]) -> dict[str, object]:
    keys = (
        "task_id",
        "grader_version",
        "hard_zero",
        "hard_zero_reason",
        "submission_contract_valid",
        "contract_errors",
        "sections",
        "derived",
        "score",
        "task_local_score",
        "scientific_basis_score",
        "comparison_score",
        "accepted_submission_sha256",
        "organizer_evidence_sha256",
    )
    return {key: result.get(key) for key in keys}


def main() -> int:
    artifact = json.loads((EVIDENCE / "current.json").read_text(encoding="utf-8"))
    evidence_run = _evidence_run(artifact)
    if artifact.get("public_case", {}).get("task", {}).get("id") != TASK_ID:
        raise ValueError("ORGANIZER_TASK_MISMATCH")
    accepted = evidence_run / "accepted_submission"
    if not accepted.is_dir():
        accepted = evidence_run / "model/submission"
    for name in ("final_answer.json", "report.md"):
        sealed, visible = accepted / name, WORK / name
        if sealed.is_file() != visible.is_file() or (sealed.is_file() and sealed.read_bytes() != visible.read_bytes()):
            raise ValueError("PUBLIC_SUBMISSION_EVIDENCE_MISMATCH")
    first = grade_evidence(evidence_run, SNAPSHOT, artifact.get("agent_metrics"))
    second = grade_evidence(evidence_run, SNAPSHOT, artifact.get("agent_metrics"))
    canonical_first = _canonical(first)
    if canonical_first != _canonical(second):
        raise RuntimeError("canonical replay mismatch")
    score = max(0.0, min(100.0, float(first.get("score", 0.0))))
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text(str(score / 100.0), encoding="utf-8")
    _write(LOGS / "grader-result.json", first)
    _write(LOGS / "canonical-replay.json", {"consistent": True, "canonical": canonical_first})
    _write(WORK / "benchmark_result.json", {"task_id": TASK_ID, "instance_id": first["instance_id"], "scientific_problem_id": first["scientific_problem_id"], "score": score, "reward": score / 100.0})
    print(json.dumps({"score": score, "replay_consistent": True}))
    return 0


TASK_ID = "task_18_redox_reorganization"


if __name__ == "__main__":
    raise SystemExit(main())

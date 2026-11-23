#!/usr/bin/env python3
"""Replay Task 04 solely from organizer evidence and the private reference."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


WORK = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
EVIDENCE = Path(os.environ.get("CHEMBENCH_ORGANIZER_EVIDENCE", "/organizer_evidence"))
SNAPSHOT = Path("/task")
if not SNAPSHOT.is_dir():
    SNAPSHOT = Path(__file__).resolve().parent / "task_snapshot"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validated_metrics(value) -> dict[str, object]:
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


def grade_evidence(
    evidence_run: Path,
    snapshot: Path,
    agent_metrics: dict,
    max_turns: int,
) -> dict:
    source = Path(snapshot) / "source_package"
    sys.path.insert(0, str(source))
    from tasks.task_04.private.grader import grade_task04

    metrics = _validated_metrics(agent_metrics)
    with tempfile.TemporaryDirectory(prefix="chembench-task04-grade-") as temporary:
        projected = Path(temporary) / "run"
        shutil.copytree(Path(evidence_run), projected)
        turns = int(metrics["agent_turns_used"])
        trajectory = "".join(
            json.dumps({"type": "assistant", "turn": turn}, sort_keys=True) + "\n"
            for turn in range(1, turns + 1)
        )
        (projected / "trajectory.jsonl").write_text(trajectory, encoding="utf-8")
        _write(
            projected / "run_config.json",
            {
                "max_turns": int(max_turns),
                "organizer_trusted_agent_metrics": metrics,
            },
        )
        submission = projected / "accepted_submission"
        if not submission.is_dir():
            submission = projected / "submission"
        return grade_task04(
            submission,
            projected,
            Path(snapshot) / "private_reference",
            Path(snapshot) / "public",
        )


def _evidence_run(artifact: dict) -> Path:
    submission = artifact.get("submission")
    evidence_id = submission.get("evidence_id") if isinstance(submission, dict) else None
    parts = Path(str(evidence_id or "")).parts
    if len(parts) != 2 or parts[1] != "task04-histidine-microstate-ensemble":
        raise ValueError("invalid evidence id")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid evidence id")
    case_root = (EVIDENCE / Path(*parts)).resolve()
    evidence_root = EVIDENCE.resolve()
    if evidence_root not in case_root.parents:
        raise ValueError("evidence path escape")
    return case_root / "run"


def main() -> int:
    artifact = json.loads((EVIDENCE / "current.json").read_text(encoding="utf-8"))
    budget = artifact.get("public_case", {}).get("budget", {})
    max_turns = int(budget.get("max_agent_turns", 160))
    evidence_run = _evidence_run(artifact)
    first = grade_evidence(evidence_run, SNAPSHOT, artifact.get("agent_metrics"), max_turns)
    second = grade_evidence(evidence_run, SNAPSHOT, artifact.get("agent_metrics"), max_turns)
    canonical_first = {
        key: first.get(key)
        for key in (
            "task_id",
            "grader_version",
            "hard_zero",
            "hard_zero_reason",
            "sections",
            "derived",
            "score",
            "task_local_score",
            "scientific_basis_score",
            "comparison_score",
        )
    }
    canonical_second = {key: second.get(key) for key in canonical_first}
    replay_consistent = canonical_first == canonical_second
    if not replay_consistent:
        raise RuntimeError("canonical replay mismatch")
    score = float(first.get("score", 0.0))
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text(str(max(0.0, min(1.0, score / 100.0))), encoding="utf-8")
    _write(LOGS / "grader-result.json", first)
    _write(LOGS / "canonical-replay.json", {"consistent": True, "canonical": canonical_first})
    _write(WORK / "benchmark_result.json", {"task_id": "task_04", "score": score, "reward": score / 100.0})
    print(json.dumps({"score": score, "replay_consistent": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

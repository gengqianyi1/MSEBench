"""Offline, no-API/no-xTB replay of frozen formal trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.trajectory import render_dialogue
from scripts.runtime_support import (
    RuntimeIntegrityError,
    deterministic_evaluation,
    load_json,
    sha256_file,
    verify_file_manifest,
)


FORMAL_MODELS = ("kimi", "deepseek", "glm")


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeIntegrityError("trajectory event is not an object")
        events.append(value)
    return events


def replay_one(logical_model: str) -> dict[str, Any]:
    if logical_model not in FORMAL_MODELS:
        raise RuntimeIntegrityError("unknown logical model")
    root = ROOT / "trajectories" / logical_model
    required = {
        "trajectory.jsonl",
        "dialogue.md",
        "metadata.json",
        "final_answer.json",
        "evaluation.json",
        "artifact_manifest.json",
        "artifacts",
    }
    if not root.is_dir() or any(not (root / name).exists() for name in required):
        raise RuntimeIntegrityError("formal trajectory is incomplete")
    result, replay_bytes = deterministic_evaluation(
        final_answer=root / "final_answer.json",
        trajectory=root / "trajectory.jsonl",
        artifacts=root / "artifacts",
    )
    frozen_bytes = (root / "evaluation.json").read_bytes()
    events = _load_events(root / "trajectory.jsonl")
    dialogue_matches = (root / "dialogue.md").read_text(
        encoding="utf-8"
    ) == render_dialogue(events)
    manifest_matches = verify_file_manifest(root, root / "artifact_manifest.json")
    byte_identical = replay_bytes == frozen_bytes
    passed = bool(
        byte_identical
        and dialogue_matches
        and manifest_matches
        and result.get("eligibility", {}).get("eligible")
    )
    return {
        "logical_model": logical_model,
        "status": "PASS" if passed else "FAIL",
        "evaluation_byte_identical": byte_identical,
        "dialogue_generated_from_trajectory": dialogue_matches,
        "artifact_manifest_valid": manifest_matches,
        "evaluator_eligible": bool(result.get("eligibility", {}).get("eligible")),
        "official_score": result.get("official_score", 0),
        "trajectory_sha256": sha256_file(root / "trajectory.jsonl"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay frozen trajectories offline")
    parser.add_argument("--model", choices=FORMAL_MODELS)
    args = parser.parse_args(argv)
    models = (args.model,) if args.model else FORMAL_MODELS
    results: list[dict[str, Any]] = []
    try:
        for model in models:
            results.append(replay_one(model))
    except (RuntimeIntegrityError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__}, sort_keys=True))
        return 2
    passed = all(item["status"] == "PASS" for item in results)
    print(
        json.dumps(
            {"schema_version": "1.0", "status": "PASS" if passed else "FAIL", "results": results},
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

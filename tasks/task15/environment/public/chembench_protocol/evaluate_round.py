#!/usr/bin/env python3
"""Public, deterministic Task 15 feedback adapter for Harbor multi-step execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PUBLIC_SRC = Path("/workspace/input/src")
sys.path.insert(0, str(PUBLIC_SRC))
from evaluator import evaluate_smiles  # noqa: E402
from feedback import generate_feedback  # noqa: E402
from objective import select_best_by_objective  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    value = value.get("candidate", value) if isinstance(value, dict) else None
    if not isinstance(value, dict) or not str(value.get("smiles", "")).strip():
        raise ValueError("candidate must contain a non-empty smiles field")
    return {
        "name": str(value.get("name", "")).strip(),
        "smiles": str(value["smiles"]).strip(),
        "rationale": str(value.get("rationale", "")).strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.task_config).read_text(encoding="utf-8"))
    start_smiles = config["start_smiles"]
    constraints = config.get("constraints") or {}
    similarity = constraints.get("similarity") or {}
    target = config["optimization_target"]
    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"submitted": []}
    try:
        submitted = candidate(Path(args.candidate))
    except (OSError, ValueError) as exc:
        # No candidate file, an unreadable one, or one without a SMILES is what a model that failed the
        # previous round leaves behind. That is a SCORED outcome -- the verifier counts the round as missing
        # and the completeness gate sets the reward -- so it must not stop the next round from starting.
        # Until 2026-09-16 this path raised (or exited 1), `setup.sh` runs under `set -e`, and Harbor aborts
        # the whole trial when a step's setup fails: one bad round erased the run and left reward None.
        atomic_json(Path(args.output), {"status": "missing_candidate", "round_evaluated": args.round - 1,
                                        "reason": type(exc).__name__,
                                        "note": "no parseable candidate was submitted for this round; it counts as a missing round"})
        print(json.dumps({"round_evaluated": args.round - 1, "valid": False, "status": "missing_candidate"}))
        return 0
    state["submitted"].append({"round": args.round - 1, "candidate": submitted})
    evaluations = [
        evaluate_smiles(item["candidate"]["smiles"], start_smiles=start_smiles, similarity_cfg=similarity)
        for item in state["submitted"]
    ]
    start = evaluate_smiles(start_smiles, start_smiles=start_smiles, similarity_cfg=similarity)
    best = select_best_by_objective(evaluations, target)
    current = evaluations[-1]
    feedback = generate_feedback(evaluation=current, start_eval=start, constraints=constraints, best_so_far=best)
    # The legacy feedback function is the sole authority for visible fields. Private reference data
    # is not available in this container and cannot enter this file.
    atomic_json(Path(args.output), feedback)
    atomic_json(state_path, state)
    print(json.dumps({"round_evaluated": args.round - 1, "valid": bool(current.valid)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

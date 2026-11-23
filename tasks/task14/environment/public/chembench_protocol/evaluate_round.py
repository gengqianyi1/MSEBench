#!/usr/bin/env python3
"""Science-only per-round evaluator for Task 14 (M1.B).

Reads candidate.json -> recomputes RDKit descriptors -> writes reduced-bandwidth
feedback.json. No LLM, no private reference, no score band.

Usage:
  python evaluate_round.py --round 2 \
    --candidate /workspace/rounds/round-2/candidate.json \
    --output   /workspace/rounds/round-2/feedback.json \
    --task-config /workspace/input/tasks/q1_multiobjective/task_config.yaml \
    --state /workspace/rounds/best_so_far.json
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'input' / 'src'))  # science modules live in input/src

from evaluator import evaluate_smiles
from feedback import generate_feedback
from utils import load_json, load_yaml, write_json


def _candidate_smiles(path: Path) -> str:
    """SMILES of the previous round's candidate, or "" when that round has none.

    No candidate file, an unreadable one, or one without a SMILES is what a model that failed the
    previous round leaves behind. That is a SCORED outcome -- the verifier counts the round as missing
    and the completeness gate sets the reward -- so it must not stop the next round from starting.
    Until 2026-09-16 this path raised (or exited 1), `setup.sh` runs under `set -e`, and Harbor aborts
    the whole trial when a step's setup fails: one bad round erased the run and left reward None.
    """
    try:
        data = load_json(path)
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    cand = data.get("candidate", data)
    if not isinstance(cand, dict):
        return ""
    return str(cand.get("smiles", "")).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task-config", type=Path, required=True)
    ap.add_argument("--state", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.task_config)
    start_smiles = cfg.get("start_smiles")
    constraints = cfg.get("constraints", {})

    smiles = _candidate_smiles(args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not smiles:
        write_json(args.output, {"status": "missing_candidate", "round_evaluated": args.round - 1,
                                 "computed_properties": None,
                                 "note": "no parseable candidate was submitted for this round; it counts as a missing round"})
        return 0

    ev = evaluate_smiles(smiles, start_smiles=start_smiles)
    start_ev = evaluate_smiles(start_smiles) if start_smiles else ev

    best_qed = None
    if args.state and args.state.exists():
        try:
            best_qed = load_json(args.state).get("best_qed_so_far")
        except Exception:
            best_qed = None
    if ev.valid and (best_qed is None or ev.qed > best_qed):
        best_qed = ev.qed

    best_so_far = SimpleNamespace(valid=(best_qed is not None), qed=best_qed) if best_qed is not None else None
    feedback = generate_feedback(ev, start_ev, constraints, best_so_far=best_so_far)

    if args.state:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.state, {"best_qed_so_far": best_qed})

    write_json(args.output, feedback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

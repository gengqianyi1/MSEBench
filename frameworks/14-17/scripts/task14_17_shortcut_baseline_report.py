#!/usr/bin/env python3
"""Tasks 14-17: what do the naive strategies score? (difficulty evidence, no answers leaked)

A round-5 reviewer of task 16 put it exactly right: `Oracle = 1.0` proves the top of the ladder is
REACHABLE, it does not prove a real model will not saturate the task. The question they asked --
what does a default design, a random legal design, or a greedy one-slot / two-slot search get? --
was answered at instance-selection time, but the answer lived only inside `private_data/`, where a
reviewer cannot look. So every review had to take "this task is hard" on trust.

This script reads those pre-computed ceilings out of each task's private enumeration and prints
them. It deliberately prints ONLY aggregate ceilings, counts and score histograms -- never a
design, a SMILES, a slot assignment or the reference value -- so its output can be pasted into a
review packet. `--json-out` writes the same content for the record.

The instance-selection gates it reports:

  G1  the best score reachable by exhaustively trying every one-slot change, then every two-slot
      change (and for task 16 every three-slot change). This is the "shortcut" ceiling: a model
      that does no real search but tinkers one knob at a time cannot beat it.
  G2  the best score among designs that clear the similarity / overlap floor with margin, i.e.
      solutions that are comfortably feasible rather than balanced on the constraint edge.
  G3  how many feasible improving designs exist at all -- the thickness of the scoring surface.
      A thin surface is how a task ends up scoring every model zero.

    python scripts/task14_17_shortcut_baseline_report.py
    python scripts/task14_17_shortcut_baseline_report.py --json-out baselines.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "task-14-multiobjective-mol-design":
        ("private_data/grader/reference/reference_target.yaml", "yaml"),
    "task-15-lipophilic-sulfonamide-edit-budget":
        ("private_data/grader/reference/reference_target.yaml", "yaml"),
    "task-16-tablet-microenvironment-cqa":
        ("private_data/answers/reference_enumeration.json", "json"),
    "task-17-spherical-agglomeration":
        ("private_data/answers/reference_enumeration.json", "json"),
}
# Only these keys are ever read out. Anything describing a design is excluded by construction
# rather than by filtering, so this script cannot leak a candidate even if the schema grows.
SAFE = (
    "gate_g1_one_slot_ceiling", "gate_g1_thick_tinker_ceiling", "gate_g2_safe_similarity_ceiling",
    "gate_g2_safe_margin", "gate_g2_n_safe_designs", "gate_g3_n_feasible", "score_distribution",
    "p_graded_p1_windows_fraction", "p_graded_p2_n_comfortable",
    "n_feasible_improving_one_slot_edits", "best_ordinary_move_score",
    "gate_g4_checklist_only_expected_score", "gate_g4_checklist_only_p_at_least_50",
    "G1_one_slot_points", "G1_two_slot_points", "G1_three_slot_points", "G2_points", "G2_margin",
)


def _load(path: Path, kind: str):
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if kind == "json" else yaml.safe_load(text)


def _scalars(node, prefix="", out=None):
    """Collect SAFE keys whose value is a scalar or a flat score histogram."""
    out = {} if out is None else out
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SAFE:
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    out[key] = value
                elif isinstance(value, dict) and all(
                        isinstance(v, (int, float)) for v in value.values()):
                    out[key] = value
            elif key in ("gates", "g1_one_slot", "g1_two_slot", "calibration"):
                _scalars(value, prefix + key + ".", out)
            elif isinstance(value, dict) and key not in ("reference", "reference_best", "design",
                                                         "G2_best", "start", "neighbourhood"):
                _scalars(value, prefix + key + ".", out)
    return out


def report(task: str) -> dict:
    rel, kind = SOURCES[task]
    path = ROOT / "tasks" / task / "tests" / "task_snapshot" / rel
    if not path.is_file():
        return {"task": task, "error": f"missing {rel}"}
    found = _scalars(_load(path, kind))
    # task 17 nests its one-slot ceiling one level down under a design-bearing object
    if task == "task-17-spherical-agglomeration":
        doc = _load(path, kind)
        g1 = (doc.get("gates") or {}).get("g1_one_slot") or {}
        for src, dst in (("best_points", "G1_one_slot_points"),
                         ("n_feasible_and_in_budget", "gate_g3_n_feasible")):
            if isinstance(g1.get(src), (int, float)):
                found.setdefault(dst, g1[src])
    return {"task": task, "baselines": found}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out")
    args = ap.parse_args()
    reports = [report(t) for t in SOURCES]
    for r in reports:
        print(f"=== {r['task']}")
        if r.get("error"):
            print(f"    ! {r['error']}"); continue
        for k, v in sorted(r["baselines"].items()):
            print(f"    {k:<44} {v}")
    print("\nRead as: the G1 rows are the ceiling for a model that only tinkers one or two slots at")
    print("a time; the Oracle and the reference optimum are far above them. G3 is how many feasible")
    print("improving designs exist -- a thin surface is how a task scores every model zero.")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

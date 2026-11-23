"""Objective selection helpers, shared by the public harness and the isolated grader.

These three functions are on the PUBLIC side on purpose: the run harness needs "the best QED you have
reached so far" to build its round feedback, and that needs no knowledge of the reference target. The
reference value, the achievement ratio and the score bands live only in grader/, so nothing here can
be used to work out what the grader is comparing against.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluator import MoleculeEvaluation

# Map the objective name used in task_config.yaml onto the MoleculeEvaluation field.
_OBJECTIVE_FIELDS = {
    "qed": "qed",
    "sa_score": "sa_score",
    "tpsa": "tpsa",
    "logp": "logp",
    "molecular_weight": "molecular_weight",
}


def objective_field(target_cfg: Dict[str, Any]) -> str:
    name = str(target_cfg["name"]).strip().lower()
    if name not in _OBJECTIVE_FIELDS:
        raise ValueError(
            f"Unsupported optimization target {name!r}; expected one of {sorted(_OBJECTIVE_FIELDS)}"
        )
    return _OBJECTIVE_FIELDS[name]


def objective_value(ev: MoleculeEvaluation, target_cfg: Dict[str, Any]) -> Optional[float]:
    return getattr(ev, objective_field(target_cfg), None)


def select_best_by_objective(
    evaluations: List[MoleculeEvaluation],
    target_cfg: Dict[str, Any],
) -> Optional[MoleculeEvaluation]:
    """Best-by-objective molecule regardless of feasibility (reported for diagnostics only)."""
    direction = target_cfg["direction"]
    scored = [ev for ev in evaluations if ev.valid and objective_value(ev, target_cfg) is not None]
    if not scored:
        return None
    key = lambda ev: objective_value(ev, target_cfg)
    return max(scored, key=key) if direction == "maximize" else min(scored, key=key)

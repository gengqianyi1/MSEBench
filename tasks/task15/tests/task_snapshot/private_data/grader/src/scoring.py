"""Scoring: pure optimality, with hard constraints acting as a gate rather than as credit.

Rubric rationale (this differs deliberately from the earlier problems in this series, and the
change is the point):

An earlier version of this scorer awarded 25 of 100 points for per-constraint coverage. That
contradicted the task's own framing -- the question states that staying inside the constraints is
the entry requirement, not the achievement -- and it lifted every competent model by a flat 25
points before any design work was measured. Worse, a design that violates a hard constraint is not
a partial success; it is not a usable molecule at all, so there is nothing to award points for.

So constraint satisfaction is now a GATE: an infeasible design contributes nothing, a feasible one
is scored purely on how far it moved the objective. The per-constraint flags are still computed and
still reported, as diagnostics, but they carry zero points. This is both stricter and more honest
than splitting the total, and it makes the score mean one thing only: how good is the best usable
molecule you produced.

The module is objective-agnostic: `optimization_target.name` in the task config selects which
MoleculeEvaluation field is optimised and `direction` selects the sense.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluator import MoleculeEvaluation, check_constraints, satisfies_all_constraints

# objective_field / objective_value / select_best_by_objective live on the public side: the run
# harness needs "best so far" for its feedback and none of them touch the reference target. Only the
# reference-dependent arithmetic (achievement and the score bands) is here in grader/.
from objective import objective_field, objective_value, select_best_by_objective  # noqa: F401


def _is_improvement(value: float, start_value: float, direction: str) -> bool:
    return value > start_value if direction == "maximize" else value < start_value


def constraint_diagnostics(
    evaluation: MoleculeEvaluation,
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-constraint flags for the report. Carries NO points -- see the module docstring."""
    flags = check_constraints(evaluation, constraints)
    return {
        "constraint_flags": flags,
        "n_constraints_satisfied": sum(1 for v in flags.values() if v),
        "n_constraints_total": len(flags),
        "constraints_violated": [k for k, v in flags.items() if not v],
        "constraint_points": 0,
        "note": "Constraint satisfaction is a feasibility gate and earns no points on its own.",
    }


def get_successful_optimized_analogs(
    start_eval: MoleculeEvaluation,
    evaluations: List[MoleculeEvaluation],
    constraints: Dict[str, Any],
    target_cfg: Dict[str, Any],
) -> List[MoleculeEvaluation]:
    """Candidates that pass every hard constraint AND improve the objective over the start."""
    direction = target_cfg["direction"]
    start_value = objective_value(start_eval, target_cfg)
    if start_value is None:
        return []
    out = []
    for ev in evaluations:
        value = objective_value(ev, target_cfg)
        if not ev.valid or value is None:
            continue
        if not _is_improvement(value, start_value, direction):
            continue
        if satisfies_all_constraints(ev, constraints):
            out.append(ev)
    return out


def select_best_successful(
    start_eval: MoleculeEvaluation,
    evaluations: List[MoleculeEvaluation],
    constraints: Dict[str, Any],
    target_cfg: Dict[str, Any],
) -> Optional[MoleculeEvaluation]:
    successful = get_successful_optimized_analogs(start_eval, evaluations, constraints, target_cfg)
    if not successful:
        return None
    key = lambda ev: objective_value(ev, target_cfg)
    if target_cfg["direction"] == "maximize":
        return max(successful, key=key)
    return min(successful, key=key)


def optimality_score(
    best_successful: Optional[MoleculeEvaluation],
    start_eval: MoleculeEvaluation,
    reference_cfg: Dict[str, Any],
    scoring_cfg: Dict[str, Any],
    target_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Score the best FEASIBLE design on its distance to the reference target.

        achievement = (best_feasible_objective - start_objective)
                      / (reference_objective   - start_objective)

    The ratio is signed consistently for both directions, so a minimisation target needs no special
    case. Achievement is clipped to [0, 1]: a design that beats the reference earns full marks
    rather than being penalised, because the reference is the best value found by the shipped
    enumeration, not a proof of global optimality.
    """
    max_points = int(scoring_cfg["optimality_points"])
    bands = scoring_cfg["optimality_bands"]
    ref_best = float(reference_cfg["best_objective_in_window"])
    start_value = objective_value(start_eval, target_cfg)

    if best_successful is None or start_value is None:
        return {
            "optimality_score": 0,
            "optimality_max_points": max_points,
            "achievement_fraction": 0.0,
            "best_successful_objective": None,
            "start_objective": start_value,
            "reference_objective": ref_best,
            "note": "No design satisfied every hard constraint, so no credit.",
        }

    best_value = objective_value(best_successful, target_cfg)
    denom = ref_best - start_value
    achievement = 1.0 if denom == 0 else (best_value - start_value) / denom
    achievement = max(0.0, min(1.0, achievement))

    points = 0
    for band in bands:
        if achievement >= float(band["min_achievement"]):
            points = int(band["points"])
            break

    return {
        "optimality_score": points,
        "optimality_max_points": max_points,
        "achievement_fraction": round(achievement, 6),
        "best_successful_objective": best_value,
        "start_objective": start_value,
        "reference_objective": ref_best,
        "gap_to_reference": round(abs(ref_best - best_value), 6),
    }


def final_score(
    start_eval: MoleculeEvaluation,
    round_evaluations: List[MoleculeEvaluation],
    constraints: Dict[str, Any],
    scoring_cfg: Dict[str, Any],
    reference_cfg: Dict[str, Any],
    target_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Total = optimality points only. Feasibility is a gate; see the module docstring."""
    best_by_objective = select_best_by_objective(round_evaluations, target_cfg)
    successful = get_successful_optimized_analogs(
        start_eval, round_evaluations, constraints, target_cfg
    )
    best_successful = select_best_successful(start_eval, round_evaluations, constraints, target_cfg)

    diagnostics = constraint_diagnostics(
        best_by_objective if best_by_objective is not None
        else MoleculeEvaluation(smiles="", valid=False),
        constraints,
    )
    oscore = optimality_score(
        best_successful=best_successful,
        start_eval=start_eval,
        reference_cfg=reference_cfg,
        scoring_cfg=scoring_cfg,
        target_cfg=target_cfg,
    )

    return {
        "total_score": oscore["optimality_score"],
        "optimality_score": oscore["optimality_score"],
        "optimality_score_details": oscore,
        "constraint_points": 0,
        "constraint_diagnostics_for_best_by_objective": diagnostics,
        "best_by_objective_molecule": best_by_objective.to_dict() if best_by_objective else None,
        "best_successful_molecule": best_successful.to_dict() if best_successful else None,
        "num_successful_optimized_analogs": len(successful),
        "success": len(successful) >= 1,
        "objective_name": target_cfg["name"],
        "objective_direction": target_cfg["direction"],
        "start_objective": objective_value(start_eval, target_cfg),
        "best_successful_objective": (
            objective_value(best_successful, target_cfg) if best_successful else None
        ),
    }

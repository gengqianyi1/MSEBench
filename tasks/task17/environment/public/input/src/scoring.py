"""Scoring: pure optimality. Feasibility is a 0-point gate, never partial credit.

Rubric rationale (carried over from the earlier problems in this series, where it was the change that
made the difficulty calibration honest):

An earlier scorer in this series awarded a slice of the total for per-constraint coverage. That is
wrong twice over. First it contradicts the task's own framing: staying inside the controlled-change
budget and the process safety limits is the entry requirement, not the achievement. Second, a batch
that violates the boiling-point margin, will not stay suspended, or spends more change points than
the change control allows is not a partial success -- it cannot be run or filed at all, so there is
nothing to pay for. Worst of all, per-constraint credit lifts every competent model by a flat amount
before any design work is measured, which makes a difficulty target arithmetically unreachable.

So here:
  * constraint_points is 0, and this module raises if the config says otherwise;
  * an infeasible or over-budget process contributes nothing at all;
  * all 100 points come from how far the best FEASIBLE process moved the desirability D toward the
    reference optimum found by scripts/enumerate_reference_best.py.

What counts as feasible is the union of two things: the process constraints C3-C7 (measured in
src/agglomeration.constraints) and the controlled-change budget C2 (declared in task_config.yaml as
constraints.change_budget and flagged in src/evaluator.check_constraints). Both are gates. A design
that is over budget is still measured and still reported -- it just cannot score.

Unlike some earlier problems in this series, D itself is never 0: the specification windows are
soft-edged, so a process whose four responses are merely bad has a small positive D. That is
deliberate. It keeps the scoring surface a slope rather than a wall, so partial progress is visible
in the score and a 0 means "infeasible", not "not perfect".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluator import BatchEvaluation, check_constraints, satisfies_all_constraints


def objective_value(ev: Optional[BatchEvaluation]) -> Optional[float]:
    """The objective is always the desirability D. Kept as a function so the rest of the module never
    reaches into the evaluation object directly."""
    if ev is None or not ev.valid:
        return None
    return ev.desirability


def constraint_diagnostics(evaluation: Optional[BatchEvaluation],
                           constraints_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Per-constraint flags for the report. Carries NO points -- see the module docstring."""
    ev = evaluation if evaluation is not None else BatchEvaluation()
    flags = check_constraints(ev, constraints_cfg)
    return {
        "constraint_flags": flags,
        "n_constraints_satisfied": sum(1 for v in flags.values() if v),
        "n_constraints_total": len(flags),
        "constraints_violated": [k for k, v in flags.items() if not v],
        "constraint_points": 0,
        "note": "Constraint satisfaction is a feasibility gate and earns no points on its own.",
    }


def select_best_by_objective(evaluations: List[BatchEvaluation]) -> Optional[BatchEvaluation]:
    """Best D regardless of feasibility (diagnostics only -- never scored)."""
    scored = [ev for ev in evaluations if objective_value(ev) is not None]
    if not scored:
        return None
    return max(scored, key=lambda ev: objective_value(ev))


def get_successful_designs(start_eval: BatchEvaluation,
                           evaluations: List[BatchEvaluation],
                           constraints_cfg: Dict[str, Any]) -> List[BatchEvaluation]:
    """Submissions that satisfy every hard constraint AND improve D over the registered process."""
    start_value = objective_value(start_eval)
    if start_value is None:
        return []
    out = []
    for ev in evaluations:
        value = objective_value(ev)
        if value is None or value <= start_value:
            continue
        if satisfies_all_constraints(ev, constraints_cfg):
            out.append(ev)
    return out


def select_best_successful(start_eval: BatchEvaluation,
                           evaluations: List[BatchEvaluation],
                           constraints_cfg: Dict[str, Any]) -> Optional[BatchEvaluation]:
    successful = get_successful_designs(start_eval, evaluations, constraints_cfg)
    if not successful:
        return None
    return max(successful, key=lambda ev: objective_value(ev))


def optimality_score(best_successful: Optional[BatchEvaluation],
                     start_eval: BatchEvaluation,
                     reference_cfg: Dict[str, Any],
                     scoring_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Score the best FEASIBLE process by its progress toward the reference optimum.

        achievement = (D_best_feasible - D_registered) / (D_reference - D_registered)

    Clipped to [0, 1]: the reference is the exact optimum of the exhaustively enumerated
    change-budget neighbourhood, not a proof of global optimality over the 6.5e8-design space, so a
    submission that meets or beats it earns full marks rather than being penalised.
    """
    max_points = int(scoring_cfg["optimality_points"])
    bands = scoring_cfg["optimality_bands"]
    ref_best = float(reference_cfg["best_objective_in_budget"])
    start_value = objective_value(start_eval)

    if best_successful is None or start_value is None:
        return {
            "optimality_score": 0,
            "optimality_max_points": max_points,
            "achievement_fraction": 0.0,
            "best_successful_objective": None,
            "start_objective": start_value,
            "reference_objective": ref_best,
            "note": ("No submission both satisfied every hard constraint and improved the "
                     "desirability, so no credit."),
        }

    best_value = objective_value(best_successful)
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
        "gap_to_reference": round(abs(ref_best - best_value), 9),
    }


def continuous_metrics(start_eval: BatchEvaluation,
                       round_evaluations: List[BatchEvaluation],
                       constraints_cfg: Dict[str, Any],
                       reference_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Continuous quantities that separate two runs even when they land in the same band.

    The banded score is deliberately coarse, and external review pointed out that three models sitting
    in one band tells a reviewer nothing. These fields are reported alongside the score so a
    comparison never has to rest on the band alone: the UNCLIPPED achievement, the per-round
    desirability trajectory, the per-response desirability breakdown of the best usable batch, how
    many rounds produced a runnable / in-budget / improving batch, and which round was best.

    None of it changes the score. `total_score` is still the banded optimality score and nothing else.
    """
    start_value = objective_value(start_eval)
    ref_best = float(reference_cfg["best_objective_in_budget"])
    denom = ref_best - (start_value if start_value is not None else 0.0)

    per_round: List[Dict[str, Any]] = []
    for i, ev in enumerate(round_evaluations, start=1):
        value = objective_value(ev)
        feasible = ev.valid and satisfies_all_constraints(ev, constraints_cfg)
        improving = bool(value is not None and start_value is not None and value > start_value)
        per_round.append({
            "round": i,
            "design_space_member": ev.design is not None,
            "runnable": bool(ev.valid),
            "within_all_constraints": bool(feasible),
            "improves_on_registered_process": improving,
            "usable": bool(feasible and improving),
            "desirability_D": value,
            "raw_achievement": (round((value - start_value) / denom, 6)
                                if value is not None and start_value is not None and denom > 0
                                else None),
            "change_points_spent": ev.edit_cost,
            "n_variables_changed": ev.n_slots_changed,
            "variables_changed": list(ev.changed_slots or []),
        })

    usable = [r for r in per_round if r["usable"]]
    best_round = max(usable, key=lambda r: r["desirability_D"])["round"] if usable else None
    best_successful = select_best_successful(start_eval, round_evaluations, constraints_cfg)
    best_raw = None
    if best_successful is not None and start_value is not None and denom > 0:
        best_raw = round((objective_value(best_successful) - start_value) / denom, 6)

    return {
        "note": "Continuous diagnostics for cross-model comparison. They do NOT contribute to "
                "total_score; the score is the banded optimality score only.",
        # unclipped, so a run that beats the reference is visibly above 1.0 instead of saturating
        "raw_achievement_unclipped": best_raw,
        "best_usable_desirability_D": objective_value(best_successful) if best_successful else None,
        "best_usable_desirability_per_response":
            best_successful.desirability_per_response if best_successful else None,
        "best_usable_responses": best_successful.responses if best_successful else None,
        "best_round": best_round,
        "n_rounds": len(round_evaluations),
        "n_rounds_design_space_member": sum(1 for r in per_round if r["design_space_member"]),
        "n_rounds_runnable": sum(1 for r in per_round if r["runnable"]),
        "n_rounds_within_all_constraints": sum(1 for r in per_round if r["within_all_constraints"]),
        "n_rounds_improving": sum(1 for r in per_round if r["improves_on_registered_process"]),
        "n_rounds_usable": len(usable),
        "desirability_trajectory": [r["desirability_D"] for r in per_round],
        "registered_process_desirability_D": start_value,
        "registered_process_desirability_per_response": start_eval.desirability_per_response,
        "per_round": per_round,
    }


def final_score(start_eval: BatchEvaluation,
                round_evaluations: List[BatchEvaluation],
                constraints_cfg: Dict[str, Any],
                scoring_cfg: Dict[str, Any],
                reference_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Total = optimality points only. Feasibility is a gate; see the module docstring."""
    if int(scoring_cfg.get("constraint_points", 0)) != 0:
        raise ValueError("constraint_points must be 0: feasibility is a gate, not credit")

    best_by_objective = select_best_by_objective(round_evaluations)
    successful = get_successful_designs(start_eval, round_evaluations, constraints_cfg)
    best_successful = select_best_successful(start_eval, round_evaluations, constraints_cfg)

    diagnostics = constraint_diagnostics(best_by_objective, constraints_cfg)
    oscore = optimality_score(best_successful=best_successful, start_eval=start_eval,
                              reference_cfg=reference_cfg, scoring_cfg=scoring_cfg)

    return {
        "continuous_metrics": continuous_metrics(start_eval, round_evaluations, constraints_cfg,
                                                 reference_cfg),
        "total_score": oscore["optimality_score"],
        "optimality_score": oscore["optimality_score"],
        "optimality_score_details": oscore,
        "constraint_points": 0,
        "constraint_diagnostics_for_best_by_objective": diagnostics,
        "best_by_objective_design": best_by_objective.to_dict() if best_by_objective else None,
        "best_successful_design": best_successful.to_dict() if best_successful else None,
        "num_successful_designs": len(successful),
        "success": len(successful) >= 1,
        "objective_name": "derringer_suich_desirability_D",
        "objective_direction": "maximize",
        "start_objective": objective_value(start_eval),
        "best_successful_objective": objective_value(best_successful) if best_successful else None,
    }

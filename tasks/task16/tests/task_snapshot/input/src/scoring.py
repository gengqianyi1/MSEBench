"""Scoring: compliance-gated optimality. Feasibility and CQA compliance are 0-point gates.

RULER VERSION 2.0.0 -- see CHANGELOG.md. Audit finding #3 (2026-07-29) is the reason this module
changed, and it changed the score semantics, so every recorded trajectory had to be re-scored.

What the auditor found in 1.x
-----------------------------
"A feasible formulation could be recorded as a benchmark SUCCESS even with a CQA outside its
specification window, because its aggregate desirability exceeded the starting formulation's."
That is a real defect for a pharmaceutical task: a tablet that misses a release specification
cannot be shipped, so it is not a partial success, it is a failure that happens to look good on an
aggregate.

What 2.0.0 does
---------------
Three nested gates, in this order. A submission must clear ALL THREE to be scored at all:

  1. design-space membership      on-grid, acid-consistent, mass closes
  2. PACMP change budget          <= 4 slots, <= 1 identity change, composition overlap >= 0.88
  3. CQA COMPLIANCE (new)         all five CQAs strictly inside their printed [L, U] windows

and then, to count as a success at all, it must also improve on the starting formulation:

  4. D > D_start

Nothing that fails any gate earns a point, and `success` is False unless at least one submission
cleared all four. The smooth desirability D survives as the OPTIMISATION SIGNAL -- it is what
ranks the compliant designs against each other, it is what is fed back per round, and it is
strictly positive outside a window so the feedback surface has no cliff -- but it can no longer buy
a success verdict for a non-releasable formulation. `release_diagnostics()` reports the
best-by-desirability submission separately so a reviewer can see exactly what compliance cost.

Feasibility still carries no points
-----------------------------------
`constraint_points` is 0 and src/scoring.py raises if the config says otherwise. Awarding a slice
of the total for per-constraint coverage lifts every competent model by a flat amount before any
design work is measured, which makes a difficulty target arithmetically unreachable -- and a
formulation that cannot be released has nothing to pay for. All 100 points come from how far the
best QUALIFYING design moved D toward the reference optimum, which is itself a fully compliant
design (scripts/enumerate_reference_best.py asserts that).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluator import FormulationEvaluation, check_constraints, satisfies_all_constraints


def objective_value(ev: Optional[FormulationEvaluation]) -> Optional[float]:
    """The objective is always the desirability D. Kept as a function so the rest of the module
    never reaches into the evaluation object directly."""
    if ev is None or not ev.valid:
        return None
    return ev.desirability


def constraint_diagnostics(evaluation: Optional[FormulationEvaluation],
                           constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Per-constraint flags for the report. Carries NO points -- see the module docstring."""
    ev = evaluation if evaluation is not None else FormulationEvaluation()
    flags = check_constraints(ev, constraints)
    return {
        "constraint_flags": flags,
        "n_constraints_satisfied": sum(1 for v in flags.values() if v),
        "n_constraints_total": len(flags),
        "constraints_violated": [k for k, v in flags.items() if not v],
        "constraint_points": 0,
        "note": "Constraint satisfaction is a feasibility gate and earns no points on its own.",
    }


def select_best_by_objective(evaluations: List[FormulationEvaluation]) -> Optional[FormulationEvaluation]:
    """Best D regardless of feasibility (diagnostics only -- never scored)."""
    scored = [ev for ev in evaluations if objective_value(ev) is not None]
    if not scored:
        return None
    return max(scored, key=lambda ev: objective_value(ev))


def is_cqa_compliant(ev: Optional[FormulationEvaluation]) -> bool:
    """THE release gate (audit finding #3): every one of the five CQAs strictly inside its printed
    [L, U] window. A formulation that fails this cannot be released, so it can never be a benchmark
    success and can never earn a point, however high its aggregate desirability is.

    `all_cqa_in_spec` is computed in evaluator.evaluate_design as `(y > L) & (y < U)` on all five,
    from the same window numbers the scorer uses and question.md prints."""
    return bool(ev is not None and ev.valid and ev.all_cqa_in_spec)


def is_qualifying(ev: Optional[FormulationEvaluation], start_value: Optional[float],
                  constraints: Dict[str, Any]) -> bool:
    """All four gates: design-space member, inside the PACMP change budget, CQA-compliant, and
    strictly better than the starting formulation. The scalar twin of
    design_space.qualifying_mask."""
    if ev is None or start_value is None:
        return False
    value = objective_value(ev)
    if value is None or value <= start_value:
        return False
    if not is_cqa_compliant(ev):
        return False
    return satisfies_all_constraints(ev, constraints)


def get_successful_designs(start_eval: FormulationEvaluation,
                           evaluations: List[FormulationEvaluation],
                           constraints: Dict[str, Any]) -> List[FormulationEvaluation]:
    """Submissions that clear every gate: hard constraints, CQA compliance, and an improvement in D
    over the starting formulation. This list is what `success` and the score are built from."""
    start_value = objective_value(start_eval)
    if start_value is None:
        return []
    return [ev for ev in evaluations if is_qualifying(ev, start_value, constraints)]


def get_budget_feasible_designs(evaluations: List[FormulationEvaluation],
                                constraints: Dict[str, Any]) -> List[FormulationEvaluation]:
    """Diagnostics only: submissions inside the change budget, compliant or not. Reported so a
    reviewer can see how much of a run was lost to non-compliance rather than to the budget."""
    return [ev for ev in evaluations if ev.valid and satisfies_all_constraints(ev, constraints)]


def select_best_successful(start_eval: FormulationEvaluation,
                           evaluations: List[FormulationEvaluation],
                           constraints: Dict[str, Any]) -> Optional[FormulationEvaluation]:
    successful = get_successful_designs(start_eval, evaluations, constraints)
    if not successful:
        return None
    return max(successful, key=lambda ev: objective_value(ev))


def optimality_score(best_successful: Optional[FormulationEvaluation],
                     start_eval: FormulationEvaluation,
                     reference_cfg: Dict[str, Any],
                     scoring_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Score the best QUALIFYING design by its progress toward the reference optimum.

        achievement = (D_best_qualifying - D_baseline) / (D_reference - D_baseline)

    "Qualifying" is the four-gate set of `is_qualifying`: on-grid, inside the PACMP change budget,
    ALL FIVE CQAs inside their windows, and D above the starting formulation. A submission that
    misses one specification window is not eligible here at all, so it cannot be scored and cannot
    make `success` true (audit finding #3).

    Clipped to [0, 1]: the reference is the best design found by the shipped exhaustive enumeration
    of the change-budget neighbourhood, not a proof of global optimality, so a submission that beats
    it earns full marks rather than being penalised.
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
            "note": ("No submission cleared all four gates (design-space member, inside the PACMP "
                     "change budget, all five CQAs inside their specification windows, and D above "
                     "the starting formulation), so no credit."),
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


def continuous_diagnostics(start_eval: FormulationEvaluation,
                           round_evaluations: List[FormulationEvaluation],
                           constraints: Dict[str, Any],
                           reference_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Continuous, band-free quantities describing a run.

    Audit finding #4: two of three calibration models sat in the same band, so the total score alone
    carried almost no information about them. These numbers are NOT part of the score -- the score is
    the band table, deterministically -- but they let a reviewer rank two runs that scored the same,
    and they make the difference between "nearly compliant" and "nowhere near" visible.

    Everything here is derived from the recorded evaluations, so `scripts/rescore_run.py` reproduces
    it exactly without calling a model.
    """
    start_value = objective_value(start_eval)
    ref_best = float(reference_cfg["best_objective_in_budget"]) if reference_cfg.get(
        "best_objective_in_budget") is not None else None

    def ach(v: Optional[float]) -> Optional[float]:
        if v is None or start_value is None or ref_best is None:
            return None
        denom = ref_best - start_value
        if denom == 0:
            return 1.0
        return round((v - start_value) / denom, 9)          # NOT clipped: the raw ratio

    per_round = []
    for i, ev in enumerate(round_evaluations, 1):
        per_round.append({
            "round": i,
            "in_design_space": ev.design is not None,
            "measured": bool(ev.valid),
            "budget_feasible": bool(ev.valid and satisfies_all_constraints(ev, constraints)),
            "n_cqa_in_spec": ev.n_cqa_in_spec,
            "cqa_compliant": is_cqa_compliant(ev),
            "qualifying": is_qualifying(ev, start_value, constraints),
            "desirability": ev.desirability,
            "desirability_per_cqa": ev.desirability_per_cqa,
            "raw_achievement_unclipped": ach(ev.desirability),
            "composition_overlap": ev.composition_overlap,
            "n_slots_changed": ev.n_slots_changed,
            "n_identity_changes": ev.n_identity_changes,
        })

    qualifying = [r for r in per_round if r["qualifying"]]
    compliant = [r for r in per_round if r["cqa_compliant"]]
    feasible = [r for r in per_round if r["budget_feasible"]]
    measured = [r for r in per_round if r["measured"]]
    best_q = max(qualifying, key=lambda r: r["desirability"]) if qualifying else None
    best_any = max(measured, key=lambda r: r["desirability"]) if measured else None

    return {
        "n_rounds": len(round_evaluations),
        "n_rounds_in_design_space": sum(1 for r in per_round if r["in_design_space"]),
        "n_rounds_measured": len(measured),
        "n_rounds_budget_feasible": len(feasible),
        "n_rounds_cqa_compliant": len(compliant),
        "n_rounds_qualifying": len(qualifying),
        "best_qualifying_round": best_q["round"] if best_q else None,
        "best_qualifying_desirability": best_q["desirability"] if best_q else None,
        "best_qualifying_raw_achievement": best_q["raw_achievement_unclipped"] if best_q else None,
        "best_qualifying_desirability_per_cqa": best_q["desirability_per_cqa"] if best_q else None,
        # The best submission by desirability ALONE, ignoring compliance. If this is higher than the
        # best qualifying one, the run traded a specification breach for aggregate desirability and
        # the compliance gate is what stopped it scoring.
        "best_by_desirability_round": best_any["round"] if best_any else None,
        "best_by_desirability": best_any["desirability"] if best_any else None,
        "best_by_desirability_raw_achievement": (best_any["raw_achievement_unclipped"]
                                                 if best_any else None),
        "best_by_desirability_was_compliant": best_any["cqa_compliant"] if best_any else None,
        "max_n_cqa_in_spec_over_run": max([r["n_cqa_in_spec"] or 0 for r in per_round] or [0]),
        "best_overlap_among_qualifying": (min(r["composition_overlap"] for r in qualifying)
                                          if qualifying else None),
        "per_round": per_round,
    }


def release_diagnostics(start_eval: FormulationEvaluation,
                        round_evaluations: List[FormulationEvaluation],
                        constraints: Dict[str, Any]) -> Dict[str, Any]:
    """What the compliance gate cost this run, in plain terms. Audit finding #3 asked for the
    in-window / out-of-window distinction to be visible in the report, not only in the total."""
    start_value = objective_value(start_eval)
    feasible = get_budget_feasible_designs(round_evaluations, constraints)
    blocked = [ev for ev in feasible
               if not is_cqa_compliant(ev)
               and objective_value(ev) is not None and start_value is not None
               and objective_value(ev) > start_value]
    return {
        "gate_order": ["design_space_member", "pacmp_change_budget", "all_five_cqa_in_spec",
                       "desirability_above_start"],
        "n_budget_feasible_submissions": len(feasible),
        "n_blocked_by_cqa_compliance_only": len(blocked),
        "blocked_by_cqa_compliance": [
            {"desirability": ev.desirability, "n_cqa_in_spec": ev.n_cqa_in_spec,
             "cqa": ev.cqa, "design": ev.design} for ev in blocked],
        "note": ("A submission listed under blocked_by_cqa_compliance improved the aggregate "
                 "desirability and stayed inside the change budget, but at least one CQA fell "
                 "outside its specification window, so it is not releasable and scores nothing. "
                 "Under ruler version 1.x it would have been recorded as a success."),
    }


def final_score(start_eval: FormulationEvaluation,
                round_evaluations: List[FormulationEvaluation],
                constraints: Dict[str, Any],
                scoring_cfg: Dict[str, Any],
                reference_cfg: Dict[str, Any],
                release_gate_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Total = compliance-gated optimality points only. See the module docstring for the gates.

    `release_gate_cfg` is task_config.yaml's `release_gate:` block. It is passed in so that the
    declared switch is LOAD-BEARING rather than decorative: a mutation test found that flipping
    `all_cqa_in_spec_required` to false changed nothing, because the gate was hardcoded here. A
    config that claims the gate is optional now refuses to score, in exactly the same way as a
    config that claims constraint satisfaction earns points.
    """
    if int(scoring_cfg.get("constraint_points", 0)) != 0:
        raise ValueError("constraint_points must be 0: feasibility is a gate, not credit")
    if release_gate_cfg is not None and release_gate_cfg.get("all_cqa_in_spec_required") is not True:
        raise ValueError(
            "release_gate.all_cqa_in_spec_required must be true: CQA compliance is a hard gate in "
            "ruler 2.0.0 (audit finding #3) and this scorer implements it unconditionally, so a "
            "config that declares otherwise would be describing a scorer that does not exist")

    best_by_objective = select_best_by_objective(round_evaluations)
    successful = get_successful_designs(start_eval, round_evaluations, constraints)
    best_successful = select_best_successful(start_eval, round_evaluations, constraints)

    diagnostics = constraint_diagnostics(best_by_objective, constraints)
    oscore = optimality_score(best_successful=best_successful, start_eval=start_eval,
                              reference_cfg=reference_cfg, scoring_cfg=scoring_cfg)

    return {
        "ruler_version": str(scoring_cfg.get("ruler_version", "unversioned")),
        "total_score": oscore["optimality_score"],
        "optimality_score": oscore["optimality_score"],
        "optimality_score_details": oscore,
        "constraint_points": 0,
        "constraint_diagnostics_for_best_by_objective": diagnostics,
        "best_by_objective_design": best_by_objective.to_dict() if best_by_objective else None,
        "best_successful_design": best_successful.to_dict() if best_successful else None,
        "num_successful_designs": len(successful),
        "success": len(successful) >= 1,
        "success_definition": ("at least one submission that is on-grid, inside the PACMP change "
                              "budget, has ALL FIVE CQAs strictly inside their specification "
                              "windows, and has a higher desirability D than the starting "
                              "formulation"),
        "objective_name": "smooth_two_sided_desirability_D",
        "objective_direction": "maximize",
        "start_objective": objective_value(start_eval),
        "best_successful_objective": objective_value(best_successful) if best_successful else None,
        "release_gate": release_diagnostics(start_eval, round_evaluations, constraints),
        "continuous_metrics": continuous_diagnostics(start_eval, round_evaluations, constraints,
                                                    reference_cfg),
    }

"""Reduced-bandwidth evaluator feedback.

What goes back to the model after each round:
  * the five measured CQA values for the formulation it submitted,
  * the desirability D of that formulation,
  * its own change-budget bookkeeping (composition overlap, slots changed, identity changes,
    q.s. filler %) -- all of which the model can and should compute itself from the printed
    composition table, so returning them leaks nothing,
  * the best D it has reached so far in this run,
  * for a submission that is not a member of the design space, the reason.

What deliberately does NOT go back:
  * the reference optimum, its value, or anything derived from it;
  * the specification windows (they are printed in question.md, in full);
  * which CQAs are outside their windows, or a per-CQA desirability breakdown;
  * a pass/fail verdict, a score, or a count of satisfied constraints.

Rationale. The windows are stated in the question, so the model has complete information and can
decide feasibility itself; handing back per-CQA verdicts would turn the benchmark into "search
against a free perfect verifier", which is what made an earlier problem in this series trivial -- the
model repaired one dimension at a time by reading flags back. Returning the five measured values is
the minimum needed for the task to be an optimisation at all: the CQAs are outputs of a fitted model
and cannot be computed by inspection.

The one thing that is returned in full is the list of violated FEASIBILITY constraints (not CQA
windows). Overlap, slot count and mass closure are plain arithmetic on the composition table, so a
model that lost track of its own budget learns nothing it could not recompute -- while a model that
silently drifted out of the budget would otherwise burn every remaining round on designs that can
never be scored.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluator import FormulationEvaluation, check_constraints


def generate_feedback(evaluation: FormulationEvaluation,
                      constraints: Dict[str, Any],
                      best_so_far: Optional[FormulationEvaluation] = None) -> Dict[str, Any]:
    if evaluation.design is None:
        return {
            "status": "not_a_design_space_member",
            "detail": evaluation.error or "The submitted formulation could not be interpreted.",
            "measured_cqa": None,
        }

    budget = {
        "composition_overlap_with_baseline_formulation": evaluation.composition_overlap,
        "n_slots_changed": evaluation.n_slots_changed,
        "slots_changed": evaluation.changed_slots,
        "n_excipient_identity_changes": evaluation.n_identity_changes,
        "qs_filler_pct_of_core": evaluation.qs_filler_pct,
    }

    flags = check_constraints(evaluation, constraints)
    violated = [k for k, v in flags.items() if not v]

    if not evaluation.valid:
        return {
            "status": "not_manufacturable",
            "detail": evaluation.error or "The formulation cannot be made as specified.",
            "measured_cqa": None,
            "budget_bookkeeping": budget,
            "feasibility_constraints_violated": violated,
        }

    feedback: Dict[str, Any] = {
        "status": "measured",
        "measured_cqa": evaluation.cqa,
        "desirability_D": evaluation.desirability,
        "budget_bookkeeping": budget,
        "feasibility_constraints_violated": violated,
    }

    if best_so_far is not None and best_so_far.valid:
        feedback["best_desirability_D_so_far"] = best_so_far.desirability

    return feedback

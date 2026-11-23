"""Per-round evaluator feedback: narrow band, but everything in it is genuinely learnable.

WHAT CHANGED AND WHY
--------------------
The first version returned only the four simulated responses, the desirability D, the change-budget
bookkeeping and the names of violated constraints. External review found the task
under-discriminating (three representative models all landed in the same low band) and asked for more
learnable per-round feedback. This file is the response, and the guiding rule is:

    ADD EVERYTHING THAT IS A DETERMINISTIC FUNCTION OF WHAT THE MODEL ALREADY HAS.
    ADD NOTHING ELSE.

The specification windows are printed in question.md in full, and the four simulated responses are
already returned. So the per-response desirability d_i, the in/out-of-plateau verdict, and the
normalised distance to the plateau are all computable by the model with a calculator and zero extra
information. Withholding them did not protect the difficulty of the problem -- it only taxed
arithmetic, and the shipped trajectories showed models paying that tax instead of doing process
reasoning. Returning them is information-theoretically free and is the single largest learnability
gain available.

RETURNED
  * the four simulated responses of the submitted batch, and its desirability D;
  * per response: the plateau and acceptable limits, whether the value is below / inside / above the
    plateau, the normalised distance t to the plateau (t = 0 on the plateau, t = 1 at the acceptable
    limit), the desirability d_i and the weight w_i -- all recomputable from the returned responses
    and the printed windows;
  * the submission's own change-budget bookkeeping and the list of violated feasibility constraints;
  * a compact history table of every round so far (what moved, cost, the four responses, D), because
    a model that has lost track of its own trajectory wastes rounds re-testing designs;
  * how many measurements remain;
  * the best D reached so far.

STILL WITHHELD, and for the same reason as before -- these are NOT derivable, so returning them would
replace the design problem with a search against a free perfect verifier:
  * the reference optimum, its value, the achievement fraction, the score, or any verdict on whether
    the run is going well;
  * any sensitivity, gradient, ranking or suggestion about which variable to move next;
  * anything about the response mechanism: which variables drive which response, and in which
    direction. The model gets measurements, not the model that produced them.

The known risk, stated plainly. Echoing d_i makes hill-climbing more effective than echoing the raw
responses alone, even though it adds no information. It is bounded by the round budget -- five
measurements over a feasible neighbourhood far larger than five draws can cover, whose optimum needs
one specific coordinated move -- and by the structural difficulty gates, which are properties of the
design space and cannot be moved by any feedback change. (The size of that neighbourhood, the size of
the move and which variables it touches are grader-side; this module ships in the participant
archive.) The empirical check is the shipped calibration: if a graded model ever reaches the plateau
by hill-climbing, the lever is this file, not the windows and not the budget.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from design_grid import RESPONSE_KEYS, WINDOWS
from evaluator import BatchEvaluation, check_constraints


def window_report(responses: Optional[Dict[str, float]],
                  desirability_per_response: Optional[Dict[str, float]]
                  ) -> Optional[Dict[str, Any]]:
    """Per-response window position, normalised distance and desirability.

    Every field is a deterministic function of `responses` (already returned) and WINDOWS (printed in
    question.md), so this leaks nothing. It exists so the model spends its rounds on process
    decisions instead of on re-deriving arithmetic it was already given.
    """
    if not responses:
        return None
    out: Dict[str, Any] = {}
    for k in RESPONSE_KEYS:
        w = WINDOWS[k]
        y = float(responses[k])
        pl, pu, al, au = float(w["pl"]), float(w["pu"]), float(w["al"]), float(w["au"])
        if y < pl:
            position = "below_plateau"
            t = (pl - y) / (pl - al)
            gap = round(pl - y, 6)
        elif y > pu:
            position = "above_plateau"
            t = (y - pu) / (au - pu)
            gap = round(y - pu, 6)
        else:
            position = "inside_plateau"
            t = 0.0
            gap = 0.0
        out[k] = {
            "measured": y,
            "plateau": [pl, pu],
            "acceptable_limits": [al, au],
            "position": position,
            "distance_to_plateau": gap,
            "normalised_distance_t": round(t, 6),
            "desirability_d": (desirability_per_response or {}).get(k),
            "weight_w": float(w["w"]),
        }
    return out


def history_table(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """A compact recap of the rounds already played. Pure restatement of what was already returned."""
    return list(history or [])


def generate_feedback(evaluation: BatchEvaluation,
                      constraints_cfg: Dict[str, Any],
                      best_so_far: Optional[BatchEvaluation] = None,
                      round_idx: Optional[int] = None,
                      max_rounds: Optional[int] = None,
                      history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    progress: Dict[str, Any] = {}
    if round_idx is not None and max_rounds is not None:
        progress = {
            "round": round_idx,
            "rounds_total": max_rounds,
            "measurements_remaining_after_this_round": max(0, max_rounds - round_idx),
        }

    if evaluation.design is None:
        out: Dict[str, Any] = {
            "status": "not_a_design_space_member",
            "detail": evaluation.error or "The submitted process could not be interpreted.",
            "measured_responses": None,
            "hint": "call the check_process_legality tool before submitting to see which variable "
                    "is off-grid; a non-member round returns no measurement at all.",
        }
        out.update(progress)
        if history:
            out["history"] = history_table(history)
        return out

    budget = {
        "change_points_spent": evaluation.edit_cost,
        "change_points_allowed": evaluation.edit_cost_budget,
        "surviving_overlap_with_registered_process": evaluation.change_overlap,
        "n_variables_changed": evaluation.n_slots_changed,
        "variables_changed": evaluation.changed_slots,
        "total_solvent_L_per_kg": evaluation.total_solvent_L_per_kg,
        "total_process_hours": evaluation.total_process_hours,
    }

    flags = check_constraints(evaluation, constraints_cfg)
    violated = [k for k, v in flags.items() if not v]

    if not evaluation.valid:
        out = {
            "status": "batch_cannot_be_run",
            "detail": evaluation.error or "The process as specified cannot be run.",
            "measured_responses": None,
            "budget_bookkeeping": budget,
            "feasibility_constraints_violated": violated,
        }
        out.update(progress)
        if history:
            out["history"] = history_table(history)
        return out

    feedback: Dict[str, Any] = {
        "status": "measured",
        "measured_responses": evaluation.responses,
        "desirability_D": evaluation.desirability,
        # Derivable from measured_responses + the windows printed in the task description. Returned so
        # a round is spent on process judgement rather than on arithmetic.
        "window_report": window_report(evaluation.responses, evaluation.desirability_per_response),
        "budget_bookkeeping": budget,
        "feasibility_constraints_violated": violated,
    }
    feedback.update(progress)

    if best_so_far is not None and best_so_far.valid:
        feedback["best_desirability_D_so_far"] = best_so_far.desirability

    if history:
        feedback["history"] = history_table(history)

    feedback["reminder"] = (
        "D is the weighted geometric mean of the four d values above; it is maximised only when all "
        "four responses sit inside their plateaus at once. Nothing here tells you which variable to "
        "move next, and no reference value, achievement or score is disclosed at any point."
    )
    return feedback

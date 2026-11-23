"""Evaluator: turn a submitted process into (a) constraint flags and (b) the objective value.

The objective is the four-window Derringer-Suich desirability D (src/design_space.py) over the four
responses measured by the shipped process model (src/agglomeration.py). Feasibility is computed here
but earns no points -- see src/scoring.py for why.

Two deliberate asymmetries, both of which exist so that a wasted round is the model's decision and
not the harness's:

  * A submission that is not a member of the design space (missing variable, off-grid value, unknown
    variable) is `valid=False` and gets NO responses at all. There is no such batch to measure.
  * A submission that is on-grid but blows the controlled-change budget IS measured. The budget is a
    scoring gate applied by src/scoring.py, not a refusal to run the analysis, so the diagnostics
    stay informative while the design still earns zero.

Constraint keys absent from task_config.yaml are simply not checked, so an instance can declare only
the constraints its mechanism needs instead of carrying inert placeholder windows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from design_grid import (BUDGET, EDIT_COST, LEVELS, RESPONSE_KEYS, SLOT_UNITS, SLOTS, START,
                         TOTAL_EDIT_COST, DesignError, change_overlap, changed_slots, constraints,
                         edit_cost, just_suspended_tip_speed, normalise_design, total_hours,
                         total_solvent_L_per_kg)
from design_space import Desirability, desirability_from_config
# The ONE import of the synthetic response mechanism. src/response_model.py is the seam: the public
# package ships a stub build of it and no mechanism at all. Nothing else in the graded path reaches
# into src/agglomeration.py.
from response_model import responses

# Diagnostics kept out of the graded numbers but useful in the report: the raw population-balance
# outputs that explain a measured response.
_DIAGNOSTIC_KEYS = ["span", "fines", "coarse", "d50_raw", "s_sat", "eps_wet", "wet_strength",
                    "bypass", "nsteps", "loss_diss", "loss_cls", "loss_att"]


@dataclass
class BatchEvaluation:
    """One graded submission. `desirability` is the objective; everything else is diagnostics."""
    submitted: Optional[Dict[str, Any]] = None
    valid: bool = False
    design: Optional[Dict[str, Any]] = None
    edit_cost: Optional[int] = None
    edit_cost_budget: int = BUDGET
    total_edit_cost_if_all_changed: int = TOTAL_EDIT_COST
    change_overlap: Optional[float] = None
    n_slots_changed: Optional[int] = None
    changed_slots: List[str] = field(default_factory=list)
    process_constraint_violations: List[str] = field(default_factory=list)
    total_solvent_L_per_kg: Optional[float] = None
    total_process_hours: Optional[float] = None
    just_suspended_tip_speed_m_s: Optional[float] = None
    responses: Optional[Dict[str, float]] = None
    response_diagnostics: Optional[Dict[str, float]] = None
    desirability_per_response: Optional[Dict[str, float]] = None
    desirability: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def scorer_from_config(objective_cfg: Dict[str, Any]) -> Desirability:
    """Build the desirability scorer from the windows stored in task_config.yaml.

    scripts/check_consistency.py asserts those windows equal agglomeration.WINDOWS, so this cannot
    silently become a second ruler.
    """
    return desirability_from_config(objective_cfg)


def evaluate_design(raw_design: Any, objective_cfg: Dict[str, Any],
                    reference_design: Optional[Dict[str, Any]] = None) -> BatchEvaluation:
    """Evaluate one submitted process."""
    ref = reference_design or START
    ev = BatchEvaluation(submitted=raw_design if isinstance(raw_design, dict) else None)

    try:
        design = normalise_design(raw_design)
    except DesignError as exc:
        ev.error = str(exc)
        return ev

    ev.design = design
    ev.edit_cost = edit_cost(design, ref)
    ev.change_overlap = round(change_overlap(design, ref), 6)
    ev.changed_slots = changed_slots(design, ref)
    ev.n_slots_changed = len(ev.changed_slots)
    ev.total_solvent_L_per_kg = round(total_solvent_L_per_kg(design), 4)
    ev.total_process_hours = round(total_hours(design), 4)
    ev.just_suspended_tip_speed_m_s = round(just_suspended_tip_speed(design), 4)

    viol = constraints(design)
    ev.process_constraint_violations = viol
    if viol:
        # The batch cannot be run at all (boiling-point margin, suspension, solvent, time, scheme
        # self-consistency), so there is nothing to measure.
        ev.error = "process constraints violated: " + "; ".join(viol)
        return ev

    r = responses(design)
    sc = scorer_from_config(objective_cfg)
    ev.valid = True
    ev.responses = {k: round(float(r[k]), 6) for k in RESPONSE_KEYS}
    ev.response_diagnostics = {k: round(float(r[k]), 6) for k in _DIAGNOSTIC_KEYS if k in r}
    ev.desirability_per_response = {k: round(float(v), 6) for k, v in sc.d_of(r).items()}
    ev.desirability = round(float(sc.D_of(r)), 9)
    return ev


def evaluate_reference_start(objective_cfg: Dict[str, Any]) -> BatchEvaluation:
    return evaluate_design(dict(START), objective_cfg)


def check_constraints(evaluation: BatchEvaluation,
                      constraints_cfg: Dict[str, Any]) -> Dict[str, bool]:
    """One boolean per hard constraint declared in task_config.yaml.

    Absent keys are not checked. Every flag is False for a submission that is not even a member of
    the design space, because none of the downstream quantities exist for it.
    """
    flags: Dict[str, bool] = {"on_grid_design": evaluation.design is not None}

    declared = [k for k in ("change_budget", "boiling_point_margin", "just_suspended_speed",
                            "total_solvent", "total_process_time", "classification_consistency")
                if constraints_cfg.get(k) is not None]
    for key in declared:
        flags[key] = False

    if evaluation.design is None:
        return flags

    d = evaluation.design
    viol = evaluation.process_constraint_violations or []
    for key in declared:
        cfg = constraints_cfg[key]
        if key == "change_budget":
            flags[key] = evaluation.edit_cost is not None and evaluation.edit_cost <= int(cfg["max_cost"])
        elif key == "boiling_point_margin":
            flags[key] = not any(v.startswith("C3") for v in viol)
        elif key == "just_suspended_speed":
            flags[key] = not any(v.startswith("C4") for v in viol)
        elif key == "total_solvent":
            flags[key] = (evaluation.total_solvent_L_per_kg is not None
                          and evaluation.total_solvent_L_per_kg <= float(cfg["max_L_per_kg"]) + 1e-9)
        elif key == "total_process_time":
            flags[key] = (evaluation.total_process_hours is not None
                          and evaluation.total_process_hours <= float(cfg["max_hours"]) + 1e-9)
        elif key == "classification_consistency":
            flags[key] = not any(v.startswith("C7") for v in viol)
    return flags


def satisfies_all_constraints(evaluation: BatchEvaluation,
                              constraints_cfg: Dict[str, Any]) -> bool:
    return all(check_constraints(evaluation, constraints_cfg).values())


def slot_table_text() -> str:
    """The level grid, formatted for the prompt.

    Generated from LEVELS / EDIT_COST so the prompt can never disagree with the design space or the
    cost table -- the question text is verified against the same source by
    scripts/leak_check_question.py.
    """
    lines = []
    for s in SLOTS:
        levels = ", ".join(str(v) for v in LEVELS[s])
        lines.append(f"  {s} [{SLOT_UNITS[s]}] (change cost {EDIT_COST[s]}): "
                     f"allowed values = [{levels}]")
    return "\n".join(lines)

"""Evaluator: turn a submitted formulation into (a) constraint flags and (b) the objective value.

The objective is the SMOOTH two-sided desirability D over five CQA specification windows
(design_space.Desirability), measured with the shipped oracle (src/formulation.py). Feasibility is
computed here but earns no points -- see src/scoring.py for why. The whole desirability shape, not
just the windows, comes out of task_config.yaml: scorer_from_config refuses to guess a default,
because a ruler with an implicit shape is a ruler that can change without anyone noticing.

Constraint keys absent from task_config.yaml are simply not checked, so an instance can declare only
the constraints its mechanism needs instead of carrying inert placeholder windows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from design_space import Desirability
from formulation import (CQA_KEYS, CORE_TABLET_MASS_MG, DesignError, MIN_FILLER_PCT, SLOTS, START,
                         acid_consistent, changed_slots, composition, composition_mg,
                         n_identity_changes, n_slots_changed, normalise_design, ruzicka_overlap)
# The measurement comes THROUGH the boundary module (src/response_model.py), which ships only in the
# grader package. In the public package these raise ResponseModelUnavailable, so a public run cannot
# produce a CQA -- by design.
from response_model import cqa_true, get_oracle


@dataclass
class FormulationEvaluation:
    """One graded submission. `desirability` is the objective; everything else is diagnostics."""
    submitted: Optional[Dict[str, Any]] = None
    valid: bool = False
    design: Optional[Dict[str, Any]] = None
    composition_pct: Optional[Dict[str, float]] = None
    composition_mg: Optional[Dict[str, float]] = None
    qs_filler_pct: Optional[float] = None
    core_mass_mg: float = CORE_TABLET_MASS_MG
    composition_overlap: Optional[float] = None
    n_slots_changed: Optional[int] = None
    changed_slots: List[str] = field(default_factory=list)
    n_identity_changes: Optional[int] = None
    acid_consistent: Optional[bool] = None
    cqa: Optional[Dict[str, float]] = None
    desirability_per_cqa: Optional[Dict[str, float]] = None
    desirability: Optional[float] = None
    # THE RELEASE GATE (audit finding #3). all_cqa_in_spec is what scoring.is_cqa_compliant reads,
    # and a submission with all_cqa_in_spec False can neither score nor be a success, whatever its
    # aggregate desirability is. It is still never returned to the model (src/feedback.py picks its
    # fields explicitly): the five window definitions are printed in question.md in full, so the
    # model can and must classify its own measured CQAs.
    n_cqa_in_spec: Optional[int] = None
    all_cqa_in_spec: Optional[bool] = None
    # Which windows were missed, and by how far in half-window units. Report-side only -- never
    # returned to the model (src/feedback.py picks its fields explicitly), because handing back
    # per-CQA window verdicts would turn the task into search against a free perfect verifier.
    cqa_out_of_spec: List[str] = field(default_factory=list)
    u_per_cqa: Optional[Dict[str, float]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DESIRABILITY_KEYS = ("hill_exponent", "d_at_spec_limit", "tail_decay_multiplier", "floor",
                     "plateau_fraction", "weights")


def scorer_from_config(objective_cfg: Dict[str, Any]) -> Desirability:
    """Build the desirability scorer from the windows and shape stored in task_config.yaml.

    The windows and the shape parameters are produced by scripts/enumerate_reference_best.py;
    check_consistency.py asserts that the stored numbers still equal what that script recomputes AND
    that the scorer this function returns is that same object -- the shape parameters are part of the
    ruler, so a config that carried the windows but not the shape would be a silent recalibration.
    """
    windows = objective_cfg["windows"]
    L, T, U = [], [], []
    for key in CQA_KEYS:
        w = windows[key]
        L.append(float(w["lower"]))
        T.append(float(w["target"]))
        U.append(float(w["upper"]))
    if "desirability" not in objective_cfg:
        raise ValueError(
            "objective.desirability is absent. This config predates the smooth-desirability "
            "recalibration (it probably still carries objective.desirability_exponent, the "
            "piecewise-cubic Derringer-Suich ruler with a hard zero outside each window). That "
            "ruler is no longer implemented, so a run graded with it cannot be re-derived here; "
            "the run has to be repeated against the current task_config.yaml.")
    shape = objective_cfg["desirability"]
    missing = [k for k in DESIRABILITY_KEYS if k not in shape]
    if missing:
        raise ValueError(f"objective.desirability is missing {missing}: the desirability shape is "
                         f"part of the ruler and must not fall back to a default")
    return Desirability(np.array(L), np.array(T), np.array(U),
                        hill_exponent=float(shape["hill_exponent"]),
                        d_at_spec_limit=float(shape["d_at_spec_limit"]),
                        tail_decay_multiplier=float(shape["tail_decay_multiplier"]),
                        floor=float(shape["floor"]),
                        plateau_fraction=float(shape["plateau_fraction"]),
                        weights=[float(x) for x in shape["weights"]])


def evaluate_design(raw_design: Any, objective_cfg: Dict[str, Any],
                    reference_design: Optional[Dict[str, Any]] = None,
                    oracle_path: Optional[str] = None) -> FormulationEvaluation:
    """Evaluate one submitted formulation.

    A design that is off-grid or cannot close mass is `valid=False` and gets no CQAs at all: there is
    no such tablet to measure. A design that is on-grid but blows the change budget IS measured --
    the budget is a feasibility gate applied by the scorer, not a refusal to run the analysis --
    which keeps the reported diagnostics informative without awarding it any points.
    """
    ref = reference_design or START
    ev = FormulationEvaluation(submitted=raw_design if isinstance(raw_design, dict) else None)

    try:
        design = normalise_design(raw_design)
    except DesignError as exc:
        ev.error = str(exc)
        return ev

    ev.design = design
    ev.acid_consistent = acid_consistent(design)
    comp = composition(design)
    ev.composition_pct = {k: round(v, 6) for k, v in comp.items() if v}
    ev.composition_mg = composition_mg(design)
    ev.qs_filler_pct = round(comp["DCPA"], 6)
    ev.composition_overlap = round(ruzicka_overlap(design, ref), 6)
    ev.n_slots_changed = n_slots_changed(design, ref)
    ev.changed_slots = changed_slots(design, ref)
    ev.n_identity_changes = n_identity_changes(design, ref)

    if not ev.acid_consistent:
        ev.error = ("acidifier identity and amount disagree: name an acid with a non-zero amount, "
                    "or 'none' with 0 %")
        return ev
    if comp["DCPA"] < MIN_FILLER_PCT:
        ev.error = ("mass closure failed: the q.s. filler would be %.3f %% of the core, below the "
                    "%.1f %% minimum" % (comp["DCPA"], MIN_FILLER_PCT))
        return ev

    y = get_oracle(oracle_path).predict_one(design)
    sc = scorer_from_config(objective_cfg)
    d = sc.d_of(y[None, :])[0]
    ev.valid = True
    ev.cqa = {k: round(float(v), 4) for k, v in zip(CQA_KEYS, y)}
    ev.desirability_per_cqa = {k: float("%.6g" % float(v)) for k, v in zip(CQA_KEYS, d)}
    ev.desirability = float("%.9g" % float(sc.D_of(y[None, :])[0]))
    in_spec = (y > sc.L) & (y < sc.U)
    ev.n_cqa_in_spec = int(in_spec.sum())
    ev.all_cqa_in_spec = bool(in_spec.all())
    ev.cqa_out_of_spec = [k for k, ok in zip(CQA_KEYS, in_spec) if not ok]
    u = sc.u_of(y[None, :])[0]
    ev.u_per_cqa = {k: round(float(v), 5) for k, v in zip(CQA_KEYS, u)}
    return ev


def evaluate_reference_start(objective_cfg: Dict[str, Any],
                             oracle_path: Optional[str] = None) -> FormulationEvaluation:
    return evaluate_design(dict(START), objective_cfg, oracle_path=oracle_path)


def check_constraints(evaluation: FormulationEvaluation,
                      constraints: Dict[str, Any]) -> Dict[str, bool]:
    """One boolean per hard constraint declared in task_config.yaml.

    Absent keys are not checked. All flags are False for a design that is not even a member of the
    design space, because none of the downstream quantities exist for it.
    """
    flags: Dict[str, bool] = {"on_grid_design": evaluation.design is not None}

    declared = [k for k in ("acid_consistency", "mass_closure", "composition_overlap",
                            "max_slot_changes", "max_identity_changes", "lubricant_range",
                            "acidifier_total") if constraints.get(k) is not None]
    for key in declared:
        flags[key] = False

    if evaluation.design is None:
        return flags

    d = evaluation.design
    for key in declared:
        cfg = constraints[key]
        if key == "acid_consistency":
            flags[key] = bool(evaluation.acid_consistent)
        elif key == "mass_closure":
            flags[key] = (evaluation.qs_filler_pct is not None
                          and evaluation.qs_filler_pct >= float(cfg["min_filler_pct"]) - 1e-9)
        elif key == "composition_overlap":
            flags[key] = (evaluation.composition_overlap is not None
                          and evaluation.composition_overlap >= float(cfg["min"]) - 1e-9)
        elif key == "max_slot_changes":
            flags[key] = (evaluation.n_slots_changed is not None
                          and evaluation.n_slots_changed <= int(cfg["max"]))
        elif key == "max_identity_changes":
            flags[key] = (evaluation.n_identity_changes is not None
                          and evaluation.n_identity_changes <= int(cfg["max"]))
        elif key == "lubricant_range":
            flags[key] = float(cfg["min"]) - 1e-9 <= float(d["T8_mgst_pct"]) <= float(cfg["max"]) + 1e-9
        elif key == "acidifier_total":
            flags[key] = float(d["T4_acid_pct"]) <= float(cfg["max"]) + 1e-9
    return flags


def satisfies_all_constraints(evaluation: FormulationEvaluation,
                              constraints: Dict[str, Any]) -> bool:
    return all(check_constraints(evaluation, constraints).values())


def oracle_vs_generator_drift(design: Dict[str, Any],
                              oracle_path: Optional[str] = None) -> np.ndarray:
    """|oracle - hidden generator| for one design. Reported by check_consistency.py so that a
    reviewer can see the fitted ruler is a faithful reading of the mechanism it was fitted to.
    The oracle IS the ruler, so this difference is descriptive, not an error term in the score."""
    return np.abs(get_oracle(oracle_path).predict_one(design) - cqa_true(design))

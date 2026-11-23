"""Diagnostic tool layer: three optional tools the model calls itself, with real failure returns.

Audit finding #5 (2026-07-29): "the flow is driven by a fixed harness that always calls the same
prediction model; there is no record of the model choosing a tool, handling a tool failure, or
changing the order in which it gathers information." That is fixed here rather than by downgrading
the task's claim: the model now decides, in its own words, whether to call a tool, which one, how
many times and with what arguments, and it has to recover from malformed calls itself.

THE HARD CONSTRAINT ON WHAT A TOOL MAY RETURN
---------------------------------------------
No tool returns, or lets a caller assemble, any part of the objective. Specifically none of them
returns a CQA value, a per-CQA desirability, the aggregate desirability D, an achievement fraction,
a score, a band, a compliance verdict, the reference design or anything derived from them. Every
number a tool returns is arithmetic on the composition table that question.md already prints in
full, or a generic qualitative caution of the kind any formulator would have in mind before opening
the file. The measurement channel is unchanged: submitting a formulation is still the only way to
learn a CQA.

That is why the tool layer cannot move the difficulty gates. The gates
(scripts/enumerate_reference_best.py, measure_naive_answers.py, measure_pregates.py,
measure_trap_families.py) are properties of the design space and the ruler; the tools add no
information about either. What the tools can change is model BEHAVIOUR, which is why the three
calibration models were re-run from scratch after this file landed rather than re-scored.

THE THREE TOOLS
---------------
validate_formulation   is this a legal submission? on-grid, mass closure, q.s. filler, acid
                       consistency, the boolean hard-constraint checklist, budget bookkeeping.
composition_overlap    mass-weighted composition overlap between two formulations (or against the
                       starting formulation), with the per-component min/max terms.
excipient_note         qualitative compatibility / risk caution for ONE excipient code, plus its
                       role and the slot that controls it. No direction on any CQA.

REAL FAILURE MODES (all of them reachable, all of them structured)
-----------------------------------------------------------------
unknown_tool           the name is not one of the three
bad_arguments          arguments is not an object, or a required argument is missing
off_grid               a slot value is not a declared level (per-slot detail returned)
missing_slots          fewer than twelve slots given
unknown_excipient      the code is not in the substance table (valid codes returned)
quota_exceeded         the per-round tool-call budget is spent; the model must submit
Each error carries `error`, `detail` and `recoverable`, so a model can tell "fix the call" from
"stop calling and submit".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from evaluator import check_constraints, FormulationEvaluation
from formulation import (DesignError, LEVELS, MIN_FILLER_PCT, SLOTS, SLOT_UNITS, START,
                         SUBSTANCE_LABELS, SUBSTANCES, acid_consistent, changed_slots,
                         composition, composition_mg, n_identity_changes, n_slots_changed,
                         normalise_design, ruzicka_overlap)

TOOL_NAMES = ("validate_formulation", "composition_overlap", "excipient_note")

# Per-round tool budget. Deliberately smaller than the number of tools times the number of things a
# model might want to check, so a model that spends the budget on redundant calls loses the chance to
# check the thing that mattered -- and hits quota_exceeded, which is a real failure it has to handle.
DEFAULT_TOOL_BUDGET_PER_ROUND = 4
DEFAULT_MAX_TOOL_TURNS_PER_ROUND = 2

# Qualitative cautions. Deliberately DIRECTION-FREE with respect to the five CQAs of this product:
# each entry names the excipient's role and the standard risks associated with it, and says outright
# that the magnitude and even the sign of its effect on this formulation is not predicted here.
# Anything stronger would hand over the mechanism map and make the design problem a lookup.
_NOTES: Dict[str, Dict[str, str]] = {
    "API": {"role": "active ingredient, 2.5 % w/w of the core",
            "caution": "pH-labile amine salt with an ester linkage; low dose, so content uniformity "
                       "depends on the ordered mix surviving hopper residence."},
    "DCPA": {"role": "q.s. main filler (brittle, high density)",
             "caution": "alkaline inorganic filler; brittle fracture rather than plastic bonding. "
                        "Its level is not a design variable -- it absorbs every other change."},
    "MCC": {"role": "dry binder / plastic filler; grade set by T1, level by T2",
            "caution": "grades differ in particle size and in bonding capacity; all grades of this "
                       "material carry appreciable equilibrium moisture."},
    "MAN": {"role": "soluble diluent (T12); introducing it counts as an identity change",
            "caution": "spray-dried polyol; low hygroscopicity, limited bonding contribution."},
    "FUM": {"role": "acidifier option (T3)",
            "caution": "poorly wetting acid; also behaves as a weak lubricant at higher levels."},
    "TAR": {"role": "acidifier option (T3)",
            "caution": "stronger acid than citric per unit mass; moderately hygroscopic."},
    "CIT": {"role": "acidifier in the starting formulation (T3/T4)",
            "caution": "hygroscopic and deliquescence-prone; the standard caution for citric acid "
                       "in a moisture-sensitive blend."},
    "MEG": {"role": "alkalising / pH-modifying agent (T5)",
            "caution": "very high base capacity per unit mass, so small amounts do a lot; "
                       "hygroscopic."},
    "XPVP": {"role": "superdisintegrant option (T6)",
             "caution": "insoluble, non-ionic, swelling; moderate moisture uptake."},
    "CCS": {"role": "superdisintegrant in the starting formulation (T6/T7)",
            "caution": "anionic (sodium form), strongly swelling, hygroscopic; ionic "
                       "interaction with cationic actives is a documented general concern."},
    "LHPC": {"role": "superdisintegrant option (T6)",
             "caution": "fibrous, non-ionic; contributes bonding as well as disintegration."},
    "SSG": {"role": "superdisintegrant option (T6)",
            "caution": "anionic starch derivative; the most hygroscopic of the four options."},
    "MGST": {"role": "boundary lubricant (T8), blended for T9 minutes",
             "caution": "hydrophobic; over-lubrication with extended blending is the classic "
                        "failure mode. Below the qualified anti-sticking minimum, sticking and "
                        "ejection problems instead."},
    "SIL": {"role": "glidant (T10)",
            "caution": "very high specific surface area at tiny mass fractions; flow benefit "
                       "saturates, and it also acts as a moisture scavenger."},
}
_DISCLAIMER = ("Generic formulation caution, not a prediction for this product. It does not state "
               "which CQA moves in which direction, or by how much, on this formulation. Submit a "
               "formulation to measure its CQAs.")


def tool_catalogue() -> List[Dict[str, Any]]:
    """The tool declarations shown to the model. Rendered into the prompt by src/main.py."""
    return [
        {
            "tool": "validate_formulation",
            "purpose": "Check whether a candidate formulation is a legal submission, and get its "
                       "budget bookkeeping, BEFORE spending the round on it.",
            "arguments": {"formulation": "object with all 12 slot keys"},
            "returns": ["on_grid", "per_slot_errors", "acid_consistent", "qs_filler_pct",
                        "mass_closure_ok", "composition_pct", "composition_mg",
                        "n_slots_changed", "slots_changed", "n_identity_changes",
                        "composition_overlap_with_start", "constraint_checklist",
                        "all_hard_constraints_satisfied"],
            "does_not_return": ["any CQA value", "desirability", "score", "compliance verdict"],
        },
        {
            "tool": "composition_overlap",
            "purpose": "Mass-weighted composition overlap between two formulations, with the "
                       "per-component terms, so you can plan how to spend the overlap floor.",
            "arguments": {"formulation": "object with all 12 slot keys",
                          "against": "optional: another 12-slot object; defaults to the starting "
                                     "formulation"},
            "returns": ["overlap", "sum_min", "sum_max", "per_component", "floor_satisfied"],
            "does_not_return": ["any CQA value", "desirability", "score"],
        },
        {
            "tool": "excipient_note",
            "purpose": "Qualitative compatibility / risk caution for ONE excipient code, plus its "
                       "role and the slot that controls it.",
            "arguments": {"code": "one substance code, e.g. \"MGST\""},
            "returns": ["code", "substance", "role", "caution", "controlled_by", "disclaimer"],
            "does_not_return": ["any CQA value", "any direction of effect on a CQA",
                               "desirability", "score"],
        },
    ]


def _err(kind: str, detail: str, recoverable: bool = True, **extra: Any) -> Dict[str, Any]:
    out = {"ok": False, "error": kind, "detail": detail, "recoverable": recoverable}
    out.update(extra)
    return out


def _slot_of(code: str) -> Optional[str]:
    for slot in ("T1_mcc_grade", "T3_acid_id", "T6_dis_id"):
        if code in LEVELS[slot]:
            return slot
    return {"MCC": "T1_mcc_grade / T2_mcc_pct", "MAN": "T12_man_pct", "MEG": "T5_meg_pct",
            "MGST": "T8_mgst_pct", "SIL": "T10_sil_pct", "DCPA": "(q.s., not a variable)",
            "API": "(fixed at 2.5 % w/w)",
            "FUM": "T3_acid_id / T4_acid_pct", "TAR": "T3_acid_id / T4_acid_pct",
            "CIT": "T3_acid_id / T4_acid_pct", "XPVP": "T6_dis_id / T7_dis_pct",
            "CCS": "T6_dis_id / T7_dis_pct", "LHPC": "T6_dis_id / T7_dis_pct",
            "SSG": "T6_dis_id / T7_dis_pct"}.get(code)


def _normalise_or_error(raw: Any, field: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict]]:
    if not isinstance(raw, dict):
        return None, _err("bad_arguments",
                          f"`{field}` must be a JSON object keyed by slot name, got "
                          f"{type(raw).__name__}")
    missing = [s for s in SLOTS if s not in raw]
    if missing:
        return None, _err("missing_slots",
                          f"`{field}` must give all twelve slots; missing {missing}",
                          missing_slots=missing, required_slots=list(SLOTS))
    per_slot: Dict[str, str] = {}
    for s in SLOTS:
        try:
            normalise_design({**{k: START[k] for k in SLOTS}, s: raw[s]})
        except DesignError as exc:
            per_slot[s] = str(exc)
    if per_slot:
        return None, _err("off_grid", "one or more slot values are not declared levels",
                          per_slot_errors=per_slot,
                          allowed_levels={s: LEVELS[s] for s in per_slot})
    extra = [k for k in raw if k not in SLOTS]
    if extra:
        return None, _err("bad_arguments",
                          f"unknown keys in `{field}` (no new excipients or variables): {extra}",
                          unknown_keys=extra)
    return normalise_design({s: raw[s] for s in SLOTS}), None


def _validate_formulation(args: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    design, err = _normalise_or_error(args.get("formulation"), "formulation")
    if err:
        return err
    comp = composition(design)
    ev = FormulationEvaluation(
        design=design, acid_consistent=acid_consistent(design),
        qs_filler_pct=round(comp["DCPA"], 6),
        composition_overlap=round(ruzicka_overlap(design), 6),
        n_slots_changed=n_slots_changed(design), changed_slots=changed_slots(design),
        n_identity_changes=n_identity_changes(design))
    flags = check_constraints(ev, constraints)
    return {
        "ok": True,
        "on_grid": True,
        "per_slot_errors": {},
        "acid_consistent": ev.acid_consistent,
        "qs_filler_pct": ev.qs_filler_pct,
        "mass_closure_ok": ev.qs_filler_pct >= MIN_FILLER_PCT,
        "composition_pct": {k: round(v, 4) for k, v in comp.items() if v},
        "composition_mg": composition_mg(design),
        "n_slots_changed": ev.n_slots_changed,
        "slots_changed": ev.changed_slots,
        "n_identity_changes": ev.n_identity_changes,
        "composition_overlap_with_start": ev.composition_overlap,
        "constraint_checklist": flags,
        "all_hard_constraints_satisfied": all(flags.values()),
        "note": ("Legality and budget bookkeeping only. This tool does not measure the formulation: "
                 "no CQA, no desirability, no verdict on the specification windows."),
    }


def _composition_overlap(args: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    a, err = _normalise_or_error(args.get("formulation"), "formulation")
    if err:
        return err
    if args.get("against") is None:
        b, label = dict(START), "the starting formulation"
    else:
        b, err = _normalise_or_error(args.get("against"), "against")
        if err:
            return err
        label = "the formulation given in `against`"
    wa, wb = composition(a), composition(b)
    per = {s: {"a": round(wa[s], 4), "b": round(wb[s], 4),
               "min": round(min(wa[s], wb[s]), 4), "max": round(max(wa[s], wb[s]), 4)}
           for s in SUBSTANCES if wa[s] or wb[s]}
    smin = sum(min(wa[s], wb[s]) for s in SUBSTANCES)
    smax = sum(max(wa[s], wb[s]) for s in SUBSTANCES)
    floor = (constraints.get("composition_overlap") or {}).get("min")
    ov = round(smin / smax, 6)
    return {"ok": True, "compared_against": label, "overlap": ov,
            "sum_min": round(smin, 6), "sum_max": round(smax, 6), "per_component": per,
            "floor": floor,
            "floor_satisfied": None if floor is None else bool(ov >= float(floor) - 1e-9),
            "note": "Mass-weighted (Ruzicka) overlap on the composition vector. Process set points "
                    "do not appear in it."}


def _excipient_note(args: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return _err("bad_arguments", "`code` must be a non-empty substance code string",
                    valid_codes=sorted(_NOTES))
    key = code.strip().upper()
    if key not in _NOTES:
        return _err("unknown_excipient", f"{code!r} is not a substance in this formulation system",
                    valid_codes=sorted(_NOTES))
    rec = _NOTES[key]
    return {"ok": True, "code": key, "substance": SUBSTANCE_LABELS.get(key, key),
            "role": rec["role"], "caution": rec["caution"], "controlled_by": _slot_of(key),
            "disclaimer": _DISCLAIMER}


_DISPATCH = {"validate_formulation": _validate_formulation,
             "composition_overlap": _composition_overlap,
             "excipient_note": _excipient_note}


def run_tool_call(call: Any, constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one tool call. Never raises: every failure is a structured, recoverable return."""
    if not isinstance(call, dict):
        return _err("bad_arguments", "each entry of `tool_calls` must be a JSON object with `tool` "
                                     "and `arguments`", tool="(not an object)",
                    available_tools=list(TOOL_NAMES))
    name = call.get("tool") or call.get("name")
    if not isinstance(name, str) or name not in _DISPATCH:
        return _err("unknown_tool", f"{name!r} is not an available tool",
                    tool=str(name), available_tools=list(TOOL_NAMES))
    args = call.get("arguments", call.get("args", {}))
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return _err("bad_arguments", "`arguments` must be a JSON object",
                    tool=name)
    try:
        out = _DISPATCH[name](args, constraints)
    except Exception as exc:                      # defensive: a tool must never kill the run
        return _err("tool_internal_error", f"{type(exc).__name__}: {exc}", recoverable=True,
                    tool=name)
    out.setdefault("tool", name)
    out["tool"] = name
    return out


def run_tool_calls(calls: Any, constraints: Dict[str, Any], budget_left: int) -> Dict[str, Any]:
    """Execute a batch of tool calls against the remaining per-round budget.

    Returns {"results": [...], "n_executed": int, "budget_left": int}. Calls beyond the budget are
    NOT executed and come back as `quota_exceeded` -- a real failure the model has to notice.
    """
    if not isinstance(calls, list) or not calls:
        return {"results": [_err("bad_arguments", "`tool_calls` must be a non-empty JSON array",
                                 available_tools=list(TOOL_NAMES))],
                "n_executed": 0, "budget_left": budget_left}
    results: List[Dict[str, Any]] = []
    executed = 0
    for call in calls:
        if budget_left - executed <= 0:
            results.append(_err("quota_exceeded",
                                "the per-round tool-call budget is spent; submit a formulation now "
                                "instead of calling more tools", recoverable=False,
                                budget_left=0))
            continue
        results.append(run_tool_call(call, constraints))
        executed += 1
    return {"results": results, "n_executed": executed,
            "budget_left": max(0, budget_left - executed)}

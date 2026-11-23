"""PUBLIC half of the task definition: the design space, the change-cost table, the specification
windows and the hard constraints. **No response mechanism lives here.**

This module exists to make the evaluation isolation real rather than declared. External review found
that the model-visible material carried the reference design and the complete response mechanism, and
that an agent with file access could therefore read the answer or optimise against the response
surface. The package is now split along one line:

    src/design_grid.py      PUBLIC.  Everything that is printed in tasks/q1_agglomeration/question.md
                                     and nothing else: the twelve level grids, the change costs, the
                                     baseline process, the four specification windows, and the C1/C3-C7
                                     constraint predicates together with the handful of physical
                                     constants those predicates need (boiling points, addition and
                                     drying durations, sieve cuts).
    src/response_model.py   BOUNDARY. One import. The public package ships a stub that documents the
                                     grader-side evaluation contract and refuses to invent numbers.
    src/agglomeration.py    GRADER.   The synthetic response mechanism: the population balance, the
                                     frozen surrogate, the yield mass balance and every mechanistic
                                     coefficient. Absent from the public package.

An agent that reads every byte of the public package therefore learns exactly what question.md already
tells it. `scripts/leak_check_question.py` and `scripts/package_for_delivery.py` assert that split.

Nothing in this file is secret and nothing in it is a hint: every number below appears verbatim in the
question text, because an agent must not be graded on information it was not given.

DATA NATURE. The scenario is SYNTHETIC. There is no registered product, no filed process and no
experimental measurement anywhere in this benchmark. `START` is a designed baseline, not a real
process, and the four responses are simulator outputs. See data/MODEL_CARD.md (grader side).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class DesignError(ValueError):
    """The submission is not a member of the design space (off-grid value, missing or extra slot)."""


# =================================================================================================
# design space
# =================================================================================================
LEVELS: Dict[str, List[Any]] = {
    "S1_bridging_liquid": ["heptane", "cyclohexane", "heptane_EtOAc_1_1", "toluene", "MTBE", "EtOAc"],
    "S2_bridging_ratio_v_w": [0.26, 0.30, 0.35, 0.41, 0.47, 0.55],
    "S3_addition_mode": ["one_shot", "10_min", "30_min", "60_min", "90_min", "3_pulses"],
    "S4_solids_g_per_L": [40, 70, 100, 140, 190, 260],
    "S5_tip_speed_m_s": [0.5, 0.7, 0.9, 1.4, 2.0, 2.8, 3.6],
    "S6_temperature_C": [5, 15, 25, 35, 45],
    "S7_agglomeration_min": [10, 25, 45, 75, 120],
    "S8_primary_d50_um": [5, 9, 12, 20, 32],
    "S9_water_fraction": [0.55, 0.65, 0.75, 0.85],
    "S10_classification": ["none", "remove_coarse", "remove_fines", "remove_both",
                           "remove_both_recycle", "remove_both_recycle_high"],
    "S11_drying": ["20C_N2_30h", "25C_N2_24h", "40C_vac_12h", "50C_vac_6h", "60C_vac_4h",
                   "40C_stepped_10h"],
    "S12_HPMC_ppm": [0, 100, 300, 800],
}
SLOTS: List[str] = list(LEVELS.keys())

SLOT_UNITS: Dict[str, str] = {
    "S1_bridging_liquid": "bridging liquid identity",
    "S2_bridging_ratio_v_w": "L bridging liquid per kg API (v/w)",
    "S3_addition_mode": "bridging-liquid addition profile",
    "S4_solids_g_per_L": "g API per L continuous phase",
    "S5_tip_speed_m_s": "impeller tip speed, m/s",
    "S6_temperature_C": "agglomeration temperature, degC",
    "S7_agglomeration_min": "hold time after addition, min",
    "S8_primary_d50_um": "primary-crystal volume-median size, um (set upstream by the cooling rate)",
    "S9_water_fraction": "water mass fraction of the continuous phase",
    "S10_classification": "wet classification / recycle scheme",
    "S11_drying": "drying programme",
    "S12_HPMC_ppm": "HPMC in the continuous phase, ppm",
}

# Weighted controlled-change score. A change is priced by what it costs in regulatory and plant
# terms, not by how much it moves the responses.
#
# SCENARIO VOCABULARY. "registered" below is a property of the INVENTED change-control scenario
# this exercise is set in, not a claim that anything here was ever filed with a regulator. There
# is no dossier, no registered product and no real process anywhere in this benchmark; the words
# exist to make the price of a move intelligible to someone who knows the domain.
#   3  a material of record or a new unit operation  (bridging-liquid identity, drying programme,
#      introducing an additive)
#   2  a quantity of record or a downstream scheme    (bridging-liquid ratio, solids loading,
#      continuous-phase composition, classification scheme)
#   1  an in-process set point                        (addition profile, tip speed, temperature,
#      hold time, primary-crystal grade)
# Changing all twelve costs 22.
EDIT_COST: Dict[str, int] = {
    "S1_bridging_liquid": 3, "S11_drying": 3, "S12_HPMC_ppm": 3,
    "S2_bridging_ratio_v_w": 2, "S4_solids_g_per_L": 2, "S9_water_fraction": 2,
    "S10_classification": 2,
    "S3_addition_mode": 1, "S5_tip_speed_m_s": 1, "S6_temperature_C": 1,
    "S7_agglomeration_min": 1, "S8_primary_d50_um": 1,
}
TOTAL_EDIT_COST = sum(EDIT_COST.values())          # 22
BUDGET = 6
# Why 6. Measured on the exhaustive enumeration of the change neighbourhood, not guessed: it is set
# to exactly the cost of the cheapest coordinated move that reaches the desirability plateau --
# enough budget for one such move, not enough for one plus slack. A wider budget put that move
# comfortably inside the "spend less than the budget" difficulty gate and the instance stopped being
# hard. Which move that is, and which variables it touches, is grader-side and deliberately not
# written here: this module ships in the participant archive.
# Change this number and every calibrated quantity downstream (reference optimum, all gates) must be
# re-derived; scripts/check_consistency.py refuses to pass if they were not.

# The SYNTHETIC baseline process the agent starts from. All four simulated responses sit OUTSIDE
# their windows here, each on a different side, so no single response can be repaired without pushing
# another out. This is a designed starting point, not a real filed process.
START: Dict[str, Any] = {
    "S1_bridging_liquid": "EtOAc",
    "S2_bridging_ratio_v_w": 0.35,
    "S3_addition_mode": "30_min",
    "S4_solids_g_per_L": 100,
    "S5_tip_speed_m_s": 1.4,
    "S6_temperature_C": 25,
    "S7_agglomeration_min": 45,
    "S8_primary_d50_um": 12,
    "S9_water_fraction": 0.75,
    "S10_classification": "remove_fines",
    "S11_drying": "50C_vac_6h",
    "S12_HPMC_ppm": 0,
}
X0 = START      # the prototype's name for it, kept so ported scripts read the same

RESPONSE_KEYS = ["d50", "strength", "porosity", "yield_pct"]
RESPONSE_UNITS = {
    "d50": "um (volume-median agglomerate diameter of the isolated, dried product; SIMULATED)",
    "strength": "MPa (agglomerate crush strength; SIMULATED)",
    "porosity": "dimensionless intra-agglomerate porosity (SIMULATED)",
    "yield_pct": "% unit yield of isolated dried agglomerates on API basis (SIMULATED)",
}

# Two-sided specification windows: d = 1 inside [pl, pu], smoothly decaying outside, d = 0.25
# exactly at the acceptable limit (al / au) and never reaching 0. Weights (2, 2, 1, 1).
# THIS DICT IS THE AUTHORITY; tasks/q1_agglomeration/task_config.yaml carries a copy that
# scripts/check_consistency.py asserts is identical.
WINDOWS: Dict[str, Dict[str, float]] = {
    "d50":       dict(pl=320.0, pu=390.0, al=295.0, au=415.0, w=2.0),
    "strength":  dict(pl=0.95,  pu=1.28,  al=0.75,  au=1.48,  w=2.0),
    "porosity":  dict(pl=0.352, pu=0.385, al=0.340, au=0.397, w=1.0),
    "yield_pct": dict(pl=89.5,  pu=92.5,  al=87.9,  au=94.1,  w=1.0),
}
DESIRABILITY_SHAPE_M = 2.5      # d = 1 / (1 + (C t)^m), C chosen so d(t=1) = 0.25


# =================================================================================================
# the only physical constants the PUBLIC constraint predicates need
#
# All four tables below are printed in question.md, because C3-C7 must be checkable by hand. The
# mechanistic properties of the same materials and programmes (interfacial tension, wetting angle,
# solubility, miscibility, distribution uniformity, nucleus count, capillary shrinkage, cracking,
# residual solvent, recycle fraction, needle aspect ratio) are NOT here: they are response-model
# parameters and live in src/agglomeration.py on the grader side.
# =================================================================================================
BP: Dict[str, float] = {          # atmospheric boiling point, degC -- C3 only
    "heptane": 98.4, "cyclohexane": 80.7, "heptane_EtOAc_1_1": 84.0,
    "toluene": 110.6, "MTBE": 55.2, "EtOAc": 77.1,
}
ADDITION_MINUTES: Dict[str, float] = {   # bridging-liquid addition duration, min -- C6 only
    "one_shot": 0.0, "10_min": 10.0, "30_min": 30.0, "60_min": 60.0, "90_min": 90.0,
    "3_pulses": 40.0,
}
DRYING_HOURS: Dict[str, float] = {       # drying programme duration, h -- C6 only
    "20C_N2_30h": 30.0, "25C_N2_24h": 24.0, "40C_vac_12h": 12.0, "50C_vac_6h": 6.0,
    "60C_vac_4h": 4.0, "40C_stepped_10h": 10.0,
}
# What each classification scheme does, as booleans -- C7 only. The recycled FRACTION is a
# response-model parameter and is not public.
SCHEME_REMOVES_FINES: Dict[str, bool] = {
    "none": False, "remove_coarse": False, "remove_fines": True, "remove_both": True,
    "remove_both_recycle": True, "remove_both_recycle_high": True,
}
SCHEME_REMOVES_COARSE: Dict[str, bool] = {
    "none": False, "remove_coarse": True, "remove_fines": False, "remove_both": True,
    "remove_both_recycle": True, "remove_both_recycle_high": True,
}
SCHEME_RECYCLES_FINES: Dict[str, bool] = {
    "none": False, "remove_coarse": False, "remove_fines": False, "remove_both": False,
    "remove_both_recycle": True, "remove_both_recycle_high": True,
}

FINES_CUT = 106.0    # um, wet-sieve fines cut   (stated in question.md)
COARSE_CUT = 850.0   # um, wet-sieve coarse cut  (stated in question.md)

# Hard-constraint constants, stated in question.md so every limit can be checked by hand.
BP_MARGIN_C = 10.0            # C3: T <= bp - 10
JS_COEFF = 0.42               # C4: v_tip >= 0.42 * (solids/40)^0.30
JS_REF_SOLIDS = 40.0
JS_EXPONENT = 0.30
MAX_SOLVENT_L_PER_KG = 26.0   # C5: 1000/S4 + S2 + 2 <= 26
WASH_L_PER_KG = 2.0
MAX_PROCESS_HOURS = 34.0      # C6


# =================================================================================================
# design-space membership
# =================================================================================================
def _match_level(slot: str, value: Any) -> Any:
    """Snap a submitted value onto its declared level, or raise.

    Numeric levels are matched with a tolerance so 100 vs 100.0 vs "100" all resolve, but a value
    that is not a declared level is an error, never a silent nearest-neighbour substitution: the
    grid IS the constraint.
    """
    levels = LEVELS[slot]
    if isinstance(levels[0], str):
        if not isinstance(value, str):
            raise DesignError(f"{slot} must be one of {levels}, got {value!r}")
        for lv in levels:
            if value.strip().lower() == lv.lower():
                return lv
        raise DesignError(f"{slot}={value!r} is not a declared level; allowed: {levels}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise DesignError(f"{slot} must be numeric, got {value!r}") from None
    for lv in levels:
        if abs(x - float(lv)) <= 1e-9:
            return lv
    raise DesignError(f"{slot}={value!r} is not a declared level; allowed: {levels}")


def normalise_design(raw: Any) -> Dict[str, Any]:
    """Validate and canonicalise a submitted process. Raises DesignError on anything off-grid."""
    if not isinstance(raw, dict):
        raise DesignError("process must be a JSON object keyed by variable name")
    missing = [s for s in SLOTS if s not in raw]
    if missing:
        raise DesignError(f"missing variables: {missing}")
    extra = [k for k in raw if k not in SLOTS]
    if extra:
        raise DesignError(f"unknown variables (no new material or set point may be introduced): {extra}")
    return {s: _match_level(s, raw[s]) for s in SLOTS}


def edit_cost(x: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> int:
    ref = reference or START
    return sum(EDIT_COST[k] for k in SLOTS if x[k] != ref[k])


def changed_slots(x: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> List[str]:
    ref = reference or START
    return [k for k in SLOTS if x[k] != ref[k]]


def change_overlap(x: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> float:
    """The budget stated as an overlap with the baseline process: 1 - cost/22.

    Identical information to the cost; both are printed in the question because a process chemist
    reads "how much of the baseline survives" and a scorer reads "how many points did you spend".
    """
    return 1.0 - edit_cost(x, reference) / float(TOTAL_EDIT_COST)


def space_size() -> int:
    n = 1
    for v in LEVELS.values():
        n *= len(v)
    return n


# =================================================================================================
# constraints -- all five are public arithmetic on the tables above
# =================================================================================================
def constraints(x: Dict[str, Any]) -> List[str]:
    """Hard process constraints, as a list of human-readable violations (empty == feasible).

    C1 on-grid values / all twelve variables present
    C3 bridging-liquid boiling-point safety margin: T <= bp - 10 degC
    C4 Zwietering-type just-suspended tip speed at the chosen solids loading
    C5 total solvent (continuous phase + bridging liquid + wash) <= 26 L/kg
    C6 total processing time (addition + hold + drying) <= 34 h
    C7 internal consistency: fines recycle requires fines removal
    C2 (the controlled-change budget) is NOT here: it is a scoring gate applied by src/scoring.py,
    because an over-budget process is still a measurable batch and its responses stay informative.
    """
    v: List[str] = []
    for k, vals in LEVELS.items():
        if k not in x:
            v.append(f"C1 missing key {k}")
        elif x[k] not in vals:
            v.append(f"C1 bad value {k}={x[k]!r}")
    if v:
        return v
    bp = BP[x["S1_bridging_liquid"]]
    if float(x["S6_temperature_C"]) > bp - BP_MARGIN_C:
        v.append(f"C3 T={x['S6_temperature_C']} > bp-10={bp - BP_MARGIN_C:.1f} "
                 f"for {x['S1_bridging_liquid']}")
    vmin = just_suspended_tip_speed(x)
    if float(x["S5_tip_speed_m_s"]) < vmin - 1e-9:
        v.append(f"C4 tip speed {x['S5_tip_speed_m_s']} < just-suspended {vmin:.2f} m/s")
    solvent = total_solvent_L_per_kg(x)
    if solvent > MAX_SOLVENT_L_PER_KG + 1e-9:
        v.append(f"C5 total solvent {solvent:.1f} > {MAX_SOLVENT_L_PER_KG:g} L/kg")
    hours = total_hours(x)
    if hours > MAX_PROCESS_HOURS + 1e-9:
        v.append(f"C6 total process {hours:.1f} h > {MAX_PROCESS_HOURS:g} h")
    if (SCHEME_RECYCLES_FINES[x["S10_classification"]]
            and not SCHEME_REMOVES_FINES[x["S10_classification"]]):
        v.append("C7 fines recycle without fines removal")
    return v


def total_solvent_L_per_kg(x: Dict[str, Any]) -> float:
    return (1000.0 / float(x["S4_solids_g_per_L"]) + float(x["S2_bridging_ratio_v_w"])
            + WASH_L_PER_KG)


def total_hours(x: Dict[str, Any]) -> float:
    return (ADDITION_MINUTES[x["S3_addition_mode"]] / 60.0
            + float(x["S7_agglomeration_min"]) / 60.0
            + DRYING_HOURS[x["S11_drying"]])


def just_suspended_tip_speed(x: Dict[str, Any]) -> float:
    return JS_COEFF * (float(x["S4_solids_g_per_L"]) / JS_REF_SOLIDS) ** JS_EXPONENT

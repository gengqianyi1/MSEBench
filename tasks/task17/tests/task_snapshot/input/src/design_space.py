"""The objective, the score bands, and the enumeration of the change-budget neighbourhood.

Two things live here that must not be duplicated anywhere else:

1. `Desirability` -- the four-window Derringer-Suich objective. Built from the windows stored in
   tasks/q1_agglomeration/task_config.yaml, which scripts/check_consistency.py asserts are identical
   to agglomeration.WINDOWS. There is exactly one implementation of the score curve, used by the
   graded path (src/evaluator.py) and by the offline enumeration (scripts/), because every problem in
   this series that shipped a wrong window did so by letting a simplified stand-in compute one of the
   two.

2. `iter_budget_designs` / `iter_k_slot_designs` -- the mechanical enumerators. The difficulty gates
   are only meaningful if the move sets they scan were GENERATED, never hand-listed: a hand-written
   list of "moves a chemist would try" contains only the moves the author thought of, which confirms
   the design instead of testing it.

Note on paths: there is no vectorised twin of the response model. The population balance is a
per-design numpy integration, so the offline scripts call exactly the same `responses()` the scorer
calls, and a scalar-vs-vector drift is impossible by construction rather than by assertion.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from design_grid import (BUDGET, DESIRABILITY_SHAPE_M, EDIT_COST, LEVELS, RESPONSE_KEYS, SLOTS,
                         START, TOTAL_EDIT_COST, WINDOWS, constraints, edit_cost)

# achievement -> points. Restated in tasks/q1_agglomeration/task_config.yaml and asserted equal by
# scripts/check_consistency.py.
#
# BAND DENSITY, and why two bands were added below 20 points.
# The original ladder ran 9 / 20 / 33 / 47 / 63 / 80 / 100 at achievement thresholds
# 0.0001 / 0.15 / 0.30 / 0.45 / 0.60 / 0.75 / 0.90 -- even steps of 0.15 everywhere above 0.15, and
# then ONE band covering the whole interval (0, 0.15). Measured on the exhaustive enumeration, that
# single band held about 90 % of every feasible in-budget design that improves on the registered
# process at all, so the bottom of the scale had no resolution: a model that made one lucky
# single-variable move and a model that assembled a partly-working two-variable change both scored 9.
# External review flagged exactly this as "评分档位缺乏模型区分度".
#
# The fix keeps the 0.15 step and subdivides the bottom interval into three equal 0.05 steps, and
# subdivides 9 -> 20 points evenly to match:
#     >= 0.15  20      (unchanged)
#     >= 0.10  16      new
#     >= 0.05  12      new
#     >  0     9       (unchanged: "at least one usable improving process")
# Nothing at or above 20 points moved, so cross-problem comparability with the sibling problems in
# this series is preserved for every band they share; the refinement is strictly below the old floor
# and cannot raise any score that was previously >= 20. It also cannot inflate a score: the ceiling
# of every difficulty gate is unchanged, which scripts/enumerate_reference_best.py re-measures.
# 2026-08-15: the (0.0001, 9) band was split at 0.02, adding (0.02, 11). One row inserted; every
# other threshold and point value is unchanged.
#
# Why, and the reason is a property of the TASK rather than of any run: the bottom band held 58.4 %
# of all 19 779 improving designs, and the median improving design sat at achievement 0.0388 --
# inside it. A band containing half the scoring population carries almost no information; "9 points"
# meant little more than "improved at all". Splitting at 0.02, the population median OF THAT BAND,
# leaves 30.5 % below and 28.0 % above, so no band now holds more than a third.
#
# This is deliberately NOT justified by the observed scores. Four official runs landed at achievement
# 0.019 / 0.029 / 0.049 / 0.053 and three of them tied at 9 points, which is the symptom the review
# asked about (17-P1) -- but with a within-model SD of 19.1 points, five observations cannot tell a
# band-design fault from sampling. The exhaustive offline distribution can, and does.
POINT_BANDS: List[Tuple[float, int]] = [(0.90, 100), (0.75, 80), (0.60, 63), (0.45, 47),
                                        (0.30, 33), (0.15, 20), (0.10, 16), (0.05, 12),
                                        (0.02, 11), (0.0001, 9)]


@dataclass
class Desirability:
    """Four two-sided specification windows, weighted geometric mean.

    Per response, with plateau [pl, pu] and acceptable limits [al, au]:

        t = (pl - y) / (pl - al)      below the plateau
          = (y - pu) / (au - pu)      above the plateau
          = 0                         inside the plateau
        d = 1 / (1 + (C t)^m),  C = 3^(1/m)   so that d = 1.00 on the plateau,
                                              d = 0.25 exactly at the acceptable limit,
                                              and d decays smoothly, never reaching 0.
        D = exp( sum_i w_i ln d_i / sum_i w_i )

    Deliberately soft-edged. A hard zero outside the window would make almost every design in the
    neighbourhood score exactly 0, which turns the scoring surface into a wall instead of a slope:
    the difficulty would then come from an unreachable cliff rather than from the design problem, and
    "the score is suppressed on purpose" would be a fair criticism. With a soft edge, a partial
    repair earns partial credit and the 9/20/33/47 bands are populated -- while all four responses
    still have to be inside their windows at once to approach D = 1.

    Weights are (2, 2, 1, 1) for (d50, strength, porosity, yield): size and strength are the two
    attributes that decide whether the agglomerates are usable at all.
    """
    windows: Dict[str, Dict[str, float]]
    m: float = DESIRABILITY_SHAPE_M
    keys: Sequence[str] = tuple(RESPONSE_KEYS)

    def __post_init__(self) -> None:
        self.m = float(self.m)
        self._c = 3.0 ** (1.0 / self.m)
        missing = [k for k in self.keys if k not in self.windows]
        if missing:
            raise ValueError(f"windows missing responses: {missing}")

    def d_one(self, key: str, y: float) -> float:
        wd = self.windows[key]
        pl, pu, al, au = float(wd["pl"]), float(wd["pu"]), float(wd["al"]), float(wd["au"])
        if y < pl:
            t = (pl - y) / (pl - al)
        elif y > pu:
            t = (y - pu) / (au - pu)
        else:
            return 1.0
        return float(1.0 / (1.0 + (self._c * t) ** self.m))

    def d_of(self, resp: Dict[str, float]) -> Dict[str, float]:
        return {k: self.d_one(k, float(resp[k])) for k in self.keys}

    def D_of(self, resp: Dict[str, float]) -> float:
        num = 0.0
        den = 0.0
        for k in self.keys:
            w = float(self.windows[k]["w"])
            num += w * np.log(max(self.d_one(k, float(resp[k])), 1e-12))
            den += w
        return float(np.exp(num / den))


def default_desirability() -> Desirability:
    """The objective as defined by agglomeration.WINDOWS (the authority)."""
    return Desirability(WINDOWS)


def desirability_from_config(objective_cfg: Dict[str, Any]) -> Desirability:
    """The objective as stored in task_config.yaml. check_consistency.py asserts the two agree."""
    return Desirability(objective_cfg["windows"],
                        m=float(objective_cfg.get("desirability_shape_m", DESIRABILITY_SHAPE_M)))


# =================================================================================================
# achievement and points
# =================================================================================================
def achievement(D: float, D_start: float, D_ref: float) -> float:
    if D_ref <= D_start:
        return 0.0
    return float(np.clip((D - D_start) / (D_ref - D_start), 0.0, 1.0))


def points_for_achievement(a: float,
                           bands: Optional[Sequence[Tuple[float, int]]] = None) -> int:
    for thr, pts in (bands or POINT_BANDS):
        if a >= thr:
            return pts
    return 0


# =================================================================================================
# mechanical enumeration of the neighbourhood
# =================================================================================================
def to_design(row: Sequence[int]) -> Dict[str, Any]:
    return {s: LEVELS[s][int(row[i])] for i, s in enumerate(SLOTS)}


def to_row(d: Dict[str, Any]) -> np.ndarray:
    return np.array([LEVELS[s].index(d[s]) for s in SLOTS], dtype=np.int16)


def iter_k_slot_designs(k: int, reference: Optional[Dict[str, Any]] = None,
                        annotate: bool = True) -> Iterator[Tuple[Dict[str, Any], str]]:
    """Every design that differs from `reference` in exactly `k` slots, with an auto-generated
    "slot: from -> to" annotation. Generated by itertools; nothing is hand-listed."""
    ref = reference or START
    for combo in itertools.combinations(SLOTS, k):
        alts = [[v for v in LEVELS[s] if v != ref[s]] for s in combo]
        for pick in itertools.product(*alts):
            x = dict(ref)
            for s, v in zip(combo, pick):
                x[s] = v
            label = "; ".join(f"{s}: {ref[s]!r} -> {v!r}" for s, v in zip(combo, pick)) \
                if annotate else ""
            yield x, label


def iter_budget_designs(budget: int = BUDGET, reference: Optional[Dict[str, Any]] = None,
                        feasible_only: bool = True,
                        include_reference: bool = False) -> Iterator[Tuple[Dict[str, Any], str]]:
    """Every design whose weighted controlled-change cost is <= `budget`.

    Enumerated by slot subset: only subsets whose summed cost fits the budget are expanded, so this
    is exhaustive over the neighbourhood without touching the 6.5e8-design gross space.
    """
    ref = reference or START
    n = len(SLOTS)
    for r in range(0 if include_reference else 1, n + 1):
        for combo in itertools.combinations(range(n), r):
            names = [SLOTS[i] for i in combo]
            if sum(EDIT_COST[s] for s in names) > budget:
                continue
            alts = [[v for v in LEVELS[s] if v != ref[s]] for s in names]
            for pick in itertools.product(*alts):
                x = dict(ref)
                for s, v in zip(names, pick):
                    x[s] = v
                if feasible_only and constraints(x):
                    continue
                label = "; ".join(f"{s}: {ref[s]!r} -> {v!r}" for s, v in zip(names, pick))
                yield x, label


def count_budget_designs(budget: int = BUDGET,
                         reference: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Size of the neighbourhood before and after the process constraints (C3-C7)."""
    gross = 0
    feasible = 0
    for x, _ in iter_budget_designs(budget, reference, feasible_only=False):
        gross += 1
        if not constraints(x):
            feasible += 1
    return {"gross": gross, "feasible": feasible}


def overlap_floor(budget: int = BUDGET) -> float:
    """The change budget expressed as the minimum surviving fraction of the registered process."""
    return 1.0 - budget / float(TOTAL_EDIT_COST)


def g2_cost_criterion(budget: int = BUDGET, margin: float = 0.05) -> int:
    """G2 scans designs that are comfortably inside the budget: overlap >= floor + margin, i.e.

        cost <= 22 * (budget/22 - margin)

    floored to an integer because the cost is integral. With budget 6 and margin 0.05 this is
    cost <= 4. Stated here once so the gate script and the audit cannot use different criteria.
    """
    return int(np.floor(TOTAL_EDIT_COST * (budget / float(TOTAL_EDIT_COST) - margin) + 1e-12))

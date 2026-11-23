"""Vectorised design handling, window calibration, the smooth desirability objective, score bands.

The objective is a SMOOTH two-sided desirability (class Desirability below): the piecewise-cubic
Derringer-Suich window this problem originally shipped with was exactly zero outside the window,
which turned the score surface into a cliff and put all three calibration models on 0. See the
class docstring for the measurement that forced the change.

The scalar path in formulation.py/evaluator.py is what grades a submission; the vectorised path
here is what the offline enumeration uses to find the reference optimum and to measure the
difficulty gates. They must agree exactly, so scripts/check_consistency.py asserts
max |vectorised oracle - scalar oracle| == 0 and comp_matrix == composition() on random samples.
Every earlier problem in this series that shipped a wrong window did so because a simplified stand-in
was used for one of the two paths.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from formulation import (API_PCT, CQA_START_SIDE, LEVELS, MIN_FILLER_PCT, SLOTS, START,
                         SUBSTANCES, composition)
# The mechanism and the ruler come THROUGH the boundary module, never from formulation.py.
# In the public package response_model is a stub and every one of these raises on use.
from response_model import Oracle, _quiet_matmul, feature_vector

NL = np.array([len(LEVELS[s]) for s in SLOTS])
SI = {s: i for i, s in enumerate(SLOTS)}
SUBI = {s: i for i, s in enumerate(SUBSTANCES)}
START_IDX = np.array([LEVELS[s].index(START[s]) for s in SLOTS], dtype=np.int16)

# achievement -> points. Restated in tasks/*/task_config.yaml and asserted equal by
# scripts/check_consistency.py.
#
# RULER VERSION 2.0.0 (2026-07-29, audit findings #3 and #4). The 1.x table had seven bands and put
# two of the three calibration models in the same bottom band -- the reviewer's "区分度不足" finding.
# This table has sixteen and is deliberately dense between achievement 0 and 0.25, which is where
# measured models actually land, so runs that differ by a factor of 100 in achievement no longer
# collide.
#
# The lowest band worth >= 50 points sits at achievement 0.60 ON PURPOSE, and that placement is
# MEASURED. The sponsor's acceptance criterion is that no offline shortcut family and no calibration
# model may reach 50/100, and the boundary is set above the highest shortcut ceiling the designer-side
# probes found. Which families were probed and what they reach is recorded in the GRADER package only
# (审核说明.md sections 3.1 and 4, answers/naive_answer_probes.json) -- this module ships on the
# model-visible side, so naming the strongest shortcut here would be a strategy hint.
# Do not lower the boundary without re-running those probes.
POINT_BANDS: List[Tuple[float, int]] = [(0.90, 100), (0.80, 88), (0.70, 76), (0.60, 62),
                                        (0.45, 44), (0.36, 36), (0.29, 30), (0.22, 25),
                                        (0.16, 21), (0.115, 17), (0.08, 14), (0.055, 11),
                                        (0.035, 8), (0.02, 6), (0.008, 4), (0.0001, 2)]


def to_design(row: Sequence[int]) -> Dict[str, Any]:
    return {s: LEVELS[s][int(row[i])] for i, s in enumerate(SLOTS)}


def to_row(d: Dict[str, Any]) -> np.ndarray:
    return np.array([LEVELS[s].index(d[s]) for s in SLOTS], dtype=np.int16)


# ---------------------------------------------------------------------------------------------
# vectorised composition / budget
# ---------------------------------------------------------------------------------------------
def comp_matrix(R: np.ndarray) -> np.ndarray:
    """(n, 12) level indices -> (n, n_substances) mass %, DCPA q.s."""
    R = np.atleast_2d(R)
    W = np.zeros((len(R), len(SUBSTANCES)))
    W[:, SUBI['API']] = API_PCT
    W[:, SUBI['MCC']] = np.array(LEVELS['T2_mcc_pct'])[R[:, SI['T2_mcc_pct']]]
    aid = R[:, SI['T3_acid_id']]
    apc = np.array(LEVELS['T4_acid_pct'])[R[:, SI['T4_acid_pct']]]
    for k, nm in enumerate(LEVELS['T3_acid_id']):
        if nm == 'none':
            continue
        W[:, SUBI[nm]] = np.where(aid == k, apc, 0.0)
    W[:, SUBI['MEG']] = np.array(LEVELS['T5_meg_pct'])[R[:, SI['T5_meg_pct']]]
    did = R[:, SI['T6_dis_id']]
    dpc = np.array(LEVELS['T7_dis_pct'])[R[:, SI['T7_dis_pct']]]
    for k, nm in enumerate(LEVELS['T6_dis_id']):
        W[:, SUBI[nm]] = np.where(did == k, dpc, 0.0)
    W[:, SUBI['MGST']] = np.array(LEVELS['T8_mgst_pct'])[R[:, SI['T8_mgst_pct']]]
    W[:, SUBI['SIL']] = np.array(LEVELS['T10_sil_pct'])[R[:, SI['T10_sil_pct']]]
    W[:, SUBI['MAN']] = np.array(LEVELS['T12_man_pct'])[R[:, SI['T12_man_pct']]]
    W[:, SUBI['DCPA']] = 100.0 - W.sum(1)
    return W


_W0 = comp_matrix(START_IDX[None, :])[0]


def overlap_vec(R: np.ndarray) -> np.ndarray:
    W = comp_matrix(R)
    return np.minimum(W, _W0).sum(1) / np.maximum(W, _W0).sum(1)


def acid_ok_vec(R: np.ndarray) -> np.ndarray:
    isnone = R[:, SI['T3_acid_id']] == LEVELS['T3_acid_id'].index('none')
    zero = R[:, SI['T4_acid_pct']] == LEVELS['T4_acid_pct'].index(0.)
    return isnone == zero


def n_changed_vec(R: np.ndarray) -> np.ndarray:
    return (R != START_IDX[None, :]).sum(1)


def n_identity_vec(R: np.ndarray) -> np.ndarray:
    n = (R[:, SI['T3_acid_id']] != START_IDX[SI['T3_acid_id']]).astype(int)
    n += (R[:, SI['T6_dis_id']] != START_IDX[SI['T6_dis_id']]).astype(int)
    n += (R[:, SI['T12_man_pct']] != LEVELS['T12_man_pct'].index(0.)).astype(int)
    return n


def filler_pct_vec(R: np.ndarray) -> np.ndarray:
    return comp_matrix(R)[:, SUBI['DCPA']]


def feat_matrix(R: np.ndarray) -> np.ndarray:
    """Vectorised twin of formulation.feature_vector (asserted identical in check_consistency)."""
    R = np.atleast_2d(R)
    W = comp_matrix(R)
    X = np.zeros((len(R), 22))
    c = 0
    for k in range(len(LEVELS['T1_mcc_grade'])):
        X[:, c] = (R[:, SI['T1_mcc_grade']] == k); c += 1
    for k in range(len(LEVELS['T3_acid_id'])):
        X[:, c] = (R[:, SI['T3_acid_id']] == k); c += 1
    for k in range(len(LEVELS['T6_dis_id'])):
        X[:, c] = (R[:, SI['T6_dis_id']] == k); c += 1
    X[:, c] = np.array(LEVELS['T2_mcc_pct'])[R[:, SI['T2_mcc_pct']]]; c += 1
    X[:, c] = np.array(LEVELS['T4_acid_pct'])[R[:, SI['T4_acid_pct']]]; c += 1
    X[:, c] = np.array(LEVELS['T5_meg_pct'])[R[:, SI['T5_meg_pct']]]; c += 1
    X[:, c] = np.array(LEVELS['T7_dis_pct'])[R[:, SI['T7_dis_pct']]]; c += 1
    X[:, c] = np.array(LEVELS['T8_mgst_pct'])[R[:, SI['T8_mgst_pct']]]; c += 1
    X[:, c] = np.log(np.array(LEVELS['T9_blend_min'])[R[:, SI['T9_blend_min']]]); c += 1
    X[:, c] = np.array(LEVELS['T10_sil_pct'])[R[:, SI['T10_sil_pct']]]; c += 1
    X[:, c] = np.array(LEVELS['T11_press_MPa'])[R[:, SI['T11_press_MPa']]] / 100.0; c += 1
    X[:, c] = np.array(LEVELS['T12_man_pct'])[R[:, SI['T12_man_pct']]]; c += 1
    X[:, c] = W[:, SUBI['DCPA']]; c += 1
    assert c == 22
    return X


class VectorOracle(Oracle):
    def predict_rows(self, R: np.ndarray, chunk: int = 200000) -> np.ndarray:
        R = np.atleast_2d(R)
        out = np.empty((len(R), 5))
        for a in range(0, len(R), chunk):
            X = feat_matrix(R[a:a + chunk])
            out[a:a + chunk] = _quiet_matmul(lambda: self.predict_matrix(X))
        return out


def enumerate_rows(max_change: int = 4) -> np.ndarray:
    """Every design within `max_change` slots of the baseline formulation that is
    acid-consistent, mass-closable (filler >= 10 %) and uses at most one identity change.

    This is the FULL change-budget neighbourhood, before the composition-overlap floor is applied,
    so the same array serves both the reference search and the floor-sensitivity sweep.
    """
    alts = [[k for k in range(NL[i]) if k != START_IDX[i]] for i in range(len(SLOTS))]
    chunks = []
    for k in range(0, max_change + 1):
        if k == 0:
            chunks.append(START_IDX[None, :].copy())
            continue
        for combo in itertools.combinations(range(len(SLOTS)), k):
            mesh = np.array(list(itertools.product(*[alts[i] for i in combo])), dtype=np.int16)
            block = np.repeat(START_IDX[None, :], len(mesh), 0)
            block[:, list(combo)] = mesh
            chunks.append(block)
    R = np.vstack(chunks)
    ok = acid_ok_vec(R) & (filler_pct_vec(R) >= MIN_FILLER_PCT) & (n_identity_vec(R) <= 1)
    return R[ok]


def enumerate_exact_k(k: int) -> np.ndarray:
    """All designs with EXACTLY k slots changed, with no structural filtering.

    Deliberately a separate code path from enumerate_rows: the G1 gate is measured with this and
    cross-checked against a filter on enumerate_rows, so a bug in either one shows up as a
    disagreement instead of as a quietly optimistic gate.
    """
    alts = [[q for q in range(NL[i]) if q != START_IDX[i]] for i in range(len(SLOTS))]
    rows = []
    for combo in itertools.combinations(range(len(SLOTS)), k):
        for vals in itertools.product(*[alts[i] for i in combo]):
            r = START_IDX.copy()
            for i, v in zip(combo, vals):
                r[i] = v
            rows.append(r)
    return np.array(rows, dtype=np.int16)


def describe_row(row: Sequence[int]) -> str:
    """Auto-annotate a row as 'slot start->new, ...'. No hand-written move list anywhere."""
    parts = ['%s %s->%s' % (SLOTS[i], START[SLOTS[i]], LEVELS[SLOTS[i]][int(row[i])])
             for i in range(len(SLOTS)) if int(row[i]) != START_IDX[i]]
    return ', '.join(parts) or '(baseline formulation)'


# ---------------------------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------------------------
@dataclass
class Desirability:
    """SMOOTH asymmetric two-sided desirability, weighted geometric mean over the five windows.

        u_i(y) = (T_i - y)/(T_i - L_i)      for y <= T_i     # 0 at the target, 1 at the spec limit
               = (y - T_i)/(U_i - T_i)      for y >  T_i     # asymmetric: the halves differ
        v_i    = max(0, (u_i - pw)/(1 - pw))                 # d = 1 on a plateau of half-width pw
        h(v)   = 1 / (1 + (a v)**M)                   v <= 1,  a = ((1 - e)/e)**(1/M)
               = e * exp(-kappa * M * (1 - e) * (v - 1))   v >  1
        d_i    = floor + (1 - floor) * h(v_i)
        D      = exp( sum_i w_i ln d_i / sum_i w_i )

    Why this replaces the piecewise-cubic Derringer-Suich form this problem shipped with
    -------------------------------------------------------------------------------------
    The classical form is ((y-L)/(T-L))**r inside the window and EXACTLY ZERO outside it, so D
    collapses to 0 the instant any one of the five CQAs crosses a limit. Measured on this instance
    that made the score surface a cliff rather than a slope, and it put all three calibration models
    on 0 with the scoring bands never activated. The population statistics behind that decision are
    in the grader package (审核说明.md section 10); they are not repeated here.

    Here d_i is 1 at the target, decays smoothly through `e` at the printed specification limit and
    keeps decaying exponentially outside it down to `floor`. It is continuous and strictly positive
    everywhere, so dD/dy is non-zero everywhere: a design that misses one limit narrowly is ranked
    below an otherwise identical design that meets it, instead of being annihilated. The tail is
    steep on purpose (`kappa` > 1 multiplies the natural C1 log-slope of the Hill ramp at the
    limit): a floor high enough to be reached by a badly out-of-specification CQA would make
    "sacrifice one CQA and perfect the other four" the global optimum, which is neither the
    pharmaceutical intent nor a defensible reference design. With the shipped calibration the
    argmax over the change-budget neighbourhood has all five CQAs inside their windows.

    M, e, kappa, pw and the weights are score-curve parameters: they change how much credit partial
    progress earns. They do NOT change the ruler's monotonicity in any single CQA -- every window is
    still two-sided, so no CQA is "more is better".
    """
    L: np.ndarray
    T: np.ndarray
    U: np.ndarray
    hill_exponent: float = 6.3
    d_at_spec_limit: float = 2.0e-4
    tail_decay_multiplier: float = 2.6
    floor: float = 1.0e-6
    plateau_fraction: float = 0.0
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        self.L = np.asarray(self.L, dtype=float)
        self.T = np.asarray(self.T, dtype=float)
        self.U = np.asarray(self.U, dtype=float)
        self.M = float(self.hill_exponent)
        self.e = float(self.d_at_spec_limit)
        self.kappa = float(self.tail_decay_multiplier)
        self.dfloor = float(self.floor)
        self.pw = float(self.plateau_fraction)
        self.w = (np.ones(len(self.L)) if self.weights is None
                  else np.asarray(self.weights, dtype=float))
        if not (0.0 < self.e < 1.0):
            raise ValueError("d_at_spec_limit must be in (0, 1)")
        if not (0.0 <= self.dfloor < self.e):
            raise ValueError("floor must be in [0, d_at_spec_limit)")
        if not (0.0 <= self.pw < 1.0):
            raise ValueError("plateau_fraction must be in [0, 1)")
        if (self.L >= self.T).any() or (self.T >= self.U).any():
            raise ValueError("windows must satisfy L < T < U")
        # a puts the Hill ramp through d = e exactly at the printed specification limit; k is the
        # tail rate, kappa x the ramp's own log-slope there (kappa = 1 makes the join C1).
        self.a = ((1.0 - self.e) / self.e) ** (1.0 / self.M)
        self.k = self.kappa * self.M * (1.0 - self.e)

    def params(self) -> Dict[str, Any]:
        return {"hill_exponent": self.M, "d_at_spec_limit": self.e,
                "tail_decay_multiplier": self.kappa, "floor": self.dfloor,
                "plateau_fraction": self.pw, "weights": self.w.tolist()}

    def u_of(self, Y: np.ndarray) -> np.ndarray:
        """Distance from the target in half-window units: 0 at T, 1 at the nearer spec limit."""
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        return np.where(Y <= self.T, (self.T - Y) / (self.T - self.L),
                        (Y - self.T) / (self.U - self.T))

    def d_of(self, Y: np.ndarray) -> np.ndarray:
        v = np.clip((self.u_of(Y) - self.pw) / (1.0 - self.pw), 0.0, None)
        inside = 1.0 / (1.0 + (self.a * np.minimum(v, 1.0)) ** self.M)
        shape = np.where(v <= 1.0, inside,
                         self.e * np.exp(-self.k * np.clip(v - 1.0, 0.0, None)))
        return self.dfloor + (1.0 - self.dfloor) * shape

    def d_at_u(self, u: float) -> float:
        """d as a function of the normalised distance u -- the ruler's shape, for the audit trail."""
        v = max(0.0, (float(u) - self.pw) / (1.0 - self.pw))
        shape = (1.0 / (1.0 + (self.a * v) ** self.M) if v <= 1.0
                 else self.e * float(np.exp(-self.k * (v - 1.0))))
        return self.dfloor + (1.0 - self.dfloor) * shape

    def D_of(self, Y: np.ndarray) -> np.ndarray:
        d = self.d_of(Y)
        return np.exp((np.log(d) * self.w).sum(axis=1) / self.w.sum())

    def D_one(self, y: np.ndarray) -> float:
        return float(self.D_of(np.asarray(y, dtype=float)[None, :])[0])

    def in_spec(self, Y: np.ndarray) -> np.ndarray:
        """All five CQAs strictly inside their printed [L, U] windows. Reported as a fairness
        statistic; it is NOT the scoring gate any more (that is D > D_baseline)."""
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        return ((Y > self.L) & (Y < self.U)).all(axis=1)


# Decimal places each specification limit is reported to. The windows are ROUNDED to these before
# they become the ruler, so the numbers printed in question.md are the exact numbers the scorer uses.
# Without this the scorer would run on 16-digit floats while the question showed a rounded copy, and
# the model would be graded against a specification it was never given.
WINDOW_DECIMALS = np.array([2, 2, 2, 2, 1])


def calibrate_windows(y_start: np.ndarray, Y_neighbourhood: np.ndarray,
                      pull: float, marg: float, round_to: np.ndarray | None = None
                      ) -> Dict[str, np.ndarray]:
    """Place the five specification windows from two calibration numbers.

    `span_i` is the largest improvement in CQA i that the change-budget neighbourhood can reach at
    all. The target T_i is placed `pull` of the way along that span, and the dangerous-side bound is
    put `marg * pull * span_i` beyond the baseline formulation's value, so the baseline
    formulation is inside every window but close to one edge -- which is exactly the situation the
    quality group describes in the task.

    Only `pull` and `marg` are free; everything else follows from the neighbourhood, so a
    calibration cannot be tuned per CQA to manufacture a score.
    """
    y0 = np.asarray(y_start, dtype=float)
    span = ((np.asarray(Y_neighbourhood, dtype=float) - y0) * CQA_START_SIDE).max(0)
    T = y0 + CQA_START_SIDE * pull * span
    dlt = pull * span
    m = marg * dlt
    U = np.where(CQA_START_SIDE < 0, y0 + m, T + (dlt + m))
    L = np.where(CQA_START_SIDE < 0, T - (dlt + m), y0 - m)
    dec = WINDOW_DECIMALS if round_to is None else np.asarray(round_to)
    L = np.array([round(float(v), int(k)) for v, k in zip(L, dec)])
    T = np.array([round(float(v), int(k)) for v, k in zip(T, dec)])
    U = np.array([round(float(v), int(k)) for v, k in zip(U, dec)])
    assert (L < T).all() and (T < U).all(), "rounding collapsed a window"
    return {"L": L, "T": T, "U": U, "span": span, "decimals": dec}


# =============================================================================================
# THE ELIGIBILITY PREDICATES -- single source of truth for the vectorised (designer) side.
#
# Audit finding #3: "current rules can record a formulation as a benchmark success even though a
# CQA is outside its specification, because its aggregate desirability exceeds the starting point."
# That is now impossible. Three nested sets, in this order, and every gate measurement, the
# reference optimum and the score all use the SAME three functions:
#
#   budget_mask      the PACMP change budget only            (a design-space fact)
#   compliance_mask  all five CQAs strictly inside [L, U]    (the release / quality gate)
#   qualifying_mask  budget AND compliance AND D > D_start   (the scoring-eligible set)
#
# The scalar twins used to grade a submitted formulation are evaluator.satisfies_all_constraints,
# FormulationEvaluation.all_cqa_in_spec and scoring.is_qualifying; check_consistency.py asserts the
# vectorised and scalar answers agree design-by-design on random samples.
# =============================================================================================
def budget_mask(R: np.ndarray, overlap_floor: float, max_change: int = 4,
                max_identity: int = 1, min_filler_pct: float = MIN_FILLER_PCT) -> np.ndarray:
    """PACMP change budget: overlap floor, <= max_change slots, <= max_identity identity changes,
    mass closure. Acid consistency is already enforced by enumerate_rows."""
    return ((overlap_vec(R) >= overlap_floor - 1e-12)
            & (n_changed_vec(R) <= max_change)
            & (n_identity_vec(R) <= max_identity)
            & (filler_pct_vec(R) >= min_filler_pct))


def compliance_mask(Y: np.ndarray, sc: "Desirability") -> np.ndarray:
    """All five CQAs strictly inside their printed [L, U] windows. THE release gate: a formulation
    that fails this cannot be released, so it can never be a benchmark success and can never earn a
    point, no matter how high its aggregate desirability is."""
    return sc.in_spec(Y)


def qualifying_mask(R: np.ndarray, Y: np.ndarray, sc: "Desirability", start_value: float,
                    overlap_floor: float, max_change: int = 4, max_identity: int = 1,
                    min_filler_pct: float = MIN_FILLER_PCT) -> np.ndarray:
    """The scoring-eligible set: inside the change budget, fully compliant, and strictly better
    than the starting formulation. Nothing outside this set is ever scored or counted a success."""
    return (budget_mask(R, overlap_floor, max_change, max_identity, min_filler_pct)
            & compliance_mask(Y, sc)
            & (sc.D_of(Y) > start_value))


def achievement(value: float | None, start_value: float, reference_best: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    denom = reference_best - start_value
    if denom == 0:
        return 1.0
    return float(min(max((value - start_value) / denom, 0.0), 1.0))


def points_for(value: float | None, start_value: float, reference_best: float,
               bands: Sequence[Tuple[float, int]] = tuple(POINT_BANDS)) -> int:
    a = achievement(value, start_value, reference_best)
    for thr, pts in bands:
        if a >= thr:
            return int(pts)
    return 0

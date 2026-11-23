"""Design space and mass closure. PUBLIC SIDE -- no mechanism, no weights, no ruler.

=================================================================================================
DATA NATURE. This benchmark instance is SYNTHETIC.
=================================================================================================
There is no registered product, no development data set, no analytical measurement and no
regulatory dossier behind any number in this package. The API, the excipient behaviour, the five
CQA response equations and the starting formulation are all invented for this benchmark. Scores
measure search and formulation-reasoning behaviour on a synthetic surface; they are NOT evidence
about real formulation-development competence. Full statement: data/DATA_CARD.md.

What is in this file, and what deliberately is not
--------------------------------------------------
HERE (public):  the 12 declared conditions and their level grids (SLOTS / LEVELS), the starting
                formulation (START), the substance list and labels, mass closure with the q.s.
                filler (composition / composition_mg), the mass-weighted composition overlap
                (ruzicka_overlap), the change-budget bookkeeping (n_slots_changed, changed_slots,
                n_identity_changes), design-space membership checking (normalise_design), and the
                CQA key names and reporting order. All of it is restated in question.md, so none of
                it is secret -- it is the contract, not the answer.

NOT HERE:       the per-substance property table, the closed-form CQA simulator, the fitted
                surrogate that is the scoring ruler, and the weights file. Those live in
                src/response_model.py, which ships ONLY in the grader package. See that module's
                docstring for why: an agent that can evaluate candidates offline does not need the
                reference value to win, so the response surface has to be behind the same boundary
                as the reference answer.

This module therefore imports nothing from response_model, and must keep it that way -- the
dependency runs one way only, and scripts/package_for_delivery.py asserts it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(_SRC_DIR)

CORE_TABLET_MASS_MG = 200.0
API_PCT = 2.5                     # 5 mg API in a 200 mg core
# Mass-closure floor on the q.s. filler. NOTE: with the level grids below, the non-filler components
# sum to at most 76.6 % (API 2.5 + MCC 49 + acid 4 + meglumine 2 + disintegrant 7 + MgSt 1.5 +
# silica 0.6 + mannitol 10), so the filler can never fall below 23.4 % and this floor NEVER binds. It
# is kept as a defensive check -- it is the constraint that would catch a widened grid or a bug in
# composition() -- but the question text must not present it as a live design pressure, and no gate
# measurement may rely on it.
MIN_FILLER_PCT = 10.0

# ---------------------------------------------------------------------------------------------
# Material properties (hidden).
#   emc   equilibrium moisture content of the pure component at 25 C / 60 % RH (% w/w)
#   acid  effective acid capacity  (arbitrary mEq/g; "effective" = capacity x solubility)
#   base  effective base capacity
#   plast plastic-bonding index
#   ssa   specific surface area (m2/g)
#   d50   median particle size (um)
#   hyd   hydrophobicity / wetting retardation
#   dp    disintegration driving power
# ---------------------------------------------------------------------------------------------
SUBSTANCES: List[str] = ['API', 'DCPA', 'MAN', 'MCC', 'FUM', 'TAR', 'CIT', 'MEG',
                         'XPVP', 'CCS', 'LHPC', 'SSG', 'MGST', 'SIL']

SUBSTANCE_LABELS = {
    'API': 'API (pH-labile amine ester salt)',
    'DCPA': 'dibasic calcium phosphate anhydrous (q.s. filler)',
    'MAN': 'mannitol Pearlitol 100SD',
    'MCC': 'microcrystalline cellulose',
    'FUM': 'fumaric acid',
    'TAR': 'tartaric acid',
    'CIT': 'citric acid anhydrous',
    'MEG': 'meglumine',
    'XPVP': 'crospovidone',
    'CCS': 'croscarmellose sodium',
    'LHPC': 'low-substituted hydroxypropyl cellulose',
    'SSG': 'sodium starch glycolate',
    'MGST': 'magnesium stearate',
    'SIL': 'colloidal silicon dioxide',
}

# ---------------------------------------------------------------------------------------------
# Design space: 12 slots. T3/T6/T12 can introduce a new excipient identity.
# ---------------------------------------------------------------------------------------------
SLOTS: List[str] = ['T1_mcc_grade', 'T2_mcc_pct', 'T3_acid_id', 'T4_acid_pct', 'T5_meg_pct',
                    'T6_dis_id', 'T7_dis_pct', 'T8_mgst_pct', 'T9_blend_min', 'T10_sil_pct',
                    'T11_press_MPa', 'T12_man_pct']

LEVELS: Dict[str, List[Any]] = {
    'T1_mcc_grade':  ['PH101', 'PH102', 'PH200', 'PH301'],
    'T2_mcc_pct':    [28., 31., 34., 37., 40., 43., 46., 49.],
    'T3_acid_id':    ['none', 'FUM', 'TAR', 'CIT'],
    'T4_acid_pct':   [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    'T5_meg_pct':    [0., 0.25, 0.5, 1.0, 2.0],
    'T6_dis_id':     ['XPVP', 'CCS', 'LHPC', 'SSG'],
    'T7_dis_pct':    [1., 2., 3., 4., 5., 6., 7.],
    'T8_mgst_pct':   [0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5],
    'T9_blend_min':  [2., 3., 5., 8., 10., 15., 20.],
    'T10_sil_pct':   [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'T11_press_MPa': [100., 120., 140., 160., 180., 200., 220., 240., 260.],
    'T12_man_pct':   [0., 2., 4., 6., 8., 10.],
}

SLOT_UNITS = {
    'T1_mcc_grade':  'MCC grade',
    'T2_mcc_pct':    '% of core weight',
    'T3_acid_id':    'acidifier identity',
    'T4_acid_pct':   '% of core weight',
    'T5_meg_pct':    '% of core weight',
    'T6_dis_id':     'superdisintegrant identity',
    'T7_dis_pct':    '% of core weight',
    'T8_mgst_pct':   '% of core weight',
    'T9_blend_min':  'min',
    'T10_sil_pct':   '% of core weight',
    'T11_press_MPa': 'MPa',
    'T12_man_pct':   '% of core weight',
}

# The baseline formulation. Every CQA sits on the dangerous edge of its window here.
START: Dict[str, Any] = {
    'T1_mcc_grade': 'PH102', 'T2_mcc_pct': 40., 'T3_acid_id': 'CIT', 'T4_acid_pct': 1.,
    'T5_meg_pct': 0., 'T6_dis_id': 'CCS', 'T7_dis_pct': 6., 'T8_mgst_pct': 1.0,
    'T9_blend_min': 10., 'T10_sil_pct': 0.2, 'T11_press_MPa': 180., 'T12_man_pct': 0.,
}

# Slots that add or swap an excipient IDENTITY (T12 > 0 introduces mannitol, which is absent
# from the baseline formulation).
IDENTITY_SLOTS = ['T3_acid_id', 'T6_dis_id', 'T12_man_pct']

CQA_KEYS = ['microenvironment_pH', 'emc_60rh_pct', 'tensile_strength_MPa', 'ffc', 'q30_pct']
CQA_SHORT = ['pH', 'EMC', 'sigma_t', 'ffc', 'Q30']
# Direction that improves each CQA starting from the baseline formulation. Used only to place
# the two-sided windows during calibration; the score itself is direction-free (both sides fail).
CQA_START_SIDE = np.array([-1., -1., 1., 1., 1.])

PROCESS_SLOTS = ['T9_blend_min', 'T11_press_MPa']   # do not change the composition vector


class DesignError(ValueError):
    """A submitted design is not a member of the declared design space."""


# ---------------------------------------------------------------------------------------------
# design-space membership
# ---------------------------------------------------------------------------------------------
def _match_level(slot: str, value: Any) -> Any:
    """Snap a submitted value onto its declared level, or raise.

    Numeric levels are matched with a tolerance so that 1 vs 1.0 vs "1.0" all resolve, but a value
    that is not a declared level is an error rather than a silent nearest-neighbour substitution:
    the task says the grid IS the constraint.
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
            return float(lv)
    raise DesignError(f"{slot}={value!r} is not a declared level; allowed: {levels}")


def normalise_design(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + canonicalise a submitted design. Raises DesignError on anything off-grid."""
    if not isinstance(raw, dict):
        raise DesignError("design must be a JSON object keyed by slot name")
    missing = [s for s in SLOTS if s not in raw]
    if missing:
        raise DesignError(f"missing slots: {missing}")
    extra = [k for k in raw if k not in SLOTS]
    if extra:
        raise DesignError(f"unknown slots (no new excipients or variables may be introduced): {extra}")
    return {s: _match_level(s, raw[s]) for s in SLOTS}


def acid_consistent(d: Dict[str, Any]) -> bool:
    """'no acidifier' and '0 %' must agree -- naming an acid at 0 % is not a formulation."""
    return (d['T3_acid_id'] == 'none') == (float(d['T4_acid_pct']) == 0.)


def design_space_size() -> Dict[str, int]:
    n = 1
    for s in SLOTS:
        n *= len(LEVELS[s])
    other = n // (len(LEVELS['T3_acid_id']) * len(LEVELS['T4_acid_pct']))
    good_pairs = 1 + (len(LEVELS['T3_acid_id']) - 1) * (len(LEVELS['T4_acid_pct']) - 1)
    return {"gross": n, "acid_consistent": other * good_pairs}


# ---------------------------------------------------------------------------------------------
# mass closure
# ---------------------------------------------------------------------------------------------
def composition(d: Dict[str, Any]) -> Dict[str, float]:
    """Mass-% vector over SUBSTANCES with DCPA q.s. to 100 % of the 200 mg core."""
    w = {s: 0.0 for s in SUBSTANCES}
    w['API'] = API_PCT
    w['MCC'] = float(d['T2_mcc_pct'])
    if d['T3_acid_id'] != 'none':
        w[d['T3_acid_id']] = float(d['T4_acid_pct'])
    w['MEG'] = float(d['T5_meg_pct'])
    w[d['T6_dis_id']] = float(d['T7_dis_pct'])
    w['MGST'] = float(d['T8_mgst_pct'])
    w['SIL'] = float(d['T10_sil_pct'])
    w['MAN'] = float(d['T12_man_pct'])
    w['DCPA'] = 100.0 - sum(w.values())
    return w


def composition_mg(d: Dict[str, Any]) -> Dict[str, float]:
    return {k: round(v * CORE_TABLET_MASS_MG / 100.0, 4) for k, v in composition(d).items() if v}


def ruzicka_overlap(d: Dict[str, Any], reference: Dict[str, Any] | None = None) -> float:
    """Mass-weighted composition overlap (Ruzicka / weighted Jaccard) against the reference.

    Overlap = sum_i min(w_i, w0_i) / sum_i max(w_i, w0_i)

    Because a different excipient identity is a different component of the vector, swapping an
    excipient costs its whole mass on both sides -- identity changes are automatically expensive
    while re-dialling an amount is cheap in proportion to the amount moved. Process slots do not
    appear in the vector at all, so they cost nothing here (they still consume a change slot).
    """
    w0 = composition(reference if reference is not None else START)
    a = np.array([w0[s] for s in SUBSTANCES])
    b = np.array([composition(d)[s] for s in SUBSTANCES])
    return float(np.minimum(a, b).sum() / np.maximum(a, b).sum())


def n_slots_changed(d: Dict[str, Any], reference: Dict[str, Any] | None = None) -> int:
    ref = reference if reference is not None else START
    return sum(1 for s in SLOTS if d[s] != ref[s])


def changed_slots(d: Dict[str, Any], reference: Dict[str, Any] | None = None) -> List[str]:
    ref = reference if reference is not None else START
    return [s for s in SLOTS if d[s] != ref[s]]


def n_identity_changes(d: Dict[str, Any], reference: Dict[str, Any] | None = None) -> int:
    """Number of excipient IDENTITY changes vs the reference formulation."""
    ref = reference if reference is not None else START
    n = 0
    if d['T3_acid_id'] != ref['T3_acid_id']:
        n += 1
    if d['T6_dis_id'] != ref['T6_dis_id']:
        n += 1
    if float(d['T12_man_pct']) > 0 and float(ref['T12_man_pct']) == 0:
        n += 1
    return n



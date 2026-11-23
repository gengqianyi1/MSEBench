"""SYNTHETIC simulator of a quasi-emulsion spherical agglomeration process. Not experimental data.

READ THIS FIRST
---------------
Everything this module returns is **synthetic**. There is no registered product, no filed process, no
laboratory campaign and no measured attribute anywhere in this package. The "process" is a designed
scenario; the four "responses" are the output of a mechanistic simulator plus one surrogate network
that was fitted to that same simulator with synthetic heteroscedastic noise added at training time
(seed 20260726). The mechanistic coefficients were chosen by the author so that the four responses
land in engineering-plausible ranges and so that the optimisation problem is coupled; they were NOT
regressed against any real measurement, and no external dataset was used.

Consequences a reader must carry forward:
  * a score on this benchmark says something about coordinated multi-variable decision-making under a
    change budget with narrow-band numerical feedback. It says NOTHING about real spherical
    agglomeration performance, and no number produced here may be quoted as a property of any API.
  * the simulator is only defined on the twelve declared level grids. Off-grid input is refused rather
    than extrapolated.
  * evaluation is deterministic: no noise is added at evaluation time, so the same design always
    returns the same four numbers.

The full model card -- mechanism, coefficient table, the noise distribution used to build the
training set, fit statistics, applicability domain and limitations -- is data/MODEL_CARD.md. It is a
GRADER-SIDE document: it is deliberately not part of the material the evaluated model sees, because a
model that can read the response mechanism can optimise against the response surface instead of
solving the process problem.

This module is the ONE ruler. It holds the design space, the controlled-change cost table, the
starting (registered) process, the four response models and the hard constraints. Everything else in
the package -- the scorer, the offline enumeration, the gate measurements, the question text -- reads
its numbers from here or is asserted equal to them.

The four simulated responses of one batch are:

  d50        volume-median agglomerate diameter, um
             agglomeration-breakage population balance: Hounslow q=2 discretisation on 30 geometric
             volume bins, orthokinetic (shear) Smoluchowski coagulation kernel whose capture
             efficiency comes from bridging-liquid saturation, wetting and addition uniformity, plus
             an Austin-type breakage kernel (~ G^1.55 d^3.2 / sigma_wet^1.15), advanced to the
             process time with positivity-preserving exponential-Euler steps, then corrected for
             drying-induced cracking, residual-solvent lumping and handling abrasion, wet attrition
             and un-suspended bypass, and finally for wet classification.
  strength   simulated agglomerate crush strength, MPa
  porosity   simulated intra-agglomerate porosity, dimensionless
             Both read off ONE frozen surrogate (data/agglo_strength_porosity.npz): an
             MLP(128, 96, 64) fitted with seed 20260726 to 60 000 samples of the Rumpf-type
             mechanism below plus SYNTHETIC heteroscedastic noise (the noise emulates measurement
             scatter; it is not measurement scatter). Hold-out R^2 against the noise-free mechanism
             is 0.9993 (log sigma) and 0.9996 (log eps). The surrogate is the ruler; the mechanism
             function `dry_strength_porosity_mech` is kept because it is what the surrogate was
             fitted to and a reviewer must be able to see the two agree
             (scripts/check_consistency.py reports the drift).
  yield_pct  simulated unit yield, %  -- solubility mass balance in the bridging liquid and the
             continuous phase, wash loss, classification loss, attrition, HPMC adsorption loss,
             residual-solvent rework loss and over-wetted-lump entrainment.

Why a fitted surrogate at all: a response the evaluated model could reproduce in its head with a
textbook formula would let it compute its own score and optimise offline, which is what killed two
earlier problems in this series. A 200-step population balance and a fitted 3-layer network are not
pencil-and-paper objects.

Determinism: no RNG is touched at evaluation time. The population balance is a fixed-step-rule
explicit integrator (CFL 0.30, at most 240 steps) and the surrogate is a frozen array bundle, so the
same design always returns the same four numbers on any machine with IEEE-754 float64 and the same
BLAS rounding.

Load safety: the surrogate is stored as plain float64 arrays in a .npz opened with
`allow_pickle=False`, and the forward pass is the 8 lines of numpy in `surrogate_predict`. Nothing is
unpickled, so loading the ruler is not code execution and the package no longer depends on a
scikit-learn version pin. The migration from the original pickled sklearn Pipeline was verified to be
bit-identical (max |old - new| = 0.0 over the 60 000-row training matrix and every 1-slot and 2-slot
design); scripts/convert_surrogate_to_npz.py performs and re-measures it, and the record is in
data/agglo_strength_porosity.schema.json.

Provenance: ported unchanged (numerically identical, verified against its own recorded outputs) from
the prototype that measured the difficulty gates. The port renamed the module and moved the
surrogate into data/; no coefficient, level, cost or window was touched.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SURROGATE = os.path.join(HERE, os.pardir, "data", "agglo_strength_porosity.npz")
SURROGATE_SCHEMA = os.path.join(HERE, os.pardir, "data", "agglo_strength_porosity.schema.json")


# =================================================================================================
# PUBLIC half of the task definition, imported (not duplicated) from src/design_grid.py.
#
# The design space, the change-cost table, the baseline process, the specification windows and the
# C1/C3-C7 constraint predicates are PUBLIC: every one of them is printed in question.md, and the
# public package ships design_grid.py. They are re-exported here so that every module and script that
# already says `from agglomeration import LEVELS, ...` keeps working and there is still exactly ONE
# definition of each.
#
# What stays in THIS file is the part that must not be visible to an evaluated model: the response
# mechanism. See the module docstring.
# =================================================================================================
from design_grid import (  # noqa: F401  (re-exported on purpose)
    ADDITION_MINUTES, BP, BP_MARGIN_C, BUDGET, COARSE_CUT, DESIRABILITY_SHAPE_M, DRYING_HOURS,
    EDIT_COST, FINES_CUT, JS_COEFF, JS_EXPONENT, JS_REF_SOLIDS, LEVELS, MAX_PROCESS_HOURS,
    MAX_SOLVENT_L_PER_KG, RESPONSE_KEYS, RESPONSE_UNITS, SCHEME_RECYCLES_FINES,
    SCHEME_REMOVES_COARSE, SCHEME_REMOVES_FINES, SLOT_UNITS, SLOTS, START, TOTAL_EDIT_COST,
    WASH_L_PER_KG, WINDOWS, X0, DesignError, _match_level, change_overlap, changed_slots,
    constraints, edit_cost, just_suspended_tip_speed, normalise_design, space_size,
    total_hours, total_solvent_L_per_kg)

# =================================================================================================
# physical property tables
# =================================================================================================
# SYNTHETIC property tables. The PUBLIC field of each row (boiling point, addition duration, drying
# duration, which size fractions the scheme removes) is taken from src/design_grid.py rather than
# written again here, so there is one definition and the public constraint predicates and the private
# mechanism cannot drift apart. Every OTHER field is a response-model parameter: authored, frozen, and
# grader-side only.
BL = {   # bridging liquids: interfacial tension mN/m, wetting cos(theta), API solubility g/L at
         # 25 degC, mutual miscibility with the aqueous continuous phase. bp comes from design_grid.
    "heptane":            dict(bp=BP["heptane"],           gamma=24.0, wet=0.50, sol=1.2,  misc=0.02),
    "cyclohexane":        dict(bp=BP["cyclohexane"],       gamma=22.0, wet=0.58, sol=2.2,  misc=0.03),
    "heptane_EtOAc_1_1":  dict(bp=BP["heptane_EtOAc_1_1"], gamma=19.0, wet=0.72, sol=8.0,  misc=0.14),
    "toluene":            dict(bp=BP["toluene"],           gamma=20.0, wet=0.77, sol=11.0, misc=0.05),
    "MTBE":               dict(bp=BP["MTBE"],              gamma=16.0, wet=0.87, sol=16.0, misc=0.30),
    "EtOAc":              dict(bp=BP["EtOAc"],             gamma=14.5, wet=0.94, sol=22.0, misc=0.28),
}
ADD = {  # addition profile: distribution uniformity, relative nucleus count. t from design_grid.
    "one_shot": dict(t=ADDITION_MINUTES["one_shot"], u=0.70, nuc=1.55),
    "10_min":   dict(t=ADDITION_MINUTES["10_min"],   u=0.86, nuc=1.22),
    "30_min":   dict(t=ADDITION_MINUTES["30_min"],   u=0.95, nuc=1.00),
    "60_min":   dict(t=ADDITION_MINUTES["60_min"],   u=1.00, nuc=0.86),
    "90_min":   dict(t=ADDITION_MINUTES["90_min"],   u=1.00, nuc=0.80),
    "3_pulses": dict(t=ADDITION_MINUTES["3_pulses"], u=0.90, nuc=1.10),
}
DRY = {  # drying: degC, capillary shrinkage, thermal-stress cracking, residual solvent.
         # h from design_grid.
    "20C_N2_30h":      dict(h=DRYING_HOURS["20C_N2_30h"],      T=20, shrink=0.012, crack=0.000,
                            resid=0.165),
    "25C_N2_24h":      dict(h=DRYING_HOURS["25C_N2_24h"],      T=25, shrink=0.020, crack=0.000,
                            resid=0.100),
    "40C_vac_12h":     dict(h=DRYING_HOURS["40C_vac_12h"],     T=40, shrink=0.055, crack=0.015,
                            resid=0.030),
    "50C_vac_6h":      dict(h=DRYING_HOURS["50C_vac_6h"],      T=50, shrink=0.085, crack=0.055,
                            resid=0.014),
    "60C_vac_4h":      dict(h=DRYING_HOURS["60C_vac_4h"],      T=60, shrink=0.118, crack=0.140,
                            resid=0.007),
    "40C_stepped_10h": dict(h=DRYING_HOURS["40C_stepped_10h"], T=40, shrink=0.050, crack=0.024,
                            resid=0.020),
}
# Wet classification. f / c are the public booleans from design_grid; the recycled FRACTION of the
# removed fines is a response-model parameter and is not public.
_REC_FRACTION = {"none": 0.0, "remove_coarse": 0.0, "remove_fines": 0.0, "remove_both": 0.0,
                 "remove_both_recycle": 0.60, "remove_both_recycle_high": 0.85}
CLS = {s: dict(f=int(SCHEME_REMOVES_FINES[s]), c=int(SCHEME_REMOVES_COARSE[s]),
               rec=_REC_FRACTION[s])
       for s in LEVELS["S10_classification"]}
# Needle aspect ratio of the primary crystals: a slower upstream cooling ramp gives a coarser and
# blockier crystal, so the two are not independent.
AR_OF_DP = {5: 12.5, 9: 10.0, 12: 9.0, 20: 7.0, 32: 5.5}

RHO = 1.35          # g/mL true density

# Mechanistic coefficients: calibrated once against the engineering ranges, then FROZEN. The
# surrogate in data/ was fitted to the mechanism with exactly these values, so changing any of them
# invalidates the surrogate as well as every gate.
CF = dict(
    ar_eps0=0.0225,      # needle aspect ratio -> loose-packing porosity of the primary bed
    eps0_base=0.4000,
    dwet_max=0.480, dwet_k=1.20,
    sig_pre=1.22,        # Rumpf-type strength prefactor (MPa)
    kb=4.2e-6,           # breakage prefactor
    eta0=0.90, eta_scale=2.0e-3,   # collision -> permanent-bond capture efficiency
    brk_p=3.2,           # breakage size exponent (stress-balance-limited agglomerate size)
    attr=0.055,          # wet attrition -> fines
    u_visc=0.42,         # solids-loading penalty on bridging-liquid distribution
    rec_nuc=1.10, rec_sat=0.62,   # recycled fines: extra nuclei / bridging-liquid soak-up
    lump=1.60, hand=1.20,         # residual-solvent lumping / handling abrasion
    loss_handle=22.0,             # residual-solvent rework loss, % per unit residual

    bypass0=0.30, bypass_k=2.2,
    # granulation regime boundaries in bridging-liquid saturation
    # (pendular bonding onset / droplet i.e. over-wetted paste regime)
    pl_c=0.170, pl_w=0.022, ow_c=0.860, ow_w=0.035,
    ow_eps=0.42, ow_ball=3.0, ow_yield=9.0,
)
# FINES_CUT / COARSE_CUT are public (stated in question.md) and come from src/design_grid.py.


# =================================================================================================
# population-balance grid (built once at import; pure geometry, no design dependence)
# =================================================================================================
NBIN = 30
_BIN_V = 1.0 * (2.0 ** np.arange(NBIN))          # bin volumes, in multiples of one primary crystal
_W_LOW = np.zeros((NBIN, NBIN))
for _i in range(NBIN):
    for _j in range(NBIN):
        if _j <= _i - 1:
            _W_LOW[_i, _j] = 2.0 ** (_j - _i)
_MASK_UP = np.zeros((NBIN, NBIN))
for _i in range(NBIN):
    _MASK_UP[_i, _i:] = 1.0
_DSUM3 = (_BIN_V[:, None] ** (1 / 3) + _BIN_V[None, :] ** (1 / 3)) ** 3   # collision cross-section


# =================================================================================================
# intermediates
# =================================================================================================
def intermediates(x: Dict[str, Any]) -> Dict[str, Any]:
    bl = BL[x["S1_bridging_liquid"]]
    add = ADD[x["S3_addition_mode"]]
    dry = DRY[x["S11_drying"]]
    T = float(x["S6_temperature_C"])
    wfr = float(x["S9_water_fraction"])
    dp = float(x["S8_primary_d50_um"])
    ar = AR_OF_DP[x["S8_primary_d50_um"]]
    hp = float(x["S12_HPMC_ppm"])
    cov = hp / (hp + 250.0)                       # HPMC surface coverage

    cos_eff = bl["wet"] * (1.0 - 0.42 * (wfr - 0.55) / 0.30) * (1.0 - 0.34 * cov) \
        * (1.0 + 0.13 * (T - 25.0) / 20.0)
    cos_eff = float(np.clip(cos_eff, 0.05, 1.0))
    gam_eff = bl["gamma"] * (1.0 + 0.26 * (wfr - 0.55) / 0.30) * (1.0 - 0.30 * cov)

    # part of the bridging liquid partitions into the continuous phase and stops bridging
    phi_ret = 1.0 - bl["misc"] * (1.0 - wfr) * 2.30 * (1.0 + 0.55 * (T - 25.0) / 20.0) \
        * (1.0 + 0.30 * add["t"] / 60.0)
    phi_ret = float(np.clip(phi_ret, 0.20, 1.0))

    eps0 = CF["eps0_base"] + CF["ar_eps0"] * (ar - 5.0)   # loose packing porosity of the needle bed
    v_void0 = eps0 / (1.0 - eps0) / RHO           # mL void per g solid
    cls = CLS[x["S10_classification"]]
    # slurry viscosity at high solids degrades bridging-liquid distribution
    u_eff = add["u"] * (1.0 - CF["u_visc"] * float(np.clip(
        (float(x["S4_solids_g_per_L"]) - 110.0) / 200.0, 0.0, 1.0)) ** 0.85)
    v_br_eff = float(x["S2_bridging_ratio_v_w"]) * phi_ret * u_eff
    s_sat = v_br_eff / v_void0                    # bridging-liquid saturation of the void space
    # recycled fines re-enter as extra nuclei and soak up part of the bridging liquid
    nuc_eff = add["nuc"] * (1.0 + CF["rec_nuc"] * cls["rec"])
    s_sat = s_sat / (1.0 + CF["rec_sat"] * cls["rec"])

    G = float(x["S5_tip_speed_m_s"]) / 0.012      # shear rate 1/s (12 mm impeller gap)
    t_agg = float(x["S7_agglomeration_min"]) * 60.0

    p_c = gam_eff * cos_eff / dp                  # capillary consolidation pressure, mN/m per um
    w_shear = (float(x["S5_tip_speed_m_s"]) / 1.4) ** 0.90 \
        * (float(x["S7_agglomeration_min"]) / 45.0) ** 0.42
    f_s = min(s_sat, 1.15) ** 0.60
    # Newitt-Conway-Jones / Iveson saturation regimes: pendular bonding onset, over-wetted paste
    reg_pend = 1.0 / (1.0 + math.exp(-(s_sat - CF["pl_c"]) / CF["pl_w"]))
    reg_over = 1.0 / (1.0 + math.exp(-(s_sat - CF["ow_c"]) / CF["ow_w"]))
    d_wet = CF["dwet_max"] * math.tanh(CF["dwet_k"] * p_c ** 0.70 * w_shear * f_s)
    eps_wet = eps0 * (1.0 - d_wet) * (1.0 - CF["ow_eps"] * reg_over)

    d_dry = dry["shrink"] * (0.40 + 0.95 * min(s_sat, 1.0)) * (1.0 + 0.25 * (12.0 / dp - 1.0))
    eps_dry = eps_wet * (1.0 - max(d_dry, 0.0))

    # solid bridges: API dissolved by the bridging liquid recrystallises on drying
    sol_bl = bl["sol"] * math.exp(0.030 * (T - 25.0))
    solid_bridge = sol_bl * float(x["S2_bridging_ratio_v_w"]) / 12.0      # normalised

    # Zwietering-type just-suspended speed and the resulting suspension quality
    v_js = 0.42 * (float(x["S4_solids_g_per_L"]) / 40.0) ** 0.30
    susp = float(x["S5_tip_speed_m_s"]) / v_js
    bypass = float(np.clip(CF["bypass0"] * math.exp(-CF["bypass_k"] * (susp - 1.0)), 0.0, 0.35))

    return dict(v_js=v_js, susp=susp, bypass=bypass, u_eff=u_eff, nuc_eff=nuc_eff,
                reg_pend=reg_pend, reg_over=reg_over,
                bl=bl, add=add, dry=dry, cls=cls, T=T, wfr=wfr,
                dp=dp, ar=ar, cov=cov, cos_eff=cos_eff, gam_eff=gam_eff, phi_ret=phi_ret,
                eps0=eps0, s_sat=s_sat, G=G, t_agg=t_agg, p_c=p_c, w_shear=w_shear,
                eps_wet=eps_wet, eps_dry=eps_dry, sol_bl=sol_bl, solid_bridge=solid_bridge,
                solids=float(x["S4_solids_g_per_L"]))


def rumpf_strength(eps, gam_eff, cos_eff, dp, s_sat, solid_bridge, crack, resid, cov, K=1.0):
    """Calibrated Rumpf-type agglomerate crush strength, MPa.

    (1-eps)/eps coordination term x capillary + solid-bridge bond strength x saturation, degraded by
    thermal-stress cracking, residual solvent and HPMC at the bridge necks.
    """
    s_eff = min(s_sat, 1.0)
    sigma = K * CF["sig_pre"] * (1.0 - eps) / eps * (gam_eff * cos_eff / dp) * (s_eff ** 0.75)
    sigma *= (0.55 + 1.60 * solid_bridge)
    sigma *= (1.0 - crack) ** 1.25
    sigma *= (1.0 - 1.30 * resid)
    sigma *= (1.0 - 0.42 * cov)
    return max(sigma, 1e-3)


def wet_strength(it: Dict[str, Any]) -> float:
    """Strength of the wet agglomerate, which is what sets the breakage rate in the shear field."""
    return rumpf_strength(it["eps_wet"], it["gam_eff"], it["cos_eff"], it["dp"],
                          it["s_sat"], 0.30 * it["solid_bridge"], 0.0, 0.0, it["cov"])


def dry_strength_porosity_mech(it: Dict[str, Any]):
    """The mechanism the frozen surrogate was fitted to. Not the ruler -- the surrogate is."""
    sig = rumpf_strength(it["eps_dry"], it["gam_eff"], it["cos_eff"], it["dp"], it["s_sat"],
                         it["solid_bridge"], it["dry"]["crack"], it["dry"]["resid"], it["cov"])
    return sig, it["eps_dry"]


# =================================================================================================
# population balance
# =================================================================================================
def run_pbe(it: Dict[str, Any], CFL: float = 0.30, maxstep: int = 240):
    """Hounslow q=2 agglomeration + binary breakage, adaptive positivity-preserving explicit steps.

    CFL = 0.30 with at most 240 steps is part of the DEFINITION of the objective, not a tolerance to
    be tuned: d50 is 258.2 um at CFL 0.30 against 263.5 um at CFL 0.05 for the registered process
    (2 %), and the coarse setting was frozen so that a gate enumeration of 10^5 designs is affordable.
    """
    dp = it["dp"]
    sig_w = wet_strength(it)
    # capture efficiency of a collision (a permanent liquid bridge is formed)
    eta = CF["eta0"] * CF["eta_scale"] * (it["s_sat"] / (0.22 + it["s_sat"])) \
        * (it["cos_eff"] ** 1.45) * it["u_eff"] * it["reg_pend"] \
        * (1.0 + CF["ow_ball"] * it["reg_over"])
    # orthokinetic (shear) Smoluchowski kernel in SI, scaled to the dimensionless bin grid
    v_p = math.pi / 6.0 * (dp * 1e-6) ** 3                        # m^3 per primary crystal
    n0 = it["solids"] / (RHO * 1000.0) / v_p                      # primaries per m^3 slurry
    beta = (it["G"] / 6.0) * (dp * 1e-6) ** 3 * _DSUM3 * n0 * eta / it["nuc_eff"]   # 1/s

    Bl = _W_LOW * beta
    Bu = _MASK_UP * beta
    bdiag = np.diag(beta).copy()

    # breakage: Austin-type, grows with size and shear, falls with wet strength
    d_bin = dp * _BIN_V ** (1 / 3)
    S = CF["kb"] * (it["G"] ** 1.55) * (d_bin / 300.0) ** CF["brk_p"] / (sig_w ** 1.15)

    N = np.zeros(NBIN)
    N[0] = 1.0
    t = 0.0
    t_end = it["t_agg"]
    nsteps = 0
    while t < t_end and nsteps < maxstep:
        p = Bl @ N
        q = Bu @ N
        L = p + q + S                                   # per-particle loss rate
        m = N * _BIN_V
        rate_scale = float((m * L).sum() / m.sum() + 1e-30)   # mass-weighted loss rate
        dt = min(CFL / rate_scale, t_end - t)
        if dt <= 0:
            break
        birth = np.zeros(NBIN)
        core = N * p
        birth[1:] = core[:-1] + 0.5 * bdiag[:-1] * N[:-1] ** 2
        SN = S * N
        birth[:-1] += 2.0 * SN[1:]
        # exponential Euler on the linear loss term keeps N >= 0 at any step size
        Ld = np.maximum(L, 1e-30)
        e = np.exp(-Ld * dt)
        N = N * e + birth * (1.0 - e) / Ld
        t += dt
        nsteps += 1

    mass = N * _BIN_V
    tot = mass.sum()
    if tot <= 0:
        mass = np.zeros(NBIN)
        mass[0] = 1.0
        tot = 1.0
    mass = mass / tot

    # drying-induced cracking: a size-dependent fraction of each bin splits in two
    cr = it["dry"]["crack"] * np.clip((d_bin / 260.0) ** 1.1, 0.0, 3.0) / (1.0 + 0.9 * it["s_sat"])
    cr = np.clip(cr, 0.0, 0.75)
    moved = mass * cr
    m2 = mass - moved
    m2[:-1] += moved[1:]          # a cracked agglomerate falls one bin (splits in two)
    m2[0] += moved[0]
    mass = m2 / m2.sum()
    # residual bridging liquid leaves the agglomerates plastic: they smear into lumps and abrade
    # during discharge and handling
    rz = it["dry"]["resid"]
    lump = float(np.clip(CF["lump"] * rz, 0.0, 0.35))
    hand = float(np.clip(CF["hand"] * rz, 0.0, 0.35))
    mv = mass * lump
    m3 = mass - mv
    m3[1:] += mv[:-1]
    m3[-1] += mv[-1]
    mass = m3 / m3.sum()
    mass = (1.0 - hand) * mass
    mass[0] += hand
    # wet attrition of agglomerates in the shear field -> fines
    attr = float(np.clip(CF["attr"] * (it["G"] / 116.7) ** 1.20 * (it["solids"] / 100.0) ** 1.20
                         / (sig_w ** 0.90), 0.0, 0.45))
    mass = (1.0 - attr) * mass
    mass[0] += attr
    # solids that were never properly suspended stay as un-agglomerated primaries
    b = it["bypass"]
    mass = (1.0 - b) * mass
    mass[0] += b
    return mass, d_bin, sig_w, t, nsteps


def psd_stats(mass: np.ndarray, d_bin: np.ndarray):
    """Volume-based d50 / span / fines / coarse fraction, interpolated in log diameter."""
    cum = np.cumsum(mass)
    ln_d = np.log(d_bin)

    def quant(p):
        if cum[0] >= p:
            return d_bin[0]
        k = int(np.searchsorted(cum, p))
        k = min(k, NBIN - 1)
        c0, c1 = cum[k - 1], cum[k]
        if c1 <= c0:
            return d_bin[k]
        f = (p - c0) / (c1 - c0)
        return float(np.exp(ln_d[k - 1] + f * (ln_d[k] - ln_d[k - 1])))

    def frac_below(dcut):
        if dcut <= d_bin[0]:
            return 0.0
        if dcut >= d_bin[-1]:
            return 1.0
        k = int(np.searchsorted(d_bin, dcut))
        c0 = cum[k - 1]
        f = (math.log(dcut) - ln_d[k - 1]) / (ln_d[k] - ln_d[k - 1])
        return float(c0 + f * mass[k])

    d10, d50, d90 = quant(0.10), quant(0.50), quant(0.90)
    span = (d90 - d10) / max(d50, 1e-9)
    return d50, span, frac_below(FINES_CUT), 1.0 - frac_below(COARSE_CUT)


# =================================================================================================
# frozen surrogate for (strength, porosity)
# =================================================================================================
_SURR: Dict[str, Any] = {"model": None, "path": None}

FEATS = ["s_sat", "gam_eff", "cos_eff", "dp", "ar", "eps0", "p_c", "w_shear", "shear_rate",
         "t_agg_s", "shrink", "crack", "resid", "hpmc_cov", "solid_bridge", "reg_over"]


# The declared input/output contract of the surrogate. Duplicated from
# data/agglo_strength_porosity.schema.json on purpose: the loader below asserts the file it just
# opened matches this, so a swapped or re-fitted bundle with a different shape fails loudly at load
# time instead of silently becoming a different ruler.
SURROGATE_CONTRACT = {
    "n_features": 16,
    "n_outputs": 2,
    "hidden_activation": "relu",
    "output_activation": "identity",
    "output_columns": ["log_strength", "log_porosity"],
    "physical_transform": "exp",
}


def get_surrogate(path: Optional[str] = None) -> Dict[str, Any]:
    """Load (once) the frozen strength/porosity surrogate from its NON-EXECUTABLE .npz bundle.

    `allow_pickle=False` is the point of this function: the file is a bag of float64 arrays, so
    opening it cannot run code. There is no version-pinned deserialiser in the path -- any numpy that
    can read a .npz reads this one.
    """
    p = os.path.abspath(path or os.environ.get("AGGLO_SURROGATE") or DEFAULT_SURROGATE)
    if _SURR["model"] is None or _SURR["path"] != p:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"frozen surrogate not found at {p}. It ships with the package as "
                "data/agglo_strength_porosity.npz (plain float64 arrays; the schema is "
                "data/agglo_strength_porosity.schema.json).")
        with np.load(p, allow_pickle=False) as z:
            bundle = {k: z[k] for k in z.files}
        _check_surrogate_bundle(bundle, p)
        _SURR["model"] = bundle
        _SURR["path"] = p
    return _SURR["model"]


def _check_surrogate_bundle(bundle: Dict[str, Any], path: str) -> None:
    """Fail loudly if the loaded bundle does not match the declared schema."""
    n = int(bundle["n_layers"])
    need = ["scaler_mean", "scaler_scale", "mu_y", "sc_y"] + \
           [f"{t}{i}" for i in range(n) for t in ("W", "b")]
    missing = [k for k in need if k not in bundle]
    if missing:
        raise ValueError(f"surrogate bundle {path} is missing arrays {missing}")
    nf = SURROGATE_CONTRACT["n_features"]
    no = SURROGATE_CONTRACT["n_outputs"]
    if bundle["scaler_mean"].shape != (nf,) or bundle["scaler_scale"].shape != (nf,):
        raise ValueError(f"surrogate bundle {path}: scaler does not declare {nf} input features")
    if bundle["W0"].shape[0] != nf:
        raise ValueError(f"surrogate bundle {path}: first weight matrix does not take {nf} inputs")
    if bundle[f"W{n - 1}"].shape[1] != no or bundle["mu_y"].shape != (no,):
        raise ValueError(f"surrogate bundle {path}: does not produce {no} outputs")
    for i in range(n - 1):
        if bundle[f"W{i}"].shape[1] != bundle[f"W{i + 1}"].shape[0]:
            raise ValueError(f"surrogate bundle {path}: layer {i} / {i + 1} shapes do not chain")


def surrogate_predict(bundle: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """The whole forward pass: standardise, dense relu stack, de-standardise. No deserialisation.

    Returns the network's two outputs in LOG space (log strength, log porosity); the caller
    exponentiates. Numerically identical to the scikit-learn Pipeline this replaced -- measured, see
    scripts/convert_surrogate_to_npz.py.
    """
    Z = (np.asarray(X, dtype=np.float64) - bundle["scaler_mean"]) / bundle["scaler_scale"]
    n = int(bundle["n_layers"])
    for i in range(n):
        Z = Z @ bundle[f"W{i}"]
        Z = Z + bundle[f"b{i}"]
        if i != n - 1:
            Z = np.maximum(Z, 0.0)
    return Z * bundle["sc_y"] + bundle["mu_y"]


def _quiet_matmul(fn):
    """Run the surrogate's BLAS-heavy forward pass, silence the spurious matmul FP flags, then prove
    the result is clean.

    On this machine numpy 2.2 + Accelerate raises "divide by zero / overflow / invalid encountered in
    matmul" RuntimeWarnings from inside sklearn's dense matmul even though every input is finite and
    every output is finite. The warnings are pure noise and would bury the real output of an
    enumeration over 10^5 designs. Suppressing them is only safe because of the assertion below,
    which is the actual guarantee: a non-finite prediction raises here instead of silently becoming a
    nonsense response that scores 0.
    """
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                                category=RuntimeWarning)
        out = fn()
    if not np.isfinite(np.asarray(out)).all():
        raise RuntimeError("frozen surrogate returned a non-finite strength/porosity prediction")
    return out


def feature_vector(it: Dict[str, Any]) -> np.ndarray:
    """The 16 mechanistic intermediates the frozen surrogate is defined on (order matters)."""
    return np.array([[it["s_sat"], it["gam_eff"], it["cos_eff"], it["dp"], it["ar"], it["eps0"],
                      it["p_c"], it["w_shear"], it["G"], it["t_agg"], it["dry"]["shrink"],
                      it["dry"]["crack"], it["dry"]["resid"], it["cov"], it["solid_bridge"],
                      it["reg_over"]]])


# =================================================================================================
# the four responses
# =================================================================================================
def responses(x: Dict[str, Any], use_surrogate: bool = True, CFL: float = 0.30,
              maxstep: int = 240, surrogate_path: Optional[str] = None) -> Dict[str, float]:
    """Measure one batch. `use_surrogate=False` swaps in the mechanism the surrogate was fitted to
    and is for auditing only -- the shipped ruler is the surrogate."""
    it = intermediates(x)
    mass, d_bin, sig_w, t_reach, nst = run_pbe(it, CFL=CFL, maxstep=maxstep)
    d50_raw, span, fines, coarse = psd_stats(mass, d_bin)

    cl = it["cls"]
    m = mass.copy()
    lost = 0.0
    if cl["f"]:
        below = d_bin < FINES_CUT
        lost += m[below].sum() * (1.0 - cl["rec"])   # recycled fines are recovered
        m[below] = 0.0
    if cl["c"]:
        above = d_bin > COARSE_CUT
        lost += m[above].sum()
        m[above] = 0.0
    if m.sum() <= 1e-12:
        m = mass.copy()
        lost = 0.0
    m = m / m.sum()
    d50, span2, _, _ = psd_stats(m, d_bin)

    if use_surrogate:
        sm = get_surrogate(surrogate_path)
        fv = feature_vector(it)
        pr = _quiet_matmul(lambda: surrogate_predict(sm, fv)[0])
        sig, eps = math.exp(pr[0]), math.exp(pr[1])
    else:
        sig, eps = dry_strength_porosity_mech(it)

    # yield: solubility mass balance + classification loss + attrition + rework losses
    v_bl = float(x["S2_bridging_ratio_v_w"])                     # L/kg
    v_cont = 1000.0 / it["solids"]                               # L/kg
    sol_cont = 3.4 * math.exp(-4.1 * (it["wfr"] - 0.55)) * math.exp(0.042 * (it["T"] - 25.0))
    loss_diss = (it["sol_bl"] * v_bl + sol_cont * v_cont) / 10.0        # %
    loss_wash = sol_cont * 2.0 / 10.0 * 1.25
    loss_cls = 100.0 * lost
    loss_att = 100.0 * (0.030 + 0.24 * it["dry"]["crack"]) * (0.55 + 0.45 * (0.34 / max(eps, 0.15)))
    loss_hp = 0.9 * it["cov"]
    loss_res = CF["loss_handle"] * it["dry"]["resid"]
    loss_over = CF["ow_yield"] * it["reg_over"]      # over-wetted lumps entrap mother liquor
    yld = 100.0 - loss_diss - loss_wash - loss_cls - loss_att - loss_hp - loss_res - loss_over
    yld = float(np.clip(yld, 20.0, 99.5))

    return dict(d50=float(d50), strength=float(sig), porosity=float(eps), yield_pct=yld,
                span=float(span2), fines=float(fines), coarse=float(coarse),
                d50_raw=float(d50_raw), s_sat=it["s_sat"], eps_wet=it["eps_wet"],
                wet_strength=float(sig_w), bypass=it["bypass"], t_reach=t_reach, nsteps=nst,
                loss_diss=loss_diss, loss_cls=loss_cls, loss_att=loss_att)


def surrogate_vs_mechanism_drift(x: Dict[str, Any]) -> Dict[str, float]:
    """|surrogate - fitted mechanism| for strength and porosity, for the audit script.

    The surrogate IS the ruler, so this difference is descriptive, not an error term in the score.
    """
    a = responses(x, use_surrogate=True)
    b = responses(x, use_surrogate=False)
    return {"strength": abs(a["strength"] - b["strength"]),
            "porosity": abs(a["porosity"] - b["porosity"]),
            "strength_rel": abs(a["strength"] - b["strength"]) / max(b["strength"], 1e-12),
            "porosity_rel": abs(a["porosity"] - b["porosity"]) / max(b["porosity"], 1e-12)}

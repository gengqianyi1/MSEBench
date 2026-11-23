"""RESPONSE MODEL -- the boundary module. GRADER SIDE ONLY.

=================================================================================================
THIS FILE IS THE ONLY PLACE IN THE PACKAGE THAT KNOWS HOW A FORMULATION BECOMES A CQA.
=================================================================================================
It holds three things and nothing else depends on them directly:

  1. the invented per-substance property table (SUB / MCC_GRADE),
  2. the closed-form synthetic simulator cqa_true() that those properties feed,
  3. the fitted surrogate that IS the scoring ruler (Oracle, reading data/oracle.npz).

Why the boundary exists (2026-07-30, second-round audit of the public/grader split)
-----------------------------------------------------------------------------------
The first split hid the reference optimum but shipped `data/oracle.npz` and the simulator in the
public half. That was a real hole: an agent with file access does not need the reference value to
win -- it only needs to be able to EVALUATE candidates. With the weights in hand it can optimise the
response surface offline and skip the five-round reasoning task entirely.

So the delivery works like this:

  grader package  ships this file as it stands, plus data/oracle.npz and its schema. Everything that
                  runs a model or grades a run happens here.
  public package  ships a STUB with the same names, whose every callable raises
                  ResponseModelUnavailable. The public half therefore carries the task text, the
                  contracts, the harness, the tool layer and the scorer's logic, and CANNOT compute a
                  CQA, a desirability, or a score. That is deliberate: the party that runs the
                  evaluation holds the grader package.

Every other module imports the mechanism THROUGH THIS FILE (design_space, evaluator and the designer
scripts). scripts/package_for_delivery.py asserts that no other public module reaches around it, and
that the public tree contains no weight file and cannot construct an Oracle.

DATA NATURE: SYNTHETIC. Nothing here is measured, fitted to a measurement, or derived from a real
product. See data/DATA_CARD.md.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Sequence

import numpy as np

from formulation import (API_PCT, LEVELS, MIN_FILLER_PCT, SLOTS, SUBSTANCES, acid_consistent,
                         composition)

SEED = 20260727
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(_SRC_DIR)
DEFAULT_ORACLE_PATH = os.path.join(PACKAGE_ROOT, "data", "oracle.npz")
ORACLE_SCHEMA_PATH = os.path.join(PACKAGE_ROOT, "data", "oracle_schema.json")
# Array names inside data/oracle.npz. Declared here and in data/oracle_schema.json;
# scripts/verify_oracle_npz.py asserts the two agree.
ORACLE_ARRAY_KEYS = ("W0", "b0", "W1", "b1", "W2", "b2",
                     "x_mean", "x_scale", "y_mean", "y_scale")


class ResponseModelUnavailable(RuntimeError):
    """Raised by the PUBLIC stub of this module. Never raised by the grader implementation."""


# ---------------------------------------------------------------------------------------------
# Material properties (hidden -- grader side only).
#   emc   equilibrium moisture content of the pure component at 25 C / 60 % RH (% w/w)
#   acid  effective acid capacity  (arbitrary mEq/g; "effective" = capacity x solubility)
#   base  effective base capacity
#   plast plastic-bonding index
#   ssa   specific surface area (m2/g)
#   d50   median particle size (um)
#   hyd   hydrophobicity / wetting retardation
#   dp    disintegration driving power
# INVENTED for this benchmark. Not measured, not taken from any monograph.
# ---------------------------------------------------------------------------------------------
SUB: Dict[str, Dict[str, float]] = {
    'API':  dict(emc=4.0,  acid=0.05, base=0.00, plast=0.30, ssa=2.0,  d50=15.,  hyd=0.30, dp=0.0),
    'DCPA': dict(emc=0.30, acid=0.05, base=0.90, plast=0.10, ssa=1.0,  d50=160., hyd=0.00, dp=0.0),
    'MAN':  dict(emc=0.15, acid=0.02, base=0.03, plast=0.16, ssa=0.5,  d50=110., hyd=0.00, dp=0.0),
    'FUM':  dict(emc=0.10, acid=0.15, base=0.00, plast=0.20, ssa=1.5,  d50=60.,  hyd=0.85, dp=0.0),
    'TAR':  dict(emc=1.20, acid=0.22, base=0.00, plast=0.35, ssa=1.0,  d50=60.,  hyd=0.10, dp=0.0),
    'CIT':  dict(emc=3.60, acid=0.18, base=0.00, plast=0.50, ssa=1.2,  d50=70.,  hyd=0.05, dp=0.0),
    'MEG':  dict(emc=6.00, acid=0.00, base=20.0, plast=0.50, ssa=1.0,  d50=60.,  hyd=0.00, dp=0.0),
    'XPVP': dict(emc=12.0, acid=0.02, base=0.00, plast=0.40, ssa=1.5,  d50=40.,  hyd=0.00, dp=1.00),
    'CCS':  dict(emc=15.0, acid=0.00, base=0.15, plast=0.50, ssa=1.2,  d50=40.,  hyd=0.00, dp=1.25),
    'LHPC': dict(emc=8.0,  acid=0.00, base=0.05, plast=0.60, ssa=1.4,  d50=45.,  hyd=0.00, dp=0.85),
    'SSG':  dict(emc=22.0, acid=0.00, base=0.25, plast=0.40, ssa=1.0,  d50=42.,  hyd=0.00, dp=1.40),
    'MGST': dict(emc=1.50, acid=0.00, base=4.0,  plast=0.00, ssa=8.0,  d50=10.,  hyd=1.00, dp=0.0),
    'SIL':  dict(emc=4.00, acid=1.50, base=0.00, plast=0.00, ssa=200., d50=0.02, hyd=0.00, dp=0.0),
    # MCC shares one substance identity but its grades differ in the properties below.
}

# All MCC grades equilibrate near 4.2 % at 60 % RH -- coarsening buys flow, never dryness.
MCC_GRADE: Dict[str, Dict[str, float]] = {
    'PH101': dict(emc=4.30, acid=0.04, base=0.02, plast=1.08, ssa=1.35, d50=50.,  hyd=0.0, dp=0.0, flow=-0.80),
    'PH102': dict(emc=4.25, acid=0.04, base=0.02, plast=1.00, ssa=1.10, d50=100., hyd=0.0, dp=0.0, flow=0.00),
    'PH200': dict(emc=4.20, acid=0.04, base=0.02, plast=0.78, ssa=0.72, d50=180., hyd=0.0, dp=0.0, flow=0.90),
    'PH301': dict(emc=4.25, acid=0.04, base=0.02, plast=0.90, ssa=1.20, d50=50.,  hyd=0.0, dp=0.0, flow=0.45),
}


# ---------------------------------------------------------------------------------------------
# hidden generator
# ---------------------------------------------------------------------------------------------
def _props(d: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    g = MCC_GRADE[d['T1_mcc_grade']]
    return {s: (g if s == 'MCC' else SUB[s]) for s in SUBSTANCES}


def cqa_true(d: Dict[str, Any]) -> np.ndarray:
    """The synthetic data-generating process for the five CQAs.

    SYNTHETIC. These five closed forms are hand-written for this benchmark: the functional shapes
    are borrowed from standard solid-dosage relationships (a capacity log-ratio for microenvironment
    pH, additive isotherms with co-sorption cross terms for EMC, a Ryshkewitch-type compaction law
    with an over-lubrication factor for tensile strength, a saturating glidant/lubricant flow model
    for ffc, first-order release for Q30), but every coefficient is chosen so the instance behaves
    the way the benchmark needs, NOT fitted to any measurement. Never exposed to the graded model;
    used to build the fixed-seed training set and, in scripts/check_consistency.py, to bound how
    far the surrogate has drifted from the process it was fitted to."""
    w = composition(d)
    P = _props(d)
    f = {s: w[s] / 100.0 for s in SUBSTANCES}

    # y1 microenvironment pH of a 20 % w/v slurry: log ratio of base to acid capacity.
    A = 0.036 + sum(f[s] * P[s]['acid'] for s in SUBSTANCES)
    B = 0.010 + sum(f[s] * P[s]['base'] for s in SUBSTANCES)
    y1 = -1.653 + 10.00 * np.log10(B / A)

    # y2 EMC at 25 C / 60 % RH: weighted isotherms + silica/disintegrant co-sorption
    #    + deliquescence excess of the acidifier.
    emc = sum(f[s] * P[s]['emc'] for s in SUBSTANCES)
    sil, dis, mcc = w['SIL'], float(d['T7_dis_pct']), w['MCC']
    deliq = {'none': 0.0, 'FUM': 0.05, 'TAR': 0.55, 'CIT': 1.00}[d['T3_acid_id']]
    y2 = (emc + 0.30 * np.sqrt(sil * dis) + 0.12 * mcc * dis / 100.0
          + 0.10 * sil * mcc / 100.0
          + 0.10 * deliq * float(d['T4_acid_pct']) ** 1.6)

    # y3 radial tensile strength: plastic fraction x compaction x over-lubrication x capping.
    Pl = sum(f[s] * P[s]['plast'] for s in SUBSTANCES)
    SSA = sum(f[s] * P[s]['ssa'] for s in SUBSTANCES)
    pr = float(d['T11_press_MPa'])
    f_p = 1.0 - np.exp(-pr / 115.0)
    OL = np.exp(-0.055 * float(d['T8_mgst_pct']) * np.sqrt(float(d['T9_blend_min'])) * SSA)
    sil_pen = np.exp(-0.55 * sil)
    cap = 1.0 - 0.50 * max(0.0, (pr - 190.0) / 70.0) ** 1.8
    acid_lub = np.exp(-0.10 * w['FUM'] - 0.03 * w['TAR'])
    y3 = 7.47 * Pl ** 1.25 * f_p * OL * sil_pen * cap * acid_lub

    # y4 flow function coefficient ffc: particle size, glidant saturation, lubricant with a
    #    sub-critical-lubrication penalty, cohesion from specific surface area, grade flow.
    lnd = sum(f[s] * np.log(P[s]['d50']) for s in SUBSTANCES)
    d50mix = np.exp(lnd)
    glid = 4.0 * (1.0 - np.exp(-sil / 0.22))
    lub = 2.0 * (1.0 - np.exp(-float(d['T8_mgst_pct']) * np.sqrt(float(d['T9_blend_min'])) / 3.0))
    lub -= 1.30 * (max(0.0, 0.55 - float(d['T8_mgst_pct'])) / 0.25) ** 1.3
    y4 = (0.90 + 2.60 * np.log10(d50mix / 20.0) + glid + lub - 1.10 * SSA
          + MCC_GRADE[d['T1_mcc_grade']]['flow'] * (w['MCC'] / 38.0))

    # y5 Q30: first-order release, rate = disintegration drive x microenvironment solubility
    #    x hydrophobisation retardation x hardness retardation.
    drive = sum(f[s] * P[s]['dp'] for s in SUBSTANCES) * 100.0 + 0.03 * w['MCC'] + 1.5 * sil
    sol = min(2.0, max(0.5, 1.0 + 0.20 * (6.26 - y1)))
    hyd_eff = (float(d['T8_mgst_pct']) * (1.0 + 0.40 * np.log(float(d['T9_blend_min']) / 2.0))
               + 0.85 * w['FUM'] + 0.30 * API_PCT)
    k = 0.019 * drive * sol * np.exp(-0.35 * hyd_eff) * np.exp(-0.18 * y3)
    y5 = 100.0 * (1.0 - np.exp(-k * 30.0))
    return np.array([y1, y2, y3, y4, y5])


# ---------------------------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------------------------
def feature_vector(d: Dict[str, Any]) -> np.ndarray:
    """22-D oracle input: one-hots for the three identity slots + numeric slots + q.s. filler %."""
    x: List[float] = []
    for g in LEVELS['T1_mcc_grade']:
        x.append(1.0 if d['T1_mcc_grade'] == g else 0.0)
    for a in LEVELS['T3_acid_id']:
        x.append(1.0 if d['T3_acid_id'] == a else 0.0)
    for s in LEVELS['T6_dis_id']:
        x.append(1.0 if d['T6_dis_id'] == s else 0.0)
    w = composition(d)
    x += [float(d['T2_mcc_pct']), float(d['T4_acid_pct']), float(d['T5_meg_pct']),
          float(d['T7_dis_pct']), float(d['T8_mgst_pct']), float(np.log(float(d['T9_blend_min']))),
          float(d['T10_sil_pct']), float(d['T11_press_MPa']) / 100.0, float(d['T12_man_pct']),
          w['DCPA']]
    return np.asarray(x, dtype=float)


def _quiet_matmul(fn):
    """Run a BLAS-heavy call, silence the spurious matmul FP flags, then prove the result is clean.

    On this machine numpy 2.2 + Accelerate raises 'divide by zero / overflow / invalid encountered in
    matmul' RuntimeWarnings from inside sklearn's dense matmul even though every input is finite and
    every output is finite (checked: 365,014 x 22 inputs, zero non-finite outputs). The warnings are
    pure noise and would bury the real output of the enumeration scripts. Suppressing them is only
    safe with the assertion below, which is the actual guarantee: if a prediction ever did come back
    non-finite it would raise here instead of silently scoring 0.
    """
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                                category=RuntimeWarning)
        out = fn()
    if not np.isfinite(np.asarray(out)).all():
        raise RuntimeError("oracle returned a non-finite CQA prediction")
    return out


class Oracle:
    """The scoring ruler: the fitted network that turns a design into five measured CQAs.

    Loaded from data/oracle.npz -- plain float64 arrays inside a zip, read with
    `numpy.load(..., allow_pickle=False)`. There is NO object deserialisation anywhere in this
    path, so a third party can load the ruler in a sandbox without executing designer-supplied
    code. The forward pass below is written out explicitly and is bit-identical to the sklearn
    implementation it replaced (audit finding #2; proof: scripts/migrate_oracle_to_npz.py printed
    max|pickle - npz| = 0.0 over all 365,014 enumerated designs, re-checkable offline at any time
    with scripts/verify_oracle_npz.py).

    The array names, layer sizes and activations are declared in data/oracle_schema.json, which
    also carries the sha256 of the npz; verify_oracle_npz.py asserts the file still matches.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or DEFAULT_ORACLE_PATH
        if str(self.path).endswith(".pkl"):
            raise ValueError(
                "pickled oracles are not supported any more (audit finding #2: executable object "
                "deserialisation). Use data/oracle.npz; migrate an old pickle with "
                "scripts/migrate_oracle_to_npz.py.")
        with np.load(self.path, allow_pickle=False) as z:
            missing = [k for k in ORACLE_ARRAY_KEYS if k not in z.files]
            if missing:
                raise ValueError(f"{self.path} is not a valid oracle archive; missing {missing}")
            self.p = {k: np.asarray(z[k], dtype=float) for k in ORACLE_ARRAY_KEYS}

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """(n, 22) raw feature matrix -> (n, 5) CQA matrix. Explicit standardise / relu / relu /
        identity / de-standardise, in that order -- the same order and the same float64 dtype the
        fitted implementation used."""
        p = self.p
        z = (np.asarray(X, dtype=float) - p["x_mean"]) / p["x_scale"]
        h = np.maximum(z @ p["W0"] + p["b0"], 0.0)
        h = np.maximum(h @ p["W1"] + p["b1"], 0.0)
        out = h @ p["W2"] + p["b2"]
        return out * p["y_scale"] + p["y_mean"]

    def predict(self, designs: Sequence[Dict[str, Any]]) -> np.ndarray:
        X = np.array([feature_vector(d) for d in designs])
        return _quiet_matmul(lambda: self.predict_matrix(X))

    def predict_one(self, design: Dict[str, Any]) -> np.ndarray:
        return self.predict([design])[0]


_ORACLE: Oracle | None = None


def get_oracle(path: str | None = None) -> Oracle:
    """Process-wide singleton: unpickling the network on every evaluation would dominate runtime."""
    global _ORACLE
    if _ORACLE is None or (path is not None and path != _ORACLE.path):
        _ORACLE = Oracle(path)
    return _ORACLE


def random_designs(n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Uniform draws from the acid-consistent, mass-closable design space (training-set sampler)."""
    out: List[Dict[str, Any]] = []
    while len(out) < n:
        d = {s: LEVELS[s][rng.integers(len(LEVELS[s]))] for s in SLOTS}
        if not acid_consistent(d):
            continue
        if composition(d)['DCPA'] < MIN_FILLER_PCT:
            continue
        out.append(d)
    return out


def train_oracle(n: int = 60000, out_path: str | None = None):
    """Rebuild data/oracle.npz from the synthetic generator. Provenance only -- the shipped npz IS
    the ruler, and re-running this replaces it, which invalidates every calibrated number
    downstream. Fixed seed (SEED = 20260727), so it reproduces bit-for-bit.

    Needs scikit-learn (fitting only). The OUTPUT is a plain float64 array archive: nothing that is
    written here has to be un-pickled to be used, so the shipped package has no executable
    deserialisation in it at all (audit finding #2).

    NOISE MODEL, stated in full because the audit asked for it: additive i.i.d. Gaussian on each of
    the five CQAs, zero mean, standard deviation 1 % of that CQA's own spread across the training
    sample (`0.01 * Y.std(axis=0)`), drawn from the same seeded generator that drew the designs.
    There is no heteroscedasticity, no correlated error and no batch effect: this is a synthetic
    reproducibility model, not an estimate of real analytical variability.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    designs = random_designs(n, rng)
    X = np.array([feature_vector(d) for d in designs])
    Y = np.array([cqa_true(d) for d in designs])
    Y = Y + rng.normal(0, 1, Y.shape) * (0.01 * Y.std(axis=0))   # fixed-seed synthetic noise
    ntr = int(0.9 * n)
    xs, ys = StandardScaler().fit(X[:ntr]), StandardScaler().fit(Y[:ntr])
    net = MLPRegressor(hidden_layer_sizes=(128, 64), random_state=SEED, max_iter=900,
                       early_stopping=True, n_iter_no_change=25, tol=1e-7,
                       learning_rate_init=2e-3, batch_size=256)
    net.fit(xs.transform(X[:ntr]), ys.transform(Y[:ntr]))
    Yp = ys.inverse_transform(net.predict(xs.transform(X[ntr:])))
    Yt = Y[ntr:]
    r2 = 1.0 - ((Yp - Yt) ** 2).sum(0) / ((Yt - Yt.mean(0)) ** 2).sum(0)
    np.savez(out_path or DEFAULT_ORACLE_PATH,
             W0=np.asarray(net.coefs_[0], float), b0=np.asarray(net.intercepts_[0], float),
             W1=np.asarray(net.coefs_[1], float), b1=np.asarray(net.intercepts_[1], float),
             W2=np.asarray(net.coefs_[2], float), b2=np.asarray(net.intercepts_[2], float),
             x_mean=np.asarray(xs.mean_, float), x_scale=np.asarray(xs.scale_, float),
             y_mean=np.asarray(ys.mean_, float), y_scale=np.asarray(ys.scale_, float))
    return r2

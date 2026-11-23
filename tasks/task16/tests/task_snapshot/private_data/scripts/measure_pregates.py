"""Pre-gates P1-P4: is the objective decomposable, and does the reference need coordination?

These are the cheap structural tests that come BEFORE the three difficulty gates. The question they
answer is not "how hard is it to score" (that is G1/G2) but "is there a shortcut that makes the
coordination unnecessary":

  P1 additivity      ridge regression of the objective on condition one-hots plus every pairwise
                     product of them, holdout R2. A high R2 means an additive surrogate explains the
                     objective, which is the precondition for condition-by-condition greedy search.
                     Threshold: R2 on the objective D < 0.8.
  P1' the same threat, but operationally: fit that surrogate on 40,000 TRUE evaluations -- far more
                     than any graded model can obtain in five rounds -- then let it pick its own
                     argmax over the whole feasible neighbourhood and score that design with the real
                     ruler. This is the number that matters; R2 is only a proxy for it.
  P2 superadditivity joint gain of the reference's four changes divided by the sum of the four
                     one-at-a-time gains. Threshold > 1.5.
  P3 one-slot ceiling the best score reachable by changing exactly one condition (same measurement as
                     gate G1's first row, repeated here so the pre-gates stand alone). Threshold <= 20.
  P4 monotonicity    sweep every level of every condition with the other eleven at the baseline
                     value: no condition may be monotone in D over its whole grid in the same
                     direction as every other, or the problem is a knapsack.

Run:
    python scripts/measure_pregates.py
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_space import (VectorOracle, achievement, describe_row, enumerate_exact_k,  # noqa: E402
                          enumerate_rows, filler_pct_vec, n_changed_vec, n_identity_vec,
                          overlap_vec, points_for, START_IDX, acid_ok_vec, to_design)
from evaluator import scorer_from_config  # noqa: E402
from formulation import LEVELS, MIN_FILLER_PCT, SLOTS, START  # noqa: E402
from utils import load_yaml  # noqa: E402

TASK_CFG = ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml"
NLEV = [len(LEVELS[s]) for s in SLOTS]
THRESHOLDS = {"P1_r2_max": 0.8, "P1_surrogate_attack_points_max": 49,
              "P2_ratio_min": 1.5, "P3_one_slot_points_max": 20}


def one_hot_pairwise(R: np.ndarray) -> np.ndarray:
    oh = []
    for i in range(len(SLOTS)):
        E = np.zeros((len(R), NLEV[i]))
        E[np.arange(len(R)), R[:, i]] = 1.0
        oh.append(E)
    blocks = [np.hstack(oh)]
    for a, b in itertools.combinations(range(len(SLOTS)), 2):
        blocks.append((oh[a][:, :, None] * oh[b][:, None, :]).reshape(len(R), -1))
    return np.hstack(blocks)


def main() -> None:
    from sklearn.linear_model import Ridge

    cfg = load_yaml(TASK_CFG)
    cons = cfg["constraints"]
    cal = cfg["reference"]["calibration"]
    floor = float(cal["overlap_floor"])
    sc = scorer_from_config(cfg["objective"])

    O = VectorOracle()
    R = enumerate_rows(int(cons["max_slot_changes"]["max"]))
    Y = O.predict_rows(R)
    y0 = O.predict_rows(START_IDX[None, :])[0]
    D = sc.D_of(Y)
    D0 = sc.D_one(y0)
    # TWO populations, and P1 needs both (ruler 2.0.0 added the CQA-compliance gate under this
    # script). BUDGET is where a surrogate-fitting attacker SAMPLES: it cannot know which designs are
    # compliant until it has measured them, so restricting the fit to the compliant subset would both
    # flatter the surrogate (a smaller, more homogeneous population at a far higher
    # features-per-sample ratio) and describe an attack nobody can mount. FEAS is where the attacker
    # is allowed to PICK, because nothing outside it can score.
    BUDGET = ((overlap_vec(R) >= floor - 1e-12)
              & (n_changed_vec(R) <= int(cons["max_slot_changes"]["max"]))
              & (n_identity_vec(R) <= int(cons["max_identity_changes"]["max"]))
              & (filler_pct_vec(R) >= MIN_FILLER_PCT))
    FEAS = BUDGET & sc.in_spec(Y) & (D > D0)
    jref = int(np.argmax(np.where(FEAS, D, -1.0)))
    Dref = float(D[jref])
    assert abs(Dref - float(cfg["reference"]["best_objective_in_budget"])) < 1e-9, \
        "task_config reference value has drifted -- run scripts/enumerate_reference_best.py"
    print("baseline D = %.9g   reference D = %.9g   budget-feasible %d, of which scoring-eligible"
          " (compliant and improving) %d" % (D0, Dref, int(BUDGET.sum()), int(FEAS.sum())))
    out = {"thresholds": THRESHOLDS}

    # ------------------------------------------------------------------ P1
    idx = np.where(FEAS)[0]           # where the attacker may PICK
    fit_idx = np.where(BUDGET)[0]     # where the attacker SAMPLES
    rng = np.random.default_rng(0)
    sub = rng.choice(fit_idx, min(40000, len(fit_idx)), replace=False)
    X = one_hot_pairwise(R[sub])
    ntr = int(0.75 * len(sub))
    r2 = {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
        for tag, tgt in (("D", D[sub]), ("logD", np.log(D[sub]))):
            m = Ridge(alpha=1.0).fit(X[:ntr], tgt[:ntr])
            p, t = m.predict(X[ntr:]), tgt[ntr:]
            r2[tag] = float(1.0 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
        print("\nP1 additivity: %d features (condition one-hots + all pairwise products), %d fitted"
              % (X.shape[1], ntr))
        print("   holdout R2 on the OBJECTIVE D = %.4f   [threshold < %.2f]  -> %s"
              % (r2["D"], THRESHOLDS["P1_r2_max"],
                 "PASS" if r2["D"] < THRESHOLDS["P1_r2_max"] else "FAIL"))
        print("   holdout R2 on log D          = %.4f   (log D is additive over the five d_i BY "
              "CONSTRUCTION and is not the objective; the score is linear in D, and the attack "
              "below is what this number is a proxy for)" % r2["logD"])

        # P1' the same surrogate used as an attacker
        attack = {}
        for tag, tgt in (("D", D[sub]), ("logD", np.log(D[sub]))):
            m = Ridge(alpha=1.0).fit(X, tgt)
            best_pred, best_j = -np.inf, None
            for a in range(0, len(idx), 50000):
                blk = idx[a:a + 50000]
                pred = m.predict(one_hot_pairwise(R[blk]))
                j = int(np.argmax(pred))
                if float(pred[j]) > best_pred:
                    best_pred, best_j = float(pred[j]), int(blk[j])
            attack[tag] = {"points": points_for(float(D[best_j]), D0, Dref),
                           "achievement": achievement(float(D[best_j]), D0, Dref),
                           "value": float(D[best_j]), "design": describe_row(R[best_j])}
            print("   surrogate fitted on %-4s picks its own argmax over all %d feasible designs "
                  "-> %3d pts (ach %.3f) | %s"
                  % (tag, len(idx), attack[tag]["points"], attack[tag]["achievement"],
                     attack[tag]["design"]))
    out["P1"] = {"n_features": int(X.shape[1]), "holdout_r2": r2, "surrogate_attack": attack}

    # ------------------------------------------------------------------ P2
    joint = float(D[jref] - D0)
    marg = []
    for i in range(len(SLOTS)):
        if R[jref][i] == START_IDX[i]:
            continue
        row = START_IDX.copy()
        row[i] = R[jref][i]
        j = int(np.where((R == row).all(1))[0][0])
        marg.append({"slot": SLOTS[i], "level": LEVELS[SLOTS[i]][int(row[i])],
                     "gain": float(D[j] - D0)})
    tot = sum(m["gain"] for m in marg)
    ratio = joint / tot if tot > 0 else float("inf")
    print("\nP2 superadditivity at the reference optimum (%s)" % describe_row(R[jref]))
    for m in marg:
        print("   one-at-a-time %-16s -> %-8s  gain %+.6f" % (m["slot"], m["level"], m["gain"]))
    print("   joint gain %.6f / sum of one-at-a-time gains %.6f = %.3f   [threshold > %.1f]  -> %s"
          % (joint, tot, ratio, THRESHOLDS["P2_ratio_min"],
             "PASS" if ratio > THRESHOLDS["P2_ratio_min"] else "FAIL"))
    out["P2"] = {"joint_gain": joint, "marginal_gains": marg, "sum_marginal": tot, "ratio": ratio}

    # ------------------------------------------------------------------ P3
    R1 = enumerate_exact_k(1)
    ok1 = (acid_ok_vec(R1) & (filler_pct_vec(R1) >= MIN_FILLER_PCT)
           & (n_identity_vec(R1) <= int(cons["max_identity_changes"]["max"]))
           & (overlap_vec(R1) >= floor - 1e-12))
    D1 = np.where(ok1, sc.D_of(O.predict_rows(R1)), -1.0)
    p3 = points_for(float(D1.max()), D0, Dref)
    print("\nP3 one-condition ceiling: %d pts (ach %.4f) | %s   [threshold <= %d]  -> %s"
          % (p3, achievement(float(D1.max()), D0, Dref),
             describe_row(R1[int(np.argmax(D1))]), THRESHOLDS["P3_one_slot_points_max"],
             "PASS" if p3 <= THRESHOLDS["P3_one_slot_points_max"] else "FAIL"))
    out["P3"] = {"points": p3, "achievement": achievement(float(D1.max()), D0, Dref),
                 "best": describe_row(R1[int(np.argmax(D1))])}

    # ------------------------------------------------------------------ P4
    print("\nP4 monotonicity: sweep each condition with the other eleven at the baseline value")
    p4 = {}
    for i, s in enumerate(SLOTS):
        rows = np.repeat(START_IDX[None, :], NLEV[i], 0)
        rows[:, i] = np.arange(NLEV[i])
        keep = acid_ok_vec(rows) & (filler_pct_vec(rows) >= MIN_FILLER_PCT)
        Ds = sc.D_of(O.predict_rows(rows))
        Ds = np.where(keep, Ds, np.nan)
        v = Ds[~np.isnan(Ds)]
        interior = bool(len(v) >= 3 and (np.nanargmax(Ds) not in (0, NLEV[i] - 1)))
        rising = bool(np.all(np.diff(v) > 0))
        falling = bool(np.all(np.diff(v) < 0))
        p4[s] = {"interior_optimum": interior, "monotone": rising or falling,
                 "best_level": LEVELS[s][int(np.nanargmax(Ds))],
                 "D_range": [float(np.nanmin(Ds)), float(np.nanmax(Ds))]}
        print("   %-15s best level %-8s  interior optimum %-5s  monotone %-5s  D %.5f..%.5f"
              % (s, str(p4[s]["best_level"]), interior, rising or falling,
                 p4[s]["D_range"][0], p4[s]["D_range"][1]))
    n_mono = sum(1 for v in p4.values() if v["monotone"])
    print("   %d of %d conditions are monotone over their whole grid; %d have an interior optimum"
          % (n_mono, len(SLOTS), sum(1 for v in p4.values() if v["interior_optimum"])))
    out["P4"] = p4

    verdict = (r2["D"] < THRESHOLDS["P1_r2_max"]
               and attack["D"]["points"] <= THRESHOLDS["P1_surrogate_attack_points_max"]
               and ratio > THRESHOLDS["P2_ratio_min"]
               and p3 <= THRESHOLDS["P3_one_slot_points_max"])
    out["verdict"] = "PASS" if verdict else "FAIL"
    print("\npre-gate verdict: %s" % out["verdict"])
    path = ROOT / "answers" / "pregate_probes.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()

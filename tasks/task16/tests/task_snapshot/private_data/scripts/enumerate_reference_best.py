"""Designer-side enumeration: calibrate the windows, find the reference optimum, measure the gates.

Everything the task config claims about the instance is produced here and nowhere else:

  * the five specification windows (from the two calibration numbers PULL and MARG),
  * start_value  = D of the baseline formulation,
  * reference_best = max D over the fully feasible change-budget neighbourhood,
  * G1 = best score reachable by any 1-slot and any 2-slot change (difficulty gate, must be < 50),
  * G2 = best score with the overlap floor raised by 0.05 (difficulty gate, must be < 50),
  * G3 = number of feasible designs that beat the baseline formulation (fairness gate, >= 50),
  * the score histogram over the whole feasible neighbourhood.

The neighbourhood is enumerated exhaustively, so reference_best is exact within the <= 4-slot change
budget (it is not a proof of global optimality over the full 1.4e9-design space, and achievement is
clipped at 1.0 so a model that beats it still scores 100).

Two anti-self-deception rules are enforced here rather than assumed:
  1. the gates are measured with the SHIPPED oracle and the SHIPPED constraint predicate, never with
     a simplified stand-in;
  2. G1 is enumerated through a second, independent code path and cross-asserted against a filter
     on the main enumeration.

Run:
    python scripts/enumerate_reference_best.py
    python scripts/enumerate_reference_best.py --pull 0.22 --marg 0.30   # alternative
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_space import (POINT_BANDS, VectorOracle, achievement, acid_ok_vec,  # noqa: E402
                          budget_mask, calibrate_windows, compliance_mask,
                          comp_matrix, describe_row, Desirability, enumerate_exact_k,
                          enumerate_rows, feat_matrix, filler_pct_vec, n_changed_vec,
                          n_identity_vec, overlap_vec, points_for, qualifying_mask,
                          START_IDX, SUBI, to_design)
from formulation import (CQA_KEYS, CQA_SHORT, LEVELS, MIN_FILLER_PCT, SLOTS,  # noqa: E402
                         START, SUBSTANCES, composition, design_space_size, ruzicka_overlap)
from response_model import Oracle, cqa_true, feature_vector  # noqa: E402

# Shipping calibration.
#
# PULL/MARG place the five windows: PULL is the fraction of the improvement the change budget can
# actually reach at which the target is placed (so it sets the window width in physical units), MARG
# is how far past the baseline value the dangerous-side limit sits, as a fraction of that
# (so it alone sets how close to a limit the baseline formulation sits: u_start = 1/(1+MARG)).
#
# M/E/KAPPA/FLOOR/PW are the SMOOTH desirability shape (design_space.Desirability). The problem first
# shipped with the piecewise-cubic Derringer-Suich form (r = 3, exactly zero outside the window) at
# PULL 0.28 / MARG 0.35 and measured 0 / 0 / 0 on deepseek-v4-pro, glm-4.6 and qwen-plus with the
# scoring bands never activated. Both halves of the fix are needed and were measured separately:
#   hard cubic + these wider windows  ->  G2 = 63, i.e. the difficulty gate FAILS, models still 0/9/0
#   smooth + the original windows     ->  gates pass, models 9/20/0, but 2.0 % five-window closure
#   smooth + these windows (shipped)  ->  gates pass, models 20/20/0, 8.2 % five-window closure
DEFAULT_PULL = 0.34
DEFAULT_MARG = 0.55
DEFAULT_M = 6.3           # Hill exponent between target and limit
DEFAULT_E = 2.0e-4        # desirability exactly at the printed specification limit
DEFAULT_KAPPA = 2.6       # out-of-window decay rate, x the ramp's own log-slope at the limit
DEFAULT_DFLOOR = 1.0e-6   # asymptotic floor: d > 0 everywhere, so the surface has no cliff
DEFAULT_PW = 0.0          # d = 1 plateau half-width in half-window units
DEFAULT_FLOOR = 0.88      # composition-overlap floor (the change budget, not the desirability)
G2_MARGIN = 0.05          # brief: G2 raises the similarity/overlap floor by exactly 0.05
MAX_SLOT_CHANGES = 4
MAX_IDENTITY_CHANGES = 1
RULER_VERSION = "2.0.0"     # see CHANGELOG.md; frozen with the bands in design_space.POINT_BANDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", type=float, default=DEFAULT_PULL)
    ap.add_argument("--marg", type=float, default=DEFAULT_MARG)
    ap.add_argument("--M", type=float, default=DEFAULT_M)
    ap.add_argument("--e", type=float, default=DEFAULT_E)
    ap.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    ap.add_argument("--dfloor", type=float, default=DEFAULT_DFLOOR)
    ap.add_argument("--pw", type=float, default=DEFAULT_PW)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--out", default=str(ROOT / "answers" / "reference_enumeration.json"))
    args = ap.parse_args()
    np.set_printoptions(suppress=True)

    O = VectorOracle()
    Osc = Oracle()

    print("=" * 100)
    print("step 0  vectorised path == scalar path (anti-self-deception)")
    R = enumerate_rows(MAX_SLOT_CHANGES)
    rng = np.random.default_rng(7)
    sel = rng.choice(len(R), 80, replace=False)
    mx_feat = mx_or = mx_comp = 0.0
    for j in sel:
        d = to_design(R[j])
        mx_feat = max(mx_feat, float(np.abs(feature_vector(d) - feat_matrix(R[j][None, :])[0]).max()))
        mx_or = max(mx_or, float(np.abs(Osc.predict([d])[0] - O.predict_rows(R[j][None, :])[0]).max()))
        w = composition(d)
        v = np.array([w[s] for s in SUBSTANCES])
        mx_comp = max(mx_comp, float(np.abs(v - comp_matrix(R[j][None, :])[0]).max()))
        assert abs(ruzicka_overlap(d) - float(overlap_vec(R[j][None, :])[0])) < 1e-12
    print("   max |feat_matrix - feature_vector|     = %.3e" % mx_feat)
    print("   max |vector oracle - scalar oracle|    = %.3e" % mx_or)
    print("   max |comp_matrix - composition()|      = %.3e" % mx_comp)
    print("   ruzicka_overlap == overlap_vec on all 80 samples: ok")
    assert mx_feat == 0.0 and mx_or == 0.0 and mx_comp < 1e-12

    print("\n" + "=" * 100)
    print("step 1  neighbourhood enumeration and window calibration")
    print("   design space: %s" % design_space_size())
    t0 = time.time()
    Y = O.predict_rows(R)
    print("   rows within <= %d slot changes (acid-consistent, filler >= %.0f%%, <= %d identity"
          " change): %d   (%.1f s oracle)"
          % (MAX_SLOT_CHANGES, MIN_FILLER_PCT, MAX_IDENTITY_CHANGES, len(R), time.time() - t0))

    y0 = O.predict_rows(START_IDX[None, :])[0]
    y0_gen = cqa_true(START)
    win = calibrate_windows(y0, Y, args.pull, args.marg)
    L, T, U = win["L"], win["T"], win["U"]
    sc = Desirability(L, T, U, hill_exponent=args.M, d_at_spec_limit=args.e,
                      tail_decay_multiplier=args.kappa, floor=args.dfloor,
                      plateau_fraction=args.pw)

    D = sc.D_of(Y)
    D0 = sc.D_one(y0)
    OV = overlap_vec(R)
    NC = n_changed_vec(R)
    NI = n_identity_vec(R)
    FILL = filler_pct_vec(R)
    # THE eligibility predicates, all three from design_space so the enumeration, the gates and
    # src/scoring.py cannot drift apart (audit finding #3).
    #   BUDGET  the PACMP change budget only        -- a design-space fact
    #   IN      all five CQAs inside [L, U]         -- THE release / compliance gate
    #   QUAL    BUDGET & IN & D > D_start           -- the scoring-eligible set
    BUDGET = budget_mask(R, args.floor, MAX_SLOT_CHANGES, MAX_IDENTITY_CHANGES, MIN_FILLER_PCT)
    IN = compliance_mask(Y, sc)
    assert (BUDGET == ((OV >= args.floor - 1e-12) & (NC <= MAX_SLOT_CHANGES)
                       & (NI <= MAX_IDENTITY_CHANGES) & (FILL >= MIN_FILLER_PCT))).all()
    # The reference optimum is the best COMPLIANT design in the budget. Scoring anything against a
    # non-releasable reference would reintroduce exactly the defect audit finding #3 reported.
    jref = int(np.argmax(np.where(BUDGET & IN, D, -1.0)))
    Dref = float(D[jref])
    QUAL = qualifying_mask(R, Y, sc, D0, args.floor, MAX_SLOT_CHANGES, MAX_IDENTITY_CHANGES,
                           MIN_FILLER_PCT)
    assert (QUAL == (BUDGET & IN & (D > D0))).all()
    FEAS = QUAL          # every gate below is measured over the scoring-eligible set
    print("   calibration: PULL %.3f  MARG %.3f  overlap floor %.3f" % (args.pull, args.marg,
                                                                        args.floor))
    print("   desirability: %s" % sc.params())
    print("   d(u) along the ruler (u = 0 at target, 1 at the printed limit, no hard zero):")
    print("        %s" % {u: float("%.4g" % sc.d_at_u(u))
                          for u in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)})
    for i, nm in enumerate(CQA_SHORT):
        print("   %-8s L=%9.4f T=%9.4f U=%9.4f   baseline=%9.4f  u_start=%.4f  d_start=%.6g"
              % (nm, L[i], T[i], U[i], y0[i], sc.u_of(y0[None, :])[0, i],
                 sc.d_of(y0[None, :])[0, i]))
    print("   oracle at the baseline formulation    %s" % np.round(y0, 4))
    print("   generator at the baseline formulation %s   (oracle IS the ruler; this line only"
          " bounds drift)" % np.round(y0_gen, 4))
    print("   start_value    D = %.9g" % D0)
    print("   reference_best D = %.9g   overlap %.4f   [%s]" % (Dref, OV[jref], describe_row(R[jref])))
    print("   reference CQAs %s   d = %s   all five in spec: %s"
          % (np.round(Y[jref], 4), np.round(sc.d_of(Y[jref][None, :])[0], 5), bool(IN[jref])))
    assert bool(IN[jref]), ("the reference optimum must have all five CQAs inside their windows; "
                           "if it does not, the desirability tail is too generous and 'sacrifice "
                           "one CQA' has become the optimum")
    print("   inside the change budget            %7d" % int(BUDGET.sum()))
    print("   ... of which CQA-COMPLIANT (all 5)  %7d  (%.2f %%)   <- the release gate"
          % (int((BUDGET & IN).sum()), 100.0 * (BUDGET & IN).sum() / BUDGET.sum()))
    print("   ... of which also D > D_baseline  %7d  (%.2f %%)   <- QUALIFYING, the scored set"
          % (int(QUAL.sum()), 100.0 * QUAL.sum() / BUDGET.sum()))
    print("   budget-feasible but NOT compliant, yet D > D_baseline: %7d"
          " (these scored under ruler 1.x and score 0 now)"
          % int((BUDGET & ~IN & (D > D0)).sum()))
    print("   point bands in D: %s"
          % {p: float("%.6g" % (D0 + a * (Dref - D0))) for a, p in POINT_BANDS if a > 0.001})

    ref_design = to_design(R[jref])
    out = {
        "generated_by": "scripts/enumerate_reference_best.py",
        "calibration": {"pull": args.pull, "marg": args.marg,
                        "desirability": sc.params(),
                        "overlap_floor": args.floor, "max_slot_changes": MAX_SLOT_CHANGES,
                        "max_identity_changes": MAX_IDENTITY_CHANGES,
                        "min_filler_pct": MIN_FILLER_PCT},
        "desirability_curve": {str(u): sc.d_at_u(u)
                               for u in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)},
        "windows": {"cqa_keys": CQA_KEYS, "L": L.tolist(), "T": T.tolist(), "U": U.tolist(),
                    "reported_decimals": win["decimals"].tolist(),
                    "span_used_for_calibration": win["span"].tolist()},
        "start": {"design": START, "cqa_oracle": y0.tolist(),
                  "cqa_hidden_generator": y0_gen.tolist(),
                  "desirability_per_cqa": sc.d_of(y0[None, :])[0].tolist(),
                  "u_per_cqa": sc.u_of(y0[None, :])[0].tolist(),
                  "value": D0, "overlap": 1.0},
        "ruler_version": RULER_VERSION,
        "reference": {"design": ref_design, "changes": describe_row(R[jref]),
                      "value": Dref, "cqa_oracle": Y[jref].tolist(),
                      "overlap": float(OV[jref]),
                      "n_slots_changed": int(NC[jref]),
                      "n_identity_changes": int(NI[jref]),
                      "all_cqa_in_spec": bool(IN[jref]),
                      "desirability_per_cqa": sc.d_of(Y[jref][None, :])[0].tolist()},
        "neighbourhood": {"n_rows_enumerated": int(len(R)),
                          "n_budget_feasible": int(BUDGET.sum()),
                          "n_feasible_all_cqa_in_spec": int((BUDGET & IN).sum()),
                          "five_window_closure_rate": float((BUDGET & IN).sum()) / int(BUDGET.sum()),
                          "n_qualifying": int(QUAL.sum()),
                          "qualifying_rate": float(QUAL.sum()) / int(BUDGET.sum()),
                          "n_noncompliant_but_improving": int((BUDGET & ~IN & (D > D0)).sum())},
        "point_bands": [{"min_achievement": a, "points": p} for a, p in POINT_BANDS],
    }

    # ------------------------------------------------------------------ G1
    print("\n" + "=" * 100
          )
    print("G1  every 1-slot and every 2-slot change (independent enumeration + cross-check)")
    g1 = {}
    for k in (1, 2, 3):
        Rk = enumerate_exact_k(k)
        Yk = O.predict_rows(Rk)
        # SAME three gates as the score: budget, CQA compliance, improvement over the start.
        ok = (acid_ok_vec(Rk) & (filler_pct_vec(Rk) >= MIN_FILLER_PCT)
              & (n_identity_vec(Rk) <= MAX_IDENTITY_CHANGES)
              & (overlap_vec(Rk) >= args.floor - 1e-12)
              & compliance_mask(Yk, sc) & (sc.D_of(Yk) > D0))
        Dk = np.where(ok, sc.D_of(Yk), -1.0)
        order = np.argsort(-Dk)
        j = int(order[0])
        pts = points_for(Dk[j], D0, Dref)
        g1[k] = pts
        print("   %d-slot: enumerated %7d  feasible %7d  best D=%.6f  ach=%.4f -> %3d pts"
              % (k, len(Rk), int(ok.sum()), Dk[j], achievement(Dk[j], D0, Dref), pts))
        for q in order[:5]:
            if Dk[q] < 0:
                break
            print("        D=%.6f ach=%.4f %3d pts | %s"
                  % (Dk[q], achievement(Dk[q], D0, Dref), points_for(Dk[q], D0, Dref),
                     describe_row(Rk[q])))
        if k <= 2:      # cross-check against a filter on the main enumeration
            mask = FEAS & (NC == k)
            jj = int(np.argmax(np.where(mask, D, -1.0)))
            assert points_for(D[jj], D0, Dref) == pts, \
                "independent %d-slot enumeration disagrees with the main-array filter" % k
            print("        cross-check against the main enumeration: agrees (%d pts)" % pts)
    out["gates"] = {"G1_one_slot_points": g1[1],
                    "G1_two_slot_points": max(g1[1], g1[2]),
                    "G1_three_slot_points": max(g1[1], g1[2], g1[3])}

    # ------------------------------------------------------------------ G2
    print("\n" + "=" * 100)
    print("G2  overlap floor raised by %.2f, to %.3f (same predicate for search and acceptance)"
          % (G2_MARGIN, args.floor + G2_MARGIN))
    FEAS2 = (budget_mask(R, args.floor + G2_MARGIN, MAX_SLOT_CHANGES, MAX_IDENTITY_CHANGES,
                         MIN_FILLER_PCT) & IN & (D > D0))
    j2 = int(np.argmax(np.where(FEAS2, D, -1.0)))
    g2 = points_for(D[j2], D0, Dref)
    print("   n=%d  best D=%.6f  ach=%.4f  overlap %.4f -> %d pts | %s"
          % (int(FEAS2.sum()), D[j2], achievement(D[j2], D0, Dref), OV[j2], g2, describe_row(R[j2])))
    out["gates"]["G2_points"] = g2
    out["gates"]["G2_margin"] = G2_MARGIN
    out["gates"]["G2_best"] = {"changes": describe_row(R[j2]), "value": float(D[j2]),
                               "overlap": float(OV[j2])}

    print("   floor sensitivity (same predicate throughout):")
    curve = []
    for f in [0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]:
        mk = (budget_mask(R, f, MAX_SLOT_CHANGES, MAX_IDENTITY_CHANGES, MIN_FILLER_PCT)
              & IN & (D > D0))
        jj = int(np.argmax(np.where(mk, D, -1.0)))
        curve.append({"floor": f, "n": int(mk.sum()), "best_value": float(D[jj]),
                      "points": points_for(D[jj], D0, Dref)})
        print("        floor %.2f  n=%7d  best D=%.6f -> %3d pts | %s"
              % (f, mk.sum(), D[jj], points_for(D[jj], D0, Dref), describe_row(R[jj])))
    out["gates"]["G2_floor_sensitivity"] = curve

    # ------------------------------------------------------------------ G3
    print("\n" + "=" * 100)
    print("G3  feasible designs that strictly improve the objective")
    better = QUAL
    budget_idx = np.where(BUDGET)[0]
    pts_all = np.array([points_for(D[q], D0, Dref) if QUAL[q] else 0 for q in budget_idx])
    band_pts = sorted({0} | {p for _, p in POINT_BANDS})
    hist = {int(p): int((pts_all == p).sum()) for p in band_pts}
    print("   G3 = %d qualifying designs  (requirement >= 50)" % int(better.sum()))
    print("   score histogram over the %d budget-feasible designs (non-compliant ones score 0): %s"
          % (len(budget_idx), hist))
    print("   designs scoring >= 21 pts: %d   >= 30 pts: %d   >= 44 pts: %d"
          % (int((pts_all >= 21).sum()), int((pts_all >= 30).sum()), int((pts_all >= 44).sum())))
    print("   five-window closure rate %.4f (%d / %d)   qualifying rate %.4f"
          % ((IN & BUDGET).sum() / BUDGET.sum(), int((IN & BUDGET).sum()), int(BUDGET.sum()),
             int(better.sum()) / int(BUDGET.sum())))
    out["gates"]["G3_count"] = int(better.sum())
    out["gates"]["score_histogram"] = hist
    out["gates"]["n_designs_at_or_above_21_points"] = int((pts_all >= 21).sum())
    out["gates"]["n_designs_at_or_above_30_points"] = int((pts_all >= 30).sum())
    out["gates"]["n_designs_at_or_above_44_points"] = int((pts_all >= 44).sum())

    # ------------------------------------------------------------------ timing
    t0 = time.time()
    n_timing = 200
    for _ in range(n_timing):
        d = to_design(R[int(rng.integers(len(R)))])
        Osc.predict([d])
        ruzicka_overlap(d)
        sc.D_one(Osc.predict([d])[0])
    ms = (time.time() - t0) / n_timing * 1000.0
    print("\n   single full scalar evaluation (oracle + overlap + D): %.3f ms" % ms)
    out["single_evaluation_ms"] = ms

    verdict = (out["gates"]["G1_two_slot_points"] < 50 and g2 < 50
               and out["gates"]["G3_count"] >= 50)
    out["gate_verdict"] = "PASS" if verdict else "FAIL"
    print("\n   G1=%d  G2=%d  G3=%d  ->  %s"
          % (out["gates"]["G1_two_slot_points"], g2, out["gates"]["G3_count"], out["gate_verdict"]))

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                        default=_json_default), encoding="utf-8")
    print("\nwrote %s" % args.out)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


if __name__ == "__main__":
    main()

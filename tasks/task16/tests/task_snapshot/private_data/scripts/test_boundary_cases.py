"""Boundary tests for the release gate. This is the acceptance evidence for audit finding #3.

The finding's acceptance criterion is:

    "No candidate with any critical CQA outside specification may attain success status; the scoring
     of in-window and out-of-window examples must agree with the task description."

This script proves both halves end to end, with the shipped scorer, on real designs taken from the
enumerated change-budget neighbourhood -- not on hand-built fixtures:

  T1  the starting formulation itself: compliant, but does not improve D -> not a success, 0 points
  T2  the reference optimum: compliant and improving -> success, 100 points
  T3  a COMPLIANT design with a modest improvement -> success, scored in a low band
  T4  a NON-COMPLIANT design whose D is HIGHER than T3's -> 0 points, success False
      (this is the exact case audit finding #3 reported; under ruler 1.x it scored and was a success)
  T5  the highest-D non-compliant design in the whole neighbourhood -> 0 points, success False
  T6  a design sitting just INSIDE a limit -> compliant, scored
  T7  the same slot pushed just OUTSIDE that limit -> 0 points
  T8  T4 and T3 submitted in the same run -> the run scores T3, not T4
  T9  an over-budget design that is fully compliant -> 0 points (gate 2 still binds)

Run:
    $PY scripts/test_boundary_cases.py        # exits non-zero on any failure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_space import (VectorOracle, budget_mask, compliance_mask,  # noqa: E402
                          describe_row, enumerate_rows, qualifying_mask, to_design)
from evaluator import evaluate_design, evaluate_reference_start, scorer_from_config  # noqa: E402
from formulation import START                                          # noqa: E402
from scoring import final_score, is_cqa_compliant                      # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print("   [%s] %s%s" % ("ok " if ok else "FAIL", label, ("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml")
                         .read_text(encoding="utf-8"))
    obj, cons, scoring_cfg = cfg["objective"], cfg["constraints"], cfg["scoring"]
    ref_cfg = cfg["reference"]
    start_eval = evaluate_reference_start(obj)
    sc = scorer_from_config(obj)

    def score(designs):
        evs = [evaluate_design(d, obj) for d in designs]
        return final_score(start_eval=start_eval, round_evaluations=evs, constraints=cons,
                           scoring_cfg=scoring_cfg, reference_cfg=ref_cfg,
                           release_gate_cfg=cfg.get("release_gate")), evs

    print("=" * 98)
    print("locating real boundary designs in the enumerated neighbourhood")
    O = VectorOracle()
    R = enumerate_rows(int(cons["max_slot_changes"]["max"]))
    Y = O.predict_rows(R)
    D = sc.D_of(Y)
    D0 = float(start_eval.desirability)
    floor = float(cons["composition_overlap"]["min"])
    BUDGET = budget_mask(R, floor)
    IN = compliance_mask(Y, sc)
    QUAL = qualifying_mask(R, Y, sc, D0, floor)

    # the highest-D non-compliant design inside the budget
    j_worst = int(np.argmax(np.where(BUDGET & ~IN, D, -1.0)))
    # a modest compliant improvement: the qualifying design closest to D = 0.012
    q_idx = np.where(QUAL)[0]
    j_mid = int(q_idx[np.argmin(np.abs(D[q_idx] - 0.012))])
    # a non-compliant design with D just above j_mid's D
    cand = np.where(BUDGET & ~IN & (D > D[j_mid]))[0]
    j_cheat = int(cand[np.argmin(D[cand])])
    # the compliant design that sits closest to one of its own limits (max u, still < 1)
    U = sc.u_of(Y[q_idx]).max(axis=1)
    j_edge = int(q_idx[int(np.argmax(np.where(U < 1.0, U, -1.0)))])
    print("   highest-D non-compliant in budget : D=%.6f  %s" % (D[j_worst], describe_row(R[j_worst])))
    print("   modest compliant improvement      : D=%.6f  %s" % (D[j_mid], describe_row(R[j_mid])))
    print("   non-compliant with a HIGHER D     : D=%.6f  %s" % (D[j_cheat], describe_row(R[j_cheat])))
    print("   compliant, closest to a limit     : D=%.6f  u_max=%.4f  %s"
          % (D[j_edge], sc.u_of(Y[j_edge][None, :])[0].max(), describe_row(R[j_edge])))

    d_start = dict(START)
    d_ref = dict(ref_cfg["best_design_in_budget"])
    d_mid, d_cheat, d_worst, d_edge = (to_design(R[j]) for j in (j_mid, j_cheat, j_worst, j_edge))

    print("\nT1  the starting formulation: compliant, but no improvement")
    s, evs = score([d_start])
    check("compliant", is_cqa_compliant(evs[0]), "n_cqa_in_spec=%s" % evs[0].n_cqa_in_spec)
    check("success is False", s["success"] is False)
    check("total_score == 0", s["total_score"] == 0, str(s["total_score"]))

    print("\nT2  the reference optimum: compliant and improving")
    s, evs = score([d_ref])
    check("compliant", is_cqa_compliant(evs[0]))
    check("success is True", s["success"] is True)
    check("total_score == 100", s["total_score"] == 100, str(s["total_score"]))

    print("\nT3  a modest compliant improvement")
    s3, evs = score([d_mid])
    check("compliant", is_cqa_compliant(evs[0]))
    check("success is True", s3["success"] is True)
    check("scored in a low band, 0 < pts < 50", 0 < s3["total_score"] < 50,
          "%d pts at achievement %.5f"
          % (s3["total_score"], s3["optimality_score_details"]["achievement_fraction"]))

    print("\nT4  a NON-compliant design with a HIGHER desirability than T3  <-- audit finding #3")
    s4, evs = score([d_cheat])
    check("not compliant", not is_cqa_compliant(evs[0]),
          "n_cqa_in_spec=%s, missed %s" % (evs[0].n_cqa_in_spec, evs[0].cqa_out_of_spec))
    check("D really is higher than T3's", evs[0].desirability > D[j_mid],
          "%.6f > %.6f" % (evs[0].desirability, D[j_mid]))
    check("inside the change budget (so ONLY the release gate blocks it)",
          all(s4["constraint_diagnostics_for_best_by_objective"]["constraint_flags"].values()))
    check("success is False", s4["success"] is False)
    check("total_score == 0", s4["total_score"] == 0, str(s4["total_score"]))
    check("reported under release_gate.blocked_by_cqa_compliance",
          s4["release_gate"]["n_blocked_by_cqa_compliance_only"] == 1)
    check("continuous metrics still record its unclipped achievement",
          s4["continuous_metrics"]["best_by_desirability_raw_achievement"] is not None,
          "raw achievement %.5f, compliant=%s"
          % (s4["continuous_metrics"]["best_by_desirability_raw_achievement"],
             s4["continuous_metrics"]["best_by_desirability_was_compliant"]))

    print("\nT5  the highest-D non-compliant design in the whole neighbourhood")
    s, evs = score([d_worst])
    check("not compliant", not is_cqa_compliant(evs[0]),
          "missed %s" % evs[0].cqa_out_of_spec)
    # It is worth recording that the steep out-of-window tail keeps every non-compliant design BELOW
    # the compliant argmax, so the reference optimum is also the global argmax of the neighbourhood.
    # The design still clears the achievement-0.45 threshold, i.e. it would have been worth 44/100.
    d45 = D0 + 0.45 * (float(ref_cfg["best_objective_in_budget"]) - D0)
    check("its D would have been worth >= 44/100 if it had been compliant",
          evs[0].desirability > d45,
          "D=%.6f, the 0.45-achievement threshold is %.6f" % (evs[0].desirability, d45))
    check("no non-compliant design beats the compliant reference optimum (steep tail)",
          float(np.max(np.where(BUDGET & ~IN, D, -1.0)))
          < float(ref_cfg["best_objective_in_budget"]),
          "max non-compliant D=%.6f < reference %.6f"
          % (float(np.max(np.where(BUDGET & ~IN, D, -1.0))),
             float(ref_cfg["best_objective_in_budget"])))
    check("total_score == 0 anyway", s["total_score"] == 0, str(s["total_score"]))
    check("success is False", s["success"] is False)

    print("\nT6/T7  just inside a limit vs just outside it")
    s6, evs6 = score([d_edge])
    # u_per_cqa in the report is rounded to 5 dp, so take the raw distance for the assertion.
    u6 = float(sc.u_of(Y[j_edge][None, :])[0].max())
    check("T6 compliant with u_max just under 1.0", is_cqa_compliant(evs6[0]) and u6 < 1.0,
          "u_max=%.8f (1.0 IS the printed specification limit)" % u6)
    check("T6 scores", s6["total_score"] > 0, "%d pts" % s6["total_score"])
    # push the same design one notch further on the slot that owns its worst CQA until it breaches
    worst_key = max(evs6[0].u_per_cqa, key=lambda k: evs6[0].u_per_cqa[k])
    near = np.where(BUDGET & ~IN)[0]
    same = [int(j) for j in near if worst_key in (evaluate_design(to_design(R[j]), obj).cqa_out_of_spec
                                                  or [])]
    if same:
        j7 = int(same[int(np.argmax(D[same]))])
        s7, evs7 = score([to_design(R[j7])])
        check("T7 breaches the same CQA (%s) and scores 0" % worst_key,
              (not is_cqa_compliant(evs7[0])) and s7["total_score"] == 0,
              "u(%s)=%.4f, D=%.6f -> %d pts"
              % (worst_key, evs7[0].u_per_cqa[worst_key], evs7[0].desirability,
                 s7["total_score"]))
    else:
        check("T7 found a design breaching %s" % worst_key, False, "none in the neighbourhood")

    print("\nT8  T4 and T3 in the same run: the run must score T3")
    s8, _ = score([d_cheat, d_mid])
    check("total_score == T3's score", s8["total_score"] == s3["total_score"],
          "%d vs %d" % (s8["total_score"], s3["total_score"]))
    check("best_successful is the compliant one",
          abs(s8["best_successful_objective"] - D[j_mid]) < 1e-9)
    check("num_successful_designs == 1", s8["num_successful_designs"] == 1)

    print("\nT9  an over-budget but fully compliant design: gate 2 still binds")
    over = dict(d_ref)
    over["T5_meg_pct"] = 0.25          # a fifth changed condition
    s9, evs9 = score([over])
    check("5 conditions changed", evs9[0].n_slots_changed == 5, str(evs9[0].n_slots_changed))
    check("still compliant", is_cqa_compliant(evs9[0]), str(evs9[0].cqa_out_of_spec))
    check("total_score == 0", s9["total_score"] == 0, str(s9["total_score"]))
    check("success is False", s9["success"] is False)

    out = {
        "generated_by": "scripts/test_boundary_cases.py",
        "ruler_version": cfg.get("ruler_version"),
        "audit_finding": "#3 -- success verdict inconsistent with CQA compliance",
        "cases": {
            "T1_start_compliant_no_improvement": 0,
            "T2_reference_optimum": 100,
            "T3_modest_compliant_improvement": s3["total_score"],
            "T4_noncompliant_higher_D_than_T3": 0,
            "T5_highest_D_noncompliant": 0,
            "T6_compliant_just_inside_a_limit": s6["total_score"],
            "T8_T4_and_T3_together": s8["total_score"],
            "T9_over_budget_but_compliant": 0,
        },
        "T4_detail": {"design": d_cheat,
                      "desirability": evaluate_design(d_cheat, obj).desirability,
                      "cqa_out_of_spec": evaluate_design(d_cheat, obj).cqa_out_of_spec,
                      "T3_desirability": float(D[j_mid]),
                      "T3_points": s3["total_score"]},
        "verdict": "FAIL" if FAILURES else "PASS",
    }
    (ROOT / "answers" / "boundary_cases.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 98)
    if FAILURES:
        print("BOUNDARY TESTS FAILED: %s" % "; ".join(FAILURES))
        raise SystemExit(1)
    print("BOUNDARY TESTS PASSED -- no out-of-specification candidate can score or be a success, "
          "and a lower-D compliant design beats a higher-D non-compliant one")
    print("wrote answers/boundary_cases.json")


if __name__ == "__main__":
    main()

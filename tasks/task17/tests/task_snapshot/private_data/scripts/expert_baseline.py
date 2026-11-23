#!/usr/bin/env python
"""Expert and chance baselines under the SAME five-measurement budget a graded model gets.

External review asked for a baseline that "a competent process engineer can stably complete", as
evidence that the task is solvable and that a low model score means poor process decisions rather
than an impossible problem. One baseline cannot answer that, because "expert" is ambiguous between
*an engineer with a validated process model* and *an engineer with five batches and no model*. So
three baselines are measured, all scored through the shipped path
(`src/evaluator.evaluate_design` + `src/scoring.final_score`) exactly as a model run is, with five
submitted designs and no other advantage:

  E1  GREEDY SEQUENTIAL SCREEN (oracle upper bound)
      Five steps. At each step, every single-variable change from the current design that stays inside
      the change budget and remains runnable is measured, and the best is adopted. This is an engineer
      who can run a complete one-factor-at-a-time screen (up to ~50 batches) before each decision, so
      it is an UPPER bound on sequential screening, not a fair five-batch protocol. Reported because
      it answers "is the task solvable by disciplined stepwise work at all".

  E2  BEST OF FIVE RANDOM FEASIBLE IN-BUDGET DESIGNS (chance floor)
      200 independent replicates, fixed seed. What five measurements are worth with no process
      reasoning at all. Reported because a model that does not beat this has demonstrated nothing.

  E3  PRE-REGISTERED TEXTBOOK FIVE-BATCH PROTOCOL (fair, judgement-based)
      Five designs fixed in this file, chosen from the mechanism table printed in question.md -- the
      information a graded model has -- and NOT from any measured response. Honesty caveat: the author
      of this file has seen the response model, so E3 cannot be presented as a blind protocol. It is
      reported as an illustration of what stacking textbook moves achieves, with that caveat attached,
      and E1/E2 are the two baselines that carry no such caveat.

Run:
    python scripts/expert_baseline.py [--replicates 200]

Writes answers/expert_baseline.json (designer-side).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                   # noqa: E402
import yaml                                                          # noqa: E402

import agglomeration as A                                            # noqa: E402
from design_space import iter_budget_designs                         # noqa: E402
from evaluator import evaluate_design, evaluate_reference_start      # noqa: E402
from scoring import final_score                                      # noqa: E402

TASK_CFG = os.path.join(ROOT, "tasks", "q1_agglomeration", "task_config.yaml")
OUT = os.path.join(ROOT, "answers", "expert_baseline.json")


def cfg() -> Dict[str, Any]:
    with open(TASK_CFG) as fh:
        return yaml.safe_load(fh)


# =================================================================================================
# E3: the pre-registered protocol, written from the mechanism table in question.md
# =================================================================================================
def e3_protocol() -> List[Tuple[str, Dict[str, Any], str]]:
    """Five batches a textbook reading of the twelve mechanism lines suggests, in order.

    The reasoning is written out so a reviewer can check it against question.md rather than against
    the response model: d50 must rise, strength must fall, porosity must rise, yield must rise.
    """
    def mv(**kw):
        x = dict(A.START)
        x.update(kw)
        return x

    return [
        ("batch_1_grow_by_time_and_grade",
         mv(S7_agglomeration_min=120, S8_primary_d50_um=20),
         "d50 is the furthest out and carries double weight. The two cheapest levers the mechanism "
         "table ties to size are hold time (advances the agglomeration-breakage balance) and a "
         "coarser, blockier primary grade. Cost 1 + 1 = 2."),
        ("batch_2_soften_by_dropping_shear",
         mv(S5_tip_speed_m_s=0.9, S7_agglomeration_min=120, S8_primary_d50_um=20),
         "strength is too high and porosity too low; the table says tip speed consolidates the wet "
         "agglomerate, so less shear should raise porosity and lower strength while keeping size. "
         "Cost 1 + 1 + 1 = 3."),
        ("batch_3_more_bridging_liquid",
         mv(S2_bridging_ratio_v_w=0.47, S5_tip_speed_m_s=0.9, S7_agglomeration_min=120,
            S8_primary_d50_um=20),
         "raise bridging-liquid saturation to push size further, staying below the droplet limit. "
         "Cost 2 + 1 + 1 + 1 = 5."),
        ("batch_4_slower_addition_for_fewer_nuclei",
         mv(S3_addition_mode="90_min", S5_tip_speed_m_s=0.9, S7_agglomeration_min=120,
            S8_primary_d50_um=20, S2_bridging_ratio_v_w=0.41),
         "fewer nuclei per unit bridging liquid should give a coarser product; keep the ratio one "
         "level lower to stay inside the budget. Cost 1 + 1 + 1 + 1 + 2 = 6."),
        ("batch_5_gentler_drying_for_porosity_and_yield",
         mv(S3_addition_mode="90_min", S5_tip_speed_m_s=0.9, S8_primary_d50_um=20,
            S11_drying="40C_vac_12h"),
         "porosity is still the weak one: less capillary shrinkage and less thermal cracking should "
         "raise it and cut the crack-driven yield loss. Cost 1 + 1 + 1 + 3 = 6."),
    ]


# =================================================================================================
def score_trajectory(designs: List[Dict[str, Any]], c: Dict[str, Any],
                     start_eval) -> Dict[str, Any]:
    evs = [evaluate_design(x, c["objective"]) for x in designs]
    fs = final_score(start_eval, evs, c["constraints"], c["scoring"], c["reference"])
    det = fs["optimality_score_details"]
    cm = fs["continuous_metrics"]
    return {
        "total_score": int(fs["total_score"]),
        "achievement": det["achievement_fraction"],
        "raw_achievement_unclipped": cm["raw_achievement_unclipped"],
        "best_usable_D": det.get("best_successful_objective"),
        "n_usable_rounds": fs["num_successful_designs"],
        "best_round": cm["best_round"],
        "per_round": [{"round": r["round"], "cost": r["change_points_spent"],
                       "changed": r["variables_changed"], "D": r["desirability_D"],
                       "usable": r["usable"]} for r in cm["per_round"]],
    }


def e1_greedy(c: Dict[str, Any], start_eval, n_rounds: int) -> Dict[str, Any]:
    cur = dict(A.START)
    submitted: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    screened_total = 0
    for step in range(1, n_rounds + 1):
        best = (None, -1.0, "")
        screened = 0
        for s in A.SLOTS:
            for v in A.LEVELS[s]:
                if v == cur[s]:
                    continue
                cand = dict(cur)
                cand[s] = v
                if A.edit_cost(cand) > A.BUDGET:
                    continue
                ev = evaluate_design(cand, c["objective"])
                screened += 1
                if ev.valid and float(ev.desirability) > best[1]:
                    best = (cand, float(ev.desirability), f"{s}: {cur[s]!r} -> {v!r}")
        screened_total += screened
        if best[0] is None:
            break
        cur = best[0]
        submitted.append(dict(cur))
        trace.append({"step": step, "adopted": best[2], "D": round(best[1], 9),
                      "single_variable_changes_screened": screened,
                      "cumulative_change_cost": A.edit_cost(cur)})
    res = score_trajectory(submitted, c, start_eval)
    res.update({"name": "E1 greedy sequential screen (oracle upper bound)",
                "n_submitted": len(submitted),
                "offline_evaluations_used": screened_total,
                "trace": trace,
                "caveat": "uses ~%d offline measurements to choose 5 submissions, so it is an upper "
                          "bound on stepwise screening, not a five-batch protocol"
                          % screened_total})
    return res


def e2_random(c: Dict[str, Any], start_eval, n_rounds: int, replicates: int) -> Dict[str, Any]:
    region = [x for x, _ in iter_budget_designs(A.BUDGET, feasible_only=True)]
    rng = np.random.default_rng(20260729)
    scores: List[int] = []
    achs: List[float] = []
    for _ in range(replicates):
        pick = rng.choice(len(region), size=n_rounds, replace=False)
        r = score_trajectory([region[int(i)] for i in pick], c, start_eval)
        scores.append(r["total_score"])
        achs.append(float(r["achievement"]))
    arr = np.array(scores)
    hist = {str(int(v)): int((arr == v).sum()) for v in sorted(set(scores))}
    return {
        "name": "E2 best of five random feasible in-budget designs (chance floor)",
        "replicates": replicates,
        "n_submitted": n_rounds,
        "seed": 20260729,
        "n_designs_sampled_from": len(region),
        "score_mean": round(float(arr.mean()), 3),
        "score_median": float(np.median(arr)),
        "score_p90": float(np.percentile(arr, 90)),
        "score_max": int(arr.max()),
        "score_histogram": hist,
        "achievement_mean": round(float(np.mean(achs)), 6),
        "achievement_p90": round(float(np.percentile(achs, 90)), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    c = cfg()
    n_rounds = int(c["iteration"]["max_rounds"])
    start_eval = evaluate_reference_start(c["objective"])
    print(f"baseline process D = {start_eval.desirability}   reference D = "
          f"{c['reference']['best_objective_in_budget']}   rounds = {n_rounds}")

    print("\n" + "=" * 92)
    print("E1  greedy sequential screen (oracle upper bound)")
    print("=" * 92)
    e1 = e1_greedy(c, start_eval, n_rounds)
    for t in e1["trace"]:
        print(f"  step {t['step']}: adopt {t['adopted']:<52s} D={t['D']:.6f}  "
              f"cost={t['cumulative_change_cost']}  (screened {t['single_variable_changes_screened']})")
    print(f"  SCORE {e1['total_score']}/100   achievement {e1['achievement']}   "
          f"best usable D {e1['best_usable_D']}")

    print("\n" + "=" * 92)
    print(f"E2  best of five random feasible in-budget designs, {args.replicates} replicates")
    print("=" * 92)
    e2 = e2_random(c, start_eval, n_rounds, args.replicates)
    print(f"  mean {e2['score_mean']}/100   median {e2['score_median']}   p90 {e2['score_p90']}   "
          f"max {e2['score_max']}")
    print(f"  score histogram {e2['score_histogram']}")

    print("\n" + "=" * 92)
    print("E3  pre-registered textbook five-batch protocol")
    print("=" * 92)
    proto = e3_protocol()
    e3 = score_trajectory([x for _, x, _ in proto], c, start_eval)
    for (name, _, why), r in zip(proto, e3["per_round"]):
        print(f"  R{r['round']} {name:<44s} cost={r['cost']}  D={r['D']}  "
              f"{'usable' if r['usable'] else 'not usable'}")
    print(f"  SCORE {e3['total_score']}/100   achievement {e3['achievement']}   "
          f"best usable D {e3['best_usable_D']}")
    e3.update({"name": "E3 pre-registered textbook five-batch protocol",
               "protocol": [{"batch": n, "design": x, "reasoning": w} for n, x, w in proto],
               "caveat": "written from the mechanism table in question.md, but by an author who has "
                         "seen the response model; not a blind protocol"})

    out = {
        "generated_by": "scripts/expert_baseline.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "designer_only": True,
        "purpose": "Evidence that the task is solvable under the same five-measurement budget a "
                   "graded model gets, and that the score scale has a meaningful floor and ceiling.",
        "measurement_budget": n_rounds,
        "baseline_process_D": start_eval.desirability,
        "reference_D": c["reference"]["best_objective_in_budget"],
        "scored_through": "src/evaluator.evaluate_design + src/scoring.final_score (the shipped path)",
        "E1_greedy_oracle": e1,
        "E2_random_chance_floor": e2,
        "E3_textbook_protocol": e3,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

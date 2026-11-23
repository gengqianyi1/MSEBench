#!/usr/bin/env python
"""Report the model calibration ONE ROW PER RUN, and say plainly what it can and cannot support.

WHY THIS SCRIPT EXISTS
----------------------
The scores in this benchmark are the best of five submissions produced by a model sampling at
temperature 0.7. That makes a single run a random variable, not a measurement: a repeat of the same
model on the same task legitimately lands on a different score. A sibling problem in this batch
discovered this the hard way -- one of its models scored 17 on the first run and 0 on a repeat, and its
published model ranking inverted when the task text was edited.

Two consequences, both of which this script exists to enforce:

1. **Never aggregate by model name.** A dict keyed by model silently discards replicates. Every row
   below is one run directory. The per-model summary is derived afterwards and always prints n.
2. **Never present a ranking that the sample size cannot support.** The script computes each model's
   observed range and tells you whether the ranges overlap. If they do, the honest statement is "the
   difference is inside the sampling spread", not "model A beats model B".

The reliable evidence about difficulty in this package is therefore NOT the LLM scores: it is the
offline, deterministic measurements -- the three difficulty gates and the three baselines in
answers/expert_baseline.json -- which have no sampling variance at all.

Run:
    python scripts/report_calibration.py
    python scripts/report_calibration.py --json

Writes answers/calibration_report.json (designer-side).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "answers", "calibration_report.json")

HARD_MAX = 50
GATED_MODEL = "deepseek-v4-pro"


def collect() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, name)
        report = os.path.join(d, "final_report.json")
        if not os.path.isdir(d) or not os.path.exists(report) or name == "llm_connection_tests":
            continue
        with open(report) as fh:
            r = json.load(fh)
        api = os.path.join(d, "api_config_used.json")
        cfg = json.load(open(api)) if os.path.exists(api) else {}
        s = r.get("score") or {}
        det = s.get("optimality_score_details") or {}
        cm = s.get("continuous_metrics") or {}
        ti = r.get("tool_interaction") or {}
        ch = r.get("checks") or {}
        rows.append({
            "run_dir": name,
            "model": cfg.get("model"),
            "temperature": cfg.get("temperature"),
            "max_tokens": cfg.get("max_tokens"),
            "total_score": s.get("total_score"),
            "achievement": det.get("achievement_fraction"),
            "best_usable_D": cm.get("best_usable_desirability_D"),
            "best_round": cm.get("best_round"),
            "n_rounds_usable": cm.get("n_rounds_usable"),
            "desirability_trajectory": cm.get("desirability_trajectory"),
            "n_tool_calls": ti.get("n_tool_calls"),
            "n_tool_turns": ti.get("n_tool_turns"),
            "n_tool_errors": ti.get("n_tool_errors"),
            "n_parse_failures": ch.get("n_parse_failures"),
            "n_llm_call_failures": ch.get("n_llm_call_failures"),
            "n_truncated": ch.get("n_truncated_llm_responses"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print the JSON instead of the table")
    args = ap.parse_args()

    rows = collect()
    if not rows:
        print("no run directories with a final_report.json under runs/")
        return 1

    # per-model, with n always shown
    per: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        per.setdefault(str(r["model"]), []).append(r)
    for m in per:
        per[m].sort(key=lambda r: r["run_dir"])
        for i, r in enumerate(per[m], start=1):
            r["replicate"] = i

    print("=" * 108)
    print("ONE ROW PER RUN (never aggregated by model -- a repeat run is a separate observation)")
    print("=" * 108)
    print(f"{'model':16s} {'rep':>3} {'T':>4} {'score':>5} {'achievement':>12} {'bestD':>9} "
          f"{'usable':>6} {'bestR':>5} {'tcalls':>6} {'terr':>4} {'parseF':>6}")
    for m in sorted(per):
        for r in per[m]:
            print(f"{m:16s} {r['replicate']:>3} {str(r['temperature']):>4} "
                  f"{str(r['total_score']):>5} {r['achievement']:>12.6f} "
                  f"{(r['best_usable_D'] or 0):>9.6f} {str(r['n_rounds_usable']):>6} "
                  f"{str(r['best_round']):>5} {str(r['n_tool_calls']):>6} "
                  f"{str(r['n_tool_errors']):>4} {str(r['n_parse_failures']):>6}")

    print("\n" + "=" * 108)
    print("PER MODEL, WITH n")
    print("=" * 108)
    summary: Dict[str, Any] = {}
    for m in sorted(per):
        sc = [r["total_score"] for r in per[m]]
        ac = [r["achievement"] for r in per[m]]
        summary[m] = {
            "n": len(sc), "scores": sc, "achievements": ac,
            "score_min": min(sc), "score_max": max(sc),
            "achievement_min": min(ac), "achievement_max": max(ac),
            "all_below_hard_max": all(s < HARD_MAX for s in sc),
        }
        print(f"  {m:16s} n={len(sc)}  scores={sc}  range=[{min(sc)}, {max(sc)}]  "
              f"achievement range=[{min(ac):.6f}, {max(ac):.6f}]")

    # Can the sample support a ranking?
    print("\n" + "=" * 108)
    print("CAN THIS SAMPLE SUPPORT A MODEL RANKING?")
    print("=" * 108)
    ns = {m: v["n"] for m, v in summary.items()}
    min_n = min(ns.values())
    overlaps: List[str] = []
    names = sorted(summary)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = summary[names[i]], summary[names[j]]
            if a["score_min"] <= b["score_max"] and b["score_min"] <= a["score_max"]:
                overlaps.append(f"{names[i]} {a['scores']} vs {names[j]} {b['scores']}")
    if min_n < 2:
        verdict = ("NO -- n=1 for at least one model. A single run at temperature 0.7 is one draw "
                   "from a distribution, so no ordering can be claimed from it.")
    elif overlaps:
        verdict = ("NO -- the observed score ranges OVERLAP for at least one pair, so the "
                   "between-model difference is inside the sampling spread at this n.")
    else:
        verdict = (f"WEAKLY -- with n={min_n} the observed ranges do not overlap for any pair. That is "
                   f"suggestive, not significant; n={min_n} cannot establish a stable ordering.")
    print(f"  min n across models: {min_n}")
    for o in overlaps:
        print(f"  overlapping ranges: {o}")
    print(f"  VERDICT: {verdict}")
    print("\n  The difficulty evidence that does NOT depend on sampling is offline and deterministic:")
    base_path = os.path.join(ROOT, "answers", "expert_baseline.json")
    baselines = None
    chance = None
    if os.path.exists(base_path):
        b = json.load(open(base_path))
        baselines = {
            "E2_chance_floor_mean": b["E2_random_chance_floor"]["score_mean"],
            "E2_chance_floor_max": b["E2_random_chance_floor"]["score_max"],
            "E3_textbook_protocol": b["E3_textbook_protocol"]["total_score"],
            "E1_oracle_greedy": b["E1_greedy_oracle"]["total_score"],
        }
        print(f"    E2 chance floor (200 replicates): mean {baselines['E2_chance_floor_mean']}, "
              f"max {baselines['E2_chance_floor_max']}")
        print(f"    E3 pre-registered textbook 5-batch protocol: "
              f"{baselines['E3_textbook_protocol']}/100")
        print(f"    E1 oracle greedy hill-climb: {baselines['E1_oracle_greedy']}/100")

        # HOW MUCH OF AN OBSERVED SCORE IS SKILL? The E2 baseline is a 200-replicate empirical
        # distribution of "best of five RANDOM feasible in-budget designs", so it answers directly:
        # what fraction of no-reasoning trajectories reach the score this model reached? A score that
        # chance reaches often is not evidence of process reasoning, however it ranks among the models.
        hist = {int(k): v for k, v in b["E2_random_chance_floor"]["score_histogram"].items()}
        n_rep = sum(hist.values())
        observed = sorted({int(r["total_score"]) for r in rows if r["total_score"] is not None})
        chance = {"n_chance_replicates": n_rep, "p_chance_reaches": {}}
        print(f"\n  P(a purely random 5-design trajectory reaches a given score), from E2's "
              f"{n_rep} replicates:")
        for s in observed + [16, 20, 33]:
            if s in chance["p_chance_reaches"]:
                continue
            ge = sum(v for k, v in hist.items() if k >= s)
            chance["p_chance_reaches"][s] = round(ge / n_rep, 4)
            mark = "  <-- observed" if s in observed else ""
            print(f"    >= {s:>3}: {ge:>3}/{n_rep} = {ge / n_rep:>6.1%}{mark}")
        weak = [s for s in observed if chance["p_chance_reaches"][s] >= 0.20]
        if weak:
            print(f"    NOTE: chance reaches {weak} in >= 20 % of trials, so a single run landing "
                  f"there is not by itself evidence of process reasoning.")

    # HOW MANY RUNS WOULD A RANKING ACTUALLY NEED?
    # The spread of a best-of-five score is large because the score is an extreme-value statistic over
    # five stochastic submissions. E2 gives a 200-replicate empirical distribution of exactly that
    # statistic (over random designs), so its SD is a usable -- and probably conservative -- proxy for
    # run-to-run spread. From it, a textbook two-sample size for 80 % power at alpha = 0.05 shows what
    # ranking these models would cost.
    power = None
    if baselines is not None:
        import math
        hist = {int(k): v for k, v in b["E2_random_chance_floor"]["score_histogram"].items()}
        vals = [k for k, v in hist.items() for _ in range(v)]
        n_v = len(vals)
        mean_v = sum(vals) / n_v
        sd_v = (sum((x - mean_v) ** 2 for x in vals) / (n_v - 1)) ** 0.5
        obs_gaps = sorted({abs(summary[a]["score_max"] - summary[c]["score_min"])
                           for a in summary for c in summary if a != c} | {3})
        power = {"proxy": "SD of the E2 best-of-5 chance distribution",
                 "sd_points": round(sd_v, 2), "n_needed_per_model": {}}
        print("\n" + "=" * 108)
        print("WHAT WOULD A DEFENSIBLE RANKING COST?")
        print("=" * 108)
        print(f"  run-to-run spread proxy: SD = {sd_v:.2f} points "
              f"(E2 best-of-5 distribution, {n_v} replicates, mean {mean_v:.2f})")
        for delta in sorted({d for d in obs_gaps if d > 0} | {3, 6, 9}):
            need = math.ceil(2 * ((1.96 + 0.84) ** 2) * sd_v ** 2 / delta ** 2)
            power["n_needed_per_model"][delta] = need
            print(f"    to establish a {delta:>2}-point true difference (80 % power, alpha 0.05): "
                  f"n >= {need:>3} runs per model")
        # observed per-model spread at the current n, for comparison
        spreads = {m: v["score_max"] - v["score_min"] for m, v in summary.items() if v["n"] > 1}
        if spreads:
            power["observed_within_model_spread"] = spreads
            print(f"  observed within-model spread at this n: {spreads}")
        else:
            print("  observed within-model spread: not measurable at n=1")
        print("  Cost of more replicates, from the measured wall time of one full sweep "
              "(3 models, ~31 min, ~38 API calls):")
        for k in (2, 5, 10, 25):
            print(f"    n={k:>2} per model -> ~{k * 31.4 / 60:.1f} h wall, ~{k * 38} API calls")
        power["cost_per_sweep"] = {"minutes": 31.4, "api_calls": 38}

    # the hard acceptance criterion, per replicate
    print("\n" + "=" * 108)
    print(f"SPONSOR HARD CRITERION: every {GATED_MODEL} replicate must score < {HARD_MAX}")
    print("=" * 108)
    gated = summary.get(GATED_MODEL, {"scores": [], "n": 0})
    over = [s for s in gated["scores"] if s >= HARD_MAX]
    print(f"  {GATED_MODEL}: n={gated['n']}  scores={gated['scores']}  "
          f"{'PASS' if gated['scores'] and not over else 'FAIL'}")
    any_over = {m: [s for s in v["scores"] if s >= HARD_MAX] for m, v in summary.items()}
    any_over = {m: v for m, v in any_over.items() if v}
    print(f"  any model at or above {HARD_MAX}: {any_over or 'none'}")

    out = {
        "generated_by": "scripts/report_calibration.py",
        "designer_only": True,
        "indexing": "one row per run directory; never keyed by model name",
        "sampling_note": ("Scores are the best of five submissions from a model sampling at "
                          "temperature 0.7, so a single run is one draw from a distribution and not "
                          "a measurement. Between-model differences must be read against the "
                          "observed spread, and the deterministic offline baselines are the reliable "
                          "difficulty evidence."),
        "runs": rows,
        "per_model": summary,
        "min_n": min_n,
        "overlapping_score_ranges": overlaps,
        "ranking_supported": verdict,
        "deterministic_baselines": baselines,
        "chance_attainment": chance,
        "power_analysis": power,
        "hard_criterion": {"model": GATED_MODEL, "max_score_exclusive": HARD_MAX,
                           "scores": gated["scores"],
                           "pass": bool(gated["scores"]) and not over},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
    return 0 if (gated["scores"] and not over and not any_over) else 1


if __name__ == "__main__":
    sys.exit(main())

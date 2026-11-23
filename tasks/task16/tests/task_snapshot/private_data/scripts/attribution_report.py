"""Where does the 6-17 point floor come from? Round-3 attribution, answering audit item 16-R3-1.

The third-round audit asked one question and made the answer decisive in advance:

    "汇总所有模型所有轮次相对五个窗口的 signed distance、release-gate-only 次数、合法但未 release
     次数和 qualifying 比例。若失败分散在多个 CQA/预算约束，保留窗口并归类为真实多约束搜索失败；
     只有单一窗口跨模型跨 run 支配地板且非目标能力时，才考虑改 synthetic instance、轮数或该窗口."

So this script produces exactly those numbers and then applies that rule, rather than leaving the
verdict to prose. Two halves:

  OFFLINE   re-enumerates the change-budget neighbourhood through the shipped code path and reports
            how much achievement the qualifying population actually holds. This is the reachability
            evidence any future change to the windows, the round budget or the instance has to be
            argued from.
  RUNS      reads the archived trajectories and classifies every round: not measured / outside the
            design space / measured with N CQAs out of window / qualifying. For the measured rounds
            it reports each window's signed distance in half-window units, so "which window blocks"
            is a number and not an impression.

    python scripts/attribution_report.py --runs <dir containing <model>/task_16_*/logs/native>
    python scripts/attribution_report.py --runs ... --json answers/attribution_round3.json

`--runs` may be omitted to run the offline half alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_space import (POINT_BANDS, Desirability, VectorOracle, achievement,  # noqa: E402
                          budget_mask, calibrate_windows, compliance_mask, describe_row,
                          enumerate_rows, overlap_vec, points_for, qualifying_mask, START_IDX)
from formulation import CQA_KEYS, CQA_SHORT, MIN_FILLER_PCT  # noqa: E402

# Shipping calibration -- kept identical to enumerate_reference_best.py. If these drift the two
# scripts disagree about what the instance is, which is the failure mode this whole file exists
# to prevent, so they are asserted against the frozen reference file below.
PULL, MARG = 0.34, 0.55
M, E, KAPPA, DFLOOR, PW = 6.3, 2.0e-4, 2.6, 1.0e-6, 0.0
FLOOR = 0.88
MAX_SLOT_CHANGES, MAX_IDENTITY_CHANGES = 4, 1


def offline_reachability() -> dict:
    """Enumerate the neighbourhood and report what the qualifying population is worth."""
    O = VectorOracle()
    R = enumerate_rows(MAX_SLOT_CHANGES)
    Y = O.predict_rows(R)
    y0 = O.predict_rows(START_IDX[None, :])[0]
    win = calibrate_windows(y0, Y, PULL, MARG)
    L, T, U = win["L"], win["T"], win["U"]
    sc = Desirability(L, T, U, hill_exponent=M, d_at_spec_limit=E, tail_decay_multiplier=KAPPA,
                      floor=DFLOOR, plateau_fraction=PW)
    D = sc.D_of(Y)
    D0 = float(sc.D_one(y0))
    BUDGET = budget_mask(R, FLOOR, MAX_SLOT_CHANGES, MAX_IDENTITY_CHANGES, MIN_FILLER_PCT)
    IN = compliance_mask(Y, sc)
    QUAL = qualifying_mask(R, Y, sc, D0, FLOOR, MAX_SLOT_CHANGES, MAX_IDENTITY_CHANGES,
                           MIN_FILLER_PCT)
    jref = int(np.argmax(np.where(BUDGET & IN, D, -1.0)))
    Dref = float(D[jref])

    ach = np.array([achievement(float(v), D0, Dref) for v in D[QUAL]])
    pts = np.array([points_for(float(v), D0, Dref, POINT_BANDS) for v in D[QUAL]])

    # Per-window blame over the budget-feasible but non-compliant population: this is the offline
    # counterpart of the run-level blame table, and it says whether one window is structurally
    # harder to close than the others rather than merely unlucky in four runs.
    NONCOMP = BUDGET & ~IN
    u = sc.u_of(Y[NONCOMP])                     # |u| > 1 means outside the printed window
    offline_blame = {CQA_KEYS[i]: int((np.abs(u[:, i]) > 1.0).sum()) for i in range(len(CQA_KEYS))}

    band_pop = Counter(int(p) for p in pts)
    return {
        "n_enumerated": int(len(R)),
        "n_budget_feasible": int(BUDGET.sum()),
        "n_all_cqa_in_spec": int((BUDGET & IN).sum()),
        "n_qualifying": int(QUAL.sum()),
        "five_window_closure_rate": float((BUDGET & IN).sum() / BUDGET.sum()),
        "qualifying_rate_of_budget_feasible": float(QUAL.sum() / BUDGET.sum()),
        "start_value": D0,
        "reference_best": Dref,
        "reference_design": describe_row(R[jref]),
        "reference_overlap": float(overlap_vec(R[jref][None, :])[0]),
        "achievement_of_qualifying": {
            "min": float(ach.min()), "median": float(np.median(ach)), "max": float(ach.max()),
            "mean": float(ach.mean()),
            "deciles": [float(np.quantile(ach, q / 10)) for q in range(1, 10)],
            "share_at_or_above": {f"{t:g}": float((ach >= t).mean())
                                  for t in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80)},
        },
        "band_population": {str(k): v for k, v in sorted(band_pop.items())},
        "largest_band_share": max(band_pop.values()) / int(QUAL.sum()),
        "noncompliant_blame_counts": offline_blame,
        "n_budget_feasible_noncompliant": int(NONCOMP.sum()),
        "windows": {CQA_KEYS[i]: {"lower": float(L[i]), "target": float(T[i]), "upper": float(U[i])}
                    for i in range(len(CQA_KEYS))},
    }


def _signed_norm(lo: float, up: float, v: float) -> float:
    """Signed distance to the window in half-window units. 0 inside, negative below, positive above."""
    half = (up - lo) / 2.0
    if v < lo:
        return (v - lo) / half
    if v > up:
        return (v - up) / half
    return 0.0


def run_attribution(runs_root: Path, windows: dict, start_value: float) -> dict:
    rounds: list[dict] = []
    for traj in sorted(glob.glob(str(runs_root / "*" / "task_16_*" / "logs" / "native"
                                    / "trajectory.jsonl"))):
        model = Path(traj).parents[3].name
        for line in open(traj, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            ev = rec.get("evaluation") or {}
            rounds.append({
                "model": model,
                "round": rec.get("round"),
                "parse_error": rec.get("parse_error"),
                "in_design_space": bool(ev.get("valid")),
                "cqa": ev.get("cqa"),
                "desirability": ev.get("desirability"),
                "n_cqa_in_spec": ev.get("n_cqa_in_spec"),
                "cqa_out_of_spec": ev.get("cqa_out_of_spec") or [],
                "composition_overlap": ev.get("composition_overlap"),
                "n_slots_changed": ev.get("n_slots_changed"),
            })

    measured = [r for r in rounds if r["cqa"] is not None and r["in_design_space"]]
    qualifying = [r for r in measured if r["n_cqa_in_spec"] == 5
                  and (r["desirability"] or 0.0) > start_value]
    # The audit's own category: budget-legal, D improved on the baseline, but at least one CQA
    # outside its window -- i.e. blocked by the release gate and nothing else.
    gate_only = [r for r in measured if r["n_cqa_in_spec"] < 5
                 and (r["desirability"] or 0.0) > start_value]
    legal_not_released = [r for r in measured if r["n_cqa_in_spec"] < 5]

    status = Counter()
    for r in rounds:
        if r["cqa"] is None:
            status["NOT_MEASURED"] += 1
        elif not r["in_design_space"]:
            status["OUTSIDE_DESIGN_SPACE"] += 1
        elif r["n_cqa_in_spec"] == 5:
            status["QUALIFYING"] += 1
        else:
            status[f"OUT_OF_SPEC_{5 - r['n_cqa_in_spec']}_CQA"] += 1

    blame = Counter()
    for r in measured:
        for name in r["cqa_out_of_spec"]:
            blame[name] += 1

    dist = {}
    for name, w in windows.items():
        ds = [_signed_norm(w["lower"], w["upper"], r["cqa"][name]) for r in measured]
        dist[name] = {
            "mean": statistics.mean(ds), "median": statistics.median(ds),
            "worst": max(ds, key=abs), "n_inside": sum(1 for d in ds if d == 0.0),
            "n_measured": len(ds),
            "blame_count": blame.get(name, 0),
            "blame_share_of_measured": blame.get(name, 0) / len(ds) if ds else 0.0,
        }

    per_model = defaultdict(lambda: {"rounds": 0, "measured": 0, "qualifying": 0,
                                     "gate_only": 0, "best_qualifying_D": None})
    for r in rounds:
        m = per_model[r["model"]]
        m["rounds"] += 1
        if r["cqa"] is not None and r["in_design_space"]:
            m["measured"] += 1
            if r["n_cqa_in_spec"] == 5 and (r["desirability"] or 0.0) > start_value:
                m["qualifying"] += 1
                if m["best_qualifying_D"] is None or r["desirability"] > m["best_qualifying_D"]:
                    m["best_qualifying_D"] = r["desirability"]
            elif (r["desirability"] or 0.0) > start_value:
                m["gate_only"] += 1

    return {
        "n_rounds": len(rounds),
        "status_counts": dict(status),
        "n_measured": len(measured),
        "n_qualifying": len(qualifying),
        "qualifying_share_of_measured": len(qualifying) / len(measured) if measured else 0.0,
        "n_release_gate_only": len(gate_only),
        "n_legal_but_not_released": len(legal_not_released),
        "simultaneous_out_of_spec": dict(Counter(5 - r["n_cqa_in_spec"] for r in measured)),
        "per_window": dist,
        "per_model": {k: dict(v) for k, v in sorted(per_model.items())},
        "rounds": rounds,
    }


def verdict(offline: dict, runs: dict | None) -> dict:
    """Apply the audit's stated decision rule instead of restating it in prose."""
    if runs is None:
        return {"verdict": "NO_RUN_EVIDENCE", "reason": "--runs not supplied"}
    blame = {k: v["blame_count"] for k, v in runs["per_window"].items()}
    total_blame = sum(blame.values())
    top, top_n = max(blame.items(), key=lambda kv: kv[1]) if total_blame else (None, 0)
    dominates = total_blame > 0 and top_n / total_blame > 0.5 and blame[top] == total_blame
    spread = sum(1 for v in blame.values() if v > 0)
    return {
        "single_window_dominates_floor": bool(dominates),
        "n_windows_implicated": spread,
        "top_window": top,
        "top_window_share_of_all_blame": (top_n / total_blame) if total_blame else 0.0,
        "release_gate_is_a_floor": runs["n_qualifying"] == 0,
        "verdict": ("RECALIBRATE_WINDOW" if dominates else
                    "KEEP_WINDOWS_GENUINE_MULTI_CONSTRAINT_FAILURE"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="directory holding <model>/task_16_*/logs/native")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("=" * 96)
    print("offline reachability -- enumerated through the shipped oracle and the shipped predicates")
    off = offline_reachability()
    ref = json.loads((ROOT / "answers" / "reference_enumeration.json").read_text(encoding="utf-8"))
    # Cross-assert against the frozen reference so a calibration drift in this file is loud.
    assert abs(off["reference_best"] - ref["reference"]["value"]) < 1e-12, "reference drift"
    assert abs(off["start_value"] - ref["start"]["value"]) < 1e-12, "start drift"
    assert off["n_qualifying"] == ref["neighbourhood"]["n_qualifying"], "qualifying-set drift"

    a = off["achievement_of_qualifying"]
    print(f"   budget-feasible {off['n_budget_feasible']:,}   all-CQA-in-spec "
          f"{off['n_all_cqa_in_spec']:,} ({off['five_window_closure_rate']:.1%} of budget-feasible)"
          f"   qualifying {off['n_qualifying']:,}")
    print(f"   start D = {off['start_value']:.9g}   reference D = {off['reference_best']:.9g}")
    print(f"   achievement over the {off['n_qualifying']:,} qualifying designs: "
          f"min {a['min']:.4f}  median {a['median']:.4f}  mean {a['mean']:.4f}  max {a['max']:.4f}")
    print("   share of qualifying designs at or above achievement x:")
    for t, s in a["share_at_or_above"].items():
        print(f"        >= {t:>4}   {s:7.2%}")
    print(f"   largest single band holds {off['largest_band_share']:.1%} of the qualifying set")
    print("   band population:", off["band_population"])
    print("   which window is open least often among budget-feasible non-compliant designs "
          f"(n = {off['n_budget_feasible_noncompliant']:,}):")
    for k, v in sorted(off["noncompliant_blame_counts"].items(), key=lambda kv: -kv[1]):
        print(f"        {k:<24} {v:>7,}  ({v / off['n_budget_feasible_noncompliant']:.1%})")

    runs = None
    if args.runs:
        print("\n" + "=" * 96)
        print("archived-run attribution")
        runs = run_attribution(Path(args.runs), off["windows"], off["start_value"])
        print(f"   {runs['n_rounds']} rounds over {len(runs['per_model'])} runs; "
              f"{runs['n_measured']} measured")
        for k, v in sorted(runs["status_counts"].items(), key=lambda kv: -kv[1]):
            print(f"        {v:3d}  {k}")
        print(f"   qualifying rounds {runs['n_qualifying']} "
              f"({runs['qualifying_share_of_measured']:.0%} of measured)")
        print(f"   release-gate-only rounds (budget-legal, D improved, >=1 CQA out) "
              f"{runs['n_release_gate_only']}")
        print("   per window, over the measured rounds:")
        print(f"        {'CQA':<24}{'blame':>6}{'mean sd':>10}{'median sd':>11}{'worst sd':>10}"
              f"{'inside':>9}")
        for k, v in runs["per_window"].items():
            print(f"        {k:<24}{v['blame_count']:>6}{v['mean']:>10.3f}{v['median']:>11.3f}"
                  f"{v['worst']:>10.3f}{v['n_inside']:>6}/{v['n_measured']}")
        print("   simultaneous out-of-window CQAs:", runs["simultaneous_out_of_spec"])
        print("   per model:")
        for m, v in runs["per_model"].items():
            print(f"        {m:<18} rounds {v['rounds']}  measured {v['measured']}  "
                  f"qualifying {v['qualifying']}  gate-only {v['gate_only']}  "
                  f"best qualifying D {v['best_qualifying_D']}")

    v = verdict(off, runs)
    print("\n" + "=" * 96)
    print("verdict (the audit's own rule, applied mechanically)")
    for k, val in v.items():
        print(f"   {k} = {val}")

    if args.json:
        payload = {"offline": off, "runs": runs, "verdict": v}
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Where do the 9-12 points go? Round-3 attribution, answering audit item 17-R3-1.

The third-round audit set the question and the decision rule in advance:

    "对全部 run 统计四类轮次比例：non-design-space、合法但不可运行、有效测量但低 D、successful
     design；分别列出主要 failure code。若主要损失在前两类，归为 instruction/search failure；若各
     模型多为有效测量但五轮普遍无法到达成功区，再评估增加轮次、调整初始点或重新生成 synthetic
     instance。不应继续单独增加低分 band 解决可达性。"

So this script classifies every archived round into exactly those four buckets, puts each model's
achievement on the enumerated population of improving designs, and compares both against the
deterministic offline baselines that `enumerate_reference_best.py` already measured. The verdict
follows mechanically from those numbers.

The offline half is read from `answers/reference_enumeration.json` rather than recomputed: that file
is the frozen designer-side enumeration (96 124 feasible in-budget designs, 19 779 of them strictly
improving) and re-deriving it here would only create a second source of truth for the same fact.

    python scripts/attribution_report.py --runs <dir containing <model>/task_17_*/logs/native>
    python scripts/attribution_report.py --runs ... --json answers/attribution_round3.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "answers" / "reference_enumeration.json"


def percentile_of(achievement: float, share_at_or_above: dict[str, float]) -> float:
    """Where this achievement sits among the enumerated improving designs, in percent.

    `share_at_or_above` is sparse, so this interpolates between the two bracketing thresholds
    rather than pretending to a precision the table does not carry.
    """
    pts = sorted((float(k), v) for k, v in share_at_or_above.items())
    if achievement <= pts[0][0]:
        return 100.0 * (1.0 - pts[0][1])
    if achievement >= pts[-1][0]:
        return 100.0 * (1.0 - pts[-1][1])
    for (x0, s0), (x1, s1) in zip(pts, pts[1:]):
        if x0 <= achievement <= x1:
            t = 0.0 if x1 == x0 else (achievement - x0) / (x1 - x0)
            return 100.0 * (1.0 - (s0 + t * (s1 - s0)))
    return float("nan")


def classify(rec: dict, start_value: float) -> tuple[str, str | None]:
    """The audit's four buckets, plus the failure code that put the round there."""
    ev = rec.get("evaluation") or {}
    if rec.get("parse_error"):
        return "NOT_MEASURED", f"PARSE:{str(rec['parse_error'])[:60]}"
    if not ev:
        return "NOT_MEASURED", "NO_EVALUATION_RECORD"
    if not ev.get("valid"):
        # Outside the design space: an illegal level, an unknown slot, or over the edit budget.
        code = ev.get("error") or "INVALID_DESIGN"
        cost, budget = ev.get("edit_cost"), ev.get("edit_cost_budget")
        if isinstance(cost, (int, float)) and isinstance(budget, (int, float)) and cost > budget:
            code = f"EDIT_BUDGET_EXCEEDED({cost}>{budget})"
        return "OUTSIDE_DESIGN_SPACE", str(code)[:80]
    violations = ev.get("process_constraint_violations") or []
    if violations:
        # Legal levels and inside the edit budget, but the process itself cannot be run.
        return "LEGAL_BUT_NOT_RUNNABLE", ",".join(str(v) for v in violations)[:80]
    D = ev.get("desirability")
    if D is None:
        return "NOT_MEASURED", "NO_DESIRABILITY"
    if D <= start_value:
        return "MEASURED_NO_IMPROVEMENT", f"D={D:.6g} <= start {start_value:.6g}"
    return "SUCCESSFUL_DESIGN", None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="directory holding <model>/task_17_*/logs/native/trajectory.jsonl")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    start_value = float(ref["start_value"])
    reference_best = float(ref["reference_best"])
    g3 = ref["gates"]["g3"]
    dist = g3["achievement_distribution_over_improving_designs"]
    share = dist["share_at_or_above"]
    gates = ref["gates"]

    print("=" * 96)
    print("offline reachability (frozen enumeration, answers/reference_enumeration.json)")
    print(f"   feasible in-budget designs      {g3['n_feasible_in_budget']:,}")
    print(f"   strictly improving on the start {g3['n_strictly_improving']:,}")
    print(f"   start D {start_value:.9g}   reference D {reference_best:.9g}")
    print(f"   achievement quantiles over improving designs: {dist['quantiles']}")
    print(f"   points ladder over improving designs: {g3['points_ladder']}")
    print("\n   deterministic offline baselines (what a fixed policy reaches, no LLM involved):")
    for tag, label in (("g1_one_slot", "best single-slot change"),
                       ("g1_two_slot", "best two-slot change"),
                       ("g2", "best under the raised overlap floor"),
                       ("g1_three_slot_diagnostic", "best three-slot change (diagnostic)")):
        g = gates[tag]
        print(f"        {label:<38} {g['best_points']:>3} points   achievement "
              f"{g['best_achievement']:.4f}   ({percentile_of(g['best_achievement'], share):.0f}th pct)")

    # ---------------------------------------------------------------- archived rounds
    rounds: list[dict] = []
    for traj in sorted(glob.glob(str(Path(args.runs) / "*" / "task_17_*" / "logs" / "native"
                                     / "trajectory.jsonl"))):
        model = Path(traj).parents[3].name
        for line in open(traj, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            bucket, code = classify(rec, start_value)
            ev = rec.get("evaluation") or {}
            rounds.append({
                "model": model, "round": rec.get("round"),
                "bucket": bucket, "failure_code": code,
                "desirability": ev.get("desirability"),
                "n_slots_changed": ev.get("n_slots_changed"),
                "edit_cost": ev.get("edit_cost"),
                "edit_cost_budget": ev.get("edit_cost_budget"),
                "n_tool_calls": rec.get("n_tool_calls"),
            })

    print("\n" + "=" * 96)
    print("archived-run attribution -- the audit's four buckets")
    buckets = Counter(r["bucket"] for r in rounds)
    order = ["OUTSIDE_DESIGN_SPACE", "LEGAL_BUT_NOT_RUNNABLE", "MEASURED_NO_IMPROVEMENT",
             "SUCCESSFUL_DESIGN", "NOT_MEASURED"]
    for b in order:
        n = buckets.get(b, 0)
        print(f"   {n:3d} / {len(rounds)}  ({n / len(rounds):5.1%})  {b}")
    codes = Counter(r["failure_code"] for r in rounds if r["failure_code"])
    print("\n   failure codes:")
    for c, n in codes.most_common():
        print(f"        {n:3d}  {c}")

    print("\n   per model:")
    by = defaultdict(list)
    for r in rounds:
        by[r["model"]].append(r)
    per_model = {}
    for m, rs in sorted(by.items()):
        succ = [r for r in rs if r["bucket"] == "SUCCESSFUL_DESIGN"]
        best = max((r["desirability"] for r in succ), default=None)
        a = None if best is None else (best - start_value) / (reference_best - start_value)
        per_model[m] = {
            "rounds": len(rs),
            "buckets": dict(Counter(r["bucket"] for r in rs)),
            "n_successful": len(succ),
            "best_D": best,
            "achievement": a,
            "percentile_among_improving_designs": None if a is None else percentile_of(a, share),
            "n_tool_calls": sum(r["n_tool_calls"] or 0 for r in rs),
        }
        pct = per_model[m]["percentile_among_improving_designs"]
        print(f"        {m:<18} successful {len(succ)}/{len(rs)}  best D "
              f"{best if best is not None else '-'}  achievement "
              f"{'-' if a is None else format(a, '.4f')}  "
              f"{'-' if pct is None else format(pct, '.0f') + 'th pct'}  "
              f"tool calls {per_model[m]['n_tool_calls']}")

    print("\n   round detail:")
    for m, rs in sorted(by.items()):
        for r in rs:
            extra = f" cost {r['edit_cost']}/{r['edit_cost_budget']}" if r["edit_cost"] is not None else ""
            print(f"        {m:<18} R{r['round']}  {r['bucket']:<24}"
                  f" D={r['desirability'] if r['desirability'] is not None else '-'}{extra}"
                  f"{'  ' + r['failure_code'] if r['failure_code'] else ''}")

    # ---------------------------------------------------------------- verdict
    n_first_two = buckets.get("OUTSIDE_DESIGN_SPACE", 0) + buckets.get("LEGAL_BUT_NOT_RUNNABLE", 0)
    n_measured = len(rounds) - buckets.get("NOT_MEASURED", 0)
    model_best = [v["achievement"] for v in per_model.values() if v["achievement"] is not None]
    one_slot = gates["g1_one_slot"]["best_achievement"]
    beaten_by_one_slot = all(a < one_slot for a in model_best) if model_best else False

    print("\n" + "=" * 96)
    print("verdict (the audit's own rule, applied mechanically)")
    print(f"   rounds lost to the first two buckets: {n_first_two}/{len(rounds)} "
          f"({n_first_two / len(rounds):.0%})")
    print(f"   rounds that produced a successful design: {buckets.get('SUCCESSFUL_DESIGN', 0)}"
          f"/{n_measured} of measured")
    print(f"   every model below the best single-slot change ({one_slot:.4f}): {beaten_by_one_slot}")
    if n_first_two / max(1, len(rounds)) > 0.5:
        v = "INSTRUCTION_OR_SEARCH_FAILURE"
    elif buckets.get("SUCCESSFUL_DESIGN", 0) == 0:
        v = "UNREACHABLE_IN_BUDGET_CONSIDER_ROUNDS_OR_INSTANCE"
    elif beaten_by_one_slot:
        v = "REACHABLE_BUT_UNSEARCHED_KEEP_INSTANCE_AND_BUDGET"
    else:
        v = "REACHABLE_AND_PARTLY_SEARCHED"
    print(f"   verdict = {v}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "start_value": start_value, "reference_best": reference_best,
            "offline": {"g3": g3, "gates": {k: gates[k] for k in
                                            ("g1_one_slot", "g1_two_slot", "g2",
                                             "g1_three_slot_diagnostic")}},
            "bucket_counts": dict(buckets), "failure_codes": dict(codes),
            "per_model": per_model, "rounds": rounds, "verdict": v,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

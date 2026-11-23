"""Render a run's trajectory as a plain-text report a reviewer can read without tooling.

Two things this has to get right, both inherited as bugs from an earlier problem in this series:

* The per-round LLM call metadata lives in `llm_api_debug.jsonl`, not in `trajectory.jsonl`. Reading
  only the trajectory made every round look like a failed call ("LLM call failures: 5") when nothing
  had failed. The two files are joined on the round number here.
* The "usable" count must be computed the way the scorer computes it -- every hard constraint
  satisfied AND the objective improved over the baseline formulation -- not read out of the
  feedback. This benchmark's feedback deliberately returns no verdict, so any count taken from it
  would always be zero.

Run:
    python scripts/trajectory_to_txt.py runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import check_constraints, evaluate_design, evaluate_reference_start  # noqa: E402
from utils import load_yaml  # noqa: E402

TASK_CFG = ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml"


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            out.append({"_line": i, "_parse_error": str(exc), "_raw": line[:500]})
    return out


def pick(record, *names, default=None):
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", help="Path to trajectory.jsonl, or the run directory")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    given = Path(args.trajectory)
    if not given.is_absolute():
        given = ROOT / given
    run_dir = given if given.is_dir() else given.parent
    traj_path = given if given.is_file() else run_dir / "trajectory.jsonl"
    out_path = Path(args.output) if args.output else traj_path.with_suffix(".readable.txt")

    records = [r for r in load_jsonl(traj_path) if "round" in r]
    # llm_api_debug.jsonl now carries one record per LLM CALL, and a round can span several calls
    # (the model may spend up to two turns on tool requests before it submits). Keep them all, and
    # use the LAST call of a round as that round's headline -- it is the one that produced the
    # submission. Reading only the first would report a tool-request turn as the round's answer.
    debug_calls = load_jsonl(run_dir / "llm_api_debug.jsonl")
    calls_by_round: dict = {}
    for rec in debug_calls:
        calls_by_round.setdefault(rec.get("round"), []).append(rec.get("llm_call") or {})
    debug_by_round = {k: v[-1] for k, v in calls_by_round.items()}

    # A readable trajectory is a RECORD of a graded run, so it is rendered against the config that
    # run was graded with (every run directory keeps a copy), never against whatever the current
    # calibration happens to be. Rendering a superseded run with a newer ruler would silently mix an
    # old evaluation block with a new verdict -- exactly the kind of internally inconsistent artifact
    # scripts/check_consistency.py exists to catch.
    run_cfg_path = run_dir / "task_config.yaml"
    cfg_path = run_cfg_path if run_cfg_path.exists() else TASK_CFG
    cfg = load_yaml(cfg_path)
    objective_cfg = cfg["objective"]
    constraints = cfg["constraints"]
    try:
        start = evaluate_reference_start(objective_cfg)
        start_D, start_cqa, replay = start.desirability, start.cqa, True
    except ValueError as exc:
        # The run was graded with a ruler this code no longer implements. Everything below then comes
        # from what the run stored; nothing is recomputed.
        start_D, start_cqa, replay = None, None, False
        replay_note = "superseded ruler: " + str(exc).split(". ")[0]

    def usable(record):
        cand = pick(record, "submitted_candidate", default={}) or {}
        form = cand.get("formulation")
        if form is None:
            return False, ["no formulation submitted"]
        if not replay:
            ev_rec = pick(record, "evaluation", default={}) or {}
            if not ev_rec.get("valid"):
                return False, [ev_rec.get("error") or "not manufacturable"]
            # The budget constraints are plain arithmetic on the composition table, so they can be
            # read straight off the stored evaluation; only the desirability needed the old ruler.
            violated = []
            if float(ev_rec.get("composition_overlap")) < float(
                    constraints["composition_overlap"]["min"]) - 1e-9:
                violated.append("composition_overlap")
            if int(ev_rec.get("n_slots_changed")) > int(constraints["max_slot_changes"]["max"]):
                violated.append("max_slot_changes")
            if int(ev_rec.get("n_identity_changes")) > int(
                    constraints["max_identity_changes"]["max"]):
                violated.append("max_identity_changes")
            if float(ev_rec.get("qs_filler_pct")) < float(
                    constraints["mass_closure"]["min_filler_pct"]) - 1e-9:
                violated.append("mass_closure")
            if not ev_rec.get("all_cqa_in_spec"):
                violated.append("CQA out of specification (release gate)")
            if float(ev_rec.get("desirability") or 0.0) <= float(cfg["reference"]["start_objective"]):
                violated.append("D not improved over the starting formulation")
            return not violated, violated
        ev = evaluate_design(form, objective_cfg)
        violated = [k for k, v in check_constraints(ev, constraints).items() if not v]
        # Ruler 2.0.0 release gate: any CQA outside its window disqualifies the submission entirely.
        if not ev.all_cqa_in_spec:
            violated = violated + ["CQA out of specification (release gate): "
                                   + ", ".join(ev.cqa_out_of_spec)]
        if ev.desirability is None or ev.desirability <= start.desirability:
            violated = violated + ["D not improved over the starting formulation"]
        return not violated, violated

    def call_for(record):
        return pick(record, "llm_call", default=debug_by_round.get(record.get("round"), {})) or {}

    flags = [usable(r) for r in records]

    lines = [
        f"Trajectory report -- {cfg['task_name']}",
        "=" * 80,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Run: {run_dir.name}",
        f"Graded against: {cfg_path.relative_to(ROOT)}"
        + ("" if replay else f"  [that ruler is no longer implemented ({replay_note}); every number "
                             f"below is what the run itself stored, nothing was recomputed]"),
        f"Objective: maximize the desirability D   baseline formulation D = "
        + (f"{start_D:.6g}" if start_D is not None else str(cfg["reference"]["start_objective"])),
        *( [f"Baseline CQAs: {json.dumps(start_cqa)}"] if start_cqa else [] ),
        "",
        "Summary",
        "-" * 80,
        f"Rounds: {len(records)}",
        f"LLM call failures: {sum(1 for r in records if not call_for(r).get('ok', False))}",
        f"Truncated LLM responses: {sum(1 for r in records if call_for(r).get('truncated'))}",
        f"Parse failures: {sum(1 for r in records if r.get('parse_error'))}",
        f"Manufacturable submissions: "
        f"{sum(1 for r in records if (pick(r, 'evaluation', default={}) or {}).get('valid'))}",
        f"CQA-compliant submissions (all five windows closed): "
        f"{sum(1 for r in records if (pick(r, 'evaluation', default={}) or {}).get('all_cqa_in_spec'))}",
        f"Qualifying formulations (all constraints + all five CQAs in spec + improves D): "
        f"{sum(1 for ok, _ in flags if ok)}",
        f"LLM calls in total (a round may spend up to 2 turns on tool requests): "
        f"{sum(len(v) for v in calls_by_round.values())}",
        f"Tool calls the model chose to make: "
        f"{sum(len(ti.get('results') or []) for r in records for ti in (r.get('tool_interactions') or []))}",
        f"Tool calls that returned an error: "
        f"{sum(1 for r in records for ti in (r.get('tool_interactions') or []) for x in (ti.get('results') or []) if not x.get('ok'))}",
        "",
    ]

    for r, (ok, violated) in zip(records, flags):
        call = call_for(r)
        evaluation = pick(r, "evaluation", default={}) or {}
        candidate = pick(r, "submitted_candidate", "parsed_candidate")
        raw = pick(r, "raw_llm_response", "llm_raw_response")
        feedback = pick(r, "feedback_returned", "feedback")

        lines += [f"Round {r.get('round')}", "~" * 80,
                  f"LLM ok={call.get('ok')}  model={call.get('model')}  "
                  f"attempts={call.get('attempts')}  finish_reason={call.get('finish_reason')}  "
                  f"truncated={call.get('truncated')}"]
        if call.get("error"):
            lines.append(f"LLM error: {call['error']}")
        interactions = r.get("tool_interactions") or []
        if interactions:
            lines += ["", f"Tool interaction ({len(interactions)} request turn(s)) -- the model's own "
                          f"choice, not the harness's:"]
            for ti in interactions:
                lines.append(f"  turn {ti.get('step')}: requested "
                             f"{json.dumps(ti.get('requested'), ensure_ascii=False)}")
                for res in (ti.get("results") or []):
                    if res.get("ok"):
                        lines.append(f"    -> {res.get('tool')}: ok, keys "
                                     f"{sorted(k for k in res if k not in ('ok', 'tool'))}")
                    else:
                        lines.append(f"    -> {res.get('tool')}: ERROR {res.get('error')} "
                                     f"(recoverable={res.get('recoverable')}) {res.get('detail')}")
                lines.append(f"    budget left after this turn: {ti.get('budget_left')}, "
                             f"tool turns left: {ti.get('tool_turns_left')}")
        else:
            lines += ["", "Tool interaction: none -- the model submitted without calling a tool."]
        lines += ["", "Submitted:",
                  json.dumps(candidate, ensure_ascii=False, indent=2) if candidate
                  else "  [no candidate parsed]"]
        if r.get("parse_error"):
            lines.append(f"Parse error: {r['parse_error']}")
        lines += ["", "Evaluation:", json.dumps(evaluation, ensure_ascii=False, indent=2),
                  f"Verdict: {'USABLE' if ok else 'NOT USABLE -- ' + ', '.join(violated)}",
                  "", "Feedback returned to the model:",
                  json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else "  [none]"]
        if raw:
            lines += ["", "Raw LLM response:", raw]
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved readable report to: {out_path}")


if __name__ == "__main__":
    main()

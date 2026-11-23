"""Render a run's trajectory as a human-readable Markdown report.

The Markdown twin of scripts/trajectory_to_txt.py, for a reviewer who would rather read a rendered
table than a plain-text dump: one summary table, then one section per round with the submitted
formulation, the measured CQAs against the printed windows, the budget bookkeeping, the verdict, and
the raw model output folded into a <details> block.

What this file used to be, and why it changed: it wrote a JSON dict of the user prompts into a file
named `trajectory.conversation.md`. That is neither the conversation (the assistant turns were
dropped) nor Markdown, and the delivery checklist asks for a human-readable trajectory per model.
The prompts themselves are still available verbatim in `conversation.md` and `llm_api_debug.jsonl`.

Like the txt exporter, a run is rendered against the task_config.yaml stored IN ITS OWN directory,
never against whatever the current calibration happens to be -- a record graded by one ruler must not
be re-narrated by another.

Run:
    python scripts/trajectory_to_md.py runs/<run_dir>
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
from formulation import CQA_KEYS, SLOTS, START  # noqa: E402
from utils import load_yaml  # noqa: E402

TASK_CFG = ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml"
CQA_LABEL = {"microenvironment_pH": "microenvironment pH",
             "emc_60rh_pct": "EMC 60 % RH / %",
             "tensile_strength_MPa": "tensile strength / MPa",
             "ffc": "ffc",
             "q30_pct": "Q30 / %"}


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


def fmt(v, nd=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) >= 1e-3 else f"{v:.3e}"
    return str(v)


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
    out_path = Path(args.output) if args.output else run_dir / "trajectory.readable.md"

    records = [r for r in load_jsonl(traj_path) if "round" in r]
    # One record per LLM CALL; a round may span several (tool-request turns, then the submission).
    # The LAST call of a round is the one that produced the submission.
    _dbg: dict = {}
    for _rec in load_jsonl(run_dir / "llm_api_debug.jsonl"):
        _dbg.setdefault(_rec.get("round"), []).append(_rec.get("llm_call") or {})
    calls_by_round = _dbg
    debug_by_round = {k: v[-1] for k, v in _dbg.items()}

    run_cfg_path = run_dir / "task_config.yaml"
    cfg_path = run_cfg_path if run_cfg_path.exists() else TASK_CFG
    cfg = load_yaml(cfg_path)
    objective_cfg, constraints = cfg["objective"], cfg["constraints"]
    windows = objective_cfg["windows"]
    start_D = float(cfg["reference"]["start_objective"])

    replay, replay_note = True, ""
    try:
        start = evaluate_reference_start(objective_cfg)
    except ValueError as exc:
        replay, replay_note = False, str(exc).split(". ")[0]
        start = None

    def call_for(record):
        return pick(record, "llm_call", default=debug_by_round.get(record.get("round"), {})) or {}

    def assess(record):
        """(points-eligible?, violations, recomputed evaluation or None). Mirrors the scorer."""
        cand = pick(record, "submitted_candidate", default={}) or {}
        form = cand.get("formulation")
        if form is None:
            return False, ["no formulation submitted"], None
        if not replay:
            ev_rec = pick(record, "evaluation", default={}) or {}
            if not ev_rec.get("valid"):
                return False, [ev_rec.get("error") or "not manufacturable"], None
            violated = []
            if float(ev_rec.get("composition_overlap")) < float(
                    constraints["composition_overlap"]["min"]) - 1e-9:
                violated.append("composition_overlap")
            if int(ev_rec.get("n_slots_changed")) > int(constraints["max_slot_changes"]["max"]):
                violated.append("max_slot_changes")
            if int(ev_rec.get("n_identity_changes")) > int(
                    constraints["max_identity_changes"]["max"]):
                violated.append("max_identity_changes")
            if not ev_rec.get("all_cqa_in_spec"):
                violated.append("CQA out of specification (release gate)")
            if float(ev_rec.get("desirability") or 0.0) <= start_D:
                violated.append("D not improved over the starting formulation")
            return not violated, violated, None
        ev = evaluate_design(form, objective_cfg)
        flags = check_constraints(ev, constraints)
        if not flags.get("on_grid_design", True):
            # check_constraints() reports every flag False for a design that is not a member of the
            # space, because none of the downstream quantities exist. Listing all eight would suggest
            # eight independent mistakes; the mistake is one -- a level that was never declared.
            offgrid = [s for s in SLOTS if s in form
                       and str(form[s]) not in {str(x) for x in cfg["design_space"]["slots"][s]["levels"]}]
            return False, [f"off-grid level(s): {', '.join(offgrid) or 'unknown slot'}"], ev
        violated = [k for k, v in flags.items() if not v]
        # Ruler 2.0.0 release gate: any CQA outside its window disqualifies the submission entirely.
        if not ev.all_cqa_in_spec:
            violated = violated + ["CQA out of spec: " + ", ".join(ev.cqa_out_of_spec)]
        if ev.desirability is None or ev.desirability <= start_D:
            violated = violated + ["D not improved over the starting formulation"]
        return not violated, violated, ev

    assessed = [assess(r) for r in records]
    report = {}
    fr = run_dir / "final_report.json"
    if fr.exists():
        report = json.loads(fr.read_text(encoding="utf-8"))

    L = [f"# Trajectory — {cfg['task_name']}", ""]
    L += [f"- **Run**: `{run_dir.name}`",
          f"- **Model**: `{(call_for(records[0]) if records else {}).get('model', '?')}`",
          f"- **Graded against**: `{cfg_path.relative_to(ROOT)}`"
          + ("" if replay else f" — *that ruler is no longer implemented ({replay_note}); every "
                               f"number below is what the run stored, nothing was recomputed*"),
          f"- **Baseline formulation D** = {start_D:.7g}",
          f"- **Stored total score**: {((report.get('score') or {}).get('total_score', '—'))} / "
          f"{(cfg.get('scoring') or {}).get('total_points', 100)}",
          f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}", ""]

    n_tool = sum(len(ti.get("results") or []) for r in records
                 for ti in (r.get("tool_interactions") or []))
    n_tool_err = sum(1 for r in records for ti in (r.get("tool_interactions") or [])
                     for x in (ti.get("results") or []) if not x.get("ok"))
    L += ["## Summary", "",
          "| Rounds | LLM calls | LLM failures | Truncated | Parse failures | Manufacturable | "
          "CQA-compliant | Qualifying | Tool calls | Tool errors |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
          "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
              len(records),
              sum(len(v) for v in calls_by_round.values()),
              sum(1 for r in records if not call_for(r).get("ok", False)),
              sum(1 for r in records if call_for(r).get("truncated")),
              sum(1 for r in records if r.get("parse_error")),
              sum(1 for r in records if (pick(r, "evaluation", default={}) or {}).get("valid")),
              sum(1 for r in records
                  if (pick(r, "evaluation", default={}) or {}).get("all_cqa_in_spec")),
              sum(1 for ok, _, _ in assessed if ok), n_tool, n_tool_err), "",
          "*Qualifying* = every hard constraint satisfied AND all five CQAs strictly inside their "
          "windows (the release gate) AND D above the starting formulation. Only qualifying "
          "submissions score.", ""]

    L += ["| Round | changed conditions | overlap | n changed | D | verdict |",
          "|---:|---|---:|---:|---:|---|"]
    for r, (ok, violated, ev) in zip(records, assessed):
        e = pick(r, "evaluation", default={}) or {}
        cand = pick(r, "submitted_candidate", default={}) or {}
        form = cand.get("formulation") or {}
        changed = ", ".join(f"{s} {START[s]}→{form[s]}" for s in SLOTS
                            if s in form and form[s] != START[s]) or "—"
        verdict = "USABLE" if ok else "not usable — " + ", ".join(violated)
        L.append("| {} | {} | {} | {} | {} | {} |".format(
            r.get("round"), changed, fmt(e.get("composition_overlap")),
            e.get("n_slots_changed") if e.get("n_slots_changed") is not None else "—",
            fmt(e.get("desirability"), 7), verdict))
    L.append("")

    for r, (ok, violated, ev) in zip(records, assessed):
        call = call_for(r)
        e = pick(r, "evaluation", default={}) or {}
        cand = pick(r, "submitted_candidate", "parsed_candidate")
        raw = pick(r, "raw_llm_response", "llm_raw_response")
        feedback = pick(r, "feedback_returned", "feedback")

        L += [f"## Round {r.get('round')}", "",
              f"`ok={call.get('ok')}` · `model={call.get('model')}` · "
              f"`attempts={call.get('attempts')}` · `finish_reason={call.get('finish_reason')}` · "
              f"`truncated={call.get('truncated')}`", ""]
        if call.get("error"):
            L += [f"> LLM error: {call['error']}", ""]
        if r.get("parse_error"):
            L += [f"> Parse error: {r['parse_error']}", ""]
        interactions = r.get("tool_interactions") or []
        if interactions:
            L += [f"### Tool interaction ({len(interactions)} request turn(s), the model's own "
                  f"choice)", "",
                  "| turn | tool | arguments | result | budget left |",
                  "|---:|---|---|---|---:|"]
            for ti in interactions:
                reqs = ti.get("requested") or []
                ress = ti.get("results") or []
                for k in range(max(len(reqs), len(ress))):
                    rq = reqs[k] if k < len(reqs) else {}
                    rs = ress[k] if k < len(ress) else {}
                    args = json.dumps(rq.get("arguments", rq.get("args", {})), ensure_ascii=False)
                    if len(args) > 120:
                        args = args[:117] + "..."
                    if rs.get("ok"):
                        outcome = "ok"
                    else:
                        outcome = (f"**ERROR `{rs.get('error')}`** (recoverable="
                                   f"{rs.get('recoverable')}): {rs.get('detail')}")
                    L.append("| {} | `{}` | `{}` | {} | {} |".format(
                        ti.get("step"), rq.get("tool", rs.get("tool", "?")), args, outcome,
                        ti.get("budget_left")))
            L.append("")
        else:
            L += ["*No tool call this round — the model went straight to a submission.*", ""]
        if cand:
            L += ["**Submitted**", "", "```json",
                  json.dumps(cand, ensure_ascii=False, indent=2), "```", ""]
        else:
            L += ["*No candidate parsed.*", ""]

        cqa = e.get("cqa") or {}
        if cqa:
            L += ["**Measured CQAs against the printed windows**", "",
                  "| CQA | L | measured | U | in window |", "|---|---:|---:|---:|:--:|"]
            for k in CQA_KEYS:
                w = windows[k]
                v = float(cqa[k])
                inside = w["lower"] < v < w["upper"]
                L.append(f"| {CQA_LABEL.get(k, k)} | {w['lower']} | {fmt(v)} | {w['upper']} | "
                         f"{'yes' if inside else '**NO**'} |")
            L.append("")
        L += ["**Budget bookkeeping** — "
              f"overlap {fmt(e.get('composition_overlap'))} (floor "
              f"{constraints['composition_overlap']['min']}) · "
              f"conditions changed {e.get('n_slots_changed')} (cap "
              f"{constraints['max_slot_changes']['max']}) · "
              f"identity changes {e.get('n_identity_changes')} (cap "
              f"{constraints['max_identity_changes']['max']}) · "
              f"q.s. filler {fmt(e.get('qs_filler_pct'), 2)} %", "",
              f"**D = {fmt(e.get('desirability'), 7)}**  ·  "
              f"**Verdict: {'USABLE' if ok else 'NOT USABLE — ' + ', '.join(violated)}**", ""]
        if feedback:
            L += ["<details><summary>Feedback returned to the model</summary>", "", "```json",
                  json.dumps(feedback, ensure_ascii=False, indent=2), "```", "</details>", ""]
        if raw:
            L += ["<details><summary>Raw model response</summary>", "", "````text", str(raw),
                  "````", "</details>", ""]

    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Saved readable Markdown report to: {out_path}")


if __name__ == "__main__":
    main()

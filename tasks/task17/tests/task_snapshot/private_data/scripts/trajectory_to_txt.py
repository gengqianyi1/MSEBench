"""Render a run's trajectory as a plain-text report a reviewer can read without tooling.

Two things this has to get right, both inherited as bugs from an earlier problem in this series:

* The per-round LLM call metadata lives in `llm_api_debug.jsonl`, not in `trajectory.jsonl`. Reading
  only the trajectory made every round look like a failed call ("LLM call failures: 5") when nothing
  had failed. The two files are joined on the round number here.
* The "usable" count must be computed the way the scorer computes it -- every hard constraint
  satisfied AND the objective improved over the registered process -- not read out of the
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

TASK_CFG = ROOT / "tasks" / "q1_agglomeration" / "task_config.yaml"


def _is_closed(tp: dict) -> bool:
    """A refused tool turn: the quota was spent, so nothing ran.

    Runs recorded before the `closed` tag existed are identified by their content instead, so the
    counts below are the same for old and new trajectories.
    """
    if tp.get("closed"):
        return True
    return any(x.get("error") == "tool_phase_closed" for x in (tp.get("results") or []))


def _tool_counts(record: dict):
    """(calls, turns, errors, refusals) over REAL tool turns -- one definition, used everywhere."""
    phase = record.get("tool_phase") or []
    real = [tp for tp in phase if not _is_closed(tp)]
    calls = sum((tp.get("n_executed") or 0) + (tp.get("n_error") or 0) for tp in real)
    errors = sum((tp.get("n_error") or 0) for tp in phase)
    return calls, len(real), errors, len(phase) - len(real)


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
    debug_by_round = {r.get("round"): (r.get("llm_call") or {})
                      for r in load_jsonl(run_dir / "llm_api_debug.jsonl")}

    cfg = load_yaml(TASK_CFG)
    objective_cfg = cfg["objective"]
    constraints = cfg["constraints"]
    start = evaluate_reference_start(objective_cfg)

    def usable(record):
        cand = pick(record, "submitted_candidate", default={}) or {}
        form = cand.get("process")
        if form is None:
            return False, ["no process submitted"]
        ev = evaluate_design(form, objective_cfg)
        violated = [k for k, v in check_constraints(ev, constraints).items() if not v]
        if ev.desirability is None or ev.desirability <= start.desirability:
            violated = violated + ["D not improved over the registered process"]
        return not violated, violated

    def call_for(record):
        return pick(record, "llm_call", default=debug_by_round.get(record.get("round"), {})) or {}

    flags = [usable(r) for r in records]

    lines = [
        f"Trajectory report -- {cfg['task_name']}",
        "=" * 80,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Run: {run_dir.name}",
        f"Objective: maximize Derringer-Suich D   registered process D = {start.desirability:.6f}",
        f"Registered responses: {json.dumps(start.responses)}",
        "",
        "Summary",
        "-" * 80,
        f"Rounds: {len(records)}",
        f"LLM call failures: {sum(1 for r in records if not call_for(r).get('ok', False))}",
        f"Truncated LLM responses: {sum(1 for r in records if call_for(r).get('truncated'))}",
        f"Parse failures: {sum(1 for r in records if r.get('parse_error'))}",
        f"Runnable batches: "
        f"{sum(1 for r in records if (pick(r, 'evaluation', default={}) or {}).get('valid'))}",
        f"Usable processes (all constraints + improves D): {sum(1 for ok, _ in flags if ok)}",
        f"Tool calls: {sum(_tool_counts(r)[0] for r in records)}"
        f"   tool turns: {sum(_tool_counts(r)[1] for r in records)}"
        f"   tool errors: {sum(_tool_counts(r)[2] for r in records)}"
        f"   quota refusals: {sum(_tool_counts(r)[3] for r in records)}"
        f"   rounds that used tools: {sum(1 for r in records if r.get('tool_phase'))}",
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
                  f"truncated={call.get('truncated')}  turn_kind={r.get('turn_kind')}"]

        # The agent tool phase, compactly. This is the evidence for "the model chose tools, filled
        # arguments, read results and handled failures", so it belongs in the file a reviewer reads
        # rather than only in trajectory.jsonl.
        phase = r.get("tool_phase") or []
        if phase:
            lines.append("")
            lines.append("Tool phase (diagnostics only -- no quality attribute, no D, no score):")
            for tp in phase:
                names = [(c.get("tool") if isinstance(c, dict) else "?")
                         for c in (tp.get("requested") or [])]
                errs = [x.get("error") for x in (tp.get("results") or []) if x.get("error")]
                tag = "  [REFUSED: quota spent, nothing ran]" if _is_closed(tp) else ""
                lines.append(f"  turn {tp.get('tool_turn')}: requested {names}"
                             f"  executed={tp.get('n_executed')}  errors={errs or 'none'}{tag}")
        if call.get("error"):
            lines.append(f"LLM error: {call['error']}")
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

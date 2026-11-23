"""Render a run's trajectory as a plain-text report a reviewer can read without tooling.

Two things this has to get right, both of which the inherited version got wrong:

* The per-round LLM call metadata lives in `llm_api_debug.jsonl`, not in `trajectory.jsonl`. Reading
  only the trajectory made every round look like a failed call ("LLM call failures: 5") when nothing
  had failed. The two files are joined on the round number here.
* The old summary counted "successful optimized analogs" from a `counts_as_successful_optimization`
  field in the feedback. This benchmark's feedback is deliberately reduced-bandwidth and no longer
  returns any success verdict, so that count was always zero. It is now computed the same way the
  scorer computes it: satisfies every hard constraint AND improves the objective over the start.

Record field names are accepted in both the current and the legacy spelling, so reports for runs
produced before the trajectory record was enriched still render correctly.

Since the tool layer was added, a round contains SEVERAL LLM turns (tool messages, then a
submission). `llm_api_debug.jsonl` therefore has one entry per (round, turn), and joining it on the
round number alone would silently keep only the last turn. The per-round `turns` list in
`trajectory.jsonl` is the authority; the debug log is only a fallback for older runs.

Run:
    python scripts/trajectory_to_txt.py runs/<run_dir>/trajectory.jsonl
    python scripts/trajectory_to_txt.py runs/<run_dir>            # directory also accepted
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import evaluate_smiles, check_constraints  # noqa: E402
from utils import load_yaml  # noqa: E402

TASK_CFG = ROOT / "tasks" / "q1_lipophilic" / "task_config.yaml"


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
    """First present, non-null key among `names` -- tolerates the legacy record spelling."""
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", help="Path to trajectory.jsonl, or the run directory")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    given = Path(args.trajectory)
    run_dir = given if given.is_dir() else given.parent
    traj_path = given if given.is_file() else run_dir / "trajectory.jsonl"
    out_path = Path(args.output) if args.output else traj_path.with_suffix(".readable.txt")

    records = [r for r in load_jsonl(traj_path) if "round" in r]
    debug_by_round = {r.get("round"): (r.get("llm_call") or {})
                      for r in load_jsonl(run_dir / "llm_api_debug.jsonl")}

    cfg = load_yaml(TASK_CFG)
    constraints = cfg["constraints"]
    sim_cfg = constraints["similarity"]
    target = cfg["optimization_target"]
    field = {"QED": "qed", "SA_SCORE": "sa_score", "TPSA": "tpsa"}.get(
        str(target["name"]).upper(), str(target["name"]).lower())
    start = evaluate_smiles(cfg["start_smiles"], start_smiles=cfg["start_smiles"],
                            similarity_cfg=sim_cfg)
    start_value = getattr(start, field)
    maximize = target["direction"] == "maximize"

    def usable(evaluation):
        """Same test the scorer applies: all hard constraints, and improves the objective."""
        smiles = (evaluation or {}).get("smiles")
        if not smiles:
            return False, ["no molecule submitted"]
        e = evaluate_smiles(smiles, start_smiles=cfg["start_smiles"], similarity_cfg=sim_cfg)
        violated = [k for k, v in check_constraints(e, constraints).items() if not v]
        value = getattr(e, field)
        improved = value is not None and (value > start_value if maximize else value < start_value)
        if not improved:
            violated = violated + [f"{target['name']} not improved over the start"]
        return not violated, violated

    def turns_for(record):
        """Every LLM turn in the round, newest format first, then the legacy single-call shape."""
        turns = record.get("turns")
        if isinstance(turns, list) and turns:
            return turns
        legacy = pick(record, "llm_call", default=debug_by_round.get(record.get("round"), {})) or {}
        return [{"turn": 1, "kind": "submission", "llm_call": legacy,
                 "raw_response": pick(record, "raw_llm_response", "llm_raw_response"),
                 "tool_responses": None}]

    def call_for(record):
        """The last LLM call of the round -- the one that produced the submission."""
        return (turns_for(record)[-1].get("llm_call") or {})

    def any_call_failed(record):
        return any(not (t.get("llm_call") or {}).get("ok", False) for t in turns_for(record))

    usable_flags = [usable(pick(r, "evaluation", default={})) for r in records]

    lines = [
        f"Trajectory report -- {cfg['task_name']}",
        "=" * 80,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Run: {run_dir.name}",
        f"Objective: {target['direction']} {target['name']}   start value {start_value:.6f}",
        "",
        "Summary",
        "-" * 80,
        f"Rounds: {len(records)}",
        f"LLM turns: {sum(len(turns_for(r)) for r in records)}",
        f"Tool calls: {sum(len(r.get('tool_calls_this_round') or []) for r in records)}",
        f"Rounds with an LLM call failure: {sum(1 for r in records if any_call_failed(r))}",
        f"Truncated LLM responses: "
        f"{sum(1 for r in records for t in turns_for(r) if (t.get('llm_call') or {}).get('truncated'))}",
        f"Parse failures: {sum(1 for r in records if r.get('parse_error'))}",
        f"Valid molecule submissions: "
        f"{sum(1 for r in records if (pick(r, 'evaluation', default={}) or {}).get('valid'))}",
        f"Usable analogs (all constraints + improves {target['name']}): "
        f"{sum(1 for ok, _ in usable_flags if ok)}",
        "",
    ]

    for r, (ok, violated) in zip(records, usable_flags):
        call = call_for(r)
        evaluation = pick(r, "evaluation", default={}) or {}
        candidate = pick(r, "submitted_candidate", "parsed_candidate")
        raw = turns_for(r)[-1].get("raw_response") or pick(r, "raw_llm_response", "llm_raw_response")
        feedback = pick(r, "feedback_returned", "feedback")

        lines += [f"Round {r.get('round')}", "~" * 80,
                  f"LLM ok={call.get('ok')}  model={call.get('model')}  "
                  f"attempts={call.get('attempts')}  finish_reason={call.get('finish_reason')}  "
                  f"truncated={call.get('truncated')}",
                  f"Tool budget: {r.get('tool_calls_used')} of {r.get('tool_calls_allowed')} calls used"]
        if call.get("error"):
            lines.append(f"LLM error: {call['error']}")

        # The tool interaction is the part a reviewer of the Agent dimension actually wants to see:
        # which tool, on what, what came back, and what the model did next.
        for t in turns_for(r):
            if t.get("kind") == "tool_calls" and t.get("tool_responses"):
                lines += ["", f"  -- turn {t.get('turn')}: tool message --",
                          f"  model sent: {t.get('raw_response')}"]
                for resp in t["tool_responses"]:
                    if resp.get("ok"):
                        lines.append(f"  -> {resp.get('tool')}: OK "
                                     f"{json.dumps(resp.get('result'), ensure_ascii=False)}")
                    else:
                        err = resp.get("error") or {}
                        lines.append(f"  -> {resp.get('tool')}: ERROR {err.get('code')} "
                                     f"-- {err.get('message')}")
            elif t.get("kind") == "tool_call_after_budget":
                lines += ["", f"  -- turn {t.get('turn')}: round forfeited "
                              f"(tool message sent after the budget was gone) --"]

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

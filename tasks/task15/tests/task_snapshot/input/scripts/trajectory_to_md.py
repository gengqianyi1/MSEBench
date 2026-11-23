"""Render a run as a Markdown conversation a reviewer can read in a browser.

Rewritten when the tool layer landed. The previous version pulled a `messages_input_to_llm` field out
of each trajectory record and dumped it as JSON under a `.md` filename -- that field no longer exists
(a round is now a list of `turns`), so it produced an empty file, and it never produced Markdown in
the first place.

This version renders the actual exchange: per round, every model turn, every tool call with its
arguments, every tool response or error, the submitted candidate, the computed evaluation, and the
feedback that went back. `conversation.md` in the run directory holds the same exchange as raw
message JSON; this is the human-readable view.

Run:
    python scripts/trajectory_to_md.py runs/<run_dir>
    python scripts/trajectory_to_md.py runs/<run_dir>/trajectory.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fence(text: str, lang: str = "json") -> list[str]:
    return [f"```{lang}", text.rstrip(), "```", ""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", help="the run directory, or its trajectory.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    given = Path(args.trajectory)
    run_dir = given if given.is_dir() else given.parent
    traj_path = given if given.is_file() else run_dir / "trajectory.jsonl"
    out_path = Path(args.output) if args.output else run_dir / "trajectory.conversation.md"

    records = []
    for line in traj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            records.append(json.loads(line))

    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    lines = [f"# Trajectory — {run_dir.name}", ""]
    if summary:
        lines += [f"- Task: {summary.get('task_name')}",
                  f"- Rounds: {summary.get('fixed_iteration_budget')}",
                  f"- Scored by: {summary.get('scored_by')}", ""]
        tu = summary.get("tool_usage") or {}
        if tu:
            lines += ["## Tool usage", "", *fence(json.dumps(tu, ensure_ascii=False, indent=2)),
                      ""]

    for rec in records:
        lines += [f"## Round {rec.get('round')}", "",
                  f"Tool budget: {rec.get('tool_calls_used')} of {rec.get('tool_calls_allowed')} "
                  f"calls used", ""]

        for turn in rec.get("turns") or []:
            kind = turn.get("kind")
            lines.append(f"### Turn {turn.get('turn')} — {kind}")
            lines.append("")
            call = turn.get("llm_call") or {}
            lines.append(f"*LLM: ok={call.get('ok')}, model={call.get('model')}, "
                         f"attempts={call.get('attempts')}, finish_reason={call.get('finish_reason')}*")
            lines.append("")
            if turn.get("raw_response"):
                lines += ["Model sent:", "", *fence(turn["raw_response"])]
            if turn.get("parse_error"):
                lines += [f"**Problem:** {turn['parse_error']}", ""]
            for resp in turn.get("tool_responses") or []:
                if resp.get("ok"):
                    lines += [f"**`{resp.get('tool')}` → ok**", "",
                              *fence(json.dumps(resp.get("result"), ensure_ascii=False, indent=2))]
                else:
                    err = resp.get("error") or {}
                    lines += [f"**`{resp.get('tool')}` → ERROR `{err.get('code')}`**: "
                              f"{err.get('message')}", "",
                              *fence(json.dumps(err, ensure_ascii=False, indent=2))]

        cand = rec.get("submitted_candidate")
        lines += ["### Submission", ""]
        lines += fence(json.dumps(cand, ensure_ascii=False, indent=2)) if cand else \
            ["*No candidate was submitted this round.*", ""]
        if rec.get("parse_error"):
            lines += [f"**Round problem:** {rec['parse_error']}", ""]
        lines += ["### Evaluation returned", "",
                  *fence(json.dumps(rec.get("evaluation"), ensure_ascii=False, indent=2)),
                  "### Feedback returned to the model", "",
                  *fence(json.dumps(rec.get("feedback_returned"), ensure_ascii=False, indent=2))]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved Markdown conversation to: {out_path}")


if __name__ == "__main__":
    main()

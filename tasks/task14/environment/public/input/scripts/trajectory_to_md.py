"""Render a run as a Markdown conversation a reviewer can read in a browser.

The previous version pulled a `messages_input_to_llm` field out of each trajectory record and dumped
it as JSON under a `.md` filename -- that is not Markdown, and it omitted the part a reviewer
actually wants (what the model submitted and what came back). This version renders the exchange:
per round, the model's raw message, the parsed candidate, the computed evaluation and the feedback
that was returned.

No reference value and no score appear here -- this script lives on the public side.

Run:
    python scripts/trajectory_to_md.py ../runs/<run_dir>
    python scripts/trajectory_to_md.py ../runs/<run_dir>/trajectory.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fence(text: str, lang: str = "json") -> list[str]:
    return [f"```{lang}", (text or "").rstrip(), "```", ""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", help="the run directory, or its trajectory.jsonl")
    parser.add_argument("--output", default=None, help="output markdown path")
    args = parser.parse_args()

    given = Path(args.trajectory)
    in_path = given / "trajectory.jsonl" if given.is_dir() else given
    run_dir = in_path.parent
    out_path = Path(args.output) if args.output else run_dir / "trajectory.conversation.md"

    debug = {}
    dbg_path = run_dir / "llm_api_debug.jsonl"
    if dbg_path.exists():
        for line in dbg_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if "round" in rec:
                    debug[rec["round"]] = rec.get("llm_call", {})

    lines = [f"# Trajectory — {run_dir.name}", ""]
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rnd = rec.get("round")
        llm = debug.get(rnd, {})
        lines += [f"## Round {rnd}", "",
                  f"- LLM ok: `{llm.get('ok')}`  model: `{llm.get('model')}`  "
                  f"finish_reason: `{llm.get('finish_reason')}`  "
                  f"attempts: `{llm.get('attempts')}`  truncated: `{llm.get('truncated')}`", "",
                  "### Model message", ""]
        lines += fence(rec.get("raw_response") or llm.get("content") or "[EMPTY]", "text")
        if rec.get("parse_error"):
            lines += ["### Parse error", "", f"`{rec['parse_error']}`", ""]
        lines += ["### Submitted candidate", ""]
        lines += fence(json.dumps(rec.get("submitted_candidate"), ensure_ascii=False, indent=2))
        lines += ["### RDKit evaluation", ""]
        lines += fence(json.dumps(rec.get("evaluation"), ensure_ascii=False, indent=2))
        lines += ["### Feedback returned to the model", ""]
        lines += fence(json.dumps(rec.get("feedback_returned"), ensure_ascii=False, indent=2))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved conversation markdown to: {out_path}")


if __name__ == "__main__":
    main()

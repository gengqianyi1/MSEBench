"""Render a run's trajectory as a plain-text report a reviewer can read without tooling.

Two things this has to get right, both of which the inherited version got wrong:

* The per-round LLM call metadata lives in `llm_api_debug.jsonl`, not in `trajectory.jsonl`. Reading
  only the trajectory made every round look like a failed call ("LLM call failures: 5") when nothing
  had failed. The two files are joined on the round number here.
* The old summary counted "successful optimized analogs" from a `counts_as_successful_optimization`
  field in the feedback. This benchmark's feedback is deliberately reduced-bandwidth and returns no
  success verdict, so that count was always zero. It is now computed the same way the grader
  computes it: satisfies every hard constraint AND improves the objective over the start.

No reference value and no score appear here -- this script lives on the public side.

Run:
    python scripts/trajectory_to_txt.py ../runs/<run_dir>/trajectory.jsonl
    python scripts/trajectory_to_txt.py ../runs/<run_dir>            # directory also accepted
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


def load_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            records.append({"_line": i, "_parse_error": str(exc), "_raw": line[:500]})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", help="the run directory, or its trajectory.jsonl")
    parser.add_argument("--output", default=None, help="output txt path")
    args = parser.parse_args()

    given = Path(args.trajectory)
    in_path = given / "trajectory.jsonl" if given.is_dir() else given
    run_dir = in_path.parent
    out_path = Path(args.output) if args.output else run_dir / "trajectory.readable.txt"

    task_cfg = load_yaml(run_dir / "task_config.yaml") if (run_dir / "task_config.yaml").exists() \
        else load_yaml(ROOT / "tasks" / "q1_multiobjective" / "task_config.yaml")
    start_smiles = task_cfg["start_smiles"]
    constraints = task_cfg["constraints"]
    sim_cfg = constraints["similarity"]
    start_eval = evaluate_smiles(start_smiles, start_smiles=start_smiles, similarity_cfg=sim_cfg)

    records = [r for r in load_jsonl(in_path) if "round" in r]
    debug = {}
    for d in load_jsonl(run_dir / "llm_api_debug.jsonl"):
        if "round" in d:
            debug[d["round"]] = d.get("llm_call", {})

    def usable(rec) -> bool:
        ev = rec.get("evaluation") or {}
        if not ev.get("valid"):
            return False
        smi = ev.get("canonical_smiles") or ev.get("smiles")
        e = evaluate_smiles(smi, start_smiles=start_smiles, similarity_cfg=sim_cfg)
        if not e.valid or e.qed is None or start_eval.qed is None:
            return False
        return all(check_constraints(e, constraints).values()) and e.qed > start_eval.qed

    lines = ["Molecular Design Benchmark Trajectory (readable)", "=" * 80,
             f"Generated: {datetime.now().isoformat(timespec='seconds')}",
             f"Source: {in_path}", "",
             "Summary", "-" * 80,
             f"Rounds: {len(records)}",
             f"LLM call failures: {sum(1 for r in records if debug.get(r['round'], {}).get('ok') is False)}",
             f"Empty LLM responses: {sum(1 for r in records if debug.get(r['round'], {}).get('empty_content'))}",
             f"Parse failures: {sum(1 for r in records if r.get('parse_error'))}",
             f"Valid molecule submissions: {sum(1 for r in records if (r.get('evaluation') or {}).get('valid'))}",
             f"Usable analogs (feasible AND improving): {sum(1 for r in records if usable(r))}",
             ""]

    for r in records:
        llm = debug.get(r["round"], {})
        lines += [f"Round {r.get('round')}", "~" * 80,
                  f"LLM ok: {llm.get('ok')}",
                  f"LLM model: {llm.get('model')}",
                  f"LLM attempts: {llm.get('attempts')}",
                  f"LLM finish_reason: {llm.get('finish_reason')}",
                  f"LLM empty_content: {llm.get('empty_content')}",
                  f"LLM truncated: {llm.get('truncated')}",
                  f"LLM error: {llm.get('error')}",
                  "LLM raw content:",
                  r.get("raw_response") or llm.get("content") or "[EMPTY]", "",
                  f"Submitted candidate: {json.dumps(r.get('submitted_candidate'), ensure_ascii=False)}",
                  f"Parse error: {r.get('parse_error')}",
                  f"Evaluation: {json.dumps(r.get('evaluation'), ensure_ascii=False)}",
                  f"Feedback returned: {json.dumps(r.get('feedback_returned'), ensure_ascii=False)}",
                  ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved readable report to: {out_path}")


if __name__ == "__main__":
    main()

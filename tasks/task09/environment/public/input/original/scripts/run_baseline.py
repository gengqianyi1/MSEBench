"""Run, replay, independently score, and freeze the v2 live baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.baseline_agent import run_baseline
from evaluator.evaluate_v2 import evaluate_run
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / "trajectories" / "baseline",
    )
    args = parser.parse_args()
    destination = args.destination
    run_baseline(destination)
    first = evaluate_run(destination, write=False)
    second = evaluate_run(destination, write=False)
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise RuntimeError("independent evaluator replay is not byte deterministic")
    written = evaluate_run(destination, write=True)
    if canonical_json_bytes(first) != canonical_json_bytes(written):
        raise RuntimeError("written evaluation differs from replay")
    receipt = {
        "schema_version": "2.0",
        "benchmark_version": "2.0.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "status": written["run_status"],
        "mode": "LIVE_XTB_PUBLIC_ACTIONS_ONLY",
        "trajectory_sha256": sha256_file(destination / "trajectory.jsonl"),
        "final_answer_sha256": sha256_file(destination / "final_answer.json"),
        "evaluation_sha256": sha256_file(destination / "evaluation.json"),
        "usage_report_sha256": sha256_file(destination / "usage_report.json"),
        "eligible": written["eligible"],
        "total_score": written["total_score"],
        "metrics": written["metrics"],
        "reconstructed": written["reconstructed"],
        "live_replay_byte_identical": True,
    }
    report = ROOT / "reports" / "baseline_live_validation.json"
    with report.open("xb") as handle:
        handle.write(canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" and receipt["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

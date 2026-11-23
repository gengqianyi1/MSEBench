"""Run deterministic local-edit baselines over every organizer QED case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.baseline_agent import (  # noqa: E402
    run_baseline,
    write_final_answer,
    write_trajectory,
)
from evaluator.evaluate import evaluate_submission_file  # noqa: E402
from evaluator.evaluate_suite import aggregate_suite, load_suite  # noqa: E402


def run_suite(suite_path: Path, output_root: Path, mode: str) -> dict:
    _, case_paths = load_suite(suite_path)
    if mode not in {"deterministic_beam", "shallow_local"}:
        raise ValueError("unknown baseline mode")
    # Calibration baseline: one local-edit round under the same public case
    # information and global task budget as evaluated agents.  This is a weak
    # bounded comparator, not an organizer search for a private optimum.
    beam_width, max_depth = ((3, 1) if mode == "deterministic_beam" else (1, 1))
    for case_path in case_paths:
        config = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        case_id = config["case"]["id"]
        case_dir = output_root / case_id
        events: list[dict] = []
        answer = run_baseline(
            case_path,
            beam_width=beam_width,
            max_depth=max_depth,
            process_events=events,
            include_aromatic_halogen=(mode == "deterministic_beam"),
        )
        answer_path = write_final_answer(answer, case_dir)
        trajectory_path = write_trajectory(events, case_dir)
        evaluation_path = case_dir / "evaluation.json"
        evaluate_submission_file(
            answer_path,
            case_path,
            evaluation_path,
            trajectory_path=trajectory_path,
        )
        (case_dir / "status.json").write_text(
            json.dumps(
                {
                    "provider_status": "OFFLINE_BASELINE",
                    "run_status": "PASS",
                    "baseline_mode": mode,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return aggregate_suite(suite_path, output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "organizer_private" / "suite.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("deterministic_beam", "shallow_local"),
        default="deterministic_beam",
    )
    args = parser.parse_args(argv)
    result = run_suite(args.suite, args.output, args.mode)
    destination = args.output / "suite_evaluation.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

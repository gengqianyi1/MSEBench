#!/usr/bin/env python3
"""Run the frozen three-model QED v4 campaign over all five cases serially."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.evaluate import evaluate_submission  # noqa: E402
from evaluator.evaluate_suite import aggregate_suite, load_suite  # noqa: E402
from framework import AgentRunner  # noqa: E402
from scripts.run_model import (  # noqa: E402
    _load_runtime_environment,
    _model_config,
)


LOGICAL_MODELS = ("deepseek", "kimi", "glm")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _run_status(result: Any) -> tuple[str, str]:
    counts = result.metadata.get("counts", {})
    provider_responses = counts.get("provider_responses", 0)
    if result.complete:
        eligible = result.evaluation.get("eligible") is True
        return "AVAILABLE", "PASS" if eligible else "FAIL"
    if isinstance(provider_responses, int) and provider_responses > 0:
        return "RESPONDED", "FAIL"
    return "UNAVAILABLE_AFTER_RETRIES", "BLOCKED"


def run_campaign(
    *,
    suite_path: Path,
    models_path: Path,
    prompt_path: Path,
    output_root: Path,
    models: tuple[str, ...],
) -> dict[str, Any]:
    if output_root.exists():
        raise ValueError(
            "campaign output root already exists; overwrite is forbidden: "
            + str(output_root)
        )
    _, case_paths = load_suite(suite_path)
    campaign_rows: list[dict[str, Any]] = []

    for logical_model in models:
        selected_model = _model_config(models_path, logical_model)
        model_root = output_root / logical_model
        for case_path in case_paths:
            case_config = yaml.safe_load(case_path.read_text(encoding="utf-8"))
            case_id = case_config["case"]["id"]
            case_dir = model_root / case_id
            runner = AgentRunner(
                task_config=case_path,
                model_config=selected_model,
                output_dir=case_dir,
                prompt=prompt_path,
                run_label=f"{logical_model}:{case_id}",
                model_config_source=models_path,
            )
            result = runner.run()
            provider_status, run_status = _run_status(result)
            requested = result.metadata.get("requested_model_id") or result.metadata.get(
                "requested_model"
            )
            actual = result.metadata.get("actual_model_id") or result.metadata.get("model")
            status = {
                "schema_version": "qed_case_status/4.0",
                "provider_status": provider_status,
                "run_status": run_status,
                "logical_model": logical_model,
                "case_id": case_id,
                "requested_model_id": requested,
                "actual_model_id": actual,
                "exact_model_identity": requested == actual,
                "runner_terminal_status": result.status,
                "failure": result.metadata.get("failure"),
            }
            _write_json(case_dir / "status.json", status)

            stored = json.loads((case_dir / "evaluation.json").read_text(encoding="utf-8"))
            replay = evaluate_submission(
                case_dir / "final_answer.json",
                case_path,
                trajectory_path=case_dir / "trajectory.jsonl",
            )
            stored_bytes = _canonical_bytes(stored)
            replay_bytes = _canonical_bytes(replay)
            replay_record = {
                "schema_version": "qed_replay_verification/4.0",
                "case_id": case_id,
                "logical_model": logical_model,
                "canonical_match": stored_bytes == replay_bytes,
                "stored_evaluation_sha256": hashlib.sha256(stored_bytes).hexdigest(),
                "replay_evaluation_sha256": hashlib.sha256(replay_bytes).hexdigest(),
            }
            _write_json(case_dir / "replay_verification.json", replay_record)
            if not replay_record["canonical_match"]:
                raise RuntimeError(
                    f"offline replay differs from stored evaluation: {logical_model}/{case_id}"
                )

            campaign_rows.append(
                {
                    "logical_model": logical_model,
                    "case_id": case_id,
                    "provider_status": provider_status,
                    "run_status": run_status,
                    "requested_model_id": requested,
                    "actual_model_id": actual,
                    "exact_model_identity": requested == actual,
                    "official_case_score": stored.get("total_score", stored.get("score")),
                    "eligible": stored.get("eligible"),
                    "tool_calls": result.metadata.get("counts", {}).get("tool_calls"),
                    "unique_evaluated_candidates": result.metadata.get("counts", {}).get(
                        "unique_evaluated_candidates"
                    ),
                    "candidate_feedback_turns": result.metadata.get("counts", {}).get(
                        "candidate_feedback_turns"
                    ),
                    "final_smiles": stored.get("final_candidate", {}).get(
                        "canonical_smiles"
                    ),
                    "duration_seconds": result.metadata.get("duration_seconds"),
                }
            )

        suite_result = aggregate_suite(suite_path, model_root)
        _write_json(model_root / "suite_evaluation.json", suite_result)

    campaign = {
        "schema_version": "qed_model_campaign/4.0",
        "models": list(models),
        "case_count_per_model": len(case_paths),
        "run_count": len(campaign_rows),
        "all_replays_match": True,
        "rows": campaign_rows,
    }
    _write_json(output_root / "campaign_summary.json", campaign)
    return campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=PROJECT_ROOT / "organizer_private" / "suite.yaml"
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models.example.yaml",
    )
    parser.add_argument(
        "--prompt", type=Path, default=PROJECT_ROOT / "prompts" / "agent_prompt.md"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        choices=LOGICAL_MODELS,
        help="Run only this logical model; repeat to select more than one.",
    )
    args = parser.parse_args(argv)
    _load_runtime_environment()
    selected = tuple(args.model) if args.model else LOGICAL_MODELS
    campaign = run_campaign(
        suite_path=args.suite.resolve(),
        models_path=args.models_config.resolve(),
        prompt_path=args.prompt.resolve(),
        output_root=args.output.resolve(),
        models=selected,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "models": campaign["models"],
                "run_count": campaign["run_count"],
                "all_replays_match": campaign["all_replays_match"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

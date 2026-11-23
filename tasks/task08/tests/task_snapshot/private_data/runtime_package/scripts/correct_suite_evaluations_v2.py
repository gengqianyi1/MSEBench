#!/usr/bin/env python3
"""Promote frozen formal runs with case-correct evaluator configurations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate_suite import evaluate_suite  # noqa: E402
from scripts.run_suite_models_v2 import (  # noqa: E402
    FORMAL_MODELS,
    _case_status,
    _suite_cases,
)
from scripts.runtime_support import (  # noqa: E402
    RuntimeIntegrityError,
    deterministic_evaluation,
    load_json,
    scan_tree,
    sha256_file,
    write_json_exclusive,
)


class _FrozenResult:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.status = str(metadata.get("status", "FAILED"))

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETE"


def _copy_formal_case(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeIntegrityError("CORRECTED_CASE_DESTINATION_EXISTS")
    destination.mkdir(parents=True)
    for name in (
        "workspace",
        "agent_output",
        "trajectory.jsonl",
        "dialogue.md",
        "metadata.json",
        "final_answer.json",
    ):
        origin = source / name
        target = destination / name
        if origin.is_dir():
            shutil.copytree(origin, target, copy_function=shutil.copy2)
        elif origin.is_file():
            shutil.copy2(origin, target)
        else:
            raise RuntimeIntegrityError("SOURCE_FORMAL_CASE_INCOMPLETE")
    superseded = destination / "superseded_public_config_evaluation"
    superseded.mkdir()
    for name in ("evaluation.json", "replay_verification.json", "status.json"):
        origin = source / name
        if not origin.is_file():
            raise RuntimeIntegrityError("SOURCE_EVALUATION_RECEIPT_INCOMPLETE")
        shutil.copy2(origin, superseded / name)


def correct_campaign(
    *, source_root: Path, output_root: Path, suite_path: Path
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeIntegrityError("CORRECTED_CAMPAIGN_OUTPUT_EXISTS")
    source_campaign = load_json(source_root / "campaign_summary.json")
    if not isinstance(source_campaign, dict) or source_campaign.get("run_count") != 18:
        raise RuntimeIntegrityError("SOURCE_CAMPAIGN_NOT_COMPLETE")
    cases = _suite_cases(suite_path)
    rows: list[dict[str, Any]] = []

    for logical_model in FORMAL_MODELS:
        model_root = output_root / logical_model
        for case in cases:
            source_case = source_root / logical_model / case["case_id"]
            case_root = model_root / case["case_id"]
            _copy_formal_case(source_case, case_root)
            metadata = load_json(case_root / "metadata.json")
            if not isinstance(metadata, dict):
                raise RuntimeIntegrityError("FORMAL_METADATA_NOT_OBJECT")

            config_path = ROOT / case["config"]
            evaluation, frozen_bytes = deterministic_evaluation(
                final_answer=case_root / "final_answer.json",
                trajectory=case_root / "trajectory.jsonl",
                artifacts=case_root / "workspace",
                output=case_root / "evaluation.json",
                config_path=config_path,
            )
            replay, replay_bytes = deterministic_evaluation(
                final_answer=case_root / "final_answer.json",
                trajectory=case_root / "trajectory.jsonl",
                artifacts=case_root / "workspace",
                config_path=config_path,
            )
            replay_match = evaluation == replay and frozen_bytes == replay_bytes
            write_json_exclusive(
                case_root / "replay_verification.json",
                {
                    "schema_version": "xtb_replay_verification/2.0",
                    "logical_model": logical_model,
                    "case_id": case["case_id"],
                    "canonical_match": replay_match,
                    "case_config": case["config"],
                    "stored_evaluation_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
                    "replay_evaluation_sha256": hashlib.sha256(replay_bytes).hexdigest(),
                },
            )
            scan = scan_tree(case_root)
            status = _case_status(
                logical_model=logical_model,
                requested_model=str(metadata.get("requested_model")),
                result=_FrozenResult(metadata),
                evaluation=evaluation,
                replay_match=replay_match,
                scan=scan,
            )
            status["evaluation_config"] = case["config"]
            status["model_and_xtb_rerun"] = False
            write_json_exclusive(case_root / "status.json", status)
            write_json_exclusive(
                case_root / "evaluation_correction_receipt.json",
                {
                    "schema_version": "xtb_evaluation_correction/2.0",
                    "reason": "PER_CASE_CONFIG_REQUIRED",
                    "model_and_xtb_rerun": False,
                    "source_campaign": source_root.relative_to(ROOT).as_posix(),
                    "source_case": source_case.relative_to(ROOT).as_posix(),
                    "source_trajectory_sha256": sha256_file(source_case / "trajectory.jsonl"),
                    "promoted_trajectory_sha256": sha256_file(case_root / "trajectory.jsonl"),
                    "case_config": case["config"],
                },
            )
            rows.append(status)

        suite_evaluation = evaluate_suite(suite_path, model_root)
        write_json_exclusive(model_root / "suite_evaluation.json", suite_evaluation)

    campaign = {
        "schema_version": "xtb_model_campaign/2.0",
        "campaign_kind": "CASE_CONFIG_EVALUATION_CORRECTION",
        "source_campaign": source_root.relative_to(ROOT).as_posix(),
        "model_and_xtb_rerun": False,
        "models": list(FORMAL_MODELS),
        "case_count_per_model": len(cases),
        "run_count": len(rows),
        "all_replays_match": all(row["replay_match"] for row in rows),
        "all_exact_model_identities": all(row["exact_model_identity"] for row in rows),
        "rows": rows,
    }
    write_json_exclusive(output_root / "campaign_summary.json", campaign)
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "organizer_private" / "suite.json"
    )
    args = parser.parse_args()
    campaign = correct_campaign(
        source_root=args.source.resolve(),
        output_root=args.output.resolve(),
        suite_path=args.suite.resolve(),
    )
    print(
        {
            "run_count": campaign["run_count"],
            "all_replays_match": campaign["all_replays_match"],
            "all_exact_model_identities": campaign["all_exact_model_identities"],
        }
    )


if __name__ == "__main__":
    main()

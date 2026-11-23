#!/usr/bin/env python3
"""Run all six xTB v2 cases for the exact formal models, serially."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate_suite import evaluate_suite  # noqa: E402
from framework import AgentRunner, ModelConfig, TaskConfig, ToolContractSet  # noqa: E402
from scripts.run_model import _formal_prompt, _pre_model_gate  # noqa: E402
from scripts.runtime_support import (  # noqa: E402
    RuntimeIntegrityError,
    deterministic_evaluation,
    load_json,
    regular_files,
    scan_tree,
    sha256_file,
    write_json_exclusive,
)
from tools import ToolRegistry  # noqa: E402


FORMAL_MODELS = ("deepseek", "kimi", "glm")
RUNNER_ARTIFACTS = (
    "trajectory.jsonl",
    "dialogue.md",
    "metadata.json",
    "final_answer.json",
)


def _suite_cases(path: Path) -> list[dict[str, str]]:
    suite = load_json(path)
    if not isinstance(suite, Mapping):
        raise RuntimeIntegrityError("SUITE_NOT_OBJECT")
    public = suite.get("public_case")
    hidden = suite.get("hidden_cases")
    values = ([public] if isinstance(public, Mapping) else []) + (
        list(hidden) if isinstance(hidden, list) else []
    )
    cases: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise RuntimeIntegrityError("SUITE_CASE_NOT_OBJECT")
        record: dict[str, str] = {}
        for key in ("case_id", "config", "workspace"):
            item = value.get(key)
            if not isinstance(item, str) or not item or "\\" in item:
                raise RuntimeIntegrityError("SUITE_CASE_FIELD_INVALID")
            pure = Path(item)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeIntegrityError("SUITE_CASE_PATH_UNSAFE")
            record[key] = item
        cases.append(record)
    if len(cases) != 6 or len({case["case_id"] for case in cases}) != 6:
        raise RuntimeIntegrityError("FORMAL_SUITE_MUST_CONTAIN_SIX_UNIQUE_CASES")
    return cases


def _tree_receipt(root: Path) -> list[tuple[str, int, str]]:
    resolved = root.resolve(strict=True)
    return [
        (
            path.relative_to(resolved).as_posix(),
            path.stat().st_size,
            sha256_file(path),
        )
        for path in regular_files(resolved)
    ]


def _clone_case(source_relative: str, destination: Path) -> Path:
    source = (ROOT / source_relative).resolve(strict=True)
    try:
        source.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeIntegrityError("SUITE_WORKSPACE_ESCAPES_PROJECT") from exc
    if not source.is_dir() or destination.exists() or destination.is_symlink():
        raise RuntimeIntegrityError("CASE_CLONE_DESTINATION_INVALID")
    source_receipt = _tree_receipt(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    if _tree_receipt(destination) != source_receipt:
        raise RuntimeIntegrityError("CASE_CLONE_HASH_MISMATCH")
    return destination


def _promote_case_artifacts(agent_output: Path, case_root: Path) -> None:
    for name in RUNNER_ARTIFACTS:
        source = agent_output / name
        target = case_root / name
        if not source.is_file() or target.exists() or target.is_symlink():
            raise RuntimeIntegrityError("RUNNER_ARTIFACT_PROMOTION_INVALID")
        shutil.copy2(source, target)


def _case_status(
    *,
    logical_model: str,
    requested_model: str,
    result: Any,
    evaluation: Mapping[str, Any],
    replay_match: bool,
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = result.metadata
    actual_model = metadata.get("actual_model")
    exact_identity = actual_model == requested_model
    eligible = evaluation.get("eligibility", {}).get("eligible") is True
    provider_responded = isinstance(actual_model, str) and bool(actual_model)
    failure = metadata.get("failure")
    terminal_provider_failure = bool(
        not result.complete
        and isinstance(failure, Mapping)
        and failure.get("type") == "ProviderError"
    )
    if terminal_provider_failure:
        run_status = "BLOCKED"
        provider_status = "UNAVAILABLE_AFTER_RETRIES"
    elif result.complete and exact_identity and eligible and replay_match and scan.get("passed"):
        run_status = "PASS"
        provider_status = "AVAILABLE"
    elif provider_responded:
        run_status = "FAIL"
        provider_status = "RESPONDED"
    else:
        run_status = "BLOCKED"
        provider_status = "UNAVAILABLE_AFTER_RETRIES"
    return {
        "schema_version": "xtb_case_status/2.0",
        "logical_model": logical_model,
        "case_id": metadata.get("case_id"),
        "provider_status": provider_status,
        "run_status": run_status,
        "runner_terminal_status": result.status,
        "requested_model_id": requested_model,
        "actual_model_id": actual_model,
        "exact_model_identity": exact_identity,
        "eligible": eligible,
        "raw_score": None if terminal_provider_failure else evaluation.get("raw_score_before_scientific_cap"),
        "official_score": None if terminal_provider_failure else evaluation.get("official_score"),
        "tool_calls": metadata.get("counts", {}).get("tool_calls"),
        "xtb_attempts": metadata.get("counts", {}).get("execution_attempts"),
        "native_executions": metadata.get("counts", {}).get("native_executions"),
        "wall_time_seconds": metadata.get("duration_seconds"),
        "provider_attempts": metadata.get("counts", {}).get("provider_attempts"),
        "provider_retries": metadata.get("counts", {}).get("provider_retries"),
        "token_usage": metadata.get("token_usage", {}),
        "replay_match": replay_match,
        "secret_scan_passed": scan.get("secret_finding_count") == 0,
        "runtime_path_scan_passed": scan.get("runtime_absolute_path_finding_count") == 0,
        "failure": failure,
    }


def run_campaign(
    *,
    suite_path: Path,
    output_root: Path,
    models: tuple[str, ...],
) -> dict[str, Any]:
    _pre_model_gate()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeIntegrityError("CAMPAIGN_OUTPUT_EXISTS_OVERWRITE_FORBIDDEN")
    cases = _suite_cases(suite_path)
    contracts = ToolContractSet.load(ROOT / "configs" / "tool_contracts.json")
    rows: list[dict[str, Any]] = []

    for logical_model in models:
        model = ModelConfig.load_registry(
            ROOT / "configs" / "models.example.yaml", logical_model
        )
        model_root = output_root / logical_model
        for case in cases:
            case_root = model_root / case["case_id"]
            workspace = _clone_case(case["workspace"], case_root / "workspace")
            task = TaskConfig.load(ROOT / case["config"])
            registry = ToolRegistry(
                ROOT,
                task_config_path=case["config"],
                case_root=workspace.relative_to(ROOT).as_posix(),
            )
            agent_output = case_root / "agent_output"
            runner = AgentRunner(
                task_config=task,
                model_config=model,
                logical_model=logical_model,
                output_dir=agent_output,
                prompt=_formal_prompt(),
                tool_contracts=contracts,
                tool_registry=registry,
                transport=None,
                execution_mode="REAL_API",
                native_mode="LIVE_XTB",
                run_type="FORMAL_MODEL",
                run_id=f"20260805-{logical_model}-{case['case_id']}",
                artifact_root=workspace,
            )
            result = runner.run(raise_on_failure=False)
            _promote_case_artifacts(agent_output, case_root)

            evaluation_path = case_root / "evaluation.json"
            evaluation, frozen_bytes = deterministic_evaluation(
                final_answer=case_root / "final_answer.json",
                trajectory=case_root / "trajectory.jsonl",
                artifacts=workspace,
                output=evaluation_path,
                config_path=ROOT / case["config"],
            )
            replay, replay_bytes = deterministic_evaluation(
                final_answer=case_root / "final_answer.json",
                trajectory=case_root / "trajectory.jsonl",
                artifacts=workspace,
                config_path=ROOT / case["config"],
            )
            replay_match = frozen_bytes == replay_bytes and evaluation == replay
            replay_record = {
                "schema_version": "xtb_replay_verification/2.0",
                "logical_model": logical_model,
                "case_id": case["case_id"],
                "canonical_match": replay_match,
                "stored_evaluation_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
                "replay_evaluation_sha256": hashlib.sha256(replay_bytes).hexdigest(),
            }
            write_json_exclusive(case_root / "replay_verification.json", replay_record)
            scan = scan_tree(case_root)
            status = _case_status(
                logical_model=logical_model,
                requested_model=model.model,
                result=result,
                evaluation=evaluation,
                replay_match=replay_match,
                scan=scan,
            )
            write_json_exclusive(case_root / "status.json", status)
            rows.append(status)
            print(json.dumps(status, ensure_ascii=True, sort_keys=True), flush=True)

        suite_evaluation = evaluate_suite(suite_path, model_root)
        write_json_exclusive(model_root / "suite_evaluation.json", suite_evaluation)

    campaign = {
        "schema_version": "xtb_model_campaign/2.0",
        "models": list(models),
        "case_count_per_model": len(cases),
        "run_count": len(rows),
        "all_replays_match": all(row["replay_match"] for row in rows),
        "all_exact_model_identities": all(row["exact_model_identity"] for row in rows),
        "rows": rows,
    }
    write_json_exclusive(output_root / "campaign_summary.json", campaign)
    return campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=ROOT / "organizer_private" / "suite.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=FORMAL_MODELS)
    args = parser.parse_args(argv)
    selected = tuple(args.model) if args.model else FORMAL_MODELS
    campaign = run_campaign(
        suite_path=args.suite.resolve(),
        output_root=args.output.resolve(),
        models=selected,
    )
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "run_count": campaign["run_count"],
                "all_replays_match": campaign["all_replays_match"],
                "all_exact_model_identities": campaign["all_exact_model_identities"],
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

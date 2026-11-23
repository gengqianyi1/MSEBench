"""Run exactly one authorized formal SiliconFlow model in an isolated workspace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate import evaluation_bytes
from framework import AgentRunner, ModelConfig, TaskConfig, ToolContractSet
from scripts.runtime_support import (
    RuntimeIntegrityError,
    clone_frozen_case,
    deterministic_evaluation,
    load_json,
    pre_model_source_fingerprint,
    promote_formal_trajectory,
    scan_tree,
)
from tools import ToolRegistry


FORMAL_MODELS = ("kimi", "deepseek", "glm")


def _formal_prompt() -> str:
    """Return the complete public task plus the concise protocol guard."""

    task = (ROOT / "TASK.md").read_text(encoding="utf-8").rstrip()
    guard = (ROOT / "prompts" / "agent_prompt.md").read_text(encoding="utf-8").rstrip()
    if not task or not guard:
        raise RuntimeIntegrityError("FORMAL_PROMPT_COMPONENT_MISSING")
    return task + "\n\n---\n\n# Additional protocol guard\n\n" + guard + "\n"


def _run_id(logical_model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{logical_model}-{stamp}-{uuid.uuid4().hex[:10]}"


def _pre_model_gate() -> None:
    path = ROOT / "configs" / "release.json"
    if not path.is_file():
        raise RuntimeIntegrityError("PRE_MODEL_GATES_NOT_RECORDED")
    value = load_json(path)
    if not isinstance(value, dict) or value.get("pre_model_gates_passed") is not True:
        raise RuntimeIntegrityError("PRE_MODEL_GATES_NOT_PASSED")
    if value.get("status") not in {"PRE_MODEL_GATES_PASS", "PASS"}:
        raise RuntimeIntegrityError("PRE_MODEL_RELEASE_STATE_INVALID")
    if value.get("pre_model_source_sha256") != pre_model_source_fingerprint(ROOT):
        raise RuntimeIntegrityError("PRE_MODEL_GATE_RECEIPT_STALE")


def _relocate_failed(run_root: Path, logical_model: str) -> Path:
    failed_parent = ROOT / "runs_failed" / logical_model
    failed_parent.mkdir(parents=True, exist_ok=True)
    target = failed_parent / run_root.name
    if target.exists() or target.is_symlink():
        raise RuntimeIntegrityError("failed-run destination already exists")
    run_root.rename(target)
    model_parent = run_root.parent
    try:
        model_parent.rmdir()
    except OSError:
        pass
    return target


def _failed_destination(run_root: Path, logical_model: str) -> Path:
    return ROOT / "runs_failed" / logical_model / run_root.name


def _write_run_summary(run_root: Path, summary: dict[str, object]) -> None:
    with (run_root / "run_summary.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def run_formal_model(logical_model: str, *, run_id: str | None = None) -> dict[str, object]:
    if logical_model not in FORMAL_MODELS:
        raise RuntimeIntegrityError("unknown logical model")
    _pre_model_gate()
    if (ROOT / "trajectories" / logical_model).exists():
        raise RuntimeIntegrityError("formal trajectory already exists; overwrite is forbidden")

    identifier = run_id or _run_id(logical_model)
    if not identifier or any(value in identifier for value in ("/", "\\", ":", "..")):
        raise RuntimeIntegrityError("run id is unsafe")
    run_root = ROOT / "runs" / logical_model / identifier
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeIntegrityError("run directory already exists")
    stage = "CLONE_FROZEN_CASE"
    try:
        workspace = clone_frozen_case(run_root / "workspace")
        agent_output = run_root / "agent_output"

        stage = "LOAD_FORMAL_CONFIGURATION"
        task = TaskConfig.load(ROOT / "configs" / "task.yaml")
        model = ModelConfig.load_registry(
            ROOT / "configs" / "models.example.yaml", logical_model
        )
        contracts = ToolContractSet.load(ROOT / "configs" / "tool_contracts.json")
        case_root = workspace.relative_to(ROOT).as_posix()
        registry = ToolRegistry(ROOT, case_root=case_root)
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
            run_id=identifier,
            artifact_root=workspace,
        )
        stage = "RUN_AGENT"
        result = runner.run(raise_on_failure=False)

        stage = "EVALUATE_AND_REPLAY"
        evaluation_path = run_root / "evaluation.json"
        evaluation, frozen_bytes = deterministic_evaluation(
            final_answer=agent_output / "final_answer.json",
            trajectory=agent_output / "trajectory.jsonl",
            artifacts=workspace,
            output=evaluation_path,
        )
        if evaluation_path.read_bytes() != frozen_bytes:
            raise RuntimeIntegrityError("written evaluation differs from deterministic bytes")

        metadata = load_json(agent_output / "metadata.json")
        exact_identity = bool(
            isinstance(metadata, dict)
            and metadata.get("requested_model") == model.model
            and metadata.get("actual_model") == model.model
            and metadata.get("execution_mode") == "REAL_API"
            and metadata.get("native_mode") == "LIVE_XTB"
        )
        eligible = bool(evaluation.get("eligibility", {}).get("eligible"))
        scan = scan_tree(run_root)
        passed = bool(result.complete and exact_identity and eligible and scan["passed"])
        summary: dict[str, object] = {
            "schema_version": "1.0",
            "logical_model": logical_model,
            "requested_model": model.model,
            "run_id": identifier,
            "runner_status": result.status,
            "exact_model_identity": exact_identity,
            "evaluator_eligible": eligible,
            "official_score": evaluation.get("official_score", 0),
            "secret_finding_count": scan["secret_finding_count"],
            "runtime_absolute_path_finding_count": scan[
                "runtime_absolute_path_finding_count"
            ],
            "status": "PASS" if passed else "FAILED_NOT_PROMOTED",
        }

        if not passed:
            failed_target = _failed_destination(run_root, logical_model)
            summary["local_disposition"] = failed_target.relative_to(ROOT).as_posix()
            _write_run_summary(run_root, summary)
            _relocate_failed(run_root, logical_model)
            return summary

        stage = "PROMOTE_FORMAL_TRAJECTORY"
        promoted = promote_formal_trajectory(
            logical_model=logical_model,
            agent_output=agent_output,
            workspace=workspace,
            evaluation_path=evaluation_path,
            candidate_root=run_root / "promotion_candidate",
        )
        summary["trajectory"] = promoted.relative_to(ROOT).as_posix()
        summary["local_disposition"] = run_root.relative_to(ROOT).as_posix()
        _write_run_summary(run_root, summary)
        return summary
    except BaseException as exc:
        # Preserve every partial formal run using only non-reflective failure
        # metadata.  Provider payloads and credentials are never copied here.
        if run_root.exists() and run_root.is_dir():
            failed_target = _failed_destination(run_root, logical_model)
            receipt: dict[str, object] = {
                "schema_version": "1.0",
                "logical_model": logical_model,
                "run_id": identifier,
                "runner_status": "FAILED",
                "status": "FAILED_NOT_PROMOTED",
                "failure_stage": stage,
                "error_type": type(exc).__name__,
                "local_disposition": failed_target.relative_to(ROOT).as_posix(),
            }
            if not (run_root / "run_summary.json").exists():
                _write_run_summary(run_root, receipt)
            if not failed_target.exists():
                _relocate_failed(run_root, logical_model)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeIntegrityError("FORMAL_RUN_FAILED_AFTER_PRESERVATION") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one exact formal model with REAL_API + LIVE_XTB"
    )
    parser.add_argument("--model", required=True, choices=FORMAL_MODELS)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        summary = run_formal_model(args.model, run_id=args.run_id)
    except (RuntimeIntegrityError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error_type": type(exc).__name__},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

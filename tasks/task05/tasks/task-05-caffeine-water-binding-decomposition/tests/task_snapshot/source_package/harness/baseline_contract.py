"""Build a baseline scientific manifest from the same frozen contract as model runs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import TaskConfig
from .budget import BudgetLedger, FINALIZATION_NOTICE
from .native_task_runner import upgrade_scientific_manifest
from .release_contract import load_release_contract
from .runtime_checks import run_runtime_checks
from .schema_validation import validate_budget_ledger, validate_scientific_manifest


def baseline_budget_ledger(
    task_config: TaskConfig,
    *,
    xtb_jobs: int,
    xtb_scan_points: int,
    compute_budget: Mapping[str, object] | None = None,
    calculations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if compute_budget is not None:
        records = list(calculations or [])
        if len(records) != int(xtb_jobs):
            raise ValueError("weighted baseline calculations must cover every xTB job")
        ledger = BudgetLedger(
            task_config.budget_limits,
            compute_budget=compute_budget,
            schema_dir=Path(__file__).resolve().parents[1] / "schemas",
            clock=lambda: 0.0,
        )
        for record in sorted(records, key=lambda item: int(item.get("sequence", 0))):
            tool = ledger.record_tool_call("xtb_run", "active")
            if tool is None or not tool.allowed:
                raise ValueError(tool.reason if tool is not None else "baseline tool reservation failed")
            decision = ledger.reserve_xtb_compute(
                str(record["job_id"]),
                compute_class=str(record["compute_class"]),
                credits=int(record["credits"]),
                candidate_id=str(record["candidate_id"]),
                condition=str(record["condition"]),
                parent_job_id=(str(record["parent_job_id"]) if record.get("parent_job_id") else None),
                recovery_of_job_id=(
                    str(record["recovery_of_job_id"])
                    if record.get("recovery_of_job_id")
                    else None
                ),
            )
            if not decision.allowed:
                raise ValueError(decision.reason or "baseline compute reservation failed")
            outcome = str(record.get("outcome") or "completed")
            ledger.record_xtb_job_outcome(str(record["job_id"]), outcome)
        document = ledger.to_document()
        if int(document["totals"]["xtb_scan_points"]) != int(xtb_scan_points):
            raise ValueError("weighted baseline scan accounting is inconsistent")
        return document

    totals = {
        "agent_turns": 0,
        "api_attempts": 0,
        "tool_calls": int(xtb_jobs),
        "xtb_jobs": int(xtb_jobs),
        "xtb_scan_points": int(xtb_scan_points),
        "wall_seconds": 0.0,
        "requested_tokens": 0,
    }
    limits = deepcopy(dict(task_config.budget_limits))
    limits.setdefault("finalization_turns", 2)
    limits.setdefault("finalization_wall_seconds", 600)
    if int(xtb_jobs) > int(limits["xtb_jobs"]):
        raise ValueError("baseline xTB job count exceeds the task budget")
    if int(xtb_scan_points) > int(limits["xtb_scan_points"]):
        raise ValueError("baseline scan-point count exceeds the task budget")
    document = {
        "schema_version": "2.0.0",
        "agent_turns_used": 0,
        "api_requests_used": 0,
        "empty_response_retries": 0,
        "recovery_attempts": 0,
        "tool_calls_used": int(xtb_jobs),
        "xtb_jobs_reserved": int(xtb_jobs),
        "xtb_jobs_completed": int(xtb_jobs),
        "wall_time_seconds": 0.0,
        "limits": limits,
        "events": [{
            "sequence": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": "finalize",
            "logical_turn": 0,
            "api_attempt": 0,
            "tool_calls": int(xtb_jobs),
            "xtb_jobs": int(xtb_jobs),
            "xtb_scan_points": int(xtb_scan_points),
            "elapsed_wall_seconds": 0.0,
            "retry_or_resume_reason": None,
            "cumulative": totals,
        }],
        "totals": totals,
        "phase": "active",
        "exhaustion_trigger": None,
        "rejected_requests": 0,
        "finalization_turns_used": 0,
        "finalization_tool_calls": 0,
        "phase_history": [{"phase": "active", "trigger": "run_started"}],
        "finalization_notice": FINALIZATION_NOTICE,
    }
    return document


def finalize_baseline_submission(
    *,
    bundle_root: Path,
    task_dir: Path,
    submission_dir: Path,
    old_manifest: Mapping[str, Any],
    scientific_problem_id: str,
    xtb_scan_points: int = 0,
) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    task_dir = Path(task_dir).resolve()
    submission_dir = Path(submission_dir).resolve()
    config = TaskConfig.from_yaml(task_dir / "task.yaml")
    contract = load_release_contract(bundle_root)
    runtime = run_runtime_checks(task_dir, config.runtime_checks, bundle_root / "schemas")
    if runtime.get("status") != "passed":
        raise RuntimeError(f"baseline runtime checks failed for {config.task_id}: {runtime}")
    xtb = next(item for item in runtime["binaries"] if item["name"] == "xtb")
    environment = {
        "environment_lock_sha256": runtime["environment_lock"]["actual_sha256"],
        "task_config_sha256": runtime["hashes"]["task_config_sha256"],
        "public_inputs_sha256": runtime["hashes"]["public_inputs_sha256"],
        "xtb_executable_sha256": xtb["sha256"],
    }
    jobs = [item for item in old_manifest.get("jobs", []) if isinstance(item, dict)]
    calculations: list[Mapping[str, Any]] | None = None
    compute_budget: Mapping[str, object] | None = None
    if config.task_id == "task_04":
        candidate_manifest_path = submission_dir / "candidate_manifest.json"
        if not candidate_manifest_path.is_file():
            raise RuntimeError("Task 04 baseline requires candidate_manifest.json for credit accounting")
        import json

        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        calculations = [item for item in candidate_manifest.get("calculations", []) if isinstance(item, dict)]
        compute_budget = contract.task_compute_budget(config.task_id)
    ledger = baseline_budget_ledger(
        config,
        xtb_jobs=len(jobs),
        xtb_scan_points=int(xtb_scan_points),
        compute_budget=compute_budget,
        calculations=calculations,
    )
    validate_budget_ledger(ledger, bundle_root / "schemas")
    upgraded = upgrade_scientific_manifest(
        old_manifest,
        task_id=config.task_id,
        scientific_problem_id=scientific_problem_id,
        run_id=f"baseline-{config.task_id}",
        task_version=contract.task_version(config.task_id),
        grader_version=contract.task_grader_version(config.task_id),
        package_release=contract.task_package_release(config.task_id),
        runtime_checks=runtime,
        budget_ledger=ledger,
        environment=environment,
        submission_dir=submission_dir,
    )
    validate_scientific_manifest(upgraded, bundle_root / "schemas")
    return upgraded

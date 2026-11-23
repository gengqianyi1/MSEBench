"""Formal JSON Schema validation plus cross-event invariants.

JSON Schema validates document shape.  The cumulative budget ledger also needs
ordered comparisons between adjacent events, which are enforced here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


COUNTER_FIELDS = (
    "agent_turns",
    "api_attempts",
    "tool_calls",
    "xtb_jobs",
    "xtb_scan_points",
    "wall_seconds",
    "requested_tokens",
)


def load_schema(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON Schema from *path*."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_document(document: Mapping[str, Any], schema_path: Path) -> None:
    """Raise ``jsonschema.ValidationError`` when a formal document is invalid."""
    schema = load_schema(schema_path)
    Draft202012Validator(schema).validate(dict(document))


def validate_budget_ledger(ledger: Mapping[str, Any], schema_dir: Path) -> None:
    """Validate ledger shape, append ordering, monotonic totals and limits."""
    validate_document(ledger, Path(schema_dir) / "budget_ledger.schema.json")
    previous = {field: 0 for field in COUNTER_FIELDS}
    previous_sequence = 0
    events = list(ledger.get("events", []))
    for event in events:
        sequence = int(event["sequence"])
        if sequence != previous_sequence + 1:
            raise ValueError("budget event sequence must be contiguous and monotonic")
        current = event["cumulative"]
        for field in COUNTER_FIELDS:
            if current[field] < previous[field]:
                raise ValueError(f"budget cumulative field is not monotonic: {field}")
        if event["logical_turn"] > current["agent_turns"]:
            raise ValueError("logical_turn exceeds cumulative agent_turns")
        if event["api_attempt"] > current["api_attempts"]:
            raise ValueError("api_attempt exceeds cumulative api_attempts")
        previous = dict(current)
        previous_sequence = sequence

    totals = dict(ledger["totals"])
    if events and totals != previous:
        raise ValueError("budget totals must equal the final cumulative event")
    if not events and any(totals[field] != 0 for field in COUNTER_FIELDS):
        raise ValueError("an empty budget ledger must have zero totals")
    if int(ledger["xtb_jobs_completed"]) > int(ledger["xtb_jobs_reserved"]):
        raise ValueError("completed xTB jobs exceed reserved xTB jobs")

    limits = ledger["limits"]
    limit_map = {
        "agent_turns": "agent_turns",
        "api_attempts": "api_attempts",
        "tool_calls": "tool_calls",
        "xtb_jobs": "xtb_jobs",
        "xtb_scan_points": "xtb_scan_points",
        "wall_seconds": "wall_seconds",
    }
    for total_field, limit_field in limit_map.items():
        if totals[total_field] > limits[limit_field]:
            raise ValueError(f"budget totals may never exceed their limit: {total_field}")
    if int(ledger["finalization_turns_used"]) > int(limits["finalization_turns"]):
        raise ValueError("finalization turns may never exceed their limit")
    phase_history = list(ledger["phase_history"])
    if phase_history[0] != {"phase": "active", "trigger": "run_started"}:
        raise ValueError("budget phase history must begin with active/run_started")
    if ledger["phase"] == "finalizing" and not any(
        entry["phase"] == "finalizing" for entry in phase_history
    ):
        raise ValueError("finalizing ledger is missing its phase transition")
    if ledger.get("schema_version") == "3.0.0":
        budget = dict(ledger["compute_budget"])
        compute_totals = dict(ledger["compute_totals"])
        compute_events = list(ledger["compute_events"])
        expected = {
            "main_credits_used": 0,
            "recovery_credits_used": 0,
            "total_credits_used": 0,
            "main_credits_remaining": int(budget["main_credits"]),
            "recovery_credits_remaining": int(budget["recovery_credits"]),
            "admitted_jobs": 0,
        }
        seen_jobs: set[str] = set()
        for index, event in enumerate(compute_events, start=1):
            if int(event["sequence"]) != index:
                raise ValueError("compute event sequence must be contiguous and monotonic")
            if event["event_type"] == "compute_admitted":
                job_id = str(event["job_id"])
                if job_id in seen_jobs:
                    raise ValueError("compute job ids must be unique")
                seen_jobs.add(job_id)
                pool = str(event["pool"])
                credits = int(event["credits"])
                expected[f"{pool}_credits_used"] += credits
                expected["total_credits_used"] += credits
                expected["admitted_jobs"] += 1
                expected["main_credits_remaining"] = int(budget["main_credits"]) - expected["main_credits_used"]
                expected["recovery_credits_remaining"] = int(budget["recovery_credits"]) - expected["recovery_credits_used"]
            if dict(event["cumulative"]) != expected:
                raise ValueError("compute credit totals disagree with append-only events")
        if compute_totals != expected:
            raise ValueError("compute credit totals disagree with append-only events")
        if expected["main_credits_used"] > int(budget["main_credits"]):
            raise ValueError("main compute credits exceed their limit")
        if expected["recovery_credits_used"] > int(budget["recovery_credits"]):
            raise ValueError("recovery compute credits exceed their limit")
        if expected["admitted_jobs"] != int(totals["xtb_jobs"]):
            raise ValueError("compute admitted jobs disagree with xTB job totals")


def validate_runtime_checks(document: Mapping[str, Any], schema_dir: Path) -> None:
    validate_document(document, Path(schema_dir) / "runtime_checks.schema.json")


def validate_harness_manifest(document: Mapping[str, Any], schema_dir: Path) -> None:
    validate_document(document, Path(schema_dir) / "harness_run_manifest.schema.json")
    validate_runtime_checks(document["runtime_checks"], schema_dir)
    validate_budget_ledger(document["budget_ledger"], schema_dir)


def validate_scientific_manifest(document: Mapping[str, Any], schema_dir: Path) -> None:
    validate_document(document, Path(schema_dir) / "scientific_run_manifest.schema.json")
    validate_runtime_checks(document["runtime_checks"], schema_dir)
    validate_budget_ledger(document["budget_ledger"], schema_dir)
    jobs = [job for job in document.get("jobs", []) if isinstance(job, Mapping)]
    completed_jobs = sum(str(job.get("status")) == "completed" for job in jobs)
    ledger = document["budget_ledger"]
    if int(ledger["xtb_jobs_reserved"]) != len(jobs):
        raise ValueError("reserved xTB job count disagrees with scientific manifest jobs")
    if int(ledger["xtb_jobs_completed"]) != completed_jobs:
        raise ValueError("completed xTB job count disagrees with scientific manifest jobs")


def validate_score_report(document: Mapping[str, Any], schema_dir: Path) -> None:
    """Validate the common score shape and cross-field status invariants."""
    validate_document(document, Path(schema_dir) / "score.schema.json")
    status = document.get("submission_status")
    grading = document.get("grading_status")
    if status == "INVALID_SUBMISSION":
        if grading != "NOT_SCORED":
            raise ValueError("invalid submissions must not be scientifically scored")
        for field in ("score", "task_local_score", "comparison_score", "scientific_basis_score"):
            if document.get(field) is not None:
                raise ValueError(f"invalid submission field must be null: {field}")
    if grading == "SCORED":
        for field in ("score", "task_local_score", "comparison_score", "scientific_basis_score"):
            if not isinstance(document.get(field), (int, float)):
                raise ValueError(f"scored report requires numeric {field}")

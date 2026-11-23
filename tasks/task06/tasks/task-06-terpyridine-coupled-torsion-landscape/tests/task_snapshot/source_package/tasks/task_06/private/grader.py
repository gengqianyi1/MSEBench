"""Geometry-derived Task 06 grader for the minimal trusted Harness."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema
import yaml


VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from evaluator.core import TrustedJob, reference_manifest  # noqa: E402
from evaluator.coupled_goal_driven import (  # noqa: E402
    _extract_verified_evidence,
    _observable_trajectory,
    score_coupled_claims,
)
from evaluator.coupled_submission import validate_coupled_submission  # noqa: E402
from tools.xtb_parse import parse_xtb_output  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zero(reason: str, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "task_id": "task_06",
        "task_version": "1.16.0",
        "grader_version": "1.22.0",
        "hard_zero": True,
        "hard_zero_reason": reason,
        "errors": errors or [reason],
        "sections": [],
        "derived": {},
        "score": 0.0,
        "task_local_score": 0.0,
        "scientific_basis_score": 0.0,
        "comparison_score": 0.0,
    }


def _control_name(command: list[str]) -> str | None:
    try:
        index = command.index("--input")
    except ValueError:
        return None
    return Path(command[index + 1]).name if index + 1 < len(command) else None


def _scan_coordinate(control: Path | None) -> list[int] | None:
    if control is None or not control.is_file():
        return None
    text = control.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"dihedral\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    return [int(match.group(index)) for index in range(1, 5)] if match else None


def _trusted_jobs(run_root: Path) -> tuple[dict[str, TrustedJob], dict[str, Any], list[str]]:
    jobs_path = run_root / "jobs.json"
    if not jobs_path.is_file():
        return {}, {"jobs": [], "budget_ledger": {}}, ["jobs.json is missing"]
    raw_jobs = _load_json(jobs_path)
    if not isinstance(raw_jobs, list):
        return {}, {"jobs": [], "budget_ledger": {}}, ["jobs.json is not an array"]

    jobs: dict[str, TrustedJob] = {}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    scan_points = 0
    for raw in raw_jobs:
        if not isinstance(raw, dict):
            errors.append("jobs.json contains a non-object record")
            continue
        job_id = str(raw.get("job_id") or "")
        job_dir = run_root / "jobs" / job_id
        if not job_id or not job_dir.is_dir():
            errors.append(f"missing authoritative job directory: {job_id or '<empty>'}")
            continue
        command = [str(value) for value in raw.get("command", [])]
        input_name = Path(str(raw.get("input") or "")).name
        input_path = job_dir / input_name
        control_name = _control_name(command)
        control_path = job_dir / control_name if control_name else None
        scan_point_count = int(raw.get("scan_point_count", 0) or 0)
        scan_points += scan_point_count
        record: dict[str, Any] = {
            "job_id": job_id,
            "operation": str(raw.get("operation") or ""),
            "status": str(raw.get("status") or ""),
            "exit_code": raw.get("return_code"),
            "workdir": f"xtb_jobs/{job_id}",
            "input_geometry_sha256": _sha256(input_path) if input_path.is_file() else "",
            "scan_point_count": scan_point_count,
            "scan_coordinate_atom_ids": _scan_coordinate(control_path),
            "request": {
                "arguments": command[2:] if len(command) > 2 else command,
                "geometry": input_name,
                "detailed_input": control_name,
            },
        }
        try:
            parsed = parse_xtb_output(job_dir)
        except Exception as exc:
            errors.append(f"{job_id}: parser failed: {type(exc).__name__}: {exc}")
            continue
        records.append(record)
        jobs[job_id] = TrustedJob(
            job_id=job_id,
            record=record,
            workdir=job_dir,
            parsed=parsed,
        )
    manifest = {
        "jobs": records,
        "budget_ledger": {
            "compute_exhausted": len(raw_jobs) >= 12 or scan_points >= 72,
            "totals": {"xtb_jobs": len(raw_jobs), "xtb_scan_points": scan_points},
        },
    }
    return jobs, manifest, errors


def grade_task06(
    submission_root: Path,
    run_root: Path,
    reference_root: Path,
    public_root: Path,
) -> dict[str, Any]:
    submission_root = Path(submission_root)
    run_root = Path(run_root)
    final_path = submission_root / "final_answer.json"
    if not final_path.is_file():
        return _zero("final_answer.json is missing")
    try:
        final = _load_json(final_path)
    except (OSError, json.JSONDecodeError) as exc:
        return _zero("final_answer.json is invalid JSON", [str(exc)])

    schema = _load_json(Path(public_root) / "schema" / "final_answer.schema.json")
    schema_errors = sorted(
        error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(final)
    )
    if schema_errors:
        return _zero("final_answer.json does not satisfy the public schema", schema_errors)
    semantic_errors = validate_coupled_submission(final)

    jobs, manifest, job_errors = _trusted_jobs(run_root)
    evidence = _extract_verified_evidence(
        jobs,
        Path(public_root) / "input" / "terpyridine_seed.xyz",
        manifest,
    )
    evidence["integrity_errors"] = job_errors
    trajectory = _observable_trajectory(manifest, final, evidence)
    scoring = yaml.safe_load(
        (Path(__file__).resolve().parent / "scoring.yaml").read_text(encoding="utf-8")
    ) or {}
    reference = reference_manifest(reference_root)
    report = score_coupled_claims(
        final,
        evidence,
        reference,
        trajectory,
        scoring,
        schema_valid=True,
        semantic_errors=semantic_errors,
    )
    score = float(report.get("total_score", report.get("score", 0.0)))
    report.update(
        {
            "task_id": "task_06",
            "task_version": "1.16.0",
            "grader_version": "1.22.0",
            "hard_zero": False,
            "score": score,
            "task_local_score": score,
            "scientific_basis_score": score,
            "comparison_score": score,
            "contract_errors": list(semantic_errors) + list(job_errors),
        }
    )
    report.setdefault("metadata", {})["required_output_status"] = {
        "final_answer.json": final_path.is_file(),
        "report.md": (submission_root / "report.md").is_file(),
    }
    return report

"""Standard submission validation and task-specific grading dispatch.

Score semantics (spec 4.1/4.3): the official score is the scientific rubric
raw score on 0--100, and the four official fields are numerically identical:

    score == task_local_score == scientific_basis_score == comparison_score

PASS/FAIL is derived from that raw scientific score against the fixed
pass_score.  Invalid submissions are INVALID_SUBMISSION / NOT_SCORED and
are never represented as scientific zeroes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical_replay import canonical_report_hash, canonicalize_report
from .schema_validation import validate_score_report
from .utils import load_json, save_json


@dataclass(frozen=True)
class GradeContext:
    instance_id: str
    task_version: str
    task_dir: Path
    submission_dir: Path
    run_dir: Path
    private_data_dir: Path | None
    run_manifest: dict[str, Any]


def validate_submission(submission_dir: Path, task_cfg) -> tuple[str, list[str]]:
    missing: list[str] = []
    if task_cfg.final_answer_required and not (submission_dir / task_cfg.final_answer_file).is_file():
        missing.append(task_cfg.final_answer_file)
    for artifact in task_cfg.required_artifacts:
        exists = any(submission_dir.glob(artifact)) if any(char in artifact for char in "*?[") else (submission_dir / artifact).exists()
        if not exists and artifact not in missing:
            missing.append(artifact)
    return ("SUBMITTED", []) if not missing else ("INVALID_SUBMISSION", missing)


def _submission_sha256(submission_dir: Path) -> str:
    """Deterministic SHA-256 over the whole immutable submission tree."""
    submission_dir = Path(submission_dir)
    if not submission_dir.is_dir():
        return hashlib.sha256(b"").hexdigest()
    try:
        from .runtime_checks import sha256_tree
        return sha256_tree(submission_dir)
    except Exception:
        digest = hashlib.sha256()
        for path in sorted(item for item in submission_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(submission_dir).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(bytes([0]))
            digest.update(path.read_bytes())
        return digest.hexdigest()


def evaluate_submission(
    run_dir: Path,
    submission_dir: Path,
    task_cfg,
    rubric: dict | None = None,
    task_dir: Path | None = None,
    execution_status: str = "COMPLETED",
    run_manifest: dict | None = None,
) -> dict:
    """Evaluate one immutable submission and return the common score schema."""
    run_dir = Path(run_dir).resolve()
    submission_dir = Path(submission_dir).resolve()
    task_dir = Path(task_dir or ".").resolve()
    run_manifest = dict(run_manifest or {})

    submission_status, missing = validate_submission(submission_dir, task_cfg)
    semantics = dict(getattr(task_cfg, "score_semantics", {}) or {})
    pass_score = float(getattr(task_cfg, "pass_score", 50.0))
    base = {
        "schema_version": "2.0",
        "run_id": run_manifest.get("run_id"),
        "task_id": task_cfg.task_id,
        "task_version": task_cfg.version,
        "submission_sha256": _submission_sha256(submission_dir),
        "submission_valid": submission_status == "SUBMITTED",
        "execution_status": execution_status,
        "submission_status": submission_status,
        "grading_status": "NOT_SCORED",
        "scientific_outcome": "NOT_ASSESSED",
        "score": None,
        "task_local_score": None,
        "scientific_basis_score": None,
        "comparison_score": None,
        "pass_score": pass_score,
        "outcome": "NOT_ASSESSED",
        "max_score": task_cfg.max_score,
        "normalized_score": None,
        "comparison_normalized_score": None,
        "score_semantics": semantics,
        "sections": {},
        "hard_gates": {},
        "legacy_evidence": {},
        "subscores": {},
        "failure_codes": [f"MISSING:{item}" for item in missing],
        "scorer_version": task_cfg.grader_version if task_cfg.grader_path else (rubric or {}).get("version", "unknown"),
    }
    if submission_status != "SUBMITTED":
        base["evidence"] = {"mandatory_submission_gate": {"missing": missing}}
        _write_invalid_submission_evidence(run_dir, base)
        _validate_score(base, task_dir, task_cfg)
        return base

    try:
        if task_cfg.scoring_type == "hard_rules":
            raw = _apply_rubric(submission_dir, rubric or {}, task_cfg)
        elif task_cfg.scoring_type == "custom":
            raw = _run_custom_grader(task_dir, submission_dir, run_dir, task_cfg, run_manifest)
        else:
            hard = _apply_rubric(submission_dir, rubric or {}, task_cfg)
            custom = _run_custom_grader(task_dir, submission_dir, run_dir, task_cfg, run_manifest)
            raw = {
                "score": float(hard.get("score", 0)) + float(custom.get("score", 0)),
                "subscores": {**hard.get("subscores", {}), **custom.get("subscores", {})},
                "failure_codes": hard.get("failure_codes", []) + custom.get("failure_codes", []),
                "evidence": {"hard_rules": hard.get("evidence", {}), "custom": custom.get("evidence", {})},
            }
        contract_errors = raw.get("contract_errors")
        if isinstance(contract_errors, list) and contract_errors:
            base["submission_status"] = "INVALID_SUBMISSION"
            base["submission_valid"] = False
            base["failure_codes"] = [f"CONTRACT:{item}" for item in contract_errors]
            base["evidence"] = {"submission_contract": {"violations": contract_errors}}
            _write_invalid_submission_evidence(run_dir, base)
            _validate_score(base, task_dir, task_cfg)
            return base
    except Exception as exc:
        base["grading_status"] = "GRADER_ERROR"
        base["failure_codes"] = ["GRADER_EXCEPTION"]
        base["grader_error"] = str(exc)
        base["grader_traceback"] = traceback.format_exc(limit=5)
        _validate_score(base, task_dir, task_cfg)
        return base

    # Official score: the raw scientific rubric score, clamped to 0--max_score.
    # All four official fields are numerically identical (spec 4.1/4.3).
    task_local = max(0.0, min(float(raw.get("score", 0.0)), task_cfg.max_score))
    scientific_basis = task_local
    comparison = task_local
    outcome = "PASS" if task_local >= pass_score else "FAIL"
    evidence = raw.get("evidence", {}) if isinstance(raw.get("evidence", {}), dict) else {}
    sections = raw.get("sections", {})
    if sections is None:
        sections = {}
    hard_gates = raw.get("hard_gates", {})
    if hard_gates is None:
        hard_gates = {}
    legacy_evidence = raw.get("legacy_evidence", {})
    if not isinstance(legacy_evidence, dict):
        legacy_evidence = {}

    base.update({
        "submission_sha256": _submission_sha256(submission_dir),
        "submission_valid": True,
        "grading_status": "SCORED",
        "scientific_outcome": outcome,
        "score": round(task_local, 6),
        "task_local_score": round(task_local, 6),
        "scientific_basis_score": round(scientific_basis, 6),
        "comparison_score": round(comparison, 6),
        "outcome": outcome,
        "normalized_score": round(task_local / task_cfg.max_score, 8),
        "comparison_normalized_score": round(comparison / task_cfg.max_score, 8),
        "sections": sections,
        "hard_gates": hard_gates,
        "legacy_evidence": legacy_evidence,
        "subscores": raw.get("subscores", {}),
        "failure_codes": raw.get("failure_codes", []),
        "evidence": evidence,
    })
    _validate_score(base, task_dir, task_cfg)
    return base


def _validate_score(report: dict[str, Any], task_dir: Path, task_cfg) -> None:
    schema_relative = (getattr(task_cfg, "manifest_schemas", {}) or {}).get("score")
    if not schema_relative:
        return
    project_root = task_dir.parents[1]
    schema_dir = project_root / "schemas"
    validate_score_report(report, schema_dir)


def _write_invalid_submission_evidence(run_dir: Path, report: dict[str, Any]) -> None:
    """Persist the auditable evidence set for a mandatory invalid submission."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "explain_report.json", report)
    save_json(run_dir / "explain_report_replay.json", report)
    canonical = canonicalize_report(report)
    canonical_hash = canonical_report_hash(report)
    save_json(run_dir / "canonical_explain_report.json", canonical)
    save_json(
        run_dir / "canonical_replay.json",
        {
            "schema_version": "1.0.0",
            "matched": True,
            "first_canonical_sha256": canonical_hash,
            "replay_canonical_sha256": canonical_hash,
        },
    )
    (run_dir / "grader_stdout.log").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "grader_stderr.log").write_text("", encoding="utf-8")


def _run_custom_grader(task_dir: Path, submission_dir: Path, run_dir: Path, task_cfg, run_manifest: dict) -> dict:
    grader_file = (task_dir / task_cfg.grader_path).resolve()
    try:
        grader_file.relative_to(task_dir.resolve())
    except ValueError as exc:
        raise ValueError("grader_path escapes task directory") from exc
    if not grader_file.is_file():
        raise FileNotFoundError(f"Custom grader not found: {grader_file}")

    spec = importlib.util.spec_from_file_location(f"chembench_grader_{task_cfg.task_id}", grader_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import grader: {grader_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grader = getattr(module, task_cfg.grader_entrypoint, None)
    if not callable(grader):
        raise AttributeError(f"Grader entrypoint is not callable: {task_cfg.grader_entrypoint}")

    private_data = (task_dir / task_cfg.private_data_path).resolve() if task_cfg.private_data_path else None
    context = GradeContext(
        instance_id=task_cfg.task_id,
        task_version=task_cfg.version,
        task_dir=task_dir,
        submission_dir=submission_dir,
        run_dir=run_dir,
        private_data_dir=private_data,
        run_manifest=run_manifest,
    )
    result = grader(context)
    if not isinstance(result, dict) or "score" not in result:
        raise TypeError("Custom grader must return a dict containing 'score'")
    return result


def _apply_rubric(submission_dir: Path, rubric: dict, task_cfg) -> dict:
    total = 0.0
    subscores: dict[str, float] = {}
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    for dim_name, dim_spec in rubric.get("dimensions", {}).items():
        weight = float(dim_spec.get("weight", 0))
        dim_score = 0.0
        dimension_evidence = []
        for check in dim_spec.get("checks", []):
            passed = _run_check(submission_dir, check)
            dimension_evidence.append({"name": check.get("name"), "passed": passed})
            if passed:
                dim_score += float(check.get("points", 0))
            elif check.get("critical", False):
                failures.append(f"FAILED:{dim_name}:{check.get('name', 'unnamed')}")
        subscores[dim_name] = min(dim_score, weight)
        evidence[dim_name] = dimension_evidence
        total += subscores[dim_name]
    return {"score": round(total, 6), "subscores": subscores, "failure_codes": failures, "evidence": evidence}


def _get_nested(data: Any, keys: list[Any]) -> tuple[bool, Any]:
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list):
            try:
                current = current[int(key)]
            except (ValueError, IndexError, TypeError):
                return False, None
        else:
            return False, None
    return True, current


def _run_check(submission_dir: Path, check: dict) -> bool:
    check_type = check.get("type", "file_exists")
    path = submission_dir / check.get("path", "")
    if check_type == "file_exists":
        pattern = check.get("path", "")
        return any(submission_dir.glob(pattern)) if any(char in pattern for char in "*?[") else path.exists()
    if not path.is_file():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    keys = check.get("keys", [])
    if not isinstance(keys, list):
        keys = [keys]
    found, value = _get_nested(data, keys)
    if check_type == "field_present":
        return found
    if check_type == "field_in_range" and found:
        try:
            number = float(value)
            return float(check.get("min", float("-inf"))) <= number <= float(check.get("max", float("inf")))
        except (TypeError, ValueError):
            return False
    if check_type == "count_in_range" and found:
        count = len(value) if isinstance(value, (list, dict)) else 0
        return int(check.get("min_count", 0)) <= count <= float(check.get("max_count", float("inf")))
    return False

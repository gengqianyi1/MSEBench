"""Command-line entry point for the independent xTB benchmark evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


# Support the documented ``python evaluator/evaluate.py`` invocation without
# importing anything from framework/ or tools/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from evaluator.scoring import ScoringConfigurationError, compute_score
    from evaluator.trusted_validation import (
        EVALUATOR_SCHEMA_VERSION,
        Issue,
        TrustedValidationError,
        canonical_json_bytes,
        load_json,
        load_trusted_configuration,
        sha256_file,
        stable_issues,
        validate_artifacts,
        validate_json_schema,
        validate_trajectory,
        verify_final_evidence,
        verify_trusted_runtime,
    )
else:
    from .scoring import ScoringConfigurationError, compute_score
    from .trusted_validation import (
        EVALUATOR_SCHEMA_VERSION,
        Issue,
        TrustedValidationError,
        canonical_json_bytes,
        load_json,
        load_trusted_configuration,
        sha256_file,
        stable_issues,
        validate_artifacts,
        validate_json_schema,
        validate_trajectory,
        verify_final_evidence,
        verify_trusted_runtime,
    )


class EvaluationError(ValueError):
    """The evaluator could not safely construct a result."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _final_answer_checks(
    final_answer: Any,
    artifact_schema: Mapping[str, Any],
    submission_schema: Mapping[str, Any],
) -> list[Issue]:
    if type(final_answer) is not dict:
        return [Issue("FINAL_SCHEMA", "final answer must be a JSON object")]
    issues = validate_json_schema(final_answer, artifact_schema)
    model_submission = final_answer.get("model_submission")
    issues.extend(validate_json_schema(model_submission, submission_schema, "$.model_submission"))
    diagnosis = _mapping(_mapping(model_submission).get("diagnosis"))
    evidence = diagnosis.get("evidence")
    if isinstance(evidence, list):
        for index, item in enumerate(evidence):
            if not isinstance(item, Mapping):
                continue
            start = item.get("line_start")
            end = item.get("line_end")
            if type(start) is int and type(end) is int and end < start:
                issues.append(
                    Issue("FINAL_SCHEMA", f"$.diagnosis.evidence[{index}] has a reversed line range")
                )
            path_value = item.get("path")
            if type(path_value) is str and (".." in path_value.replace("\\", "/").split("/") or path_value.startswith(("/", "\\"))):
                issues.append(
                    Issue("FINAL_SCHEMA", f"$.diagnosis.evidence[{index}] has an unsafe path")
                )
    return issues


def _tool_summary(trace: Any) -> dict[str, Any]:
    by_name: dict[str, dict[str, int]] = {}
    for fact in trace.tool_calls:
        entry = by_name.setdefault(fact.name, {"calls": 0, "successful": 0})
        entry["calls"] += 1
        entry["successful"] += int(fact.ok)
    return {
        "tool_call_count": len(trace.tool_calls),
        "successful_tool_call_count": trace.successful_tool_calls,
        "by_name": {name: by_name[name] for name in sorted(by_name)},
        "agent_turn_count": trace.agent_turns,
        "provider_retry_count": trace.provider_retries,
    }


def _trusted_evidence(trace: Any, artifacts: Any, evidence_valid: bool) -> dict[str, Any]:
    return {
        "trace": {
            "event_count": len(trace.events),
            "lifecycle_valid": trace.lifecycle_valid,
            "protocol_valid": trace.protocol_valid,
            "hash_chain_valid": trace.hash_chain_valid,
            "terminal_event": trace.terminal_event,
            "trace_sha256": trace.trace_sha256,
            "final_event_hash": trace.final_event_hash,
        },
        "run": {
            "run_type": trace.run_type,
            "execution_mode": trace.execution_mode,
            "native_mode": trace.native_mode,
            "logical_model": trace.logical_model,
            "requested_model": trace.requested_model,
            "actual_models": list(trace.actual_models),
            "provider_request_ids_present": len(trace.provider_request_ids),
            "exact_model_identity": trace.exact_model_identity,
            "identity_gate_exempt": trace.identity_exempt,
            "mode_truthful": trace.mode_truthful,
        },
        "tools": _tool_summary(trace),
        "process": {
            "attempt_prepared_count": len(trace.attempt_prepared),
            "native_execution_count": len(trace.native_executions),
            "attempt_validation_count": len(trace.attempt_validations),
            "attempt_event_contracts_valid": trace.attempt_event_contracts_valid,
            "native_event_contracts_valid": trace.native_event_contracts_valid,
            "validation_event_contracts_valid": trace.validation_event_contracts_valid,
            "budgets_respected": trace.budgets_respected,
            "budget_state_valid": trace.budget_state_valid,
            "decisive_log_evidence_read": trace.decisive_log_evidence_read,
            "final_evidence_verified": evidence_valid,
        },
        "artifacts": {
            "workspace_sha256": artifacts.workspace_sha256,
            "file_count": len(artifacts.files),
            "manifests_valid": artifacts.manifests_valid,
            "canonical_attempt_files_valid": artifacts.canonical_attempt_files_valid,
            "case_manifest_valid": artifacts.case_manifest_valid,
            "attempt_ids": sorted(artifacts.prepared_attempts),
            "attempt_isolation_valid": artifacts.attempt_isolation_valid,
            "provenance_valid": artifacts.provenance_valid,
            "actual_argv_safe": artifacts.actual_argv_safe,
            "native_xtb_executed": artifacts.native_xtb_executed,
            "optimization_converged": artifacts.optimization_converged,
            "hessian_completed": artifacts.hessian_completed,
            "final_artifacts_current": artifacts.final_artifacts_current,
            "final_xyz_valid": artifacts.final_xyz_valid,
            "scientific_invariants_preserved": artifacts.scientific_invariants_preserved,
            "legacy_artifact_present": artifacts.legacy_artifact_present,
            "legacy_used_as_current": artifacts.legacy_used_as_current,
            "initial_scheduler_completed": artifacts.initial_scheduler_completed,
            "initial_failure_max_cycles": artifacts.initial_failure_max_cycles,
        },
        "security": {
            "oracle_accessed": trace.oracle_accessed,
            "private_evaluator_material_present": artifacts.oracle_present,
            "unsafe_path_or_argv": trace.unsafe_path_or_argv or not artifacts.actual_argv_safe,
            "secret_finding": trace.secret_finding or artifacts.secret_finding,
            "absolute_path_finding": artifacts.absolute_path_finding,
            "filesystem_safe": artifacts.filesystem_safe,
            "hidden_reasoning_present": trace.hidden_reasoning_present,
            "fallback_or_manual_marker": trace.fallback_or_manual_marker,
        },
    }


def evaluate_files(
    *,
    final_answer_path: Path | str,
    trajectory_path: Path | str,
    artifacts_path: Path | str,
    config_path: Path | str,
) -> dict[str, Any]:
    """Evaluate one frozen run without executing a model, xTB, or network call."""

    final_path = Path(final_answer_path)
    trace_path = Path(trajectory_path)
    artifact_root = Path(artifacts_path)
    task_config_path = Path(config_path)

    trusted_config = load_trusted_configuration(task_config_path)
    final_value = load_json(final_path, label="final answer")
    final_answer = final_value if type(final_value) is dict else {}
    model_submission = _mapping(final_answer.get("model_submission"))
    final_issues = _final_answer_checks(
        final_value,
        trusted_config.final_answer_artifact_schema,
        trusted_config.submission_schema,
    )
    final_schema_valid = not final_issues

    trace = validate_trajectory(trace_path, trusted_config, model_submission)
    artifacts = validate_artifacts(artifact_root, trusted_config, trace, model_submission)
    trusted_runtime_valid, trusted_runtime_issues = verify_trusted_runtime(
        final_answer.get("trusted_runtime"),
        model_submission,
        trace,
        artifacts,
    )
    evidence_valid, evidence_issues = verify_final_evidence(
        model_submission, trace, artifact_root
    )
    score = compute_score(
        final_answer=model_submission,
        config=trusted_config,
        trace=trace,
        artifacts=artifacts,
        final_schema_valid=final_schema_valid,
        trusted_runtime_valid=trusted_runtime_valid,
        evidence_valid=evidence_valid,
    )
    issues = stable_issues(
        trusted_config.issues,
        final_issues,
        trusted_runtime_issues,
        trace.issues,
        artifacts.issues,
        evidence_issues,
    )
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "task_id": trusted_config.task_id,
        "case_id": trusted_config.case_id,
        "evaluation_status": "COMPLETED",
        "inputs": {
            "config_sha256": sha256_file(task_config_path),
            "final_answer_sha256": sha256_file(final_path),
            "trajectory_sha256": trace.trace_sha256,
            "artifact_workspace_sha256": artifacts.workspace_sha256,
        },
        "trusted_evidence": _trusted_evidence(trace, artifacts, evidence_valid),
        "scores": score["categories"],
        "raw_score_before_scientific_cap": score["raw_score_before_scientific_cap"],
        "scientific_core_correct": score["scientific_core_correct"],
        "scientific_core_cap_applied": score["scientific_core_cap_applied"],
        "ungated_score": score["ungated_score"],
        "eligibility": score["eligibility"],
        "official_score": score["official_score"],
        "maximum_score": score["maximum_score"],
        "issues": [item.as_dict() for item in issues],
    }


# A short public name is useful for import-based tests and integrations.
evaluate = evaluate_files


def evaluation_bytes(result: Mapping[str, Any], *, indent: int = 2) -> bytes:
    return canonical_json_bytes(result, newline=True, indent=indent)


def write_evaluation(path: Path | str, result: Mapping[str, Any], *, indent: int = 2) -> None:
    """Atomically write exactly one deterministic UTF-8/LF JSON document."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = evaluation_bytes(result, indent=indent)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _error_result(error_type: str) -> dict[str, Any]:
    """Return a deterministic non-reflective error document."""

    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "evaluation_status": "ERROR",
        "error": {"type": error_type},
        "ungated_score": 0,
        "raw_score_before_scientific_cap": 0,
        "scientific_core_correct": False,
        "scientific_core_cap_applied": False,
        "eligibility": {
            "eligible": False,
            "failed_gates": ["evaluation_completed"],
            "gates": {
                "evaluation_completed": {"passed": False, "status": "FAIL"}
            },
            "identity_gate_exempt": False,
        },
        "official_score": 0,
        "maximum_score": 100,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently evaluate one xTB failure-recovery trajectory."
    )
    parser.add_argument(
        "--final-answer",
        "--submission",
        dest="final_answer",
        required=True,
        help="Path to strict final_answer.json",
    )
    parser.add_argument("--trajectory", required=True, help="Path to trajectory.jsonl")
    parser.add_argument(
        "--artifacts",
        required=True,
        help="Path to the isolated benchmark_case artifact workspace",
    )
    parser.add_argument("--config", required=True, help="Path to configs/task.yaml")
    parser.add_argument("--output", required=True, help="Output evaluation.json path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_files(
            final_answer_path=args.final_answer,
            trajectory_path=args.trajectory,
            artifacts_path=args.artifacts,
            config_path=args.config,
        )
        indent = 2
        write_evaluation(args.output, result, indent=indent)
        return 0
    except (TrustedValidationError, ScoringConfigurationError, EvaluationError, OSError, ValueError) as exc:
        # Do not echo paths, provider payloads, or attacker-controlled values.
        result = _error_result(type(exc).__name__)
        try:
            write_evaluation(args.output, result, indent=2)
        except OSError:
            return 2
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

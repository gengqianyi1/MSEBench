"""Frozen 100-point rubric for the xTB recovery benchmark."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from .trusted_validation import (
    ArtifactFacts,
    REQUIRED_ELIGIBILITY_GATES,
    TraceFacts,
    TrustedConfiguration,
)


class ScoringConfigurationError(ValueError):
    """Raised when the trusted rubric no longer totals 100 points."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _component(points: int, passed: bool, *, awarded: int | None = None) -> dict[str, Any]:
    value = points if passed else 0
    if awarded is not None:
        value = awarded
    status = "PASS" if value == points else "PARTIAL" if value > 0 else "FAIL"
    return {"maximum": points, "awarded": value, "passed": value == points, "status": status}


def _event_sequence(trace: TraceFacts, event_name: str, predicate: Any = None) -> int | None:
    for event in trace.events:
        if event.get("event") != event_name:
            continue
        payload = event.get("payload")
        if predicate is None or predicate(payload if isinstance(payload, Mapping) else {}):
            sequence = event.get("sequence")
            return sequence if type(sequence) is int else None
    return None


def _successful_tool_sequence(trace: TraceFacts, name: str) -> int | None:
    values = [fact.result_sequence for fact in trace.tool_calls if fact.name == name and fact.ok and fact.result_sequence is not None]
    return min(values) if values else None


def _tool_budget_ok(trace: TraceFacts, config: TrustedConfiguration) -> bool:
    budget = config.budget
    maximum = budget.get("max_tool_calls")
    minimum = budget.get("min_successful_tool_calls")
    return (
        type(maximum) is int
        and type(minimum) is int
        and len(trace.tool_calls) <= maximum
        and trace.successful_tool_calls >= minimum
    )


def _attempt_budget_ok(trace: TraceFacts, artifacts: ArtifactFacts, config: TrustedConfiguration) -> bool:
    maximum = config.budget.get("max_execution_attempts")
    return (
        type(maximum) is int
        and len(trace.attempt_prepared) <= maximum
        and len(artifacts.prepared_attempts) <= maximum
    )


def _feedback_features(trace: TraceFacts, artifacts: ArtifactFacts) -> dict[str, bool]:
    read_sequences = [fact.result_sequence for fact in trace.direct_log_reads if fact.result_sequence is not None]
    first_read = min(read_sequences) if read_sequences else None
    first_prepare = _event_sequence(trace, "attempt_prepared")
    evidence_before_action = (
        first_read is not None and first_prepare is not None and first_read < first_prepare
    )

    opt_validation = _event_sequence(
        trace,
        "attempt_validated",
        lambda payload: payload.get("phase") == "geometry_optimization"
        and (
            payload.get("optimization_converged") is True
            or payload.get("valid") is True
        ),
    )
    hessian_prepare = _event_sequence(
        trace,
        "attempt_prepared",
        lambda payload: payload.get("phase") == "hessian",
    )
    optimization_feedback_before_hessian = (
        opt_validation is not None
        and hessian_prepare is not None
        and opt_validation < hessian_prepare
    )

    fingerprints: list[bytes] = []
    from .trusted_validation import canonical_json_bytes

    for attempt_id in sorted(artifacts.prepared_attempts):
        prepared = artifacts.prepared_attempts[attempt_id]
        fingerprint_material = {
            "phase": prepared.get("phase"),
            "source_geometry": prepared.get("source_geometry"),
            "request": prepared.get("request"),
            "argv": _mapping(prepared.get("execution")).get("argv"),
        }
        fingerprints.append(canonical_json_bytes(fingerprint_material))
    no_identical_retry = len(fingerprints) == len(set(fingerprints))
    feedback_consistent = (
        evidence_before_action
        and optimization_feedback_before_hessian
        and artifacts.targeted_cycle_repair
    )
    return {
        "evidence_before_action": evidence_before_action,
        "optimization_feedback_before_hessian": optimization_feedback_before_hessian,
        "no_identical_retry": no_identical_retry,
        "feedback_consistent": feedback_consistent,
    }


def _gate_record(passed: bool, *, exempt: bool = False) -> dict[str, Any]:
    return {
        "passed": bool(passed or exempt),
        "status": "EXEMPT" if exempt else ("PASS" if passed else "FAIL"),
    }


def build_eligibility(
    *,
    config: TrustedConfiguration,
    trace: TraceFacts,
    artifacts: ArtifactFacts,
    final_schema_valid: bool,
    trusted_runtime_valid: bool,
    evidence_valid: bool,
) -> dict[str, Any]:
    """Build all hard gates from independently derived facts."""

    values: dict[str, dict[str, Any]] = {
        "complete_lifecycle": _gate_record(
            trace.lifecycle_valid and trace.protocol_valid and trace.terminal_event == "run_complete"
        ),
        "formal_mode_truthful": _gate_record(trace.mode_truthful),
        "native_xtb_executed": _gate_record(
            trace.native_mode == "LIVE_XTB"
            and bool(trace.native_executions)
            and trace.native_event_contracts_valid
            and artifacts.native_xtb_executed
        ),
        "decisive_log_evidence_read": _gate_record(
            trace.decisive_log_evidence_read
            and evidence_valid
        ),
        "optimization_converged": _gate_record(
            trace.validation_event_contracts_valid and artifacts.optimization_converged
        ),
        "hessian_completed": _gate_record(
            trace.validation_event_contracts_valid and artifacts.hessian_completed
        ),
        "final_artifacts_current": _gate_record(
            artifacts.final_artifacts_current
            and artifacts.provenance_valid
            and artifacts.final_xyz_valid
            and trace.attempt_event_contracts_valid
        ),
        "legacy_not_used_as_current": _gate_record(not artifacts.legacy_used_as_current),
        "scientific_invariants_preserved": _gate_record(
            artifacts.scientific_invariants_preserved
        ),
        "oracle_not_accessed": _gate_record(
            not trace.oracle_accessed and not artifacts.oracle_present
        ),
        "path_and_argv_safe": _gate_record(
            not trace.unsafe_path_or_argv
            and artifacts.actual_argv_safe
            and not artifacts.absolute_path_finding
            and artifacts.filesystem_safe
        ),
        "budgets_respected": _gate_record(
            trace.budgets_respected
            and _tool_budget_ok(trace, config)
            and _attempt_budget_ok(trace, artifacts, config)
        ),
        "trace_hash_chain_valid": _gate_record(trace.hash_chain_valid),
        "final_answer_schema_valid": _gate_record(final_schema_valid),
        "trusted_runtime_verified": _gate_record(trusted_runtime_valid),
        "exact_model_identity_confirmed": _gate_record(
            trace.exact_model_identity,
            exempt=trace.identity_exempt,
        ),
        "no_fallback_or_manual_rewrite": _gate_record(
            trace.hash_chain_valid and not trace.fallback_or_manual_marker
        ),
        "no_secret_finding": _gate_record(
            not trace.secret_finding
            and not artifacts.secret_finding
            and not trace.hidden_reasoning_present
        ),
    }
    configured = config.config.get("scoring", {}).get("eligibility_gates", [])
    configured_set = set(configured) if isinstance(configured, list) else set()
    config_valid = not config.issues and configured_set == REQUIRED_ELIGIBILITY_GATES
    values["trusted_configuration_valid"] = _gate_record(config_valid)
    failed = sorted(name for name, record in values.items() if not record["passed"])
    return {
        "eligible": not failed,
        "failed_gates": failed,
        "gates": {name: values[name] for name in sorted(values)},
        "identity_gate_exempt": trace.identity_exempt,
    }


def compute_score(
    *,
    final_answer: Mapping[str, Any],
    config: TrustedConfiguration,
    trace: TraceFacts,
    artifacts: ArtifactFacts,
    final_schema_valid: bool,
    trusted_runtime_valid: bool,
    evidence_valid: bool,
) -> dict[str, Any]:
    """Apply the v2 science-weighted rubric using Decimal arithmetic."""

    categories = config.config.get("scoring", {}).get("categories", {})
    expected_categories = {
        "failure_diagnosis_and_evidence": 40,
        "restart_strategy_and_provenance": 20,
        "actual_task_recovery": 30,
        "compliance_and_efficiency": 10,
    }
    if categories != expected_categories:
        raise ScoringConfigurationError("configured scoring weights do not match the frozen 100-point rubric")

    diagnosis = _mapping(final_answer.get("diagnosis"))
    restart = _mapping(final_answer.get("restart_summary"))
    result = _mapping(final_answer.get("result_summary"))

    reference = _mapping(config.config.get("scoring", {}).get("diagnosis_reference"))
    category_correct = diagnosis.get("primary_category") == reference.get("primary_category")
    cause_correct = diagnosis.get("primary_cause_code") == reference.get("primary_cause_code")
    scheduler_correct = diagnosis.get("scheduler_assessment") == reference.get("scheduler_assessment")
    conflict_correct = diagnosis.get("conflict_resolution") == reference.get("conflict_resolution")
    legacy_correct = (
        diagnosis.get("legacy_disposition") == reference.get("legacy_disposition")
        and not artifacts.legacy_used_as_current
    )
    category_a_components = {
        "primary_failure_category": _component(8, category_correct),
        "primary_failure_cause": _component(12, cause_correct),
        "scheduler_vs_scientific_state": _component(5, scheduler_correct),
        "decisive_cited_log_evidence": _component(
            8, evidence_valid and trace.decisive_log_evidence_read
        ),
        "legacy_artifact_classification": _component(3, legacy_correct),
        "conflicting_evidence_handled": _component(4, conflict_correct),
    }

    source_kind = restart.get("starting_geometry_source_kind")
    declared_source = _mapping(restart.get("starting_geometry_source"))
    actual_prepared: Mapping[str, Any] | None = None
    actual_source: Mapping[str, Any] = {}
    for prepared in artifacts.prepared_attempts.values():
        source = _mapping(prepared.get("source_geometry"))
        if prepared.get("phase") == "geometry_optimization" and (
            source.get("path") == declared_source.get("path")
            and source.get("frame_index") == declared_source.get("frame_index")
        ):
            actual_prepared = prepared
            actual_source = source
            break
    declared_attempt = declared_source.get("attempt_id")
    source_path = declared_source.get("path")
    attempt_bound = (
        type(declared_attempt) is str
        and type(source_path) is str
        and declared_attempt in source_path.replace("\\", "/").split("/")
    )
    source_reference_matches = bool(
        actual_prepared is not None
        and attempt_bound
        and declared_source.get("file_sha256") == actual_source.get("file_sha256")
        and declared_source.get("frame_sha256") == actual_source.get("frame_sha256")
    )
    source_kind_matches = bool(
        (source_kind == "CURRENT_TRAJECTORY_FRAME" and artifacts.restart_from_last_current_frame)
        or (source_kind == "ORIGINAL_INPUT" and artifacts.restart_from_original_input)
        or source_kind == "CURRENT_RESTART_ARTIFACT"
    )
    source_validated_before_prepare = False
    if actual_prepared is not None:
        prepared_id = actual_prepared.get("attempt_id")
        prepare_sequence = _event_sequence(
            trace,
            "attempt_prepared",
            lambda payload: payload.get("attempt_id") == prepared_id,
        )
        for fact in trace.tool_calls:
            if fact.name != "validate_xyz" or not fact.ok or fact.result_sequence is None:
                continue
            if (
                fact.arguments.get("path") == actual_source.get("path")
                and fact.arguments.get("frame_index") == actual_source.get("frame_index")
                and prepare_sequence is not None
                and fact.result_sequence < prepare_sequence
            ):
                source_validated_before_prepare = True
                break
    category_b_components = {
        "restart_source_hash_bound": _component(6, source_reference_matches and source_kind_matches),
        "source_geometry_validation_bound": _component(5, source_validated_before_prepare),
        "targeted_cycle_limit_repair": _component(4, artifacts.targeted_cycle_repair),
        "scientific_invariants_preserved": _component(
            3, artifacts.scientific_invariants_preserved
        ),
        "attempt_isolation_and_provenance": _component(
            2, artifacts.attempt_isolation_valid and artifacts.provenance_valid
        ),
    }

    final_claim_consistent = (
        result.get("optimization_completed") == artifacts.optimization_converged
        and result.get("hessian_completed") == artifacts.hessian_completed
        and type(result.get("optimization_completed")) is bool
        and type(result.get("hessian_completed")) is bool
        and artifacts.final_xyz_valid
    )
    category_c_components = {
        "optimization_converged": _component(10, artifacts.optimization_converged),
        "hessian_completed": _component(10, artifacts.hessian_completed),
        "current_artifacts_and_provenance": _component(
            5, artifacts.final_artifacts_current and artifacts.provenance_valid
        ),
        "independent_final_validation": _component(5, final_claim_consistent),
    }

    tool_budget = _tool_budget_ok(trace, config)
    attempt_budget = _attempt_budget_ok(trace, artifacts, config)
    trace_consistent = (
        trace.hash_chain_valid
        and trace.lifecycle_valid
        and artifacts.manifests_valid
        and artifacts.provenance_valid
    )
    safe = (
        not trace.unsafe_path_or_argv
        and artifacts.actual_argv_safe
        and not artifacts.secret_finding
        and artifacts.filesystem_safe
        and not trace.secret_finding
        and not trace.oracle_accessed
        and not artifacts.oracle_present
    )
    category_e_components = {
        "tool_budget": _component(2, tool_budget),
        "attempt_budget": _component(2, attempt_budget),
        "strict_final_json": _component(2, final_schema_valid),
        "provenance_and_trace_consistency": _component(3, trace_consistent),
        "safe_and_relevant_operations": _component(1, safe),
    }

    component_groups = {
        "failure_diagnosis_and_evidence": category_a_components,
        "restart_strategy_and_provenance": category_b_components,
        "actual_task_recovery": category_c_components,
        "compliance_and_efficiency": category_e_components,
    }
    category_scores: dict[str, dict[str, Any]] = {}
    ungated = Decimal("0")
    for name, components in component_groups.items():
        maximum = sum(component["maximum"] for component in components.values())
        awarded = sum(component["awarded"] for component in components.values())
        if maximum != expected_categories[name]:
            raise ScoringConfigurationError(f"internal rubric mismatch for {name}")
        category_scores[name] = {
            "maximum": maximum,
            "score": awarded,
            "components": components,
        }
        ungated += Decimal(awarded)

    places = config.config.get("scoring", {}).get("rounding_decimal_places", 4)
    if type(places) is not int or places < 0 or places > 8:
        raise ScoringConfigurationError("rounding_decimal_places must be between 0 and 8")
    quantum = Decimal(1).scaleb(-places)
    raw_score = ungated.quantize(quantum, rounding=ROUND_HALF_UP)
    scientific_core_correct = bool(
        category_correct and cause_correct and scheduler_correct and conflict_correct
    )
    configured_cap = reference.get("scientific_core_cap_if_wrong", 20)
    cap = Decimal(str(configured_cap))
    ungated = raw_score if scientific_core_correct else min(raw_score, cap)
    eligibility = build_eligibility(
        config=config,
        trace=trace,
        artifacts=artifacts,
        final_schema_valid=final_schema_valid,
        trusted_runtime_valid=trusted_runtime_valid,
        evidence_valid=evidence_valid,
    )
    official = ungated if eligibility["eligible"] else Decimal("0").quantize(quantum)
    return {
        "categories": category_scores,
        "raw_score_before_scientific_cap": (
            int(raw_score) if raw_score == raw_score.to_integral_value() else float(raw_score)
        ),
        "scientific_core_correct": scientific_core_correct,
        "scientific_core_cap_applied": not scientific_core_correct and raw_score > cap,
        "ungated_score": int(ungated) if ungated == ungated.to_integral_value() else float(ungated),
        "eligibility": eligibility,
        "official_score": int(official) if official == official.to_integral_value() else float(official),
        "maximum_score": 100,
    }

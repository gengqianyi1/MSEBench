"""Deterministic score calculation for the constrained-QED benchmark.

This module deliberately contains no RDKit or file-system code. It accepts
values that have already been recomputed by the evaluator and applies the
weights in ``configs/task.yaml``.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from math import isfinite
from typing import Any, Mapping


SCORE_DECIMAL_PLACES = 4
_SCORE_QUANTUM = Decimal("1").scaleb(-SCORE_DECIMAL_PLACES)


class ScoringConfigurationError(ValueError):
    """Raised when the scoring section of a task config is unusable."""


def _decimal(value: Any, path: str, *, minimum: Decimal | None = None) -> Decimal:
    """Convert a real, finite config value to ``Decimal`` without float noise."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ScoringConfigurationError(f"{path} must be a real number")
    if isinstance(value, float) and not isfinite(value):
        raise ScoringConfigurationError(f"{path} must be finite")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ScoringConfigurationError(f"{path} must be a real number") from exc
    if not result.is_finite():
        raise ScoringConfigurationError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise ScoringConfigurationError(f"{path} must be >= {minimum}")
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScoringConfigurationError(f"{path} must be an object")
    return value


def _clamp(
    value: Decimal,
    low: Decimal = Decimal("0"),
    high: Decimal = Decimal("1"),
) -> Decimal:
    return max(low, min(high, value))


def _rounded(value: Decimal) -> float:
    return float(value.quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_UP))


def _observed_decimal(value: Any, path: str) -> Decimal:
    """Validate an evaluator-produced numeric observation."""

    try:
        return _decimal(value, path)
    except ScoringConfigurationError as exc:
        raise ValueError(str(exc)) from exc


def compute_score(
    *,
    final_qed: float,
    start_qed: float,
    unique_feasible_candidates: int,
    process_checks: Mapping[str, bool],
    hard_constraints_passed: bool,
    process_eligible: bool,
    config: Mapping[str, Any],
    objective_passed: bool = True,
    feedback_revision_quality: bool = False,
) -> dict[str, Any]:
    """Return a deterministic, config-driven score breakdown.

    The absolute-QED and improvement components are linearly interpolated and
    clipped. Exploration is based only on unique, evaluator-confirmed feasible
    history molecules. When a configured hard/process gate fails, the final
    score is zero, while ``ungated_score`` remains available for diagnostics.
    The objective gate is independent from the hard-constraint and process
    gates so a constraint-valid molecule cannot earn a positive total merely
    by tying or underperforming the configured starting molecule.
    """

    if isinstance(unique_feasible_candidates, bool) or not isinstance(
        unique_feasible_candidates, int
    ):
        raise ValueError("unique_feasible_candidates must be an integer")
    if unique_feasible_candidates < 0:
        raise ValueError("unique_feasible_candidates must be non-negative")
    if (
        type(hard_constraints_passed) is not bool
        or type(process_eligible) is not bool
        or type(objective_passed) is not bool
        or type(feedback_revision_quality) is not bool
    ):
        raise ValueError("eligibility flags must be booleans")

    scoring = _mapping(config.get("scoring"), "scoring")
    if scoring.get("version") == "qed_suite_v4":
        quality_cfg = _mapping(
            scoring.get("final_quality"), "scoring.final_quality"
        )
        revision_cfg = _mapping(
            scoring.get("feedback_revision"), "scoring.feedback_revision"
        )
        final = _observed_decimal(final_qed, "final_qed")
        start = _observed_decimal(start_qed, "start_qed")
        quality_points = _decimal(
            quality_cfg.get("points"),
            "scoring.final_quality.points",
            minimum=Decimal("0"),
        )
        quality_floor = _decimal(
            quality_cfg.get("floor_qed"), "scoring.final_quality.floor_qed"
        )
        quality_target = _decimal(
            quality_cfg.get("target_qed"), "scoring.final_quality.target_qed"
        )
        if quality_target <= quality_floor:
            raise ScoringConfigurationError(
                "scoring.final_quality.target_qed must exceed floor_qed"
            )
        quality_ratio = _clamp(
            (final - quality_floor) / (quality_target - quality_floor)
        )
        quality_score = quality_points * quality_ratio
        revision_points = _decimal(
            revision_cfg.get("points"),
            "scoring.feedback_revision.points",
            minimum=Decimal("0"),
        )
        revision_score = (
            revision_points if feedback_revision_quality else Decimal("0")
        )
        ungated = quality_score + revision_score
        hard_gate_enabled = scoring.get("hard_constraint_gate")
        process_gate_enabled = scoring.get("process_eligibility_gate")
        if type(hard_gate_enabled) is not bool or type(process_gate_enabled) is not bool:
            raise ScoringConfigurationError("scoring gate settings must be booleans")
        gate_passed = (
            objective_passed
            and (hard_constraints_passed or not hard_gate_enabled)
            and (process_eligible or not process_gate_enabled)
        )
        gated_total = ungated if gate_passed else Decimal("0")
        total = _clamp(gated_total, Decimal("0"), Decimal("90"))
        ungated_capped = _clamp(ungated, Decimal("0"), Decimal("90"))
        return {
            "score_schema": "qed_case_score/4.0",
            "components": {
                "final_chemical_quality": _rounded(quality_score),
                "feedback_revision": _rounded(revision_score),
            },
            "qed_delta": float(
                (final - start).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )
            ),
            "quality_ratio": _rounded(quality_ratio),
            "ungated_score": _rounded(ungated_capped),
            "objective_gate_passed": objective_passed,
            "gate_passed": gate_passed,
            "total_score": _rounded(total),
            "case_score_max": 90.0,
            "rounding_decimal_places": SCORE_DECIMAL_PLACES,
        }

    absolute_cfg = _mapping(scoring.get("qed_absolute"), "scoring.qed_absolute")
    improvement_cfg = _mapping(scoring.get("qed_improvement"), "scoring.qed_improvement")
    exploration_cfg = _mapping(scoring.get("exploration"), "scoring.exploration")
    process_cfg = _mapping(scoring.get("process"), "scoring.process")

    final = _observed_decimal(final_qed, "final_qed")
    start = _observed_decimal(start_qed, "start_qed")

    absolute_points = _decimal(
        absolute_cfg.get("points"),
        "scoring.qed_absolute.points",
        minimum=Decimal("0"),
    )
    absolute_floor = _decimal(absolute_cfg.get("floor"), "scoring.qed_absolute.floor")
    absolute_target = _decimal(absolute_cfg.get("target"), "scoring.qed_absolute.target")
    if absolute_target <= absolute_floor:
        raise ScoringConfigurationError("scoring.qed_absolute.target must exceed floor")
    absolute_ratio = _clamp((final - absolute_floor) / (absolute_target - absolute_floor))
    absolute_score = absolute_points * absolute_ratio

    improvement_points = _decimal(
        improvement_cfg.get("points"),
        "scoring.qed_improvement.points",
        minimum=Decimal("0"),
    )
    target_delta = _decimal(
        improvement_cfg.get("target_delta"),
        "scoring.qed_improvement.target_delta",
        minimum=Decimal("0"),
    )
    if target_delta == 0:
        raise ScoringConfigurationError("scoring.qed_improvement.target_delta must be > 0")
    qed_delta = final - start
    improvement_ratio = _clamp(qed_delta / target_delta)
    improvement_score = improvement_points * improvement_ratio

    exploration_points = _decimal(
        exploration_cfg.get("points"),
        "scoring.exploration.points",
        minimum=Decimal("0"),
    )
    full_credit_count = exploration_cfg.get(
        "unique_feasible_candidates_for_full_credit"
    )
    if (
        isinstance(full_credit_count, bool)
        or not isinstance(full_credit_count, int)
        or full_credit_count <= 0
    ):
        raise ScoringConfigurationError(
            "scoring.exploration.unique_feasible_candidates_for_full_credit "
            "must be a positive integer"
        )
    exploration_ratio = _clamp(
        Decimal(unique_feasible_candidates) / Decimal(full_credit_count)
    )
    exploration_score = exploration_points * exploration_ratio

    process_points_configured = _decimal(
        process_cfg.get("points"),
        "scoring.process.points",
        minimum=Decimal("0"),
    )
    process_components = (
        ("minimum_calls_met", "minimum_calls_points"),
        ("final_candidate_evaluated", "final_evaluated_points"),
        ("within_tool_budget", "within_budget_points"),
    )
    process_score = Decimal("0")
    for check_name, config_name in process_components:
        check_value = process_checks.get(check_name)
        if type(check_value) is not bool:
            raise ValueError(f"process_checks.{check_name} must be a boolean")
        component_points = _decimal(
            process_cfg.get(config_name),
            f"scoring.process.{config_name}",
            minimum=Decimal("0"),
        )
        if check_value:
            process_score += component_points
    # A malformed config must not silently award more process credit than its
    # declared category weight.
    process_score = min(process_score, process_points_configured)

    ungated = absolute_score + improvement_score + exploration_score + process_score
    hard_gate_enabled = scoring.get("hard_constraint_gate")
    process_gate_enabled = scoring.get("process_eligibility_gate")
    if type(hard_gate_enabled) is not bool or type(process_gate_enabled) is not bool:
        raise ScoringConfigurationError("scoring gate settings must be booleans")

    gate_passed = (
        objective_passed
        and (hard_constraints_passed or not hard_gate_enabled)
        and (process_eligible or not process_gate_enabled)
    )
    gated_total = ungated if gate_passed else Decimal("0")
    # The benchmark scale is always 0..100 even if a custom config contains
    # inconsistent category totals.
    total = _clamp(gated_total, Decimal("0"), Decimal("100"))
    ungated_capped = _clamp(ungated, Decimal("0"), Decimal("100"))

    return {
        "components": {
            "qed_absolute": _rounded(absolute_score),
            "qed_improvement": _rounded(improvement_score),
            "exploration": _rounded(exploration_score),
            "process": _rounded(process_score),
        },
        "qed_delta": float(
            qed_delta.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        ),
        "ungated_score": _rounded(ungated_capped),
        "objective_gate_passed": objective_passed,
        "gate_passed": gate_passed,
        "total_score": _rounded(total),
        "rounding_decimal_places": SCORE_DECIMAL_PLACES,
    }


# Compatibility alias for callers that use the noun form.
calculate_score = compute_score


__all__ = [
    "SCORE_DECIMAL_PLACES",
    "ScoringConfigurationError",
    "calculate_score",
    "compute_score",
]

"""Frozen v2 scoring with symmetry-aware extrema and barrier metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from evaluator.symmetry import evaluate_symmetry_aware_profile


SCORING_VERSION = "adaptive-torsion-scoring/2.0"
GRID = list(range(0, 360, 5))


class ScoringV2Error(ValueError):
    pass


def _finite_curve(values: Sequence[Any], label: str) -> np.ndarray:
    try:
        result = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ScoringV2Error(f"{label} curve is not numeric") from exc
    if result.shape != (72,) or not np.all(np.isfinite(result)):
        raise ScoringV2Error(f"{label} curve must have 72 finite values")
    return result - float(np.min(result))


def _linear(error: float, full: float, zero: float, points: float) -> float:
    if error <= full:
        return points
    if error >= zero:
        return 0.0
    return points * (zero - error) / (zero - full)


def _round(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == 0.0 else result


def score_profile_v2(
    *,
    predicted_energies_kcal_mol: Sequence[Any],
    oracle_energies_kcal_mol: Sequence[Any],
    facts: Mapping[str, Any],
    eligibility_gates: Mapping[str, bool],
) -> dict[str, Any]:
    predicted = _finite_curve(predicted_energies_kcal_mol, "predicted")
    oracle = _finite_curve(oracle_energies_kcal_mol, "oracle")
    delta = predicted - oracle
    rmse = float(np.sqrt(np.mean(delta**2)))
    maxae = float(np.max(np.abs(delta)))
    # Full-credit anchors are the worst-case errors of the frozen, legal
    # 12-angle reference optimum.  The zero anchors preserve a graded tail.
    rmse_points = _linear(rmse, 0.96, 2.00, 40.0)
    maxae_points = _linear(maxae, 4.23, 6.00, 15.0)
    curve_points = rmse_points + maxae_points

    symmetry = evaluate_symmetry_aware_profile(
        GRID,
        predicted.tolist(),
        GRID,
        oracle.tolist(),
    )
    basin_error = float(
        symmetry["global_minimum_basins"]["circular_error_deg"]
    )
    global_points = _linear(basin_error, 5.0, 30.0, 7.0)
    min_f1 = float(symmetry["local_extrema"]["minima"]["f1"])
    max_f1 = float(symmetry["local_extrema"]["maxima"]["f1"])
    extrema_points = 4.0 * min_f1 + 4.0 * max_f1
    barrier_mae = float(
        symmetry["adjacent_periodic_barriers"]["mae_kcal_mol"]
    )
    barrier_max = float(
        symmetry["adjacent_periodic_barriers"]["maximum_error_kcal_mol"]
    )
    barrier_mae_points = _linear(barrier_mae, 1.30, 3.00, 5.0)
    barrier_max_points = _linear(barrier_max, 2.60, 4.00, 5.0)
    scientific_points = global_points + extrema_points + barrier_mae_points + barrier_max_points

    adaptive_source = facts.get("adaptive_sampling", {})
    if not isinstance(adaptive_source, Mapping):
        adaptive_source = {}
    adaptive_fractions = {
        "periodic_coverage": float(adaptive_source.get("coverage_fraction", 0.0)),
        "feedback_use": float(adaptive_source.get("feedback_fraction", 0.0)),
        "refinement": float(adaptive_source.get("refinement_fraction", 0.0)),
        "call_efficiency": float(adaptive_source.get("efficiency_fraction", 0.0)),
    }
    adaptive_fractions = {
        key: min(1.0, max(0.0, value))
        for key, value in adaptive_fractions.items()
        if math.isfinite(value)
    }
    adaptive_points = 2.5 * sum(adaptive_fractions.values())

    process_source = facts.get("process_and_reproducibility", {})
    if not isinstance(process_source, Mapping):
        process_source = {}
    process_checks = {
        "trajectory_integrity": process_source.get("trajectory_integrity") is True,
        "artifact_provenance": process_source.get("artifact_provenance") is True,
        "exact_identity": process_source.get("exact_identity") is True,
        "scientific_invariants": process_source.get("scientific_invariants") is True,
        "replay_consistency": process_source.get("replay_consistency") is True,
    }
    process_points = 2.0 * sum(process_checks.values())
    ungated = min(
        100.0,
        max(
            0.0,
            curve_points
            + scientific_points
            + adaptive_points
            + process_points,
        ),
    )
    normalized_gates = {
        str(name): value is True for name, value in eligibility_gates.items()
    }
    failed = sorted(
        name for name, passed in normalized_gates.items() if not passed
    )
    eligible = bool(normalized_gates) and not failed
    official = ungated if eligible else 0.0
    return {
        "schema_version": "2.0",
        "scoring_version": SCORING_VERSION,
        "weights": {
            "curve_accuracy": 55,
            "extrema_and_barriers": 25,
            "adaptive_sampling": 10,
            "process_and_reproducibility": 10,
        },
        "categories": {
            "curve_accuracy": {
                "score": _round(curve_points),
                "max_score": 55.0,
                "rmse": {
                    "value_kcal_mol": _round(rmse),
                    "score": _round(rmse_points),
                    "max_score": 40.0,
                },
                "maxae": {
                    "value_kcal_mol": _round(maxae),
                    "score": _round(maxae_points),
                    "max_score": 15.0,
                },
            },
            "extrema_and_barriers": {
                "score": _round(scientific_points),
                "max_score": 25.0,
                "global_minimum_basins": {
                    "circular_error_deg": _round(basin_error),
                    "score": _round(global_points),
                    "max_score": 7.0,
                },
                "local_extrema": {
                    "minimum_f1": _round(min_f1),
                    "maximum_f1": _round(max_f1),
                    "score": _round(extrema_points),
                    "max_score": 8.0,
                },
                "symmetry_aware_barriers": {
                    "mae_kcal_mol": _round(barrier_mae),
                    "maximum_error_kcal_mol": _round(barrier_max),
                    "score": _round(barrier_mae_points + barrier_max_points),
                    "max_score": 10.0,
                },
                "full_diagnostics": symmetry,
            },
            "adaptive_sampling": {
                "score": _round(adaptive_points),
                "max_score": 10.0,
                "fractions": {
                    key: _round(value)
                    for key, value in adaptive_fractions.items()
                },
            },
            "process_and_reproducibility": {
                "score": _round(process_points),
                "max_score": 10.0,
                "checks": process_checks,
            },
        },
        "metrics": {
            "rmse_kcal_mol": _round(rmse),
            "maxae_kcal_mol": _round(maxae),
            "global_minimum_basin_error_deg": _round(basin_error),
            "local_minimum_f1": _round(min_f1),
            "local_maximum_f1": _round(max_f1),
            "barrier_mae_kcal_mol": _round(barrier_mae),
            "maximum_barrier_error_kcal_mol": _round(barrier_max),
        },
        "eligibility": {
            "eligible": eligible,
            "gates": normalized_gates,
            "failed_gates": failed,
        },
        "eligible": eligible,
        "ungated_total_score": _round(ungated),
        "total_score": _round(official),
        "official_score": _round(official),
    }


__all__ = ["SCORING_VERSION", "ScoringV2Error", "score_profile_v2"]

"""Independent NumPy-only scoring for the torsion-profile benchmark.

The scorer deliberately accepts only evaluator-reconstructed ``facts``, a
frozen 72-point agent ``profile``, and the hidden 72-point ``oracle``.  It has
no model-submission parameter and therefore cannot award points for model
self-reported counters, extrema, barriers, or eligibility claims.

All curves are independently shifted to a zero minimum before comparison.
Local extrema and directional barriers are recomputed from the frozen grid.
The caller supplies process/adaptive facts only after independently rebuilding
them from the trajectory and artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


SCORING_SCHEMA_VERSION = "1.0"
STANDARD_GRID_DEG = np.arange(0.0, 360.0, 5.0, dtype=np.float64)

CATEGORY_MAX_POINTS = {
    "curve_accuracy": 45.0,
    "extrema_and_barriers": 25.0,
    "adaptive_sampling": 15.0,
    "process_and_reproducibility": 15.0,
}

DEFAULT_THRESHOLDS = {
    # Full-credit anchors are calibrated to the frozen 12-angle reference
    # optimum.  This makes the top of the published 100-point scale reachable
    # under the same action/angle budget used for evaluated agents.
    "rmse_full_kcal_mol": 0.96,
    "rmse_zero_kcal_mol": 2.00,
    "maxae_full_kcal_mol": 4.23,
    "maxae_zero_kcal_mol": 6.00,
    "global_min_full_deg": 5.0,
    "global_min_zero_deg": 30.0,
    "local_extrema_match_tolerance_deg": 10.0,
    "barrier_full_error_kcal_mol": 2.60,
    "barrier_zero_error_kcal_mol": 4.00,
    "adaptive_rmse_gate_kcal_mol": 2.00,
    "adaptive_maxae_gate_kcal_mol": 6.00,
    "energy_equality_tolerance_kcal_mol": 1.0e-9,
}

_ANGLE_KEYS = (
    "grid_deg",
    "grid_angles_deg",
    "angles_deg",
    "angle_grid_deg",
    "angles",
)
_ENERGY_KEYS = (
    "relative_energies_kcal_mol",
    "energies_kcal_mol",
    "predicted_energy_kcal_mol",
    "predicted_relative_energies_kcal_mol",
    "predicted_energies_kcal_mol",
    "predicted_energies",
    "relative_energy_kcal_mol",
    "energy_kcal_mol",
    "values",
)
_CURVE_KEYS = ("curve", "predicted_curve", "oracle_curve", "points", "profile")
_POINT_ANGLE_KEYS = ("angle_deg", "angle", "grid_angle_deg")
_POINT_ENERGY_KEYS = (
    "relative_energy_kcal_mol",
    "energy_kcal_mol",
    "predicted_energy_kcal_mol",
    "value",
)


class ScoringInputError(ValueError):
    """The independent facts or curves are incomplete or malformed."""


@dataclass(frozen=True)
class _Curve:
    angles_deg: np.ndarray
    relative_energies_kcal_mol: np.ndarray


def _round(value: float) -> float:
    """Return a stable, JSON-safe score/metric value."""

    result = round(float(value), 12)
    return 0.0 if result == 0.0 else result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ScoringInputError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ScoringInputError(f"{label} must be a finite number")
    return number


def _thresholds(config: Mapping[str, Any] | None) -> dict[str, float]:
    values = dict(DEFAULT_THRESHOLDS)
    if config is not None:
        if not isinstance(config, Mapping):
            raise ScoringInputError("config must be a mapping")
        scoring = config.get("scoring")
        parent = scoring if isinstance(scoring, Mapping) else config
        source = parent.get("thresholds", parent)
        if not isinstance(source, Mapping):
            raise ScoringInputError("config.thresholds must be a mapping")
        for key in values:
            if key in source:
                values[key] = _finite_number(source[key], f"config.{key}")

    ordered_pairs = (
        ("rmse_full_kcal_mol", "rmse_zero_kcal_mol"),
        ("maxae_full_kcal_mol", "maxae_zero_kcal_mol"),
        ("global_min_full_deg", "global_min_zero_deg"),
        ("barrier_full_error_kcal_mol", "barrier_zero_error_kcal_mol"),
    )
    for full_key, zero_key in ordered_pairs:
        if values[full_key] < 0.0 or values[zero_key] <= values[full_key]:
            raise ScoringInputError(
                f"thresholds require 0 <= {full_key} < {zero_key}"
            )
    for key in (
        "local_extrema_match_tolerance_deg",
        "adaptive_rmse_gate_kcal_mol",
        "adaptive_maxae_gate_kcal_mol",
        "energy_equality_tolerance_kcal_mol",
    ):
        if values[key] < 0.0:
            raise ScoringInputError(f"config.{key} must be non-negative")
    if values["local_extrema_match_tolerance_deg"] > 180.0:
        raise ScoringInputError("local-extrema tolerance cannot exceed 180 degrees")
    return values


def linear_error_points(
    error: float,
    *,
    full_at_or_below: float,
    zero_at_or_above: float,
    max_points: float,
) -> float:
    """Piecewise-linear error score used by curve, minimum, and barrier terms."""

    error_value = _finite_number(error, "error")
    full = _finite_number(full_at_or_below, "full_at_or_below")
    zero = _finite_number(zero_at_or_above, "zero_at_or_above")
    points = _finite_number(max_points, "max_points")
    if error_value < 0.0 or full < 0.0 or zero <= full or points < 0.0:
        raise ScoringInputError("invalid linear scoring thresholds")
    if error_value <= full:
        return points
    if error_value >= zero:
        return 0.0
    return points * (zero - error_value) / (zero - full)


def circular_distance_deg(first: float, second: float) -> float:
    """Shortest unsigned angular distance in degrees, in ``[0, 180]``."""

    a = _finite_number(first, "first angle")
    b = _finite_number(second, "second angle")
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _angle_array(values: Sequence[Any], label: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScoringInputError(f"{label} must contain numeric angles") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ScoringInputError(f"{label} must be a finite one-dimensional array")
    return np.mod(result, 360.0)


def one_to_one_circular_matches(
    predicted_angles_deg: Sequence[Any],
    reference_angles_deg: Sequence[Any],
    *,
    tolerance_deg: float,
) -> list[dict[str, float | int]]:
    """Maximum-cardinality deterministic one-to-one circular matching.

    Each returned prediction and reference index occurs at most once.  The
    augmenting-path matcher maximizes the number of pairs within the frozen
    tolerance; adjacency and traversal ties are resolved by distance then
    canonical angle.
    """

    tolerance = _finite_number(tolerance_deg, "tolerance_deg")
    if not 0.0 <= tolerance <= 180.0:
        raise ScoringInputError("tolerance_deg must lie in [0, 180]")
    predicted = _angle_array(predicted_angles_deg, "predicted_angles_deg")
    reference = _angle_array(reference_angles_deg, "reference_angles_deg")
    if predicted.size == 0 or reference.size == 0:
        return []

    adjacency: list[list[tuple[int, float]]] = []
    for angle in predicted:
        edges = [
            (index, circular_distance_deg(float(angle), float(reference_angle)))
            for index, reference_angle in enumerate(reference)
        ]
        edges = [edge for edge in edges if edge[1] <= tolerance + 1.0e-12]
        edges.sort(key=lambda edge: (edge[1], float(reference[edge[0]]), edge[0]))
        adjacency.append(edges)

    # Constrained predictions are visited first.  Standard augmenting paths
    # still guarantee maximum cardinality; the ordering only freezes ties.
    order = sorted(
        range(len(predicted)),
        key=lambda index: (len(adjacency[index]), float(predicted[index]), index),
    )
    matched_reference: list[int | None] = [None] * len(reference)

    def augment(predicted_index: int, seen: set[int]) -> bool:
        for reference_index, _distance in adjacency[predicted_index]:
            if reference_index in seen:
                continue
            seen.add(reference_index)
            current = matched_reference[reference_index]
            if current is None or augment(current, seen):
                matched_reference[reference_index] = predicted_index
                return True
        return False

    for predicted_index in order:
        augment(predicted_index, set())

    pairs: list[dict[str, float | int]] = []
    for reference_index, predicted_index in enumerate(matched_reference):
        if predicted_index is None:
            continue
        pairs.append(
            {
                "predicted_index": predicted_index,
                "reference_index": reference_index,
                "predicted_angle_deg": _round(float(predicted[predicted_index])),
                "reference_angle_deg": _round(float(reference[reference_index])),
                "distance_deg": _round(
                    circular_distance_deg(
                        float(predicted[predicted_index]),
                        float(reference[reference_index]),
                    )
                ),
            }
        )
    pairs.sort(
        key=lambda item: (
            float(item["predicted_angle_deg"]),
            float(item["reference_angle_deg"]),
        )
    )
    return pairs


def _sequence_like(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _point_curve(values: Any, label: str) -> tuple[Any, Any] | None:
    if not _sequence_like(values):
        return None
    items = list(values)
    if not items or not all(isinstance(item, Mapping) for item in items):
        return None
    angles: list[Any] = []
    energies: list[Any] = []
    for index, item in enumerate(items):
        angle = next((item[key] for key in _POINT_ANGLE_KEYS if key in item), None)
        energy = next((item[key] for key in _POINT_ENERGY_KEYS if key in item), None)
        if angle is None or energy is None:
            raise ScoringInputError(f"{label}[{index}] lacks angle or energy")
        angles.append(angle)
        energies.append(energy)
    return angles, energies


def _curve_arrays(value: Any, label: str) -> tuple[Any | None, Any]:
    if isinstance(value, Mapping):
        angles = next((value[key] for key in _ANGLE_KEYS if key in value), None)
        energies = next((value[key] for key in _ENERGY_KEYS if key in value), None)
        if energies is not None:
            return angles, energies
        for key in _CURVE_KEYS:
            if key not in value:
                continue
            nested = value[key]
            points = _point_curve(nested, f"{label}.{key}")
            if points is not None:
                return points
            if _sequence_like(nested):
                return angles, nested
            if isinstance(nested, Mapping):
                return _curve_arrays(nested, f"{label}.{key}")
        raise ScoringInputError(f"{label} does not contain a recognized 72-point curve")
    points = _point_curve(value, label)
    if points is not None:
        return points
    if _sequence_like(value):
        return None, value
    raise ScoringInputError(f"{label} must be a curve mapping or numeric sequence")


def _load_curve(value: Any, label: str) -> _Curve:
    raw_angles, raw_energies = _curve_arrays(value, label)
    try:
        energies = np.asarray(raw_energies, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScoringInputError(f"{label} energies must be numeric") from exc
    if energies.ndim != 1 or energies.size != STANDARD_GRID_DEG.size:
        raise ScoringInputError(f"{label} must contain exactly 72 energy values")
    if not np.all(np.isfinite(energies)):
        raise ScoringInputError(f"{label} energies must all be finite")

    if raw_angles is None:
        angles = STANDARD_GRID_DEG.copy()
    else:
        angles = _angle_array(raw_angles, f"{label} angles")
        if angles.size != STANDARD_GRID_DEG.size:
            raise ScoringInputError(f"{label} must contain exactly 72 angles")
        order = np.argsort(angles, kind="stable")
        angles = angles[order]
        energies = energies[order]
        if not np.allclose(angles, STANDARD_GRID_DEG, rtol=0.0, atol=1.0e-9):
            raise ScoringInputError(
                f"{label} angles must be the standard 0,5,...,355 degree grid"
            )
    relative = energies - float(np.min(energies))
    if not np.all(np.isfinite(relative)):
        raise ScoringInputError(f"{label} normalization produced non-finite values")
    return _Curve(angles.astype(np.float64), relative.astype(np.float64))


def _extrema_groups(energies: np.ndarray, atol: float) -> tuple[list[int], list[int]]:
    count = int(energies.size)
    if count < 3 or np.allclose(energies, energies[0], rtol=0.0, atol=atol):
        return [], []
    boundaries = [
        index
        for index in range(count)
        if not math.isclose(
            float(energies[index]),
            float(energies[(index - 1) % count]),
            rel_tol=0.0,
            abs_tol=atol,
        )
    ]
    start = boundaries[0]
    order = [(start + offset) % count for offset in range(count)]
    groups: list[list[int]] = []
    position = 0
    while position < count:
        group = [order[position]]
        value = float(energies[order[position]])
        position += 1
        while position < count and math.isclose(
            float(energies[order[position]]), value, rel_tol=0.0, abs_tol=atol
        ):
            group.append(order[position])
            position += 1
        groups.append(group)

    minima: list[int] = []
    maxima: list[int] = []
    for group_index, group in enumerate(groups):
        value = float(energies[group[0]])
        previous = float(energies[groups[(group_index - 1) % len(groups)][0]])
        following = float(energies[groups[(group_index + 1) % len(groups)][0]])
        # The benchmark profile analysis freezes the smallest grid index as a
        # plateau representative.  The evaluator duplicates that rule rather
        # than importing the public tool implementation.
        representative = min(group)
        if value < previous - atol and value < following - atol:
            minima.append(representative)
        elif value > previous + atol and value > following + atol:
            maxima.append(representative)
    return sorted(minima), sorted(maxima)


def periodic_local_extrema(
    angles_deg: Sequence[Any],
    energies: Sequence[Any],
    *,
    equality_tolerance: float = 1.0e-9,
) -> dict[str, list[float]]:
    """Recompute deterministic local extrema on a circular discrete curve."""

    angles = _angle_array(angles_deg, "angles_deg")
    try:
        values = np.asarray(energies, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScoringInputError("energies must be numeric") from exc
    if values.ndim != 1 or values.size != angles.size or values.size < 3:
        raise ScoringInputError("angles and energies must be equal one-dimensional arrays")
    if not np.all(np.isfinite(values)):
        raise ScoringInputError("energies must all be finite")
    tolerance = _finite_number(equality_tolerance, "equality_tolerance")
    if tolerance < 0.0:
        raise ScoringInputError("equality_tolerance must be non-negative")
    minima, maxima = _extrema_groups(values, tolerance)
    return {
        "local_minima_deg": [_round(float(angles[index])) for index in minima],
        "local_maxima_deg": [_round(float(angles[index])) for index in maxima],
    }


def directional_barriers_from_curve(
    angles_deg: Sequence[Any],
    relative_energies_kcal_mol: Sequence[Any],
    *,
    equality_tolerance: float = 1.0e-9,
) -> dict[str, float]:
    """Compute forward/reverse barriers from a primary global minimum.

    The primary minimum is the lowest-angle local representative at the global
    energy.  Each direction ends at the nearest other local minimum.  If no
    other minimum exists, that direction traverses one complete period back to
    the primary minimum.  A barrier is the maximum energy on that inclusive
    path minus the primary-minimum energy.
    """

    angles = _angle_array(angles_deg, "angles_deg")
    try:
        energies = np.asarray(relative_energies_kcal_mol, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScoringInputError("relative energies must be numeric") from exc
    if energies.ndim != 1 or energies.size != angles.size or energies.size < 3:
        raise ScoringInputError("angles and relative energies must have equal length")
    if not np.all(np.isfinite(energies)):
        raise ScoringInputError("relative energies must all be finite")
    tolerance = _finite_number(equality_tolerance, "equality_tolerance")
    minima, _maxima = _extrema_groups(energies, tolerance)
    # np.argmin supplies the frozen smallest-index tie break used by profile
    # analysis.  Include it in the minima set even when it lies on a plateau
    # represented by another index.
    start = int(np.argmin(energies))

    count = int(energies.size)
    minima_for_paths = sorted(set([*minima, start]))
    others = [index for index in minima_for_paths if index != start]
    forward_steps = min(((index - start) % count for index in others), default=count)
    reverse_steps = min(((start - index) % count for index in others), default=count)
    forward_indices = [(start + step) % count for step in range(forward_steps + 1)]
    reverse_indices = [(start - step) % count for step in range(reverse_steps + 1)]
    forward_end = forward_indices[-1]
    reverse_end = reverse_indices[-1]
    return {
        "reference_minimum_angle_deg": _round(float(angles[start])),
        "forward_endpoint_angle_deg": _round(float(angles[forward_end])),
        "reverse_endpoint_angle_deg": _round(float(angles[reverse_end])),
        "forward_barrier_kcal_mol": _round(
            float(np.max(energies[forward_indices])) - float(energies[start])
        ),
        "reverse_barrier_kcal_mol": _round(
            float(np.max(energies[reverse_indices])) - float(energies[start])
        ),
    }


def _global_minimum_angles(curve: _Curve, tolerance: float) -> list[float]:
    minimum = float(np.min(curve.relative_energies_kcal_mol))
    indices = np.flatnonzero(
        np.isclose(
            curve.relative_energies_kcal_mol,
            minimum,
            rtol=0.0,
            atol=tolerance,
        )
    )
    return [float(curve.angles_deg[index]) for index in indices]


def _minimum_set_distance(first: Sequence[float], second: Sequence[float]) -> float:
    if not first or not second:
        return 180.0
    return min(circular_distance_deg(a, b) for a in first for b in second)


def _f1(match_count: int, predicted_count: int, reference_count: int) -> float:
    if predicted_count == reference_count == 0:
        return 1.0
    if match_count <= 0 or predicted_count <= 0 or reference_count <= 0:
        return 0.0
    precision = match_count / predicted_count
    recall = match_count / reference_count
    return 2.0 * precision * recall / (precision + recall)


def _fraction(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        number = float(value)
        if math.isfinite(number):
            return min(1.0, max(0.0, number))
    return None


def _explicit_fraction(section: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        if name in section:
            return _fraction(section[name])
    return None


def _evidence_fraction(section: Mapping[str, Any], names: Sequence[str]) -> float:
    values = [_fraction(section[name]) for name in names if name in section]
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else 0.0


def _section(facts: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = facts.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _adaptive_score(
    facts: Mapping[str, Any], *, curve_gate_passed: bool
) -> tuple[float, dict[str, Any]]:
    section = _section(facts, "adaptive_sampling", "adaptive")
    coverage = _explicit_fraction(
        section, ("coverage_fraction", "angular_coverage_fraction", "coverage")
    )
    information = _explicit_fraction(
        section,
        ("information_gain_fraction", "normalized_information_gain", "information_gain"),
    )
    refinement = _explicit_fraction(
        section,
        ("refinement_fraction", "high_value_refinement_fraction", "refinement"),
    )
    if refinement is None:
        refinement = _evidence_fraction(
            section,
            (
                "high_curvature_refinement_fraction",
                "model_disagreement_refinement_fraction",
                "uncertainty_refinement_fraction",
            ),
        )
    efficiency = _explicit_fraction(
        section, ("efficiency_fraction", "call_efficiency_fraction", "efficiency")
    )
    values = {
        "coverage": coverage if coverage is not None else 0.0,
        "information_gain": information if information is not None else 0.0,
        "refinement": refinement if refinement is not None else 0.0,
        "efficiency": efficiency if efficiency is not None else 0.0,
    }
    weights = {"coverage": 4.0, "information_gain": 4.0, "refinement": 4.0, "efficiency": 3.0}
    raw = sum(values[name] * weights[name] for name in weights)
    score = raw if curve_gate_passed else 0.0
    return score, {
        "score": _round(score),
        "max_score": 15.0,
        "raw_score_before_curve_gate": _round(raw),
        "curve_accuracy_gate_passed": curve_gate_passed,
        "components": {
            name: {
                "fraction": _round(values[name]),
                "score": _round(values[name] * weights[name]),
                "max_score": weights[name],
            }
            for name in weights
        },
    }


def _process_score(facts: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    section = _section(facts, "process_and_reproducibility", "process")
    definitions = {
        "trajectory_and_tool_integrity": (
            ("trajectory_and_tool_integrity_fraction",),
            ("trajectory_hash_chain_valid", "tool_pairing_valid", "closed_lifecycle_valid"),
        ),
        "native_attempt_provenance": (
            ("native_attempt_provenance_fraction", "attempt_provenance_fraction"),
            ("attempt_provenance_valid", "native_attempts_verified", "profile_points_native_backed"),
        ),
        "budget_and_protocol": (
            ("budget_and_protocol_fraction",),
            ("budgets_respected", "feedback_rounds_valid", "profile_fit_rounds_valid", "scientific_invariants_preserved"),
        ),
        "independent_recomputation": (
            ("independent_recomputation_fraction",),
            ("profile_recomputed", "extrema_recomputed", "trusted_runtime_verified"),
        ),
        "replay_consistency": (
            ("replay_consistency_fraction",),
            ("live_replay_consistent", "replay_consistent", "deterministic_replay"),
        ),
    }
    fractions: dict[str, float] = {}
    for name, (explicit_names, evidence_names) in definitions.items():
        explicit = _explicit_fraction(section, explicit_names)
        fractions[name] = (
            explicit if explicit is not None else _evidence_fraction(section, evidence_names)
        )
    score = sum(value * 3.0 for value in fractions.values())
    return score, {
        "score": _round(score),
        "max_score": 15.0,
        "components": {
            name: {
                "fraction": _round(value),
                "score": _round(value * 3.0),
                "max_score": 3.0,
            }
            for name, value in fractions.items()
        },
    }


def _gate_passed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return value.get("passed") is True
    return value is True


def _eligibility(
    facts: Mapping[str, Any], explicit_gates: Mapping[str, Any] | None
) -> dict[str, Any]:
    gates: Any = explicit_gates
    if gates is None:
        gates = facts.get("eligibility_gates")
    if gates is None and isinstance(facts.get("eligibility"), Mapping):
        gates = facts["eligibility"].get("gates")
    if not isinstance(gates, Mapping) or not gates:
        return {
            "eligible": False,
            "gates_present": False,
            "gates": {},
            "failed_gates": ["eligibility_gates_missing"],
        }
    normalized = {str(name): _gate_passed(value) for name, value in gates.items()}
    failed = sorted(name for name, passed in normalized.items() if not passed)
    return {
        "eligible": not failed,
        "gates_present": True,
        "gates": {name: normalized[name] for name in sorted(normalized)},
        "failed_gates": failed,
    }


def compute_score(
    *,
    facts: Mapping[str, Any],
    profile: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    oracle: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    eligibility_gates: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the independent 100-point score and apply hard eligibility gates.

    ``facts`` must be evaluator-reconstructed trajectory/artifact facts.  The
    model submission is intentionally absent from this API.  An empty or
    missing gate map fails closed: the ungated diagnostic score is retained,
    while the official ``total_score`` becomes zero.
    """

    if not isinstance(facts, Mapping):
        raise ScoringInputError("facts must be a mapping")
    thresholds = _thresholds(config)
    predicted = _load_curve(profile, "profile")
    reference = _load_curve(oracle, "oracle")
    errors = predicted.relative_energies_kcal_mol - reference.relative_energies_kcal_mol
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    maxae = float(np.max(np.abs(errors)))
    rmse_points = linear_error_points(
        rmse,
        full_at_or_below=thresholds["rmse_full_kcal_mol"],
        zero_at_or_above=thresholds["rmse_zero_kcal_mol"],
        max_points=35.0,
    )
    maxae_points = linear_error_points(
        maxae,
        full_at_or_below=thresholds["maxae_full_kcal_mol"],
        zero_at_or_above=thresholds["maxae_zero_kcal_mol"],
        max_points=10.0,
    )
    curve_score = rmse_points + maxae_points
    curve_category = {
        "score": _round(curve_score),
        "max_score": 45.0,
        "rmse": {
            "value_kcal_mol": _round(rmse),
            "score": _round(rmse_points),
            "max_score": 35.0,
        },
        "maxae": {
            "value_kcal_mol": _round(maxae),
            "score": _round(maxae_points),
            "max_score": 10.0,
        },
    }

    equality_tolerance = thresholds["energy_equality_tolerance_kcal_mol"]
    predicted_global = _global_minimum_angles(predicted, equality_tolerance)
    reference_global = _global_minimum_angles(reference, equality_tolerance)
    global_error = _minimum_set_distance(predicted_global, reference_global)
    global_points = linear_error_points(
        global_error,
        full_at_or_below=thresholds["global_min_full_deg"],
        zero_at_or_above=thresholds["global_min_zero_deg"],
        max_points=7.0,
    )

    predicted_extrema = periodic_local_extrema(
        predicted.angles_deg,
        predicted.relative_energies_kcal_mol,
        equality_tolerance=equality_tolerance,
    )
    reference_extrema = periodic_local_extrema(
        reference.angles_deg,
        reference.relative_energies_kcal_mol,
        equality_tolerance=equality_tolerance,
    )

    def nonglobal(values: Sequence[float], globals_: Sequence[float]) -> list[float]:
        return [
            value
            for value in values
            if _minimum_set_distance([value], globals_) > 1.0e-8
        ]

    predicted_minima = nonglobal(
        predicted_extrema["local_minima_deg"], predicted_global
    )
    reference_minima = nonglobal(
        reference_extrema["local_minima_deg"], reference_global
    )
    predicted_maxima = predicted_extrema["local_maxima_deg"]
    reference_maxima = reference_extrema["local_maxima_deg"]
    match_tolerance = thresholds["local_extrema_match_tolerance_deg"]
    minima_matches = one_to_one_circular_matches(
        predicted_minima, reference_minima, tolerance_deg=match_tolerance
    )
    maxima_matches = one_to_one_circular_matches(
        predicted_maxima, reference_maxima, tolerance_deg=match_tolerance
    )
    minima_f1 = _f1(len(minima_matches), len(predicted_minima), len(reference_minima))
    maxima_f1 = _f1(len(maxima_matches), len(predicted_maxima), len(reference_maxima))
    minima_points = 4.0 * minima_f1
    maxima_points = 4.0 * maxima_f1
    extrema_points = minima_points + maxima_points

    predicted_barriers = directional_barriers_from_curve(
        predicted.angles_deg,
        predicted.relative_energies_kcal_mol,
        equality_tolerance=equality_tolerance,
    )
    reference_barriers = directional_barriers_from_curve(
        reference.angles_deg,
        reference.relative_energies_kcal_mol,
        equality_tolerance=equality_tolerance,
    )
    forward_error = abs(
        predicted_barriers["forward_barrier_kcal_mol"]
        - reference_barriers["forward_barrier_kcal_mol"]
    )
    reverse_error = abs(
        predicted_barriers["reverse_barrier_kcal_mol"]
        - reference_barriers["reverse_barrier_kcal_mol"]
    )
    forward_points = linear_error_points(
        forward_error,
        full_at_or_below=thresholds["barrier_full_error_kcal_mol"],
        zero_at_or_above=thresholds["barrier_zero_error_kcal_mol"],
        max_points=5.0,
    )
    reverse_points = linear_error_points(
        reverse_error,
        full_at_or_below=thresholds["barrier_full_error_kcal_mol"],
        zero_at_or_above=thresholds["barrier_zero_error_kcal_mol"],
        max_points=5.0,
    )
    barrier_points = forward_points + reverse_points
    extrema_category = {
        "score": _round(global_points + extrema_points + barrier_points),
        "max_score": 25.0,
        "global_minimum": {
            "predicted_angles_deg": [_round(value) for value in predicted_global],
            "oracle_angles_deg": [_round(value) for value in reference_global],
            "circular_error_deg": _round(global_error),
            "score": _round(global_points),
            "max_score": 7.0,
        },
        "local_extrema": {
            "score": _round(extrema_points),
            "max_score": 8.0,
            "match_tolerance_deg": _round(match_tolerance),
            "minima": {
                "predicted_angles_deg": [_round(value) for value in predicted_minima],
                "oracle_angles_deg": [_round(value) for value in reference_minima],
                "matches": minima_matches,
                "f1": _round(minima_f1),
                "score": _round(minima_points),
                "max_score": 4.0,
            },
            "maxima": {
                "predicted_angles_deg": [_round(value) for value in predicted_maxima],
                "oracle_angles_deg": [_round(value) for value in reference_maxima],
                "matches": maxima_matches,
                "f1": _round(maxima_f1),
                "score": _round(maxima_points),
                "max_score": 4.0,
            },
        },
        "barriers": {
            "score": _round(barrier_points),
            "max_score": 10.0,
            "predicted": predicted_barriers,
            "oracle": reference_barriers,
            "forward": {
                "absolute_error_kcal_mol": _round(forward_error),
                "score": _round(forward_points),
                "max_score": 5.0,
            },
            "reverse": {
                "absolute_error_kcal_mol": _round(reverse_error),
                "score": _round(reverse_points),
                "max_score": 5.0,
            },
        },
    }

    curve_gate_passed = bool(
        rmse < thresholds["adaptive_rmse_gate_kcal_mol"]
        and maxae < thresholds["adaptive_maxae_gate_kcal_mol"]
    )
    adaptive_score, adaptive_category = _adaptive_score(
        facts, curve_gate_passed=curve_gate_passed
    )
    process_score, process_category = _process_score(facts)
    categories = {
        "curve_accuracy": curve_category,
        "extrema_and_barriers": extrema_category,
        "adaptive_sampling": adaptive_category,
        "process_and_reproducibility": process_category,
    }
    ungated = (
        curve_score
        + global_points
        + extrema_points
        + barrier_points
        + adaptive_score
        + process_score
    )
    ungated = min(100.0, max(0.0, ungated))
    eligibility = _eligibility(facts, eligibility_gates)
    official = ungated if eligibility["eligible"] else 0.0
    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "categories": categories,
        "metrics": {
            "rmse_kcal_mol": _round(rmse),
            "maxae_kcal_mol": _round(maxae),
            "global_minimum_circular_error_deg": _round(global_error),
            "forward_barrier_error_kcal_mol": _round(forward_error),
            "reverse_barrier_error_kcal_mol": _round(reverse_error),
        },
        "thresholds": {key: _round(value) for key, value in thresholds.items()},
        "eligibility": eligibility,
        "eligible": eligibility["eligible"],
        "ungated_score": _round(ungated),
        "ungated_total_score": _round(ungated),
        "total_score": _round(official),
        "official_score": _round(official),
    }


__all__ = [
    "CATEGORY_MAX_POINTS",
    "DEFAULT_THRESHOLDS",
    "SCORING_SCHEMA_VERSION",
    "STANDARD_GRID_DEG",
    "ScoringInputError",
    "circular_distance_deg",
    "compute_score",
    "directional_barriers_from_curve",
    "linear_error_points",
    "one_to_one_circular_matches",
    "periodic_local_extrema",
]

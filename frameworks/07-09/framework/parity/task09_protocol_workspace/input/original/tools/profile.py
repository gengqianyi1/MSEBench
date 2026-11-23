"""Deterministic, NumPy-only periodic torsion-profile fitting and analysis.

The public fitting functions deliberately know nothing about native attempts or
the hidden oracle.  Callers must supply energies from already-validated points.
All sample angles are restricted to the benchmark's frozen 5 degree grid in
``[0, 360)`` and every returned payload is JSON serializable.

Barrier definition
------------------
The global minimum (smallest angle breaks an exact tie) is the anchor well.
The forward barrier follows increasing angle to the next discrete local
minimum; the reverse barrier follows decreasing angle to the previous local
minimum.  In each direction the barrier is the largest grid energy on that
closed path minus the anchor energy.  If there is no other local minimum, the
path is one complete revolution.  This definition is invariant to an additive
energy shift and gives an unambiguous direction on the periodic grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


GRID_STEP_DEG = 5
PERIOD_DEG = 360
PROFILE_GRID_DEG = tuple(range(0, PERIOD_DEG, GRID_STEP_DEG))
FOURIER_MIN_ORDER = 1
FOURIER_MAX_ORDER = 6


class ProfileFitError(ValueError):
    """Input data cannot define a valid frozen periodic profile."""


def _clean_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProfileFitError("profile output is non-finite")
    return 0.0 if abs(result) < 1.0e-15 else result


def _float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ProfileFitError("profile values must be a finite one-dimensional sequence")
    return [_clean_float(value) for value in array]


def _sample_angle(value: Any) -> int:
    if isinstance(value, (bool, str, bytes)):
        raise ProfileFitError("sample angles must be numeric integers")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileFitError("sample angles must be numeric integers") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ProfileFitError("sample angles must be finite integers")
    angle = int(number)
    if angle < 0 or angle >= PERIOD_DEG or angle % GRID_STEP_DEG:
        raise ProfileFitError("sample angles must be 5 degree multiples in [0, 360)")
    return angle


def _energy(value: Any) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ProfileFitError("energies must be finite numeric values")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileFitError("energies must be finite numeric values") from exc
    if not math.isfinite(number):
        raise ProfileFitError("energies must be finite numeric values")
    return number


def _coerce_samples(
    angles_deg: Sequence[float] | Mapping[float, float],
    energies: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(angles_deg, Mapping):
        if energies is not None:
            raise ProfileFitError("energies must be omitted when samples are a mapping")
        pairs = list(angles_deg.items())
    else:
        if energies is None:
            raise ProfileFitError("energies are required for an angle sequence")
        try:
            angle_values = list(angles_deg)
            energy_values = list(energies)
        except TypeError as exc:
            raise ProfileFitError("angles and energies must be sequences") from exc
        if len(angle_values) != len(energy_values):
            raise ProfileFitError("angles and energies must have equal lengths")
        pairs = list(zip(angle_values, energy_values))

    if not pairs:
        raise ProfileFitError("at least one sample is required")
    parsed = [(_sample_angle(angle), _energy(value)) for angle, value in pairs]
    parsed.sort(key=lambda pair: pair[0])
    sorted_angles = [pair[0] for pair in parsed]
    if len(set(sorted_angles)) != len(sorted_angles):
        raise ProfileFitError("sample angles must be unique")
    return (
        np.asarray(sorted_angles, dtype=float),
        np.asarray([pair[1] for pair in parsed], dtype=float),
    )


def _coerce_queries(query_angles_deg: Sequence[float]) -> np.ndarray:
    try:
        queries = np.asarray(list(query_angles_deg), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileFitError("query angles must be numeric") from exc
    if queries.ndim != 1 or queries.size == 0 or not np.all(np.isfinite(queries)):
        raise ProfileFitError("query angles must be a non-empty finite sequence")
    return queries


@dataclass(frozen=True)
class _PeriodicCubicModel:
    angles: np.ndarray
    energies: np.ndarray
    second_derivatives: np.ndarray
    matrix_condition: float

    def evaluate(self, query_angles_deg: Sequence[float], derivative: int = 0) -> np.ndarray:
        if derivative not in (0, 1, 2):
            raise ProfileFitError("periodic cubic derivative must be 0, 1, or 2")
        query = _coerce_queries(query_angles_deg)
        first = float(self.angles[0])
        wrapped = np.mod(query - first, PERIOD_DEG) + first
        indices = np.searchsorted(self.angles, wrapped, side="right") - 1
        indices = np.where(indices < 0, self.angles.size - 1, indices)

        output = np.empty(query.size, dtype=float)
        count = self.angles.size
        for position, (value, raw_index) in enumerate(zip(wrapped, indices)):
            index = int(raw_index)
            following = (index + 1) % count
            left = float(self.angles[index])
            right = (
                float(self.angles[following])
                if following != 0
                else float(self.angles[0]) + PERIOD_DEG
            )
            if value < left:
                value += PERIOD_DEG
            width = right - left
            a = (right - value) / width
            b = (value - left) / width
            left_y = float(self.energies[index])
            right_y = float(self.energies[following])
            left_m = float(self.second_derivatives[index])
            right_m = float(self.second_derivatives[following])
            if derivative == 0:
                result = (
                    a * left_y
                    + b * right_y
                    + ((a**3 - a) * left_m + (b**3 - b) * right_m)
                    * width**2
                    / 6.0
                )
            elif derivative == 1:
                result = (
                    (right_y - left_y) / width
                    + width
                    * ((-3.0 * a**2 + 1.0) * left_m + (3.0 * b**2 - 1.0) * right_m)
                    / 6.0
                )
            else:
                result = a * left_m + b * right_m
            output[position] = result
        if not np.all(np.isfinite(output)):
            raise ProfileFitError("periodic cubic prediction is non-finite")
        return output


def _build_periodic_cubic_model(
    angles_deg: Sequence[float] | Mapping[float, float],
    energies: Sequence[float] | None,
) -> _PeriodicCubicModel:
    angles, values = _coerce_samples(angles_deg, energies)
    count = angles.size
    if count < 3:
        raise ProfileFitError("periodic cubic interpolation requires at least 3 unique samples")

    widths = np.empty(count, dtype=float)
    for index in range(count):
        following = (index + 1) % count
        right = angles[following] if following else angles[0] + PERIOD_DEG
        widths[index] = right - angles[index]
    if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
        raise ProfileFitError("periodic sample intervals are invalid")

    matrix = np.zeros((count, count), dtype=float)
    rhs = np.zeros(count, dtype=float)
    for index in range(count):
        previous = (index - 1) % count
        following = (index + 1) % count
        left_width = widths[previous]
        right_width = widths[index]
        matrix[index, previous] += left_width
        matrix[index, index] += 2.0 * (left_width + right_width)
        matrix[index, following] += right_width
        rhs[index] = 6.0 * (
            (values[following] - values[index]) / right_width
            - (values[index] - values[previous]) / left_width
        )
    try:
        condition = float(np.linalg.cond(matrix))
        second = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ProfileFitError("periodic cubic system is singular") from exc
    if not math.isfinite(condition) or not np.all(np.isfinite(second)):
        raise ProfileFitError("periodic cubic system is non-finite")
    return _PeriodicCubicModel(angles, values, second, condition)


def periodic_cubic_predict(
    angles_deg: Sequence[float] | Mapping[float, float],
    energies: Sequence[float] | None = None,
    query_angles_deg: Sequence[float] = PROFILE_GRID_DEG,
) -> np.ndarray:
    """Evaluate the unique C2 periodic cubic interpolant on arbitrary angles."""

    return _build_periodic_cubic_model(angles_deg, energies).evaluate(query_angles_deg)


def _validate_fourier_order(order: Any) -> int:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ProfileFitError("Fourier order must be an integer from 1 through 6")
    value = int(order)
    if value < FOURIER_MIN_ORDER or value > FOURIER_MAX_ORDER:
        raise ProfileFitError("Fourier order must be an integer from 1 through 6")
    return value


def _fourier_design(angles_deg: np.ndarray, order: int) -> np.ndarray:
    theta = np.deg2rad(np.mod(angles_deg, PERIOD_DEG))
    columns = [np.ones(theta.size, dtype=float)]
    for harmonic in range(1, order + 1):
        columns.append(np.cos(harmonic * theta))
        columns.append(np.sin(harmonic * theta))
    return np.column_stack(columns)


@dataclass(frozen=True)
class _FourierModel:
    angles: np.ndarray
    energies: np.ndarray
    order: int
    coefficients: np.ndarray
    rank: int
    matrix_condition: float

    def evaluate(self, query_angles_deg: Sequence[float]) -> np.ndarray:
        query = _coerce_queries(query_angles_deg)
        output = _fourier_design(query, self.order) @ self.coefficients
        if not np.all(np.isfinite(output)):
            raise ProfileFitError("Fourier prediction is non-finite")
        return output


def _build_fourier_model(
    angles_deg: Sequence[float] | Mapping[float, float],
    energies: Sequence[float] | None,
    order: Any,
) -> _FourierModel:
    harmonic_order = _validate_fourier_order(order)
    angles, values = _coerce_samples(angles_deg, energies)
    parameter_count = 1 + 2 * harmonic_order
    if angles.size < parameter_count:
        raise ProfileFitError(
            f"Fourier order {harmonic_order} requires at least {parameter_count} unique samples"
        )
    design = _fourier_design(angles, harmonic_order)
    rank = int(np.linalg.matrix_rank(design))
    if rank != parameter_count:
        raise ProfileFitError(
            f"Fourier design is rank deficient ({rank} < {parameter_count})"
        )
    try:
        condition = float(np.linalg.cond(design))
        coefficients, _, solved_rank, _ = np.linalg.lstsq(design, values, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ProfileFitError("Fourier least-squares solution failed") from exc
    if int(solved_rank) != parameter_count:
        raise ProfileFitError("Fourier least-squares solution is underdetermined")
    if not math.isfinite(condition) or not np.all(np.isfinite(coefficients)):
        raise ProfileFitError("Fourier fit is non-finite")
    return _FourierModel(angles, values, harmonic_order, coefficients, rank, condition)


def fourier_predict(
    angles_deg: Sequence[float] | Mapping[float, float],
    energies: Sequence[float] | None = None,
    *,
    order: int,
    query_angles_deg: Sequence[float] = PROFILE_GRID_DEG,
) -> np.ndarray:
    """Fit a determined Fourier model and evaluate it on arbitrary angles."""

    return _build_fourier_model(angles_deg, energies, order).evaluate(query_angles_deg)


def largest_circular_gaps(
    sampled_angles_deg: Sequence[float], *, limit: int = 4
) -> list[dict[str, Any]]:
    """Return the widest forward circular arcs between sampled grid angles."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ProfileFitError("signal limit must be a positive integer")
    parsed = sorted(_sample_angle(value) for value in sampled_angles_deg)
    if not parsed:
        return []
    if len(set(parsed)) != len(parsed):
        raise ProfileFitError("sample angles must be unique")
    intervals: list[dict[str, Any]] = []
    for index, start in enumerate(parsed):
        end = parsed[(index + 1) % len(parsed)]
        width = (end - start) % PERIOD_DEG
        if len(parsed) == 1:
            width = PERIOD_DEG
        steps = int(round(width / GRID_STEP_DEG))
        internal = [
            (start + step * GRID_STEP_DEG) % PERIOD_DEG
            for step in range(1, steps)
        ]
        midpoint = (start + width / 2.0) % PERIOD_DEG
        recommendation = None
        if internal:
            recommendation = min(
                internal,
                key=lambda angle: (
                    abs(((angle - start) % PERIOD_DEG) - width / 2.0),
                    angle,
                ),
            )
        intervals.append(
            {
                "start_angle_deg": start,
                "end_angle_deg": end,
                "width_deg": int(width),
                "midpoint_angle_deg": _clean_float(midpoint),
                "recommended_angle_deg": recommendation,
            }
        )
    intervals.sort(key=lambda item: (-item["width_deg"], item["start_angle_deg"]))
    return intervals[:limit]


def _canonical_grid(grid_angles_deg: Sequence[float] | None) -> np.ndarray:
    if grid_angles_deg is None:
        return np.asarray(PROFILE_GRID_DEG, dtype=float)
    try:
        grid = np.asarray(list(grid_angles_deg), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileFitError("profile grid must be numeric") from exc
    expected = np.asarray(PROFILE_GRID_DEG, dtype=float)
    if grid.shape != expected.shape or not np.array_equal(grid, expected):
        raise ProfileFitError("analysis requires the canonical 72-point 5 degree grid")
    return grid


def _profile_values(values: Sequence[float], expected_size: int = 72) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileFitError("profile energies must be numeric") from exc
    if array.ndim != 1 or array.size != expected_size or not np.all(np.isfinite(array)):
        raise ProfileFitError(f"profile must contain {expected_size} finite energies")
    return array


def _select_diverse_indices(scores: np.ndarray, limit: int) -> list[int]:
    order = sorted(range(scores.size), key=lambda index: (-float(scores[index]), index))
    chosen: list[int] = []
    for index in order:
        if all(min((index - other) % scores.size, (other - index) % scores.size) > 1 for other in chosen):
            chosen.append(index)
        if len(chosen) == limit:
            break
    return chosen


def high_curvature_intervals(
    profile_energies: Sequence[float],
    *,
    grid_angles_deg: Sequence[float] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Rank non-adjacent grid neighborhoods by periodic second difference."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ProfileFitError("signal limit must be a positive integer")
    grid = _canonical_grid(grid_angles_deg)
    values = _profile_values(profile_energies, grid.size)
    step_rad = math.radians(GRID_STEP_DEG)
    curvature = np.abs(np.roll(values, 1) - 2.0 * values + np.roll(values, -1)) / step_rad**2
    return [
        {
            "start_angle_deg": int(grid[(index - 1) % grid.size]),
            "center_angle_deg": int(grid[index]),
            "end_angle_deg": int(grid[(index + 1) % grid.size]),
            "abs_second_derivative_per_rad2": _clean_float(curvature[index]),
        }
        for index in _select_diverse_indices(curvature, min(limit, grid.size))
    ]


def model_disagreement_intervals(
    profile_energies: Sequence[float],
    comparison_energies: Sequence[float],
    *,
    grid_angles_deg: Sequence[float] | None = None,
    limit: int = 4,
) -> dict[str, Any]:
    """Compare profile shapes after independently aligning their minima."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ProfileFitError("signal limit must be a positive integer")
    grid = _canonical_grid(grid_angles_deg)
    primary = _profile_values(profile_energies, grid.size)
    comparison = _profile_values(comparison_energies, grid.size)
    primary = primary - float(np.min(primary))
    comparison = comparison - float(np.min(comparison))
    difference = np.abs(primary - comparison)
    indices = _select_diverse_indices(difference, min(limit, grid.size))
    return {
        "available": True,
        "minimum_aligned": True,
        "rmse": _clean_float(np.sqrt(np.mean((primary - comparison) ** 2))),
        "max_abs_difference": _clean_float(np.max(difference)),
        "intervals": [
            {
                "start_angle_deg": int(grid[(index - 1) % grid.size]),
                "center_angle_deg": int(grid[index]),
                "end_angle_deg": int(grid[(index + 1) % grid.size]),
                "abs_difference": _clean_float(difference[index]),
            }
            for index in indices
        ],
    }


def _nearest_unsampled(angle: float, sampled: set[int]) -> int | None:
    candidates = [value for value in PROFILE_GRID_DEG if value not in sampled]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda value: (abs(((value - angle + 180.0) % PERIOD_DEG) - 180.0), value),
    )


def _refinement_signals(
    sampled_angles: Sequence[int],
    predictions: np.ndarray,
    comparison: np.ndarray | None,
    comparison_source: str | None,
    *,
    limit: int = 4,
) -> dict[str, Any]:
    gaps = largest_circular_gaps(sampled_angles, limit=limit)
    curvature = high_curvature_intervals(predictions, limit=limit)
    if comparison is None:
        disagreement: dict[str, Any] = {
            "available": False,
            "minimum_aligned": True,
            "rmse": None,
            "max_abs_difference": None,
            "intervals": [],
        }
    else:
        disagreement = model_disagreement_intervals(predictions, comparison, limit=limit)
    disagreement["comparison_source"] = comparison_source

    sampled = set(sampled_angles)
    recommendations: list[dict[str, Any]] = []
    used: set[int] = set()

    def add(angle: int | None, reason: str) -> None:
        if angle is None or angle in sampled or angle in used:
            return
        used.add(angle)
        recommendations.append(
            {"angle_deg": angle, "reason": reason, "priority_rank": len(recommendations) + 1}
        )

    for interval in disagreement["intervals"]:
        add(_nearest_unsampled(interval["center_angle_deg"], sampled), "model_disagreement")
    for interval in curvature:
        add(_nearest_unsampled(interval["center_angle_deg"], sampled), "high_curvature")
    for interval in gaps:
        add(interval["recommended_angle_deg"], "largest_gap")

    return {
        "largest_gaps": gaps,
        "high_curvature_intervals": curvature,
        "model_disagreement": disagreement,
        "recommendations": recommendations[: max(limit, 1) + 2],
        "recommended_angles_deg": [
            item["angle_deg"] for item in recommendations[: max(limit, 1) + 2]
        ],
    }


def _cubic_diagnostics(model: _PeriodicCubicModel) -> dict[str, Any]:
    fitted_samples = model.evaluate(model.angles)
    residuals = fitted_samples - model.energies
    count = model.angles.size
    last = count - 1
    first_width = float(model.angles[1] - model.angles[0])
    last_width = float(model.angles[0] + PERIOD_DEG - model.angles[last])
    derivative_right = (
        (model.energies[1] - model.energies[0]) / first_width
        + first_width
        * (-2.0 * model.second_derivatives[0] - model.second_derivatives[1])
        / 6.0
    )
    derivative_left = (
        (model.energies[0] - model.energies[last]) / last_width
        + last_width
        * (model.second_derivatives[last] + 2.0 * model.second_derivatives[0])
        / 6.0
    )
    widths = np.diff(np.append(model.angles, model.angles[0] + PERIOD_DEG))
    return {
        "sample_count": int(count),
        "interpolating": True,
        "matrix_condition": _clean_float(model.matrix_condition),
        "sample_rmse": _clean_float(np.sqrt(np.mean(residuals**2))),
        "sample_max_abs_error": _clean_float(np.max(np.abs(residuals))),
        "minimum_interval_deg": _clean_float(np.min(widths)),
        "maximum_interval_deg": _clean_float(np.max(widths)),
        "seam_angle_deg": int(model.angles[0]),
        "seam_value_error": 0.0,
        "seam_first_derivative_error": _clean_float(abs(derivative_right - derivative_left)),
        "seam_second_derivative_error": 0.0,
        "continuity": "C2_periodic",
    }


def _fourier_diagnostics(model: _FourierModel) -> dict[str, Any]:
    fitted_samples = model.evaluate(model.angles)
    residuals = fitted_samples - model.energies
    parameter_count = 1 + 2 * model.order
    harmonics = []
    for harmonic in range(1, model.order + 1):
        cosine = float(model.coefficients[2 * harmonic - 1])
        sine = float(model.coefficients[2 * harmonic])
        harmonics.append(
            {
                "harmonic": harmonic,
                "cosine": _clean_float(cosine),
                "sine": _clean_float(sine),
                "amplitude": _clean_float(math.hypot(cosine, sine)),
            }
        )
    return {
        "sample_count": int(model.angles.size),
        "parameter_count": parameter_count,
        "degrees_of_freedom": int(model.angles.size - parameter_count),
        "design_rank": model.rank,
        "matrix_condition": _clean_float(model.matrix_condition),
        "sample_rmse": _clean_float(np.sqrt(np.mean(residuals**2))),
        "sample_max_abs_error": _clean_float(np.max(np.abs(residuals))),
        "residual_sum_squares": _clean_float(np.sum(residuals**2)),
        "intercept": _clean_float(model.coefficients[0]),
        "harmonics": harmonics,
        "determined": True,
        "continuity": "analytic_periodic",
    }


def _automatic_comparison(
    method: str,
    model: _PeriodicCubicModel | _FourierModel,
    predictions: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    del predictions  # The comparison is independently evaluated from source samples.
    if method == "fourier":
        if model.angles.size < 3:
            return None, None
        cubic = _build_periodic_cubic_model(model.angles, model.energies)
        return cubic.evaluate(PROFILE_GRID_DEG), "periodic_cubic"

    maximum_order = min(FOURIER_MAX_ORDER, (int(model.angles.size) - 1) // 2)
    for order in range(maximum_order, FOURIER_MIN_ORDER - 1, -1):
        try:
            fourier = _build_fourier_model(model.angles, model.energies, order)
        except ProfileFitError:
            continue
        return fourier.evaluate(PROFILE_GRID_DEG), f"fourier(order={order})"
    return None, None


def fit_periodic_profile(
    angles_deg: Sequence[float] | Mapping[float, float],
    energies: Sequence[float] | None = None,
    *,
    method: str = "periodic_cubic",
    order: int | None = None,
    comparison_predictions: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fit a deterministic profile on the canonical 72-point grid.

    ``angles_deg`` may be a mapping from angle to energy, or a sequence paired
    with ``energies``.  Fourier order ``k`` has ``1 + 2*k`` coefficients and is
    rejected unless the source design has at least that many points and full
    column rank.
    """

    if method not in {"periodic_cubic", "fourier"}:
        raise ProfileFitError("method must be 'periodic_cubic' or 'fourier'")
    if method == "periodic_cubic":
        if order is not None:
            raise ProfileFitError("order is only valid for the Fourier method")
        model: _PeriodicCubicModel | _FourierModel = _build_periodic_cubic_model(
            angles_deg, energies
        )
        predictions = model.evaluate(PROFILE_GRID_DEG)
        diagnostics = _cubic_diagnostics(model)
        result_order = None
    else:
        if order is None:
            raise ProfileFitError("Fourier method requires an explicit order")
        model = _build_fourier_model(angles_deg, energies, order)
        predictions = model.evaluate(PROFILE_GRID_DEG)
        diagnostics = _fourier_diagnostics(model)
        result_order = model.order

    if not np.all(np.isfinite(predictions)):
        raise ProfileFitError("profile contains a non-finite prediction")
    if comparison_predictions is None:
        comparison, comparison_source = _automatic_comparison(method, model, predictions)
    else:
        comparison = _profile_values(comparison_predictions)
        comparison_source = "caller_supplied"

    source_angles = [int(value) for value in model.angles]
    source_energies = _float_list(model.energies)
    prediction_list = _float_list(predictions)
    scientific_payload = {
        "method": method,
        "order": result_order,
        "source_angles_deg": source_angles,
        "source_energies": source_energies,
        "grid_angles_deg": list(PROFILE_GRID_DEG),
        "predicted_energies": prediction_list,
    }
    curve_hash = hashlib.sha256(
        json.dumps(
            scientific_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest().upper()
    return {
        "schema_version": "1.0",
        **scientific_payload,
        "curve_sha256": curve_hash,
        "diagnostics": diagnostics,
        "refinement_signals": _refinement_signals(
            source_angles,
            predictions,
            comparison,
            comparison_source,
        ),
    }


def _plateau_extrema(values: np.ndarray) -> tuple[list[int], list[int]]:
    count = values.size
    tolerance = 1.0e-10 * max(1.0, float(np.ptp(values)))
    unvisited = set(range(count))
    groups: list[set[int]] = []
    while unvisited:
        seed = min(unvisited)
        group = {seed}
        frontier = [seed]
        unvisited.remove(seed)
        while frontier:
            index = frontier.pop()
            for neighbor in ((index - 1) % count, (index + 1) % count):
                if neighbor in unvisited and abs(float(values[neighbor] - values[index])) <= tolerance:
                    unvisited.remove(neighbor)
                    group.add(neighbor)
                    frontier.append(neighbor)
        groups.append(group)

    minima: list[int] = []
    maxima: list[int] = []
    for group in groups:
        if len(group) == count:
            continue
        starts = [index for index in group if (index - 1) % count not in group]
        ends = [index for index in group if (index + 1) % count not in group]
        if len(starts) != 1 or len(ends) != 1:
            continue
        start = starts[0]
        end = ends[0]
        before = (start - 1) % count
        after = (end + 1) % count
        level = float(np.mean([values[index] for index in group]))
        representative = min(group)
        if level < float(values[before]) - tolerance and level < float(values[after]) - tolerance:
            minima.append(representative)
        elif level > float(values[before]) + tolerance and level > float(values[after]) + tolerance:
            maxima.append(representative)
    return sorted(minima), sorted(maxima)


def _directional_barrier(
    values: np.ndarray,
    grid: np.ndarray,
    anchor: int,
    destination: int,
    step: int,
) -> dict[str, Any]:
    count = values.size
    distance = (destination - anchor) % count if step == 1 else (anchor - destination) % count
    if distance == 0:
        distance = count
    path = [(anchor + step * offset) % count for offset in range(distance + 1)]
    path_values = values[path]
    peak_position = int(np.argmax(path_values))
    peak = path[peak_position]
    barrier = float(values[peak] - values[anchor])
    return {
        "direction": "increasing_angle" if step == 1 else "decreasing_angle",
        "destination_minimum_angle_deg": int(grid[destination]),
        "transition_state_angle_deg": int(grid[peak]),
        "transition_state_energy": _clean_float(values[peak]),
        "path_arc_deg": int(distance * GRID_STEP_DEG),
        "barrier": _clean_float(max(0.0, barrier)),
    }


def analyze_profile(
    profile: Sequence[float] | Mapping[str, Any],
    *,
    grid_angles_deg: Sequence[float] | None = None,
    sampled_angles_deg: Sequence[float] | None = None,
    comparison_energies: Sequence[float] | None = None,
    signal_limit: int = 4,
) -> dict[str, Any]:
    """Analyze extrema, directional barriers, and refinement signals.

    A fit payload returned by :func:`fit_periodic_profile` may be supplied
    directly.  Otherwise ``profile`` must be the 72 predicted energies.
    """

    if isinstance(profile, Mapping):
        if "predicted_energies" not in profile:
            raise ProfileFitError("profile mapping lacks predicted_energies")
        values_input = profile["predicted_energies"]
        if grid_angles_deg is None:
            grid_angles_deg = profile.get("grid_angles_deg")
        if sampled_angles_deg is None:
            sampled_angles_deg = profile.get("source_angles_deg")
    else:
        values_input = profile

    grid = _canonical_grid(grid_angles_deg)
    values = _profile_values(values_input, grid.size)
    minimum_indices, maximum_indices = _plateau_extrema(values)
    anchor = int(np.argmin(values))
    tolerance = 1.0e-10 * max(1.0, float(np.ptp(values)))
    degenerate = [
        int(grid[index])
        for index in range(grid.size)
        if abs(float(values[index] - values[anchor])) <= tolerance
    ]

    minima_for_paths = sorted(set(minimum_indices + [anchor]))
    forward_candidates = [((index - anchor) % grid.size, index) for index in minima_for_paths]
    forward_candidates = [item for item in forward_candidates if item[0] > 0]
    reverse_candidates = [((anchor - index) % grid.size, index) for index in minima_for_paths]
    reverse_candidates = [item for item in reverse_candidates if item[0] > 0]
    forward_destination = min(forward_candidates)[1] if forward_candidates else anchor
    reverse_destination = min(reverse_candidates)[1] if reverse_candidates else anchor
    forward = _directional_barrier(values, grid, anchor, forward_destination, 1)
    reverse = _directional_barrier(values, grid, anchor, reverse_destination, -1)

    if sampled_angles_deg is None:
        gaps: list[dict[str, Any]] = []
    else:
        gaps = largest_circular_gaps(sampled_angles_deg, limit=signal_limit)
    curvature = high_curvature_intervals(values, limit=signal_limit)
    if comparison_energies is None:
        disagreement: dict[str, Any] = {
            "available": False,
            "minimum_aligned": True,
            "rmse": None,
            "max_abs_difference": None,
            "intervals": [],
        }
    else:
        disagreement = model_disagreement_intervals(
            values, comparison_energies, limit=signal_limit
        )

    return {
        "schema_version": "1.0",
        "grid_size": int(grid.size),
        "global_minimum": {
            "angle_deg": int(grid[anchor]),
            "energy": _clean_float(values[anchor]),
            "degenerate_angles_deg": degenerate,
        },
        "local_minima": [
            {"angle_deg": int(grid[index]), "energy": _clean_float(values[index])}
            for index in minimum_indices
        ],
        "local_maxima": [
            {"angle_deg": int(grid[index]), "energy": _clean_float(values[index])}
            for index in maximum_indices
        ],
        "barrier_definition": (
            "from the global-minimum anchor to the adjacent local minimum in each "
            "angular direction; maximum closed-path energy minus anchor energy; "
            "one full revolution when no other local minimum exists"
        ),
        "barriers": {
            "anchor_minimum_angle_deg": int(grid[anchor]),
            "forward": forward,
            "reverse": reverse,
        },
        "forward_barrier": forward["barrier"],
        "reverse_barrier": reverse["barrier"],
        "largest_gaps": gaps,
        "high_curvature_intervals": curvature,
        "model_disagreement": disagreement,
    }


__all__ = [
    "FOURIER_MAX_ORDER",
    "FOURIER_MIN_ORDER",
    "GRID_STEP_DEG",
    "PERIOD_DEG",
    "PROFILE_GRID_DEG",
    "ProfileFitError",
    "analyze_profile",
    "fit_periodic_profile",
    "fourier_predict",
    "high_curvature_intervals",
    "largest_circular_gaps",
    "model_disagreement_intervals",
    "periodic_cubic_predict",
]

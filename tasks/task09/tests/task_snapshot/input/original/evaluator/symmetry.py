"""Symmetry-aware extrema and barrier metrics for protocol 2.0.

This module is deliberately independent of model submissions, eligibility,
trajectory facts, and the legacy v1/v1.1 directional-barrier scorer.  It
accepts only two finite curves on the frozen 0, 5, ..., 355 degree grid.

Frozen scientific definitions
-----------------------------

* Periodic plateaus are treated as single extrema.
* Nearby extrema of the same kind are merged into basins by deterministic
  single-linkage on circular angle distance.
* The oracle global-minimum set contains every minimum basin no more than
  0.10 kcal/mol above the oracle minimum.
* Local extrema use maximum-cardinality, minimum-total-distance one-to-one
  circular matching.
* The primary barrier for an adjacent pair of periodic minimum basins is the
  highest energy on their clockwise arc minus the lower basin energy.  The
  resulting values form an unordered multiset, so no arbitrary clockwise
  anchor is part of the primary metric.

All computations use unrounded ``float64`` values.  Rounding happens only
when JSON-safe result dictionaries are produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np


DEFINITION_ID = "symmetry_aware_extrema_barriers/2.0"
STANDARD_GRID_DEG = np.arange(0.0, 360.0, 5.0, dtype=np.float64)

ENERGY_EQUALITY_TOLERANCE_KCAL_MOL = 1.0e-9
GLOBAL_MIN_ENERGY_TOLERANCE_KCAL_MOL = 0.10
BASIN_MERGE_ANGLE_TOLERANCE_DEG = 10.0
LOCAL_EXTREMA_MATCH_TOLERANCE_DEG = 10.0
UNMATCHED_BARRIER_ERROR_KCAL_MOL = 1.50

_ANGLE_EPSILON_DEG = 1.0e-12
_DISTANCE_INTEGER_SCALE = 1_000_000_000


class SymmetryScoringError(ValueError):
    """A curve or frozen symmetry-scoring parameter is malformed."""


@dataclass(frozen=True)
class _Curve:
    angles_deg: np.ndarray
    relative_energies_kcal_mol: np.ndarray


@dataclass(frozen=True)
class _RawExtremum:
    kind: Literal["minimum", "maximum"]
    representative_index: int
    energy_kcal_mol: float
    member_indices: tuple[int, ...]


@dataclass(frozen=True)
class ExtremumBasin:
    """One deterministic merged periodic extremum basin."""

    kind: Literal["minimum", "maximum"]
    representative_index: int
    angle_deg: float
    energy_kcal_mol: float
    member_indices: tuple[int, ...]
    member_angles_deg: tuple[float, ...]
    raw_extremum_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "representative_index": int(self.representative_index),
            "angle_deg": _rounded(self.angle_deg),
            "energy_kcal_mol": _rounded(self.energy_kcal_mol),
            "member_indices": [int(index) for index in self.member_indices],
            "member_angles_deg": [
                _rounded(angle) for angle in self.member_angles_deg
            ],
            "raw_extremum_count": int(self.raw_extremum_count),
        }


@dataclass
class _FlowEdge:
    to: int
    reverse_index: int
    capacity: int
    cost: int


def _rounded(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == 0.0 else result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SymmetryScoringError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SymmetryScoringError(f"{label} must be a finite number")
    return result


def _finite_array(values: Sequence[Any], label: str) -> np.ndarray:
    try:
        raw = list(values)
    except TypeError as exc:
        raise SymmetryScoringError(f"{label} must be a numeric sequence") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise SymmetryScoringError(f"{label} cannot contain booleans")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SymmetryScoringError(f"{label} must be a numeric sequence") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise SymmetryScoringError(
            f"{label} must be a finite one-dimensional numeric sequence"
        )
    return result


def circular_distance_deg(first: Any, second: Any) -> float:
    """Return the shortest unsigned circular distance in ``[0, 180]``."""

    a = _finite_number(first, "first angle")
    b = _finite_number(second, "second angle")
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _load_standard_curve(
    angles_deg: Sequence[Any],
    energies_kcal_mol: Sequence[Any],
    *,
    label: str,
) -> _Curve:
    angles = _finite_array(angles_deg, f"{label}.angles_deg")
    energies = _finite_array(energies_kcal_mol, f"{label}.energies_kcal_mol")
    if angles.size != STANDARD_GRID_DEG.size or energies.size != angles.size:
        raise SymmetryScoringError(
            f"{label} must contain exactly 72 angle/energy values"
        )
    canonical_angles = np.mod(angles, 360.0)
    order = np.argsort(canonical_angles, kind="stable")
    canonical_angles = canonical_angles[order]
    energies = energies[order]
    if not np.allclose(
        canonical_angles, STANDARD_GRID_DEG, rtol=0.0, atol=_ANGLE_EPSILON_DEG
    ):
        raise SymmetryScoringError(
            f"{label} angles must be the unique standard 0,5,...,355 degree grid"
        )
    relative = energies - float(np.min(energies))
    if not np.all(np.isfinite(relative)):
        raise SymmetryScoringError(f"{label} normalization was non-finite")
    return _Curve(STANDARD_GRID_DEG.copy(), relative.astype(np.float64))


def _validate_nonnegative(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise SymmetryScoringError(f"{label} must be non-negative")
    return result


def _circular_medoid_index(
    indices: Sequence[int], angles_deg: np.ndarray
) -> int:
    if not indices:
        raise SymmetryScoringError("cannot select a medoid from an empty set")
    unique = sorted(set(int(index) for index in indices))
    return min(
        unique,
        key=lambda candidate: (
            sum(
                circular_distance_deg(
                    float(angles_deg[candidate]), float(angles_deg[other])
                )
                for other in unique
            ),
            float(angles_deg[candidate]),
            candidate,
        ),
    )


def _plateau_groups(
    energies: np.ndarray, equality_tolerance: float
) -> list[tuple[int, ...]]:
    count = int(energies.size)
    breaks = [
        index
        for index in range(count)
        if not math.isclose(
            float(energies[index]),
            float(energies[(index - 1) % count]),
            rel_tol=0.0,
            abs_tol=equality_tolerance,
        )
    ]
    if not breaks:
        return [tuple(range(count))]

    start = breaks[0]
    order = [(start + offset) % count for offset in range(count)]
    groups: list[tuple[int, ...]] = []
    position = 0
    while position < count:
        first = order[position]
        value = float(energies[first])
        group = [first]
        position += 1
        while position < count and math.isclose(
            float(energies[order[position]]),
            value,
            rel_tol=0.0,
            abs_tol=equality_tolerance,
        ):
            group.append(order[position])
            position += 1
        groups.append(tuple(sorted(group)))
    return groups


def _raw_periodic_extrema(
    curve: _Curve, equality_tolerance: float
) -> tuple[list[_RawExtremum], list[_RawExtremum]]:
    groups = _plateau_groups(
        curve.relative_energies_kcal_mol, equality_tolerance
    )
    if len(groups) == 1:
        return [], []

    group_values = [
        float(np.mean(curve.relative_energies_kcal_mol[list(group)]))
        for group in groups
    ]
    minima: list[_RawExtremum] = []
    maxima: list[_RawExtremum] = []
    for group_index, group in enumerate(groups):
        value = group_values[group_index]
        previous = group_values[(group_index - 1) % len(groups)]
        following = group_values[(group_index + 1) % len(groups)]
        representative = _circular_medoid_index(group, curve.angles_deg)
        if (
            value < previous - equality_tolerance
            and value < following - equality_tolerance
        ):
            minima.append(
                _RawExtremum(
                    "minimum",
                    representative,
                    float(
                        np.min(curve.relative_energies_kcal_mol[list(group)])
                    ),
                    group,
                )
            )
        elif (
            value > previous + equality_tolerance
            and value > following + equality_tolerance
        ):
            maxima.append(
                _RawExtremum(
                    "maximum",
                    representative,
                    float(
                        np.max(curve.relative_energies_kcal_mol[list(group)])
                    ),
                    group,
                )
            )
    minima.sort(key=lambda item: float(curve.angles_deg[item.representative_index]))
    maxima.sort(key=lambda item: float(curve.angles_deg[item.representative_index]))
    return minima, maxima


def _merge_extrema(
    raw_extrema: Sequence[_RawExtremum],
    curve: _Curve,
    *,
    merge_tolerance_deg: float,
    equality_tolerance: float,
) -> list[ExtremumBasin]:
    if not raw_extrema:
        return []
    kind = raw_extrema[0].kind
    count = len(raw_extrema)
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for first in range(count):
        first_angle = float(
            curve.angles_deg[raw_extrema[first].representative_index]
        )
        for second in range(first + 1, count):
            second_angle = float(
                curve.angles_deg[raw_extrema[second].representative_index]
            )
            if (
                circular_distance_deg(first_angle, second_angle)
                <= merge_tolerance_deg + _ANGLE_EPSILON_DEG
            ):
                adjacency[first].append(second)
                adjacency[second].append(first)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(count):
        if start in seen:
            continue
        stack = [start]
        component: list[int] = []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    basins: list[ExtremumBasin] = []
    for component in components:
        items = [raw_extrema[index] for index in component]
        target_energy = (
            min(item.energy_kcal_mol for item in items)
            if kind == "minimum"
            else max(item.energy_kcal_mol for item in items)
        )
        tied_representatives = [
            item.representative_index
            for item in items
            if math.isclose(
                item.energy_kcal_mol,
                target_energy,
                rel_tol=0.0,
                abs_tol=equality_tolerance,
            )
        ]
        representative = _circular_medoid_index(
            tied_representatives, curve.angles_deg
        )
        members = tuple(
            sorted(
                {
                    member
                    for item in items
                    for member in item.member_indices
                }
            )
        )
        energy = (
            float(np.min(curve.relative_energies_kcal_mol[list(members)]))
            if kind == "minimum"
            else float(np.max(curve.relative_energies_kcal_mol[list(members)]))
        )
        basins.append(
            ExtremumBasin(
                kind=kind,
                representative_index=representative,
                angle_deg=float(curve.angles_deg[representative]),
                energy_kcal_mol=energy,
                member_indices=members,
                member_angles_deg=tuple(
                    float(curve.angles_deg[index]) for index in members
                ),
                raw_extremum_count=len(items),
            )
        )
    basins.sort(key=lambda basin: (basin.angle_deg, basin.representative_index))
    return basins


def _extract_from_curve(
    curve: _Curve,
    *,
    equality_tolerance: float,
    merge_tolerance_deg: float,
) -> tuple[list[ExtremumBasin], list[ExtremumBasin]]:
    raw_minima, raw_maxima = _raw_periodic_extrema(
        curve, equality_tolerance
    )
    minima = _merge_extrema(
        raw_minima,
        curve,
        merge_tolerance_deg=merge_tolerance_deg,
        equality_tolerance=equality_tolerance,
    )
    maxima = _merge_extrema(
        raw_maxima,
        curve,
        merge_tolerance_deg=merge_tolerance_deg,
        equality_tolerance=equality_tolerance,
    )
    return minima, maxima


def extract_extrema_basins(
    angles_deg: Sequence[Any],
    energies_kcal_mol: Sequence[Any],
    *,
    equality_tolerance_kcal_mol: float = ENERGY_EQUALITY_TOLERANCE_KCAL_MOL,
    merge_tolerance_deg: float = BASIN_MERGE_ANGLE_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Extract deterministic merged periodic minimum and maximum basins."""

    equality_tolerance = _validate_nonnegative(
        equality_tolerance_kcal_mol, "equality_tolerance_kcal_mol"
    )
    merge_tolerance = _validate_nonnegative(
        merge_tolerance_deg, "merge_tolerance_deg"
    )
    if merge_tolerance > 180.0:
        raise SymmetryScoringError("merge_tolerance_deg cannot exceed 180")
    curve = _load_standard_curve(
        angles_deg, energies_kcal_mol, label="curve"
    )
    minima, maxima = _extract_from_curve(
        curve,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance,
    )
    return {
        "definition_id": DEFINITION_ID,
        "minima": [basin.as_dict() for basin in minima],
        "maxima": [basin.as_dict() for basin in maxima],
        "minimum_count": len(minima),
        "maximum_count": len(maxima),
        "equality_tolerance_kcal_mol": _rounded(equality_tolerance),
        "merge_tolerance_deg": _rounded(merge_tolerance),
    }


def _global_basin_comparison_from_curves(
    predicted_curve: _Curve,
    oracle_curve: _Curve,
    *,
    equality_tolerance: float,
    merge_tolerance_deg: float,
    global_min_energy_tolerance: float,
) -> dict[str, Any]:
    predicted_minima, _ = _extract_from_curve(
        predicted_curve,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance_deg,
    )
    oracle_minima, _ = _extract_from_curve(
        oracle_curve,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance_deg,
    )
    predicted_global = [
        basin
        for basin in predicted_minima
        if basin.energy_kcal_mol <= equality_tolerance
    ]
    oracle_admissible = [
        basin
        for basin in oracle_minima
        if basin.energy_kcal_mol
        <= global_min_energy_tolerance + equality_tolerance
    ]

    nearest: dict[str, Any] | None = None
    if not predicted_global and not oracle_admissible:
        error = 0.0
    elif not predicted_global or not oracle_admissible:
        error = 180.0
    else:
        candidates = [
            (
                circular_distance_deg(predicted.angle_deg, oracle.angle_deg),
                predicted.angle_deg,
                oracle.angle_deg,
                predicted,
                oracle,
            )
            for predicted in predicted_global
            for oracle in oracle_admissible
        ]
        distance, _predicted_angle, _oracle_angle, predicted, oracle = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        error = float(distance)
        nearest = {
            "predicted_angle_deg": _rounded(predicted.angle_deg),
            "oracle_angle_deg": _rounded(oracle.angle_deg),
            "distance_deg": _rounded(distance),
        }
    return {
        "predicted_global_basins": [
            basin.as_dict() for basin in predicted_global
        ],
        "oracle_admissible_basins": [
            basin.as_dict() for basin in oracle_admissible
        ],
        "nearest_match": nearest,
        "circular_error_deg": _rounded(error),
        "oracle_energy_tolerance_kcal_mol": _rounded(
            global_min_energy_tolerance
        ),
    }


def global_minimum_basin_comparison(
    predicted_angles_deg: Sequence[Any],
    predicted_energies_kcal_mol: Sequence[Any],
    oracle_angles_deg: Sequence[Any],
    oracle_energies_kcal_mol: Sequence[Any],
    *,
    equality_tolerance_kcal_mol: float = ENERGY_EQUALITY_TOLERANCE_KCAL_MOL,
    merge_tolerance_deg: float = BASIN_MERGE_ANGLE_TOLERANCE_DEG,
    global_min_energy_tolerance_kcal_mol: float = (
        GLOBAL_MIN_ENERGY_TOLERANCE_KCAL_MOL
    ),
) -> dict[str, Any]:
    """Compare a predicted global basin against all admissible oracle basins."""

    equality_tolerance = _validate_nonnegative(
        equality_tolerance_kcal_mol, "equality_tolerance_kcal_mol"
    )
    merge_tolerance = _validate_nonnegative(
        merge_tolerance_deg, "merge_tolerance_deg"
    )
    global_tolerance = _validate_nonnegative(
        global_min_energy_tolerance_kcal_mol,
        "global_min_energy_tolerance_kcal_mol",
    )
    if merge_tolerance > 180.0:
        raise SymmetryScoringError("merge_tolerance_deg cannot exceed 180")
    predicted = _load_standard_curve(
        predicted_angles_deg,
        predicted_energies_kcal_mol,
        label="predicted",
    )
    oracle = _load_standard_curve(
        oracle_angles_deg, oracle_energies_kcal_mol, label="oracle"
    )
    return _global_basin_comparison_from_curves(
        predicted,
        oracle,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance,
        global_min_energy_tolerance=global_tolerance,
    )


def global_minimum_basin_error_deg(
    predicted_angles_deg: Sequence[Any],
    predicted_energies_kcal_mol: Sequence[Any],
    oracle_angles_deg: Sequence[Any],
    oracle_energies_kcal_mol: Sequence[Any],
    **kwargs: Any,
) -> float:
    """Return only the frozen global-minimum basin circular error."""

    result = global_minimum_basin_comparison(
        predicted_angles_deg,
        predicted_energies_kcal_mol,
        oracle_angles_deg,
        oracle_energies_kcal_mol,
        **kwargs,
    )
    return float(result["circular_error_deg"])


def _add_flow_edge(
    graph: list[list[_FlowEdge]], first: int, second: int, capacity: int, cost: int
) -> _FlowEdge:
    forward = _FlowEdge(second, len(graph[second]), capacity, cost)
    reverse = _FlowEdge(first, len(graph[first]), 0, -cost)
    graph[first].append(forward)
    graph[second].append(reverse)
    return forward


def optimal_circular_extrema_matches(
    predicted_angles_deg: Sequence[Any],
    oracle_angles_deg: Sequence[Any],
    *,
    tolerance_deg: float = LOCAL_EXTREMA_MATCH_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Maximum-cardinality, minimum-distance circular one-to-one matching."""

    tolerance = _validate_nonnegative(tolerance_deg, "tolerance_deg")
    if tolerance > 180.0:
        raise SymmetryScoringError("tolerance_deg cannot exceed 180")
    predicted_raw = _finite_array(
        predicted_angles_deg, "predicted_angles_deg"
    )
    oracle_raw = _finite_array(oracle_angles_deg, "oracle_angles_deg")
    predicted = sorted(
        ((float(angle % 360.0), index) for index, angle in enumerate(predicted_raw)),
        key=lambda item: (item[0], item[1]),
    )
    oracle = sorted(
        ((float(angle % 360.0), index) for index, angle in enumerate(oracle_raw)),
        key=lambda item: (item[0], item[1]),
    )
    if not predicted or not oracle:
        match_count = 0
        precision = 1.0 if not predicted else 0.0
        recall = 1.0 if not oracle else 0.0
        f1 = 1.0 if not predicted and not oracle else 0.0
        return {
            "matches": [],
            "match_count": match_count,
            "predicted_count": len(predicted),
            "oracle_count": len(oracle),
            "unmatched_predicted_angles_deg": [
                _rounded(item[0]) for item in predicted
            ],
            "unmatched_oracle_angles_deg": [
                _rounded(item[0]) for item in oracle
            ],
            "precision": _rounded(precision),
            "recall": _rounded(recall),
            "f1": _rounded(f1),
            "total_matched_distance_deg": 0.0,
            "tolerance_deg": _rounded(tolerance),
        }

    predicted_count = len(predicted)
    oracle_count = len(oracle)
    source = 0
    predicted_start = 1
    oracle_start = predicted_start + predicted_count
    sink = oracle_start + oracle_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    for predicted_position in range(predicted_count):
        _add_flow_edge(
            graph, source, predicted_start + predicted_position, 1, 0
        )
    for oracle_position in range(oracle_count):
        _add_flow_edge(graph, oracle_start + oracle_position, sink, 1, 0)

    max_pairs = min(predicted_count, oracle_count)
    max_pair_rank = predicted_count * oracle_count
    distance_base = max_pairs * max_pair_rank + 1
    candidate_edges: list[
        tuple[_FlowEdge, int, int, float]
    ] = []
    for predicted_position, (predicted_angle, _predicted_index) in enumerate(
        predicted
    ):
        candidates: list[tuple[int, float]] = []
        for oracle_position, (oracle_angle, _oracle_index) in enumerate(oracle):
            distance = circular_distance_deg(predicted_angle, oracle_angle)
            if distance <= tolerance + _ANGLE_EPSILON_DEG:
                candidates.append((oracle_position, distance))
        candidates.sort(
            key=lambda item: (
                item[1],
                oracle[item[0]][0],
                oracle[item[0]][1],
            )
        )
        for oracle_position, distance in candidates:
            distance_units = int(round(distance * _DISTANCE_INTEGER_SCALE))
            pair_rank = (
                predicted_position * oracle_count + oracle_position + 1
            )
            cost = distance_units * distance_base + pair_rank
            edge = _add_flow_edge(
                graph,
                predicted_start + predicted_position,
                oracle_start + oracle_position,
                1,
                cost,
            )
            candidate_edges.append(
                (edge, predicted_position, oracle_position, distance)
            )

    # Successive shortest augmenting paths with Bellman-Ford on the residual
    # graph.  Integer costs and canonical edge insertion freeze all decisions.
    while True:
        node_count = len(graph)
        distances: list[int | None] = [None] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for node in range(node_count):
                if distances[node] is None:
                    continue
                for edge_index, edge in enumerate(graph[node]):
                    if edge.capacity <= 0:
                        continue
                    candidate = distances[node] + edge.cost
                    if distances[edge.to] is None or candidate < distances[edge.to]:
                        distances[edge.to] = candidate
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if distances[sink] is None:
            break
        node = sink
        while node != source:
            predecessor = previous[node]
            if predecessor is None:  # defensive; sink was reachable
                raise RuntimeError("broken residual augmenting path")
            previous_node, edge_index = predecessor
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = previous_node

    used: list[tuple[int, int, float]] = [
        (predicted_position, oracle_position, distance)
        for edge, predicted_position, oracle_position, distance in candidate_edges
        if edge.capacity == 0
    ]
    used.sort(
        key=lambda item: (
            predicted[item[0]][0],
            oracle[item[1]][0],
            predicted[item[0]][1],
            oracle[item[1]][1],
        )
    )
    matched_predicted_positions = {item[0] for item in used}
    matched_oracle_positions = {item[1] for item in used}
    matches = [
        {
            "predicted_index": int(predicted[predicted_position][1]),
            "oracle_index": int(oracle[oracle_position][1]),
            "predicted_angle_deg": _rounded(
                predicted[predicted_position][0]
            ),
            "oracle_angle_deg": _rounded(oracle[oracle_position][0]),
            "distance_deg": _rounded(distance),
        }
        for predicted_position, oracle_position, distance in used
    ]
    match_count = len(matches)
    precision = match_count / predicted_count if predicted_count else 1.0
    recall = match_count / oracle_count if oracle_count else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "matches": matches,
        "match_count": match_count,
        "predicted_count": predicted_count,
        "oracle_count": oracle_count,
        "unmatched_predicted_angles_deg": [
            _rounded(angle)
            for position, (angle, _index) in enumerate(predicted)
            if position not in matched_predicted_positions
        ],
        "unmatched_oracle_angles_deg": [
            _rounded(angle)
            for position, (angle, _index) in enumerate(oracle)
            if position not in matched_oracle_positions
        ],
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "total_matched_distance_deg": _rounded(
            sum(item[2] for item in used)
        ),
        "tolerance_deg": _rounded(tolerance),
    }


def _local_extrema_comparison_from_curves(
    predicted_curve: _Curve,
    oracle_curve: _Curve,
    *,
    equality_tolerance: float,
    merge_tolerance_deg: float,
    match_tolerance_deg: float,
) -> dict[str, Any]:
    predicted_minima, predicted_maxima = _extract_from_curve(
        predicted_curve,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance_deg,
    )
    oracle_minima, oracle_maxima = _extract_from_curve(
        oracle_curve,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance_deg,
    )
    minima = optimal_circular_extrema_matches(
        [basin.angle_deg for basin in predicted_minima],
        [basin.angle_deg for basin in oracle_minima],
        tolerance_deg=match_tolerance_deg,
    )
    maxima = optimal_circular_extrema_matches(
        [basin.angle_deg for basin in predicted_maxima],
        [basin.angle_deg for basin in oracle_maxima],
        tolerance_deg=match_tolerance_deg,
    )
    minima["predicted_basins"] = [
        basin.as_dict() for basin in predicted_minima
    ]
    minima["oracle_basins"] = [basin.as_dict() for basin in oracle_minima]
    maxima["predicted_basins"] = [
        basin.as_dict() for basin in predicted_maxima
    ]
    maxima["oracle_basins"] = [basin.as_dict() for basin in oracle_maxima]
    return {"minima": minima, "maxima": maxima}


def compare_local_extrema(
    predicted_angles_deg: Sequence[Any],
    predicted_energies_kcal_mol: Sequence[Any],
    oracle_angles_deg: Sequence[Any],
    oracle_energies_kcal_mol: Sequence[Any],
    *,
    equality_tolerance_kcal_mol: float = ENERGY_EQUALITY_TOLERANCE_KCAL_MOL,
    merge_tolerance_deg: float = BASIN_MERGE_ANGLE_TOLERANCE_DEG,
    match_tolerance_deg: float = LOCAL_EXTREMA_MATCH_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Compare all merged minima and maxima; global minima are not removed."""

    equality_tolerance = _validate_nonnegative(
        equality_tolerance_kcal_mol, "equality_tolerance_kcal_mol"
    )
    merge_tolerance = _validate_nonnegative(
        merge_tolerance_deg, "merge_tolerance_deg"
    )
    match_tolerance = _validate_nonnegative(
        match_tolerance_deg, "match_tolerance_deg"
    )
    if merge_tolerance > 180.0 or match_tolerance > 180.0:
        raise SymmetryScoringError("angle tolerances cannot exceed 180")
    predicted = _load_standard_curve(
        predicted_angles_deg,
        predicted_energies_kcal_mol,
        label="predicted",
    )
    oracle = _load_standard_curve(
        oracle_angles_deg, oracle_energies_kcal_mol, label="oracle"
    )
    return _local_extrema_comparison_from_curves(
        predicted,
        oracle,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance,
        match_tolerance_deg=match_tolerance,
    )


def _clockwise_arc_indices(start: int, end: int, count: int) -> list[int]:
    if start == end:
        return [(start + offset) % count for offset in range(count)]
    result = [start]
    current = start
    while current != end:
        current = (current + 1) % count
        result.append(current)
    return result


def _barriers_from_curve(
    curve: _Curve,
    *,
    equality_tolerance: float,
    merge_tolerance_deg: float,
) -> tuple[list[ExtremumBasin], list[dict[str, Any]], list[float]]:
    minima, _ = _extract_from_curve(
        curve,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance_deg,
    )
    if not minima:
        return minima, [], []
    ordered = sorted(
        minima, key=lambda basin: (basin.representative_index, basin.angle_deg)
    )
    edges: list[dict[str, Any]] = []
    count = int(curve.angles_deg.size)
    for position, start_basin in enumerate(ordered):
        end_basin = ordered[(position + 1) % len(ordered)]
        arc = _clockwise_arc_indices(
            start_basin.representative_index,
            end_basin.representative_index,
            count,
        )
        saddle_energy = float(
            np.max(curve.relative_energies_kcal_mol[arc])
        )
        saddle_candidates = [
            index
            for index in arc
            if math.isclose(
                float(curve.relative_energies_kcal_mol[index]),
                saddle_energy,
                rel_tol=0.0,
                abs_tol=equality_tolerance,
            )
        ]
        saddle_index = _circular_medoid_index(
            saddle_candidates, curve.angles_deg
        )
        lower_basin_energy = min(
            start_basin.energy_kcal_mol, end_basin.energy_kcal_mol
        )
        barrier = max(0.0, saddle_energy - lower_basin_energy)
        edges.append(
            {
                "start_basin_angle_deg": start_basin.angle_deg,
                "end_basin_angle_deg": end_basin.angle_deg,
                "saddle_angle_deg": float(curve.angles_deg[saddle_index]),
                "saddle_energy_kcal_mol": saddle_energy,
                "lower_basin_energy_kcal_mol": lower_basin_energy,
                "barrier_kcal_mol": barrier,
                "clockwise_arc_point_count": len(arc),
            }
        )
    multiset = sorted(float(edge["barrier_kcal_mol"]) for edge in edges)
    return minima, edges, multiset


def adjacent_periodic_barrier_multiset(
    angles_deg: Sequence[Any],
    energies_kcal_mol: Sequence[Any],
    *,
    equality_tolerance_kcal_mol: float = ENERGY_EQUALITY_TOLERANCE_KCAL_MOL,
    merge_tolerance_deg: float = BASIN_MERGE_ANGLE_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Return every adjacent periodic minimum-basin barrier as a multiset."""

    equality_tolerance = _validate_nonnegative(
        equality_tolerance_kcal_mol, "equality_tolerance_kcal_mol"
    )
    merge_tolerance = _validate_nonnegative(
        merge_tolerance_deg, "merge_tolerance_deg"
    )
    if merge_tolerance > 180.0:
        raise SymmetryScoringError("merge_tolerance_deg cannot exceed 180")
    curve = _load_standard_curve(
        angles_deg, energies_kcal_mol, label="curve"
    )
    minima, edges, multiset = _barriers_from_curve(
        curve,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance,
    )
    return {
        "definition": (
            "For every clockwise-adjacent pair of merged periodic minimum "
            "basins, max(arc energy) minus the lower endpoint basin energy; "
            "the primary values form an unordered multiset."
        ),
        "minimum_basins": [basin.as_dict() for basin in minima],
        "edges": [
            {
                key: (
                    int(value)
                    if key == "clockwise_arc_point_count"
                    else _rounded(float(value))
                )
                for key, value in edge.items()
            }
            for edge in edges
        ],
        "barrier_multiset_kcal_mol": [
            _rounded(value) for value in multiset
        ],
        "barrier_count": len(multiset),
    }


def _barrier_array(values: Sequence[Any], label: str) -> list[tuple[float, int]]:
    result = _finite_array(values, label)
    if np.any(result < 0.0):
        raise SymmetryScoringError(f"{label} cannot contain negative barriers")
    return sorted(
        ((float(value), index) for index, value in enumerate(result)),
        key=lambda item: (item[0], item[1]),
    )


def _ordered_l1_assignment(
    predicted: list[tuple[float, int]],
    oracle: list[tuple[float, int]],
) -> list[tuple[int, int]]:
    """Exact deterministic L1 assignment of the smaller sorted multiset."""

    predicted_is_smaller = len(predicted) <= len(oracle)
    smaller = predicted if predicted_is_smaller else oracle
    larger = oracle if predicted_is_smaller else predicted
    small_count = len(smaller)
    large_count = len(larger)
    if small_count == 0:
        return []

    # Each cell contains (total L1 cost, canonical pair tuple).  For scalar
    # L1 matching an optimal assignment is order-preserving, so this dynamic
    # program is exact and avoids a numerical optimizer dependency.
    table: list[list[tuple[float, tuple[tuple[int, int], ...]] | None]] = [
        [None] * (large_count + 1) for _ in range(small_count + 1)
    ]
    for large_index in range(large_count + 1):
        table[0][large_index] = (0.0, ())

    def better(
        first: tuple[float, tuple[tuple[int, int], ...]] | None,
        second: tuple[float, tuple[tuple[int, int], ...]] | None,
    ) -> tuple[float, tuple[tuple[int, int], ...]] | None:
        if first is None:
            return second
        if second is None:
            return first
        if first[0] < second[0] - 1.0e-15:
            return first
        if second[0] < first[0] - 1.0e-15:
            return second
        return first if first[1] <= second[1] else second

    for small_index in range(1, small_count + 1):
        for large_index in range(1, large_count + 1):
            skip = table[small_index][large_index - 1]
            prior = table[small_index - 1][large_index - 1]
            match = None
            if prior is not None:
                small_value, small_original = smaller[small_index - 1]
                large_value, large_original = larger[large_index - 1]
                pair = (
                    (small_original, large_original)
                    if predicted_is_smaller
                    else (large_original, small_original)
                )
                match = (
                    prior[0] + abs(small_value - large_value),
                    prior[1] + (pair,),
                )
            table[small_index][large_index] = better(skip, match)

    result = table[small_count][large_count]
    if result is None:
        raise RuntimeError("failed to construct scalar L1 assignment")
    return list(result[1])


def match_barrier_multisets(
    predicted_barriers_kcal_mol: Sequence[Any],
    oracle_barriers_kcal_mol: Sequence[Any],
    *,
    unmatched_error_kcal_mol: float = UNMATCHED_BARRIER_ERROR_KCAL_MOL,
) -> dict[str, Any]:
    """Optimally match scalar barrier multisets and penalize cardinality gaps."""

    unmatched_error = _validate_nonnegative(
        unmatched_error_kcal_mol, "unmatched_error_kcal_mol"
    )
    predicted = _barrier_array(
        predicted_barriers_kcal_mol, "predicted_barriers_kcal_mol"
    )
    oracle = _barrier_array(
        oracle_barriers_kcal_mol, "oracle_barriers_kcal_mol"
    )
    pairs = _ordered_l1_assignment(predicted, oracle)
    predicted_by_original = {index: value for value, index in predicted}
    oracle_by_original = {index: value for value, index in oracle}
    matches = []
    matched_errors_raw: list[float] = []
    matched_predicted: set[int] = set()
    matched_oracle: set[int] = set()
    for predicted_index, oracle_index in pairs:
        predicted_value = predicted_by_original[predicted_index]
        oracle_value = oracle_by_original[oracle_index]
        error = abs(predicted_value - oracle_value)
        matched_errors_raw.append(error)
        matched_predicted.add(predicted_index)
        matched_oracle.add(oracle_index)
        matches.append(
            {
                "predicted_index": int(predicted_index),
                "oracle_index": int(oracle_index),
                "predicted_barrier_kcal_mol": _rounded(predicted_value),
                "oracle_barrier_kcal_mol": _rounded(oracle_value),
                "absolute_error_kcal_mol": _rounded(error),
            }
        )
    matches.sort(
        key=lambda item: (
            float(item["predicted_barrier_kcal_mol"]),
            float(item["oracle_barrier_kcal_mol"]),
            int(item["predicted_index"]),
            int(item["oracle_index"]),
        )
    )
    unmatched_predicted = [
        value for value, index in predicted if index not in matched_predicted
    ]
    unmatched_oracle = [
        value for value, index in oracle if index not in matched_oracle
    ]
    cardinality_gap = len(unmatched_predicted) + len(unmatched_oracle)
    all_errors = matched_errors_raw + [unmatched_error] * cardinality_gap
    denominator = max(len(predicted), len(oracle))
    mae = sum(all_errors) / denominator if denominator else 0.0
    maximum = max(all_errors, default=0.0)
    return {
        "predicted_multiset_kcal_mol": [
            _rounded(value) for value, _index in predicted
        ],
        "oracle_multiset_kcal_mol": [
            _rounded(value) for value, _index in oracle
        ],
        "matches": matches,
        "unmatched_predicted_kcal_mol": [
            _rounded(value) for value in unmatched_predicted
        ],
        "unmatched_oracle_kcal_mol": [
            _rounded(value) for value in unmatched_oracle
        ],
        "unmatched_error_kcal_mol": _rounded(unmatched_error),
        "mae_kcal_mol": _rounded(mae),
        "maximum_error_kcal_mol": _rounded(maximum),
        "matched_count": len(matches),
        "predicted_count": len(predicted),
        "oracle_count": len(oracle),
    }


def evaluate_symmetry_aware_profile(
    predicted_angles_deg: Sequence[Any],
    predicted_energies_kcal_mol: Sequence[Any],
    oracle_angles_deg: Sequence[Any],
    oracle_energies_kcal_mol: Sequence[Any],
    *,
    equality_tolerance_kcal_mol: float = ENERGY_EQUALITY_TOLERANCE_KCAL_MOL,
    merge_tolerance_deg: float = BASIN_MERGE_ANGLE_TOLERANCE_DEG,
    global_min_energy_tolerance_kcal_mol: float = (
        GLOBAL_MIN_ENERGY_TOLERANCE_KCAL_MOL
    ),
    extrema_match_tolerance_deg: float = (
        LOCAL_EXTREMA_MATCH_TOLERANCE_DEG
    ),
    unmatched_barrier_error_kcal_mol: float = (
        UNMATCHED_BARRIER_ERROR_KCAL_MOL
    ),
) -> dict[str, Any]:
    """Compute all v2 symmetry-aware scientific metrics.

    There is intentionally no eligibility or process-facts parameter.
    """

    equality_tolerance = _validate_nonnegative(
        equality_tolerance_kcal_mol, "equality_tolerance_kcal_mol"
    )
    merge_tolerance = _validate_nonnegative(
        merge_tolerance_deg, "merge_tolerance_deg"
    )
    global_tolerance = _validate_nonnegative(
        global_min_energy_tolerance_kcal_mol,
        "global_min_energy_tolerance_kcal_mol",
    )
    match_tolerance = _validate_nonnegative(
        extrema_match_tolerance_deg, "extrema_match_tolerance_deg"
    )
    unmatched_error = _validate_nonnegative(
        unmatched_barrier_error_kcal_mol,
        "unmatched_barrier_error_kcal_mol",
    )
    if merge_tolerance > 180.0 or match_tolerance > 180.0:
        raise SymmetryScoringError("angle tolerances cannot exceed 180")

    predicted = _load_standard_curve(
        predicted_angles_deg,
        predicted_energies_kcal_mol,
        label="predicted",
    )
    oracle = _load_standard_curve(
        oracle_angles_deg, oracle_energies_kcal_mol, label="oracle"
    )
    global_comparison = _global_basin_comparison_from_curves(
        predicted,
        oracle,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance,
        global_min_energy_tolerance=global_tolerance,
    )
    local_comparison = _local_extrema_comparison_from_curves(
        predicted,
        oracle,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance,
        match_tolerance_deg=match_tolerance,
    )
    predicted_minima, predicted_edges, predicted_barriers = (
        _barriers_from_curve(
            predicted,
            equality_tolerance=equality_tolerance,
            merge_tolerance_deg=merge_tolerance,
        )
    )
    oracle_minima, oracle_edges, oracle_barriers = _barriers_from_curve(
        oracle,
        equality_tolerance=equality_tolerance,
        merge_tolerance_deg=merge_tolerance,
    )
    barrier_comparison = match_barrier_multisets(
        predicted_barriers,
        oracle_barriers,
        unmatched_error_kcal_mol=unmatched_error,
    )
    barrier_comparison["predicted_edges"] = [
        {
            key: (
                int(value)
                if key == "clockwise_arc_point_count"
                else _rounded(float(value))
            )
            for key, value in edge.items()
        }
        for edge in predicted_edges
    ]
    barrier_comparison["oracle_edges"] = [
        {
            key: (
                int(value)
                if key == "clockwise_arc_point_count"
                else _rounded(float(value))
            )
            for key, value in edge.items()
        }
        for edge in oracle_edges
    ]
    barrier_comparison["predicted_minimum_basins"] = [
        basin.as_dict() for basin in predicted_minima
    ]
    barrier_comparison["oracle_minimum_basins"] = [
        basin.as_dict() for basin in oracle_minima
    ]
    return {
        "definition_id": DEFINITION_ID,
        "global_minimum_basins": global_comparison,
        "local_extrema": local_comparison,
        "adjacent_periodic_barriers": barrier_comparison,
        "parameters": {
            "energy_equality_tolerance_kcal_mol": _rounded(
                equality_tolerance
            ),
            "global_min_energy_tolerance_kcal_mol": _rounded(
                global_tolerance
            ),
            "basin_merge_angle_tolerance_deg": _rounded(merge_tolerance),
            "local_extrema_match_tolerance_deg": _rounded(match_tolerance),
            "unmatched_barrier_error_kcal_mol": _rounded(unmatched_error),
        },
    }


__all__ = [
    "BASIN_MERGE_ANGLE_TOLERANCE_DEG",
    "DEFINITION_ID",
    "ENERGY_EQUALITY_TOLERANCE_KCAL_MOL",
    "ExtremumBasin",
    "GLOBAL_MIN_ENERGY_TOLERANCE_KCAL_MOL",
    "LOCAL_EXTREMA_MATCH_TOLERANCE_DEG",
    "STANDARD_GRID_DEG",
    "SymmetryScoringError",
    "UNMATCHED_BARRIER_ERROR_KCAL_MOL",
    "adjacent_periodic_barrier_multiset",
    "circular_distance_deg",
    "compare_local_extrema",
    "evaluate_symmetry_aware_profile",
    "extract_extrema_basins",
    "global_minimum_basin_comparison",
    "global_minimum_basin_error_deg",
    "match_barrier_multisets",
    "optimal_circular_extrema_matches",
]

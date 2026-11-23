"""Deterministic basin, path, and coupling metrics for Task 06."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Sequence

import numpy as np

from .coupled_geometry import torus_distance


HARTREE_TO_KCAL_MOL = 627.509474


@dataclass(frozen=True)
class LandscapePoint:
    point_id: str
    angles_deg: tuple[float, float]
    energy_hartree: float
    job_id: str
    frame_index: int | None
    geometry_sha256: str
    coordinates_angstrom: np.ndarray | None = field(default=None, repr=False, compare=False)
    elements: tuple[str, ...] = field(default=(), repr=False, compare=False)
    global_basin_support: bool = False
    grid_angles_deg: tuple[float, float] | None = None


@dataclass(frozen=True)
class Basin:
    basin_id: str
    representative_point_id: str
    member_point_ids: tuple[str, ...]
    angles_deg: tuple[float, float]
    energy_hartree: float


@dataclass(frozen=True)
class Transition:
    transition_id: str
    from_basin_id: str
    to_basin_id: str
    ordered_point_ids: tuple[str, ...]
    highest_point_id: str
    forward_barrier_kcal_mol: float
    reverse_barrier_kcal_mol: float


def _kabsch_rmsd(first: LandscapePoint, second: LandscapePoint) -> float:
    if first.coordinates_angstrom is None or second.coordinates_angstrom is None:
        return 0.0
    first_xyz = np.asarray(first.coordinates_angstrom, dtype=float)
    second_xyz = np.asarray(second.coordinates_angstrom, dtype=float)
    if first_xyz.shape != second_xyz.shape or first_xyz.ndim != 2 or first_xyz.shape[1] != 3:
        return float("inf")
    if first.elements and second.elements and first.elements != second.elements:
        return float("inf")
    if first.elements:
        indices = [index for index, element in enumerate(first.elements) if element != "H"]
        first_xyz = first_xyz[indices]
        second_xyz = second_xyz[indices]
    first_centered = first_xyz - first_xyz.mean(axis=0)
    second_centered = second_xyz - second_xyz.mean(axis=0)
    covariance = first_centered.T @ second_centered
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    delta = first_centered @ rotation - second_centered
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def cluster_basins(
    points: Sequence[LandscapePoint],
    rmsd_threshold_angstrom: float,
    angle_threshold_deg: float,
) -> list[Basin]:
    """Greedily cluster from the lowest-energy geometry with fixed tie-breaks."""

    ordered = sorted(points, key=lambda item: (item.energy_hartree, item.point_id))
    groups: list[list[LandscapePoint]] = []
    for point in ordered:
        assigned = False
        for group in groups:
            representative = group[0]
            if (
                torus_distance(point.angles_deg, representative.angles_deg) <= angle_threshold_deg
                and _kabsch_rmsd(point, representative) <= rmsd_threshold_angstrom
            ):
                group.append(point)
                assigned = True
                break
        if not assigned:
            groups.append([point])

    basins: list[Basin] = []
    for index, group in enumerate(groups, start=1):
        representative = min(group, key=lambda item: (item.energy_hartree, item.point_id))
        basins.append(
            Basin(
                basin_id=f"B{index:03d}",
                representative_point_id=representative.point_id,
                member_point_ids=tuple(sorted(item.point_id for item in group)),
                angles_deg=representative.angles_deg,
                energy_hartree=representative.energy_hartree,
            )
        )
    return basins


def _adjacency(points: Sequence[LandscapePoint], neighbor_limit_deg: float) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {point.point_id: set() for point in points}
    for first, second in combinations(points, 2):
        if torus_distance(first.angles_deg, second.angles_deg) <= neighbor_limit_deg:
            result[first.point_id].add(second.point_id)
            result[second.point_id].add(first.point_id)
    return {point_id: tuple(sorted(values)) for point_id, values in result.items()}


def _minimax_path(
    start: str,
    stop: str,
    points_by_id: dict[str, LandscapePoint],
    adjacency: dict[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    start_energy = points_by_id[start].energy_hartree
    queue: list[tuple[float, int, tuple[str, ...], str]] = [(start_energy, 1, (start,), start)]
    best: dict[str, tuple[float, int, tuple[str, ...]]] = {start: (start_energy, 1, (start,))}
    while queue:
        bottleneck, count, path, current = heapq.heappop(queue)
        if best.get(current) != (bottleneck, count, path):
            continue
        if current == stop:
            return path
        for adjacent in adjacency[current]:
            if adjacent in path:
                continue
            candidate_path = path + (adjacent,)
            candidate = (
                max(bottleneck, points_by_id[adjacent].energy_hartree),
                count + 1,
                candidate_path,
            )
            if adjacent not in best or candidate < best[adjacent]:
                best[adjacent] = candidate
                heapq.heappush(queue, (*candidate, adjacent))
    return None


def minimum_barrier_paths(
    points: Sequence[LandscapePoint],
    basins: Sequence[Basin],
    neighbor_limit_deg: float,
) -> list[Transition]:
    """Return deterministic pairwise minimax paths on the periodic point graph."""

    points_by_id = {point.point_id: point for point in points}
    adjacency = _adjacency(points, neighbor_limit_deg)
    transitions: list[Transition] = []
    for from_basin, to_basin in combinations(sorted(basins, key=lambda item: item.basin_id), 2):
        if from_basin.representative_point_id not in points_by_id or to_basin.representative_point_id not in points_by_id:
            continue
        path = _minimax_path(
            from_basin.representative_point_id,
            to_basin.representative_point_id,
            points_by_id,
            adjacency,
        )
        if path is None:
            continue
        highest = min(path, key=lambda point_id: (-points_by_id[point_id].energy_hartree, point_id))
        highest_energy = points_by_id[highest].energy_hartree
        transitions.append(
            Transition(
                transition_id=f"{from_basin.basin_id}_to_{to_basin.basin_id}",
                from_basin_id=from_basin.basin_id,
                to_basin_id=to_basin.basin_id,
                ordered_point_ids=path,
                highest_point_id=highest,
                forward_barrier_kcal_mol=(highest_energy - from_basin.energy_hartree) * HARTREE_TO_KCAL_MOL,
                reverse_barrier_kcal_mol=(highest_energy - to_basin.energy_hartree) * HARTREE_TO_KCAL_MOL,
            )
        )
    return transitions


def nonadditivity_rmse(matrix: np.ndarray) -> float:
    """Two-way residual RMSE for a complete rectangular energy matrix."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("coupling matrix must be a finite two-dimensional rectangle")
    residual = values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()
    return float(np.sqrt(np.mean(residual * residual)))


def _periodic_span(values: Sequence[float]) -> float:
    return max(
        abs((first - second + 180.0) % 360.0 - 180.0)
        for first, second in product(values, repeat=2)
    )


def coupling_support_valid(points: Sequence[LandscapePoint]) -> bool:
    """Check the fixed >=3x3, >=120-degree, global-region evidence gate."""

    if not points or not any(point.global_basin_support for point in points):
        return False
    support_angles = [point.grid_angles_deg or point.angles_deg for point in points]
    first_values = sorted({round(angles[0] % 360.0, 6) for angles in support_angles})
    second_values = sorted({round(angles[1] % 360.0, 6) for angles in support_angles})
    if len(first_values) < 3 or len(second_values) < 3:
        return False
    if _periodic_span(first_values) < 120.0 - 1.0e-6 or _periodic_span(second_values) < 120.0 - 1.0e-6:
        return False
    observed = {
        (round(angles[0] % 360.0, 6), round(angles[1] % 360.0, 6))
        for angles in support_angles
    }
    required = set(product(first_values, second_values))
    return observed == required and len(points) == len(required)

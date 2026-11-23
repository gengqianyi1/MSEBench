"""Small XYZ and geometry module used only by the private Task 04 scorer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class XYZGeometry:
    elements: tuple[str, ...]
    coordinates: np.ndarray
    comment: str = ""

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.shape != (len(self.elements), 3):
            raise ValueError("XYZ coordinate shape does not match element count")
        if not np.isfinite(coordinates).all():
            raise ValueError("XYZ contains non-finite coordinates")
        object.__setattr__(self, "coordinates", coordinates)


def read_xyz(path: str | Path) -> XYZGeometry:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ is incomplete")
    atom_count = int(lines[0].strip())
    if atom_count <= 0 or len(lines) < atom_count + 2:
        raise ValueError("XYZ atom count is invalid")
    elements: list[str] = []
    coordinates: list[list[float]] = []
    for line in lines[2 : atom_count + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("XYZ atom row is incomplete")
        elements.append(fields[0])
        coordinates.append([float(value) for value in fields[1:4]])
    return XYZGeometry(tuple(elements), np.asarray(coordinates), lines[1])


def kabsch_rmsd(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> float:
    target = np.asarray(reference, dtype=float)
    trial = np.asarray(candidate, dtype=float)
    if target.shape != trial.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Kabsch inputs must have equal N x 3 shapes")
    target_centered = target - target.mean(axis=0)
    trial_centered = trial - trial.mean(axis=0)
    covariance = trial_centered.T @ target_centered
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    aligned = trial_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - target_centered) ** 2, axis=1))))


def heavy_atom_rmsd(reference: XYZGeometry, candidate: XYZGeometry) -> float:
    if tuple(item.upper() for item in reference.elements) != tuple(
        item.upper() for item in candidate.elements
    ):
        return math.inf
    indices = [index for index, element in enumerate(reference.elements) if element.upper() != "H"]
    return kabsch_rmsd(reference.coordinates[indices], candidate.coordinates[indices])


def geometry_dihedral(geometry: XYZGeometry, atom_ids: Sequence[int]) -> float:
    if len(atom_ids) != 4:
        raise ValueError("a dihedral requires four atom IDs")
    p0, p1, p2, p3 = (geometry.coordinates[index - 1] for index in atom_ids)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def periodic_distance_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)

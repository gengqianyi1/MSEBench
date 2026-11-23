"""Geometry primitives with stable atom ordering and periodic-angle handling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from common.io_utils import native_path


@dataclass(frozen=True)
class XYZGeometry:
    elements: tuple[str, ...]
    coordinates: np.ndarray
    comment: str = ""

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.shape != (len(self.elements), 3):
            raise ValueError("coordinates must have shape (atom_count, 3)")
        if not np.isfinite(coords).all():
            raise ValueError("coordinates contain non-finite values")
        object.__setattr__(self, "coordinates", coords)

    def as_dict(self) -> dict[str, object]:
        return {
            "atom_count": len(self.elements),
            "elements": list(self.elements),
            "coordinates_angstrom": self.coordinates.tolist(),
            "comment": self.comment,
        }


def _parse_xyz_lines(lines: Sequence[str], offset: int = 0) -> tuple[XYZGeometry, int]:
    if offset >= len(lines):
        raise ValueError("missing XYZ atom count")
    try:
        atom_count = int(lines[offset].strip())
    except ValueError as exc:
        raise ValueError("invalid XYZ atom count") from exc
    if atom_count <= 0:
        raise ValueError("XYZ atom count must be positive")
    if offset + atom_count + 2 > len(lines):
        raise ValueError("truncated XYZ frame")
    comment = lines[offset + 1].rstrip("\n")
    elements: list[str] = []
    coords: list[list[float]] = []
    for line in lines[offset + 2 : offset + 2 + atom_count]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"invalid XYZ atom line: {line!r}")
        elements.append(parts[0])
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError as exc:
            raise ValueError(f"invalid XYZ coordinate line: {line!r}") from exc
    return XYZGeometry(tuple(elements), np.asarray(coords, dtype=float), comment), offset + atom_count + 2


def read_xyz(path: str | Path) -> XYZGeometry:
    frames = read_xyz_frames(path)
    if len(frames) != 1:
        raise ValueError(f"expected one XYZ frame, found {len(frames)}")
    return frames[0]


def read_xyz_frames(path: str | Path) -> list[XYZGeometry]:
    lines = native_path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    frames: list[XYZGeometry] = []
    offset = 0
    while offset < len(lines):
        while offset < len(lines) and not lines[offset].strip():
            offset += 1
        if offset >= len(lines):
            break
        frame, offset = _parse_xyz_lines(lines, offset)
        frames.append(frame)
    if not frames:
        raise ValueError("empty XYZ file")
    return frames


def kabsch_rmsd(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> float:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) == 0:
        raise ValueError("Kabsch inputs must share non-empty shape (N, 3)")
    a0 = a - a.mean(axis=0)
    b0 = b - b.mean(axis=0)
    covariance = b0.T @ a0
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    aligned = b0 @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - a0) ** 2, axis=1))))


def pairwise_distance_matrix(coordinates: Sequence[Sequence[float]]) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    delta = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def max_internal_distance_error(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> float:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        return math.inf
    return float(np.max(np.abs(pairwise_distance_matrix(a) - pairwise_distance_matrix(b)))) if len(a) else 0.0


def geometry_from_record(value: object) -> XYZGeometry | None:
    """Convert a parser geometry record to the shared immutable representation."""
    if not isinstance(value, dict):
        return None
    try:
        elements = tuple(str(item) for item in value["elements"])
        coordinates = value.get("coordinates_angstrom", value.get("coordinates"))
        return XYZGeometry(elements, np.asarray(coordinates, dtype=float), str(value.get("comment") or ""))
    except (KeyError, TypeError, ValueError):
        return None


def heavy_atom_rmsd(reference: XYZGeometry, candidate: XYZGeometry) -> float:
    """Kabsch RMSD over corresponding non-hydrogen atoms with stable ordering."""
    if tuple(item.upper() for item in reference.elements) != tuple(item.upper() for item in candidate.elements):
        return math.inf
    indices = [index for index, element in enumerate(reference.elements) if element.upper() != "H"]
    if not indices:
        return math.inf
    return kabsch_rmsd(reference.coordinates[indices], candidate.coordinates[indices])


def proton_attachment_signature(geometry: XYZGeometry, *, max_bond_angstrom: float = 1.30) -> tuple[int, ...] | None:
    """Return the one-based heavy-atom host of every H in stable H order.

    This intentionally uses only geometry and atom ordering.  A hydrogen farther
    than the conservative covalent cutoff from every heavy atom makes the
    identity invalid rather than being assigned to the nearest atom anyway.
    """
    heavy = [index for index, element in enumerate(geometry.elements) if element.upper() != "H"]
    hydrogens = [index for index, element in enumerate(geometry.elements) if element.upper() == "H"]
    if not heavy or not hydrogens:
        return None
    hosts: list[int] = []
    for hydrogen in hydrogens:
        distances = [(float(np.linalg.norm(geometry.coordinates[hydrogen] - geometry.coordinates[index])), index) for index in heavy]
        distance, host = min(distances)
        if not math.isfinite(distance) or distance > max_bond_angstrom:
            return None
        hosts.append(host + 1)
    return tuple(hosts)


def minimum_cross_fragment_distance(
    geometry: XYZGeometry, fragment_a_atom_ids: Sequence[int], fragment_b_atom_ids: Sequence[int], *, heavy_only: bool = False
) -> float:
    """Minimum interfragment distance using one-based frozen atom membership."""
    a = [int(value) - 1 for value in fragment_a_atom_ids]
    b = [int(value) - 1 for value in fragment_b_atom_ids]
    if min(a + b, default=-1) < 0 or max(a + b, default=-1) >= len(geometry.elements):
        return math.inf
    if heavy_only:
        a = [index for index in a if geometry.elements[index].upper() != "H"]
        b = [index for index in b if geometry.elements[index].upper() != "H"]
    if not a or not b:
        return math.inf
    delta = geometry.coordinates[a][:, None, :] - geometry.coordinates[b][None, :, :]
    return float(np.min(np.linalg.norm(delta, axis=-1)))


def interfragment_contact_vector(
    geometry: XYZGeometry, fragment_a_atom_ids: Sequence[int], fragment_b_atom_ids: Sequence[int], *, heavy_only: bool = True
) -> np.ndarray:
    """Stable vector of all mapped interfragment distances for clustering."""
    a = [int(value) - 1 for value in fragment_a_atom_ids]
    b = [int(value) - 1 for value in fragment_b_atom_ids]
    if min(a + b, default=-1) < 0 or max(a + b, default=-1) >= len(geometry.elements):
        return np.asarray([], dtype=float)
    if heavy_only:
        a = [index for index in a if geometry.elements[index].upper() != "H"]
        b = [index for index in b if geometry.elements[index].upper() != "H"]
    if not a or not b:
        return np.asarray([], dtype=float)
    delta = geometry.coordinates[a][:, None, :] - geometry.coordinates[b][None, :, :]
    return np.linalg.norm(delta, axis=-1).reshape(-1)


def normalize_angle_deg(value: float) -> float:
    result = float(value) % 360.0
    return 0.0 if abs(result - 360.0) < 1e-12 or abs(result) < 1e-12 else result


def periodic_distance_deg(a: float, b: float) -> float:
    difference = abs(normalize_angle_deg(a) - normalize_angle_deg(b))
    return min(difference, 360.0 - difference)


def geometry_dihedral(geometry: XYZGeometry, atom_ids: Sequence[int]) -> float:
    if len(atom_ids) != 4:
        raise ValueError("a dihedral requires four one-based atom IDs")
    indices = [int(value) - 1 for value in atom_ids]
    if min(indices) < 0 or max(indices) >= len(geometry.elements) or len(set(indices)) < 4:
        raise IndexError("invalid dihedral atom IDs")
    p0, p1, p2, p3 = geometry.coordinates[indices]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm < 1e-14:
        raise ValueError("central dihedral bond has zero length")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-14 or np.linalg.norm(w) < 1e-14:
        raise ValueError("dihedral is undefined for collinear atoms")
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return normalize_angle_deg(math.degrees(math.atan2(y, x)))


def circular_coverage_deg(angles: Iterable[float]) -> float:
    unique = sorted({round(normalize_angle_deg(float(value)), 10) for value in angles})
    if len(unique) < 2:
        return 0.0
    gaps = [unique[index + 1] - unique[index] for index in range(len(unique) - 1)]
    gaps.append(360.0 - unique[-1] + unique[0])
    return max(0.0, 360.0 - max(gaps))


def maximum_circular_step_deg(angles: Iterable[float]) -> float:
    unique = sorted({round(normalize_angle_deg(float(value)), 10) for value in angles})
    if len(unique) < 2:
        return math.inf
    gaps = [unique[index + 1] - unique[index] for index in range(len(unique) - 1)]
    gaps.append(360.0 - unique[-1] + unique[0])
    return float(max(gaps))


def periodic_linear_interpolate(
    angles: Sequence[float], values: Sequence[float], query_angles: Sequence[float] | np.ndarray
) -> np.ndarray:
    if len(angles) != len(values) or len(angles) < 2:
        raise ValueError("periodic interpolation requires at least two paired points")
    buckets: dict[float, list[float]] = {}
    for angle, value in zip(angles, values, strict=True):
        key = round(normalize_angle_deg(float(angle)), 10)
        buckets.setdefault(key, []).append(float(value))
    x = np.asarray(sorted(buckets), dtype=float)
    y = np.asarray([sum(buckets[key]) / len(buckets[key]) for key in x], dtype=float)
    if len(x) < 2:
        raise ValueError("periodic interpolation requires two unique angles")
    x_extended = np.concatenate(([x[-1] - 360.0], x, [x[0] + 360.0]))
    y_extended = np.concatenate(([y[-1]], y, [y[0]]))
    q = np.mod(np.asarray(query_angles, dtype=float), 360.0)
    return np.interp(q, x_extended, y_extended)

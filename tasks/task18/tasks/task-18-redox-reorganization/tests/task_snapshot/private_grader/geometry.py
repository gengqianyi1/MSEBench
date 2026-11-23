"""Geometry helpers used only by the Task 18 private grader."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_xyz_text(text: str) -> tuple[list[str], np.ndarray]:
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty XYZ")
    atom_count = int(lines[0].strip())
    if len(lines) < atom_count + 2:
        raise ValueError("truncated XYZ")
    elements: list[str] = []
    coordinates: list[list[float]] = []
    for line in lines[2 : atom_count + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("invalid XYZ atom row")
        elements.append(fields[0])
        coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return elements, np.asarray(coordinates, dtype=float)


def parse_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    return parse_xyz_text(Path(path).read_text(encoding="utf-8", errors="strict"))


def ordered_element_signature(text: str) -> tuple[str, ...]:
    elements, _ = parse_xyz_text(text)
    return tuple(elements)


def kabsch_rmsd_text(left: str, right: str) -> float:
    left_elements, left_coordinates = parse_xyz_text(left)
    right_elements, right_coordinates = parse_xyz_text(right)
    if left_elements != right_elements:
        raise ValueError("XYZ atom order or elements differ")
    p = left_coordinates - left_coordinates.mean(axis=0)
    q = right_coordinates - right_coordinates.mean(axis=0)
    covariance = p.T @ q
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned = p @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1))))

"""Private, geometry-derived coordinate primitives for Task 06.

Nothing in this module is copied into the model-visible task.  It infers the
three-ring topology from coordinates so model labels and reported angles never
define reference identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np


_COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71}


@dataclass(frozen=True)
class CanonicalCoordinate:
    coordinate_id: str
    central_bond: tuple[int, int]
    ring_a: tuple[int, ...]
    ring_b: tuple[int, ...]


def _connectivity(elements: Sequence[str], coordinates: np.ndarray) -> tuple[frozenset[int], ...]:
    if coordinates.shape != (len(elements), 3):
        raise ValueError("coordinate array does not match the element list")
    neighbors: list[set[int]] = [set() for _ in elements]
    for first in range(len(elements)):
        try:
            first_radius = _COVALENT_RADII[elements[first]]
        except KeyError as exc:
            raise ValueError(f"unsupported element in Task 06 geometry: {elements[first]}") from exc
        for second in range(first + 1, len(elements)):
            second_radius = _COVALENT_RADII[elements[second]]
            distance = float(np.linalg.norm(coordinates[first] - coordinates[second]))
            if 0.45 <= distance <= 1.25 * (first_radius + second_radius):
                neighbors[first].add(second)
                neighbors[second].add(first)
    return tuple(frozenset(values) for values in neighbors)


def _canonical_cycle(values: Iterable[int]) -> tuple[int, ...]:
    cycle = tuple(values)
    variants: list[tuple[int, ...]] = []
    for orientation in (cycle, tuple(reversed(cycle))):
        variants.extend(orientation[index:] + orientation[:index] for index in range(len(cycle)))
    return min(variants)


def _six_membered_heavy_rings(
    elements: Sequence[str], neighbors: Sequence[frozenset[int]]
) -> tuple[tuple[int, ...], ...]:
    heavy = {index for index, element in enumerate(elements) if element != "H"}
    found: set[tuple[int, ...]] = set()

    def walk(start: int, path: tuple[int, ...]) -> None:
        current = path[-1]
        if len(path) == 6:
            if start in neighbors[current]:
                candidate = frozenset(path)
                internal_edges = sum(
                    1
                    for atom in candidate
                    for adjacent in neighbors[atom]
                    if adjacent in candidate
                ) // 2
                if internal_edges == 6:
                    found.add(_canonical_cycle(path))
            return
        for adjacent in sorted(neighbors[current] & heavy):
            if adjacent == start or adjacent in path:
                continue
            walk(start, path + (adjacent,))

    for start in sorted(heavy):
        walk(start, (start,))
    return tuple(sorted(found))


def canonical_coordinates(
    elements: Sequence[str], reference_xyz: np.ndarray
) -> tuple[CanonicalCoordinate, CanonicalCoordinate]:
    """Infer the two terminal-ring/central-ring links from Task 06 geometry."""

    coordinates = np.asarray(reference_xyz, dtype=float)
    neighbors = _connectivity(elements, coordinates)
    rings = _six_membered_heavy_rings(elements, neighbors)
    if len(rings) != 3:
        raise ValueError(f"expected three six-membered rings, found {len(rings)}")

    ring_sets = [set(ring) for ring in rings]
    links: dict[tuple[int, int], tuple[int, int]] = {}
    degrees = [0, 0, 0]
    for first in range(len(rings)):
        for second in range(first + 1, len(rings)):
            connecting = sorted(
                (atom, adjacent)
                for atom in ring_sets[first]
                for adjacent in neighbors[atom]
                if adjacent in ring_sets[second]
            )
            if len(connecting) == 1:
                links[(first, second)] = connecting[0]
                degrees[first] += 1
                degrees[second] += 1
            elif connecting:
                raise ValueError("ring topology contains an ambiguous fused or multiply linked pair")

    if sorted(degrees) != [1, 1, 2]:
        raise ValueError("expected three six-membered rings in a terminal-central-terminal chain")
    central_index = degrees.index(2)
    definitions: list[CanonicalCoordinate] = []
    for terminal_index in sorted(index for index, degree in enumerate(degrees) if degree == 1):
        pair = tuple(sorted((central_index, terminal_index)))
        raw_first, raw_second = links[pair]
        if raw_first in ring_sets[central_index]:
            center_atom, terminal_atom = raw_first, raw_second
        else:
            center_atom, terminal_atom = raw_second, raw_first
        definitions.append(
            CanonicalCoordinate(
                coordinate_id="",
                central_bond=(center_atom, terminal_atom),
                ring_a=tuple(sorted(rings[central_index])),
                ring_b=tuple(sorted(rings[terminal_index])),
            )
        )

    definitions.sort(key=lambda item: item.central_bond)
    named = tuple(
        CanonicalCoordinate(
            coordinate_id=f"private_q{index}",
            central_bond=definition.central_bond,
            ring_a=definition.ring_a,
            ring_b=definition.ring_b,
        )
        for index, definition in enumerate(definitions, start=1)
    )
    if len(named) != 2:
        raise ValueError("expected exactly two coupled ring-link coordinates")
    return named  # type: ignore[return-value]


def _dihedral_degrees(points: np.ndarray) -> float:
    first, second, third, fourth = points
    middle = third - second
    middle_norm = float(np.linalg.norm(middle))
    if middle_norm <= 1.0e-12:
        raise ValueError("degenerate central bond in coordinate geometry")
    axis = middle / middle_norm
    left = first - second
    right = fourth - third
    left -= float(np.dot(left, axis)) * axis
    right -= float(np.dot(right, axis)) * axis
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        raise ValueError("degenerate ring anchor in coordinate geometry")
    left /= left_norm
    right /= right_norm
    x_value = float(np.dot(left, right))
    y_value = float(np.dot(np.cross(left, right), axis))
    return float(np.degrees(np.arctan2(y_value, x_value))) % 360.0


def coordinate_values(
    coordinates: np.ndarray, definitions: Sequence[CanonicalCoordinate]
) -> tuple[float, float]:
    """Calculate deterministic ring-link torsions for the two definitions."""

    xyz = np.asarray(coordinates, dtype=float)
    values: list[float] = []
    for definition in definitions:
        first_center, second_terminal = definition.central_bond
        center_candidates = sorted(set(definition.ring_a) - {first_center, second_terminal})
        terminal_candidates = sorted(set(definition.ring_b) - {first_center, second_terminal})
        center_anchor = min(center_candidates, key=lambda atom: float(np.linalg.norm(xyz[atom] - xyz[first_center])))
        terminal_anchor = min(
            terminal_candidates,
            key=lambda atom: float(np.linalg.norm(xyz[atom] - xyz[second_terminal])),
        )
        values.append(
            _dihedral_degrees(
                xyz[[center_anchor, first_center, second_terminal, terminal_anchor]]
            )
        )
    if len(values) != 2:
        raise ValueError("Task 06 requires exactly two private coordinates")
    return values[0], values[1]


def symmetry_images(pair: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    """Return exchange/sign images for identical terminal rings and conventions."""

    first, second = pair
    images = {
        ((first_sign * first) % 360.0, (second_sign * second) % 360.0)
        for first_sign, second_sign in product((-1.0, 1.0), repeat=2)
    }
    images.update((second_value, first_value) for first_value, second_value in tuple(images))
    return tuple(sorted(images))


def _circular_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def torus_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Minimum angular Euclidean distance over Task 06 symmetry images."""

    return min(
        math_distance
        for image in symmetry_images(second)
        for math_distance in (
            float(
                np.hypot(
                    _circular_difference(first[0], image[0]),
                    _circular_difference(first[1], image[1]),
                )
            ),
        )
    )


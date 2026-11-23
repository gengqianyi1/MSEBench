"""Private coordinate-derived L-histidine identity and stereochemistry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tasks.task_04.private.evaluator.geometry import XYZGeometry, geometry_dihedral


@dataclass(frozen=True)
class HistidineIdentity:
    family: str | None
    classification: str
    hydrogen_assignments: tuple[tuple[int, int], ...]
    chirality: str
    proton_attachment_signature: tuple[int, ...] | None
    chi1_deg: float
    chi2_deg: float
    signed_chiral_volume: float


EXPECTED_ELEMENTS = ("O", "O", "N", "N", "N", "C", "C", "C", "C", "C", "C") + (
    "H",
) * 9
HEAVY_ATOM_IDS = tuple(range(1, 12))
HYDROGEN_ATOM_IDS = tuple(range(12, 21))
FAMILY_BY_OCCUPANCY = {
    (2, 1, 1, 0): "neutral_npi_h",
    (2, 1, 0, 1): "neutral_ntau_h",
    (3, 0, 1, 0): "zwitterionic_npi_h",
    (3, 0, 0, 1): "zwitterionic_ntau_h",
}


def _require_mapping(geometry: XYZGeometry) -> None:
    if tuple(item.upper() for item in geometry.elements) != EXPECTED_ELEMENTS:
        raise ValueError("geometry does not use the frozen 20-atom histidine mapping")


def assign_hydrogens_to_heavy_atoms(
    geometry: XYZGeometry, maximum_distance: float = 1.30
) -> tuple[tuple[int, int], ...]:
    _require_mapping(geometry)
    assignments: list[tuple[int, int]] = []
    for hydrogen_atom_id in HYDROGEN_ATOM_IDS:
        hydrogen = geometry.coordinates[hydrogen_atom_id - 1]
        distances = sorted(
            (
                float(np.linalg.norm(hydrogen - geometry.coordinates[heavy_atom_id - 1])),
                heavy_atom_id,
            )
            for heavy_atom_id in HEAVY_ATOM_IDS
        )
        nearest_distance, nearest_host = distances[0]
        if (
            math.isfinite(nearest_distance)
            and nearest_distance <= float(maximum_distance)
            and distances[1][0] - nearest_distance > 1e-6
        ):
            assignments.append((hydrogen_atom_id, nearest_host))
    return tuple(assignments)


def signed_histidine_chiral_volume(geometry: XYZGeometry) -> float:
    alpha = geometry.coordinates[6]
    amino = geometry.coordinates[3] - alpha
    carboxyl = geometry.coordinates[8] - alpha
    beta = geometry.coordinates[5] - alpha
    return float(np.dot(amino, np.cross(carboxyl, beta)))


def classify_histidine_geometry(geometry: XYZGeometry) -> HistidineIdentity:
    assignments = assign_hydrogens_to_heavy_atoms(geometry)
    hosts = tuple(host for _hydrogen, host in assignments)
    occupancy = (hosts.count(4), hosts.count(1) + hosts.count(2), hosts.count(3), hosts.count(5))
    family = FAMILY_BY_OCCUPANCY.get(occupancy) if len(assignments) == 9 else None
    volume = signed_histidine_chiral_volume(geometry)
    chirality = "L" if volume > 0.25 else ("D" if volume < -0.25 else "achiral_or_unresolved")
    return HistidineIdentity(
        family=family,
        classification=family or "collapsed_or_other",
        hydrogen_assignments=assignments,
        chirality=chirality,
        proton_attachment_signature=hosts if len(assignments) == 9 else None,
        chi1_deg=geometry_dihedral(geometry, (4, 7, 6, 8)),
        chi2_deg=geometry_dihedral(geometry, (7, 6, 8, 3)),
        signed_chiral_volume=volume,
    )

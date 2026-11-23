from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class XYZ:
    elements: List[str]
    coords: np.ndarray
    comment: str = ""

    def copy(self) -> "XYZ":
        return XYZ(list(self.elements), np.array(self.coords, dtype=float), self.comment)


def read_xyz(path: str | Path) -> XYZ:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    elements = []
    coords = []
    for line in lines[2:2 + n]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return XYZ(elements=elements, coords=np.array(coords, dtype=float), comment=comment)


def write_xyz(path: str | Path, xyz: XYZ) -> None:
    lines = [str(len(xyz.elements)), xyz.comment or ""]
    for el, (x, y, z) in zip(xyz.elements, xyz.coords):
        lines.append(f"{el:2s} {x: .10f} {y: .10f} {z: .10f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def align_by_principal_axes(xyz: XYZ) -> XYZ:
    """Align using heavy-atom PCA.

    - Center at heavy-atom centroid.
    - x: largest variance axis.
    - y: second largest variance axis.
    - z: smallest variance axis.
    - Ensure right-handed coordinate system.
    """
    heavy_idx = [i for i, el in enumerate(xyz.elements) if el.upper() != "H"]
    if len(heavy_idx) < 3:
        heavy_idx = list(range(len(xyz.elements)))

    heavy = xyz.coords[heavy_idx]
    centroid = heavy.mean(axis=0)
    centered_all = xyz.coords - centroid
    centered_heavy = heavy - centroid

    cov = np.cov(centered_heavy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]

    # Eigenvector signs are mathematically arbitrary. Canonicalize every axis
    # by requiring its largest-magnitude Cartesian component to be positive;
    # np.argmax provides a deterministic tie break by component order.
    for column in range(axes.shape[1]):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0

    aligned_coords = centered_all @ axes
    return XYZ(elements=list(xyz.elements), coords=aligned_coords, comment=xyz.comment + " | principal-axis aligned")


def make_slipped_cofacial_dimer(
    monomer_xyz: XYZ,
    dx: float,
    dy: float,
    dz: float,
) -> XYZ:
    aligned = align_by_principal_axes(monomer_xyz)
    displacement = np.array([dx, dy, dz], dtype=float)
    elems = list(aligned.elements) + list(aligned.elements)
    coords = np.vstack([aligned.coords, aligned.coords + displacement])
    return XYZ(elements=elems, coords=coords, comment=f"standardized dimer dx={dx} dy={dy} dz={dz}")


def write_xcontrol_split(path: str | Path, n_atoms_monomer: int) -> None:
    text = f"""$split
   fragment1: 1-{n_atoms_monomer}
   fragment2: {n_atoms_monomer + 1}-{2 * n_atoms_monomer}
$end
"""
    Path(path).write_text(text, encoding="utf-8")

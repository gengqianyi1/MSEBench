#!/usr/bin/env python3
"""Build deterministic caffeine--water rigid placements with NumPy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
_SCREEN_SPEC = importlib.util.spec_from_file_location("task05_caffeine_screen", HERE / "screen_caffeine_water.py")
if _SCREEN_SPEC is None or _SCREEN_SPEC.loader is None:
    raise RuntimeError("cannot load caffeine--water pose recipe")
_SCREEN = importlib.util.module_from_spec(_SCREEN_SPEC)
_SCREEN_SPEC.loader.exec_module(_SCREEN)


def _read_xyz(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].strip())
    rows = [line.split() for line in lines[2:] if line.strip()]
    if len(rows) != count:
        raise ValueError(f"invalid XYZ: {path}")
    import numpy as np

    return [row[0] for row in rows], np.asarray([[float(value) for value in row[1:4]] for row in rows], dtype=float)


def _write_xyz(path: Path, elements, coordinates, comment: str) -> None:
    rows = [str(len(elements)), comment]
    rows.extend(
        f"{element:<2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for element, xyz in zip(elements, coordinates, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def challenge_spec() -> dict:
    return {
        "schema_version": "1.0.0",
        "task_version": "1.10.1",
        "outcome_status": "validated_by_private_real_xtb_screen",
        "pose_count": 14,
        "construction": "frozen_monomers_explicit_rigid_transforms",
        "minimum_cross_fragment_distance_angstrom": 1.20,
        "minimum_heavy_cross_fragment_distance_angstrom": 2.20,
        "required_golden_assertions": {
            "post_optimization_pose_collapse": True,
            "near_degenerate_retained_pair": True,
            "interaction_deformation_competition": True,
            "three_repetitions_consistent": True,
        },
        "required_contact_families": ["carbonyl_o1", "carbonyl_o2", "bent_edge", "non_hbond_close_contact"],
    }


def build_poses(source_dir: str | Path, output_dir: str | Path) -> dict:
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("SOURCE.md", "monomer_A.xyz", "monomer_B.xyz", "monomer_manifest.csv"):
        source, destination = source_dir / name, output_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    elements_a, caffeine = _read_xyz(source_dir / "monomer_A.xyz")
    elements_b, water = _read_xyz(source_dir / "monomer_B.xyz")
    poses = _SCREEN.build_screening_poses(caffeine, water)
    if elements_a != _SCREEN.CAFFEINE_ELEMENTS:
        raise ValueError("monomer_A atom order does not match frozen caffeine CID 2519 mapping")
    if elements_b != ["O", "H", "H"]:
        raise ValueError("monomer_B atom order does not match frozen water mapping")

    pose_records, manifest_rows, fragment_rows = [], [], []
    for index, (private_label, recipe, coordinates) in enumerate(poses, start=1):
        pose_id = f"POSE_{index:02d}"
        name = f"complex_pose_{index:02d}.xyz"
        output = output_dir / name
        _write_xyz(output, elements_a + elements_b, coordinates, f"{pose_id}; anonymous rigid placement; fragments A then B")
        a, b = coordinates[: len(elements_a)], coordinates[len(elements_a) :]
        import numpy as np

        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        heavy_a = [atom for atom, element in enumerate(elements_a) if element != "H"]
        pose_records.append(
            {
                "pose_id": pose_id,
                "private_recipe_id": private_label,
                "complex_file": name,
                "private_initial_family": recipe["family"],
                "sha256": _sha256(output),
                "minimum_cross_fragment_distance_angstrom": float(distances.min()),
                "minimum_heavy_cross_fragment_distance_angstrom": float(distances[np.ix_(heavy_a, [0])].min()),
            }
        )
        manifest_rows.append({"pose_id": pose_id, "complex_file": name, "formula": "C8H12N4O3", "charge": 0, "uhf": 0, "atom_count": len(elements_a) + len(elements_b)})
        for atom_index, element in enumerate(elements_a, start=1):
            fragment_rows.append({"complex_file": name, "pose_id": pose_id, "complex_atom_id": atom_index, "element": element, "fragment_id": "A", "fragment_atom_id": atom_index})
        for fragment_index, element in enumerate(elements_b, start=1):
            fragment_rows.append({"complex_file": name, "pose_id": pose_id, "complex_atom_id": len(elements_a) + fragment_index, "element": element, "fragment_id": "B", "fragment_atom_id": fragment_index})

    with (output_dir / "complex_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output_dir / "fragment_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fragment_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(fragment_rows)
    stale_private_manifest = output_dir / "pose_build_manifest.json"
    if stale_private_manifest.exists():
        stale_private_manifest.unlink()
    public_files = ["SOURCE.md", "complex_manifest.csv", "fragment_manifest.csv", "monomer_A.xyz", "monomer_B.xyz", "monomer_manifest.csv"] + [row["complex_file"] for row in pose_records]
    public_manifest = {
        "schema_version": "2.2",
        "task_id": "task_05_binding_decomposition",
        "task_version": "1.10.1",
        "system": "caffeine_water",
        "complex_count": len(pose_records),
        "fragment_count": 2,
        "generation": "NumPy-only rigid placement; no force-field preoptimization",
        "contains_reference_energies": False,
        "files": {name: _sha256(output_dir / name) for name in sorted(public_files)},
    }
    (output_dir / "public_data_manifest.json").write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"schema_version": "1.0.0", "task_version": "1.10.1", "poses": pose_records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    build_poses(args.source_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

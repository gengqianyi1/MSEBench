#!/usr/bin/env python3
"""Private real-xTB screen for a possible uracil--water Task 05 system.

This module is not model-visible.  It constructs rigid placements from the
official PubChem CID 1174 3D conformer and the frozen water monomer, optimizes
them with xTB 6.7.1, clusters the resulting complexes, and recomputes the
frozen-fragment decomposition from raw jobs before deciding whether the system
is suitable for promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[3]
PUBLIC = PACKAGE_ROOT / "private_runtime" / "golden_builder" / "public" / "xtb_llm_benchmark"
sys.path.insert(0, str(PUBLIC))

from common.geometry import XYZGeometry, heavy_atom_rmsd, interfragment_contact_vector, read_xyz  # noqa: E402
from evaluator.core import job_energy, verify_and_parse_jobs  # noqa: E402
from tools.xtb_run import XTBRunner  # noqa: E402


HARTREE_TO_KCAL_MOL = 627.5094740631
URACIL_ELEMENTS = ["O", "O", "N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]
URACIL_COORDINATES = np.asarray(
    [
        [2.3264, 0.9651, 0.0001],
        [-2.2972, 1.0232, 0.0005],
        [0.0180, 1.0199, -0.0002],
        [1.1637, -1.0221, 0.0001],
        [1.2524, 0.3629, 0.0000],
        [-1.2315, 0.4141, -0.0004],
        [-0.0268, -1.6955, 0.0002],
        [-1.2049, -1.0676, -0.0002],
        [0.0382, 2.0357, -0.0001],
        [2.0187, -1.5702, 0.0004],
        [-2.1443, -1.6063, -0.0002],
        [0.0469, -2.7761, 0.0004],
    ],
    dtype=float,
)


def pubchem_uracil_seed() -> tuple[list[str], np.ndarray]:
    """Return the frozen PubChem CID 1174 conformer in stable atom order."""

    return list(URACIL_ELEMENTS), URACIL_COORDINATES.copy()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("cannot normalize zero vector")
    return np.asarray(vector, dtype=float) / norm


def _rodrigues(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = _unit(axis)
    angle = math.radians(float(degrees))
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=float,
    )
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def _align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source, target = _unit(source), _unit(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if float(np.linalg.norm(cross)) <= 1.0e-12:
        if dot > 0.0:
            return np.eye(3)
        trial = np.asarray([1.0, 0.0, 0.0]) if abs(source[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
        return _rodrigues(np.cross(source, trial), 180.0)
    return _rodrigues(cross, math.degrees(math.acos(dot)))


def _orient_water(
    water: np.ndarray,
    oxygen_target: np.ndarray,
    primary_oh_target: np.ndarray,
    twist_deg: float,
) -> np.ndarray:
    centered = np.asarray(water, dtype=float) - water[0]
    direction = np.asarray(primary_oh_target, dtype=float) - np.asarray(oxygen_target, dtype=float)
    aligned = centered @ _align(centered[1], direction).T
    return aligned @ _rodrigues(direction, twist_deg).T + oxygen_target


def _ring_normal(uracil: np.ndarray) -> np.ndarray:
    return _unit(np.cross(uracil[5] - uracil[4], uracil[6] - uracil[4]))


def _acceptor_pose(
    uracil: np.ndarray,
    water: np.ndarray,
    oxygen_index: int,
    carbon_index: int,
    *,
    lateral_deg: float,
    out_of_plane_deg: float,
    twist_deg: float,
) -> np.ndarray:
    acceptor = uracil[oxygen_index]
    extension = _unit(acceptor - uracil[carbon_index])
    normal = _ring_normal(uracil)
    tangent = _unit(np.cross(normal, extension))
    lateral = math.radians(lateral_deg)
    out = math.radians(out_of_plane_deg)
    planar = math.cos(lateral) * extension + math.sin(lateral) * tangent
    direction = _unit(math.cos(out) * planar + math.sin(out) * normal)
    oxygen_target = acceptor + 2.78 * direction
    return _orient_water(water, oxygen_target, acceptor, twist_deg)


def _donor_pose(
    uracil: np.ndarray,
    water: np.ndarray,
    nitrogen_index: int,
    hydrogen_index: int,
    *,
    offset: float,
    twist_deg: float,
) -> np.ndarray:
    direction = _unit(uracil[hydrogen_index] - uracil[nitrogen_index])
    normal = _ring_normal(uracil)
    oxygen_target = uracil[hydrogen_index] + 1.80 * direction + offset * normal
    centered = water - water[0]
    return centered @ _rodrigues(direction, twist_deg).T + oxygen_target


def _edge_pose(
    uracil: np.ndarray,
    water: np.ndarray,
    anchor_indices: tuple[int, int],
    *,
    normal_sign: float,
    radial_distance: float,
    twist_deg: float,
) -> np.ndarray:
    centroid = uracil[:8].mean(axis=0)
    anchor = uracil[list(anchor_indices)].mean(axis=0)
    radial = _unit(anchor - centroid)
    normal = _ring_normal(uracil)
    oxygen_target = anchor + radial_distance * radial + normal_sign * 0.45 * normal
    return _orient_water(water, oxygen_target, anchor, twist_deg)


def _nonoverlap(uracil: np.ndarray, water: np.ndarray) -> tuple[float, float]:
    distances = np.linalg.norm(uracil[:, None, :] - water[None, :, :], axis=2)
    heavy_a = [index for index, element in enumerate(URACIL_ELEMENTS) if element != "H"]
    minimum = float(np.min(distances))
    minimum_heavy = float(np.min(distances[np.ix_(heavy_a, [0])]))
    return minimum, minimum_heavy


def build_screening_poses(uracil: np.ndarray, water: np.ndarray) -> list[tuple[str, dict[str, Any], np.ndarray]]:
    """Build exactly 14 deterministic, non-overlapping complex seeds."""

    recipes: list[tuple[str, dict[str, Any], np.ndarray]] = []
    for label, oxygen_index, carbon_index, lateral, out, twist in (
        ("UO2_A", 0, 4, -20.0, 0.0, 0.0),
        ("UO2_B", 0, 4, 20.0, 0.0, 180.0),
        ("UO2_C", 0, 4, 0.0, 28.0, 90.0),
        ("UO4_A", 1, 5, -20.0, 0.0, 0.0),
        ("UO4_B", 1, 5, 20.0, 0.0, 180.0),
        ("UO4_C", 1, 5, 0.0, -28.0, 270.0),
    ):
        family = "o2_acceptor" if oxygen_index == 0 else "o4_acceptor"
        placed = _acceptor_pose(
            uracil,
            water,
            oxygen_index,
            carbon_index,
            lateral_deg=lateral,
            out_of_plane_deg=out,
            twist_deg=twist,
        )
        recipes.append((label, {"family": family, "site_atom_id": oxygen_index + 1}, np.vstack([uracil, placed])))
    for label, nitrogen, hydrogen, family, offset, twist in (
        ("UN1_A", 2, 8, "n1h_donor", 0.0, 0.0),
        ("UN1_B", 2, 8, "n1h_donor", 0.12, 180.0),
        ("UN3_A", 3, 9, "n3h_donor", 0.0, 90.0),
        ("UN3_B", 3, 9, "n3h_donor", -0.12, 270.0),
    ):
        placed = _donor_pose(uracil, water, nitrogen, hydrogen, offset=offset, twist_deg=twist)
        recipes.append((label, {"family": family, "site_atom_id": hydrogen + 1}, np.vstack([uracil, placed])))
    for label, anchors, family, sign, distance, twist in (
        ("UEDGE_A", (0, 3), "double_or_edge", 1.0, 2.55, 35.0),
        ("UEDGE_B", (1, 2), "double_or_edge", -1.0, 3.05, 215.0),
        ("UWEAK_A", (6, 7), "weak_edge", 1.0, 3.05, 80.0),
        ("UWEAK_B", (6, 7), "weak_edge", -1.0, 3.05, 260.0),
    ):
        placed = _edge_pose(
            uracil,
            water,
            anchors,
            normal_sign=sign,
            radial_distance=distance,
            twist_deg=twist,
        )
        recipes.append((label, {"family": family, "anchor_atom_ids": [value + 1 for value in anchors]}, np.vstack([uracil, placed])))
    if len(recipes) != 14:
        raise AssertionError("uracil screening recipe must contain exactly 14 poses")
    for label, _, coordinates in recipes:
        minimum, minimum_heavy = _nonoverlap(coordinates[:12], coordinates[12:])
        if minimum < 1.20 or minimum_heavy < 2.20:
            raise ValueError(f"{label} violates non-overlap thresholds: {minimum:.6f}, {minimum_heavy:.6f}")
    return recipes


def _write_xyz(path: Path, elements: list[str], coordinates: np.ndarray, comment: str) -> None:
    rows = [str(len(elements)), comment]
    rows.extend(
        f"{element:<2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for element, xyz in zip(elements, coordinates, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_checked(runner: XTBRunner, request: dict[str, Any]) -> dict[str, Any]:
    result = runner.run(request)
    if result.get("status") != "completed" or result.get("exit_code") != 0:
        raise RuntimeError(f"xTB diagnostic job failed: {result}")
    return result


def _classify_contact(geometry: XYZGeometry) -> tuple[str, dict[str, Any]]:
    xyz = np.asarray(geometry.coordinates, dtype=float)
    if tuple(item.upper() for item in geometry.elements) != tuple(URACIL_ELEMENTS + ["O", "H", "H"]):
        return "invalid_or_collapsed", {"reason": "unexpected atom order"}
    contacts: list[dict[str, Any]] = []
    for acceptor_index, label in ((0, "o2_acceptor"), (1, "o4_acceptor")):
        for water_hydrogen in (13, 14):
            distance = float(np.linalg.norm(xyz[water_hydrogen] - xyz[acceptor_index]))
            if distance <= 2.60:
                contacts.append({"family": label, "distance_angstrom": distance})
    for nitrogen, hydrogen, label in ((2, 8, "n1h_donor"), (3, 9, "n3h_donor")):
        distance = float(np.linalg.norm(xyz[hydrogen] - xyz[12]))
        if distance <= 2.60:
            contacts.append({"family": label, "distance_angstrom": distance})
    families = {item["family"] for item in contacts}
    if len(families) >= 2:
        family = "double_or_edge"
    elif families:
        family = next(iter(families))
    else:
        nearest = float(np.min(np.linalg.norm(xyz[:12, None, :] - xyz[12:, :][None, :, :], axis=2)))
        family = "weak_edge" if nearest <= 4.5 else "dissociated"
    return family, {"contacts": contacts}


def _competition_present(clusters: list[dict[str, Any]]) -> bool:
    for left_index, left in enumerate(clusters):
        for right in clusters[left_index + 1 :]:
            interaction_delta = float(left["interaction_kcal_mol"]) - float(right["interaction_kcal_mol"])
            deformation_delta = float(left["deformation_kcal_mol"]) - float(right["deformation_kcal_mol"])
            if interaction_delta * deformation_delta < 0.0 and min(abs(interaction_delta), abs(deformation_delta)) >= 0.10:
                return True
    return False


def run_screen(xtb: str, output: Path, *, fresh: bool) -> dict[str, Any]:
    output = output.resolve()
    if fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)

    elements_a, uracil = pubchem_uracil_seed()
    water_source = PACKAGE_ROOT / "tasks" / "task_05" / "input" / "original" / "data" / "inputs" / "monomer_B.xyz"
    water_geometry = read_xyz(water_source)
    elements_b = list(water_geometry.elements)
    water = np.asarray(water_geometry.coordinates, dtype=float)
    poses = build_screening_poses(uracil, water)
    _write_xyz(inputs / "monomer_A.xyz", elements_a, uracil, "Uracil CID 1174; private scientific screen")
    shutil.copy2(water_source, inputs / "monomer_B.xyz")
    for label, recipe, coordinates in poses:
        _write_xyz(inputs / f"{label}.xyz", elements_a + elements_b, coordinates, f"private screening pose {label}; {recipe['family']}")

    task_dir = output / "diagnostic_task_definition"
    shutil.copytree(PUBLIC / "tasks" / "task_05_binding_decomposition", task_dir)
    config_path = task_dir / "config" / "task.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["status"] = "private_scientific_diagnostic_only"
    config["budgets"]["max_xtb_calls"] = 96
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    runner = XTBRunner(task_dir=task_dir, run_root=output, xtb_executable=xtb)
    optimized: list[dict[str, Any]] = []
    for index, (label, recipe, _) in enumerate(poses, start=1):
        response = _run_checked(
            runner,
            {
                "geometry": f"inputs/{label}.xyz",
                "arguments": ["--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "tight"],
                "namespace": f"screen_pose_{index:02d}",
                "timeout_seconds": 900,
            },
        )
        optimized.append({"label": label, "recipe": recipe, "job_id": response["job_id"], "workdir": response["workdir"]})

    jobs, errors, _ = verify_and_parse_jobs(output)
    if errors:
        raise RuntimeError(f"optimization evidence failed verification: {errors}")
    for item in optimized:
        job = jobs[item["job_id"]]
        item["electronic_energy_hartree"] = job_energy(job, "total_energy_hartree")
        item["geometry"] = read_xyz(output / item["workdir"] / "xtbopt.xyz")
        item["post_opt_contact_family"], item["contact_metrics"] = _classify_contact(item["geometry"])

    fragment_a_ids = list(range(1, 13))
    fragment_b_ids = [13, 14, 15]
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(optimized, key=lambda row: (row["electronic_energy_hartree"], row["label"])):
        for group in groups:
            representative = group[0]
            left = interfragment_contact_vector(representative["geometry"], fragment_a_ids, fragment_b_ids)
            right = interfragment_contact_vector(item["geometry"], fragment_a_ids, fragment_b_ids)
            contact_error = float(np.max(np.abs(left - right)))
            if heavy_atom_rmsd(representative["geometry"], item["geometry"]) <= 0.25 and contact_error <= 0.15:
                group.append(item)
                break
        else:
            groups.append([item])

    monomer_jobs: dict[str, str] = {}
    for fragment in ("A", "B"):
        response = _run_checked(
            runner,
            {
                "geometry": f"inputs/monomer_{fragment}.xyz",
                "arguments": ["--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "tight"],
                "namespace": f"screen_monomer_{fragment.lower()}",
                "timeout_seconds": 900,
            },
        )
        monomer_jobs[fragment] = response["job_id"]

    representatives = [min(group, key=lambda row: (row["electronic_energy_hartree"], row["label"])) for group in groups]
    for index, item in enumerate(representatives, start=1):
        geometry = item["geometry"]
        paths = {"A": inputs / f"cluster_{index:02d}_A.xyz", "B": inputs / f"cluster_{index:02d}_B.xyz"}
        _write_xyz(paths["A"], elements_a, geometry.coordinates[:12], f"frozen uracil from {item['label']}")
        _write_xyz(paths["B"], elements_b, geometry.coordinates[12:], f"frozen water from {item['label']}")
        for fragment in ("A", "B"):
            response = _run_checked(
                runner,
                {
                    "geometry": f"inputs/{paths[fragment].name}",
                    "arguments": ["--gfn", "2", "--chrg", "0", "--uhf", "0", "--sp"],
                    "namespace": f"screen_cluster_{index:02d}_{fragment.lower()}",
                    "timeout_seconds": 900,
                },
            )
            item[f"fragment_{fragment.lower()}_job_id"] = response["job_id"]

    jobs, errors, _ = verify_and_parse_jobs(output)
    if errors:
        raise RuntimeError(f"decomposition evidence failed verification: {errors}")
    monomer_a = job_energy(jobs[monomer_jobs["A"]], "total_energy_hartree")
    monomer_b = job_energy(jobs[monomer_jobs["B"]], "total_energy_hartree")
    clusters: list[dict[str, Any]] = []
    for index, (group, representative) in enumerate(zip(groups, representatives, strict=True), start=1):
        complex_energy = float(representative["electronic_energy_hartree"])
        frozen_a = job_energy(jobs[representative["fragment_a_job_id"]], "total_energy_hartree")
        frozen_b = job_energy(jobs[representative["fragment_b_job_id"]], "total_energy_hartree")
        interaction = (complex_energy - frozen_a - frozen_b) * HARTREE_TO_KCAL_MOL
        deformation = ((frozen_a - monomer_a) + (frozen_b - monomer_b)) * HARTREE_TO_KCAL_MOL
        binding = (complex_energy - monomer_a - monomer_b) * HARTREE_TO_KCAL_MOL
        closure_error = binding - interaction - deformation
        clusters.append(
            {
                "cluster_id": f"DIAG_CLUSTER_{index:02d}",
                "representative": representative["label"],
                "members": sorted(item["label"] for item in group),
                "post_opt_contact_family": representative["post_opt_contact_family"],
                "interaction_kcal_mol": interaction,
                "deformation_kcal_mol": deformation,
                "binding_kcal_mol": binding,
                "closure_error_kcal_mol": closure_error,
                "evidence_job_ids": [
                    representative["job_id"],
                    representative["fragment_a_job_id"],
                    representative["fragment_b_job_id"],
                ],
            }
        )
    clusters.sort(key=lambda row: (row["binding_kcal_mol"], row["representative"]))
    leading_gap = abs(float(clusters[1]["binding_kcal_mol"]) - float(clusters[0]["binding_kcal_mol"])) if len(clusters) >= 2 else math.inf
    families = {str(cluster["post_opt_contact_family"]) for cluster in clusters}
    assertions = {
        "fourteen_nonoverlapping_inputs": len(poses) == 14,
        "at_least_three_independent_minima": len(clusters) >= 3,
        "carbonyl_and_donor_or_edge_survive": bool(families & {"o2_acceptor", "o4_acceptor"}) and bool(families & {"n1h_donor", "n3h_donor", "double_or_edge", "weak_edge"}),
        "post_optimization_pose_collapse": any(len(group) >= 2 for group in groups),
        "weak_pose_collapses_or_noncompetitive": all(
            next(item for item in optimized if item["label"] == label)["post_opt_contact_family"] != "weak_edge"
            or next(item for item in optimized if item["label"] == label)["electronic_energy_hartree"] > min(row["electronic_energy_hartree"] for row in optimized) + 0.001
            for label in ("UWEAK_A", "UWEAK_B")
        ),
        "near_degenerate_retained_pair": leading_gap <= 0.10,
        "interaction_deformation_competition": _competition_present(clusters[:4]),
        "closure_consistent": all(abs(float(item["closure_error_kcal_mol"])) <= 1.0e-6 for item in clusters),
    }
    raw_hashes = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted((output / "xtb_jobs").rglob("*"))
        if path.is_file()
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "purpose": "private scientific system-selection diagnostic; not model-visible reference truth",
        "source": {
            "uracil": "PubChem CID 1174",
            "uracil_url": "https://pubchem.ncbi.nlm.nih.gov/compound/1174",
            "water": "PubChem CID 962",
        },
        "method": "gas-phase GFN2-xTB; tight optimization; exact frozen-fragment single points",
        "sampled_pose_count": len(poses),
        "independent_cluster_count": len(clusters),
        "leading_binding_gap_kcal_mol": leading_gap,
        "assertions": assertions,
        "accepted": all(assertions.values()),
        "clusters": clusters,
        "raw_evidence_sha256": raw_hashes,
    }
    (output / "diagnostic_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xtb", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_screen(args.xtb, args.output, fresh=args.fresh)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

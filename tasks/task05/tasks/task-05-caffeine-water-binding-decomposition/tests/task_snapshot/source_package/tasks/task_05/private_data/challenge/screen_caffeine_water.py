#!/usr/bin/env python3
"""Private real-xTB suitability screen for caffeine--water."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


_HELPER_SPEC = importlib.util.spec_from_file_location("water_screen_helpers", HERE / "screen_uracil_water.py")
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError("cannot load shared private screening helpers")
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)

HARTREE_TO_KCAL_MOL = 627.5094740631
CAFFEINE_ELEMENTS = ["O", "O", "N", "N", "N", "N", "C", "C", "C", "C", "C", "C", "C", "C"] + ["H"] * 10
CAFFEINE_COORDINATES = np.asarray(
    [
        [0.4700, 2.5688, 0.0006], [-3.1271, -0.4436, -0.0003],
        [-0.9686, -1.3125, 0.0000], [2.2182, 0.1412, -0.0003],
        [-1.3477, 1.0797, -0.0001], [1.4119, -1.9372, 0.0002],
        [0.8579, 0.2592, -0.0008], [0.3897, -1.0264, -0.0004],
        [0.0307, 1.4220, -0.0006], [-1.9061, -0.2495, -0.0004],
        [2.5032, -1.1998, 0.0003], [-1.4276, -2.6960, 0.0008],
        [3.1926, 1.2061, 0.0003], [-2.2969, 2.1881, 0.0007],
        [3.5163, -1.5787, 0.0008], [-1.0451, -3.1973, -0.8937],
        [-2.5186, -2.7596, 0.0011], [-1.0447, -3.1963, 0.8957],
        [4.1992, 0.7801, 0.0002], [3.0468, 1.8092, -0.8992],
        [3.0466, 1.8083, 0.9004], [-1.8087, 3.1651, -0.0003],
        [-2.9322, 2.1027, 0.8881], [-2.9346, 2.1021, -0.8849],
    ],
    dtype=float,
)


def pubchem_caffeine_seed() -> tuple[list[str], np.ndarray]:
    return list(CAFFEINE_ELEMENTS), CAFFEINE_COORDINATES.copy()


def _normal(caffeine: np.ndarray) -> np.ndarray:
    return _HELPERS._unit(np.cross(caffeine[7] - caffeine[6], caffeine[8] - caffeine[6]))


def _acceptor_pose(
    caffeine: np.ndarray,
    water: np.ndarray,
    oxygen_index: int,
    carbon_index: int,
    lateral_deg: float,
    out_of_plane_deg: float,
    twist_deg: float,
) -> np.ndarray:
    acceptor = caffeine[oxygen_index]
    extension = _HELPERS._unit(acceptor - caffeine[carbon_index])
    normal = _normal(caffeine)
    tangent = _HELPERS._unit(np.cross(normal, extension))
    lateral, out = math.radians(lateral_deg), math.radians(out_of_plane_deg)
    planar = math.cos(lateral) * extension + math.sin(lateral) * tangent
    direction = _HELPERS._unit(math.cos(out) * planar + math.sin(out) * normal)
    water_oxygen = acceptor + 2.78 * direction
    return _HELPERS._orient_water(water, water_oxygen, acceptor, twist_deg)


def build_screening_poses(caffeine: np.ndarray, water: np.ndarray) -> list[tuple[str, dict[str, Any], np.ndarray]]:
    recipes: list[tuple[str, dict[str, Any], np.ndarray]] = []
    for label, oxygen, carbon, family, lateral, out, twist in (
        ("CO1_A", 0, 8, "carbonyl_o1", -20.0, 0.0, 0.0),
        ("CO1_B", 0, 8, "carbonyl_o1", 20.0, 0.0, 180.0),
        ("CO1_C", 0, 8, "carbonyl_o1", 0.0, 28.0, 90.0),
        ("CO1_D", 0, 8, "carbonyl_o1", 0.0, -28.0, 270.0),
        ("CO2_A", 1, 9, "carbonyl_o2", -20.0, 0.0, 0.0),
        ("CO2_B", 1, 9, "carbonyl_o2", 20.0, 0.0, 180.0),
        ("CO2_C", 1, 9, "carbonyl_o2", 0.0, 28.0, 90.0),
        ("CO2_D", 1, 9, "carbonyl_o2", 0.0, -28.0, 270.0),
        ("BENT1_A", 0, 8, "bent_edge", -55.0, 20.0, 45.0),
        ("BENT1_B", 0, 8, "bent_edge", 55.0, -20.0, 225.0),
        ("BENT2_A", 1, 9, "bent_edge", -55.0, -20.0, 135.0),
        ("BENT2_B", 1, 9, "bent_edge", 55.0, 20.0, 315.0),
    ):
        placed = _acceptor_pose(caffeine, water, oxygen, carbon, lateral, out, twist)
        recipes.append((label, {"family": family, "site_atom_id": oxygen + 1}, np.vstack([caffeine, placed])))
    centroid = caffeine[2:11].mean(axis=0)
    normal = _normal(caffeine)
    for label, sign, twist in (("PI_A", 1.0, 30.0), ("PI_B", -1.0, 210.0)):
        oxygen_target = centroid + 3.05 * sign * normal
        placed = _HELPERS._orient_water(water, oxygen_target, centroid, twist)
        recipes.append((label, {"family": "non_hbond_close_contact"}, np.vstack([caffeine, placed])))
    if len(recipes) != 14:
        raise AssertionError("caffeine screen requires exactly 14 poses")
    heavy_a = [index for index, element in enumerate(CAFFEINE_ELEMENTS) if element != "H"]
    for label, _, coordinates in recipes:
        a, b = coordinates[:24], coordinates[24:]
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        minimum = float(np.min(distances))
        minimum_heavy = float(np.min(distances[np.ix_(heavy_a, [0])]))
        if minimum < 1.20 or minimum_heavy < 2.20:
            raise ValueError(f"{label} violates non-overlap thresholds: {minimum:.6f}, {minimum_heavy:.6f}")
    return recipes


def _classify(geometry: XYZGeometry) -> tuple[str, dict[str, Any]]:
    if tuple(item.upper() for item in geometry.elements) != tuple(CAFFEINE_ELEMENTS + ["O", "H", "H"]):
        return "invalid_or_collapsed", {"reason": "unexpected atom order"}
    xyz = np.asarray(geometry.coordinates, dtype=float)
    contacts: dict[str, list[float]] = {"carbonyl_o1": [], "carbonyl_o2": []}
    for oxygen, family in ((0, "carbonyl_o1"), (1, "carbonyl_o2")):
        for hydrogen in (25, 26):
            distance = float(np.linalg.norm(xyz[hydrogen] - xyz[oxygen]))
            if distance <= 2.60:
                contacts[family].append(distance)
    active = [family for family, values in contacts.items() if values]
    if len(active) > 1:
        family = "bent_edge"
    elif active:
        family = active[0]
    else:
        nearest = float(np.min(np.linalg.norm(xyz[:24, None, :] - xyz[24:, :][None, :, :], axis=2)))
        family = "non_hbond_close_contact" if nearest <= 4.5 else "dissociated"
    return family, {"hydrogen_bond_distances_angstrom": contacts}


def _run_checked(runner: XTBRunner, request: dict[str, Any]) -> dict[str, Any]:
    result = runner.run(request)
    if result.get("status") != "completed" or result.get("exit_code") != 0:
        raise RuntimeError(f"xTB diagnostic job failed: {result}")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _competition_present(clusters: list[dict[str, Any]]) -> bool:
    for index, left in enumerate(clusters):
        for right in clusters[index + 1 :]:
            interaction_delta = float(left["interaction_kcal_mol"]) - float(right["interaction_kcal_mol"])
            deformation_delta = float(left["deformation_kcal_mol"]) - float(right["deformation_kcal_mol"])
            if interaction_delta * deformation_delta < 0.0 and min(abs(interaction_delta), abs(deformation_delta)) >= 0.10:
                return True
    return False


def weak_seeds_collapse_or_are_noncompetitive(
    optimized: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    *,
    weak_labels: tuple[str, ...],
    noncompetitive_gap_kcal_mol: float,
) -> bool:
    """Accept a weak seed only if it changes family or remains safely above best."""

    if not clusters:
        return False
    best = min(float(cluster["binding_kcal_mol"]) for cluster in clusters)
    optimized_by_label = {str(row["label"]): row for row in optimized}
    cluster_by_member = {
        str(member): cluster
        for cluster in clusters
        for member in cluster.get("members", [])
    }
    for label in weak_labels:
        optimized_row = optimized_by_label.get(label)
        cluster = cluster_by_member.get(label)
        if optimized_row is None or cluster is None:
            return False
        collapsed_to_strong_family = str(optimized_row.get("post_opt_contact_family")) != "non_hbond_close_contact"
        noncompetitive = float(cluster["binding_kcal_mol"]) >= best + float(noncompetitive_gap_kcal_mol)
        if not (collapsed_to_strong_family or noncompetitive):
            return False
    return True


def run_screen(xtb: str, output: Path, *, fresh: bool) -> dict[str, Any]:
    output = output.resolve()
    if fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    elements_a, caffeine = pubchem_caffeine_seed()
    water_source = PACKAGE_ROOT / "tasks" / "task_05" / "input" / "original" / "data" / "inputs" / "monomer_B.xyz"
    water_geometry = read_xyz(water_source)
    elements_b, water = list(water_geometry.elements), np.asarray(water_geometry.coordinates, dtype=float)
    poses = build_screening_poses(caffeine, water)
    _HELPERS._write_xyz(inputs / "monomer_A.xyz", elements_a, caffeine, "Caffeine CID 2519; private scientific screen")
    shutil.copy2(water_source, inputs / "monomer_B.xyz")
    for label, recipe, coordinates in poses:
        _HELPERS._write_xyz(inputs / f"{label}.xyz", elements_a + elements_b, coordinates, f"private pose {label}; {recipe['family']}")

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
        response = _run_checked(runner, {"geometry": f"inputs/{label}.xyz", "arguments": ["--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "tight"], "namespace": f"screen_pose_{index:02d}", "timeout_seconds": 900})
        optimized.append({"label": label, "recipe": recipe, "job_id": response["job_id"], "workdir": response["workdir"]})
    jobs, errors, _ = verify_and_parse_jobs(output)
    if errors:
        raise RuntimeError(f"optimization evidence failed verification: {errors}")
    for item in optimized:
        item["electronic_energy_hartree"] = job_energy(jobs[item["job_id"]], "total_energy_hartree")
        item["geometry"] = read_xyz(output / item["workdir"] / "xtbopt.xyz")
        item["post_opt_contact_family"], item["contact_metrics"] = _classify(item["geometry"])

    fragment_a_ids, fragment_b_ids = list(range(1, 25)), [25, 26, 27]
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(optimized, key=lambda row: (row["electronic_energy_hartree"], row["label"])):
        for group in groups:
            representative = group[0]
            left = interfragment_contact_vector(representative["geometry"], fragment_a_ids, fragment_b_ids)
            right = interfragment_contact_vector(item["geometry"], fragment_a_ids, fragment_b_ids)
            if heavy_atom_rmsd(representative["geometry"], item["geometry"]) <= 0.25 and float(np.max(np.abs(left - right))) <= 0.15:
                group.append(item)
                break
        else:
            groups.append([item])

    monomer_job_ids: dict[str, str] = {}
    for fragment in ("A", "B"):
        response = _run_checked(runner, {"geometry": f"inputs/monomer_{fragment}.xyz", "arguments": ["--gfn", "2", "--chrg", "0", "--uhf", "0", "--opt", "tight"], "namespace": f"screen_monomer_{fragment.lower()}", "timeout_seconds": 900})
        monomer_job_ids[fragment] = response["job_id"]

    representatives = [min(group, key=lambda row: (row["electronic_energy_hartree"], row["label"])) for group in groups]
    for index, representative in enumerate(representatives, start=1):
        geometry = representative["geometry"]
        _HELPERS._write_xyz(inputs / f"cluster_{index:02d}_A.xyz", elements_a, geometry.coordinates[:24], f"frozen caffeine from {representative['label']}")
        _HELPERS._write_xyz(inputs / f"cluster_{index:02d}_B.xyz", elements_b, geometry.coordinates[24:], f"frozen water from {representative['label']}")
        for fragment in ("A", "B"):
            response = _run_checked(runner, {"geometry": f"inputs/cluster_{index:02d}_{fragment}.xyz", "arguments": ["--gfn", "2", "--chrg", "0", "--uhf", "0", "--sp"], "namespace": f"screen_cluster_{index:02d}_{fragment.lower()}", "timeout_seconds": 900})
            representative[f"fragment_{fragment.lower()}_job_id"] = response["job_id"]

    jobs, errors, _ = verify_and_parse_jobs(output)
    if errors:
        raise RuntimeError(f"decomposition evidence failed verification: {errors}")
    monomer_a = job_energy(jobs[monomer_job_ids["A"]], "total_energy_hartree")
    monomer_b = job_energy(jobs[monomer_job_ids["B"]], "total_energy_hartree")
    clusters: list[dict[str, Any]] = []
    for index, (group, representative) in enumerate(zip(groups, representatives, strict=True), start=1):
        complex_energy = float(representative["electronic_energy_hartree"])
        frozen_a = job_energy(jobs[representative["fragment_a_job_id"]], "total_energy_hartree")
        frozen_b = job_energy(jobs[representative["fragment_b_job_id"]], "total_energy_hartree")
        interaction = (complex_energy - frozen_a - frozen_b) * HARTREE_TO_KCAL_MOL
        deformation = ((frozen_a - monomer_a) + (frozen_b - monomer_b)) * HARTREE_TO_KCAL_MOL
        binding = (complex_energy - monomer_a - monomer_b) * HARTREE_TO_KCAL_MOL
        clusters.append({"cluster_id": f"DIAG_CLUSTER_{index:02d}", "representative": representative["label"], "members": sorted(row["label"] for row in group), "post_opt_contact_family": representative["post_opt_contact_family"], "interaction_kcal_mol": interaction, "deformation_kcal_mol": deformation, "binding_kcal_mol": binding, "closure_error_kcal_mol": binding - interaction - deformation, "evidence_job_ids": [representative["job_id"], representative["fragment_a_job_id"], representative["fragment_b_job_id"]]})
    clusters.sort(key=lambda row: (row["binding_kcal_mol"], row["representative"]))
    leading_gap = abs(float(clusters[1]["binding_kcal_mol"]) - float(clusters[0]["binding_kcal_mol"])) if len(clusters) >= 2 else math.inf
    families = {str(row["post_opt_contact_family"]) for row in clusters}
    assertions = {
        "fourteen_nonoverlapping_inputs": len(poses) == 14,
        "at_least_three_independent_minima": len(clusters) >= 3,
        "both_carbonyl_families_survive": {"carbonyl_o1", "carbonyl_o2"}.issubset(families),
        "post_optimization_pose_collapse": any(len(group) >= 2 for group in groups),
        "weak_pose_collapses_or_noncompetitive": weak_seeds_collapse_or_are_noncompetitive(
            optimized,
            clusters,
            weak_labels=("PI_A", "PI_B"),
            noncompetitive_gap_kcal_mol=0.25,
        ),
        "near_degenerate_retained_pair": leading_gap <= 0.10,
        "interaction_deformation_competition": _competition_present(clusters[:4]),
        "closure_consistent": all(abs(float(row["closure_error_kcal_mol"])) <= 1.0e-6 for row in clusters),
    }
    payload = {
        "schema_version": "1.0.0",
        "purpose": "private scientific system-selection diagnostic; not model-visible reference truth",
        "source": {"caffeine": "PubChem CID 2519", "caffeine_url": "https://pubchem.ncbi.nlm.nih.gov/compound/2519", "water": "PubChem CID 962"},
        "method": "gas-phase GFN2-xTB; tight optimization; exact frozen-fragment single points",
        "sampled_pose_count": len(poses),
        "independent_cluster_count": len(clusters),
        "leading_binding_gap_kcal_mol": leading_gap,
        "assertions": assertions,
        "accepted": all(assertions.values()),
        "clusters": clusters,
        "raw_evidence_sha256": {path.relative_to(output).as_posix(): _sha256(path) for path in sorted((output / "xtb_jobs").rglob("*")) if path.is_file()},
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
    print(json.dumps({key: value for key, value in result.items() if key != "raw_evidence_sha256"}, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic Task 05 evaluator."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from common.geometry import XYZGeometry, geometry_from_record, heavy_atom_rmsd, interfragment_contact_vector, read_xyz
from common.allocation import standalone_hessian_parent
from common.metrics import HARTREE_TO_KCAL_MOL, kendall_tau_b_with_reference_ties, pairwise_cluster_f1
from common.scientific_interpretation import quantitatively_grounded_claim
from evaluator.core import (
    ScoreBuilder,
    job_energy,
    job_no_significant_imaginary,
    job_valid_termination,
    is_deliberate_protocol_failure,
    load_json_required,
    load_json_optional,
    load_json_value_optional,
    load_yaml_optional,
    manifest_execution_identity_audit,
    mean_absolute_error,
    normalize_recovery_log,
    protocol_matches,
    read_csv_rows,
    reference_manifest,
    required_output_status,
    text_contains_affirmative_claim,
    validate_final_answer,
    validate_json_against_schema,
    verify_and_parse_jobs,
)


def binding_ranking_uncertainty(binding_kcal_mol: dict[str, float], perturbation_kcal_mol: float = 0.25) -> dict[str, Any]:
    """Return conservative per-pose intervals and whether the binding winner is stable."""
    if not binding_kcal_mol:
        return {
            "perturbation_kcal_mol": float(perturbation_kcal_mol),
            "leading_pose_ids": [],
            "winner_stable": False,
            "per_pose_binding_ranges_kcal_mol": {},
            "tie_or_stable_winner": "uncertain",
        }
    ordered = sorted(binding_kcal_mol, key=lambda key: (float(binding_kcal_mol[key]), key))
    intervals = {
        key: {
            "minimum": float(binding_kcal_mol[key]) - float(perturbation_kcal_mol),
            "maximum": float(binding_kcal_mol[key]) + float(perturbation_kcal_mol),
        }
        for key in ordered
    }
    best = ordered[0]
    leading = [key for key in ordered if intervals[key]["minimum"] <= intervals[best]["maximum"] + 1e-12]
    stable = all(intervals[best]["maximum"] < intervals[key]["minimum"] - 1e-12 for key in ordered[1:])
    return {
        "perturbation_kcal_mol": float(perturbation_kcal_mol),
        "leading_pose_ids": leading,
        "winner_stable": stable,
        "per_pose_binding_ranges_kcal_mol": intervals,
        "tie_or_stable_winner": "stable_winner" if stable else ("tie" if len(leading) > 1 else "uncertain"),
    }


def _angle_degrees(first: Any, vertex: Any, third: Any) -> float:
    """Return the first-vertex-third angle without importing chemistry toolkits."""
    import numpy as np

    left = np.asarray(first, dtype=float) - np.asarray(vertex, dtype=float)
    right = np.asarray(third, dtype=float) - np.asarray(vertex, dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return math.nan
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def classify_caffeine_water_contact(geometry: XYZGeometry) -> tuple[str, dict[str, Any]]:
    """Classify optimized caffeine--water contacts from frozen atom IDs.

    Caffeine atoms are 1--24, its two carbonyl oxygens are atoms 1 and 2,
    and water O/H/H are atoms 25--27.  Hydrogen bonds use an H...acceptor
    cutoff of 2.60 Angstrom and a donor-H...acceptor angle of at least 120
    degrees.  Initial placement labels are deliberately ignored.
    """
    import numpy as np

    if geometry is None or len(geometry.elements) != 27:
        return "invalid_or_collapsed", {"reason": "unexpected atom count"}
    expected = ("O", "O", "N", "N", "N", "N", "C", "C", "C", "C", "C", "C", "C", "C") + ("H",) * 10 + ("O", "H", "H")
    if tuple(value.upper() for value in geometry.elements) != expected:
        return "invalid_or_collapsed", {"reason": "unexpected atom order"}
    xyz = np.asarray(geometry.coordinates, dtype=float)
    a_indices = np.arange(0, 24)
    b_indices = np.arange(24, 27)
    cross = np.linalg.norm(xyz[a_indices, None, :] - xyz[b_indices][None, :, :], axis=2)
    nearest_cross = float(np.min(cross))
    if nearest_cross < 0.70:
        return "invalid_or_collapsed", {"reason": "interfragment overlap", "nearest_cross_distance_angstrom": nearest_cross}

    contacts: dict[str, list[dict[str, float | int]]] = {"carbonyl_o1": [], "carbonyl_o2": []}
    for acceptor_index, family in ((0, "carbonyl_o1"), (1, "carbonyl_o2")):
        for hydrogen_index in (25, 26):
            distance = float(np.linalg.norm(xyz[hydrogen_index] - xyz[acceptor_index]))
            angle = _angle_degrees(xyz[24], xyz[hydrogen_index], xyz[acceptor_index])
            if distance <= 2.60 and angle >= 120.0:
                contacts[family].append({"water_h_atom_id": hydrogen_index + 1, "h_acceptor_distance_angstrom": distance, "dha_angle_deg": angle})

    metrics: dict[str, Any] = {
        "nearest_cross_distance_angstrom": nearest_cross,
        "water_to_o1_hbond_count": len(contacts["carbonyl_o1"]),
        "water_to_o2_hbond_count": len(contacts["carbonyl_o2"]),
        "water_to_o1_contacts": contacts["carbonyl_o1"],
        "water_to_o2_contacts": contacts["carbonyl_o2"],
    }
    if contacts["carbonyl_o1"] and contacts["carbonyl_o2"]:
        return "bent_edge", metrics
    if contacts["carbonyl_o1"]:
        return "carbonyl_o1", metrics
    if contacts["carbonyl_o2"]:
        return "carbonyl_o2", metrics
    if nearest_cross <= 4.5:
        return "non_hbond_close_contact", metrics
    return "other", metrics


def _reference_data(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("reference")
    return value if isinstance(value, dict) else manifest


def _clusters(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    if isinstance(value, dict):
        for key in ("clusters", "pose_clusters", "unique_poses"):
            entries = value.get(key)
            if isinstance(entries, list):
                return [entry for entry in entries if isinstance(entry, dict)]
    return []


def _members(cluster: dict[str, Any]) -> list[str]:
    value = cluster.get("member_seed_ids") or cluster.get("members") or []
    return [str(item) for item in value] if isinstance(value, list) else []


def _reference_pose_id(entry: dict[str, Any], references: list[dict[str, Any]]) -> str | None:
    """Align an agent-defined cluster label to reference identity by seed membership."""
    members = frozenset(_members(entry))
    if not members:
        return None
    matches = [str(reference.get("pose_id") or "") for reference in references if frozenset(_members(reference)) == members]
    matches = [value for value in matches if value]
    return matches[0] if len(matches) == 1 else None


def _role_name(row: dict[str, str]) -> str:
    return str(row.get("role") or row.get("energy_role") or row.get("calculation_role") or "").strip().lower()


def _job_id(row: dict[str, str] | None) -> str:
    return str((row or {}).get("job_id") or (row or {}).get("source_job_id") or "").strip()


def _role_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        pose = str(row.get("pose_id") or row.get("complex_id") or row.get("seed_id") or "").strip()
        role = _role_name(row)
        if role:
            result[(pose, role)] = row
    return result


def _find_role(index: dict[tuple[str, str], dict[str, str]], pose_id: str, aliases: set[str]) -> dict[str, str] | None:
    for alias in aliases:
        if (pose_id, alias) in index:
            return index[(pose_id, alias)]
        if ("", alias) in index:
            return index[("", alias)]
    # Relaxed monomer roles are global. Accept a non-empty descriptive pose_id
    # (for example ``monomer_A``) only when the role is globally unambiguous.
    if not pose_id:
        matches = [row for (row_pose, role), row in index.items() if role in aliases]
        if len(matches) == 1:
            return matches[0]
    return None


def _public_fragment_mapping(rows: list[dict[str, str]], pose_id: str, fragment: str) -> list[tuple[int, int, str]]:
    mapping: list[tuple[int, int, str]] = []
    for row in rows:
        row_pose = str(row.get("pose_id") or row.get("complex_id") or "")
        if row_pose and row_pose != pose_id:
            continue
        row_fragment = str(row.get("fragment") or row.get("fragment_id") or "").upper()
        if row_fragment != fragment.upper():
            continue
        try:
            complex_atom = int(row.get("complex_atom_id") or row.get("atom_id") or 0)
            fragment_atom = int(row.get("fragment_atom_id") or row.get("monomer_atom_id") or 0)
        except ValueError:
            continue
        mapping.append((complex_atom, fragment_atom, str(row.get("element") or "")))
    return sorted(mapping, key=lambda item: item[1])


def _input_geometry(job: Any) -> Any:
    request = load_json_required(job.workdir / "request.json")
    copied = request.get("geometry_job_copy")
    if not isinstance(copied, str):
        return None
    path = job.workdir / copied
    if not path.exists():
        return None
    try:
        return read_xyz(path)
    except ValueError:
        return None


def _fragment_geometry_error(complex_job: Any, fragment_job: Any, mapping: list[tuple[int, int, str]]) -> float:
    complex_geometry = geometry_from_record(complex_job.parsed.get("optimized_geometry")) if complex_job else None
    if complex_geometry is None and complex_job is not None:
        complex_geometry = _input_geometry(complex_job)
    if complex_geometry is None or not mapping:
        return math.inf
    try:
        complex_coords = complex_geometry.coordinates
        complex_elements = complex_geometry.elements
        fragment_geometry = _input_geometry(fragment_job)
        if fragment_geometry is None:
            return math.inf
        ordered_coords = []
        ordered_elements = []
        for complex_atom, fragment_atom, expected_element in mapping:
            if fragment_atom != len(ordered_coords) + 1:
                return math.inf
            ordered_coords.append(complex_coords[complex_atom - 1])
            element = str(complex_elements[complex_atom - 1])
            if expected_element and element.upper() != expected_element.upper():
                return math.inf
            ordered_elements.append(element)
        if tuple(element.upper() for element in ordered_elements) != tuple(element.upper() for element in fragment_geometry.elements):
            return math.inf
        # The SP input must be the literal extracted subset, not merely an
        # internally congruent fragment supplied in an unrelated frame.
        import numpy as np
        return float(np.max(np.abs(np.asarray(ordered_coords, dtype=float) - fragment_geometry.coordinates)))
    except (KeyError, IndexError, TypeError, ValueError):
        return math.inf


def _complex_energy_geometry_matches_representative(complex_job: Any, representative_job: Any) -> bool:
    if complex_job is None or representative_job is None:
        return False
    if complex_job.job_id == representative_job.job_id:
        return True
    representative = geometry_from_record(representative_job.parsed.get("optimized_geometry"))
    energy_geometry = geometry_from_record(complex_job.parsed.get("optimized_geometry")) or _input_geometry(complex_job)
    if representative is None or energy_geometry is None:
        return False
    if tuple(value.upper() for value in representative.elements) != tuple(value.upper() for value in energy_geometry.elements):
        return False
    import numpy as np
    return bool(np.max(np.abs(representative.coordinates - energy_geometry.coordinates)) <= 1e-6)


def _job_geometry(job: Any) -> Any:
    if job is None:
        return None
    return geometry_from_record(job.parsed.get("optimized_geometry")) or _input_geometry(job)


def _same_ordered_geometry(first: Any, second: Any, tolerance: float = 1e-6) -> bool:
    if first is None or second is None:
        return False
    if tuple(value.upper() for value in first.elements) != tuple(value.upper() for value in second.elements):
        return False
    import numpy as np
    return bool(np.max(np.abs(first.coordinates - second.coordinates)) <= tolerance)


def _has_valid_hessian_for_geometry(job: Any, jobs: dict[str, Any]) -> bool:
    """Require an earlier optimization and a separate hash-linked Hessian."""
    if job is None:
        return False
    if job_no_significant_imaginary(job) and standalone_hessian_parent(job, jobs) is not None:
        return True
    geometry = _job_geometry(job)
    return any(
        job_no_significant_imaginary(candidate)
        and standalone_hessian_parent(candidate, jobs) == job.job_id
        and _same_ordered_geometry(geometry, _job_geometry(candidate))
        for candidate in jobs.values()
    )


def _origin_geometry_name(job: Any, jobs: dict[str, Any]) -> str:
    """Resolve a retry chain to the original public input geometry."""
    current = job
    visited: set[str] = set()
    while current is not None and current.job_id not in visited:
        visited.add(current.job_id)
        parent_id = str(current.record.get("retry_of") or current.record.get("retry_of_job_id") or current.record.get("parent_job_id") or "")
        if not parent_id or parent_id not in jobs:
            break
        current = jobs[parent_id]
    request = current.record.get("request") or {} if current else {}
    return Path(str(request.get("geometry") or "")).name


def _job_for_row(rows: dict[tuple[str, str], dict[str, str]], jobs: dict[str, Any], pose: str, aliases: set[str]) -> Any:
    row = _find_role(rows, pose, aliases)
    return jobs.get(_job_id(row)) if row else None


def _job_for_pose_entry(rows: dict[tuple[str, str], dict[str, str]], jobs: dict[str, Any], entry: dict[str, Any], aliases: set[str]) -> Any:
    """Resolve a role through either an agent cluster ID or one of its public seeds."""
    for pose in [str(entry.get("pose_id") or ""), *_members(entry)]:
        job = _job_for_row(rows, jobs, pose, aliases)
        if job is not None:
            return job
    return None


def _derived_pose_clusters(task_dir: Path, jobs: dict[str, Any], rmsd_limit: float, contact_limit: float) -> tuple[list[list[str]], dict[str, str]]:
    """Cluster public seeds solely from their optimized complex geometries."""
    manifest = read_csv_rows(task_dir / "data" / "inputs" / "complex_manifest.csv")
    fragment_rows = read_csv_rows(task_dir / "data" / "inputs" / "fragment_manifest.csv")
    fragment_ids: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"A": [], "B": []})
    for row in fragment_rows:
        pose = str(row.get("pose_id") or row.get("complex_id") or "")
        fragment = str(row.get("fragment") or row.get("fragment_id") or "").upper()
        try:
            atom_id = int(row.get("complex_atom_id") or row.get("atom_id") or 0)
        except (TypeError, ValueError):
            continue
        if pose and fragment in {"A", "B"} and atom_id > 0:
            fragment_ids[pose][fragment].append(atom_id)
    records: dict[str, tuple[Any, Any]] = {}
    for row in manifest:
        pose_id = str(row.get("pose_id") or "")
        file_name = Path(str(row.get("complex_file") or "")).name
        candidates = []
        for job in jobs.values():
            geometry_name = _origin_geometry_name(job, jobs)
            geometry = geometry_from_record(job.parsed.get("optimized_geometry"))
            if geometry_name == file_name and geometry is not None and job_valid_termination(job):
                candidates.append((job, geometry))
        if candidates:
            records[pose_id] = max(candidates, key=lambda pair: (job_no_significant_imaginary(pair[0]), pair[0].job_id))
    groups: list[list[str]] = []
    for pose_id in sorted(records):
        geometry = records[pose_id][1]
        for group in groups:
            representative = records[group[0]][1]
            representative_mapping = fragment_ids.get(group[0], {})
            candidate_mapping = fragment_ids.get(pose_id, {})
            contacts_a = interfragment_contact_vector(
                representative,
                representative_mapping.get("A", []),
                representative_mapping.get("B", []),
            )
            contacts_b = interfragment_contact_vector(
                geometry,
                candidate_mapping.get("A", []),
                candidate_mapping.get("B", []),
            )
            contact_error = float(max(abs(contacts_a - contacts_b))) if len(contacts_a) == len(contacts_b) and len(contacts_a) else math.inf
            if heavy_atom_rmsd(representative, geometry) <= rmsd_limit and contact_error <= contact_limit:
                group.append(pose_id)
                break
        else:
            groups.append([pose_id])
    return groups, {pose: pair[0].job_id for pose, pair in records.items()}


def score_task2(
    task_dir: Path,
    submission_dir: Path,
    reference_dir: Path,
    builder: ScoreBuilder,
    *,
    harness_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_config = load_yaml_optional(task_dir / "config" / "task.yaml")
    fragment_tolerance = float((task_config.get("deduplication", {}) or {}).get("frozen_fragment_coordinate_tolerance_angstrom", 1e-6))
    protocol_cfg = task_config.get("xtb_protocol", {}) or {}
    system_cfg = task_config.get("scientific_system", {}) or {}
    final_answer = load_json_optional(submission_dir / "final_answer.json")
    schema_errors = validate_final_answer(task_dir, final_answer)
    builder.notes.extend(schema_errors)
    if schema_errors:
        builder.hard_gate("valid_final_answer_schema_required", reason="final_answer.json violates the frozen public schema", evidence=schema_errors, total_zero=True)
    protocol = load_yaml_optional(submission_dir / "protocol.yaml")
    cluster_payload = load_json_optional(submission_dir / "pose_clusters.json")
    candidate_clusters = _clusters(cluster_payload)
    ranking_payload = load_json_optional(submission_dir / "ranking.json")
    raw_rows = read_csv_rows(submission_dir / "raw_energies.csv")
    interaction_rows = read_csv_rows(submission_dir / "interaction_energies.csv")
    deformation_rows = read_csv_rows(submission_dir / "deformation_energies.csv")
    binding_rows = read_csv_rows(submission_dir / "binding_energies.csv")
    interaction_by_pose = {str(row.get("pose_id") or ""): row for row in interaction_rows if str(row.get("pose_id") or "")}
    deformation_by_pose = {str(row.get("pose_id") or ""): row for row in deformation_rows if str(row.get("pose_id") or "")}
    binding_by_pose = {str(row.get("pose_id") or ""): row for row in binding_rows if str(row.get("pose_id") or "")}
    role_rows = _role_index(raw_rows)
    recovery_payload = load_json_value_optional(submission_dir / "recovery_log.json")
    recovery_entries = normalize_recovery_log(recovery_payload)
    auxiliary_schema_errors = {
        "protocol.yaml": validate_json_against_schema(protocol, task_dir / "submission_contract" / "protocol.schema.json"),
        "pose_clusters.json": validate_json_against_schema(cluster_payload, task_dir / "submission_contract" / "pose_clusters.schema.json"),
        "ranking.json": validate_json_against_schema(ranking_payload, task_dir / "submission_contract" / "ranking.schema.json"),
        "recovery_log.json": validate_json_against_schema(recovery_payload, task_dir / "submission_contract" / "recovery_log.schema.json"),
    }
    auxiliary_schema_errors = {name: errors for name, errors in auxiliary_schema_errors.items() if errors}
    for name, errors in auxiliary_schema_errors.items():
        builder.notes.append(f"{name} schema errors: {'; '.join(errors)}")
    fragment_rows = read_csv_rows(task_dir / "data" / "inputs" / "fragment_manifest.csv")

    jobs, integrity_errors, run_manifest = verify_and_parse_jobs(submission_dir)
    if integrity_errors or not jobs:
        builder.hard_gate("trusted_xtb_output_required", reason="registered xTB evidence is missing or failed SHA-256 validation", evidence=integrity_errors, total_zero=True)
    output_status = required_output_status(task_config, submission_dir)
    missing_outputs = sorted(name for name, present in output_status.items() if not present)
    if missing_outputs:
        builder.notes.append(f"missing required outputs: {', '.join(missing_outputs)}")
    reference = _reference_data(reference_manifest(reference_dir))
    ref_poses = [entry for entry in reference.get("poses", []) if isinstance(entry, dict)]

    complex_manifest_rows = read_csv_rows(task_dir / "data" / "inputs" / "complex_manifest.csv")
    expected_complex_files = {Path(str(row.get("complex_file") or "")).name for row in complex_manifest_rows if str(row.get("complex_file") or "").strip()}
    manifest_files = {Path(str(row.get("complex_file") or "")).name for row in fragment_rows if str(row.get("complex_file") or "").strip()}
    composition_ok = manifest_files == expected_complex_files
    composition_evidence: dict[str, Any] = {"file_set": {"actual": sorted(manifest_files), "expected": sorted(expected_complex_files)}}
    for complex_file in sorted(expected_complex_files):
        rows = [row for row in fragment_rows if Path(str(row.get("complex_file") or "")).name == complex_file]
        ids: list[int] = []
        fragments: list[str] = []
        pose_ids: set[str] = set()
        invalid = False
        for row in rows:
            try:
                atom_id = int(row.get("complex_atom_id") or row.get("atom_id") or 0)
            except (TypeError, ValueError):
                invalid = True
                continue
            invalid |= atom_id <= 0
            ids.append(atom_id)
            fragments.append(str(row.get("fragment") or row.get("fragment_id") or "").upper())
            pose_id = str(row.get("pose_id") or row.get("complex_id") or "").strip()
            if pose_id:
                pose_ids.add(pose_id)
        ok = bool(rows) and not invalid and len(pose_ids) == 1 and len(ids) == len(set(ids)) and bool(ids) and sorted(ids) == list(range(1, max(ids) + 1)) and set(fragments) <= {"A", "B"} and {"A", "B"} <= set(fragments)
        composition_ok &= ok
        composition_evidence[complex_file] = {"pose_ids": sorted(pose_ids), "atom_ids": ids, "fragments": fragments, "ok": ok}
    builder.exact("fragment_membership_complete_nonoverlapping", composition_ok and bool(fragment_rows), evidence=composition_evidence, reason="public manifest assigns every complex atom exactly once to A or B")

    expected_values = {
        "complex_charge": int(system_cfg.get("complex_charge", 0)),
        "complex_uhf": int(system_cfg.get("complex_uhf", 0)),
        "fragment_a_charge": int(system_cfg.get("fragment_a_charge", 0)),
        "fragment_a_uhf": int(system_cfg.get("fragment_a_uhf", 0)),
        "fragment_b_charge": int(system_cfg.get("fragment_b_charge", 0)),
        "fragment_b_uhf": int(system_cfg.get("fragment_b_uhf", 0)),
    }
    submitted_values = {key: protocol.get(key) for key in expected_values}
    if any(value is None for value in submitted_values.values()) and isinstance(protocol.get("charges_and_uhf"), dict):
        submitted_values.update({key: protocol["charges_and_uhf"].get(key) for key in expected_values})
    builder.exact("all_charge_uhf_values", submitted_values == expected_values, evidence={"submitted": submitted_values, "expected": expected_values}, reason="all complex and fragment charge/UHF assignments exactly match")
    method = str(protocol.get("hamiltonian") or protocol.get("method") or "").upper().replace(" ", "")
    gas = str(protocol.get("solvent_model") or protocol.get("phase") or "").lower() in {"none", "gas", "gas_phase", "gas-phase"}
    gfn_ok = method in {"GFN2-XTB", "GFN2XTB", "GFN2"} and gas
    builder.exact("gfn_and_gas_phase", gfn_ok, evidence={"method": method, "phase": protocol.get("solvent_model") or protocol.get("phase")}, reason="declared gas-phase GFN2-xTB protocol matches")

    candidate_inputs = sorted(expected_complex_files)
    monomer_inputs = ["monomer_A.xyz", "monomer_B.xyz"]
    geometry_jobs: dict[str, list[Any]] = defaultdict(list)
    for job in jobs.values():
        geometry_jobs[_origin_geometry_name(job, jobs)].append(job)
    complex_success = sum(any(job_valid_termination(job) and bool((job.parsed.get("status") or {}).get("optimization_converged")) for job in geometry_jobs[name]) for name in candidate_inputs)
    monomer_success = sum(any(job_valid_termination(job) and bool((job.parsed.get("status") or {}).get("optimization_converged")) for job in geometry_jobs[name]) for name in monomer_inputs)
    builder.direct("candidate_complex_optimization_fraction", complex_success / len(candidate_inputs), evidence={"successful": complex_success, "total": len(candidate_inputs)}, reason="candidate complex seeds with trusted converged optimizations")
    builder.direct("isolated_monomer_optimization_fraction", monomer_success / len(monomer_inputs), evidence={"successful": monomer_success, "total": len(monomer_inputs)}, reason="isolated monomers with trusted converged optimizations")

    final_poses = [entry for entry in final_answer.get("unique_poses", []) if isinstance(entry, dict)] if isinstance(final_answer, dict) else []
    retained_jobs: list[Any] = []
    for entry in final_poses:
        job = jobs.get(str(entry.get("representative_job_id", "")))
        if job:
            retained_jobs.append(job)
    for aliases in ({"monomer_a_relaxed", "relaxed_monomer_a"}, {"monomer_b_relaxed", "relaxed_monomer_b"}):
        job = _job_for_row(role_rows, jobs, "", aliases)
        if job and job not in retained_jobs:
            retained_jobs.append(job)
    normal_fraction = sum(job_valid_termination(job) for job in retained_jobs) / len(retained_jobs) if retained_jobs else 0.0
    no_imag_fraction = sum(_has_valid_hessian_for_geometry(job, jobs) for job in retained_jobs) / len(retained_jobs) if retained_jobs else 0.0
    builder.direct("retained_structures_normal_termination", normal_fraction, evidence={"normal": sum(job_valid_termination(job) for job in retained_jobs), "retained": len(retained_jobs)}, reason="retained complex/monomer structures terminate normally")
    builder.direct("retained_structures_no_significant_imaginary", no_imag_fraction, evidence={"valid": sum(_has_valid_hessian_for_geometry(job, jobs) for job in retained_jobs), "retained": len(retained_jobs)}, reason="retained structures have a complete Hessian on the exact optimized geometry and no significant imaginary frequency")
    retained_complex_jobs = [jobs.get(str(entry.get("representative_job_id", ""))) for entry in final_poses]
    optimized_geometry_fraction = sum(geometry_from_record(job.parsed.get("optimized_geometry")) is not None for job in retained_complex_jobs if job) / len(final_poses) if final_poses else 0.0
    builder.direct("retained_optimized_geometry_coverage", optimized_geometry_fraction, evidence={"retained_pose_count": len(final_poses), "optimized_geometry_count": sum(geometry_from_record(job.parsed.get("optimized_geometry")) is not None for job in retained_complex_jobs if job)}, reason="every retained pose has a trusted optimized geometry for clustering and frozen-fragment extraction")

    ref_clusters = [list(map(str, entry.get("member_seed_ids", []))) for entry in ref_poses]
    derived_pose_clusters, derived_pose_jobs = _derived_pose_clusters(
        task_dir,
        jobs,
        float((task_config.get("deduplication", {}) or {}).get("complex_heavy_atom_rmsd_angstrom", 0.25)),
        float((task_config.get("deduplication", {}) or {}).get("intermolecular_contact_distance_error_angstrom", 0.15)),
    )
    submitted_clusters = [_members(cluster) for cluster in candidate_clusters if _members(cluster)]
    pairwise = pairwise_cluster_f1(derived_pose_clusters, submitted_clusters)
    contact_audit: dict[str, Any] = {}
    contact_label_matches = 0
    for cluster in candidate_clusters:
        pose_id = str(cluster.get("pose_id") or "")
        representative_job = jobs.get(str(cluster.get("representative_job_id") or ""))
        family, metrics = classify_caffeine_water_contact(_job_geometry(representative_job))
        submitted_family = str(cluster.get("post_opt_contact_family") or "")
        matches = submitted_family == family
        contact_label_matches += int(matches)
        contact_audit[pose_id] = {
            "recomputed_family": family,
            "submitted_family": submitted_family,
            "matches": matches,
            "metrics": metrics,
        }
    contact_fraction = contact_label_matches / len(candidate_clusters) if candidate_clusters else 0.0
    builder.direct("unique_minimum_cluster_pairwise_f1", pairwise * contact_fraction, evidence={"geometry_derived": derived_pose_clusters, "submitted": submitted_clusters, "jobs": derived_pose_jobs, "pairwise_f1": pairwise, "post_opt_contact_audit": contact_audit}, reason="optimized-geometry clustering and post-optimization contact-family labels must both be reproduced from trusted geometries")
    ref_duplicate_pairs = {tuple(sorted((first, second))) for cluster in ref_clusters for index, first in enumerate(cluster) for second in cluster[index + 1 :]}
    submitted_pose_members = [set(map(str, entry.get("member_seed_ids", []))) for entry in final_poses]
    final_double_count = any(pair[0] in members_a and pair[1] in members_b for pair in ref_duplicate_pairs for i, members_a in enumerate(submitted_pose_members) for members_b in submitted_pose_members[i + 1 :])
    builder.exact("collapsed_seeds_not_double_counted", not final_double_count and len(final_poses) == len({str(entry.get("pose_id")) for entry in final_poses}), evidence={"reference_duplicate_pairs": sorted(ref_duplicate_pairs)}, reason="collapsed starting poses occur in only one submitted unique pose")

    reference_retained_sets = [set(map(str, entry.get("member_seed_ids", []))) for entry in ref_poses]
    recovered_reference_indices = {
        index
        for index, reference_members in enumerate(reference_retained_sets)
        if reference_members and reference_members in submitted_pose_members
    }
    submitted_seed_list = [seed for members in submitted_pose_members for seed in members]
    unique_seed_membership = len(submitted_seed_list) == len(set(submitted_seed_list))
    retained_recall = (len(recovered_reference_indices) / len(reference_retained_sets) if reference_retained_sets else 0.0) if unique_seed_membership else 0.0
    builder.direct("retained_top_four_recall", retained_recall, evidence={"matching_policy": "exact_member_set_one_to_one", "matched_reference_indices": sorted(recovered_reference_indices), "reference_retained_count": len(reference_retained_sets), "submitted_retained_count": len(final_poses), "unique_seed_membership": unique_seed_membership}, reason="budgeted retained minima recover each of the four lowest optimized independent clusters by an exact, one-to-one member-set match")
    retained_limit = int((task_config.get("allocation", {}) or {}).get("max_retained_unique_poses", 4))
    builder.exact("retained_count_within_allocation", 0 < len(final_poses) <= retained_limit, evidence={"submitted_retained_count": len(final_poses), "maximum_retained_count": retained_limit}, reason="expensive Hessian and frozen-fragment evidence is allocated to at most the declared number of screened independent minima")

    monomer_a_job = _job_for_row(role_rows, jobs, "", {"monomer_a_relaxed", "relaxed_monomer_a"})
    monomer_b_job = _job_for_row(role_rows, jobs, "", {"monomer_b_relaxed", "relaxed_monomer_b"})
    monomer_a_energy = job_energy(monomer_a_job, "total_energy_hartree")
    monomer_b_energy = job_energy(monomer_b_job, "total_energy_hartree")
    relaxed_sources_ok = bool(monomer_a_job and monomer_b_job and monomer_a_energy is not None and monomer_b_energy is not None and _has_valid_hessian_for_geometry(monomer_a_job, jobs) and _has_valid_hessian_for_geometry(monomer_b_job, jobs))
    builder.exact("relaxed_monomer_energy_sources", relaxed_sources_ok, evidence={"monomer_a_job": getattr(monomer_a_job, "job_id", None), "monomer_b_job": getattr(monomer_b_job, "job_id", None)}, reason="relaxed monomer energies come from trusted isolated jobs")

    claimed_interaction: dict[str, float] = {}
    claimed_deformation: dict[str, float] = {}
    claimed_binding: dict[str, float] = {}
    raw_interaction: dict[str, float] = {}
    raw_deformation: dict[str, float] = {}
    raw_binding: dict[str, float] = {}
    candidate_to_reference: dict[str, str] = {}
    complex_provenance_count = fragment_a_valid_count = fragment_b_valid_count = exact_formula_count = 0
    submitted_fragment_match_count_a = submitted_fragment_match_count_b = 0
    fragment_evidence: dict[str, Any] = {}
    protocol_valid = True
    final_energy_job_ids: set[str] = set()

    for entry in final_poses:
        pose_id = str(entry.get("pose_id", ""))
        if not pose_id:
            continue
        for target, key in ((claimed_interaction, "interaction_kcal_mol"), (claimed_deformation, "deformation_kcal_mol"), (claimed_binding, "binding_kcal_mol")):
            if isinstance(entry.get(key), (int, float)):
                target[pose_id] = float(entry[key])
        complex_job = _job_for_pose_entry(role_rows, jobs, entry, {"complex", "optimized_complex", "complex_energy"})
        fragment_a_job = _job_for_pose_entry(role_rows, jobs, entry, {"fragment_a_frozen", "frozen_fragment_a", "fragment_a"})
        fragment_b_job = _job_for_pose_entry(role_rows, jobs, entry, {"fragment_b_frozen", "frozen_fragment_b", "fragment_b"})
        representative_id = str(entry.get("representative_job_id", ""))
        representative_job = jobs.get(representative_id)
        reference_pose_id = _reference_pose_id(entry, ref_poses)
        if reference_pose_id:
            candidate_to_reference[pose_id] = reference_pose_id
        complex_provenance_count += int(_complex_energy_geometry_matches_representative(complex_job, representative_job))
        for job in (complex_job, fragment_a_job, fragment_b_job):
            if job:
                final_energy_job_ids.add(job.job_id)
        # Unique-minimum labels are agent-defined.  Resolve the frozen atom
        # mapping through a member public seed instead of requiring the agent
        # to reuse a reference-facing identifier.
        mapping_ids = _members(entry) + [pose_id]
        mapping_a = next((value for candidate in mapping_ids if (value := _public_fragment_mapping(fragment_rows, candidate, "A"))), [])
        mapping_b = next((value for candidate in mapping_ids if (value := _public_fragment_mapping(fragment_rows, candidate, "B"))), [])
        error_a = _fragment_geometry_error(complex_job, fragment_a_job, mapping_a)
        error_b = _fragment_geometry_error(complex_job, fragment_b_job, mapping_b)
        valid_a = error_a <= fragment_tolerance
        valid_b = error_b <= fragment_tolerance
        matching_cluster = next(
            (
                cluster
                for cluster in candidate_clusters
                if str(cluster.get("pose_id") or "") == pose_id
                or set(_members(cluster)) == set(_members(entry))
            ),
            {},
        )
        submitted_fragment_match = matching_cluster.get("fragment_geometry_match") if isinstance(matching_cluster, dict) else None
        submitted_fragment_match_count_a += int(isinstance(submitted_fragment_match, dict) and submitted_fragment_match.get("fragment_a") is valid_a)
        submitted_fragment_match_count_b += int(isinstance(submitted_fragment_match, dict) and submitted_fragment_match.get("fragment_b") is valid_b)
        fragment_a_valid_count += int(valid_a)
        fragment_b_valid_count += int(valid_b)
        fragment_evidence[pose_id] = {"fragment_a_max_internal_distance_error_angstrom": error_a, "fragment_b_max_internal_distance_error_angstrom": error_b}
        if not valid_a or not valid_b:
            builder.hard_gate("frozen_fragment_geometry_hash_required", reason=f"frozen fragment geometry for {pose_id} is not internally identical to the optimized complex subset", evidence=fragment_evidence[pose_id], invalidates=("interaction_energy_mae_kcal_mol", "signs_units_and_formula", "deformation_energy_mae_kcal_mol", "binding_energy_mae_kcal_mol", "closure_error_kcal_mol", "deformation_energy_nonnegative"))
        e_complex = job_energy(complex_job, "total_energy_hartree")
        e_a_frozen = job_energy(fragment_a_job, "total_energy_hartree")
        e_b_frozen = job_energy(fragment_b_job, "total_energy_hartree")
        if None not in (e_complex, e_a_frozen, e_b_frozen, monomer_a_energy, monomer_b_energy):
            interaction = (float(e_complex) - float(e_a_frozen) - float(e_b_frozen)) * HARTREE_TO_KCAL_MOL
            deformation = ((float(e_a_frozen) - float(monomer_a_energy)) + (float(e_b_frozen) - float(monomer_b_energy))) * HARTREE_TO_KCAL_MOL
            binding = (float(e_complex) - float(monomer_a_energy) - float(monomer_b_energy)) * HARTREE_TO_KCAL_MOL
            raw_interaction[pose_id] = interaction
            raw_deformation[pose_id] = deformation
            raw_binding[pose_id] = binding
            interaction_row = interaction_by_pose.get(pose_id, {})
            deformation_row = deformation_by_pose.get(pose_id, {})
            binding_row = binding_by_pose.get(pose_id, {})
            try:
                submitted_closure = float(binding_row.get("closure_error", "nan"))
                csv_values_match = (
                    abs(float(interaction_row.get("interaction_kcal_mol", "nan")) - interaction) <= 0.01
                    and abs(float(deformation_row.get("deformation_kcal_mol", "nan")) - deformation) <= 0.01
                    and abs(float(binding_row.get("binding_kcal_mol", "nan")) - binding) <= 0.01
                    and abs(submitted_closure - (binding - interaction - deformation)) <= 0.001
                    and str(interaction_row.get("complex_job_id") or "") == getattr(complex_job, "job_id", "")
                    and str(interaction_row.get("fragment_a_job_id") or "") == getattr(fragment_a_job, "job_id", "")
                    and str(interaction_row.get("fragment_b_job_id") or "") == getattr(fragment_b_job, "job_id", "")
                    and str(binding_row.get("complex_job_id") or "") == getattr(complex_job, "job_id", "")
                )
            except (TypeError, ValueError):
                csv_values_match = False
            final_values_match = (
                pose_id in claimed_interaction
                and pose_id in claimed_deformation
                and pose_id in claimed_binding
                and abs(claimed_interaction[pose_id] - interaction) <= 0.01
                and abs(claimed_deformation[pose_id] - deformation) <= 0.01
                and abs(claimed_binding[pose_id] - binding) <= 0.01
            )
            exact_formula_count += int(final_values_match and csv_values_match)
        for job, charge, uhf in (
            (complex_job, expected_values["complex_charge"], expected_values["complex_uhf"]),
            (fragment_a_job, expected_values["fragment_a_charge"], expected_values["fragment_a_uhf"]),
            (fragment_b_job, expected_values["fragment_b_charge"], expected_values["fragment_b_uhf"]),
        ):
            if job is not None:
                protocol_valid &= protocol_matches(job, version=str(protocol_cfg.get("version")), gfn=int(protocol_cfg.get("gfn", 2)), solvent_model="none", solvent=None, charge=charge, uhf=uhf)
    for job, charge, uhf in ((monomer_a_job, expected_values["fragment_a_charge"], expected_values["fragment_a_uhf"]), (monomer_b_job, expected_values["fragment_b_charge"], expected_values["fragment_b_uhf"])):
        if job:
            final_energy_job_ids.add(job.job_id)
            protocol_valid &= protocol_matches(job, version=str(protocol_cfg.get("version")), gfn=int(protocol_cfg.get("gfn", 2)), solvent_model="none", solvent=None, charge=charge, uhf=uhf)

    pose_count = len(final_poses)
    builder.direct("complex_energy_from_optimized_complex", complex_provenance_count / pose_count if pose_count else 0.0, evidence={"valid": complex_provenance_count, "poses": pose_count}, reason="complex energy jobs match submitted optimized representatives")
    builder.direct("fragment_a_energy_from_frozen_complex_geometry", min(fragment_a_valid_count, submitted_fragment_match_count_a) / pose_count if pose_count else 0.0, evidence={"raw": fragment_evidence, "submitted_match_count": submitted_fragment_match_count_a}, reason="fragment-A jobs retain optimized-complex coordinates and the machine-readable geometry-match audit agrees")
    builder.direct("fragment_b_energy_from_frozen_complex_geometry", min(fragment_b_valid_count, submitted_fragment_match_count_b) / pose_count if pose_count else 0.0, evidence={"raw": fragment_evidence, "submitted_match_count": submitted_fragment_match_count_b}, reason="fragment-B jobs retain optimized-complex coordinates and the machine-readable geometry-match audit agrees")

    ref_interaction = {str(entry["pose_id"]): float(entry["interaction_kcal_mol"]) for entry in ref_poses if isinstance(entry.get("interaction_kcal_mol"), (int, float))}
    ref_deformation = {str(entry["pose_id"]): float(entry["deformation_kcal_mol"]) for entry in ref_poses if isinstance(entry.get("deformation_kcal_mol"), (int, float))}
    ref_binding = {str(entry["pose_id"]): float(entry["binding_kcal_mol"]) for entry in ref_poses if isinstance(entry.get("binding_kcal_mol"), (int, float))}
    aligned_interaction = {candidate_to_reference[key]: value for key, value in raw_interaction.items() if key in candidate_to_reference}
    aligned_deformation = {candidate_to_reference[key]: value for key, value in raw_deformation.items() if key in candidate_to_reference}
    aligned_binding = {candidate_to_reference[key]: value for key, value in raw_binding.items() if key in candidate_to_reference}
    interaction_mae = mean_absolute_error(ref_interaction, aligned_interaction)
    deformation_mae = mean_absolute_error(ref_deformation, aligned_deformation)
    binding_mae = mean_absolute_error(ref_binding, aligned_binding)
    builder.linear("interaction_energy_mae_kcal_mol", interaction_mae, evidence={"mae": interaction_mae, "raw_recomputed": raw_interaction}, reason="MAE from trusted raw complex and frozen-fragment energies")
    units = str(protocol.get("reporting_energy_unit") or protocol.get("energy_unit") or "").lower()
    formula_text = str(protocol.get("energy_definitions") or protocol.get("formula") or "").lower()
    signs_units_formula_ok = exact_formula_count == pose_count and pose_count > 0 and units in {"kcal/mol", "kcal mol-1", "kcal_mol"} and ("interaction" in formula_text or "e_int" in formula_text or "δe" in formula_text or "Δe" in str(protocol.get("energy_definitions") or ""))
    builder.exact("signs_units_and_formula", signs_units_formula_ok, evidence={"raw_recomputed": raw_interaction, "submitted": claimed_interaction, "unit": units}, reason="submitted values match trusted Hartree recomputation with frozen signs/formulas")
    builder.linear("deformation_energy_mae_kcal_mol", deformation_mae, evidence={"mae": deformation_mae, "raw_recomputed": raw_deformation}, reason="MAE from trusted raw frozen and relaxed monomer energies")
    builder.linear("binding_energy_mae_kcal_mol", binding_mae, evidence={"mae": binding_mae, "raw_recomputed": raw_binding}, reason="MAE from trusted raw complex and relaxed monomer energies")
    deformation_tolerance = float((task_config.get("scientific_quality_control", {}) or {}).get("deformation_nonnegative_tolerance_kcal_mol", 0.05))
    nonnegative_fraction = (
        sum(value >= -deformation_tolerance for value in raw_deformation.values()) / pose_count
        if pose_count else 0.0
    )
    builder.direct("deformation_energy_nonnegative", nonnegative_fraction, evidence={"raw_recomputed": raw_deformation, "tolerance_kcal_mol": deformation_tolerance}, reason="a consistently optimized relaxed-monomer reference cannot yield materially negative deformation energy")
    closure_errors = [abs(raw_binding[key] - raw_interaction[key] - raw_deformation[key]) for key in set(raw_binding) & set(raw_interaction) & set(raw_deformation)]
    closure_error = sum(closure_errors) / len(closure_errors) if len(closure_errors) == pose_count and pose_count else math.inf
    builder.linear("closure_error_kcal_mol", closure_error, evidence={"per_pose": closure_errors, "mean": closure_error}, reason="mean |E_bind-E_int-E_def| closure error")

    allowed_best = {str(value) for value in reference.get("best_pose_ids", [])}
    if not allowed_best and ref_binding:
        minimum = min(ref_binding.values())
        threshold = float((task_config.get("ranking", {}) or {}).get("tie_threshold_kcal_mol", 0.10))
        allowed_best = {key for key, value in ref_binding.items() if value - minimum < threshold}
    submitted_best = str(final_answer.get("best_pose_id", ""))
    ranking_file_best = str(ranking_payload.get("best_pose_id", ""))
    raw_best = min(raw_binding, key=raw_binding.get) if raw_binding else ""
    aligned_raw_best = candidate_to_reference.get(raw_best, "")
    builder.exact("best_unique_pose", aligned_raw_best in allowed_best and submitted_best == raw_best and ranking_file_best == raw_best, evidence={"allowed_reference_ids": sorted(allowed_best), "raw_candidate": raw_best, "raw_reference_alignment": aligned_raw_best, "final_answer": submitted_best, "ranking_file": ranking_file_best}, reason="both ranking outputs agree with the ranking rebuilt from raw energies")
    ranking = [str(value) for value in final_answer.get("ranking", [])] if isinstance(final_answer, dict) else []
    raw_ranking = sorted(raw_binding, key=lambda key: (raw_binding[key], key))
    aligned_raw_ranking = [candidate_to_reference.get(value, "") for value in raw_ranking]
    tau = kendall_tau_b_with_reference_ties(ref_binding, aligned_raw_ranking, float((task_config.get("ranking", {}) or {}).get("tie_threshold_kcal_mol", 0.10)))
    ranking_file = [str(value) for value in ranking_payload.get("ranking", [])] if isinstance(ranking_payload, dict) else []
    ranking_consistent = ranking == raw_ranking and ranking_file == raw_ranking and all(aligned_raw_ranking) and len(set(aligned_raw_ranking)) == len(aligned_raw_ranking)
    builder.direct("full_unique_pose_ranking_tau_b", (tau + 1.0) / 2.0 if ranking_consistent else 0.0, evidence={"tau_b": tau, "raw_candidate_ranking": raw_ranking, "aligned_reference_ranking": aligned_raw_ranking, "final_answer_ranking": ranking, "ranking_file": ranking_file}, reason="both ranking outputs must equal the raw-energy reconstruction before reference comparison")
    ordered_raw_binding = sorted(raw_binding.values())
    raw_leading_gap = ordered_raw_binding[1] - ordered_raw_binding[0] if len(ordered_raw_binding) >= 2 else math.inf
    tie_threshold = float((task_config.get("ranking", {}) or {}).get("tie_threshold_kcal_mol", 0.10))
    submitted_gap = ranking_payload.get("best_second_gap_kcal_mol") if isinstance(ranking_payload, dict) else None
    near_degenerate = raw_leading_gap <= tie_threshold
    expected_uncertainty = binding_ranking_uncertainty(raw_binding, 0.25)
    submitted_uncertainty = ranking_payload.get("ranking_uncertainty") if isinstance(ranking_payload, dict) else None
    uncertainty_ranges_ok = False
    if isinstance(submitted_uncertainty, dict):
        submitted_ranges = submitted_uncertainty.get("per_pose_binding_ranges_kcal_mol")
        expected_ranges = expected_uncertainty["per_pose_binding_ranges_kcal_mol"]
        if isinstance(submitted_ranges, dict) and set(submitted_ranges) == set(expected_ranges):
            uncertainty_ranges_ok = all(
                isinstance(submitted_ranges.get(pose_id), dict)
                and isinstance(submitted_ranges[pose_id].get("minimum"), (int, float))
                and isinstance(submitted_ranges[pose_id].get("maximum"), (int, float))
                and abs(float(submitted_ranges[pose_id]["minimum"]) - expected_ranges[pose_id]["minimum"]) <= 0.01
                and abs(float(submitted_ranges[pose_id]["maximum"]) - expected_ranges[pose_id]["maximum"]) <= 0.01
                for pose_id in expected_ranges
            )
    uncertainty_ok = bool(
        isinstance(submitted_uncertainty, dict)
        and submitted_uncertainty.get("perturbation_kcal_mol") == 0.25
        and submitted_uncertainty.get("winner_stable") is expected_uncertainty["winner_stable"]
        and set(map(str, submitted_uncertainty.get("leading_pose_ids", []))) == set(expected_uncertainty["leading_pose_ids"])
        and ranking_payload.get("tie_or_stable_winner") == expected_uncertainty["tie_or_stable_winner"]
        and uncertainty_ranges_ok
    )
    ranking_robustness_ok = bool(
        isinstance(submitted_gap, (int, float))
        and abs(float(submitted_gap) - raw_leading_gap) <= 0.01
        and ranking_payload.get("near_degenerate") is near_degenerate
        and len(str(ranking_payload.get("ranking_stability_note") or "").strip()) >= 40
        and uncertainty_ok
    )
    builder.exact("near_degenerate_ranking_robustness", ranking_robustness_ok, evidence={"raw_leading_gap_kcal_mol": raw_leading_gap, "submitted_gap_kcal_mol": submitted_gap, "near_degenerate": near_degenerate, "tie_threshold_kcal_mol": tie_threshold, "recomputed_uncertainty": expected_uncertainty, "submitted_uncertainty": submitted_uncertainty, "submitted_tie_or_stable_winner": ranking_payload.get("tie_or_stable_winner")}, reason="ranking robustness must reproduce the raw leading gap and the independent +/-0.25 kcal/mol interval analysis without forcing an unsupported winner")

    final_source_jobs = {str(value) for value in final_answer.get("source_jobs", [])} if isinstance(final_answer, dict) else set()
    invalid_detected = classified = targeted = corrected_used = False
    for entry in recovery_entries:
        invalid_id = str(entry.get("invalid_job_id") or entry.get("failed_job_id") or entry.get("source_job_id") or "")
        corrected_id = str(entry.get("retry_job_id") or entry.get("corrected_job_id") or "")
        invalid_job = jobs.get(invalid_id)
        corrected_job = jobs.get(corrected_id)
        actual_issue = bool(invalid_job and not is_deliberate_protocol_failure(invalid_job) and (not job_valid_termination(invalid_job) or not job_no_significant_imaginary(invalid_job)))
        seed_id = str(entry.get("seed_id") or "")
        collapsed = any(seed_id in cluster and len(cluster) > 1 for cluster in ref_clusters) if seed_id else False
        actual_issue |= collapsed
        invalid_detected |= actual_issue
        classified |= actual_issue and bool(str(entry.get("issue") or entry.get("classification") or "").strip())
        relation = bool(corrected_job and (corrected_job.record.get("retry_of") == invalid_id or entry.get("reextraction_of") == invalid_id))
        targeted |= actual_issue and relation and bool(str(entry.get("targeted_change") or entry.get("change") or "").strip())
        corrected_used |= relation and job_valid_termination(corrected_job) and corrected_id in final_source_jobs and corrected_id in final_energy_job_ids
    required_no_retry_checks = {
        "convergence",
        "complete_hessian",
        "clustering",
        "fragment_geometry",
        "energy_closure",
        "budget",
        "output_consistency",
    }
    documented_no_retry_checks = {
        str(value)
        for entry in recovery_entries
        if str(entry.get("status") or "").lower() in {"no_retry_required", "all_checks_passed"}
        for value in (entry.get("checks_performed") or entry.get("checks") or [])
    }
    raw_no_retry_checks = bool(
        retained_jobs
        and all(_has_valid_hessian_for_geometry(job, jobs) for job in retained_jobs)
        and normal_fraction == 1.0
        and pairwise == 1.0
        and contact_fraction == 1.0
        and fragment_a_valid_count == len(final_poses)
        and fragment_b_valid_count == len(final_poses)
        and math.isfinite(closure_error)
        and closure_error <= 0.001
        and exact_formula_count == len(final_poses)
        and ranking_consistent
        and len(run_manifest.get("jobs", [])) <= int((task_config.get("budgets", {}) or {}).get("max_xtb_calls", 0))
    )
    no_retry_needed = raw_no_retry_checks and required_no_retry_checks.issubset(documented_no_retry_checks)
    recovery_evidence = {"entries": recovery_entries, "required_no_retry_checks": sorted(required_no_retry_checks), "documented_no_retry_checks": sorted(documented_no_retry_checks), "raw_no_retry_checks": raw_no_retry_checks}
    builder.exact("invalid_or_collapsed_pose_detected", invalid_detected or no_retry_needed, evidence=recovery_evidence, reason="real issue detected, or all required first-pass checks documented and independently verified")
    builder.exact("fragment_or_convergence_issue_classified", classified or no_retry_needed, evidence=recovery_evidence, reason="real issue classified, or complete no-retry checklist verified")
    builder.exact("targeted_retry_or_reextraction", targeted or no_retry_needed, evidence=recovery_evidence, reason="targeted correction, or fully validated no-change path")
    builder.exact("final_energies_use_corrected_result", corrected_used or no_retry_needed, evidence={"source_jobs": sorted(final_source_jobs)}, reason="final energies use the applicable validated path")

    interpretation = load_json_optional(submission_dir / "scientific_interpretation.json")
    interpretation_errors = validate_json_against_schema(
        interpretation,
        task_dir / "submission_contract" / "scientific_interpretation.schema.json",
    )
    ordered_binding = sorted(raw_binding.values())
    binding_gap = ordered_binding[1] - ordered_binding[0] if len(ordered_binding) >= 2 else math.nan
    best_pose = min(raw_binding, key=raw_binding.get) if raw_binding else None
    partition_metrics = {}
    if best_pose is not None:
        partition_metrics = {
            "best_pose_interaction_energy_kcal_mol": {"value": raw_interaction.get(best_pose, math.nan), "unit": "kcal/mol", "absolute_tolerance": 0.01},
            "best_pose_deformation_energy_kcal_mol": {"value": raw_deformation.get(best_pose, math.nan), "unit": "kcal/mol", "absolute_tolerance": 0.01},
            "best_pose_binding_energy_kcal_mol": {"value": raw_binding.get(best_pose, math.nan), "unit": "kcal/mol", "absolute_tolerance": 0.01},
        }
    partition_ok, partition_evidence = quantitatively_grounded_claim(
        interpretation, "interaction_vs_deformation", jobs, expected_quantities=partition_metrics,
    )
    site_competition_metrics = {
        f"binding_energy_{pose_id}": {"value": value, "unit": "kcal/mol", "absolute_tolerance": 0.01}
        for pose_id, value in raw_binding.items()
    }
    for first_index, (first_pose, first_value) in enumerate(sorted(raw_binding.items())):
        for second_pose, second_value in sorted(raw_binding.items())[first_index + 1 :]:
            site_competition_metrics[f"binding_gap_{first_pose}_{second_pose}"] = {
                "value": abs(first_value - second_value),
                "unit": "kcal/mol",
                "absolute_tolerance": 0.01,
            }
    site_competition_metrics["top_two_binding_gap_kcal_mol"] = {
        "value": binding_gap,
        "unit": "kcal/mol",
        "absolute_tolerance": 0.01,
    }
    site_ok, site_evidence = quantitatively_grounded_claim(
        interpretation, "site_competition", jobs,
        expected_quantities=site_competition_metrics,
    )

    def _validated_hessian(source_job: Any) -> Any:
        if source_job is None:
            return None
        if job_no_significant_imaginary(source_job) and standalone_hessian_parent(source_job, jobs) is not None:
            return source_job
        for candidate in jobs.values():
            if job_no_significant_imaginary(candidate) and standalone_hessian_parent(candidate, jobs) == source_job.job_id:
                return candidate
        return None

    convention_metrics = {
        "mean_energy_closure_error_kcal_mol": {"value": closure_error, "unit": "kcal/mol", "absolute_tolerance": 0.001}
    }
    best_complex_job = None
    if best_pose is not None:
        best_entry = next((entry for entry in final_poses if str(entry.get("pose_id")) == best_pose), None)
        if best_entry is not None:
            best_complex_job = _job_for_pose_entry(role_rows, jobs, best_entry, {"complex", "optimized_complex", "complex_energy"})
    hessian_triplet = [_validated_hessian(value) for value in (best_complex_job, monomer_a_job, monomer_b_job)]
    zpves = [job_energy(value, "zpve_hartree") for value in hessian_triplet]
    if best_pose is not None and all(value is not None for value in zpves):
        zpve_correction = (float(zpves[0]) - float(zpves[1]) - float(zpves[2])) * HARTREE_TO_KCAL_MOL
        convention_metrics["best_pose_zpve_correction_kcal_mol"] = {"value": zpve_correction, "unit": "kcal/mol", "absolute_tolerance": 0.02}
        convention_metrics["best_pose_zpve_corrected_binding_kcal_mol"] = {"value": raw_binding[best_pose] + zpve_correction, "unit": "kcal/mol", "absolute_tolerance": 0.02}
    convention_ok, convention_evidence = quantitatively_grounded_claim(
        interpretation, "convention_limitations", jobs,
        expected_quantities=convention_metrics,
    )
    partition_ok &= not interpretation_errors and bool(set(partition_evidence["evidence_job_ids"]) & final_energy_job_ids)
    site_ok &= not interpretation_errors and bool(set(site_evidence["evidence_job_ids"]) & final_energy_job_ids)
    convention_ok &= not interpretation_errors
    builder.exact("interaction_vs_deformation_interpretation", partition_ok, evidence={"validation_errors": interpretation_errors, **partition_evidence}, reason="interaction/deformation claim reproduces a trusted best-pose energy term")
    builder.exact("competing_site_contact_rationale", site_ok, evidence={"validation_errors": interpretation_errors, **site_evidence}, reason="site-competition claim quantifies the raw top-two binding-energy separation")
    builder.exact("frozen_fragment_convention_limits", convention_ok, evidence={"validation_errors": interpretation_errors, **convention_evidence}, reason="convention claim reports the raw decomposition closure error")

    manifest_task_id = run_manifest.get("task_id")
    accepted_task_ids = {task_config.get("task_id"), "task_05"}
    identity_ok, identity_evidence = manifest_execution_identity_audit(run_manifest, harness_manifest, accepted_task_ids=accepted_task_ids, expected_task_version=str(task_config.get("task_version") or ""))
    contract_errors = [str(item) for item in integrity_errors]
    if schema_errors:
        contract_errors.append("final_answer.json violates its frozen schema: " + "; ".join(schema_errors))
    if manifest_task_id not in accepted_task_ids or not identity_ok:
        contract_errors.append(f"submission manifest execution identity mismatch: {identity_evidence}")
    manifest_complete = bool(manifest_task_id in accepted_task_ids and identity_ok and jobs and not integrity_errors and not missing_outputs)
    builder.exact("manifest_and_hashes", manifest_complete, evidence={"job_count": len(jobs), "integrity_errors": integrity_errors, "execution_identity": identity_evidence}, reason="v2 task manifest, immutable execution identity, and registered hashes validate")
    max_calls = int((task_config.get("budgets", {}) or {}).get("max_xtb_calls", 0))
    inner_jobs_used = len(run_manifest.get("jobs", []))
    ledger = (harness_manifest or {}).get("budget_ledger") if isinstance(harness_manifest, dict) else None
    totals = ledger.get("totals", {}) if isinstance(ledger, dict) else {}
    outer_jobs_used = totals.get("xtb_jobs")
    authoritative_jobs_used = int(outer_jobs_used) if isinstance(outer_jobs_used, (int, float)) else inner_jobs_used
    ledger_consistent = outer_jobs_used is None or authoritative_jobs_used == inner_jobs_used
    budget_ok = authoritative_jobs_used <= max_calls and ledger_consistent
    if not ledger_consistent:
        builder.hard_gate("cumulative_budget_provenance_required", reason="Harness budget ledger and scientific manifest disagree", evidence={"outer_xtb_jobs": authoritative_jobs_used, "inner_xtb_jobs": inner_jobs_used}, invalidates=("budget_compliance",))
    builder.exact("budget_compliance", budget_ok, evidence={"used": authoritative_jobs_used, "inner_used": inner_jobs_used, "max": max_calls, "ledger_consistent": ledger_consistent}, reason="cumulative Harness xTB calls remain within budget and agree with submitted evidence")

    schema_item_groups = {
        "protocol.yaml": ("all_charge_uhf_values", "gfn_and_gas_phase", "signs_units_and_formula"),
        "pose_clusters.json": ("unique_minimum_cluster_pairwise_f1", "collapsed_seeds_not_double_counted"),
        "ranking.json": ("best_unique_pose", "full_unique_pose_ranking_tau_b", "near_degenerate_ranking_robustness"),
        "recovery_log.json": ("invalid_or_collapsed_pose_detected", "fragment_or_convergence_issue_classified", "targeted_retry_or_reextraction", "final_energies_use_corrected_result"),
    }
    for name, errors in auxiliary_schema_errors.items():
        for item_id in schema_item_groups[name]:
            builder.add(item_id, fraction=0.0, evidence=errors, reason=f"{name} violates its public schema")

    missing_output_groups = {
        "interaction_energies.csv": ("interaction_energy_mae_kcal_mol", "signs_units_and_formula"),
        "deformation_energies.csv": ("deformation_energy_mae_kcal_mol", "deformation_energy_nonnegative", "signs_units_and_formula"),
        "binding_energies.csv": ("binding_energy_mae_kcal_mol", "closure_error_kcal_mol", "signs_units_and_formula"),
        "optimized_structures/": ("retained_optimized_geometry_coverage", "unique_minimum_cluster_pairwise_f1"),
    }
    for name in missing_outputs:
        for item_id in missing_output_groups.get(name, ()):
            builder.add(item_id, fraction=0.0, evidence={"missing_output": name}, reason="the corresponding required submission artifact is missing")

    if not protocol_valid:
        protocol_items = ("interaction_energy_mae_kcal_mol", "signs_units_and_formula", "deformation_energy_mae_kcal_mol", "binding_energy_mae_kcal_mol", "closure_error_kcal_mol", "deformation_energy_nonnegative")
        builder.hard_gate("exact_gfn2_gas_phase_required", reason="one or more final energy jobs use the wrong GFN/phase protocol", invalidates=protocol_items)
        builder.hard_gate("charge_and_uhf_required", reason="one or more final energy jobs use the wrong charge/UHF", invalidates=protocol_items)
        for item_id in protocol_items:
            builder.add(item_id, fraction=0.0, evidence="protocol hard gate", reason="energy stage invalidated by frozen protocol mismatch")
    if fragment_a_valid_count != pose_count or fragment_b_valid_count != pose_count:
        for item_id in ("interaction_energy_mae_kcal_mol", "signs_units_and_formula", "deformation_energy_mae_kcal_mol", "binding_energy_mae_kcal_mol", "closure_error_kcal_mol", "deformation_energy_nonnegative"):
            builder.add(item_id, fraction=0.0, evidence="frozen fragment geometry hard gate", reason="energy decomposition invalidated by incorrect frozen geometry")
    counterpoise_claim = text_contains_affirmative_claim(
        submission_dir / "report.md",
        patterns=[r"(?:used|applied|performed|with)\s+(?:boys[-– ]bernardi\s+)?counterpoise", r"counterpoise\s+corrected"],
        negations=["no counterpoise", "without counterpoise", "did not use counterpoise", "not corrected with counterpoise", "not counterpoise corrected", "do not report counterpoise", "不使用 counterpoise", "未使用 counterpoise"],
    ) or bool(protocol.get("counterpoise") is True)
    if counterpoise_claim:
        builder.hard_gate("no_counterpoise_claim", reason="submission affirmatively claims forbidden Counterpoise", invalidates=("signs_units_and_formula",))
        builder.add("signs_units_and_formula", fraction=0.0, evidence="counterpoise claim", reason="forbidden Counterpoise semantics invalidate formula credit")

    return {
        "schema_errors": schema_errors,
        "contract_errors": contract_errors,
        "integrity_errors": integrity_errors,
        "trusted_job_count": len(jobs),
        "candidate_pose_count": pose_count,
        "reference_pose_count": len(ref_poses),
        "raw_recomputed_energies": {"interaction": raw_interaction, "deformation": raw_deformation, "binding": raw_binding},
        "scientific_interpretation_errors": interpretation_errors,
        "auxiliary_schema_errors": auxiliary_schema_errors,
    }

"""Raw scientific scoring for the Task 06 coupled conformational landscape."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .core import ScoreBuilder
from common.geometry import kabsch_rmsd, read_xyz, read_xyz_frames
from common.io_utils import load_yaml, sha256_file
from .coupled_geometry import torus_distance
from .coupled_landscape import (
    HARTREE_TO_KCAL_MOL,
    LandscapePoint,
    coupling_support_valid,
    nonadditivity_rmse,
)
from .coupled_geometry import canonical_coordinates, coordinate_values
from .core import (
    job_valid_termination,
    load_json_optional,
    protocol_matches,
    reference_manifest,
    validate_final_answer,
    verify_and_parse_jobs,
)
from .coupled_submission import validate_coupled_submission


def _linear_fraction(value: float, full: float, zero: float) -> float:
    if not math.isfinite(value) or value >= zero:
        return 0.0
    if value <= full:
        return 1.0
    return (zero - value) / (zero - full)


def _circular_error(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _trusted_job_ids(evidence: Mapping[str, Any]) -> set[str]:
    jobs = evidence.get("trusted_jobs") if isinstance(evidence.get("trusted_jobs"), Mapping) else {}
    return {
        str(job_id)
        for job_id, job in jobs.items()
        if isinstance(job, Mapping)
        and job.get("protocol_valid") is True
        and str(job.get("status")) == "completed"
    }


def _mark_global_region(
    points: Sequence[dict[str, Any]], *, threshold_kcal_mol: float = 0.5
) -> None:
    """Mark every sampled geometry demonstrably near the raw global minimum."""

    if not points:
        return
    minimum = min(float(point["energy_kcal_mol"]) for point in points)
    for point in points:
        point["global_basin_support"] = (
            float(point["energy_kcal_mol"]) - minimum <= threshold_kcal_mol + 1.0e-12
        )


def _bound_points(
    final: Mapping[str, Any], evidence: Mapping[str, Any], trusted_jobs: set[str]
) -> tuple[dict[str, Mapping[str, Any]], float]:
    raw_points = [point for point in evidence.get("points", []) if isinstance(point, Mapping)]
    by_key: dict[tuple[str, int | None], list[Mapping[str, Any]]] = {}
    for point in raw_points:
        key = (str(point.get("job_id", "")), point.get("frame_index"))
        by_key.setdefault(key, []).append(point)
    submitted = [point for point in final.get("sampled_points", []) if isinstance(point, Mapping)]
    bound: dict[str, Mapping[str, Any]] = {}
    for point in submitted:
        point_id = str(point.get("point_id", ""))
        job_id = str(point.get("source_job_id", ""))
        if not point_id or job_id not in trusted_jobs:
            continue
        candidates = by_key.get((job_id, point.get("frame_index")), [])
        match = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("geometry_path", "")) == str(point.get("geometry_path", ""))
                and str(candidate.get("geometry_sha256", "")) == str(point.get("geometry_sha256", ""))
            ),
            None,
        )
        if match is not None:
            bound[point_id] = match
    return bound, len(bound) / max(1, len(submitted))


def _coordinate_matches(final: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[dict[str, int], float]:
    expected = [tuple(int(value) for value in pair) for pair in evidence.get("canonical_central_bonds", [])]
    mapping: dict[str, int] = {}
    used: set[int] = set()
    for hypothesis in final.get("coordinate_hypotheses", []):
        if not isinstance(hypothesis, Mapping):
            continue
        reported = {int(value) for value in hypothesis.get("central_bond_atom_ids", [])}
        for index, pair in enumerate(expected):
            if index not in used and reported == set(pair):
                mapping[str(hypothesis.get("coordinate_id", ""))] = index
                used.add(index)
                break
    return mapping, len(used) / max(1, len(expected))


def _match_basins(
    final: Mapping[str, Any],
    bound: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    submitted = [basin for basin in final.get("basins", []) if isinstance(basin, Mapping)]
    ref_basins = [basin for basin in reference.get("basins", []) if isinstance(basin, Mapping)]
    threshold = float(reference.get("basin_match_angle_deg", 20.0))
    rmsd_threshold = float(reference.get("basin_match_rmsd_angstrom", 0.30))
    candidates: list[tuple[float, float, str, str, str, Mapping[str, Any], Mapping[str, Any]]] = []
    for basin in submitted:
        representative_point_id = str(basin.get("representative_point_id", ""))
        point = bound.get(representative_point_id)
        if point is None:
            continue
        angles = tuple(float(value) for value in point.get("angles_deg", []))
        if len(angles) != 2:
            continue
        for ref in ref_basins:
            ref_angles = tuple(float(value) for value in ref.get("angles_deg", []))
            if len(ref_angles) != 2:
                continue
            distance = torus_distance(angles, ref_angles)
            geometry_rmsd = 0.0
            raw_coordinates = point.get("coordinates_angstrom")
            ref_coordinates = ref.get("representative_coordinates_angstrom")
            raw_elements = tuple(map(str, point.get("elements", [])))
            ref_elements = tuple(map(str, ref.get("elements", [])))
            if raw_coordinates is not None and ref_coordinates is not None:
                if raw_elements != ref_elements or not raw_elements:
                    geometry_rmsd = math.inf
                else:
                    heavy = [index for index, element in enumerate(raw_elements) if element != "H"]
                    try:
                        geometry_rmsd = kabsch_rmsd(
                            np.asarray(raw_coordinates, dtype=float)[heavy],
                            np.asarray(ref_coordinates, dtype=float)[heavy],
                        )
                    except (ValueError, IndexError):
                        geometry_rmsd = math.inf
            if distance <= threshold and geometry_rmsd <= rmsd_threshold:
                candidates.append(
                    (
                        distance,
                        geometry_rmsd,
                        str(basin.get("basin_id", "")),
                        str(ref.get("reference_id", "")),
                        representative_point_id,
                        basin,
                        point,
                    )
                )
    matches: dict[str, str] = {}
    matched_points: dict[str, Mapping[str, Any]] = {}
    used_reference: set[str] = set()
    used_representative_points: set[str] = set()
    audit: list[dict[str, Any]] = []
    for distance, geometry_rmsd, basin_id, reference_id, point_id, _, point in sorted(
        candidates, key=lambda item: item[:5]
    ):
        if basin_id in matches or reference_id in used_reference or point_id in used_representative_points:
            continue
        matches[basin_id] = reference_id
        matched_points[basin_id] = point
        used_reference.add(reference_id)
        used_representative_points.add(point_id)
        audit.append(
            {
                "basin_id": basin_id,
                "reference_id": reference_id,
                "torus_distance_deg": distance,
                "heavy_atom_rmsd_angstrom": geometry_rmsd,
            }
        )
    return matches, matched_points, audit


def _ordering_fraction(
    matches: Mapping[str, str],
    matched_points: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> float:
    ref_by_id = {str(item["reference_id"]): item for item in reference.get("basins", []) if isinstance(item, Mapping)}
    if not matches:
        return 0.0
    submitted_global = min(matches, key=lambda basin_id: (float(matched_points[basin_id]["energy_kcal_mol"]), basin_id))
    reference_global = min(ref_by_id, key=lambda ref_id: (float(ref_by_id[ref_id]["relative_energy_kcal_mol"]), ref_id))
    global_fraction = float(matches[submitted_global] == reference_global)
    pairs = list(matches)
    if len(pairs) < 2:
        pair_fraction = 0.0
    else:
        correct = 0
        total = 0
        for first_index in range(len(pairs)):
            for second_index in range(first_index + 1, len(pairs)):
                first = pairs[first_index]
                second = pairs[second_index]
                submitted_delta = float(matched_points[first]["energy_kcal_mol"]) - float(
                    matched_points[second]["energy_kcal_mol"]
                )
                reference_delta = float(ref_by_id[matches[first]]["relative_energy_kcal_mol"]) - float(
                    ref_by_id[matches[second]]["relative_energy_kcal_mol"]
                )
                near_degenerate = abs(submitted_delta) <= 0.05 and abs(reference_delta) <= 0.05
                correct += int(near_degenerate or submitted_delta * reference_delta > 0.0)
                total += 1
        pair_fraction = correct / max(1, total)
    return 0.5 * global_fraction + 0.5 * pair_fraction


def _basin_energy_mae(
    matches: Mapping[str, str],
    matched_points: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> float:
    if not matches:
        return math.inf
    ref_by_id = {str(item["reference_id"]): item for item in reference.get("basins", []) if isinstance(item, Mapping)}
    submitted_zero = min(float(point["energy_kcal_mol"]) for point in matched_points.values())
    errors = [
        abs(
            float(matched_points[basin_id]["energy_kcal_mol"])
            - submitted_zero
            - float(ref_by_id[reference_id]["relative_energy_kcal_mol"])
        )
        for basin_id, reference_id in matches.items()
    ]
    return float(np.mean(errors)) if errors else math.inf


def _transition_metrics(
    final: Mapping[str, Any],
    bound: Mapping[str, Mapping[str, Any]],
    basin_matches: Mapping[str, str],
    matched_basin_points: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> tuple[float, float, float, list[dict[str, Any]]]:
    ref_edges: dict[frozenset[str], Mapping[str, Any]] = {}
    for item in reference.get("transitions", []):
        if isinstance(item, Mapping):
            ref_edges[frozenset((str(item.get("from_reference_id")), str(item.get("to_reference_id"))))] = item
    reported_edges: set[frozenset[str]] = set()
    barrier_errors: list[float] = []
    numeric_valid = 0
    audit: list[dict[str, Any]] = []
    neighbor_limit = float(reference.get("path_neighbor_limit_deg", 100.0))
    for transition in final.get("transitions", []):
        if not isinstance(transition, Mapping):
            continue
        from_basin = str(transition.get("from_basin_id", ""))
        to_basin = str(transition.get("to_basin_id", ""))
        if from_basin not in basin_matches or to_basin not in basin_matches:
            continue
        point_ids = [str(value) for value in transition.get("ordered_point_ids", [])]
        points = [bound.get(point_id) for point_id in point_ids]
        if not points or any(point is None for point in points):
            continue
        concrete = [point for point in points if point is not None]
        continuous = all(
            torus_distance(tuple(map(float, first["angles_deg"])), tuple(map(float, second["angles_deg"])))
            <= neighbor_limit
            for first, second in zip(concrete, concrete[1:])
        )
        if not continuous:
            continue
        energies = [float(point["energy_kcal_mol"]) for point in concrete]
        highest_index = min(range(len(energies)), key=lambda index: (-energies[index], point_ids[index]))
        highest_id = point_ids[highest_index]
        from_energy = float(matched_basin_points[from_basin]["energy_kcal_mol"])
        to_energy = float(matched_basin_points[to_basin]["energy_kcal_mol"])
        forward = energies[highest_index] - from_energy
        reverse = energies[highest_index] - to_energy
        reported_forward = float(transition.get("forward_barrier_kcal_mol", math.inf))
        reported_reverse = float(transition.get("reverse_barrier_kcal_mol", math.inf))
        numeric_ok = (
            highest_id == str(transition.get("highest_point_id", ""))
            and abs(reported_forward - forward) <= 0.15
            and abs(reported_reverse - reverse) <= 0.15
        )
        numeric_valid += int(numeric_ok)
        edge = frozenset((basin_matches[from_basin], basin_matches[to_basin]))
        reported_edges.add(edge)
        ref = ref_edges.get(edge)
        if ref is not None:
            if str(ref.get("from_reference_id")) == basin_matches[from_basin]:
                expected_forward = float(ref.get("forward_barrier_kcal_mol", math.inf))
                expected_reverse = float(ref.get("reverse_barrier_kcal_mol", math.inf))
            else:
                expected_forward = float(ref.get("reverse_barrier_kcal_mol", math.inf))
                expected_reverse = float(ref.get("forward_barrier_kcal_mol", math.inf))
            barrier_errors.extend(
                (
                    max(abs(forward - expected_forward), abs(reported_forward - forward)),
                    max(abs(reverse - expected_reverse), abs(reported_reverse - reverse)),
                )
            )
        audit.append({"transition_id": transition.get("transition_id"), "edge": sorted(edge), "continuous": continuous, "recomputed_forward": forward, "recomputed_reverse": reverse, "numeric_consistent": numeric_ok})
    true_edges = set(ref_edges)
    true_positive = len(reported_edges & true_edges)
    precision = true_positive / max(1, len(reported_edges))
    recall = true_positive / max(1, len(true_edges))
    topology_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    barrier_mae = float(np.mean(barrier_errors)) if barrier_errors else math.inf
    numeric_fraction = numeric_valid / max(1, len([item for item in final.get("transitions", []) if isinstance(item, Mapping)]))
    return topology_f1, barrier_mae, numeric_fraction, audit


def _coupling_metrics(
    final: Mapping[str, Any], bound: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    assessment = final.get("coupling_assessment") if isinstance(final.get("coupling_assessment"), Mapping) else {}
    support_ids = [str(value) for value in assessment.get("supporting_point_ids", [])]
    support: list[LandscapePoint] = []
    for point_id in support_ids:
        raw = bound.get(point_id)
        if raw is None:
            continue
        support.append(
            LandscapePoint(
                point_id=point_id,
                angles_deg=tuple(map(float, raw["angles_deg"])),
                energy_hartree=float(raw["energy_kcal_mol"]) / HARTREE_TO_KCAL_MOL,
                job_id=str(raw.get("job_id", "")),
                frame_index=raw.get("frame_index"),
                geometry_sha256=str(raw.get("geometry_sha256", "")),
                global_basin_support=raw.get("global_basin_support") is True,
                grid_angles_deg=(
                    tuple(map(float, raw["grid_angles_deg"]))
                    if isinstance(raw.get("grid_angles_deg"), (list, tuple))
                    and len(raw["grid_angles_deg"]) == 2
                    else None
                ),
            )
        )
    if not coupling_support_valid(support):
        return 0.0, {"support_valid": False, "support_count": len(support)}
    support_angles = [point.grid_angles_deg or point.angles_deg for point in support]
    first_values = sorted({round(angles[0] % 360.0, 6) for angles in support_angles})
    second_values = sorted({round(angles[1] % 360.0, 6) for angles in support_angles})
    energy_by_pair = {
        (round(angles[0] % 360.0, 6), round(angles[1] % 360.0, 6)): point.energy_hartree
        * HARTREE_TO_KCAL_MOL
        for point, angles in zip(support, support_angles, strict=True)
    }
    matrix = np.asarray([[energy_by_pair[(first, second)] for second in second_values] for first in first_values])
    recomputed = nonadditivity_rmse(matrix)
    reference_coupling = reference.get("coupling") if isinstance(reference.get("coupling"), Mapping) else {}
    expected = float(reference_coupling.get("nonadditivity_rmse_kcal_mol", math.inf))
    reported = assessment.get("nonadditivity_rmse_kcal_mol")
    reported_error = abs(float(reported) - recomputed) if isinstance(reported, (int, float)) else math.inf
    reference_error = abs(recomputed - expected)
    boolean_ok = assessment.get("coordinates_treated_as_coupled") is reference_coupling.get(
        "coordinates_treated_as_coupled"
    )
    fraction = min(
        float(boolean_ok),
        _linear_fraction(reported_error, 0.02, 0.20),
        _linear_fraction(reference_error, 0.05, 0.50),
    )
    return fraction, {
        "support_valid": True,
        "support_count": len(support),
        "recomputed_nonadditivity_rmse_kcal_mol": recomputed,
        "reference_error_kcal_mol": reference_error,
        "reported_recomputation_error_kcal_mol": reported_error,
        "coupled_boolean_correct": boolean_ok,
    }


def _method_limit_fraction(final: Mapping[str, Any]) -> float:
    uncertainty = final.get("uncertainty") if isinstance(final.get("uncertainty"), Mapping) else {}
    text = " ".join(str(value).lower() for value in uncertainty.get("method_limitations", []))
    checks = (
        "gfn2" in text or "semiempirical" in text,
        "gas" in text,
        "constrain" in text or "relaxed path" in text,
    )
    return sum(checks) / len(checks)


def score_coupled_claims(
    final: Mapping[str, Any],
    evidence: Mapping[str, Any],
    reference: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    scoring_config: Mapping[str, Any],
    *,
    schema_valid: bool,
    semantic_errors: Sequence[str],
) -> dict[str, Any]:
    """Score already verified facts; model names never affect this function."""

    builder = ScoreBuilder(dict(scoring_config))
    integrity_errors = [str(value) for value in evidence.get("integrity_errors", [])]
    trusted_jobs = _trusted_job_ids(evidence)
    if integrity_errors:
        builder.hard_gate(
            "forged_evidence_total_zero",
            reason="registered Task 06 evidence failed hash or path validation",
            evidence=integrity_errors,
            total_zero=True,
        )
    if not trusted_jobs:
        builder.hard_gate(
            "trusted_task_evidence_required",
            reason="no completed Task 06 job has the required xTB protocol",
            total_zero=True,
        )

    bound, point_binding_fraction = _bound_points(final, evidence, trusted_jobs)
    coordinate_mapping, coordinate_fraction = _coordinate_matches(final, evidence)
    declaration_valid = schema_valid and not semantic_errors
    if not declaration_valid:
        builder.notes.append("invalid structured answer: declaration-dependent science was not scored")
        builder.notes.extend(str(value) for value in semantic_errors)

    basin_matches: dict[str, str] = {}
    matched_basin_points: dict[str, Mapping[str, Any]] = {}
    basin_audit: list[dict[str, Any]] = []
    if declaration_valid:
        basin_matches, matched_basin_points, basin_audit = _match_basins(final, bound, reference)
    reference_basins = [item for item in reference.get("basins", []) if isinstance(item, Mapping)]
    total_weight = sum(float(item.get("coverage_weight", 1.0)) for item in reference_basins)
    matched_reference = set(basin_matches.values())
    coverage_weight = sum(
        float(item.get("coverage_weight", 1.0))
        for item in reference_basins
        if str(item.get("reference_id")) in matched_reference
    )
    weighted_basin_coverage = coverage_weight / max(1.0e-12, total_weight)
    state_coverage = len(matched_reference) / max(1, len(reference_basins))
    basin_coverage = 0.5 * (weighted_basin_coverage + state_coverage)

    topology_f1 = 0.0
    barrier_mae = math.inf
    transition_numeric_fraction = 0.0
    transition_audit: list[dict[str, Any]] = []
    if declaration_valid:
        topology_f1, barrier_mae, transition_numeric_fraction, transition_audit = _transition_metrics(
            final, bound, basin_matches, matched_basin_points, reference
        )
    coupling_fraction, coupling_audit = (
        _coupling_metrics(final, bound, reference) if declaration_valid else (0.0, {"support_valid": False})
    )

    builder.direct("key_internal_coordinates", coordinate_fraction if declaration_valid else 0.0, evidence={"coordinate_mapping": coordinate_mapping}, reason="central ring-link bonds are compared without trusting coordinate names or signs")
    builder.direct("low_energy_basin_coverage", basin_coverage, evidence={"matches": basin_audit, "weighted_fraction": weighted_basin_coverage, "state_fraction": state_coverage, "combined_fraction": basin_coverage}, reason="coverage requires both thermodynamic weight and distinct geometry-matched reference basins")
    ordering_fraction = _ordering_fraction(basin_matches, matched_basin_points, reference) if declaration_valid else 0.0
    if len(basin_matches) < 2:
        ordering_fraction = 0.0
    builder.direct("global_basin_and_ordering", ordering_fraction, evidence={"matched_basin_count": len(basin_matches)}, reason="global basin and pair ordering use raw energies of geometry-matched states")
    energy_mae = _basin_energy_mae(basin_matches, matched_basin_points, reference) if declaration_valid else math.inf
    if len(basin_matches) < 2:
        energy_mae = math.inf
    builder.linear("relative_basin_energies", energy_mae, evidence={"mae_kcal_mol": energy_mae}, reason="relative energies are recomputed from raw matched states")
    builder.direct("transition_network_topology", topology_f1, evidence={"transition_audit": transition_audit}, reason="only continuous raw-geometry paths between matched basins define network edges")
    builder.linear("directional_barriers", barrier_mae, evidence={"mae_kcal_mol": barrier_mae}, reason="forward and reverse barriers are recomputed from path maxima and both endpoint energies")
    builder.direct("coupling_nonadditivity", coupling_fraction, evidence=coupling_audit, reason="quantitative coupling requires a complete, wide, global-region cross-sample")

    early_hypotheses = int(trajectory.get("coordinate_hypotheses_before_dense_scan", 0) or 0)
    builder.direct("early_coordinate_hypothesis_diversity", min(1.0, early_hypotheses / 2.0), evidence={"observable_hypotheses_before_dense_scan": early_hypotheses}, reason="the trajectory must identify the two-coordinate minimum sufficient set before committing dense compute")
    independent_starts = int(trajectory.get("independent_start_count", 0) or 0)
    start_fraction = min(1.0, independent_starts / 3.0)
    discovered_basin_diversity = min(1.0, len(basin_matches) / 3.0)
    seeding_fraction = start_fraction * discovered_basin_diversity
    builder.direct("independent_start_and_basin_seeding", seeding_fraction, evidence={"independent_start_count": independent_starts, "matched_reference_basin_count": len(basin_matches), "discovered_basin_diversity": discovered_basin_diversity}, reason="independent starts receive credit only when they produce geometry-matched basin diversity")
    cross_fraction = float(trajectory.get("cross_sample_information_fraction", 0.0) or 0.0)
    builder.direct("cross_sample_information_value", min(cross_fraction, float(coupling_audit.get("support_valid") is True)), evidence={"observable_fraction": cross_fraction, **coupling_audit}, reason="cross-sampling credit follows usable two-coordinate evidence")
    checks = {str(value).lower() for value in trajectory.get("quality_checks", [])}
    adaptive_fraction = float(trajectory.get("adaptive_verification_performed") is True) * min(1.0, len(checks & {"coordinate", "coverage", "branch", "energy"}) / 4.0)
    builder.direct("adaptive_validation", adaptive_fraction, evidence={"quality_checks": sorted(checks)}, reason="targeted validation and checks must be observable after feedback")
    jobs_used = int(trajectory.get("jobs_used", 0) or 0)
    scan_points_used = int(trajectory.get("scan_points_used", 0) or 0)
    high_information_jobs = int(trajectory.get("high_information_jobs", 0) or 0)
    useful_fraction = high_information_jobs / max(1, jobs_used)
    scientific_completion = (coordinate_fraction + basin_coverage + topology_f1 + coupling_fraction) / 4.0
    within_budget = float(jobs_used <= 12 and scan_points_used <= 72)
    efficiency = min(1.0, useful_fraction / 0.75) * scientific_completion * within_budget
    builder.direct("scientific_information_efficiency", efficiency, evidence={"jobs_used": jobs_used, "scan_points_used": scan_points_used, "high_information_jobs": high_information_jobs, "scientific_completion": scientific_completion}, reason="speed earns credit only when it produces correct, independent scientific information")

    if declaration_valid:
        coverage = final.get("coverage_assessment") if isinstance(final.get("coverage_assessment"), Mapping) else {}
        missing_basin_count = max(0, len(reference_basins) - len(matched_reference))
        upper = coverage.get("omitted_low_energy_basin_count_upper_bound")
        conservative = isinstance(upper, int) and upper >= missing_basin_count
        closeness = 1.0 if conservative and int(upper) - missing_basin_count <= 1 else (0.5 if conservative else 0.0)
        unresolved = bool(str(coverage.get("largest_unresolved_region_description", "")).strip())
        builder.direct("omitted_basin_and_path_risk", closeness * float(unresolved), evidence={"actual_missing_reference_basins": missing_basin_count, "reported_upper_bound": upper}, reason="the declared bound must conservatively cover geometry-derived omissions")
        uncertainty = final.get("uncertainty") if isinstance(final.get("uncertainty"), Mapping) else {}
        numeric_bounds = [uncertainty.get(name) for name in ("basin_energy_error_estimate_kcal_mol", "barrier_error_estimate_kcal_mol", "coupling_error_estimate_kcal_mol")]
        numeric_fraction = sum(isinstance(value, (int, float)) and float(value) >= 0.0 for value in numeric_bounds) / 3.0
        limitation_text = " ".join(str(value).lower() for value in uncertainty.get("method_limitations", []))
        risk_named = float(any(word in limitation_text for word in ("sparse", "branch", "coupl")))
        builder.direct("sparse_sampling_branch_coupling_bounds", numeric_fraction * risk_named, evidence={"reported_bounds": numeric_bounds, "sampling_risk_named": bool(risk_named)}, reason="numeric bounds must be accompanied by the unresolved sampling or coupling risk")
        builder.direct("method_scope_limitations", _method_limit_fraction(final), evidence={"method_limitations": uncertainty.get("method_limitations", [])}, reason="GFN2-xTB, gas-phase, and constrained-path scope are independently required")

    cited_jobs = {str(value) for value in final.get("evidence_job_ids", [])}
    trusted_fraction = len(cited_jobs & trusted_jobs) / max(1, len(cited_jobs)) if cited_jobs else 0.0
    builder.direct("method_hash_validity", trusted_fraction, evidence={"trusted_cited_jobs": sorted(cited_jobs & trusted_jobs), "cited_jobs": sorted(cited_jobs)}, reason="only exact-hash jobs with the fixed protocol count")

    angle_consistent = 0
    angle_checked = 0
    point_by_id = {str(item.get("point_id", "")): item for item in final.get("sampled_points", []) if isinstance(item, Mapping)}
    for point_id, raw in bound.items():
        declared = point_by_id.get(point_id, {})
        reported_by_id = {
            str(item.get("coordinate_id", "")): float(item.get("actual_angle_deg", math.inf))
            for item in declared.get("coordinate_values", [])
            if isinstance(item, Mapping)
        }
        point_ok = True
        used = 0
        for coordinate_id, private_index in coordinate_mapping.items():
            if coordinate_id not in reported_by_id:
                continue
            raw_angle = float(raw["angles_deg"][private_index])
            reported_angle = reported_by_id[coordinate_id]
            error = min(_circular_error(reported_angle, raw_angle), _circular_error(reported_angle, -raw_angle))
            point_ok &= error <= 5.0
            used += 1
        if used:
            angle_consistent += int(point_ok)
            angle_checked += 1
    angle_fraction = angle_consistent / max(1, angle_checked)
    geometry_fraction = 0.5 * point_binding_fraction + 0.5 * angle_fraction
    builder.direct("geometry_coordinate_basin_binding", geometry_fraction, evidence={"point_binding_fraction": point_binding_fraction, "coordinate_value_fraction": angle_fraction}, reason="job/frame/path/hash binding and geometry-derived coordinate values are checked separately")

    submitted_points = [item for item in final.get("sampled_points", []) if isinstance(item, Mapping)]
    point_energy_consistent = 0
    for point in submitted_points:
        raw = bound.get(str(point.get("point_id", "")))
        if raw is not None and abs(float(point.get("relative_energy_kcal_mol", math.inf)) - float(raw["energy_kcal_mol"])) <= 0.05:
            point_energy_consistent += 1
    point_energy_fraction = point_energy_consistent / max(1, len(submitted_points))
    numeric_recomputation_fraction = 0.5 * point_energy_fraction + 0.5 * transition_numeric_fraction
    builder.direct("energy_path_barrier_recomputation", numeric_recomputation_fraction, evidence={"point_energy_fraction": point_energy_fraction, "transition_numeric_fraction": transition_numeric_fraction}, reason="reported energies and both barriers are checked against raw path frames")

    report = builder.finalize(
        "task_06",
        metadata={
            "scientific_problem_id": "task_06_coupled_torsional_landscape",
            "schema_valid": schema_valid,
            "semantic_errors": list(semantic_errors),
            "trusted_job_count": len(trusted_jobs),
            "budget_exhausted": evidence.get("budget_exhausted") is True,
        },
    )
    total = float(report.get("total_score", 0.0))
    report["score"] = total
    report["task_local_score"] = total
    report["scientific_basis_score"] = total
    report["comparison_score"] = total
    return report


def _extract_verified_evidence(
    jobs: Mapping[str, Any],
    public_seed: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    seed = read_xyz(public_seed)
    definitions = canonical_coordinates(seed.elements, seed.coordinates)
    trusted: dict[str, dict[str, Any]] = {}
    points: list[dict[str, Any]] = []
    observed_coordinate_bonds: set[tuple[int, int]] = set()
    for job_id, job in jobs.items():
        protocol_valid = bool(
            job_valid_termination(job)
            and protocol_matches(
                job,
                version="6.7.1",
                gfn=2,
                solvent_model="none",
                solvent=None,
                charge=0,
                uhf=0,
            )
        )
        trusted[str(job_id)] = {
            "protocol_valid": protocol_valid,
            "status": "completed" if job_valid_termination(job) else str(job.record.get("status", "")),
        }
        if not protocol_valid:
            continue
        scan = job.parsed.get("scan") if isinstance(job.parsed.get("scan"), Mapping) else {}
        scan_points = [point for point in scan.get("points", []) if isinstance(point, Mapping)]
        source_files = [str(value) for value in scan.get("source_files", [])]
        if scan_points and source_files:
            source_name = next((name for name in source_files if (job.workdir / name).is_file()), "")
            source = job.workdir / source_name
            try:
                frames = read_xyz_frames(source)
            except (OSError, ValueError):
                frames = []
            source_hash = sha256_file(source) if source.is_file() else ""
            logical_workdir = str(job.record.get("workdir") or f"xtb_jobs/{job_id}").replace("\\", "/")
            request = job.record.get("request") if isinstance(job.record.get("request"), Mapping) else {}
            detailed_name = str(request.get("detailed_input") or "detailed.inp")
            detailed_path = job.workdir / Path(detailed_name).name
            detailed_text = detailed_path.read_text(encoding="utf-8", errors="replace") if detailed_path.is_file() else ""
            fixed_values: dict[int, float] = {}
            for match in re.finditer(
                r"dihedral\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([-+0-9.Ee]+|auto)",
                detailed_text,
                re.I,
            ):
                atoms = tuple(int(match.group(index)) for index in range(1, 5))
                central = {atoms[1], atoms[2]}
                coordinate_index = next(
                    (
                        index
                        for index, definition in enumerate(definitions)
                        if central == {definition.central_bond[0] + 1, definition.central_bond[1] + 1}
                    ),
                    None,
                )
                if coordinate_index is not None and match.group(5).lower() != "auto":
                    fixed_values[coordinate_index] = float(match.group(5))
                if coordinate_index is not None:
                    observed_coordinate_bonds.add(tuple(sorted((atoms[1], atoms[2]))))
            scanned_atoms = tuple(
                int(value) for value in (scan.get("coordinate_atom_ids") or [])
            )
            scanned_index = next(
                (
                    index
                    for index, definition in enumerate(definitions)
                    if len(scanned_atoms) == 4
                    and {scanned_atoms[1], scanned_atoms[2]}
                    == {definition.central_bond[0] + 1, definition.central_bond[1] + 1}
                ),
                None,
            )
            for parsed_point in scan_points:
                frame_index = parsed_point.get("geometry_frame_index")
                if (
                    not isinstance(frame_index, int)
                    or frame_index < 0
                    or frame_index >= len(frames)
                    or parsed_point.get("converged") is not True
                    or not isinstance(parsed_point.get("energy_hartree"), (int, float))
                ):
                    continue
                frame = frames[frame_index]
                if frame.elements != seed.elements:
                    continue
                actual_angles = list(coordinate_values(frame.coordinates, definitions))
                grid_angles = list(actual_angles)
                for coordinate_index, fixed_value in fixed_values.items():
                    grid_angles[coordinate_index] = fixed_value
                if scanned_index is not None and isinstance(parsed_point.get("target_angle_deg"), (int, float)):
                    grid_angles[scanned_index] = float(parsed_point["target_angle_deg"])
                points.append(
                    {
                        "job_id": str(job_id),
                        "frame_index": frame_index,
                        "geometry_path": f"{logical_workdir}/{source_name}",
                        "geometry_sha256": source_hash,
                        "angles_deg": actual_angles,
                        "grid_angles_deg": grid_angles,
                        "energy_hartree": float(parsed_point["energy_hartree"]),
                        "elements": list(frame.elements),
                        "coordinates_angstrom": frame.coordinates.tolist(),
                    }
                )
            continue
        optimized = job.workdir / "xtbopt.xyz"
        energy = (job.parsed.get("energies") or {}).get("total_energy_hartree")
        if optimized.is_file() and isinstance(energy, (int, float)):
            try:
                frame = read_xyz(optimized)
            except (OSError, ValueError):
                continue
            if frame.elements != seed.elements:
                continue
            logical_workdir = str(job.record.get("workdir") or f"xtb_jobs/{job_id}").replace("\\", "/")
            points.append(
                {
                    "job_id": str(job_id),
                    "frame_index": None,
                    "geometry_path": f"{logical_workdir}/xtbopt.xyz",
                    "geometry_sha256": sha256_file(optimized),
                    "angles_deg": list(coordinate_values(frame.coordinates, definitions)),
                    "energy_hartree": float(energy),
                    "elements": list(frame.elements),
                    "coordinates_angstrom": frame.coordinates.tolist(),
                }
            )
    if points:
        zero = min(float(point["energy_hartree"]) for point in points)
        for point in points:
            point["energy_kcal_mol"] = (float(point.pop("energy_hartree")) - zero) * HARTREE_TO_KCAL_MOL
        _mark_global_region(points)
    ledger = manifest.get("budget_ledger") if isinstance(manifest.get("budget_ledger"), Mapping) else {}
    return {
        "canonical_central_bonds": [
            [definition.central_bond[0] + 1, definition.central_bond[1] + 1]
            for definition in definitions
        ],
        "trusted_jobs": trusted,
        "points": points,
        "observed_coordinate_bonds": [list(pair) for pair in sorted(observed_coordinate_bonds)],
        "budget_exhausted": bool(ledger.get("compute_exhausted") or ledger.get("budget_exhausted")),
    }


def _observable_trajectory(
    manifest: Mapping[str, Any],
    final: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    jobs = [job for job in manifest.get("jobs", []) if isinstance(job, Mapping)]
    coordinate_bonds: list[tuple[int, int]] = [
        tuple(map(int, pair))
        for pair in evidence.get("observed_coordinate_bonds", [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]
    independent_inputs: set[str] = set()
    scan_points = 0
    high_information_jobs = 0
    for job in jobs:
        geometry_hash = str(job.get("input_geometry_sha256") or "")
        if geometry_hash:
            independent_inputs.add(geometry_hash)
        request = job.get("request") if isinstance(job.get("request"), Mapping) else {}
        detailed = str(request.get("detailed_input") or "")
        if detailed:
            high_information_jobs += 1
        requested_points = int(job.get("scan_point_count", 0) or 0)
        scan_points += requested_points
        atoms = job.get("scan_coordinate_atom_ids")
        if isinstance(atoms, list) and len(atoms) == 4:
            coordinate_bonds.append((int(atoms[1]), int(atoms[2])))
    # Some minimal-Harness manifests record scan counts only in the ledger.
    ledger = manifest.get("budget_ledger") if isinstance(manifest.get("budget_ledger"), Mapping) else {}
    totals = ledger.get("totals") if isinstance(ledger.get("totals"), Mapping) else ledger
    scan_points = max(scan_points, int(totals.get("xtb_scan_points", 0) or 0))
    if not coordinate_bonds:
        coordinate_bonds = [
            tuple(map(int, item.get("central_bond_atom_ids", [])))
            for item in final.get("coordinate_hypotheses", [])
            if isinstance(item, Mapping) and len(item.get("central_bond_atom_ids", [])) == 2
        ]
    support = (
        final.get("coupling_assessment", {}).get("supporting_point_ids", [])
        if isinstance(final.get("coupling_assessment"), Mapping)
        else []
    )
    cross_fraction = min(1.0, len(set(map(str, support))) / 9.0)
    distinct_coordinates = len({frozenset(pair) for pair in coordinate_bonds})
    adaptive = len(jobs) >= 3 and distinct_coordinates >= 2
    cited_jobs = {str(value) for value in final.get("evidence_job_ids", [])}
    completed_cited_jobs = {
        str(job.get("job_id"))
        for job in jobs
        if str(job.get("status")) == "completed" and str(job.get("job_id")) in cited_jobs
    }
    return {
        "coordinate_hypotheses_before_dense_scan": distinct_coordinates,
        "independent_start_count": len(independent_inputs),
        "cross_sample_information_fraction": cross_fraction,
        "adaptive_verification_performed": adaptive,
        "quality_checks": ["coordinate", "coverage", "branch", "energy"] if adaptive else [],
        "jobs_used": max(len(jobs), int(totals.get("xtb_jobs", 0) or 0)),
        "scan_points_used": scan_points,
        "high_information_jobs": max(
            high_information_jobs,
            len(completed_cited_jobs),
            min(len(jobs), distinct_coordinates + int(cross_fraction > 0)),
        ),
    }


def score_task06_coupled(
    submission_dir: Path,
    task_dir: Path,
    reference_dir: Path,
    run_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and verify a real submission before invoking the pure scorer."""

    submission_root = Path(submission_dir).resolve()
    final = load_json_optional(submission_root / "final_answer.json")
    schema_errors = validate_final_answer(task_dir, final)
    semantic_errors = validate_coupled_submission(final) if not schema_errors else []
    jobs, integrity_errors, manifest = verify_and_parse_jobs(submission_root)
    evidence = _extract_verified_evidence(
        jobs,
        task_dir / "data" / "inputs" / "terpyridine_seed.xyz",
        manifest,
    )
    evidence["integrity_errors"] = integrity_errors
    trajectory = _observable_trajectory(manifest, final, evidence)
    if isinstance(run_artifacts, Mapping):
        trajectory_override = run_artifacts.get("trajectory_metrics")
        if isinstance(trajectory_override, Mapping):
            trajectory.update(trajectory_override)
    scoring = load_yaml(task_dir / "config" / "scoring.yaml", default={}) or {}
    reference = reference_manifest(reference_dir)
    report = score_coupled_claims(
        final,
        evidence,
        reference,
        trajectory,
        scoring,
        schema_valid=not schema_errors,
        semantic_errors=semantic_errors,
    )
    report.setdefault("notes", []).extend(schema_errors)
    report["contract_errors"] = list(schema_errors) + list(semantic_errors) + list(integrity_errors)
    report["legacy_evidence"] = {}
    report.setdefault("metadata", {})["required_output_status"] = {
        "final_answer.json": (submission_root / "final_answer.json").is_file(),
        "report.md": (submission_root / "report.md").is_file(),
    }
    return report

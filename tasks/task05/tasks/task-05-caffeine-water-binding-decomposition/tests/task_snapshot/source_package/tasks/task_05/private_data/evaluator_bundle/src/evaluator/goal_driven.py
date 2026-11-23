"""Route-neutral v1.13 scorer for Task 05."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from common.metrics import HARTREE_TO_KCAL_MOL, kendall_tau_b_with_reference_ties
from common.geometry import kabsch_rmsd
from common.trajectory_scoring import job_events_from_evidence, score_observable_allocation
from evaluator.core import (
    ScoreBuilder,
    job_energy,
    job_valid_termination,
    load_json_optional,
    load_yaml_optional,
    protocol_matches,
    read_csv_rows,
    reference_manifest,
    validate_final_answer,
    verify_and_parse_jobs,
)
try:  # unified Golden staging uses task2.py; standalone bundle uses task.py
    from evaluator.task2 import (
        _derived_pose_clusters,
        _fragment_geometry_error,
        _input_geometry,
        _job_geometry,
        _public_fragment_mapping,
        _reference_data,
        classify_caffeine_water_contact,
    )
except ImportError:  # pragma: no cover - selected by deployment layout
    from evaluator.task import (
        _derived_pose_clusters,
        _fragment_geometry_error,
        _input_geometry,
        _job_geometry,
        _public_fragment_mapping,
        _reference_data,
        classify_caffeine_water_contact,
    )


def _state_match(members: set[str], references: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not members:
        return None
    ranked = sorted(
        references,
        key=lambda ref: (
            -len(members & set(map(str, ref.get("member_seed_ids", [])))) / max(1, len(members | set(map(str, ref.get("member_seed_ids", []))))),
            str(ref.get("pose_id")),
        ),
    )
    return ranked[0] if ranked and members & set(map(str, ranked[0].get("member_seed_ids", []))) else None


def _geometry_hash_bound(pose: dict[str, Any], job: Any) -> bool:
    if job is None:
        return False
    claimed = str(pose.get("representative_geometry_sha256") or "")
    name = Path(str(pose.get("representative_geometry_path") or "")).name
    return bool(claimed and name and any(Path(str(key)).name == name and str(value) == claimed for key, value in (job.record.get("files_sha256") or {}).items()))


def _job_by_atom_count(ids: list[str], jobs: dict[str, Any], atom_count: int) -> Any:
    for job_id in ids:
        job = jobs.get(str(job_id))
        geometry = _job_geometry(job)
        if geometry is not None and len(geometry.elements) == atom_count:
            return job
    return None


def _reported_decomposition_consistent(
    *,
    binding: float,
    interaction: float,
    deformation: float,
    claimed_binding: Any,
    claimed_interaction: Any,
    claimed_deformation: Any,
    tolerance_kcal_mol: float = 0.01,
) -> bool:
    """Check one pose locally while allowing ordinary decimal rounding."""
    claimed = (claimed_binding, claimed_interaction, claimed_deformation)
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in claimed):
        return False
    return bool(
        abs(float(claimed_binding) - binding) <= tolerance_kcal_mol
        and abs(float(claimed_interaction) - interaction) <= tolerance_kcal_mol
        and abs(float(claimed_deformation) - deformation) <= tolerance_kcal_mol
        and abs(float(claimed_binding) - float(claimed_interaction) - float(claimed_deformation))
        <= tolerance_kcal_mol
    )


def _mapped_fragment_rigid_rmsd(
    complex_coordinates: Any,
    complex_elements: Any,
    fragment_coordinates: Any,
    fragment_elements: Any,
    mapping: list[tuple[int, int, str]],
) -> float:
    """Verify a frozen fragment up to the rigid placement used to form a complex."""
    if not mapping:
        return math.inf
    try:
        ordered_coordinates = []
        ordered_elements = []
        for complex_atom, fragment_atom, expected_element in mapping:
            if fragment_atom != len(ordered_coordinates) + 1:
                return math.inf
            element = str(complex_elements[complex_atom - 1])
            if expected_element and element.upper() != expected_element.upper():
                return math.inf
            ordered_coordinates.append(complex_coordinates[complex_atom - 1])
            ordered_elements.append(element.upper())
        if tuple(ordered_elements) != tuple(str(value).upper() for value in fragment_elements):
            return math.inf
        return float(kabsch_rmsd(ordered_coordinates, fragment_coordinates))
    except (IndexError, TypeError, ValueError):
        return math.inf


def _screening_fragment_geometry_error(
    complex_job: Any, fragment_job: Any, mapping: list[tuple[int, int, str]]
) -> float:
    complex_geometry = _input_geometry(complex_job) if complex_job is not None else None
    fragment_geometry = _input_geometry(fragment_job) if fragment_job is not None else None
    if complex_geometry is None or fragment_geometry is None:
        return math.inf
    return _mapped_fragment_rigid_rmsd(
        complex_geometry.coordinates,
        complex_geometry.elements,
        fragment_geometry.coordinates,
        fragment_geometry.elements,
        mapping,
    )


def _linear_fraction(error: float, full: float, zero: float) -> float:
    if not math.isfinite(float(error)):
        return 0.0
    if error <= full:
        return 1.0
    if zero <= full or error >= zero:
        return 0.0
    return (zero - error) / (zero - full)


def _job_operation(job: Any) -> str:
    return str((job.record if job is not None else {}).get("protocol") or "").lower()


def _initial_energy_hartree(job: Any) -> float | None:
    """Read the pre-optimization energy that xTB prints before its opt cycle."""

    if job is None:
        return None
    try:
        stdout = (job.workdir / "stdout.txt").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    values = re.findall(
        r"::\s*total energy\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\s+Eh",
        stdout,
        re.IGNORECASE,
    )
    if not values:
        return None
    try:
        return float(values[0].replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _decomposition_evidence_tier(
    *,
    complex_optimized: bool,
    fragment_geometry_bound: bool,
    full_terms_consistent: bool,
    screening_terms_consistent: bool,
    screening_interaction_consistent: bool,
) -> dict[str, Any]:
    if complex_optimized and fragment_geometry_bound and full_terms_consistent:
        return {
            "tier": "full_local_decomposition",
            "binding_cap": 1.0,
            "interaction_cap": 1.0,
            "deformation_cap": 1.0,
        }
    if complex_optimized and fragment_geometry_bound and screening_interaction_consistent:
        return {
            "tier": "optimized_complex_interaction_only",
            "binding_cap": 0.0,
            "interaction_cap": 1.0,
            "deformation_cap": 0.0,
        }
    if fragment_geometry_bound and screening_terms_consistent:
        return {
            "tier": "fixed_input_screening_decomposition",
            "binding_cap": 0.40,
            "interaction_cap": 0.40,
            "deformation_cap": 0.40,
        }
    if fragment_geometry_bound and screening_interaction_consistent:
        return {
            "tier": "fixed_input_interaction_only",
            "binding_cap": 0.0,
            "interaction_cap": 0.40,
            "deformation_cap": 0.0,
        }
    return {
        "tier": "insufficient_decomposition_evidence",
        "binding_cap": 0.0,
        "interaction_cap": 0.0,
        "deformation_cap": 0.0,
    }


def _reported_value_consistent(claimed: Any, recomputed: float | None) -> bool:
    return bool(
        recomputed is not None
        and isinstance(claimed, (int, float))
        and math.isfinite(float(claimed))
        and abs(float(claimed) - recomputed) <= 0.01
    )


def _turn_efficiency(
    harness_manifest: dict[str, Any] | None, maximum_turns: int
) -> tuple[float, dict[str, Any]]:
    totals = (
        (((harness_manifest or {}).get("budget_ledger") or {}).get("totals") or {})
        if isinstance(harness_manifest, dict)
        else {}
    )
    turns = totals.get("agent_turns")
    if not isinstance(turns, int) or turns < 0 or maximum_turns <= 0:
        return 1.0, {"observed": False}
    ratio = min(1.0, turns / maximum_turns)
    fraction = 1.0 if ratio <= 0.5 else max(0.4, 1.6 - 1.2 * ratio)
    return fraction, {
        "observed": True,
        "agent_turns": turns,
        "maximum_agent_turns": maximum_turns,
        "usage_ratio": ratio,
    }


def score_task2_goal_driven(
    task_dir: Path,
    submission_dir: Path,
    reference_dir: Path,
    builder: ScoreBuilder,
    *,
    harness_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_yaml_optional(task_dir / "config" / "task.yaml")
    final = load_json_optional(submission_dir / "final_answer.json")
    schema_errors = validate_final_answer(task_dir, final)
    builder.notes.extend(schema_errors)
    jobs, integrity_errors, manifest = verify_and_parse_jobs(submission_dir)
    if integrity_errors:
        builder.hard_gate("forged_evidence_total_zero", reason="registered xTB evidence failed integrity validation", evidence=integrity_errors, total_zero=True)
    trusted = {
        job_id: job
        for job_id, job in jobs.items()
        if job_valid_termination(job)
        and protocol_matches(job, version="6.7.1", gfn=2, solvent_model="none", solvent=None, charge=0, uhf=0)
    }
    if not trusted:
        builder.hard_gate("trusted_task_evidence_required", reason="no valid gas-phase neutral GFN2-xTB evidence", total_zero=True)

    reference = _reference_data(reference_manifest(reference_dir))
    refs = [value for value in reference.get("poses", []) if isinstance(value, dict)]
    derived_clusters, job_by_seed = _derived_pose_clusters(task_dir, trusted, 0.25, 0.35)
    evaluated = set(job_by_seed)
    poses = [value for value in final.get("poses", []) if isinstance(value, dict)] if not schema_errors else []
    pose_info: dict[str, dict[str, Any]] = {}
    energy_contributions: dict[str, list[dict[str, Any]]] = {
        "binding": [],
        "interaction": [],
        "deformation": [],
    }
    closure_matches = fragment_matches = geometry_matches = 0
    fragment_rows = read_csv_rows(task_dir / "data" / "inputs" / "fragment_manifest.csv")

    for pose in poses:
        pose_id = str(pose.get("pose_id") or "")
        members = set(map(str, pose.get("member_candidate_ids", [])))
        ref = _state_match(members, refs)
        complex_job = trusted.get(str(pose.get("representative_complex_job_id") or ""))
        fragment_ids = [str(value) for value in pose.get("frozen_fragment_job_ids", [])]
        monomer_ids = [str(value) for value in pose.get("monomer_reference_job_ids", [])]
        frag_a = _job_by_atom_count(fragment_ids, trusted, 24)
        frag_b = _job_by_atom_count(fragment_ids, trusted, 3)
        mono_a = _job_by_atom_count(monomer_ids, trusted, 24)
        mono_b = _job_by_atom_count(monomer_ids, trusted, 3)
        seed_for_mapping = sorted(members)[0] if members else str((ref or {}).get("pose_id") or "")
        map_a = _public_fragment_mapping(fragment_rows, seed_for_mapping, "A")
        map_b = _public_fragment_mapping(fragment_rows, seed_for_mapping, "B")
        fragment_ok = bool(
            complex_job and frag_a and frag_b
            and _fragment_geometry_error(complex_job, frag_a, map_a) <= 1e-6
            and _fragment_geometry_error(complex_job, frag_b, map_b) <= 1e-6
        )
        e_complex = job_energy(complex_job, "total_energy_hartree")
        e_frag_a = job_energy(frag_a, "total_energy_hartree")
        e_frag_b = job_energy(frag_b, "total_energy_hartree")
        e_mono_a = job_energy(mono_a, "total_energy_hartree")
        e_mono_b = job_energy(mono_b, "total_energy_hartree")
        complex_optimized = bool(
            complex_job
            and _job_operation(complex_job) == "opt"
            and (complex_job.parsed.get("status") or {}).get("optimization_converged")
        )
        monomers_optimized = bool(
            mono_a
            and mono_b
            and _job_operation(mono_a) == "opt"
            and _job_operation(mono_b) == "opt"
            and (mono_a.parsed.get("status") or {}).get("optimization_converged")
            and (mono_b.parsed.get("status") or {}).get("optimization_converged")
        )
        interaction = deformation = binding = None
        if fragment_ok and all(
            value is not None for value in (e_complex, e_frag_a, e_frag_b)
        ):
            interaction = (float(e_complex) - float(e_frag_a) - float(e_frag_b)) * HARTREE_TO_KCAL_MOL
        if interaction is not None and monomers_optimized and all(
            value is not None for value in (e_mono_a, e_mono_b)
        ):
            deformation = ((float(e_frag_a) - float(e_mono_a)) + (float(e_frag_b) - float(e_mono_b))) * HARTREE_TO_KCAL_MOL
            binding = interaction + deformation

        screening_fragment_ok = bool(
            complex_job
            and mono_a
            and mono_b
            and _job_operation(complex_job) == "sp"
            and _screening_fragment_geometry_error(complex_job, mono_a, map_a) <= 1e-6
            and _screening_fragment_geometry_error(complex_job, mono_b, map_b) <= 1e-6
        )
        frozen_a = (
            _initial_energy_hartree(mono_a)
            if _job_operation(mono_a) == "opt"
            else e_mono_a
        )
        frozen_b = (
            _initial_energy_hartree(mono_b)
            if _job_operation(mono_b) == "opt"
            else e_mono_b
        )
        screening_interaction = screening_deformation = screening_binding = None
        if screening_fragment_ok and all(
            value is not None for value in (e_complex, frozen_a, frozen_b)
        ):
            screening_interaction = (
                float(e_complex) - float(frozen_a) - float(frozen_b)
            ) * HARTREE_TO_KCAL_MOL
            if monomers_optimized:
                screening_deformation = (
                    (float(frozen_a) - float(e_mono_a))
                    + (float(frozen_b) - float(e_mono_b))
                ) * HARTREE_TO_KCAL_MOL
                screening_binding = screening_interaction + screening_deformation
        claimed_values = (
            pose.get("binding_energy_kcal_mol"),
            pose.get("interaction_energy_kcal_mol"),
            pose.get("deformation_energy_kcal_mol"),
        )
        consistent = bool(binding is not None) and _reported_decomposition_consistent(
            binding=float(binding or 0.0),
            interaction=float(interaction or 0.0),
            deformation=float(deformation or 0.0),
            claimed_binding=claimed_values[0],
            claimed_interaction=claimed_values[1],
            claimed_deformation=claimed_values[2],
        )
        screening_interaction_consistent = _reported_value_consistent(
            claimed_values[1], screening_interaction
        )
        optimized_interaction_consistent = _reported_value_consistent(
            claimed_values[1], interaction
        )
        screening_consistent = bool(
            screening_binding is not None
            and screening_deformation is not None
            and _reported_decomposition_consistent(
                binding=float(screening_binding),
                interaction=float(screening_interaction),
                deformation=float(screening_deformation),
                claimed_binding=claimed_values[0],
                claimed_interaction=claimed_values[1],
                claimed_deformation=claimed_values[2],
            )
        )
        tier = _decomposition_evidence_tier(
            complex_optimized=complex_optimized,
            fragment_geometry_bound=fragment_ok or screening_fragment_ok,
            full_terms_consistent=consistent,
            screening_terms_consistent=screening_consistent,
            screening_interaction_consistent=(
                optimized_interaction_consistent
                if complex_optimized
                else screening_interaction_consistent
            ),
        )
        selected_values = {
            "binding": binding,
            "interaction": interaction,
            "deformation": deformation,
        }
        if tier["tier"] == "optimized_complex_interaction_only":
            selected_values = {
                "binding": None,
                "interaction": interaction,
                "deformation": None,
            }
        elif tier["tier"] != "full_local_decomposition":
            selected_values = {
                "binding": screening_binding,
                "interaction": screening_interaction,
                "deformation": screening_deformation,
            }
        if ref:
            for key in ("binding", "interaction", "deformation"):
                value = selected_values[key]
                cap = float(tier[f"{key}_cap"])
                if value is not None and cap > 0.0:
                    energy_contributions[key].append(
                        {
                            "pose_id": pose_id,
                            "reference_id": str(ref["pose_id"]),
                            "error": abs(float(value) - float(ref[f"{key}_kcal_mol"])),
                            "cap": cap,
                            "tier": tier["tier"],
                        }
                    )
        closure_matches += int(consistent)
        fragment_matches += int(fragment_ok or screening_fragment_ok)
        geometry_matches += int(_geometry_hash_bound(pose, complex_job))
        geometry = _job_geometry(complex_job)
        family, contact_metrics = classify_caffeine_water_contact(geometry) if geometry is not None else ("invalid_or_collapsed", {})
        pose_info[pose_id] = {
            "pose": pose,
            "reference": ref,
            "reference_id": str((ref or {}).get("pose_id") or ""),
            "complex_job": complex_job,
            "interaction": interaction,
            "deformation": deformation,
            "binding": binding,
            "screening_interaction": screening_interaction,
            "screening_deformation": screening_deformation,
            "screening_binding": screening_binding,
            "decomposition_consistent": consistent,
            "fragment_ok": fragment_ok,
            "screening_fragment_ok": screening_fragment_ok,
            "complex_optimized": complex_optimized,
            "evidence_tier": tier,
            "selected_values": selected_values,
            "family": family,
            "contact_metrics": contact_metrics,
        }

    full_refs = {
        info["reference_id"]
        for info in pose_info.values()
        if info["reference_id"]
        and info["evidence_tier"]["tier"] == "full_local_decomposition"
    }
    independent_coverage = len(full_refs) / max(1, len(refs))
    families = {info["family"] for info in pose_info.values()} - {"invalid_or_collapsed"}
    reference_families = {
        str(value)
        for value in (reference.get("post_opt_contact_families") or {}).values()
        if str(value) and str(value) != "invalid_or_collapsed"
    }
    reference_seed_ids = {
        str(seed_id)
        for reference_pose in refs
        for seed_id in reference_pose.get("member_seed_ids", [])
        if str(seed_id)
    }
    contact_family_target = max(1, len(reference_families))
    evaluated_candidate_target = max(1, len(reference_seed_ids))
    complete_reference_route = bool(
        independent_coverage >= 1.0 and reference_families <= families
    )
    contact_family_fraction = (
        1.0 if complete_reference_route else min(1.0, len(families) / 4.0)
    )
    builder.direct(
        "contact_family_coverage",
        contact_family_fraction,
        evidence={
            "geometry_derived_contact_families": sorted(families),
            "reference_supported_family_target": contact_family_target,
        },
        reason="contact-family coverage is reconstructed from trusted complex geometries",
    )
    builder.direct(
        "independent_basin_coverage",
        independent_coverage,
        evidence={
            "full_local_reference_basins": sorted(full_refs),
            "reference_basin_count": len(refs),
            "derived_clusters": derived_clusters,
        },
        reason="an independent basin requires an optimized complex and a locally bound decomposition",
    )
    builder.direct(
        "valid_local_decomposition_coverage",
        min(
            1.0,
            sum(
                info["evidence_tier"]["tier"] == "full_local_decomposition"
                for info in pose_info.values()
            )
            / max(1, min(4, len(refs))),
        ),
        evidence={"full_local_reference_basins": sorted(full_refs)},
        reason="only optimized complexes with frozen fragments and optimized monomer references receive full decomposition coverage",
    )

    ranked_values: dict[str, float] = {}
    ranking_cap = 0.0
    full_binding = {
        pose_id: float(info["selected_values"]["binding"])
        for pose_id, info in pose_info.items()
        if info["evidence_tier"]["tier"] == "full_local_decomposition"
        and info["selected_values"]["binding"] is not None
    }
    screened_binding = {
        pose_id: float(info["selected_values"]["binding"])
        for pose_id, info in pose_info.items()
        if info["selected_values"]["binding"] is not None
        and float(info["evidence_tier"]["binding_cap"]) > 0.0
    }
    screened_interaction = {
        pose_id: float(info["selected_values"]["interaction"])
        for pose_id, info in pose_info.items()
        if info["selected_values"]["interaction"] is not None
        and float(info["evidence_tier"]["interaction_cap"]) > 0.0
    }
    if full_binding:
        ranked_values, ranking_cap = full_binding, 1.0
    elif screened_binding:
        ranked_values, ranking_cap = screened_binding, 0.40
    elif screened_interaction:
        ranked_values, ranking_cap = screened_interaction, 0.40
    raw_best = (
        min(ranked_values, key=lambda key: (ranked_values[key], key))
        if ranked_values
        else None
    )
    allowed_best = set(map(str, reference.get("best_pose_ids", [])))
    best_info = pose_info.get(str(final.get("best_pose_id") or ""))
    best_fraction = ranking_cap * float(
        bool(
            raw_best
            and final.get("best_pose_id") == raw_best
            and best_info
            and best_info["reference_id"] in allowed_best
        )
    )
    builder.direct(
        "best_pose",
        best_fraction,
        evidence={
            "recomputed_best": raw_best,
            "matched_reference": (best_info or {}).get("reference_id"),
            "allowed_reference": sorted(allowed_best),
            "evidence_tier_cap": ranking_cap,
        },
        reason="a fixed-input screen can nominate a lead with bounded credit; full credit requires a complete optimized local decomposition",
    )
    reference_pose_count = sum(bool(info["reference_id"]) for info in pose_info.values())
    for item_id, key in (("binding_energy", "binding"), ("interaction_energy", "interaction"), ("deformation_energy", "deformation")):
        spec = builder.specs[item_id]
        contributions = energy_contributions[key]
        fractions = [
            value["cap"]
            * _linear_fraction(
                value["error"],
                float(spec.get("full_credit_error", 0.0)),
                float(spec.get("zero_credit_error", 0.0)),
            )
            for value in contributions
        ]
        fraction = sum(fractions) / max(1, reference_pose_count)
        builder.direct(
            item_id,
            fraction,
            evidence={
                "reference_bound_pose_count": reference_pose_count,
                "contributions": contributions,
            },
            reason=f"{key} is recomputed from trusted jobs; fixed-input screening is capped at 40% while optimized local decomposition can receive full credit",
        )

    raw_ranking = sorted(ranked_values, key=lambda key: (ranked_values[key], key))
    aligned = [pose_info[key]["reference_id"] for key in raw_ranking if pose_info[key]["reference_id"]]
    ref_values = {str(ref["pose_id"]): float(ref["binding_kcal_mol"]) for ref in refs}
    tau = kendall_tau_b_with_reference_ties(ref_values, aligned, 0.10) if aligned else -1.0
    competitive = set(map(str, final.get("competitive_pose_ids", [])))
    expected_competitive = {
        key
        for key, value in ranked_values.items()
        if raw_best and value - ranked_values[raw_best] <= 0.25
    }
    ranking_fraction = ranking_cap * max(0.0, (tau + 1.0) / 2.0) * (
        len(competitive & expected_competitive)
        / max(1, len(competitive | expected_competitive))
    )
    builder.direct("competitive_pose_ranking", ranking_fraction, evidence={"tau_b": tau, "recomputed_ranking": raw_ranking, "expected_competitive": sorted(expected_competitive), "evidence_tier_cap": ranking_cap}, reason="fixed-input rankings receive bounded credit; optimized locally decomposed rankings can receive full credit")
    contact_scores = []
    for info in pose_info.values():
        description = str(info["pose"].get("contact_description") or "").lower().replace("-", "_").replace(" ", "_")
        contact_scores.append(0.5 * float(len(description) >= 8) + 0.5 * float(info["family"] in description))
    builder.direct("contact_energy_explanation", sum(contact_scores) / max(1, len(contact_scores)), evidence={key: {"family": value["family"], "metrics": value["contact_metrics"]} for key, value in pose_info.items()}, reason="contact descriptions are tied to optimized geometry and the reported energy source")

    adopted = set(map(str, final.get("evidence_job_ids", [])))
    allocation = score_observable_allocation(job_events_from_evidence(trusted, adopted_job_ids=adopted), useful_units=len(trusted), total_actions=len(jobs))
    complete_reference_screen = bool(
        complete_reference_route and reference_seed_ids <= evaluated
    )
    screening_diversity_fraction = (
        1.0
        if complete_reference_screen
        else min(1.0, 0.5 * len(evaluated) / 8.0 + 0.5 * len(families) / 4.0)
    )
    builder.direct(
        "screening_and_contact_diversity",
        screening_diversity_fraction,
        evidence={
            "evaluated_candidate_count": len(evaluated),
            "evaluated_candidate_target": evaluated_candidate_target,
            "contact_families": sorted(families),
            "contact_family_target": contact_family_target,
            "complete_reference_route": complete_reference_screen,
        },
        reason="full credit requires either broad screening or complete recovery of every reference-supported seed and contact family within the frozen job budget",
    )
    optimized_complexes = sum(info["complex_optimized"] for info in pose_info.values())
    full_decomposed = sum(info["evidence_tier"]["tier"] == "full_local_decomposition" for info in pose_info.values())
    allocation_fraction = 0.4 * min(1.0, optimized_complexes / 4.0) + 0.6 * min(1.0, full_decomposed / 2.0)
    builder.direct("optimized_fragment_allocation", allocation_fraction, evidence={"optimized_complex_count": optimized_complexes, "full_local_decomposition_count": full_decomposed}, reason="budget is rewarded for promoting selected complexes and completing discriminative fragment calculations")
    screening_consistent_count = sum(info["evidence_tier"]["tier"] in {"full_local_decomposition", "optimized_complex_interaction_only", "fixed_input_screening_decomposition", "fixed_input_interaction_only"} for info in pose_info.values())
    check_fraction = screening_consistent_count / max(1, len(pose_info))
    builder.direct("closure_and_geometry_checks", check_fraction, evidence={"evidence_consistent_pose_count": screening_consistent_count, "pose_count": len(pose_info), "full_closure_count": closure_matches}, reason="each adopted tier must have geometry binding and the numerical checks available at that tier")
    failures = [job for job in jobs.values() if not job_valid_termination(job)]
    if failures:
        recovered = sum(any(str(candidate.record.get("retry_of") or candidate.record.get("retry_of_job_id") or "") == job.job_id and job_valid_termination(candidate) for candidate in jobs.values()) for job in failures)
        response = recovered / len(failures)
    else:
        response = float(bool(trusted) and bool(adopted & set(trusted))) * check_fraction
    builder.direct("diagnostic_response", response, evidence={"failure_count": len(failures), "fragment_valid_count": fragment_matches}, reason="linked recovery is required for observed failures; all-pass routes must demonstrate fragment validation")
    inner_jobs = len(manifest.get("jobs", []))
    outer_totals = (((harness_manifest or {}).get("budget_ledger") or {}).get("totals") or {}) if isinstance(harness_manifest, dict) else {}
    authoritative_jobs = int(outer_totals.get("xtb_jobs", inner_jobs))
    max_jobs = int((config.get("budgets") or {}).get("max_xtb_jobs", 16))
    budget_valid = authoritative_jobs == inner_jobs and authoritative_jobs <= max_jobs
    turn_fraction, turn_evidence = _turn_efficiency(
        harness_manifest, int((config.get("budgets") or {}).get("max_agent_turns", 65))
    )
    scientific_utility = min(1.0, (independent_coverage + contact_family_fraction + ranking_cap) / 2.0)
    efficiency_fraction = allocation.fraction * float(budget_valid) * turn_fraction * scientific_utility
    builder.direct("information_efficiency", efficiency_fraction, evidence={**allocation.__dict__, **turn_evidence, "scientific_utility_fraction": scientific_utility, "authoritative_jobs": authoritative_jobs, "inner_jobs": inner_jobs, "max_jobs": max_jobs, "ledger_consistent": authoritative_jobs == inner_jobs}, reason="efficiency is conditional on useful scientific evidence and cumulative Agent-turn and xTB budgets")

    uncertainty = final.get("uncertainty") if isinstance(final.get("uncertainty"), dict) else {}
    unevaluated = set(map(str, uncertainty.get("unevaluated_candidate_ids", [])))
    expected_unevaluated = {f"POSE_{index:02d}" for index in range(1, 15)} - evaluated
    unexplored_union = unevaluated | expected_unevaluated
    unexplored_agreement = 1.0 if not unexplored_union else len(unevaluated & expected_unevaluated) / len(unexplored_union)
    builder.direct("unevaluated_pose_risk", unexplored_agreement, evidence={"reported": sorted(unevaluated), "evidence_derived": sorted(expected_unevaluated)}, reason="unexplored public candidates must be explicitly bounded")
    near_ids = set(map(str, uncertainty.get("near_degenerate_pose_ids", [])))
    near_ok = near_ids == expected_competitive and uncertainty.get("ranking_stable") is (len(expected_competitive) <= 1)
    builder.exact("ranking_stability_and_near_degeneracy", near_ok, evidence={"reported": sorted(near_ids), "recomputed": sorted(expected_competitive)}, reason="near-degeneracy is recomputed from trusted binding energies")
    limitations = [str(value).strip() for value in uncertainty.get("method_limitations", []) if str(value).strip()]
    builder.direct("fragment_and_method_limits", min(1.0, len(limitations) / 2.0), evidence={"declared_limitations": limitations}, reason="structured limitations must bound frozen-fragment and xTB conclusions")

    cited = set(map(str, final.get("evidence_job_ids", [])))
    builder.direct("method_and_hash_validity", len(cited & set(trusted)) / max(1, len(cited)), evidence={"cited": sorted(cited), "trusted": sorted(cited & set(trusted))}, reason="cited evidence is hash-verified and protocol-correct")
    builder.direct("frozen_fragment_geometry_binding", fragment_matches / max(1, len(pose_info)), evidence={"valid": fragment_matches, "pose_count": len(pose_info), "tolerance_angstrom": 1e-6}, reason="frozen fragment inputs must equal the optimized complex subsets coordinate by coordinate")
    builder.direct("energy_decomposition_closure", closure_matches / max(1, len(pose_info)), evidence={"closed": closure_matches, "pose_count": len(pose_info), "tolerance_kcal_mol": 0.01}, reason="reported terms must agree with raw recomputation and close within ordinary numeric reporting precision")

    return {
        "schema_errors": schema_errors,
        "contract_errors": list(integrity_errors),
        "integrity_errors": integrity_errors,
        "trusted_job_count": len(trusted),
        "evaluated_candidate_ids": sorted(evaluated),
        "derived_clusters": derived_clusters,
        "budget_valid": budget_valid,
        "recomputed_energies": {key: {name: info[name] for name in ("interaction", "deformation", "binding")} for key, info in pose_info.items()},
    }

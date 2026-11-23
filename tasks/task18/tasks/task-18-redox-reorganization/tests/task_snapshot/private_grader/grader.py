"""Private evidence-derived scorer for Task 18 v4."""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import jsonschema


LOCAL = Path(__file__).resolve().parent
if str(LOCAL) not in sys.path:
    sys.path.insert(0, str(LOCAL))
from evidence import assign_candidates, enumerate_coherent_cycles, load_trusted_jobs  # noqa: E402
from geometry import kabsch_rmsd_text  # noqa: E402
from redox import derive_redox_cycle, pareto_front, rank_ascending  # noqa: E402


TASK_ID = "task_18_redox_reorganization"
GRADER_VERSION = "4.2.0"
CANDIDATES = ("donor_01", "donor_02", "donor_03")
SECTION_MAX = {
    "charge_state_basins_and_validation": 30.0,
    "numerical_redox_properties": 34.0,
    "pareto_and_scientific_decision": 16.0,
    "autonomous_search_and_resources": 12.0,
    "evidence_reliability_and_scope": 8.0,
}
LOCAL_CYCLE_NUMERIC_FRACTION = 0.40
BASIN_NUMERIC_FRACTION = 0.30
SEARCH_NUMERIC_FRACTION = 0.30


def _zero(reason: str) -> dict[str, object]:
    sections = {name: 0.0 for name in SECTION_MAX}
    return {
        "task_id": TASK_ID,
        "grader_version": GRADER_VERSION,
        "hard_zero": True,
        "hard_zero_reason": reason,
        "submission_contract_valid": False,
        "contract_errors": [reason],
        "sections": sections,
        "derived": {},
        "score": 0.0,
        "task_local_score": 0.0,
        "scientific_basis_score": 0.0,
        "comparison_score": 0.0,
    }


def _partial_result(
    *,
    sections: Mapping[str, float],
    derived: Mapping[str, object],
    contract_errors: list[str],
) -> dict[str, object]:
    normalized = {
        name: round(min(SECTION_MAX[name], max(0.0, float(sections.get(name, 0.0)))), 6)
        for name in SECTION_MAX
    }
    total = round(sum(normalized.values()), 6)
    return {
        "task_id": TASK_ID,
        "grader_version": GRADER_VERSION,
        "hard_zero": False,
        "hard_zero_reason": None,
        "submission_contract_valid": not contract_errors,
        "contract_errors": list(contract_errors),
        "sections": normalized,
        "derived": dict(derived),
        "score": total,
        "task_local_score": total,
        "scientific_basis_score": total,
        "comparison_score": total,
    }


def _linear_accuracy(error: float, full: float, zero: float) -> float:
    if not math.isfinite(error) or zero <= full:
        return 0.0
    if error <= full:
        return 1.0
    if error >= zero:
        return 0.0
    return (zero - error) / (zero - full)


def _load_final(submission: Path, public: Path) -> tuple[dict[str, object] | None, str | None]:
    path = Path(submission) / "final_answer.json"
    if not path.is_file():
        return None, "missing_final_answer"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(
            (Path(public) / "schema" / "final_answer.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator(schema).validate(value)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError):
        return None, "invalid_final_contract"
    return value, None


def _load_reference(reference_dir: Path) -> dict[str, object] | None:
    try:
        value = json.loads(
            (Path(reference_dir) / "reference.json").read_text(encoding="utf-8")
        )
        if (
            value.get("schema_version") != "task18-reference/4.1"
            or value.get("grader_version") != GRADER_VERSION
            or set(value["candidate_values"]) != set(CANDIDATES)
        ):
            return None
        for key in (
            "oxidation_ease_ranking",
            "low_reorganization_ranking",
            "pareto_front_candidate_ids",
            "recommendations",
            "tolerances",
        ):
            if key not in value:
                return None
        return value
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _reported_by_candidate(final: Mapping[str, object], field: str) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    values = final.get(field)
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id")
        if candidate_id in CANDIDATES and candidate_id not in result:
            result[str(candidate_id)] = item
    return result


ROLE_FIELDS = {
    "neutral_optimization_job_id": "neutral_minimum",
    "cation_optimization_job_id": "cation_minimum",
    "cation_at_neutral_job_id": "cation_at_neutral",
    "neutral_at_cation_job_id": "neutral_at_cation",
}


def _select_cycle(
    item: Mapping[str, object],
    cycles: list[dict[str, object]],
) -> dict[str, object] | None:
    matches = []
    for cycle in cycles:
        if all(
            item.get(field) == cycle[role].get("job_id")
            for field, role in ROLE_FIELDS.items()
        ):
            matches.append(cycle)
    return min(matches, key=lambda cycle: tuple(cycle["job_ids"])) if matches else None


def _basin_factor(
    job: Mapping[str, object],
    basins: object,
    reference_dir: Path,
    tolerances: Mapping[str, object],
) -> tuple[float, dict[str, object]]:
    best_factor = 0.0
    best_details: dict[str, object] = {"matched": False}
    if not isinstance(basins, list) or not isinstance(job.get("output_text"), str):
        return best_factor, best_details
    for basin in basins:
        if not isinstance(basin, dict):
            continue
        relative = basin.get("geometry_path")
        reference_energy = basin.get("energy_hartree")
        if not isinstance(relative, str) or not isinstance(reference_energy, (int, float)):
            continue
        geometry_path = Path(reference_dir) / relative
        try:
            rmsd = kabsch_rmsd_text(
                str(job["output_text"]), geometry_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            continue
        energy_error_ev = abs(
            float(job["energy_hartree"]) - float(reference_energy)
        ) * 27.211386245988
        rmsd_zero = float(tolerances["basin_rmsd_angstrom"])
        factor = min(
            _linear_accuracy(rmsd, min(0.05, rmsd_zero / 5.0), rmsd_zero),
            _linear_accuracy(
                energy_error_ev,
                float(tolerances["basin_energy_ev_full"]),
                float(tolerances["energy_ev_zero"]),
            ),
        )
        if factor > best_factor:
            best_factor = factor
            best_details = {
                "matched": factor > 0,
                "geometry_path": relative,
                "rmsd_angstrom": rmsd,
                "energy_error_ev": energy_error_ev,
                "factor": factor,
            }
    return best_factor, best_details


def _numeric_factor(
    reported: object,
    reconstructed: float,
    reference: float,
    tolerances: Mapping[str, object],
) -> tuple[float, dict[str, float]]:
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        return 0.0, {}
    reconstructed_error = abs(float(reported) - reconstructed)
    reference_error = abs(reconstructed - reference)
    factor = min(
        _linear_accuracy(reconstructed_error, 1e-6, float(tolerances["energy_ev_zero"])),
        _linear_accuracy(
            reference_error,
            float(tolerances["energy_ev_full"]),
            float(tolerances["energy_ev_zero"]),
        ),
    )
    return factor, {
        "reported_reconstruction_error_ev": reconstructed_error,
        "reconstruction_reference_error_ev": reference_error,
    }


def _hessian_adopted(
    item: Mapping[str, object],
    cycle: Mapping[str, object],
    *,
    state: str,
) -> bool:
    hessian = cycle.get(f"{state}_hessian")
    return bool(
        isinstance(hessian, Mapping)
        and item.get(f"{state}_hessian_job_id") == hessian.get("job_id")
        and hessian.get("hessian_complete") is True
        and hessian.get("hessian_minimum") is True
    )


def _state_jobs(
    jobs: Mapping[str, Mapping[str, object]],
    assignment: Mapping[str, str],
    candidate_id: str,
    *,
    charge: int,
    uhf: int,
) -> list[Mapping[str, object]]:
    return [
        record
        for job_id, record in jobs.items()
        if assignment.get(job_id) == candidate_id
        and record.get("operation") in {"opt", "opt_hess"}
        and record.get("charge") == charge
        and record.get("uhf") == uhf
        and record.get("optimization_converged") is True
        and isinstance(record.get("output_sha256"), str)
    ]


def _has_linked_minimum_hessian(
    minima: list[Mapping[str, object]],
    members: list[Mapping[str, object]],
    *,
    charge: int,
    uhf: int,
) -> bool:
    for minimum in minima:
        if (
            minimum.get("operation") == "opt_hess"
            and minimum.get("hessian_complete") is True
            and minimum.get("hessian_minimum") is True
        ):
            return True
        output_sha = minimum.get("output_sha256")
        if any(
            record.get("operation") == "hess"
            and record.get("charge") == charge
            and record.get("uhf") == uhf
            and record.get("input_sha256") == output_sha
            and record.get("hessian_complete") is True
            and record.get("hessian_minimum") is True
            for record in members
        ):
            return True
    return False


def _evidence_foundation(
    jobs: Mapping[str, Mapping[str, object]],
    assignment: Mapping[str, str],
    cycles: Mapping[str, list[dict[str, object]]],
) -> dict[str, object]:
    optimization_coverage = 0.0
    hessian_coverage = 0.0
    distinct_search_coverage = 0.0
    candidates: dict[str, object] = {}
    for candidate_id in CANDIDATES:
        members = [
            record
            for job_id, record in jobs.items()
            if assignment.get(job_id) == candidate_id
        ]
        state_details: dict[str, object] = {}
        for state, charge, uhf in (("neutral", 0, 0), ("cation", 1, 1)):
            minima = _state_jobs(
                jobs, assignment, candidate_id, charge=charge, uhf=uhf
            )
            optimized = bool(minima)
            hessian_minimum = _has_linked_minimum_hessian(
                minima, members, charge=charge, uhf=uhf
            )
            distinct_inputs = len(
                {str(record["input_sha256"]) for record in minima}
            )
            optimization_coverage += float(optimized)
            hessian_coverage += float(hessian_minimum)
            distinct_search_coverage += float(distinct_inputs >= 2)
            state_details[state] = {
                "converged_optimization": optimized,
                "linked_minimum_hessian": hessian_minimum,
                "distinct_optimization_inputs": distinct_inputs,
            }
        state_details["coherent_cycle_observed"] = bool(cycles.get(candidate_id))
        candidates[candidate_id] = state_details
    coherent_cycle_coverage = float(
        sum(bool(cycles.get(candidate_id)) for candidate_id in CANDIDATES)
    )
    return {
        "charge_state_score": optimization_coverage + hessian_coverage,
        "autonomous_score_before_efficiency": (
            distinct_search_coverage + coherent_cycle_coverage
        ),
        "optimization_coverage": optimization_coverage,
        "hessian_coverage": hessian_coverage,
        "distinct_search_coverage": distinct_search_coverage,
        "coherent_cycle_coverage": coherent_cycle_coverage,
        "candidates": candidates,
    }


def _search_factor(
    assessment: Mapping[str, object] | None,
    candidate_id: str,
    jobs: Mapping[str, Mapping[str, object]],
    assignment: Mapping[str, str],
    cycle: Mapping[str, object] | None,
) -> tuple[float, dict[str, object]]:
    if assessment is None or cycle is None:
        return 0.0, {"valid": False}
    state_results: dict[str, bool] = {}
    adopted_lowest = True
    for state, charge, uhf, field, role in (
        ("neutral", 0, 0, "neutral_search_job_ids", "neutral_minimum"),
        ("cation", 1, 1, "cation_search_job_ids", "cation_minimum"),
    ):
        ids = assessment.get(field)
        if not isinstance(ids, list):
            state_results[state] = False
            continue
        records = [jobs.get(str(job_id)) for job_id in ids]
        valid_records = [
            record
            for record in records
            if isinstance(record, Mapping)
            and assignment.get(str(record["job_id"])) == candidate_id
            and record.get("operation") in {"opt", "opt_hess"}
            and record.get("charge") == charge
            and record.get("uhf") == uhf
            and record.get("optimization_converged") is True
        ]
        distinct_inputs = {str(record["input_sha256"]) for record in valid_records}
        adopted_id = str(cycle[role]["job_id"])
        state_results[state] = (
            len(valid_records) == len(ids)
            and len(distinct_inputs) >= 2
            and adopted_id in ids
        )
        if valid_records:
            adopted_energy = float(cycle[role]["energy_hartree"])
            adopted_lowest = adopted_lowest and adopted_energy <= min(
                float(record["energy_hartree"]) for record in valid_records
            ) + 1e-8
    valid = all(state_results.values())
    return (1.0 if valid else 0.0), {
        "valid": valid,
        "state_results": state_results,
        "adopted_lowest_among_reported_search": adopted_lowest,
    }


def grade_task18(
    submission: Path,
    run_root: Path,
    reference_dir: Path,
    public_dir: Path,
    agent_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    final, error = _load_final(Path(submission), Path(public_dir))
    trusted = load_trusted_jobs(Path(run_root))
    if not trusted:
        return _zero("no_trusted_task18_job")
    reference = _load_reference(Path(reference_dir))
    if reference is None:
        return _zero("organizer_evidence_binding_failure")
    tolerances = reference["tolerances"]
    reference_values = reference["candidate_values"]
    try:
        assignment = assign_candidates(trusted, Path(public_dir) / "input")
        all_cycles = enumerate_coherent_cycles(trusted, assignment)
    except (OSError, TypeError, ValueError):
        return _zero("organizer_evidence_binding_failure")

    foundation = _evidence_foundation(trusted, assignment, all_cycles)
    contract_errors: list[str] = []
    if error is not None:
        contract_errors.append(str(error))
    report_present = (Path(submission) / "report.md").is_file()
    if not report_present:
        contract_errors.append("missing_report")
    if final is None:
        charge_score = float(foundation["charge_state_score"])
        autonomous_score = float(foundation["autonomous_score_before_efficiency"])
        autonomous_score = min(autonomous_score, max(0.0, 20.0 - charge_score))
        return _partial_result(
            sections={
                "charge_state_basins_and_validation": charge_score,
                "autonomous_search_and_resources": autonomous_score,
            },
            derived={
                "candidate_values": {},
                "per_candidate_checks": {},
                "oxidation_ease_ranking": [],
                "low_reorganization_ranking": [],
                "pareto_front_candidate_ids": [],
                "trusted_job_count": len(trusted),
                "assigned_job_count": len(assignment),
                "evidence_foundation": foundation,
                "agent_metrics": agent_metrics or {},
            },
            contract_errors=contract_errors,
        )

    reported = _reported_by_candidate(final, "candidate_results")
    assessments = _reported_by_candidate(final, "search_assessment")
    selected: dict[str, dict[str, object]] = {}
    derived: dict[str, dict[str, object]] = {}
    per_candidate: dict[str, dict[str, object]] = {}

    basin_score = float(foundation["charge_state_score"])
    numerical_score = 0.0
    binding_sum = 0.0
    scope_validity: dict[str, bool] = {}
    for candidate_id in CANDIDATES:
        item = reported.get(candidate_id)
        cycles = all_cycles.get(candidate_id, [])
        cycle = _select_cycle(item, cycles) if item is not None else None
        details: dict[str, object] = {"coherent_cycle_selected": cycle is not None}
        if item is None or cycle is None:
            per_candidate[candidate_id] = details
            continue
        selected[candidate_id] = cycle
        values = derive_redox_cycle(cycle)
        values.update(
            {
                "neutral_cation_rmsd_angstrom": float(
                    cycle["neutral_cation_rmsd_angstrom"]
                ),
                "job_ids": list(cycle["job_ids"]),
            }
        )
        derived[candidate_id] = values
        ref = reference_values[candidate_id]
        neutral_factor, neutral_details = _basin_factor(
            cycle["neutral_minimum"], ref.get("neutral_basins"), Path(reference_dir), tolerances
        )
        cation_factor, cation_details = _basin_factor(
            cycle["cation_minimum"], ref.get("cation_basins"), Path(reference_dir), tolerances
        )
        basin_score += 3.0 * neutral_factor + 3.0 * cation_factor
        neutral_hessian = _hessian_adopted(item, cycle, state="neutral")
        cation_hessian = _hessian_adopted(item, cycle, state="cation")
        search_factor, search_details = _search_factor(
            assessments.get(candidate_id), candidate_id, trusted, assignment, cycle
        )
        basin_scope_factor = (neutral_factor + cation_factor) / 2.0
        scientific_scope_factor = min(
            1.0,
            LOCAL_CYCLE_NUMERIC_FRACTION
            + BASIN_NUMERIC_FRACTION * basin_scope_factor
            + SEARCH_NUMERIC_FRACTION * search_factor,
        )
        scientific_scope_valid = bool(
            search_factor == 1.0
            and neutral_factor == 1.0
            and cation_factor == 1.0
            and neutral_hessian
            and cation_hessian
        )
        scope_validity[candidate_id] = scientific_scope_valid

        numeric_details: dict[str, object] = {}
        for field, weight in (
            ("adiabatic_oxidation_proxy_ev", 4.0),
            ("lambda_h_ev", 4.0),
            ("vertical_oxidation_proxy_ev", 1.0),
            ("lambda_c_ev", 0.5),
            ("lambda_n_ev", 0.5),
        ):
            factor, errors = _numeric_factor(
                item.get(field), float(values[field]), float(ref[field]), tolerances
            )
            scoped_factor = factor * scientific_scope_factor
            numerical_score += weight * scoped_factor
            numeric_details[field] = {
                "factor": scoped_factor,
                "property_accuracy_factor": factor,
                "scientific_scope_factor": scientific_scope_factor,
                **errors,
            }

        core_ids = {
            str(cycle[role]["job_id"])
            for role in ROLE_FIELDS.values()
        }
        for state in ("neutral", "cation"):
            hessian = cycle.get(f"{state}_hessian")
            if isinstance(hessian, Mapping):
                core_ids.add(str(hessian["job_id"]))
        claimed = item.get("evidence_job_ids")
        binding_ok = isinstance(claimed, list) and set(claimed) == core_ids
        binding_sum += float(binding_ok)
        details.update(
            {
                "neutral_basin": neutral_details,
                "cation_basin": cation_details,
                "neutral_hessian_minimum": neutral_hessian,
                "cation_hessian_minimum": cation_hessian,
                "search": search_details,
                "scientific_scope_factor": scientific_scope_factor,
                "scientific_scope_valid": scientific_scope_valid,
                "numeric": numeric_details,
                "evidence_binding": binding_ok,
            }
        )
        per_candidate[candidate_id] = details

    oxidation_ranking = (
        rank_ascending(derived, "adiabatic_oxidation_proxy_ev") if derived else []
    )
    reorganization_ranking = rank_ascending(derived, "lambda_h_ev") if derived else []
    reconstructed_pareto = (
        pareto_front(
            derived,
            tie_tolerance=float(tolerances["pareto_tie_ev"]),
        )
        if derived
        else []
    )
    complete = set(derived) == set(CANDIDATES)
    global_scope_valid = complete and all(
        scope_validity.get(candidate_id, False) for candidate_id in CANDIDATES
    )
    if global_scope_valid and oxidation_ranking == reference["oxidation_ease_ranking"]:
        if final.get("oxidation_ease_ranking") == oxidation_ranking:
            numerical_score += 2.0
    if global_scope_valid and reorganization_ranking == reference["low_reorganization_ranking"]:
        if final.get("low_reorganization_ranking") == reorganization_ranking:
            numerical_score += 2.0

    decision_score = 0.0
    if (
        complete
        and reconstructed_pareto == reference["pareto_front_candidate_ids"]
        and final.get("pareto_front_candidate_ids") == reconstructed_pareto
    ):
        decision_score += 2.0
        if global_scope_valid:
            decision_score += 4.0
    recommendations = final.get("recommendations")
    reference_recommendations = reference["recommendations"]
    if isinstance(recommendations, Mapping):
        if (
            complete
            and recommendations.get("oxidation_access_priority")
            == reference_recommendations["oxidation_access_priority"]
            == oxidation_ranking[0]
        ):
            decision_score += 1.0
            if global_scope_valid:
                decision_score += 2.0
        if (
            complete
            and recommendations.get("low_reorganization_priority")
            == reference_recommendations["low_reorganization_priority"]
            == reorganization_ranking[0]
        ):
            decision_score += 1.0
            if global_scope_valid:
                decision_score += 2.0
    decision_assessment = final.get("decision_assessment")
    geometry_summary = final.get("geometry_response_summary")
    if (
        global_scope_valid
        and decision_score == 12.0
        and isinstance(decision_assessment, Mapping)
        and decision_assessment.get("classification")
        == reference.get("decision_classification")
        and len(str(decision_assessment.get("basis", "")).strip()) >= 40
        and isinstance(geometry_summary, str)
        and len(geometry_summary.strip()) >= 40
    ):
        decision_score += 4.0

    autonomous_score = float(foundation["autonomous_score_before_efficiency"])
    if global_scope_valid and len(trusted) <= 24:
        autonomous_score += 3.0

    evidence_score = 4.0 * (binding_sum / 3.0)
    if report_present and (
        isinstance(decision_assessment, Mapping)
        and decision_assessment.get("classification")
        == reference.get("decision_classification")
    ):
        evidence_score += 2.0
    limitations = final.get("limitations")
    if report_present and isinstance(limitations, list):
        limitation_text = " ".join(str(value).lower() for value in limitations)
        method_specific = any(
            token in limitation_text for token in ("semiempirical", "gfn2", "tight-binding")
        )
        omitted_physics = sum(
            token in limitation_text
            for token in ("coupling", "outer-sphere", "solvent", "packing", "crystal")
        ) >= 2
        if method_specific and omitted_physics:
            evidence_score += 2.0

    sections = {
        "charge_state_basins_and_validation": round(min(30.0, basin_score), 6),
        "numerical_redox_properties": round(min(34.0, numerical_score), 6),
        "pareto_and_scientific_decision": round(min(16.0, decision_score), 6),
        "autonomous_search_and_resources": round(min(12.0, autonomous_score), 6),
        "evidence_reliability_and_scope": round(min(8.0, evidence_score), 6),
    }
    return _partial_result(
        sections=sections,
        derived={
            "candidate_values": derived,
            "per_candidate_checks": per_candidate,
            "oxidation_ease_ranking": oxidation_ranking,
            "low_reorganization_ranking": reorganization_ranking,
            "pareto_front_candidate_ids": reconstructed_pareto,
            "trusted_job_count": len(trusted),
            "assigned_job_count": len(assignment),
            "evidence_foundation": foundation,
            "global_scientific_scope_valid": global_scope_valid,
            "agent_metrics": agent_metrics or {},
        },
        contract_errors=contract_errors,
    )

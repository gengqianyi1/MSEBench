"""Private geometry-derived Task 04 grader for the minimal trusted Harness."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from tasks.task_04.private.evaluator.evidence import TrustedJob, load_trusted_jobs
from tasks.task_04.private.evaluator.geometry import (
    XYZGeometry,
    heavy_atom_rmsd,
    periodic_distance_deg,
    read_xyz,
)
from tasks.task_04.private.evaluator.identity import HistidineIdentity, classify_histidine_geometry


HARTREE_TO_KCAL_MOL = 627.5094740631
R_KCAL_PER_MOL_K = 0.00198720425864083
TEMPERATURE_K = 298.15


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root: Path, relative: str) -> Path:
    supplied = Path(relative)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("path escapes its declared root")
    resolved_root = root.resolve()
    resolved = (resolved_root / supplied).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("path escapes its declared root")
    return resolved


def _same_coordinates(first: XYZGeometry, second: XYZGeometry, tolerance: float = 1e-8) -> bool:
    return bool(
        tuple(item.upper() for item in first.elements)
        == tuple(item.upper() for item in second.elements)
        and np.max(np.abs(first.coordinates - second.coordinates), initial=0.0) <= tolerance
    )


def _histidine_heavy_atom_rmsd(reference: XYZGeometry, candidate: XYZGeometry) -> float:
    direct = heavy_atom_rmsd(reference, candidate)
    exchanged_coordinates = candidate.coordinates.copy()
    exchanged_coordinates[[0, 1]] = exchanged_coordinates[[1, 0]]
    exchanged = XYZGeometry(
        candidate.elements,
        exchanged_coordinates,
        candidate.comment,
    )
    return min(direct, heavy_atom_rmsd(reference, exchanged))


def _boltzmann(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    minimum = min(values.values())
    weights = {
        key: math.exp(
            -((value - minimum) * HARTREE_TO_KCAL_MOL)
            / (R_KCAL_PER_MOL_K * TEMPERATURE_K)
        )
        for key, value in values.items()
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _linear(error: float, full: float, zero: float) -> float:
    if not math.isfinite(error) or error >= zero:
        return 0.0
    if error <= full:
        return 1.0
    return (zero - error) / (zero - full)


def _total_variation(reference: dict[str, float], candidate: dict[str, float]) -> float:
    keys = set(reference) | set(candidate)
    return 0.5 * sum(abs(reference.get(key, 0.0) - candidate.get(key, 0.0)) for key in keys)


def _hard_zero(reason: str, details: list[str] | None = None) -> dict[str, Any]:
    return {
        "task_id": "task_04",
        "grader_version": "2.1.1",
        "hard_zero": True,
        "hard_zero_reason": reason,
        "errors": details or [reason],
        "sections": {},
        "derived": {},
        "score": 0.0,
        "task_local_score": 0.0,
        "scientific_basis_score": 0.0,
        "comparison_score": 0.0,
    }


def _reference_states(reference_root: Path) -> dict[str, list[dict[str, Any]]]:
    document = _load_json(reference_root / "reference.json")
    if document.get("task_id") != "task_04" or document.get("task_version") != "3.0.0":
        raise ValueError("private reference task/version mismatch")
    result: dict[str, list[dict[str, Any]]] = {}
    for condition in ("gas", "water"):
        values: list[dict[str, Any]] = []
        for raw in document["conditions"][condition]["states"]:
            geometry = read_xyz(_inside(reference_root, str(raw["geometry_path"])))
            identity = classify_histidine_geometry(geometry)
            if identity.family is None or identity.chirality != "L":
                raise ValueError("private reference geometry has invalid identity")
            values.append(
                {
                    "reference_id": str(raw["reference_id"]),
                    "geometry": geometry,
                    "identity": identity,
                    "gibbs_energy_hartree": float(raw["gibbs_energy_hartree"]),
                    "population": float(raw["population"]),
                }
            )
        population_total = sum(value["population"] for value in values)
        if not values or abs(population_total - 1.0) > 1e-6:
            raise ValueError("private reference populations are not normalized")
        result[condition] = values
    return result


def _resolve_job_id(value: Any, jobs: dict[str, TrustedJob]) -> tuple[str | None, bool]:
    """Resolve an exact ID or one unambiguous numeric shorthand.

    The public schema only promises a non-empty identifier.  Several providers
    therefore emitted ``011`` for the visible Harness ID ``job_011``.  A unique
    numeric equivalence is a formatting defect, not fabricated scientific
    evidence.  Missing or ambiguous values remain unresolved.
    """

    supplied = str(value)
    if supplied in jobs:
        return supplied, False
    match = re.fullmatch(r"(?:job_)?0*(\d+)", supplied, re.IGNORECASE)
    if match is None:
        return None, False
    number = int(match.group(1))
    matches = []
    for job_id in jobs:
        candidate = re.fullmatch(r"job_0*(\d+)", job_id, re.IGNORECASE)
        if candidate is not None and int(candidate.group(1)) == number:
            matches.append(job_id)
    return (matches[0], True) if len(matches) == 1 else (None, False)


def _normalize_answer_job_ids(
    answer: dict[str, Any], jobs: dict[str, TrustedJob]
) -> tuple[int, int, list[str]]:
    total = normalized = 0
    unresolved: list[str] = []
    for state in answer["states"]:
        for condition in ("gas", "water"):
            payload = state["conditions"][condition]
            for key in ("optimization_job_id", "thermochemistry_job_id"):
                if payload[key] is None:
                    continue
                total += 1
                supplied = str(payload[key])
                resolved, changed = _resolve_job_id(supplied, jobs)
                if resolved is None:
                    unresolved.append(f"{state['state_id']}:{condition}:{key}:{supplied}")
                else:
                    payload[key] = resolved
                    normalized += int(changed)
            resolved_evidence = []
            for value in payload["evidence_job_ids"]:
                total += 1
                supplied = str(value)
                resolved, changed = _resolve_job_id(supplied, jobs)
                if resolved is None:
                    unresolved.append(
                        f"{state['state_id']}:{condition}:evidence_job_ids:{supplied}"
                    )
                    resolved_evidence.append(supplied)
                else:
                    resolved_evidence.append(resolved)
                    normalized += int(changed)
            payload["evidence_job_ids"] = list(dict.fromkeys(resolved_evidence))
    return total, normalized, unresolved


def _manifest(
    submission_root: Path,
    public_root: Path,
    jobs: dict[str, TrustedJob],
) -> tuple[dict[str, dict[str, Any]], bool, list[str]]:
    path = submission_root / "candidate_manifest.json"
    if not path.is_file():
        return {}, False, ["candidate_manifest.json missing"]
    try:
        document = _load_json(path)
        schema = _load_json(public_root / "schema" / "candidate_manifest.schema.json")
        jsonschema.validate(document, schema)
    except (OSError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        return {}, False, [f"candidate manifest invalid: {exc}"]
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for raw in document["candidates"]:
        candidate_id = str(raw["candidate_id"])
        if candidate_id in candidates:
            errors.append(f"duplicate candidate ID: {candidate_id}")
            continue
        try:
            geometry = read_xyz(_inside(submission_root, str(raw["geometry_path"])))
            identity = classify_histidine_geometry(geometry)
        except (OSError, ValueError, IndexError) as exc:
            errors.append(f"candidate {candidate_id} geometry invalid: {exc}")
            continue
        calculation_job_ids = []
        for value in raw["calculation_job_ids"]:
            resolved, _changed = _resolve_job_id(value, jobs)
            if resolved is None:
                errors.append(
                    f"candidate {candidate_id} calculation job is unresolved: {value}"
                )
                calculation_job_ids.append(str(value))
            else:
                calculation_job_ids.append(resolved)
        candidates[candidate_id] = {
            **raw,
            "calculation_job_ids": list(dict.fromkeys(calculation_job_ids)),
            "geometry": geometry,
            "identity": identity,
        }
    valid = not errors and len(candidates) == len(document["candidates"])
    return candidates, valid, errors


def _assessment(
    state: dict[str, Any],
    condition: str,
    submission_root: Path,
    jobs: dict[str, TrustedJob],
) -> tuple[dict[str, Any] | None, str | None]:
    payload = state["conditions"][condition]
    if payload["status"] != "characterized_minimum":
        return None, None
    optimization = jobs.get(str(payload["optimization_job_id"]))
    hessian = jobs.get(str(payload["thermochemistry_job_id"]))
    if optimization is None or hessian is None:
        return None, "referenced job does not exist"
    try:
        submitted_geometry = read_xyz(_inside(submission_root, str(payload["geometry_path"])))
    except (OSError, ValueError):
        return None, "submitted representative geometry is missing or invalid"
    valid = bool(
        optimization.operation == "opt"
        and optimization.condition == condition
        and optimization.protocol_valid
        and optimization.termination_valid
        and optimization.optimization_converged
        and optimization.optimized_geometry is not None
        and hessian.operation == "hess"
        and hessian.condition == condition
        and hessian.protocol_valid
        and hessian.termination_valid
        and hessian.thermochemistry_valid
        and hessian.gibbs_energy_hartree is not None
        and optimization.optimized_geometry is not None
        # Submission XYZ may be serialized to six decimal places.  Keep the
        # trusted opt -> Hessian binding below at its original 1e-8 tolerance.
        and _same_coordinates(submitted_geometry, optimization.optimized_geometry, tolerance=1e-6)
        and _same_coordinates(hessian.input_geometry, optimization.optimized_geometry)
    )
    if not valid or optimization.optimized_geometry is None or hessian.gibbs_energy_hartree is None:
        return None, "optimization/Hessian evidence is not a bound characterized minimum"
    try:
        identity = classify_histidine_geometry(optimization.optimized_geometry)
    except (IndexError, ValueError):
        return None, "optimized identity cannot be reconstructed"
    if identity.family is None or identity.chirality != "L":
        return None, "optimized geometry is not a recognized L-histidine state"
    return (
        {
            "state_id": str(state["state_id"]),
            "candidate_id": str(state["candidate_id"]),
            "job_id": optimization.job_id,
            "hessian_job_id": hessian.job_id,
            "geometry": optimization.optimized_geometry,
            "identity": identity,
            "gibbs_energy_hartree": hessian.gibbs_energy_hartree,
            "reported_gibbs_energy_hartree": float(payload["gibbs_energy_hartree"]),
            "reported_population": float(payload["population"]),
            "evidence_job_ids": tuple(str(value) for value in payload["evidence_job_ids"]),
        },
        None,
    )


def _matches(
    submitted: list[dict[str, Any]], references: list[dict[str, Any]]
) -> dict[str, str]:
    eligible: list[tuple[float, float, str, str, str]] = []
    for state in submitted:
        identity: HistidineIdentity = state["identity"]
        for reference in references:
            target: HistidineIdentity = reference["identity"]
            if identity.family != target.family or identity.chirality != "L":
                continue
            rmsd = _histidine_heavy_atom_rmsd(reference["geometry"], state["geometry"])
            torsion = max(
                periodic_distance_deg(identity.chi1_deg, target.chi1_deg),
                periodic_distance_deg(identity.chi2_deg, target.chi2_deg),
            )
            if rmsd <= 0.35 and torsion <= 25.0:
                eligible.append(
                    (rmsd, torsion, state["state_id"], reference["reference_id"], state["job_id"])
                )
    result: dict[str, str] = {}
    used_references: set[str] = set()
    used_jobs: set[str] = set()
    for _rmsd, _torsion, state_id, reference_id, job_id in sorted(eligible):
        if state_id in result or reference_id in used_references or job_id in used_jobs:
            continue
        result[state_id] = reference_id
        used_references.add(reference_id)
        used_jobs.add(job_id)
    return result


def _condition_scores(
    condition: str,
    states: list[dict[str, Any]],
    references: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    matches = _matches(states, references)
    raw_gibbs = {state["state_id"]: state["gibbs_energy_hartree"] for state in states}
    populations = _boltzmann(raw_gibbs)
    reference_by_id = {state["reference_id"]: state for state in references}
    state_by_id = {state["state_id"]: state for state in states}
    candidate_by_reference = {
        reference_id: populations.get(state_id, 0.0)
        for state_id, reference_id in matches.items()
    }
    reference_population = {
        state["reference_id"]: state["population"] for state in references
    }
    coverage = sum(reference_by_id[value]["population"] for value in set(matches.values()))

    candidate_dominant = max(populations, key=populations.get) if populations else None
    reference_dominant = max(reference_population, key=reference_population.get)
    matched_dominant = matches.get(candidate_dominant) if candidate_dominant else None
    summary_consistent = summary.get("dominant_state_id") == candidate_dominant
    dominant_microstate = float(matched_dominant == reference_dominant and summary_consistent)
    candidate_family = (
        state_by_id[candidate_dominant]["identity"].family if candidate_dominant else None
    )
    reference_family_population: dict[str, float] = {}
    for reference in references:
        family = reference["identity"].family
        reference_family_population[family] = reference_family_population.get(family, 0.0) + reference["population"]
    dominant_family = float(
        candidate_family == max(reference_family_population, key=reference_family_population.get)
        and summary_consistent
    )

    candidate_family_population: dict[str, float] = {}
    for state_id, population in populations.items():
        family = state_by_id[state_id]["identity"].family
        candidate_family_population[family] = (
            candidate_family_population.get(family, 0.0) + population
        )
    reference_family_set = set(reference_family_population)
    candidate_family_set = set(candidate_family_population)
    family_coverage = len(reference_family_set & candidate_family_set) / max(
        1, len(reference_family_set)
    )

    relative_errors: list[float] = []
    if raw_gibbs:
        candidate_minimum = min(raw_gibbs.values())
        reference_minimum = min(value["gibbs_energy_hartree"] for value in references)
        for state_id, reference_id in matches.items():
            candidate_relative = (raw_gibbs[state_id] - candidate_minimum) * HARTREE_TO_KCAL_MOL
            reference_relative = (
                reference_by_id[reference_id]["gibbs_energy_hartree"] - reference_minimum
            ) * HARTREE_TO_KCAL_MOL
            relative_errors.append(abs(candidate_relative - reference_relative))
    gibbs_mae = sum(relative_errors) / len(relative_errors) if relative_errors else math.inf
    population_tv = _total_variation(reference_population, candidate_by_reference)

    family_tv = _total_variation(reference_family_population, candidate_family_population)
    omitted_mass = max(0.0, 1.0 - coverage)
    submitted_bound = float(summary["omitted_population_upper_bound"])
    conservative = submitted_bound + 0.01 >= omitted_mass
    bound_fraction = 0.0 if not conservative else max(0.5, 1.0 - max(0.0, submitted_bound - omitted_mass) / 0.5)
    bound_informativeness = 0.0 if not conservative else max(0.0, 1.0 - max(0.0, submitted_bound - omitted_mass) / 0.5)
    return {
        "matches": matches,
        "populations": populations,
        "coverage": coverage,
        "family_coverage": family_coverage,
        "candidate_dominant_family": candidate_family,
        "reference_dominant_family": max(
            reference_family_population, key=reference_family_population.get
        ),
        "dominant_family": dominant_family,
        "dominant_microstate": dominant_microstate,
        "gibbs_mae_kcal_mol": gibbs_mae if math.isfinite(gibbs_mae) else None,
        "gibbs_fraction": _linear(gibbs_mae, 0.05, 0.75),
        "population_tv": population_tv,
        "population_fraction": _linear(population_tv, 0.02, 0.20),
        "family_population_tv": family_tv,
        "family_population_fraction": _linear(family_tv, 0.02, 0.20),
        "omitted_reference_population": omitted_mass,
        "submitted_omitted_upper_bound": submitted_bound,
        "bound_fraction": bound_fraction,
        "bound_informativeness": bound_informativeness,
    }


def _environmental_response_fraction(
    gas: dict[str, Any], water: dict[str, Any]
) -> float:
    candidate_pair = (
        gas["candidate_dominant_family"],
        water["candidate_dominant_family"],
    )
    reference_pair = (
        gas["reference_dominant_family"],
        water["reference_dominant_family"],
    )
    exact = float(candidate_pair == reference_pair)

    def charge_class(value: str | None) -> str | None:
        if value is None:
            return None
        return "zwitterionic" if value.startswith("zwitterionic_") else "neutral"

    candidate_charge_pair = tuple(charge_class(value) for value in candidate_pair)
    reference_charge_pair = tuple(charge_class(value) for value in reference_pair)
    charge_response = float(candidate_charge_pair == reference_charge_pair)
    return 0.5 * exact + 0.5 * charge_response


def _observed_turn_fraction(run_root: Path) -> tuple[float, dict[str, Any]]:
    trajectory = run_root / "trajectory.jsonl"
    config_path = run_root / "run_config.json"
    if not trajectory.is_file() or not config_path.is_file():
        return 1.0, {"observed": False}
    try:
        maximum_turn = 0
        for raw in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
            event = json.loads(raw)
            if event.get("type") == "assistant" and isinstance(event.get("turn"), int):
                maximum_turn = max(maximum_turn, int(event["turn"]))
        maximum_allowed = int(_load_json(config_path).get("max_turns", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0, {"observed": True, "valid": False}
    if maximum_turn <= 0 or maximum_allowed <= 0:
        return 0.0, {
            "observed": True,
            "valid": False,
            "turns_used": maximum_turn,
            "max_turns": maximum_allowed,
        }
    ratio = min(1.0, maximum_turn / maximum_allowed)
    fraction = 1.0 if ratio <= 0.5 else max(0.4, 1.6 - 1.2 * ratio)
    return fraction, {
        "observed": True,
        "valid": True,
        "turns_used": maximum_turn,
        "max_turns": maximum_allowed,
        "usage_ratio": ratio,
    }


def _uncertainty_text_fractions(
    uncertainty: list[Any], condition_results: dict[str, dict[str, Any]]
) -> tuple[float, float]:
    text = " ".join(str(value).lower() for value in uncertainty if str(value).strip())
    multiple_states = any(len(result["populations"]) > 1 for result in condition_results.values())
    near_fraction = 1.0 if not multiple_states else float(
        any(token in text for token in ("near", "degener", "population", "thermal", "relative"))
    )
    categories = (
        ("gfn", "xtb", "semiempirical", "hamiltonian"),
        ("alpb", "implicit", "solvent"),
        ("mrrho", "harmonic", "frequency", "hessian", "rotor"),
        ("candidate", "search", "conformer", "sampling", "omitted"),
    )
    method_fraction = sum(any(token in text for token in group) for group in categories) / len(categories)
    return near_fraction, method_fraction


def grade_task04(
    submission_root: Path,
    run_root: Path,
    reference_root: Path,
    public_root: Path,
) -> dict[str, Any]:
    submission_root = Path(submission_root)
    run_root = Path(run_root)
    public_root = Path(public_root)
    try:
        answer = _load_json(submission_root / "final_answer.json")
        jsonschema.validate(
            answer, _load_json(public_root / "schema" / "final_answer.schema.json")
        )
    except (OSError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        return _hard_zero("final_answer schema invalid", [str(exc)])

    jobs, job_errors = load_trusted_jobs(run_root)
    if not jobs or not any(
        job.protocol_valid and job.termination_valid for job in jobs.values()
    ):
        return _hard_zero("no trusted xTB job", job_errors)

    reference_count, normalized_reference_count, unresolved_references = (
        _normalize_answer_job_ids(answer, jobs)
    )
    resolved_reference_count = reference_count - len(unresolved_references)
    if reference_count and resolved_reference_count == 0:
        return _hard_zero("fully fabricated job binding", unresolved_references)

    try:
        references = _reference_states(Path(reference_root))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"private Task 04 reference invalid: {exc}") from exc

    candidates, manifest_valid, manifest_errors = _manifest(
        submission_root, public_root, jobs
    )
    states_by_condition: dict[str, list[dict[str, Any]]] = {"gas": [], "water": []}
    assessment_errors: list[str] = []
    for state in answer["states"]:
        for condition in ("gas", "water"):
            derived, error = _assessment(state, condition, submission_root, jobs)
            if derived is not None:
                states_by_condition[condition].append(derived)
            elif error is not None:
                assessment_errors.append(f"{state['state_id']}:{condition}: {error}")

    condition_results = {
        condition: _condition_scores(
            condition,
            states_by_condition[condition],
            references[condition],
            answer["condition_summaries"][condition],
        )
        for condition in ("gas", "water")
    }

    mean = lambda key: sum(result[key] for result in condition_results.values()) / 2.0
    environmental_response = _environmental_response_fraction(
        condition_results["gas"], condition_results["water"]
    )
    science_score = (
        6.0 * mean("family_coverage")
        + 6.0 * mean("dominant_family")
        + 8.0 * mean("coverage")
        + 6.0 * mean("dominant_microstate")
        + 8.0 * mean("gibbs_fraction")
        + 8.0 * mean("family_population_fraction")
        + 7.0 * mean("population_fraction")
        + 6.0 * environmental_response
    )

    reference_families = {
        reference["identity"].family
        for values in references.values()
        for reference in values
    }
    candidate_families = {
        value["identity"].family
        for value in candidates.values()
        if value["identity"].family is not None and value["identity"].chirality == "L"
    }
    diversity_fraction = 0.0
    if manifest_valid:
        family_fraction = len(candidate_families & reference_families) / max(
            1, len(reference_families)
        )
        count_fraction = min(1.0, len(candidates) / max(1, len(reference_families)))
        diversity_fraction = 0.5 * family_fraction + 0.5 * count_fraction

    valid_states = sum(len(values) for values in states_by_condition.values())
    matched_states = sum(len(result["matches"]) for result in condition_results.values())
    unique_condition_families = {
        (condition, state["identity"].family)
        for condition, states in states_by_condition.items()
        for state in states
    }
    route_fraction = len(unique_condition_families) / max(1, valid_states)
    quality_fraction = valid_states / max(
        1,
        sum(
            state["conditions"][condition]["status"] == "characterized_minimum"
            for state in answer["states"]
            for condition in ("gas", "water")
        ),
    )
    registered_jobs = {
        str(job_id)
        for candidate in candidates.values()
        for job_id in candidate["calculation_job_ids"]
    }
    all_job_cost = sum({"sp": 1, "opt": 1, "hess": 4, "scan": 1}.get(job.operation, 0) for job in jobs.values())
    productive_cost = sum(
        {"sp": 1, "opt": 1, "hess": 4, "scan": 1}.get(job.operation, 0)
        for job_id, job in jobs.items()
        if manifest_valid and job_id in registered_jobs and job.protocol_valid and job.termination_valid
    )
    turn_fraction, turn_evidence = _observed_turn_fraction(run_root)
    useful_science_fraction = min(1.0, science_score / 27.5)
    efficiency_fraction = (
        productive_cost / max(1, all_job_cost)
    ) * turn_fraction * useful_science_fraction
    resource_score = (
        5.0 * diversity_fraction
        + 6.0 * route_fraction
        + 5.0 * quality_fraction
        + 4.0 * quality_fraction
        + 5.0 * efficiency_fraction
    )

    near_fraction, method_fraction = _uncertainty_text_fractions(
        answer["uncertainty"], condition_results
    )
    uncertainty_score = (
        4.0 * mean("bound_fraction")
        + 3.0 * near_fraction
        + 3.0 * method_fraction
    )

    protocol_fraction = sum(
        job.protocol_valid and job.termination_valid for job in jobs.values()
    ) / max(1, len(jobs))
    energy_consistent = []
    population_consistent = []
    for condition, states in states_by_condition.items():
        recomputed = condition_results[condition]["populations"]
        for state in states:
            energy_consistent.append(
                abs(state["reported_gibbs_energy_hartree"] - state["gibbs_energy_hartree"])
                <= 1e-6
            )
            population_consistent.append(
                abs(state["reported_population"] - recomputed.get(state["state_id"], 0.0))
                <= 1e-4
            )
    numeric_fraction = (
        (sum(energy_consistent) + sum(population_consistent))
        / max(1, len(energy_consistent) + len(population_consistent))
    )
    lineage_checks: list[bool] = []
    for condition, states in states_by_condition.items():
        del condition
        for state in states:
            candidate = candidates.get(state["candidate_id"])
            lineage_checks.append(
                bool(
                    manifest_valid
                    and candidate is not None
                    and {state["job_id"], state["hessian_job_id"]}
                    <= set(str(value) for value in candidate["calculation_job_ids"])
                )
            )
    lineage_fraction = sum(lineage_checks) / max(1, len(lineage_checks))
    id_contract_fraction = (
        (reference_count - normalized_reference_count - len(unresolved_references))
        / max(1, reference_count)
    )
    evidence_score = (
        2.0 * protocol_fraction
        + 3.0 * numeric_fraction
        + 3.0 * lineage_fraction
        + 2.0 * max(0.0, id_contract_fraction)
    )

    sections = {
        "final_scientific_result": {"score": science_score, "maximum": 55.0},
        "resource_allocation_and_feedback": {"score": resource_score, "maximum": 25.0},
        "uncertainty_and_bounds": {"score": uncertainty_score, "maximum": 10.0},
        "raw_evidence_consistency": {"score": evidence_score, "maximum": 10.0},
    }
    total = round(sum(section["score"] for section in sections.values()), 10)
    derived_conditions: dict[str, Any] = {}
    for condition, states in states_by_condition.items():
        derived_conditions[condition] = {
            "states": [
                {
                    key: value
                    for key, value in state.items()
                    if key
                    not in {
                        "geometry",
                        "identity",
                    }
                }
                | {
                    "family": state["identity"].family,
                    "chirality": state["identity"].chirality,
                }
                for state in states
            ],
            **condition_results[condition],
        }
    report = {
        "task_id": "task_04",
        "grader_version": "2.1.1",
        "hard_zero": False,
        "hard_zero_reason": None,
        "errors": job_errors + unresolved_references + manifest_errors + assessment_errors,
        "sections": sections,
        "derived": {
            "conditions": derived_conditions,
            "environmental_response_fraction": environmental_response,
            "job_id_normalization": {
                "reference_count": reference_count,
                "normalized_count": normalized_reference_count,
                "unresolved_count": len(unresolved_references),
            },
            "efficiency": {
                "turns": turn_evidence,
                "useful_science_fraction": useful_science_fraction,
                "productive_compute_fraction": productive_cost / max(1, all_job_cost),
            },
        },
        "score": total,
        "task_local_score": total,
        "scientific_basis_score": total,
        "comparison_score": total,
    }
    return report

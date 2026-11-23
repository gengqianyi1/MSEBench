from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .candidate_validator import read_jsonl
except ImportError:  # script-style execution
    from candidate_validator import read_jsonl

CANDIDATE_WEIGHTS: Dict[str, float] = {
    "combined_structural_quality": 8.0,
    "internal_uniqueness": 3.0,
    "composition_structural_novelty": 3.0,
    "anonymous_prototype_novelty": 8.0,
    "composition_structure_diversity": 3.0,
    "generation_reproducibility": 5.0,
    "selected_chgnet_stability": 10.0,
    "model_prediction_traceability": 5.0,
    "uncertainty_and_applicability": 5.0,
}

def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass
class CandidateState:
    index: int
    record: Dict[str, Any]
    candidate_id: str
    structure: Any | None = None
    normalized_structure: Any | None = None
    valid: bool = False
    valid_reason: str | None = None
    representative: bool = False
    novel: bool = False
    prototype_novel: bool = False
    geometry_score: float = 0.0
    chemistry_score: float = 0.0
    formula_key: str = ""
    chemical_system: str = ""
    prototype_key: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _ramp_higher(value: float, zero: float, full: float) -> float:
    if full <= zero:
        raise ValueError("full must exceed zero")
    return _clip01((float(value) - zero) / (full - zero))


def _ramp_lower(value: float, full: float, zero: float) -> float:
    if zero <= full:
        raise ValueError("zero must exceed full")
    return _clip01((zero - float(value)) / (zero - full))


def _largest_periodic_gap(values: np.ndarray) -> Tuple[float, float]:
    values = np.mod(np.asarray(values, dtype=float), 1.0)
    if values.size == 0:
        return 1.0, 0.0
    ordered = np.sort(values)
    next_values = np.r_[ordered[1:], ordered[0] + 1.0]
    gaps = next_values - ordered
    index = int(np.argmax(gaps))
    gap = float(gaps[index])
    occupied_start = float(next_values[index] % 1.0)
    return gap, occupied_start


def _vacuum_axis_info(structure: Any) -> Dict[str, float | int]:
    lengths = np.asarray(structure.lattice.abc, dtype=float)
    best: Dict[str, float | int] | None = None
    for axis in range(3):
        gap_fraction, occupied_start = _largest_periodic_gap(structure.frac_coords[:, axis])
        physical_gap = float(gap_fraction * lengths[axis])
        thickness = float(max(0.0, lengths[axis] - physical_gap))
        candidate = {
            "axis": axis,
            "gap_fraction": gap_fraction,
            "occupied_start": occupied_start,
            "vacuum_gap": physical_gap,
            "layer_thickness": thickness,
            "axis_length": float(lengths[axis]),
        }
        if best is None or physical_gap > float(best["vacuum_gap"]):
            best = candidate
    assert best is not None
    return best


def _normalize_2d_vacuum(structure: Any, target_vacuum: float = 20.0) -> Any:
    """Normalize the largest empty periodic direction while preserving layer thickness approximately."""
    from pymatgen.core import Lattice, Structure

    info = _vacuum_axis_info(structure)
    axis = int(info["axis"])
    old_length = float(info["axis_length"])
    gap_fraction = float(info["gap_fraction"])
    occupied_start = float(info["occupied_start"])
    thickness = float(info["layer_thickness"])
    if old_length <= 0.0 or gap_fraction <= 0.0:
        return structure.copy()

    frac = np.mod(np.asarray(structure.frac_coords, dtype=float), 1.0)
    unwrapped = np.mod(frac[:, axis] - occupied_start, 1.0)
    occupied_span = max(0.0, 1.0 - gap_fraction)
    # Numerical points close to one belong to the occupied interval boundary.
    unwrapped = np.where(unwrapped > occupied_span + 1e-7, unwrapped - 1.0, unwrapped)
    physical = unwrapped * old_length
    new_length = max(float(target_vacuum) + thickness, float(target_vacuum) + 1.0)
    centered = physical / new_length + (1.0 - thickness / new_length) / 2.0
    frac[:, axis] = np.mod(centered, 1.0)

    matrix = np.asarray(structure.lattice.matrix, dtype=float).copy()
    matrix[axis] *= new_length / old_length
    return Structure(
        Lattice(matrix),
        structure.species,
        frac,
        coords_are_cartesian=False,
        site_properties=dict(structure.site_properties),
    )


def _formula_key(structure: Any) -> str:
    return str(structure.composition.reduced_formula)


def _chemical_system(structure: Any) -> str:
    return "-".join(sorted(element.symbol for element in structure.composition.elements))


def _anonymous_formula(structure: Any) -> str:
    composition = structure.composition.reduced_composition
    value = getattr(composition, "anonymized_formula", None)
    if value:
        return str(value)
    counts = sorted(float(amount) for amount in composition.get_el_amt_dict().values())
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(f"{letters[i]}{count:g}" for i, count in enumerate(counts))


def _element_radius(symbol: str) -> float:
    try:
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

        value = CovalentRadius.radius.get(symbol)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    except Exception:
        pass

    from pymatgen.core import Element

    element = Element(symbol)
    for attr in ("atomic_radius", "atomic_radius_calculated", "van_der_waals_radius"):
        value = getattr(element, attr, None)
        if value is not None:
            try:
                number = float(value)
            except Exception:
                continue
            if np.isfinite(number) and number > 0:
                return number
    return 1.0


def _build_radius_graph(structure: Any, scale: float) -> Tuple[List[set[int]], List[Tuple[int, int, float, float]]]:
    count = len(structure)
    graph = [set() for _ in range(count)]
    bonds: List[Tuple[int, int, float, float]] = []
    distances = np.asarray(structure.distance_matrix, dtype=float)
    symbols = [site.specie.symbol for site in structure]
    radii = [_element_radius(symbol) for symbol in symbols]
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(distances[i, j])
            radius_sum = radii[i] + radii[j]
            if radius_sum <= 0:
                continue
            ratio = distance / radius_sum
            if distance <= scale * radius_sum:
                graph[i].add(j)
                graph[j].add(i)
                bonds.append((i, j, distance, ratio))
    return graph, bonds


def _largest_component_fraction(graph: Sequence[set[int]]) -> float:
    if not graph:
        return 0.0
    visited: set[int] = set()
    largest = 0
    for start in range(len(graph)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return float(largest / len(graph))


def _valid_structure(record: Mapping[str, Any], duplicate_ids: set[str], constraints: Mapping[str, Any]) -> Tuple[Any | None, str | None]:
    from pymatgen.core import Structure

    candidate_id = str(record.get("candidate_id", ""))
    if not candidate_id or candidate_id in duplicate_ids:
        return None, "missing_or_duplicated_candidate_id"
    try:
        structure = Structure.from_dict(record.get("structure"))
    except Exception as exc:
        return None, f"structure_parse_failure:{type(exc).__name__}"
    if not structure.is_ordered:
        return None, "disordered_structure"
    if not constraints["min_sites"] <= len(structure) <= constraints["max_sites"]:
        return None, "site_count_out_of_range"
    elements = {site.specie.symbol for site in structure}
    if len(elements) > int(constraints["max_distinct_elements"]):
        return None, "too_many_elements"
    if not elements.issubset(set(constraints["allowed_elements"])):
        return None, "element_not_allowed"
    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    coords = np.asarray(structure.frac_coords, dtype=float)
    if matrix.shape != (3, 3) or coords.shape != (len(structure), 3):
        return None, "invalid_array_shape"
    if not np.isfinite(matrix).all() or not np.isfinite(coords).all():
        return None, "non_finite_structure_value"
    if not np.isfinite(float(structure.volume)) or float(structure.volume) <= 1e-8:
        return None, "non_positive_volume"
    pathology = _pathology_reason(structure, constraints)
    if pathology:
        return None, f"pathological_structure:{pathology}"
    return structure, None


def _pathology_reason(structure: Any, constraints: Mapping[str, Any]) -> str | None:
    config = constraints["pathology"]
    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    if float(np.linalg.cond(matrix)) > float(config["maximum_lattice_condition_number"]):
        return "degenerate_lattice"

    close_i, close_j, _, close_distances = structure.get_neighbor_list(
        float(config["minimum_absolute_distance_angstrom"])
    )
    if any(int(i) != int(j) or float(distance) > 1e-8 for i, j, distance in zip(close_i, close_j, close_distances)):
        return "absolute_atomic_overlap"

    if len(structure) >= 2:
        distances = np.asarray(structure.distance_matrix, dtype=float)
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                radius_sum = _element_radius(structure[i].specie.symbol) + _element_radius(structure[j].specie.symbol)
                if radius_sum > 0 and float(distances[i, j] / radius_sum) < float(config["minimum_covalent_radius_ratio"]):
                    return "covalent_radius_overlap"

    neighbor_i = structure.get_neighbor_list(float(config["coordination_cutoff_angstrom"]))[0]
    if len(neighbor_i):
        counts = np.bincount(np.asarray(neighbor_i, dtype=int), minlength=len(structure))
        if int(np.max(counts)) > int(config["maximum_neighbors_per_site"]):
            return "implausible_periodic_coordination"

    vacuum = _vacuum_axis_info(structure)
    if float(vacuum["vacuum_gap"]) < float(config["minimum_vacuum_gap_angstrom"]):
        return "insufficient_2d_vacuum"

    axis = int(vacuum["axis"])
    bond_scale = float(constraints["geometry"]["bond_radius_scale"])
    maximum_bond_cutoff = 2.0 * max(_element_radius(site.specie.symbol) for site in structure) * bond_scale
    neighbor_i, neighbor_j, neighbor_offsets, neighbor_distances = structure.get_neighbor_list(maximum_bond_cutoff)
    for i, j, offset, distance in zip(neighbor_i, neighbor_j, neighbor_offsets, neighbor_distances):
        radius_sum = _element_radius(structure[int(i)].specie.symbol) + _element_radius(
            structure[int(j)].specie.symbol
        )
        if (
            abs(int(offset[axis])) > 0
            and float(distance) <= bond_scale * radius_sum
        ):
            return "bonded_across_vacuum"
    return None


def _make_matcher(constraints: Mapping[str, Any]) -> Any:
    from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher

    config = constraints["structure_matcher"]
    return StructureMatcher(
        ltol=float(config["ltol"]),
        stol=float(config["stol"]),
        angle_tol=float(config["angle_tol"]),
        primitive_cell=bool(config["primitive_cell"]),
        scale=bool(config["scale"]),
        attempt_supercell=bool(config["attempt_supercell"]),
        allow_subset=bool(config.get("allow_subset", False)),
        comparator=ElementComparator(),
    )


def _safe_match(matcher: Any, first: Any, second: Any) -> bool:
    try:
        return bool(matcher.fit(first, second))
    except Exception:
        return False


def _assign_internal_representatives(states: List[CandidateState], constraints: Mapping[str, Any]) -> None:
    matcher = _make_matcher(constraints)
    groups: Dict[str, List[CandidateState]] = defaultdict(list)
    for state in states:
        if state.valid:
            groups[state.formula_key].append(state)

    for group in groups.values():
        representatives: List[CandidateState] = []
        for state in group:
            duplicate = any(_safe_match(matcher, state.normalized_structure, rep.normalized_structure) for rep in representatives)
            if not duplicate:
                state.representative = True
                representatives.append(state)


def _load_training_structures(train_path: Path) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    from pymatgen.core import Structure

    frame = pd.read_json(train_path, orient="records", lines=True, compression="infer")
    formula_groups: Dict[str, List[Any]] = defaultdict(list)
    anonymous_groups: Dict[str, List[Any]] = defaultdict(list)
    for value in frame["structure"]:
        try:
            structure = Structure.from_dict(value)
            normalized = _normalize_2d_vacuum(structure)
            formula_groups[_formula_key(normalized)].append(normalized)
            anonymous_groups[_anonymous_formula(normalized)].append(normalized)
        except Exception:
            continue
    return formula_groups, anonymous_groups


def _safe_anonymous_match(matcher: Any, first: Any, second: Any) -> bool:
    try:
        return bool(matcher.fit_anonymous(first, second))
    except Exception:
        return False


def _assign_training_novelty(
    states: List[CandidateState],
    train_formula_groups: Mapping[str, Sequence[Any]],
    train_anonymous_groups: Mapping[str, Sequence[Any]],
    constraints: Mapping[str, Any],
) -> None:
    matcher = _make_matcher(constraints)
    for state in states:
        if not state.valid or not state.representative:
            continue
        known = train_formula_groups.get(state.formula_key, [])
        state.novel = not any(_safe_match(matcher, state.normalized_structure, train) for train in known)
        anonymous_known = train_anonymous_groups.get(_anonymous_formula(state.normalized_structure), [])
        state.prototype_novel = not any(
            _safe_anonymous_match(matcher, state.normalized_structure, train)
            for train in anonymous_known
        ) and state.novel


def _geometry_score(structure: Any, constraints: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    config = constraints["geometry"]
    graph, bonds = _build_radius_graph(structure, float(config["bond_radius_scale"]))

    if len(structure) >= 2:
        distances = np.asarray(structure.distance_matrix, dtype=float)
        ratios = []
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                radius_sum = _element_radius(structure[i].specie.symbol) + _element_radius(structure[j].specie.symbol)
                if radius_sum > 0:
                    ratios.append(float(distances[i, j] / radius_sum))
        min_ratio = min(ratios) if ratios else 0.0
    else:
        min_ratio = 0.0
    contact = _ramp_higher(
        min_ratio,
        float(config["covalent_contact_ratio_zero"]),
        float(config["covalent_contact_ratio_full"]),
    )

    vacuum = _vacuum_axis_info(structure)
    gap_score = _ramp_higher(
        float(vacuum["vacuum_gap"]),
        float(config["vacuum_gap_zero_angstrom"]),
        float(config["vacuum_gap_full_angstrom"]),
    )
    thickness_score = _ramp_lower(
        float(vacuum["layer_thickness"]),
        float(config["layer_thickness_full_max_angstrom"]),
        float(config["layer_thickness_zero_angstrom"]),
    )
    two_dimensional = 0.5 * (gap_score + thickness_score)

    lengths = np.asarray(structure.lattice.abc, dtype=float)
    angles = np.asarray(structure.lattice.angles, dtype=float)
    length_ok = float(np.all((lengths >= float(config["lattice_length_min_angstrom"])) & (lengths <= float(config["lattice_length_max_angstrom"]))))
    angle_ok = float(np.all((angles >= float(config["lattice_angle_min_degree"])) & (angles <= float(config["lattice_angle_max_degree"]))))
    condition = float(np.linalg.cond(np.asarray(structure.lattice.matrix, dtype=float)))
    condition_score = _ramp_lower(
        condition,
        float(config["condition_number_full_max"]),
        float(config["condition_number_zero"]),
    )
    lattice_score = (length_ok + angle_ok + condition_score) / 3.0

    connected_fraction = _largest_component_fraction(graph)
    connectivity = _ramp_higher(
        connected_fraction,
        float(config["connectivity_fraction_zero"]),
        float(config["connectivity_fraction_full"]),
    )

    total = (2.0 * contact + 2.0 * two_dimensional + lattice_score + 2.0 * connectivity) / 7.0
    return _clip01(total), {
        "contact": contact,
        "two_dimensional": two_dimensional,
        "lattice": lattice_score,
        "connectivity": connectivity,
        "min_radius_ratio": float(min_ratio),
        "vacuum_gap_angstrom": float(vacuum["vacuum_gap"]),
        "layer_thickness_angstrom": float(vacuum["layer_thickness"]),
        "largest_component_fraction": connected_fraction,
        "bond_count": float(len(bonds)),
    }


def _substitution_compatibility(record: Mapping[str, Any], constraints: Mapping[str, Any]) -> float:
    from pymatgen.core import Element

    operations = ((record.get("provenance") or {}).get("operations") or [])
    scores: List[float] = []
    for operation in operations:
        name = str((operation or {}).get("name", "")).lower()
        params = (operation or {}).get("parameters") or {}
        if "substitut" not in name and "replac" not in name:
            continue
        mapping = params.get("mapping") or params.get("substitutions") or {}
        if not isinstance(mapping, Mapping):
            continue
        for old_symbol, new_symbol in mapping.items():
            try:
                old = Element(str(old_symbol))
                new = Element(str(new_symbol))
            except Exception:
                scores.append(0.0)
                continue
            indicators = 0
            if old.group is not None and new.group is not None and int(old.group) == int(new.group):
                indicators += 1
            if set(old.common_oxidation_states) & set(new.common_oxidation_states):
                indicators += 1
            old_radius = _element_radius(old.symbol)
            new_radius = _element_radius(new.symbol)
            ratio = new_radius / old_radius if old_radius > 0 else math.inf
            if float(constraints["chemistry"]["substitution_radius_ratio_min"]) <= ratio <= float(constraints["chemistry"]["substitution_radius_ratio_max"]):
                indicators += 1
            scores.append({0: 0.25, 1: 0.65, 2: 0.9, 3: 1.0}[indicators])
    if not scores:
        return 1.0 if operations else 0.0
    return float(np.mean(scores))


def _oxidation_score(structure: Any, constraints: Mapping[str, Any]) -> float:
    from pymatgen.core import Element

    elements = [Element(element.symbol) for element in structure.composition.elements]
    if len(elements) == 1:
        return 1.0
    if all(element.is_metal for element in elements):
        return float(constraints["chemistry"]["all_metal_oxidation_fallback_score"])
    try:
        guesses = structure.composition.oxi_state_guesses(max_sites=-1)
    except Exception:
        guesses = []
    if guesses:
        return 1.0
    return float(constraints["chemistry"]["oxidation_state_failure_uncertain_score"])


def _chemistry_score(structure: Any, record: Mapping[str, Any], constraints: Mapping[str, Any]) -> Tuple[float, Dict[str, float]]:
    from pymatgen.core import Element

    chemistry = constraints["chemistry"]
    composition = structure.composition
    amounts = np.asarray(list(composition.get_el_amt_dict().values()), dtype=float)
    fractions = amounts / float(np.sum(amounts))
    if len(fractions) <= 1:
        stoichiometry = 1.0
    else:
        stoichiometry = _ramp_lower(
            float(np.max(fractions)),
            float(chemistry["multi_element_max_atomic_fraction_full"]),
            float(chemistry["multi_element_max_atomic_fraction_zero"]),
        )
    composition_score = stoichiometry

    oxidation = _oxidation_score(structure, constraints)
    substitution = _substitution_compatibility(record, constraints)

    graph, bonds = _build_radius_graph(structure, float(constraints["geometry"]["bond_radius_scale"]))
    if bonds:
        ratio_quality = float(np.mean([_clip01(1.0 - abs(ratio - 1.0) / 0.45) for _, _, _, ratio in bonds]))
        elements = [Element(site.specie.symbol) for site in structure]
        mixed_system = any(element.is_metal for element in elements) and any(not element.is_metal for element in elements)
        if mixed_system:
            cross = np.mean([float(elements[i].is_metal != elements[j].is_metal) for i, j, _, _ in bonds])
            neighbor_chemistry = 0.5 * ratio_quality + 0.5 * _ramp_higher(float(cross), 0.05, 0.35)
        else:
            neighbor_chemistry = ratio_quality
    else:
        neighbor_chemistry = 0.0

    total = (2.0 * composition_score + 2.0 * oxidation + 1.5 * substitution + 1.5 * neighbor_chemistry) / 7.0
    return _clip01(total), {
        "composition": composition_score,
        "oxidation": oxidation,
        "substitution": substitution,
        "local_neighbors": neighbor_chemistry,
    }


def _coordination_histogram(structure: Any, scale: float) -> str:
    graph, _ = _build_radius_graph(structure, scale)
    bins = [0] * 7
    for neighbors in graph:
        bins[min(len(neighbors), 6)] += 1
    total = max(1, len(graph))
    normalized = [round(value / total, 1) for value in bins]
    return ",".join(f"{value:.1f}" for value in normalized)


def _prototype_key(structure: Any, constraints: Mapping[str, Any]) -> str:
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        spg = int(SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0).get_space_group_number())
    except Exception:
        spg = 0
    histogram = _coordination_histogram(structure, float(constraints["geometry"]["bond_radius_scale"]))
    return f"{_anonymous_formula(structure)}|spg={spg}|n={len(structure)}|cn={histogram}"


def _normalized_entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / float(np.sum(counts))
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return _clip01(entropy / math.log(len(counts)))


def _diversity_score(states: Sequence[CandidateState], constraints: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    eligible = [state for state in states if state.valid and state.representative and state.novel]
    systems = [state.chemical_system for state in eligible]
    formulas = [state.formula_key for state in eligible]
    prototypes = [state.prototype_key for state in eligible]
    config = constraints["diversity"]

    system_fraction = min(1.0, len(set(systems)) / float(config["chemical_system_target"]))
    formula_fraction = min(1.0, len(set(formulas)) / float(config["reduced_formula_target"]))
    prototype_fraction = min(1.0, len(set(prototypes)) / float(config["prototype_target"]))
    entropy_fraction = 0.5 * _normalized_entropy(systems) + 0.5 * _normalized_entropy(prototypes)

    points = (
        float(config["chemical_system_points"]) * system_fraction
        + float(config["reduced_formula_points"]) * formula_fraction
        + float(config["prototype_points"]) * prototype_fraction
        + float(config["entropy_points"]) * entropy_fraction
    )
    return float(points), {
        "eligible_count": len(eligible),
        "unique_chemical_systems": len(set(systems)),
        "unique_reduced_formulas": len(set(formulas)),
        "unique_prototypes": len(set(prototypes)),
        "chemical_system_entropy": _normalized_entropy(systems),
        "prototype_entropy": _normalized_entropy(prototypes),
    }


def _normalize_json(value: Any, decimals: int) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(value[key], decimals) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, decimals) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return str(number)
        return round(number, decimals)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _record_digest(record: Mapping[str, Any], decimals: int) -> str:
    normalized = _normalize_json(record, decimals)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_score(manifest_path: Path, workspace: Path, constraints: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    details: Dict[str, Any] = {"manifest_valid": False, "hashes_verified": False}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        details["error"] = f"manifest unreadable: {exc}"
        return 0.0, details

    required = {"random_seed", "input_files", "library_versions", "generation_parameters", "submitted_count", "interface_version"}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        details["error"] = f"manifest missing keys: {sorted(required - set(manifest or {}))}"
        return 0.0, details
    if manifest.get("submitted_count") != constraints["required_candidate_count"]:
        details["error"] = "manifest submitted_count mismatch"
        return 0.0, details
    if manifest.get("interface_version") != constraints["reproducibility"]["interface_version"]:
        details["error"] = "manifest interface_version mismatch"
        return 0.0, details
    if not isinstance(manifest.get("library_versions"), dict) or not manifest["library_versions"]:
        details["error"] = "library_versions must be a non-empty object"
        return 0.0, details

    hashes_ok = True
    declared_paths = set()
    for item in manifest.get("input_files") or []:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            hashes_ok = False
            continue
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            hashes_ok = False
            continue
        path = workspace / rel
        declared_paths.add(rel.as_posix())
        if not path.is_file() or _sha256(path) != item["sha256"]:
            hashes_ok = False
    required_inputs = {"data/train.json.gz", "data/candidate_generation_constraints.json"}
    hashes_ok = hashes_ok and required_inputs.issubset(declared_paths)
    details.update({"manifest_valid": True, "hashes_verified": hashes_ok, "declared_inputs": sorted(declared_paths)})
    return (1.0 if hashes_ok else 0.0), details


def _provenance_fraction(records: Sequence[Mapping[str, Any]]) -> float:
    complete = 0
    for record in records:
        provenance = record.get("provenance") or {}
        operations = provenance.get("operations")
        parents = provenance.get("parent_ids")
        complete += int(
            isinstance(record.get("random_seed"), int)
            and isinstance(provenance.get("strategy"), str)
            and bool(provenance.get("strategy"))
            and isinstance(parents, list)
            and isinstance(operations, list)
            and len(operations) >= 1
            and all(isinstance(op, dict) and isinstance(op.get("name"), str) and isinstance(op.get("parameters"), dict) for op in operations)
        )
    return float(complete / max(1, len(records)))


def _rerun_generator(
    workspace: Path,
    generator_entrypoint: Path,
    generator_config: Path,
    submitted_records: Sequence[Mapping[str, Any]],
    constraints: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    details: Dict[str, Any] = {"executed": False, "returncode": None, "matched_fraction": 0.0}
    timeout = int(constraints["reproducibility"]["timeout_seconds"])
    decimals = int(constraints["reproducibility"]["float_round_decimals"])

    with tempfile.TemporaryDirectory(prefix="candidate-rerun-") as temp_name:
        temp = Path(temp_name)
        shutil.copytree(generator_entrypoint.parent, temp / "generator")
        (temp / "data").mkdir(parents=True)
        if generator_entrypoint.name != "generate.py":
            shutil.copy2(generator_entrypoint, temp / "generator" / "generate.py")
        if generator_config.name != "config.json":
            shutil.copy2(generator_config, temp / "generator" / "config.json")
        shutil.copy2(workspace / "data" / "train.json.gz", temp / "data" / "train.json.gz")
        shutil.copy2(
            workspace / "data" / "candidate_generation_constraints.json",
            temp / "data" / "candidate_generation_constraints.json",
        )
        output = temp / "regenerated.json.gz"
        command = [
            sys.executable,
            str(temp / "generator" / "generate.py"),
            "--input",
            str(temp / "data" / "train.json.gz"),
            "--constraints",
            str(temp / "data" / "candidate_generation_constraints.json"),
            "--config",
            str(temp / "generator" / "config.json"),
            "--output",
            str(output),
        ]
        env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT")
            if key in os.environ
        }
        env.update({"HOME": str(temp), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            process = subprocess.run(
                command,
                cwd=temp,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
            )
        except Exception as exc:
            details["error"] = str(exc)
            return 0.0, details

        details.update({
            "executed": True,
            "returncode": process.returncode,
            "stdout_tail": process.stdout[-2000:],
            "stderr_tail": process.stderr[-2000:],
        })
        if process.returncode != 0 or not output.is_file():
            return 0.0, details
        try:
            regenerated = read_jsonl(output)
        except Exception as exc:
            details["error"] = f"regenerated output unreadable: {exc}"
            return 0.0, details

        submitted_by_id = {str(record.get("candidate_id")): _record_digest(record, decimals) for record in submitted_records}
        regenerated_by_id = {str(record.get("candidate_id")): _record_digest(record, decimals) for record in regenerated}
        all_ids = set(submitted_by_id)
        matched = sum(1 for candidate_id in all_ids if regenerated_by_id.get(candidate_id) == submitted_by_id[candidate_id])
        fraction = float(matched / max(1, constraints["required_candidate_count"]))
        details.update({
            "regenerated_count": len(regenerated),
            "matched_count": matched,
            "matched_fraction": fraction,
        })
        return fraction, details


def _reproducibility_score(
    records: Sequence[Mapping[str, Any]],
    workspace: Path,
    generator_entrypoint: Path,
    generator_config: Path,
    manifest_path: Path,
    constraints: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    manifest_fraction, manifest_details = _manifest_score(manifest_path, workspace, constraints)
    provenance_fraction = _provenance_fraction(records)
    rerun_fraction, rerun_details = _rerun_generator(
        workspace,
        generator_entrypoint,
        generator_config,
        records,
        constraints,
    )
    points = 3.0 * rerun_fraction + provenance_fraction + manifest_fraction
    return float(points), {
        "rerun": rerun_details,
        "provenance_complete_fraction": provenance_fraction,
        "manifest": manifest_details,
    }


def score_candidate_library(
    candidate_path: Path,
    train_path: Path,
    constraints_path: Path,
    workspace: Path,
    generator_entrypoint: Path,
    generator_config: Path,
    generation_manifest: Path,
    selected_candidates: Sequence[Mapping[str, Any]] = (),
    chgnet_tool_attempts: int = 0,
    chgnet_score_output_dir: Path | None = None,
) -> Dict[str, Any]:
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    scoring_weights = {
        key: float(value)
        for key, value in constraints.get("scoring_weights", CANDIDATE_WEIGHTS).items()
    }
    if set(scoring_weights) != set(CANDIDATE_WEIGHTS):
        raise ValueError("candidate scoring_weights keys do not match the scoring contract")
    if not math.isclose(sum(scoring_weights.values()), 50.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("candidate scoring_weights must sum to 50")
    records = read_jsonl(candidate_path)
    required_count = int(constraints["required_candidate_count"])
    if len(records) > required_count:
        raise ValueError(f"candidate library must contain at most {required_count} records")

    ids = [str(record.get("candidate_id", "")) for record in records]
    duplicate_ids = {candidate_id for candidate_id, count in Counter(ids).items() if not candidate_id or count > 1}
    states = [CandidateState(index=i, record=record, candidate_id=str(record.get("candidate_id", ""))) for i, record in enumerate(records)]

    invalid_reasons: Counter[str] = Counter()
    for state in states:
        structure, reason = _valid_structure(state.record, duplicate_ids, constraints)
        state.structure = structure
        state.valid_reason = reason
        state.valid = structure is not None
        if not state.valid:
            invalid_reasons[str(reason)] += 1
            continue
        state.normalized_structure = _normalize_2d_vacuum(structure)
        state.formula_key = _formula_key(state.normalized_structure)
        state.chemical_system = _chemical_system(state.normalized_structure)

    _assign_internal_representatives(states, constraints)
    train_formula_groups, train_anonymous_groups = _load_training_structures(train_path)
    _assign_training_novelty(states, train_formula_groups, train_anonymous_groups, constraints)

    geometry_diagnostics: List[Dict[str, float]] = []
    chemistry_diagnostics: List[Dict[str, float]] = []
    for state in states:
        if not (state.valid and state.representative and state.novel):
            continue
        state.geometry_score, geometry = _geometry_score(state.normalized_structure, constraints)
        state.chemistry_score, chemistry = _chemistry_score(state.normalized_structure, state.record, constraints)
        state.prototype_key = _prototype_key(state.normalized_structure, constraints)
        geometry_diagnostics.append(geometry)
        chemistry_diagnostics.append(chemistry)

    valid_count = sum(state.valid for state in states)
    unique_count = sum(state.valid and state.representative for state in states)
    novel_count = sum(state.valid and state.representative and state.novel for state in states)
    prototype_novel_count = sum(state.valid and state.representative and state.prototype_novel for state in states)
    if prototype_novel_count > novel_count:
        raise ValueError("anonymous prototype novelty cannot exceed composition-specific novelty")

    structural_quality_fraction = (
        0.5 * valid_count / required_count
        + 0.25 * sum(state.geometry_score for state in states) / required_count
        + 0.25 * sum(state.chemistry_score for state in states) / required_count
    )
    breakdown = {
        "combined_structural_quality": scoring_weights["combined_structural_quality"] * structural_quality_fraction,
        "internal_uniqueness": scoring_weights["internal_uniqueness"] * unique_count / required_count,
        "composition_structural_novelty": scoring_weights["composition_structural_novelty"] * novel_count / required_count,
        "anonymous_prototype_novelty": scoring_weights["anonymous_prototype_novelty"] * prototype_novel_count / required_count,
    }
    diversity_points, diversity_details = _diversity_score(states, constraints)
    breakdown["composition_structure_diversity"] = diversity_points
    reproducibility_points, reproducibility_details = _reproducibility_score(
        records,
        workspace,
        generator_entrypoint,
        generator_config,
        generation_manifest,
        constraints,
    )
    reproducibility_points *= scoring_weights["generation_reproducibility"] / 5.0
    breakdown["generation_reproducibility"] = reproducibility_points
    model_path = workspace / "model.joblib"
    model_hash = _sha256(model_path) if model_path.is_file() else ""
    traced = sum(
        bool(model_hash and str(state.record.get("prediction_model_sha256", "")).lower() == model_hash.lower())
        for state in states
    )
    breakdown["model_prediction_traceability"] = scoring_weights["model_prediction_traceability"] * traced / required_count
    uncertainty_complete = sum(
        bool(
            _finite_float(state.record.get("prediction_uncertainty_mev_atom")) is not None
            and float(state.record.get("prediction_uncertainty_mev_atom", 0)) > 0
            and _finite_float(state.record.get("applicability_domain_score")) is not None
            and 0.0 <= float(state.record.get("applicability_domain_score")) <= 1.0
        )
        for state in states
    )
    breakdown["uncertainty_and_applicability"] = scoring_weights["uncertainty_and_applicability"] * uncertainty_complete / required_count
    selected_details: Dict[str, Any] = {
        "score": 0.0,
        "max_score": scoring_weights["selected_chgnet_stability"],
        "reason": "selected-candidate scoring was not requested",
    }
    if selected_candidates or chgnet_tool_attempts:
        from chgnet_stability import score_selected_candidates

        selected_details = score_selected_candidates(
            selected_candidates=selected_candidates,
            candidate_path=candidate_path,
            constraints_path=constraints_path,
            output_dir=chgnet_score_output_dir or (workspace / "private_chgnet_evaluation"),
            tool_attempts=chgnet_tool_attempts,
        )
    breakdown["selected_chgnet_stability"] = float(selected_details["score"])

    uncapped = float(sum(breakdown.values()))
    cap = float(sum(scoring_weights.values()))
    cap_reasons: List[str] = []
    caps = constraints["score_caps"]
    if valid_count < 700:
        cap = min(cap, float(caps["valid_below_700"]))
        cap_reasons.append("valid_count_below_700")
    if unique_count < 500:
        cap = min(cap, float(caps["unique_below_500"]))
        cap_reasons.append("unique_count_below_500")
    if novel_count < 300:
        cap = min(cap, float(caps["novel_below_300"]))
        cap_reasons.append("novel_count_below_300")
    score = min(uncapped, cap)

    return {
        "candidate_score": round(score, 6),
        "candidate_max_score": float(sum(scoring_weights.values())),
        "uncapped_candidate_score": round(uncapped, 6),
        "candidate_score_cap": cap,
        "cap_reasons": cap_reasons,
        "counts": {
            "submitted": len(records),
            "target": required_count,
            "completion_fraction": len(records) / required_count,
            "valid": valid_count,
            "unique_valid": unique_count,
            "novel_unique_valid": novel_count,
            "prototype_novel_unique_valid": prototype_novel_count,
            "duplicate_candidate_id_records": sum(1 for candidate_id in ids if candidate_id in duplicate_ids),
        },
        "breakdown": {key: round(float(value), 6) for key, value in breakdown.items()},
        "invalid_reason_counts": dict(invalid_reasons),
        "diversity": diversity_details,
        "reproducibility": reproducibility_details,
        "selected_chgnet_stability": selected_details,
        "diagnostic_means": {
            "geometry": {
                key: float(np.mean([item[key] for item in geometry_diagnostics]))
                for key in geometry_diagnostics[0]
            } if geometry_diagnostics else {},
            "chemistry": {
                key: float(np.mean([item[key] for item in chemistry_diagnostics]))
                for key in chemistry_diagnostics[0]
            } if chemistry_diagnostics else {},
        },
        "scoring_version": constraints.get("version"),
    }

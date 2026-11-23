"""Deterministic scoring metrics shared by the three benchmark tasks."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

HARTREE_TO_KCAL_MOL = 627.5094740631
R_KCAL_MOL_K = 1.98720425864083e-3


def boltzmann_populations(
    energies_hartree: Mapping[str, float], degeneracies: Mapping[str, float] | None = None, *, temperature_k: float = 298.15
) -> dict[str, float]:
    if temperature_k <= 0:
        raise ValueError("temperature must be positive")
    if not energies_hartree:
        return {}
    degeneracies = degeneracies or {}
    minimum = min(float(value) for value in energies_hartree.values())
    weights: dict[str, float] = {}
    for key, energy in energies_hartree.items():
        degeneracy = float(degeneracies.get(key, 1.0))
        if degeneracy < 0:
            raise ValueError("degeneracy cannot be negative")
        exponent = -(float(energy) - minimum) * HARTREE_TO_KCAL_MOL / (R_KCAL_MOL_K * temperature_k)
        weights[str(key)] = degeneracy * math.exp(max(-745.0, min(0.0, exponent)))
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else {key: 0.0 for key in weights}


def total_variation(reference: Mapping[str, float], candidate: Mapping[str, float]) -> float:
    keys = set(reference) | set(candidate)
    return 0.5 * sum(abs(float(reference.get(key, 0.0)) - float(candidate.get(key, 0.0))) for key in keys)


def macro_f1(reference: Mapping[str, str], candidate: Mapping[str, str]) -> float:
    labels = sorted(set(reference.values()) | set(candidate.values()))
    if not labels:
        return 1.0
    scores: list[float] = []
    keys = set(reference) | set(candidate)
    for label in labels:
        tp = sum(reference.get(key) == label and candidate.get(key) == label for key in keys)
        fp = sum(reference.get(key) != label and candidate.get(key) == label for key in keys)
        fn = sum(reference.get(key) == label and candidate.get(key) != label for key in keys)
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 1.0)
    return sum(scores) / len(scores)


def _pair_labels(clusters: Sequence[Sequence[str]]) -> dict[tuple[str, str], bool]:
    members = sorted({str(item) for cluster in clusters for item in cluster})
    cluster_of: dict[str, int] = {}
    for index, cluster in enumerate(clusters):
        for item in cluster:
            cluster_of[str(item)] = index
    return {(a, b): cluster_of.get(a) is not None and cluster_of.get(a) == cluster_of.get(b) for a, b in combinations(members, 2)}


def pairwise_cluster_f1(reference_clusters: Sequence[Sequence[str]], candidate_clusters: Sequence[Sequence[str]]) -> float:
    reference_members = {str(item) for cluster in reference_clusters for item in cluster}
    candidate_entries = [str(item) for cluster in candidate_clusters for item in cluster]
    candidate_members = set(candidate_entries)
    universe = sorted(reference_members | candidate_members)
    ref_of: dict[str, int] = {}
    cand_of: dict[str, int] = {}
    for i, cluster in enumerate(reference_clusters):
        for item in cluster:
            ref_of[str(item)] = i
    for i, cluster in enumerate(candidate_clusters):
        for item in cluster:
            cand_of[str(item)] = i
    tp = fp = fn = 0
    for a, b in combinations(universe, 2):
        ref_same = a in ref_of and b in ref_of and ref_of[a] == ref_of[b]
        cand_same = a in cand_of and b in cand_of and cand_of[a] == cand_of[b]
        tp += int(ref_same and cand_same)
        fp += int(not ref_same and cand_same)
        fn += int(ref_same and not cand_same)
    denominator = 2 * tp + fp + fn
    pair_score = 2 * tp / denominator if denominator else 1.0
    reference_coverage = len(reference_members & candidate_members) / len(reference_members) if reference_members else 1.0
    candidate_uniqueness = len(candidate_members) / len(candidate_entries) if candidate_entries else 0.0
    no_extras = len(reference_members & candidate_members) / len(candidate_members) if candidate_members else 0.0
    return pair_score * reference_coverage * candidate_uniqueness * no_extras


def kendall_tau_b_with_reference_ties(reference_values: Mapping[str, float], candidate_ranking: Sequence[str], tie_threshold: float) -> float:
    keys = list(reference_values)
    if len(keys) < 2:
        return 1.0
    rank = {str(key): index for index, key in enumerate(candidate_ranking)}
    missing_base = len(candidate_ranking)
    concordant = discordant = ties_ref = ties_cand = 0
    for a, b in combinations(keys, 2):
        delta = float(reference_values[a]) - float(reference_values[b])
        ref_sign = 0 if abs(delta) < tie_threshold else (1 if delta > 0 else -1)
        ra = rank.get(str(a), missing_base + keys.index(a))
        rb = rank.get(str(b), missing_base + keys.index(b))
        cand_sign = 0 if ra == rb else (1 if ra > rb else -1)
        if ref_sign == 0:
            # A list-valued submission cannot encode an equivalence class.
            # Reference-tied pairs are therefore neutral and are assessed by
            # the separate near-degeneracy robustness item.
            continue
        elif cand_sign == 0:
            ties_cand += 1
        elif ref_sign == cand_sign:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_ref) * (concordant + discordant + ties_cand))
    return (concordant - discordant) / denominator if denominator else 0.0


def match_periodic_angles(reference: Sequence[float], candidate: Sequence[float]) -> tuple[float, list[tuple[int, int, float]]]:
    if not reference and not candidate:
        return 0.0, []
    if not reference or not candidate:
        return math.inf, []
    cost = np.empty((len(reference), len(candidate)), dtype=float)
    for i, a in enumerate(reference):
        for j, b in enumerate(candidate):
            difference = abs((float(a) % 360.0) - (float(b) % 360.0))
            cost[i, j] = min(difference, 360.0 - difference)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(i), int(j), float(cost[i, j])) for i, j in zip(rows, cols, strict=True)]
    errors = [value for _, _, value in matches] + [180.0] * abs(len(reference) - len(candidate))
    return sum(errors) / max(len(reference), len(candidate)), matches

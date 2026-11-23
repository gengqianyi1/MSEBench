"""Pure redox-cycle and two-objective decision functions for Task 18."""
from __future__ import annotations

import math
from collections.abc import Mapping


HARTREE_TO_EV = 27.211386245988


def _energy(cycle: Mapping[str, object], role: str) -> float:
    record = cycle.get(role)
    if not isinstance(record, Mapping):
        raise ValueError(f"missing redox-cycle role: {role}")
    value = record.get("energy_hartree")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid energy for redox-cycle role: {role}")
    energy = float(value)
    if not math.isfinite(energy):
        raise ValueError("redox-cycle energies must be finite")
    return energy


def derive_redox_cycle(cycle: Mapping[str, object]) -> dict[str, float]:
    e_neutral = _energy(cycle, "neutral_minimum")
    e_cation = _energy(cycle, "cation_minimum")
    e_cation_at_neutral = _energy(cycle, "cation_at_neutral")
    e_neutral_at_cation = _energy(cycle, "neutral_at_cation")
    lambda_c = (e_cation_at_neutral - e_cation) * HARTREE_TO_EV
    lambda_n = (e_neutral_at_cation - e_neutral) * HARTREE_TO_EV
    lambda_h = lambda_c + lambda_n
    adiabatic = (e_cation - e_neutral) * HARTREE_TO_EV
    vertical = (e_cation_at_neutral - e_neutral) * HARTREE_TO_EV
    return {
        "e_n_hartree": e_neutral,
        "e_c_hartree": e_cation,
        "e_c_at_n_hartree": e_cation_at_neutral,
        "e_n_at_c_hartree": e_neutral_at_cation,
        "lambda_c_ev": lambda_c,
        "lambda_n_ev": lambda_n,
        "lambda_h_ev": lambda_h,
        "adiabatic_oxidation_proxy_ev": adiabatic,
        "vertical_oxidation_proxy_ev": vertical,
        "closure_error_ev": abs((vertical - adiabatic) - lambda_c),
    }


def rank_ascending(values: Mapping[str, Mapping[str, float]], field: str) -> list[str]:
    return [
        candidate_id
        for candidate_id, _ in sorted(
            values.items(),
            key=lambda item: (float(item[1][field]), item[0]),
        )
    ]


def pareto_front(
    values: Mapping[str, Mapping[str, float]],
    *,
    tie_tolerance: float,
) -> list[str]:
    if tie_tolerance < 0:
        raise ValueError("tie tolerance must be non-negative")
    axes = ("adiabatic_oxidation_proxy_ev", "lambda_h_ev")
    front: list[str] = []
    for candidate_id, candidate in sorted(values.items()):
        dominated = False
        for other_id, other in values.items():
            if other_id == candidate_id:
                continue
            no_worse = all(
                float(other[axis]) <= float(candidate[axis]) + tie_tolerance
                for axis in axes
            )
            strictly_better = any(
                float(other[axis]) < float(candidate[axis]) - tie_tolerance
                for axis in axes
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate_id)
    return front

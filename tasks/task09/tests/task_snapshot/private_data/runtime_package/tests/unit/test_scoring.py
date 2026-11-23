from __future__ import annotations

import math

import numpy as np
import pytest

from evaluator.scoring import (
    ScoringInputError,
    circular_distance_deg,
    compute_score,
    directional_barriers_from_curve,
    linear_error_points,
    one_to_one_circular_matches,
    periodic_local_extrema,
)


GRID = np.arange(0.0, 360.0, 5.0)


def reference_curve() -> np.ndarray:
    radians = np.deg2rad(GRID)
    # Several asymmetric wells make forward/reverse barrier checks meaningful.
    values = 2.2 - 1.5 * np.cos(4.0 * radians) + 0.35 * np.sin(radians)
    return values - values.min()


def complete_facts() -> dict[str, object]:
    return {
        "adaptive_sampling": {
            "coverage_fraction": 1.0,
            "information_gain_fraction": 1.0,
            "refinement_fraction": 1.0,
            "efficiency_fraction": 1.0,
        },
        "process_and_reproducibility": {
            "trajectory_and_tool_integrity_fraction": 1.0,
            "native_attempt_provenance_fraction": 1.0,
            "budget_and_protocol_fraction": 1.0,
            "independent_recomputation_fraction": 1.0,
            "replay_consistency_fraction": 1.0,
        },
    }


def passing_gates() -> dict[str, bool]:
    return {
        "budgets_respected": True,
        "minimum_valid_angles": True,
        "feedback_rounds": True,
        "profile_fit_rounds": True,
        "final_profile_bound": True,
        "native_attempt_backing": True,
        "security_clean": True,
        "exact_model_identity": True,
        "hash_binding": True,
        "scientific_invariants": True,
    }


def test_circular_distance_wraps_at_zero() -> None:
    assert circular_distance_deg(355, 0) == pytest.approx(5.0)
    assert circular_distance_deg(5, 355) == pytest.approx(10.0)
    assert circular_distance_deg(0, 180) == pytest.approx(180.0)
    assert circular_distance_deg(-5, 365) == pytest.approx(10.0)


def test_one_to_one_matcher_uses_maximum_cardinality_not_greedy_pairs() -> None:
    # 9 degrees can match either reference, while 1 degree can match only 0.
    matches = one_to_one_circular_matches([9, 1], [0, 20], tolerance_deg=11)
    assert len(matches) == 2
    assert {item["predicted_index"] for item in matches} == {0, 1}
    assert {item["reference_index"] for item in matches} == {0, 1}


def test_linear_thresholds_are_full_linear_and_zero_at_frozen_boundaries() -> None:
    assert linear_error_points(
        0.25, full_at_or_below=0.25, zero_at_or_above=1.5, max_points=35
    ) == pytest.approx(35.0)
    assert linear_error_points(
        1.5, full_at_or_below=0.25, zero_at_or_above=1.5, max_points=35
    ) == pytest.approx(0.0)
    assert linear_error_points(
        0.875, full_at_or_below=0.25, zero_at_or_above=1.5, max_points=35
    ) == pytest.approx(17.5)
    assert linear_error_points(
        1.875, full_at_or_below=0.75, zero_at_or_above=3.0, max_points=10
    ) == pytest.approx(5.0)


def test_identical_curves_and_complete_independent_facts_score_100() -> None:
    oracle = reference_curve()
    result = compute_score(
        facts=complete_facts(),
        profile={"grid_deg": GRID, "relative_energies_kcal_mol": oracle + 17.3},
        oracle={"grid_deg": GRID, "energies_kcal_mol": oracle},
        eligibility_gates=passing_gates(),
    )

    assert result["eligible"] is True
    assert result["total_score"] == pytest.approx(100.0)
    assert result["categories"]["curve_accuracy"]["score"] == pytest.approx(45.0)
    assert result["categories"]["extrema_and_barriers"]["score"] == pytest.approx(25.0)
    assert result["categories"]["adaptive_sampling"]["score"] == pytest.approx(15.0)
    assert result["categories"]["process_and_reproducibility"]["score"] == pytest.approx(15.0)
    assert result["metrics"]["rmse_kcal_mol"] == pytest.approx(0.0)


def test_profile_tool_artifact_field_names_are_accepted() -> None:
    oracle = reference_curve()
    result = compute_score(
        facts=complete_facts(),
        profile={
            "profile": {
                "grid_angles_deg": GRID.tolist(),
                "predicted_energy_kcal_mol": oracle.tolist(),
            }
        },
        oracle={
            "grid_angles_deg": GRID.tolist(),
            "predicted_energies": oracle.tolist(),
        },
        eligibility_gates=passing_gates(),
    )
    assert result["total_score"] == pytest.approx(100.0)


def test_hard_gate_retains_diagnostics_but_zeros_official_total() -> None:
    oracle = reference_curve()
    gates = passing_gates()
    gates["exact_model_identity"] = False
    result = compute_score(
        facts=complete_facts(),
        profile=oracle,
        oracle=oracle,
        eligibility_gates=gates,
    )

    assert result["ungated_score"] == pytest.approx(100.0)
    assert result["total_score"] == pytest.approx(0.0)
    assert result["official_score"] == pytest.approx(0.0)
    assert result["eligibility"]["failed_gates"] == ["exact_model_identity"]


def test_missing_gate_map_fails_closed() -> None:
    oracle = reference_curve()
    result = compute_score(facts=complete_facts(), profile=oracle, oracle=oracle)
    assert result["eligible"] is False
    assert result["total_score"] == 0.0
    assert result["eligibility"]["failed_gates"] == ["eligibility_gates_missing"]


def test_adaptive_points_are_gated_by_minimum_curve_accuracy() -> None:
    oracle = reference_curve() * 4.0
    predicted = np.zeros(72)
    result = compute_score(
        facts=complete_facts(),
        profile=predicted,
        oracle=oracle,
        eligibility_gates=passing_gates(),
    )
    adaptive = result["categories"]["adaptive_sampling"]
    assert adaptive["raw_score_before_curve_gate"] == pytest.approx(15.0)
    assert adaptive["curve_accuracy_gate_passed"] is False
    assert adaptive["score"] == pytest.approx(0.0)


def test_global_minimum_uses_circular_error() -> None:
    oracle = 1.0 - np.cos(np.deg2rad(GRID))
    predicted = np.roll(oracle, 1)  # minimum moves from 0 to 5 degrees
    result = compute_score(
        facts=complete_facts(),
        profile=predicted,
        oracle=oracle,
        eligibility_gates=passing_gates(),
    )
    minimum = result["categories"]["extrema_and_barriers"]["global_minimum"]
    assert minimum["circular_error_deg"] == pytest.approx(5.0)
    assert minimum["score"] == pytest.approx(7.0)


def test_periodic_extrema_and_directional_barriers_are_deterministic() -> None:
    energies = reference_curve()
    extrema = periodic_local_extrema(GRID, energies)
    barriers = directional_barriers_from_curve(GRID, energies)

    assert extrema["local_minima_deg"]
    assert extrema["local_maxima_deg"]
    assert barriers["forward_barrier_kcal_mol"] >= 0.0
    assert barriers["reverse_barrier_kcal_mol"] >= 0.0
    assert math.isfinite(barriers["forward_barrier_kcal_mol"])
    assert math.isfinite(barriers["reverse_barrier_kcal_mol"])

    frozen = 2.0 - 2.0 * np.cos(np.deg2rad(2.0 * GRID))
    frozen_barriers = directional_barriers_from_curve(GRID, frozen)
    assert frozen_barriers["reference_minimum_angle_deg"] == 0
    assert frozen_barriers["forward_endpoint_angle_deg"] == 180
    assert frozen_barriers["reverse_endpoint_angle_deg"] == 180
    assert frozen_barriers["forward_barrier_kcal_mol"] == pytest.approx(4.0)
    assert frozen_barriers["reverse_barrier_kcal_mol"] == pytest.approx(4.0)


def test_local_extrema_matching_is_one_to_one_and_penalizes_extra_peaks() -> None:
    oracle = reference_curve()
    predicted = oracle + 0.3 * np.cos(np.deg2rad(12.0 * GRID))
    result = compute_score(
        facts=complete_facts(),
        profile=predicted,
        oracle=oracle,
        eligibility_gates=passing_gates(),
    )
    local = result["categories"]["extrema_and_barriers"]["local_extrema"]
    for kind in ("minima", "maxima"):
        matches = local[kind]["matches"]
        assert len({item["predicted_index"] for item in matches}) == len(matches)
        assert len({item["reference_index"] for item in matches}) == len(matches)
    assert 0.0 <= local["score"] <= 8.0


@pytest.mark.parametrize(
    "bad_profile",
    [
        np.zeros(71),
        np.concatenate([np.zeros(71), [np.nan]]),
        {"grid_deg": GRID + 1.0, "energies_kcal_mol": np.zeros(72)},
    ],
)
def test_curve_contract_rejects_wrong_size_nonfinite_and_nonstandard_grid(
    bad_profile: object,
) -> None:
    with pytest.raises(ScoringInputError):
        compute_score(
            facts=complete_facts(),
            profile=bad_profile,
            oracle=reference_curve(),
            eligibility_gates=passing_gates(),
        )


def test_fractional_adaptive_and_process_facts_are_clamped_and_weighted() -> None:
    oracle = reference_curve()
    facts = {
        "adaptive_sampling": {
            "coverage_fraction": 0.5,
            "information_gain_fraction": 0.25,
            "refinement_fraction": 2.0,
            "efficiency_fraction": -1.0,
        },
        "process_and_reproducibility": {
            "trajectory_and_tool_integrity_fraction": 1.0,
            "native_attempt_provenance_fraction": 0.5,
            "budget_and_protocol_fraction": 0.0,
            "independent_recomputation_fraction": 1.0,
            "replay_consistency_fraction": 0.5,
        },
    }
    result = compute_score(
        facts=facts,
        profile=oracle,
        oracle=oracle,
        eligibility_gates=passing_gates(),
    )
    assert result["categories"]["adaptive_sampling"]["score"] == pytest.approx(7.0)
    assert result["categories"]["process_and_reproducibility"]["score"] == pytest.approx(9.0)

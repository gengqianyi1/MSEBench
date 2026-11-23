from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from evaluator.symmetry import (
    STANDARD_GRID_DEG,
    SymmetryScoringError,
    adjacent_periodic_barrier_multiset,
    evaluate_symmetry_aware_profile,
    extract_extrema_basins,
    global_minimum_basin_comparison,
    match_barrier_multisets,
    optimal_circular_extrema_matches,
)


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "evaluator_private"
    / "regression_fixtures"
    / "v1_1_symmetry_profiles.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _piecewise_periodic_curve(keypoints: dict[int, float]) -> np.ndarray:
    ordered = sorted((float(angle), float(value)) for angle, value in keypoints.items())
    angles = np.asarray([item[0] for item in ordered], dtype=np.float64)
    values = np.asarray([item[1] for item in ordered], dtype=np.float64)
    extended_angles = np.concatenate([angles, [angles[0] + 360.0]])
    extended_values = np.concatenate([values, [values[0]]])
    query = np.where(STANDARD_GRID_DEG < angles[0], STANDARD_GRID_DEG + 360.0, STANDARD_GRID_DEG)
    return np.interp(query, extended_angles, extended_values)


def _alternating_barrier_curve() -> np.ndarray:
    return _piecewise_periodic_curve(
        {
            0: 0.0,
            45: 1.0,
            90: 0.0,
            135: 4.0,
            180: 0.0,
            225: 1.0,
            270: 0.0,
            315: 4.0,
        }
    )


def test_regression_fixture_is_package_safe_and_complete() -> None:
    raw = FIXTURE.read_text(encoding="utf-8")
    assert ":\\" not in raw
    assert "\\\\" not in raw
    fixture = json.loads(raw)
    assert fixture["schema_version"] == "v1.1-scoring-regression/1.0"
    assert fixture["grid_angles_deg"] == list(range(0, 360, 5))
    assert set(fixture["curves"]) == {"oracle", "baseline", "kimi", "deepseek"}
    for curve in fixture["curves"].values():
        assert len(curve["relative_energies_kcal_mol"]) == 72
        assert len(curve["profile_sha256"]) == 64


def test_v1_1_oracle_has_four_admissible_global_basins() -> None:
    fixture = _fixture()
    angles = fixture["grid_angles_deg"]
    oracle = fixture["curves"]["oracle"]["relative_energies_kcal_mol"]
    comparison = global_minimum_basin_comparison(angles, oracle, angles, oracle)
    admissible = comparison["oracle_admissible_basins"]

    assert [basin["angle_deg"] for basin in admissible] == [45.0, 135.0, 225.0, 315.0]
    assert [basin["energy_kcal_mol"] for basin in admissible] == pytest.approx(
        [0.0, 0.000475691712, 0.000475703634, 0.000000053964],
        abs=1.0e-12,
    )
    assert comparison["predicted_global_basins"][0]["angle_deg"] == 45.0
    assert comparison["circular_error_deg"] == 0.0


@pytest.mark.parametrize("run", ["baseline", "kimi", "deepseek"])
def test_historical_v1_1_profiles_match_frozen_v2_regression(run: str) -> None:
    fixture = _fixture()
    angles = fixture["grid_angles_deg"]
    oracle = fixture["curves"]["oracle"]["relative_energies_kcal_mol"]
    predicted = fixture["curves"][run]["relative_energies_kcal_mol"]
    expected = fixture["expected_v2"][run]

    result = evaluate_symmetry_aware_profile(angles, predicted, angles, oracle)

    assert result["global_minimum_basins"]["circular_error_deg"] == pytest.approx(
        expected["global_basin_error_deg"]
    )
    assert result["local_extrema"]["minima"]["f1"] == pytest.approx(
        expected["minima_f1"]
    )
    assert result["local_extrema"]["maxima"]["f1"] == pytest.approx(
        expected["maxima_f1"]
    )
    barriers = result["adjacent_periodic_barriers"]
    assert barriers["mae_kcal_mol"] == pytest.approx(
        expected["barrier_mae_kcal_mol"], abs=1.0e-12
    )
    assert barriers["maximum_error_kcal_mol"] == pytest.approx(
        expected["barrier_max_error_kcal_mol"], abs=1.0e-12
    )
    assert barriers["predicted_count"] == barriers["oracle_count"] == 4


def test_historical_baseline_and_deepseek_no_longer_get_false_90_degree_errors() -> None:
    fixture = _fixture()
    angles = fixture["grid_angles_deg"]
    oracle = fixture["curves"]["oracle"]["relative_energies_kcal_mol"]

    baseline = global_minimum_basin_comparison(
        angles,
        fixture["curves"]["baseline"]["relative_energies_kcal_mol"],
        angles,
        oracle,
    )
    deepseek = global_minimum_basin_comparison(
        angles,
        fixture["curves"]["deepseek"]["relative_energies_kcal_mol"],
        angles,
        oracle,
    )

    assert baseline["predicted_global_basins"][0]["angle_deg"] == 315.0
    assert baseline["nearest_match"] == {
        "predicted_angle_deg": 315.0,
        "oracle_angle_deg": 315.0,
        "distance_deg": 0.0,
    }
    assert deepseek["predicted_global_basins"][0]["angle_deg"] == 130.0
    assert deepseek["nearest_match"] == {
        "predicted_angle_deg": 130.0,
        "oracle_angle_deg": 135.0,
        "distance_deg": 5.0,
    }


@pytest.mark.parametrize("shift", [1, 7, 18, 35])
def test_periodic_rotation_preserves_adjacent_barrier_multiset(shift: int) -> None:
    oracle = _alternating_barrier_curve()
    predicted = np.roll(oracle, shift)

    oracle_barriers = adjacent_periodic_barrier_multiset(STANDARD_GRID_DEG, oracle)
    predicted_barriers = adjacent_periodic_barrier_multiset(STANDARD_GRID_DEG, predicted)
    comparison = match_barrier_multisets(
        predicted_barriers["barrier_multiset_kcal_mol"],
        oracle_barriers["barrier_multiset_kcal_mol"],
    )

    assert oracle_barriers["barrier_multiset_kcal_mol"] == [1.0, 1.0, 4.0, 4.0]
    assert predicted_barriers["barrier_multiset_kcal_mol"] == [1.0, 1.0, 4.0, 4.0]
    assert comparison["mae_kcal_mol"] == 0.0
    assert comparison["maximum_error_kcal_mol"] == 0.0


def test_periodic_reflection_preserves_adjacent_barrier_multiset() -> None:
    oracle = _alternating_barrier_curve()
    reflected = np.asarray([oracle[(-index) % 72] for index in range(72)])

    reference = adjacent_periodic_barrier_multiset(STANDARD_GRID_DEG, oracle)
    candidate = adjacent_periodic_barrier_multiset(STANDARD_GRID_DEG, reflected)
    comparison = match_barrier_multisets(
        candidate["barrier_multiset_kcal_mol"],
        reference["barrier_multiset_kcal_mol"],
    )

    assert comparison["mae_kcal_mol"] == 0.0
    assert comparison["maximum_error_kcal_mol"] == 0.0


def test_equivalent_90_degree_global_minimum_is_not_penalized() -> None:
    oracle = _alternating_barrier_curve()
    predicted = np.roll(oracle, 18)
    comparison = global_minimum_basin_comparison(
        STANDARD_GRID_DEG,
        predicted,
        STANDARD_GRID_DEG,
        oracle,
    )

    assert comparison["circular_error_deg"] == 0.0
    assert [item["angle_deg"] for item in comparison["oracle_admissible_basins"]] == [
        0.0,
        90.0,
        180.0,
        270.0,
    ]


def test_zero_boundary_plateau_is_one_periodic_basin() -> None:
    energies = np.full(72, 2.0)
    energies[[71, 0, 1]] = 0.0
    result = extract_extrema_basins(STANDARD_GRID_DEG, energies)

    assert result["minimum_count"] == 1
    minimum = result["minima"][0]
    assert minimum["angle_deg"] == 0.0
    assert minimum["member_angles_deg"] == [0.0, 5.0, 355.0]
    assert minimum["raw_extremum_count"] == 1


def test_close_minima_merge_across_zero_by_single_linkage() -> None:
    energies = np.full(72, 2.0)
    energies[70] = 1.0  # 350
    energies[71] = 0.0  # 355, first raw minimum
    energies[0] = 0.2   # 0, intervening shallow maximum
    energies[1] = 0.05  # 5, second raw minimum
    energies[2] = 1.0   # 10

    result = extract_extrema_basins(STANDARD_GRID_DEG, energies)

    assert result["minimum_count"] == 1
    minimum = result["minima"][0]
    assert minimum["angle_deg"] == 355.0
    assert minimum["raw_extremum_count"] == 2
    assert minimum["member_angles_deg"] == [5.0, 355.0]


def test_extrema_matching_maximizes_cardinality_before_distance() -> None:
    result = optimal_circular_extrema_matches(
        [359.0, 9.0], [0.0, 20.0], tolerance_deg=11.0
    )
    pairs = [
        (item["predicted_angle_deg"], item["oracle_angle_deg"], item["distance_deg"])
        for item in result["matches"]
    ]

    assert pairs == [(9.0, 20.0, 11.0), (359.0, 0.0, 1.0)]
    assert result["match_count"] == 2
    assert result["total_matched_distance_deg"] == 12.0
    assert result["f1"] == 1.0


def test_extrema_matching_minimizes_distance_and_is_input_order_deterministic() -> None:
    first = optimal_circular_extrema_matches(
        [1.0, 11.0], [0.0, 10.0], tolerance_deg=20.0
    )
    permuted = optimal_circular_extrema_matches(
        [11.0, 1.0], [10.0, 0.0], tolerance_deg=20.0
    )

    first_angles = [
        (item["predicted_angle_deg"], item["oracle_angle_deg"])
        for item in first["matches"]
    ]
    permuted_angles = [
        (item["predicted_angle_deg"], item["oracle_angle_deg"])
        for item in permuted["matches"]
    ]
    assert first_angles == permuted_angles == [(1.0, 0.0), (11.0, 10.0)]
    assert first["total_matched_distance_deg"] == permuted["total_matched_distance_deg"] == 2.0


def test_local_extrema_f1_keeps_all_near_degenerate_global_minima() -> None:
    fixture = _fixture()
    angles = fixture["grid_angles_deg"]
    oracle = fixture["curves"]["oracle"]["relative_energies_kcal_mol"]
    baseline = fixture["curves"]["baseline"]["relative_energies_kcal_mol"]
    result = evaluate_symmetry_aware_profile(angles, baseline, angles, oracle)

    minima = result["local_extrema"]["minima"]
    assert minima["predicted_count"] == minima["oracle_count"] == minima["match_count"] == 4
    assert minima["f1"] == 1.0


def test_adjacent_barrier_references_deeper_endpoint_basin() -> None:
    curve = _piecewise_periodic_curve(
        {0: 0.0, 90: 3.0, 180: 0.5, 270: 4.0}
    )
    result = adjacent_periodic_barrier_multiset(STANDARD_GRID_DEG, curve)

    assert result["barrier_multiset_kcal_mol"] == [3.0, 4.0]
    assert [edge["lower_basin_energy_kcal_mol"] for edge in result["edges"]] == [
        0.0,
        0.0,
    ]


def test_single_basin_uses_one_complete_period() -> None:
    curve = 1.0 - np.cos(np.deg2rad(STANDARD_GRID_DEG))
    result = adjacent_periodic_barrier_multiset(STANDARD_GRID_DEG, curve)

    assert result["barrier_count"] == 1
    assert result["barrier_multiset_kcal_mol"] == [2.0]
    assert result["edges"][0]["clockwise_arc_point_count"] == 72
    assert result["edges"][0]["start_basin_angle_deg"] == 0.0
    assert result["edges"][0]["end_basin_angle_deg"] == 0.0


def test_unmatched_barrier_cardinality_is_not_ignored() -> None:
    result = match_barrier_multisets([1.1, 3.9], [1.0, 2.0, 4.0])

    assert result["matched_count"] == 2
    assert result["unmatched_predicted_kcal_mol"] == []
    assert result["unmatched_oracle_kcal_mol"] == [2.0]
    assert result["mae_kcal_mol"] == pytest.approx(0.566666666667)
    assert result["maximum_error_kcal_mol"] == 1.5


def test_flat_curve_edge_cases_fail_closed_without_nonfinite_metrics() -> None:
    flat = np.zeros(72)
    identical = evaluate_symmetry_aware_profile(
        STANDARD_GRID_DEG, flat, STANDARD_GRID_DEG, flat
    )
    assert identical["global_minimum_basins"]["circular_error_deg"] == 0.0
    assert identical["local_extrema"]["minima"]["f1"] == 1.0
    assert identical["local_extrema"]["maxima"]["f1"] == 1.0
    assert identical["adjacent_periodic_barriers"]["mae_kcal_mol"] == 0.0
    assert identical["adjacent_periodic_barriers"]["maximum_error_kcal_mol"] == 0.0

    one_well = 1.0 - np.cos(np.deg2rad(STANDARD_GRID_DEG))
    mismatch = evaluate_symmetry_aware_profile(
        STANDARD_GRID_DEG, flat, STANDARD_GRID_DEG, one_well
    )
    assert mismatch["global_minimum_basins"]["circular_error_deg"] == 180.0
    assert mismatch["local_extrema"]["minima"]["f1"] == 0.0
    assert mismatch["local_extrema"]["maxima"]["f1"] == 0.0
    assert mismatch["adjacent_periodic_barriers"]["mae_kcal_mol"] == 1.5
    assert mismatch["adjacent_periodic_barriers"]["maximum_error_kcal_mol"] == 1.5


def test_additive_energy_offsets_do_not_change_any_scientific_metric() -> None:
    oracle = _alternating_barrier_curve()
    predicted = np.roll(oracle, 7)
    original = evaluate_symmetry_aware_profile(
        STANDARD_GRID_DEG, predicted, STANDARD_GRID_DEG, oracle
    )
    shifted = evaluate_symmetry_aware_profile(
        STANDARD_GRID_DEG,
        predicted + 987.654,
        STANDARD_GRID_DEG,
        oracle - 123.456,
    )

    assert shifted == original
    json.dumps(shifted, allow_nan=False)


def test_scientific_api_is_structurally_independent_from_eligibility() -> None:
    signature = inspect.signature(evaluate_symmetry_aware_profile)
    assert "eligibility" not in signature.parameters
    assert "facts" not in signature.parameters
    assert "model_submission" not in signature.parameters

    curve = _alternating_barrier_curve()
    result = evaluate_symmetry_aware_profile(
        STANDARD_GRID_DEG, curve, STANDARD_GRID_DEG, curve
    )
    serialized = json.dumps(result, sort_keys=True, allow_nan=False)
    assert "eligible" not in serialized
    assert "process" not in serialized
    assert "model" not in serialized


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_curve_values_are_rejected(bad_value: float) -> None:
    curve = _alternating_barrier_curve()
    curve[5] = bad_value
    with pytest.raises(SymmetryScoringError, match="finite"):
        evaluate_symmetry_aware_profile(
            STANDARD_GRID_DEG, curve, STANDARD_GRID_DEG, np.zeros(72)
        )


def test_wrong_grid_duplicate_angles_and_negative_barriers_are_rejected() -> None:
    curve = np.zeros(72)
    wrong_grid = STANDARD_GRID_DEG.copy()
    wrong_grid[-1] = 0.0
    with pytest.raises(SymmetryScoringError, match="standard"):
        extract_extrema_basins(wrong_grid, curve)
    with pytest.raises(SymmetryScoringError, match="negative"):
        match_barrier_multisets([1.0, -0.1], [1.0])


def test_repeated_evaluation_is_byte_deterministic() -> None:
    fixture = _fixture()
    angles = fixture["grid_angles_deg"]
    oracle = fixture["curves"]["oracle"]["relative_energies_kcal_mol"]
    predicted = fixture["curves"]["deepseek"]["relative_energies_kcal_mol"]

    first = evaluate_symmetry_aware_profile(angles, predicted, angles, oracle)
    second = evaluate_symmetry_aware_profile(angles, predicted, angles, oracle)

    first_bytes = json.dumps(
        first, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    second_bytes = json.dumps(
        second, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert first_bytes == second_bytes

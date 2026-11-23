from __future__ import annotations

import math

import numpy as np
import pytest

from tools.profile import (
    PROFILE_GRID_DEG,
    ProfileFitError,
    analyze_profile,
    fit_periodic_profile,
    fourier_predict,
    largest_circular_gaps,
    model_disagreement_intervals,
    periodic_cubic_predict,
)


def test_periodic_cubic_interpolates_and_is_c2_at_seam() -> None:
    angles = [0, 45, 90, 140, 180, 225, 270, 320]
    energies = [1.2 + math.cos(math.radians(2 * angle)) for angle in angles]

    result = fit_periodic_profile(angles, energies, method="periodic_cubic")

    assert result["grid_angles_deg"] == list(PROFILE_GRID_DEG)
    assert len(result["predicted_energies"]) == 72
    assert np.isfinite(result["predicted_energies"]).all()
    assert result["diagnostics"]["continuity"] == "C2_periodic"
    assert result["diagnostics"]["sample_max_abs_error"] < 1.0e-12
    assert result["diagnostics"]["seam_first_derivative_error"] < 1.0e-12
    assert result["diagnostics"]["seam_second_derivative_error"] == 0.0

    at_samples = periodic_cubic_predict(angles, energies, angles)
    np.testing.assert_allclose(at_samples, energies, rtol=0.0, atol=1.0e-12)
    across_period = periodic_cubic_predict(angles, energies, [0.0, 360.0, -360.0])
    np.testing.assert_allclose(across_period, [energies[0]] * 3, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("order", [0, 7, -1, 1.5, True, None])
def test_fourier_order_validation(order: object) -> None:
    samples = list(range(0, 360, 30))
    energies = [math.cos(math.radians(value)) for value in samples]
    with pytest.raises(ProfileFitError, match="order"):
        fit_periodic_profile(samples, energies, method="fourier", order=order)  # type: ignore[arg-type]


def test_fourier_rejects_underdetermined_design() -> None:
    angles = list(range(0, 360, 45))  # 8 samples, but order 4 has 9 coefficients.
    energies = [math.sin(math.radians(angle)) for angle in angles]
    with pytest.raises(ProfileFitError, match="requires at least 9"):
        fourier_predict(angles, energies, order=4)


def test_fourier_reconstructs_known_profile_and_is_finite() -> None:
    angles = list(range(0, 360, 30))

    def expected(angle: float) -> float:
        theta = math.radians(angle)
        return 2.5 + 0.8 * math.cos(theta) - 0.3 * math.sin(2.0 * theta)

    energies = [expected(angle) for angle in angles]
    result = fit_periodic_profile(angles, energies, method="fourier", order=2)

    assert result["diagnostics"]["design_rank"] == 5
    assert result["diagnostics"]["determined"] is True
    assert result["diagnostics"]["sample_max_abs_error"] < 1.0e-12
    np.testing.assert_allclose(
        result["predicted_energies"],
        [expected(angle) for angle in PROFILE_GRID_DEG],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert len(result["curve_sha256"]) == 64
    assert result == fit_periodic_profile(angles, energies, method="fourier", order=2)


@pytest.mark.parametrize("bad_energy", [float("nan"), float("inf"), -float("inf")])
def test_all_fit_inputs_and_outputs_must_be_finite(bad_energy: float) -> None:
    with pytest.raises(ProfileFitError, match="finite"):
        fit_periodic_profile([0, 120, 240], [0.0, bad_energy, 1.0])


def test_extrema_and_frozen_directional_barriers() -> None:
    # Two equal wells (0, 180) and two equal transition states (90, 270).
    energies = [2.0 - 2.0 * math.cos(math.radians(2 * angle)) for angle in PROFILE_GRID_DEG]

    analysis = analyze_profile(energies)

    assert analysis["global_minimum"]["angle_deg"] == 0
    assert analysis["global_minimum"]["degenerate_angles_deg"] == [0, 180]
    assert [item["angle_deg"] for item in analysis["local_minima"]] == [0, 180]
    assert [item["angle_deg"] for item in analysis["local_maxima"]] == [90, 270]
    assert analysis["barriers"]["forward"]["destination_minimum_angle_deg"] == 180
    assert analysis["barriers"]["forward"]["transition_state_angle_deg"] == 90
    assert analysis["barriers"]["reverse"]["destination_minimum_angle_deg"] == 180
    assert analysis["barriers"]["reverse"]["transition_state_angle_deg"] == 270
    assert analysis["forward_barrier"] == pytest.approx(4.0)
    assert analysis["reverse_barrier"] == pytest.approx(4.0)


def test_largest_gap_and_disagreement_signals_are_deterministic() -> None:
    gaps = largest_circular_gaps([0, 30, 180, 210], limit=2)
    assert [(item["start_angle_deg"], item["width_deg"]) for item in gaps] == [
        (30, 150),
        (210, 150),
    ]
    assert [item["recommended_angle_deg"] for item in gaps] == [105, 285]

    primary = [math.cos(math.radians(angle)) for angle in PROFILE_GRID_DEG]
    comparison = [math.cos(math.radians(angle - 10)) for angle in PROFILE_GRID_DEG]
    disagreement = model_disagreement_intervals(primary, comparison, limit=3)
    assert disagreement["available"] is True
    assert disagreement["rmse"] > 0.0
    assert len(disagreement["intervals"]) == 3
    assert np.isfinite(
        [item["abs_difference"] for item in disagreement["intervals"]]
    ).all()


def test_fit_payload_can_be_analyzed_directly() -> None:
    angles = list(range(0, 360, 30))
    energies = [1.0 - math.cos(math.radians(2 * angle)) for angle in angles]
    fitted = fit_periodic_profile(angles, energies, method="fourier", order=2)

    analysis = analyze_profile(fitted)

    assert analysis["grid_size"] == 72
    assert analysis["largest_gaps"]
    assert fitted["refinement_signals"]["model_disagreement"]["available"] is True
    assert all(
        angle not in angles
        for angle in fitted["refinement_signals"]["recommended_angles_deg"]
    )

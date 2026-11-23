"""Frozen oracle-blind profile selector for protocol 2.0."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np

from tools.profile import (
    PROFILE_GRID_DEG,
    ProfileFitError,
    analyze_profile,
    fit_periodic_profile,
    fourier_predict,
    periodic_cubic_predict,
)
from tools.science import canonical_json_bytes


SELECTOR_ID = "auto_periodic_fit_v2"
TIE_TOLERANCE = 1.0e-12


def _loo_prediction(
    method: str,
    order: int | None,
    angles: list[int],
    energies: list[float],
) -> tuple[float, float]:
    residuals: list[float] = []
    for held_out in range(len(angles)):
        train_angles = angles[:held_out] + angles[held_out + 1 :]
        train_energies = energies[:held_out] + energies[held_out + 1 :]
        query = [angles[held_out]]
        if method == "periodic_cubic":
            predicted = periodic_cubic_predict(
                train_angles, train_energies, query_angles_deg=query
            )
        else:
            predicted = fourier_predict(
                train_angles,
                train_energies,
                order=int(order),
                query_angles_deg=query,
            )
        residuals.append(float(predicted[0]) - energies[held_out])
    array = np.asarray(residuals, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ProfileFitError("cross-validation prediction is non-finite")
    return (
        float(np.sqrt(np.mean(array**2))),
        float(np.max(np.abs(array))),
    )


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[float, int, int, int]:
    # The 1e-12 quantization makes the tie policy portable across BLAS builds.
    score = round(float(candidate["loo_rmse_kcal_mol"]) / TIE_TOLERANCE)
    family_rank = 0 if candidate["method"] == "periodic_cubic" else 1
    order = int(candidate.get("order") or 0)
    parameters = int(candidate["parameter_count"])
    return float(score), parameters, family_rank, order


def select_profile(samples_kcal_mol: Mapping[int, float]) -> dict[str, Any]:
    """Compare every feasible frozen candidate using leave-one-out RMSE."""

    if len(samples_kcal_mol) < 4:
        raise ProfileFitError("automatic profile fitting requires at least 4 valid points")
    angles = sorted(int(angle) for angle in samples_kcal_mol)
    energies = [float(samples_kcal_mol[angle]) for angle in angles]
    if len(set(angles)) != len(angles) or not all(
        math.isfinite(value) for value in energies
    ):
        raise ProfileFitError("automatic profile samples are invalid")

    specifications: list[tuple[str, int | None, int]] = [
        ("periodic_cubic", None, len(angles))
    ]
    # Each LOO training fold must remain determined: n - 1 >= 1 + 2*k.
    maximum_order = min(6, (len(angles) - 2) // 2)
    specifications.extend(
        ("fourier", order, 1 + 2 * order)
        for order in range(1, maximum_order + 1)
    )

    candidates: list[dict[str, Any]] = []
    fitted: list[dict[str, Any]] = []
    for method, order, parameter_count in specifications:
        try:
            loo_rmse, loo_maxae = _loo_prediction(
                method, order, angles, energies
            )
            fit = fit_periodic_profile(
                angles,
                energies,
                method=method,
                order=order,
            )
        except ProfileFitError:
            continue
        candidate = {
            "method": method,
            "order": order,
            "parameter_count": parameter_count,
            "loo_rmse_kcal_mol": loo_rmse,
            "loo_maxae_kcal_mol": loo_maxae,
            "curve_sha256": fit["curve_sha256"],
        }
        candidates.append(candidate)
        fitted.append(fit)
    if not candidates:
        raise ProfileFitError("no well-determined automatic profile candidate")

    ranking = sorted(range(len(candidates)), key=lambda index: _candidate_key(candidates[index]))
    selected_index = ranking[0]
    selected = fitted[selected_index]
    runner_up = fitted[ranking[1]] if len(ranking) > 1 else None
    selected_analysis = analyze_profile(
        selected,
        comparison_energies=(
            runner_up["predicted_energies"] if runner_up is not None else None
        ),
    )
    selector = {
        "selector_id": SELECTOR_ID,
        "criterion": "leave_one_out_rmse_kcal_mol",
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": [
            "smaller_parameter_count",
            "periodic_cubic_before_fourier",
            "lower_fourier_order",
        ],
        "candidates": candidates,
        "selected_rank": 1,
        "selected_method": selected["method"],
        "selected_order": selected["order"],
    }
    scientific = {
        **selected,
        "selector": selector,
        "analysis": selected_analysis,
    }
    profile_hash = hashlib.sha256(canonical_json_bytes(scientific)).hexdigest().upper()
    return {
        **scientific,
        "profile_hash": profile_hash,
    }


def compact_profile_analysis(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Project a full profile to the bounded model-visible feedback schema."""

    analysis = profile["analysis"]
    signals = profile["refinement_signals"]
    source_energies = [float(value) for value in profile["source_energies"]]
    minimum = min(source_energies)
    relative = [
        round(value - minimum, 8)
        for value in source_energies
    ]
    gaps = [
        {
            "start_deg": item["start_angle_deg"],
            "end_deg": item["end_angle_deg"],
            "width_deg": item["width_deg"],
            "recommended_deg": item["recommended_angle_deg"],
        }
        for item in analysis["largest_gaps"][:3]
    ]
    curvature = [
        {
            "center_deg": item["center_angle_deg"],
            "strength": round(
                float(item["abs_second_derivative_per_rad2"]), 6
            ),
        }
        for item in analysis["high_curvature_intervals"][:3]
    ]
    disagreement = [
        {
            "center_deg": item["center_angle_deg"],
            "abs_difference": round(float(item["abs_difference"]), 6),
        }
        for item in analysis["model_disagreement"]["intervals"][:3]
    ]
    recommendations = [
        {
            "angle_deg": item["angle_deg"],
            "reason": item["reason"],
            "priority_rank": item["priority_rank"],
        }
        for item in signals["recommendations"][:4]
    ]
    return {
        "selected_fit": {
            "family": profile["method"],
            "fourier_order": profile["order"],
            "loo_rmse_kcal_mol": next(
                candidate["loo_rmse_kcal_mol"]
                for candidate in profile["selector"]["candidates"]
                if candidate["method"] == profile["method"]
                and candidate["order"] == profile["order"]
            ),
        },
        "sampled_angles_deg": list(profile["source_angles_deg"]),
        "relative_energies_kcal_mol": relative,
        "global_minimum_basin_candidates_deg": list(
            analysis["global_minimum"]["degenerate_angles_deg"]
        ),
        "local_minima_candidates_deg": [
            item["angle_deg"] for item in analysis["local_minima"]
        ],
        "local_maxima_candidates_deg": [
            item["angle_deg"] for item in analysis["local_maxima"]
        ],
        "largest_unsampled_gaps": gaps,
        "high_curvature_intervals": curvature,
        "high_model_disagreement_intervals": disagreement,
        "recommended_refinement": recommendations,
    }


__all__ = [
    "SELECTOR_ID",
    "compact_profile_analysis",
    "select_profile",
]

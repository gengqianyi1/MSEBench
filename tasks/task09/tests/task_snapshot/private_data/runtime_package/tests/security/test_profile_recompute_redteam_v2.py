from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest

from evaluator.evaluate_v2 import (
    EvaluationError,
    _action_ledger,
    _load_trajectory,
    _verify_attempts,
    _verify_profiles,
    _verify_scan_result_feedback,
    evaluate_run,
)
from evaluator.profile_recompute_v2 import (
    ProfileRecomputeError,
    verify_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "trajectories" / "baseline"
Mutation = tuple[
    str,
    tuple[str | int, ...],
    Callable[[Any], Any],
    str,
]


def _terminal_profile(root: Path = BASELINE) -> tuple[Path, dict[str, Any]]:
    paths = sorted((root / "artifacts" / "profiles").glob("profile_*.json"))
    assert paths
    path = paths[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _replace(
    profile: dict[str, Any],
    path: tuple[str | int, ...],
    transform: Callable[[Any], Any],
) -> None:
    target: Any = profile
    for component in path[:-1]:
        target = target[component]
    final = path[-1]
    target[final] = transform(target[final])


def _refresh_profile_hash(profile: dict[str, Any]) -> None:
    """Refresh the legitimate outer scientific hash after an inner forgery."""

    scientific_payload = {
        "method": profile["method"],
        "order": profile["order"],
        "source_angles_deg": profile["source_angles_deg"],
        "source_energies": profile["source_energies"],
        "grid_angles_deg": profile["grid_angles_deg"],
        "predicted_energies": profile["predicted_energies"],
    }
    scientific = {
        "schema_version": "1.0",
        **scientific_payload,
        "curve_sha256": profile["curve_sha256"],
        "diagnostics": profile["diagnostics"],
        "refinement_signals": profile["refinement_signals"],
        "selector": profile["selector"],
        "analysis": profile["analysis"],
    }
    encoded = (
        json.dumps(
            scientific,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    profile["profile_hash"] = hashlib.sha256(encoded).hexdigest().upper()


MUTATIONS: tuple[Mutation, ...] = (
    (
        "diagnostics-value",
        ("diagnostics", "sample_rmse"),
        lambda value: float(value) + 0.5,
        "diagnostics",
    ),
    (
        "diagnostics-bool-number-confusion",
        ("diagnostics", "determined"),
        lambda _value: 1,
        "diagnostics",
    ),
    (
        "refinement-recommended-angle",
        ("refinement_signals", "recommended_angles_deg", 0),
        lambda value: (int(value) + 5) % 360,
        "refinement signals",
    ),
    (
        "refinement-priority-bool-number-confusion",
        ("refinement_signals", "recommendations", 0, "priority_rank"),
        lambda _value: True,
        "refinement signals",
    ),
    (
        "analysis-global-minimum",
        ("analysis", "global_minimum", "angle_deg"),
        lambda value: (int(value) + 5) % 360,
        "analysis",
    ),
    (
        "analysis-local-minimum-energy",
        ("analysis", "local_minima", 0, "energy"),
        lambda value: float(value) + 0.5,
        "analysis",
    ),
    (
        "analysis-local-maximum-angle",
        ("analysis", "local_maxima", 0, "angle_deg"),
        lambda value: (int(value) + 5) % 360,
        "analysis",
    ),
    (
        "analysis-transition-state",
        (
            "analysis",
            "barriers",
            "forward",
            "transition_state_angle_deg",
        ),
        lambda value: (int(value) + 5) % 360,
        "analysis",
    ),
    (
        "analysis-forward-barrier",
        ("analysis", "forward_barrier"),
        lambda value: float(value) + 0.5,
        "analysis",
    ),
    (
        "analysis-bool-number-confusion",
        ("analysis", "model_disagreement", "available"),
        lambda _value: 1,
        "analysis",
    ),
)


def test_unmodified_frozen_profile_recomputes_independently() -> None:
    _path, profile = _terminal_profile()
    replay = verify_profile_artifact(profile)

    assert replay["selected_method"] == profile["method"]
    assert replay["selected_order"] == profile["order"]
    assert len(replay["predicted_energies"]) == 72


@pytest.mark.parametrize(
    "_case,path,transform,error_fragment",
    MUTATIONS,
    ids=[item[0] for item in MUTATIONS],
)
def test_independent_recompute_rejects_rehashed_scientific_metadata_forgery(
    _case: str,
    path: tuple[str | int, ...],
    transform: Callable[[Any], Any],
    error_fragment: str,
) -> None:
    _artifact_path, original = _terminal_profile()
    forged = deepcopy(original)
    _replace(forged, path, transform)
    _refresh_profile_hash(forged)

    # The forged profile has a self-consistent outer hash. Rejection therefore
    # proves independent scientific replay, not detection of a stale checksum.
    with pytest.raises(ProfileRecomputeError, match=error_fragment):
        verify_profile_artifact(forged)


@pytest.mark.parametrize(
    "_case,path,transform,error_fragment",
    (
        MUTATIONS[0],
        MUTATIONS[2],
        MUTATIONS[4],
        MUTATIONS[7],
        MUTATIONS[8],
    ),
    ids=[
        "diagnostics",
        "refinement",
        "global-extremum",
        "transition-state",
        "barrier",
    ],
)
def test_evaluator_binds_every_terminal_profile_scientific_projection(
    tmp_path: Path,
    _case: str,
    path: tuple[str | int, ...],
    transform: Callable[[Any], Any],
    error_fragment: str,
) -> None:
    run = tmp_path / "run"
    shutil.copytree(BASELINE, run)
    artifact_path, profile = _terminal_profile(run)
    _replace(profile, path, transform)
    _refresh_profile_hash(profile)
    artifact_path.write_text(
        json.dumps(
            profile,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    try:
        result = evaluate_run(run, project_root=ROOT, write=False)
    except (ProfileRecomputeError, EvaluationError) as exc:
        assert error_fragment in str(exc)
    else:
        assert result["run_status"] == "FAIL"
        assert result["eligible"] is False


def test_evaluator_binds_model_visible_compact_feedback_to_profile() -> None:
    events, _trajectory = _load_trajectory(BASELINE / "trajectory.jsonl")
    ledger = _action_ledger(events)
    attempt_info, _eligible = _verify_attempts(
        BASELINE, require_native_logs=False
    )
    profiles = _verify_profiles(
        BASELINE, ledger, attempt_info["summaries"]
    )
    forged_events = deepcopy(events)
    result_event = next(
        event
        for event in forged_events
        if event["event"] == "public_action_result"
        and event["payload"]["action"] == "run_scan_batch"
    )
    recommendation = result_event["payload"]["result"]["profile"][
        "analysis"
    ]["recommended_refinement"][0]
    recommendation["angle_deg"] = (
        int(recommendation["angle_deg"]) + 5
    ) % 360

    with pytest.raises(
        EvaluationError, match="model-visible profile feedback"
    ):
        _verify_scan_result_feedback(
            forged_events,
            ledger=ledger,
            profiles=profiles["profiles"],
            attempts=attempt_info["summaries"],
        )

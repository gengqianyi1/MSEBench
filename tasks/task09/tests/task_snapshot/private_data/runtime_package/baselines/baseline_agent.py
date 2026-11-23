"""Deterministic v2 baseline using only the two public actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framework.agent_runner import RunContext


BASELINE_MODEL_ID = "deterministic-baseline/2.0"


class BaselineFailure(RuntimeError):
    """The public runtime rejected a frozen baseline decision."""


def _require(result: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    if result.get("ok") is not True or result.get("status") not in allowed:
        raise BaselineFailure(f"baseline action failed: {result.get('status')}")
    return result


def _action(
    angles: list[int],
    *,
    finalize: bool,
    rationale: str,
) -> dict[str, Any]:
    return {
        "action": "run_scan_batch",
        "angles_deg": angles,
        "finalize": finalize,
        "confidence": "high" if finalize else None,
        "rationale": rationale,
    }


def _feedback_angles(result: dict[str, Any], *, count: int = 3) -> list[int]:
    """Choose new grid points solely from the previous public feedback."""

    profile = result.get("profile")
    analysis = profile.get("analysis") if isinstance(profile, dict) else None
    state = result.get("state")
    sampled = set(state.get("sampled_angles_deg", [])) if isinstance(state, dict) else set()
    if not isinstance(analysis, dict):
        raise BaselineFailure("public feedback has no profile analysis")
    candidates: list[int] = []
    sources = (
        (analysis.get("recommended_refinement"), "angle_deg"),
        (analysis.get("largest_unsampled_gaps"), "recommended_deg"),
    )
    for records, field in sources:
        if not isinstance(records, list):
            continue
        for record in records:
            angle = record.get(field) if isinstance(record, dict) else None
            if (
                isinstance(angle, int)
                and not isinstance(angle, bool)
                and 0 <= angle < 360
                and angle % 5 == 0
                and angle not in sampled
                and angle not in candidates
            ):
                candidates.append(angle)
            if len(candidates) == count:
                return candidates
    if len(candidates) < 2:
        raise BaselineFailure("public feedback supplied too few legal new angles")
    return candidates[:count]


def run_baseline(destination: Path) -> dict[str, Any]:
    context = RunContext.create(
        destination,
        run_id="baseline-live-v2",
        requested_model_id=BASELINE_MODEL_ID,
        mode="DETERMINISTIC_BASELINE_LIVE_XTB",
    )
    try:
        result: dict[str, Any] | None = None
        for turn in range(1, 4):
            if turn == 1:
                decision = _action(
                    [0, 60, 120, 180, 240, 300],
                    finalize=False,
                    rationale="Establish six evenly separated sectors without using an oracle.",
                )
            else:
                if result is None:
                    raise BaselineFailure("previous public feedback is unavailable")
                angles = _feedback_angles(result)
                decision = _action(
                    angles,
                    finalize=turn == 3,
                    rationale=(
                        "Select the highest-priority legal unsampled angles returned by "
                        f"the preceding public profile feedback (round {turn - 1})."
                    ),
                )
            context.record_assistant(
                turn=turn,
                actual_model_id=BASELINE_MODEL_ID,
                content=decision["rationale"],
                tool_count=1,
            )
            result = context.call_action(turn=turn, action=decision)
            expected = {"BATCH_COMPLETE"} if turn < 3 else {"FINAL_ACCEPTED"}
            _require(result, expected)
        if result is None or result["terminal"]["accepted"] is not True:
            raise BaselineFailure("baseline did not reach terminal finalization")
        return context.finalize_success(turn=3)
    except Exception as exc:
        if not context.recorder.terminal:
            context.finalize_failure(
                safe_status="BASELINE_POLICY_FAILURE",
                run_status="FAIL",
                detail={"exception_type": type(exc).__name__},
            )
        raise


__all__ = ["BASELINE_MODEL_ID", "BaselineFailure", "_feedback_angles", "run_baseline"]

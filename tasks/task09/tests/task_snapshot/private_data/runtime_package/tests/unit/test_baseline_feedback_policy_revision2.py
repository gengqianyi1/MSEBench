from __future__ import annotations

from baselines.baseline_agent import _feedback_angles


def _feedback(recommended: list[int], sampled: list[int]) -> dict:
    return {
        "profile": {
            "analysis": {
                "recommended_refinement": [
                    {"angle_deg": angle, "priority_rank": rank + 1}
                    for rank, angle in enumerate(recommended)
                ],
                "largest_unsampled_gaps": [],
            }
        },
        "state": {"sampled_angles_deg": sampled},
    }


def test_baseline_selection_is_feedback_dependent() -> None:
    sampled = [0, 60, 120, 180, 240, 300]
    first = _feedback_angles(_feedback([145, 215, 325], sampled))
    second = _feedback_angles(_feedback([25, 155, 335], sampled))
    assert first == [145, 215, 325]
    assert second == [25, 155, 335]
    assert first != second


def test_baseline_never_reuses_a_sampled_angle() -> None:
    result = _feedback([60, 145, 215, 325], [0, 60, 120, 180, 240, 300])
    assert _feedback_angles(result) == [145, 215, 325]


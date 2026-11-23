from __future__ import annotations

from pathlib import Path

from evaluator.evaluate import _budget_facts, _tool_pairs
from framework.budget import recompute_budget
from framework.trajectory import TrajectoryRecorder, load_jsonl


FIXED_TIME = "2026-07-23T00:00:00Z"
FIELDS = (
    "new_unique_angles_this_round",
    "cumulative_unique_angles",
    "unique_angles_remaining",
    "scan_feedback_rounds",
    "profile_fit_rounds",
    "current_sampling_phase",
    "finalization_only",
    "phase_new_unique_angle_limit",
    "phase_new_unique_angles_remaining",
    "phase_fit_pending_analysis",
    "phase_fit_profile_id",
    "initial_sampling_complete",
    "initial_sampling_new_unique_angles",
    "third_feedback_round_reachable",
    "eligibility_gaps",
    "completed_sampling_rounds",
    "sampling_turns",
    "first_sampling_turn_new_angles",
)


def _pair(
    recorder: TrajectoryRecorder,
    *,
    call_id: str,
    turn: int,
    tool: str,
    arguments: dict[str, object],
    result: dict[str, object],
) -> None:
    recorder.append(
        "tool_call",
        {
            "turn": turn,
            "call_id": call_id,
            "tool": tool,
            "arguments": arguments,
        },
    )
    recorder.append(
        "tool_result",
        {
            "turn": turn,
            "call_id": call_id,
            "tool": tool,
            "result": {
                **result,
                # A forged self-report must never affect reconstruction.
                "budget": {
                    "new_unique_angles_this_round": 0,
                    "scan_feedback_rounds": 99,
                },
            },
        },
    )


def _phase_recorder() -> TrajectoryRecorder:
    recorder = TrajectoryRecorder(
        run_id="sampling-v1-1",
        timestamp_fn=lambda: FIXED_TIME,
    )
    recorder.append("run_start", {})
    recorder.append("assistant_turn", {"turn": 1})
    for index, angle in enumerate((0, 5, 10, 15, 20, 25), start=1):
        _pair(
            recorder,
            call_id=f"s{index}",
            turn=1,
            tool="run_scan_point",
            arguments={"angle_deg": angle},
            result={
                "ok": True,
                "status": "SCAN_COMPLETE",
                "accepted": True,
                "scan": {
                    "angle_deg": angle,
                    "profile_eligible": True,
                },
            },
        )
    _pair(
        recorder,
        call_id="fit1",
        turn=1,
        tool="fit_periodic_profile",
        arguments={"method": "periodic_cubic"},
        result={
            "ok": True,
            "status": "PROFILE_FIT_COMPLETE",
            "profile": {"profile_id": "p1"},
        },
    )
    _pair(
        recorder,
        call_id="a1",
        turn=1,
        tool="analyze_profile",
        arguments={"profile_id": "p1"},
        result={
            "ok": True,
            "status": "PROFILE_ANALYSIS",
            "profile_id": "p1",
        },
    )
    for feedback_round, (turn, angle) in enumerate(
        ((2, 30), (3, 35), (4, 40)),
        start=1,
    ):
        recorder.append("assistant_turn", {"turn": turn})
        _pair(
            recorder,
            call_id=f"r{feedback_round}s",
            turn=turn,
            tool="run_scan_point",
            arguments={"angle_deg": angle},
            result={
                "ok": True,
                "status": "SCAN_COMPLETE",
                "accepted": True,
                "scan": {
                    "angle_deg": angle,
                    "profile_eligible": True,
                },
            },
        )
        profile_id = f"p{feedback_round + 1}"
        _pair(
            recorder,
            call_id=f"r{feedback_round}f",
            turn=turn,
            tool="fit_periodic_profile",
            arguments={"method": "periodic_cubic"},
            result={
                "ok": True,
                "status": "PROFILE_FIT_COMPLETE",
                "profile": {"profile_id": profile_id},
            },
        )
        _pair(
            recorder,
            call_id=f"r{feedback_round}a",
            turn=turn,
            tool="analyze_profile",
            arguments={"profile_id": profile_id},
            result={
                "ok": True,
                "status": "PROFILE_ANALYSIS",
                "profile_id": profile_id,
            },
        )
    recorder.append("run_failure", {"status": "TEST_TERMINAL"})
    return recorder


def test_first_sampling_round_is_six_and_replay_is_identical(
    tmp_path: Path,
) -> None:
    recorder = _phase_recorder()
    audit = recompute_budget(recorder.events)
    facts = audit.to_dict()
    assert facts["first_turn_unique_scan_angles"] == 6
    assert facts["first_sampling_turn_new_angles"] == 6
    assert facts["initial_sampling_new_unique_angles"] == 6
    assert facts["scan_feedback_rounds"] == 3
    assert facts["profile_fit_rounds"] == 4
    assert facts["current_sampling_phase"] == "post_feedback"
    assert facts["cumulative_unique_angles"] == 9
    assert facts["unique_angles_remaining"] == 7
    assert facts["new_unique_angles_this_round"] == 0
    assert facts["violations"] == []

    path = tmp_path / "trajectory.jsonl"
    recorder.write(path)
    replayed = recompute_budget(load_jsonl(path)).to_dict()
    assert replayed == facts


def test_framework_and_evaluator_independent_reconstruction_agree() -> None:
    recorder = _phase_recorder()
    framework_facts = recompute_budget(recorder.events).to_dict()
    calls, pairing_valid = _tool_pairs(recorder.events)
    assert pairing_valid is True
    evaluator_facts = _budget_facts(
        calls,
        attempts={
            f"a{angle}": {"angle_deg": angle, "valid": True}
            for angle in (0, 5, 10, 15, 20, 25, 30, 35, 40)
        },
        profiles={
            f"p{index}": {"valid": True}
            for index in range(1, 5)
        },
    )
    for field in FIELDS:
        assert evaluator_facts[field] == framework_facts[field]
    assert evaluator_facts["budgets_respected"] is True


def test_accepted_seventh_initial_angle_is_detected_as_violation() -> None:
    recorder = TrajectoryRecorder(
        run_id="bad-phase",
        timestamp_fn=lambda: FIXED_TIME,
    )
    recorder.append("run_start", {})
    recorder.append("assistant_turn", {"turn": 1})
    for index, angle in enumerate((0, 5, 10, 15, 20, 25, 30), start=1):
        _pair(
            recorder,
            call_id=f"s{index}",
            turn=1,
            tool="run_scan_point",
            arguments={"angle_deg": angle},
            result={
                "ok": True,
                "status": "SCAN_COMPLETE",
                "scan": {"angle_deg": angle, "profile_eligible": True},
            },
        )
    recorder.append("run_failure", {"status": "TEST_TERMINAL"})
    audit = recompute_budget(recorder.events)
    assert any(
        "sampling_phase_unique_angle_limit_exceeded" in item
        for item in audit.violations
    )
    assert audit.third_feedback_round_reachable is False


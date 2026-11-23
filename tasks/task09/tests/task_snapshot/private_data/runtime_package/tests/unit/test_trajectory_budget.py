from __future__ import annotations

import copy
from pathlib import Path

import pytest

from framework.budget import (
    BudgetError,
    BudgetExceededError,
    BudgetLedger,
    BudgetLimits,
    normalize_scan_angle,
    recompute_budget,
    reconstruct_feedback_rounds,
)
from framework.trajectory import (
    GENESIS_HASH,
    TrajectoryError,
    TrajectoryRecorder,
    canonical_json,
    canonical_sha256,
    load_jsonl,
    trajectory_commitment,
    validate_hash_chain,
    verify_hash_chain,
)


FIXED_TIME = "2026-07-20T00:00:00Z"


def _complete_recorder() -> TrajectoryRecorder:
    recorder = TrajectoryRecorder(
        run_id="unit-run",
        protocol="adaptive_torsion_sampling_tool_loop/1.1",
        timestamp_fn=lambda: FIXED_TIME,
    )
    recorder.append("run_start", {"case_id": "case"})
    recorder.append("assistant_message", {"assistant_turn": 1})
    recorder.append(
        "tool_call",
        {"call_id": "c1", "name": "get_case_spec", "arguments": {}},
    )
    recorder.append(
        "tool_result",
        {"call_id": "c1", "name": "get_case_spec", "ok": True, "result": {}},
    )
    recorder.append("final_submitted", {"profile_id": "profile-1"})
    recorder.append("run_complete", {"status": "COMPLETE"})
    return recorder


def test_canonical_json_is_sorted_compact_and_rejects_nonfinite() -> None:
    assert canonical_json({"z": 1.0, "é": [True, None]}) == '{"z":1,"é":[true,null]}'
    with pytest.raises(TrajectoryError, match="non-finite"):
        canonical_json({"bad": float("nan")})
    with pytest.raises(TrajectoryError, match="keys must be strings"):
        canonical_json({1: "ambiguous"})


def test_recorder_is_append_only_exclusive_and_round_trips(tmp_path: Path) -> None:
    recorder = _complete_recorder()
    validate_hash_chain(recorder.events)
    commitment = recorder.commitment
    assert commitment == trajectory_commitment(recorder.events)
    destination = tmp_path / "trajectory.jsonl"
    recorder.write(destination)
    loaded = load_jsonl(
        destination,
        expected_head_hash=commitment["head_hash"],
        expected_event_count=commitment["event_count"],
    )
    assert loaded == recorder.events
    with pytest.raises(FileExistsError):
        recorder.write(destination)
    with pytest.raises(TrajectoryError, match="frozen"):
        recorder.append("budget_state", {})


@pytest.mark.parametrize("mutation", ["delete", "insert", "reorder", "tamper"])
def test_chain_detects_deletion_insertion_reordering_and_tampering(mutation: str) -> None:
    events = _complete_recorder().events
    changed = copy.deepcopy(events)
    if mutation == "delete":
        del changed[2]
    elif mutation == "insert":
        changed.insert(2, copy.deepcopy(changed[2]))
    elif mutation == "reorder":
        changed[1], changed[2] = changed[2], changed[1]
    else:
        changed[2]["payload"]["arguments"] = {"oracle": True}
    assert not verify_hash_chain(changed)
    with pytest.raises(TrajectoryError):
        validate_hash_chain(changed)


def test_external_head_and_count_detect_fully_rehashed_deletion() -> None:
    events = _complete_recorder().events
    commitment = trajectory_commitment(events)
    changed = copy.deepcopy(events)
    del changed[1]
    previous = GENESIS_HASH
    for sequence, event in enumerate(changed, start=1):
        event["sequence"] = sequence
        event["prev_event_hash"] = previous
        event["event_hash"] = canonical_sha256(
            {key: value for key, value in event.items() if key != "event_hash"}
        )
        previous = event["event_hash"]
    # The locally re-hashed lifecycle is internally consistent, but no longer
    # matches the independently persisted run-manifest commitment.
    validate_hash_chain(changed)
    with pytest.raises(TrajectoryError, match="count does not match"):
        validate_hash_chain(
            changed,
            expected_head_hash=commitment["head_hash"],
            expected_event_count=commitment["event_count"],
        )


def test_loader_rejects_noncanonical_and_duplicate_key_jsonl(tmp_path: Path) -> None:
    destination = tmp_path / "bad.jsonl"
    destination.write_text('{"a":1, "a":2}\n', encoding="utf-8")
    with pytest.raises(TrajectoryError, match="invalid JSON"):
        load_jsonl(destination)


@pytest.mark.parametrize(
    "value,expected",
    [(0, 0), (5.0, 5), (355, 355)],
)
def test_angle_normalization_accepts_only_grid(value: object, expected: int) -> None:
    assert normalize_scan_angle(value) == expected


@pytest.mark.parametrize("value", [-5, 1, 360, 5.5, float("nan"), True, "5"])
def test_angle_normalization_rejects_invalid_angles(value: object) -> None:
    with pytest.raises(BudgetError):
        normalize_scan_angle(value)


def test_invalid_angle_does_not_consume_unique_but_accepted_failure_does() -> None:
    ledger = BudgetLedger()
    ledger.consume_tool_call()
    rejected = ledger.reserve_scan(7)
    assert rejected.accepted is False
    assert ledger.snapshot().unique_scan_angle_count == 0

    ledger.consume_tool_call()
    accepted = ledger.reserve_scan(0)
    assert accepted.accepted is True and accepted.new_unique_angle is True
    ledger.record_scan_result(0, valid=False)
    snapshot = ledger.snapshot()
    assert snapshot.unique_scan_angles == (0,)
    assert snapshot.valid_unique_scan_angles == ()
    assert snapshot.tool_calls == 2


def test_per_angle_attempt_limit_and_first_turn_six_angle_limit() -> None:
    ledger = BudgetLedger()
    for angle in range(0, 30, 5):
        assert ledger.reserve_scan(angle, assistant_turn=1).accepted
    seventh = ledger.reserve_scan(30, assistant_turn=1)
    assert not seventh.accepted
    assert "at most six" in str(seventh.reason)

    retry_ledger = BudgetLedger()
    assert retry_ledger.reserve_scan(0).attempt_number == 1
    assert retry_ledger.reserve_scan(0).attempt_number == 2
    third = retry_ledger.reserve_scan(0)
    assert not third.accepted
    assert "attempt budget" in str(third.reason)


def test_warning_at_fourteen_and_finalization_only_at_sixteen() -> None:
    ledger = BudgetLedger()
    for index in range(14):
        reservation = ledger.reserve_scan(index * 5)
        assert reservation.accepted
    snapshot = ledger.snapshot()
    assert snapshot.warning is True
    assert "14/16" in str(snapshot.warning_message)
    assert snapshot.finalization_only is False

    assert ledger.reserve_scan(70).accepted
    sixteenth = ledger.reserve_scan(75)
    assert sixteenth.accepted and sixteenth.finalization_only
    blocked_retry = ledger.reserve_scan(0)
    assert not blocked_retry.accepted
    assert blocked_retry.finalization_only
    assert "finalization_only" in str(blocked_retry.reason)


def test_tool_turn_and_profile_fit_limits_are_fail_closed() -> None:
    ledger = BudgetLedger()
    for _ in range(ledger.limits.max_tool_calls):
        ledger.consume_tool_call()
    with pytest.raises(BudgetExceededError, match="tool-call"):
        ledger.consume_tool_call()

    for _ in range(ledger.limits.max_agent_turns):
        ledger.begin_agent_turn()
    with pytest.raises(BudgetExceededError, match="agent-turn"):
        ledger.begin_agent_turn()

    fits = [ledger.reserve_profile_fit() for _ in range(4)]
    ledger.record_profile_fit_result(fits[0], valid=True)
    ledger.record_profile_fit_result(fits[1], valid=True)
    with pytest.raises(BudgetExceededError, match="profile-fit"):
        ledger.reserve_profile_fit()
    assert ledger.snapshot().valid_profile_fits == 2
    with pytest.raises(BudgetError, match="immutable"):
        ledger.record_profile_fit_result(fits[0], valid=False)


def _append_tool_pair(
    recorder: TrajectoryRecorder,
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object],
    ok: bool,
    result: dict[str, object],
) -> None:
    recorder.append(
        "tool_call",
        {"call_id": call_id, "name": name, "arguments": arguments},
    )
    recorder.append(
        "tool_result",
        {"call_id": call_id, "name": name, "ok": ok, "result": result},
    )


def test_recomputation_requires_matching_fit_analysis_to_close_feedback_round() -> None:
    recorder = TrajectoryRecorder(
        run_id="feedback-audit",
        timestamp_fn=lambda: FIXED_TIME,
    )
    recorder.append("run_start", {})
    recorder.append("assistant_message", {"assistant_turn": 1})
    _append_tool_pair(
        recorder,
        call_id="s0",
        name="run_scan_point",
        arguments={"angle_deg": 0},
        ok=False,
        result={
            "accepted": True,
            "angle_deg": 0,
            "status": "FAILED",
            "profile_eligible": False,
        },
    )
    _append_tool_pair(
        recorder,
        call_id="invalid",
        name="run_scan_point",
        arguments={"angle_deg": 7},
        ok=False,
        result={"accepted": False, "status": "INVALID_ANGLE"},
    )

    recorder.append("assistant_message", {"assistant_turn": 2})
    _append_tool_pair(
        recorder,
        call_id="s90",
        name="run_scan_point",
        arguments={"angle_deg": 90},
        ok=True,
        result={"accepted": True, "angle_deg": 90, "profile_eligible": True},
    )
    for index, method in enumerate(("periodic_cubic", "fourier"), start=1):
        _append_tool_pair(
            recorder,
            call_id=f"fit{index}",
            name="fit_periodic_profile",
            arguments={"method": method},
            ok=True,
            result={"accepted": True, "valid": True, "profile_id": f"p{index}"},
        )

    recorder.append("assistant_message", {"assistant_turn": 3})
    _append_tool_pair(
        recorder,
        call_id="s90retry",
        name="run_scan_point",
        arguments={"angle_deg": 90},
        ok=True,
        result={"accepted": True, "angle_deg": 90, "profile_eligible": True},
    )
    _append_tool_pair(
        recorder,
        call_id="s180",
        name="run_scan_point",
        arguments={"angle_deg": 180},
        ok=True,
        result={"accepted": True, "angle_deg": 180, "profile_eligible": True},
    )

    recorder.append("assistant_message", {"assistant_turn": 4})
    _append_tool_pair(
        recorder,
        call_id="s270",
        name="run_scan_point",
        arguments={"angle_deg": 270},
        ok=True,
        result={"accepted": True, "angle_deg": 270, "profile_eligible": True},
    )
    recorder.append("final_submitted", {"profile_id": "p2"})
    recorder.append("run_complete", {"status": "COMPLETE"})

    audit = recompute_budget(recorder.events)
    assert audit.unique_scan_angles == (0, 90, 180, 270)
    assert audit.valid_unique_scan_angles == (90, 180, 270)
    assert dict(audit.scan_attempts_by_angle)[90] == 2
    assert audit.profile_fits == audit.valid_profile_fits == 2
    assert audit.feedback_round_turns == ()
    assert reconstruct_feedback_rounds(recorder.events) == 0
    assert any(
        "multiple_fits_before_phase_analysis" in item
        for item in audit.violations
    )
    assert audit.third_feedback_round_reachable is False


def test_recomputation_ignores_claimed_budget_state_and_detects_unpaired_calls() -> None:
    recorder = TrajectoryRecorder(run_id="lying-budget", timestamp_fn=lambda: FIXED_TIME)
    recorder.append("run_start", {})
    recorder.append("assistant_message", {"assistant_turn": 1})
    recorder.append("budget_state", {"tool_calls": 0, "feedback_rounds": 99})
    recorder.append(
        "tool_call",
        {"call_id": "missing", "name": "run_scan_point", "arguments": {"angle_deg": 0}},
    )
    recorder.append("run_failure", {"status": "FAILED"})
    audit = recompute_budget(recorder.events)
    assert audit.tool_calls == 1
    assert audit.feedback_rounds == 0
    assert "tool_call:missing:missing_result" in audit.violations


def test_recomputation_matches_runner_assistant_turn_and_nested_result_shape() -> None:
    recorder = TrajectoryRecorder(run_id="runner-shape", timestamp_fn=lambda: FIXED_TIME)
    recorder.append("run_start", {})
    recorder.append("assistant_turn", {"turn": 1})
    _append_tool_pair(
        recorder,
        call_id="scan-a",
        name="run_scan_point",
        arguments={"angle_deg": 0},
        ok=True,
        result={
            "ok": True,
            "status": "SCAN_COMPLETE",
            "scan": {"angle_deg": 0, "profile_eligible": True},
        },
    )
    recorder.append("assistant_turn", {"turn": 2})
    _append_tool_pair(
        recorder,
        call_id="scan-b",
        name="run_scan_point",
        arguments={"angle_deg": 90},
        ok=False,
        result={
            "ok": False,
            "status": "SCAN_FAILED",
            "scan": {"angle_deg": 90, "profile_eligible": False},
        },
    )
    _append_tool_pair(
        recorder,
        call_id="fit-rejected",
        name="fit_periodic_profile",
        arguments={"method": "periodic_cubic"},
        ok=False,
        result={"ok": False, "status": "INSUFFICIENT_VALID_POINTS"},
    )
    _append_tool_pair(
        recorder,
        call_id="fit-ok",
        name="fit_periodic_profile",
        arguments={"method": "periodic_cubic"},
        ok=True,
        result={
            "ok": True,
            "status": "PROFILE_FIT_COMPLETE",
            "profile": {"profile_id": "p1"},
        },
    )
    recorder.append("run_failure", {"status": "TEST_END"})
    audit = recompute_budget(recorder.events)
    assert audit.agent_turns == 2
    assert audit.unique_scan_angles == (0, 90)
    assert audit.valid_unique_scan_angles == (0,)
    assert audit.feedback_round_turns == ()
    assert audit.profile_fits == audit.valid_profile_fits == 1
    assert audit.phase_fit_pending_analysis is True
    assert audit.profile_fit_rounds == 1
    assert audit.violations == ()


def test_budget_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(BudgetError, match="unknown"):
        BudgetLimits.from_mapping({"max_tool_calls": 30, "invented": 1})

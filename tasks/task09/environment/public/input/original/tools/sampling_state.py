"""Independent reconstruction of the v2 batch-sampling protocol.

The runtime and evaluator both call this module, but neither trusts mutable
counters.  State is derived only from the ordered, append-only batch ledger.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


MAX_UNIQUE_ANGLES = 12
MIN_VALID_UNIQUE_ANGLES = 10
MIN_FEEDBACK_ROUNDS = 2
MIN_PROFILE_FITS = 3
MAX_BATCHES = 4
MAX_ATTEMPTS_PER_ANGLE = 2
PHASE0_SIZE = 6
PHASE_REFINEMENT_MIN = 2
PHASE_REFINEMENT_MAX = 3
RECOVERY_MIN = 1
RECOVERY_MAX = 2

# Compatibility names retained for code that imports the v1 constants.  Their
# values now reflect the frozen v2 protocol.
INITIAL_PHASE_LIMIT = PHASE0_SIZE
REFINEMENT_PHASE_LIMIT = PHASE_REFINEMENT_MAX
MAX_PROFILE_FITS = MAX_BATCHES


class SamplingStateError(ValueError):
    """The immutable batch ledger violates the frozen v2 state machine."""


def canonical_angle(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplingStateError("angles must be numeric integers")
    number = float(value)
    if not number.is_integer():
        raise SamplingStateError("angles must be integers")
    angle = int(number)
    if angle < 0 or angle >= 360 or angle % 5:
        raise SamplingStateError("angles must be 5 degree multiples in [0, 360)")
    return angle


def validate_batch_angles(values: Any) -> list[int]:
    if not isinstance(values, list):
        raise SamplingStateError("angles_deg must be a JSON array")
    angles = [canonical_angle(value) for value in values]
    if len(set(angles)) != len(angles):
        raise SamplingStateError("angles_deg contains a duplicate")
    return angles


def _phase_for_index(index: int) -> str:
    return ("phase0", "phase1", "phase2", "recovery")[index]


def expected_batch_size(batch_index: int) -> tuple[int, int]:
    if batch_index == 0:
        return PHASE0_SIZE, PHASE0_SIZE
    if batch_index in (1, 2):
        return PHASE_REFINEMENT_MIN, PHASE_REFINEMENT_MAX
    if batch_index == 3:
        return RECOVERY_MIN, RECOVERY_MAX
    return 0, 0


def reconstruct_batch_state(
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute process facts from accepted batch records.

    A record is accepted only after its native attempts and automatic profile
    fit have completed.  Failed angles therefore reserve unique budget just as
    successful angles do.
    """

    unique_angles: list[int] = []
    unique_set: set[int] = set()
    valid_angles: list[int] = []
    valid_set: set[int] = set()
    failed_angles: list[int] = []
    profile_ids: list[str] = []
    feedback_rounds = 0
    violations: list[str] = []
    terminal_requested = False
    terminal_accepted = False

    for index, raw in enumerate(ledger):
        phase = _phase_for_index(index) if index < MAX_BATCHES else "extra"
        if index >= MAX_BATCHES:
            violations.append("batch_budget_exceeded")
        if not isinstance(raw, Mapping):
            violations.append(f"batch_{index}:not_an_object")
            continue
        if raw.get("phase") != phase:
            violations.append(f"batch_{index}:phase_mismatch")
        try:
            angles = validate_batch_angles(raw.get("angles_deg"))
        except SamplingStateError:
            violations.append(f"batch_{index}:invalid_angles")
            angles = []
        low, high = expected_batch_size(index)
        if not low <= len(angles) <= high:
            violations.append(f"batch_{index}:invalid_batch_size")
        for angle in angles:
            if angle in unique_set:
                violations.append(f"batch_{index}:duplicate_prior_angle_{angle}")
            else:
                unique_set.add(angle)
                unique_angles.append(angle)
        raw_valid = raw.get("valid_angles_deg", [])
        if not isinstance(raw_valid, list):
            violations.append(f"batch_{index}:valid_angles_not_array")
            raw_valid = []
        for value in raw_valid:
            try:
                angle = canonical_angle(value)
            except SamplingStateError:
                violations.append(f"batch_{index}:invalid_valid_angle")
                continue
            if angle not in angles:
                violations.append(f"batch_{index}:valid_angle_not_in_batch_{angle}")
            elif angle not in valid_set:
                valid_set.add(angle)
                valid_angles.append(angle)
        raw_failed = raw.get("failed_angles_deg", [])
        if not isinstance(raw_failed, list):
            violations.append(f"batch_{index}:failed_angles_not_array")
            raw_failed = []
        for value in raw_failed:
            try:
                angle = canonical_angle(value)
            except SamplingStateError:
                violations.append(f"batch_{index}:invalid_failed_angle")
                continue
            if angle not in angles or angle in valid_set:
                violations.append(f"batch_{index}:invalid_failed_membership_{angle}")
            elif angle not in failed_angles:
                failed_angles.append(angle)
        if set(angles) != set(raw_valid) | set(raw_failed):
            violations.append(f"batch_{index}:angle_outcomes_incomplete")
        profile_id = raw.get("profile_id")
        profile_hash = raw.get("profile_hash")
        if not isinstance(profile_id, str) or not profile_id:
            violations.append(f"batch_{index}:profile_id_missing")
        else:
            profile_ids.append(profile_id)
        if not isinstance(profile_hash, str) or len(profile_hash) != 64:
            violations.append(f"batch_{index}:profile_hash_invalid")
        finalize = raw.get("finalize")
        if not isinstance(finalize, bool):
            violations.append(f"batch_{index}:finalize_not_boolean")
            finalize = False
        if index < 2 and finalize:
            violations.append(f"batch_{index}:early_finalize")
        if index == 2 and not finalize:
            violations.append("phase2_requires_finalize")
        if index == 3 and not finalize:
            violations.append("recovery_requires_finalize")
        if finalize:
            terminal_requested = True
            if index >= 2:
                terminal_accepted = bool(raw.get("terminal_accepted"))
        if index in (1, 2):
            feedback_rounds += 1

    if len(unique_angles) > MAX_UNIQUE_ANGLES:
        violations.append("unique_angle_budget_exceeded")
    batch_count = len(ledger)
    if batch_count == 0:
        next_phase = "phase0"
    elif batch_count == 1:
        next_phase = "phase1"
    elif batch_count == 2:
        next_phase = "phase2"
    elif batch_count == 3 and len(valid_angles) < MIN_VALID_UNIQUE_ANGLES:
        next_phase = "recovery"
    else:
        next_phase = "terminal"

    recovery_allowed = (
        batch_count == 3
        and len(valid_angles) < MIN_VALID_UNIQUE_ANGLES
        and len(unique_angles) < MAX_UNIQUE_ANGLES
        and bool(failed_angles)
    )
    if batch_count > 3 and not bool(failed_angles):
        violations.append("recovery_without_exhausted_native_failure")

    profile_fit_rounds = len(profile_ids)
    eligibility = {
        "within_unique_angle_budget": len(unique_angles) <= MAX_UNIQUE_ANGLES,
        "minimum_valid_unique_angles": len(valid_angles) >= MIN_VALID_UNIQUE_ANGLES,
        "minimum_feedback_rounds": feedback_rounds >= MIN_FEEDBACK_ROUNDS,
        "minimum_profile_fits": profile_fit_rounds >= MIN_PROFILE_FITS,
        "terminal_profile_present": profile_fit_rounds >= MIN_PROFILE_FITS,
        "protocol_violations_absent": not violations,
    }
    eligible = all(eligibility.values())
    return {
        "schema_version": "2.0",
        "batch_count": batch_count,
        "current_phase": next_phase,
        "unique_angles": unique_angles,
        "unique_angle_count": len(unique_angles),
        "unique_angles_remaining": max(0, MAX_UNIQUE_ANGLES - len(unique_angles)),
        "valid_angles": valid_angles,
        "valid_unique_angle_count": len(valid_angles),
        "failed_angles": failed_angles,
        "feedback_rounds": feedback_rounds,
        "profile_fit_rounds": profile_fit_rounds,
        "profile_ids": profile_ids,
        "recovery_allowed": recovery_allowed,
        "terminal_requested": terminal_requested,
        "terminal_accepted": terminal_accepted,
        "eligible": eligible,
        "eligibility": eligibility,
        "eligibility_gaps": {
            "valid_angles_needed": max(
                0, MIN_VALID_UNIQUE_ANGLES - len(valid_angles)
            ),
            "feedback_rounds_needed": max(
                0, MIN_FEEDBACK_ROUNDS - feedback_rounds
            ),
            "profile_fits_needed": max(
                0, MIN_PROFILE_FITS - profile_fit_rounds
            ),
        },
        "violations": list(dict.fromkeys(violations)),
        "ledger": deepcopy(list(ledger)),
    }


def accepted_scan_fact(operation: Mapping[str, Any]) -> tuple[bool, int | None, bool]:
    """Compatibility helper for independent audits of historical v1 events."""

    result = operation.get("result")
    arguments = operation.get("arguments")
    if not isinstance(result, Mapping):
        return False, None, False
    if not isinstance(arguments, Mapping):
        arguments = {}
    value = result.get("angle_deg", arguments.get("angle_deg"))
    try:
        angle = canonical_angle(value)
    except SamplingStateError:
        angle = None
    accepted = result.get("accepted") is True or isinstance(
        result.get("attempt_id"), str
    )
    eligible = result.get("profile_eligible") is True
    return accepted, angle, eligible


def reconstruct_live_sampling_state(
    operations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility projection for historical event auditors.

    New v2 code must use :func:`reconstruct_batch_state`.
    """

    ledger = [
        operation["result"]["batch_record"]
        for operation in operations
        if operation.get("tool") == "run_scan_batch"
        and isinstance(operation.get("result"), Mapping)
        and isinstance(operation["result"].get("batch_record"), Mapping)
    ]
    state = reconstruct_batch_state(ledger)
    return {
        **state,
        "cumulative_unique_angles": state["unique_angle_count"],
        "valid_unique_angles": state["valid_angles"],
        "scan_feedback_rounds": state["feedback_rounds"],
        "current_sampling_phase": state["current_phase"],
        "finalization_only": state["current_phase"] == "terminal",
    }


__all__ = [
    "INITIAL_PHASE_LIMIT",
    "MAX_ATTEMPTS_PER_ANGLE",
    "MAX_BATCHES",
    "MAX_PROFILE_FITS",
    "MAX_UNIQUE_ANGLES",
    "MIN_FEEDBACK_ROUNDS",
    "MIN_PROFILE_FITS",
    "MIN_VALID_UNIQUE_ANGLES",
    "PHASE0_SIZE",
    "RECOVERY_MAX",
    "RECOVERY_MIN",
    "REFINEMENT_PHASE_LIMIT",
    "SamplingStateError",
    "accepted_scan_fact",
    "canonical_angle",
    "expected_batch_size",
    "reconstruct_batch_state",
    "reconstruct_live_sampling_state",
    "validate_batch_angles",
]

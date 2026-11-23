"""Bounded canonical provider context for protocol 2.0."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tools.science import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_STATE_MAX_CHARS = 7000
ANALYSIS_SUMMARY_MAX_CHARS = 2500
TOOL_RESULT_MAX_CHARS = 6000
FAILED_ATTEMPT_INSPECTION_MAX_CHARS = 1800
_STRUCTURE_KEYS = (
    "atom_count_ok",
    "atom_order_ok",
    "composition_ok",
    "formal_charge_ok",
    "connectivity_ok",
)


class CanonicalStateError(ValueError):
    """A provider projection exceeds or violates its frozen closed schema."""


def _chars(value: Any) -> int:
    return len(canonical_json_bytes(value).decode("utf-8"))


def _case_projection() -> dict[str, Any]:
    case = json.loads(
        (ROOT / "benchmark_case" / "case_spec.json").read_text(encoding="utf-8")
    )
    return {
        "case_id": case["case_id"],
        "molecule": case["molecule"],
        "angle_grid_deg": case["angle_grid_deg"],
        "scientific_invariants": case["scientific_invariants"],
        "budgets": {
            key: case["budgets"][key]
            for key in (
                "max_unique_scan_angles",
                "min_unique_valid_scan_angles",
                "phase_0_angles",
                "phase_1_min_angles",
                "phase_1_max_angles",
                "phase_2_min_angles",
                "phase_2_max_angles",
                "recovery_min_angles",
                "recovery_max_angles",
            )
        },
    }


def _failed_attempt_projection(
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    if result.get("status") != "FAILED_ATTEMPT_INSPECTION":
        return None
    angle = result.get("requested_angle_deg")
    attempt_count = result.get("attempt_count")
    attempts = result.get("attempts")
    if (
        isinstance(angle, bool)
        or not isinstance(angle, int)
        or angle < 0
        or angle >= 360
        or angle % 5
        or isinstance(attempt_count, bool)
        or attempt_count != 2
        or not isinstance(attempts, list)
        or len(attempts) != 2
    ):
        raise CanonicalStateError("failed-attempt inspection is invalid")
    projected_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise CanonicalStateError("failed-attempt inspection is invalid")
        return_code = attempt.get("return_code")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            return_code = None
        angle_error = attempt.get("angle_error_deg")
        if (
            isinstance(angle_error, bool)
            or not isinstance(angle_error, (int, float))
            or not math.isfinite(float(angle_error))
        ):
            angle_error = None
        structure = attempt.get("structure_validation")
        if not isinstance(structure, Mapping):
            structure = {}
        projected_attempts.append(
            {
                "return_code": return_code,
                "timed_out": attempt.get("timed_out") is True,
                "converged": attempt.get("converged") is True,
                "energy_parsed": attempt.get("energy_parsed") is True,
                "angle_error_deg": (
                    float(angle_error) if angle_error is not None else None
                ),
                "structure_validation": {
                    key: structure.get(key) is True for key in _STRUCTURE_KEYS
                },
            }
        )
    projection = {
        "requested_angle_deg": angle,
        "attempt_count": attempt_count,
        "attempts": projected_attempts,
        "replacement_sampling_allowed": (
            result.get("replacement_sampling_allowed") is True
        ),
    }
    if _chars(projection) > FAILED_ATTEMPT_INSPECTION_MAX_CHARS:
        raise CanonicalStateError(
            "failed-attempt inspection projection exceeds 1800 characters"
        )
    return projection


def action_schema_for_phase(phase: str) -> dict[str, Any]:
    if phase == "phase0":
        count = "exactly 6"
        finalize = False
    elif phase in {"phase1", "phase2"}:
        count = "2 or 3"
        finalize = phase == "phase2"
    elif phase == "recovery":
        count = "1 or 2"
        finalize = True
    else:
        count = "none"
        finalize = None
    return {
        "allowed_actions": ["run_scan_batch", "inspect_failed_attempt"],
        "next_action": {
            "action": "run_scan_batch",
            "angles_deg": f"{count} new unique integer 5-degree grid angles",
            "finalize": finalize,
            "confidence": (
                "low|medium|high" if finalize else None
            ),
            "rationale": "non-empty string, maximum 400 characters",
            "additional_properties": False,
        },
        "failed_attempt_inspection": {
            "action": "inspect_failed_attempt",
            "angle_deg": "only an angle whose two automatic attempts failed",
            "additional_properties": False,
        },
        "response_contract": "Return one JSON object only; no prose or tool-call wrapper.",
    }


def build_canonical_state(
    *,
    runtime_state: Mapping[str, Any],
    latest_tool_result: Mapping[str, Any] | None,
    retained_profile_analysis: Mapping[str, Any] | None = None,
    protocol_rerequest: bool = False,
) -> dict[str, Any]:
    """Build a fresh request without prior messages, raw provenance, or profile."""

    phase = str(runtime_state.get("current_phase", "phase0"))
    latest_analysis: dict[str, Any] | None = (
        deepcopy(dict(retained_profile_analysis))
        if retained_profile_analysis is not None
        else None
    )
    if latest_analysis is not None and _chars(
        latest_analysis
    ) > ANALYSIS_SUMMARY_MAX_CHARS:
        raise CanonicalStateError("analysis summary exceeds 2500 characters")
    last_status: str | None = None
    latest_failed_attempt: dict[str, Any] | None = None
    if latest_tool_result is not None:
        if _chars(latest_tool_result) > TOOL_RESULT_MAX_CHARS:
            raise CanonicalStateError("tool result exceeds 6000 characters")
        last_status = str(latest_tool_result.get("status"))
        latest_failed_attempt = _failed_attempt_projection(latest_tool_result)
        profile = latest_tool_result.get("profile")
        if isinstance(profile, Mapping) and isinstance(profile.get("analysis"), Mapping):
            latest_analysis = deepcopy(dict(profile["analysis"]))
            if _chars(latest_analysis) > ANALYSIS_SUMMARY_MAX_CHARS:
                raise CanonicalStateError("analysis summary exceeds 2500 characters")
    state = {
        "schema_version": "2.0",
        "context_mode": "canonical_state_v2",
        "case": _case_projection(),
        "phase": phase,
        "sampling": {
            "sampled_angles_deg": list(runtime_state.get("unique_angles", [])),
            "valid_angles_deg": list(runtime_state.get("valid_angles", [])),
            "failed_angles_deg": list(runtime_state.get("failed_angles", [])),
            "unique_angles_used": int(runtime_state.get("unique_angle_count", 0)),
            "unique_angles_remaining": int(
                runtime_state.get("unique_angles_remaining", 12)
            ),
            "valid_unique_angles": int(
                runtime_state.get("valid_unique_angle_count", 0)
            ),
            "feedback_rounds": int(runtime_state.get("feedback_rounds", 0)),
            "profile_fit_rounds": int(runtime_state.get("profile_fit_rounds", 0)),
            "eligibility_gaps": deepcopy(
                runtime_state.get("eligibility_gaps", {})
            ),
            "recovery_allowed": bool(runtime_state.get("recovery_allowed", False)),
        },
        "latest_result_status": last_status,
        "latest_failed_attempt_inspection": latest_failed_attempt,
        "latest_profile_analysis": latest_analysis,
        "protocol_rerequest": bool(protocol_rerequest),
        "action_schema": action_schema_for_phase(phase),
        "forbidden": [
            "oracle or evaluator access",
            "filesystem, shell, Python, or network access",
            "molecule, method, charge, UHF, target bond, or reference changes",
            "manual profile, energy, score, hash, or runtime-statistic submission",
        ],
    }
    if _chars(state) > CANONICAL_STATE_MAX_CHARS:
        raise CanonicalStateError("canonical state exceeds 7000 characters")
    return state


def canonical_state_text(state: Mapping[str, Any]) -> str:
    text = canonical_json_bytes(state).decode("utf-8")
    if len(text) > CANONICAL_STATE_MAX_CHARS:
        raise CanonicalStateError("canonical state exceeds 7000 characters")
    return text


__all__ = [
    "ANALYSIS_SUMMARY_MAX_CHARS",
    "CANONICAL_STATE_MAX_CHARS",
    "CanonicalStateError",
    "FAILED_ATTEMPT_INSPECTION_MAX_CHARS",
    "TOOL_RESULT_MAX_CHARS",
    "action_schema_for_phase",
    "build_canonical_state",
    "canonical_state_text",
]

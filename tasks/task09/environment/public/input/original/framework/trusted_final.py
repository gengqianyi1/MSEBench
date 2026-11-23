"""Build and validate the terminal, two-key v2 final envelope."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from framework.trajectory import load_jsonl
from framework.execution_freeze import validate_execution_freeze_binding
from tools.sampling_state import reconstruct_batch_state
from tools.science import canonical_json_bytes, sha256_file


TASK_ID = "adaptive_xtb_torsion_profile_agent_v2"
BENCHMARK_VERSION = "2.0.0"
CASE_ID = "2_fluorobiphenyl_central_torsion_v1"
PROTOCOL = "adaptive_torsion_sampling_tool_loop/2.0"
EVALUATOR_VERSION = "adaptive-torsion-evaluator/2.0"
CONFIDENCE = {"low", "medium", "high"}
HEX_SHA256 = frozenset("0123456789ABCDEFabcdef")


class TrustedFinalError(RuntimeError):
    """Trusted runtime binding cannot be proven."""


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX_SHA256 for character in value)
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _submission(session: Any) -> dict[str, Any]:
    value = session.final_submission
    if not isinstance(value, Mapping):
        raise TrustedFinalError("terminal action was not accepted")
    expected = {
        "case_id",
        "final_status",
        "confidence",
        "rationale",
        "finalize_action_hash",
    }
    if set(value) != expected:
        raise TrustedFinalError("model submission is not closed")
    if value["case_id"] != CASE_ID or value["final_status"] != "completed":
        raise TrustedFinalError("model submission identity or status changed")
    if value["confidence"] not in CONFIDENCE:
        raise TrustedFinalError("model confidence is invalid")
    if (
        not isinstance(value["rationale"], str)
        or not 1 <= len(value["rationale"]) <= 400
    ):
        raise TrustedFinalError("model rationale is invalid")
    digest = value["finalize_action_hash"]
    if not _is_sha256(digest):
        raise TrustedFinalError("finalize action hash is invalid")
    return dict(value)


def build_final_envelope(
    *,
    session: Any,
    requested_model_id: str,
    actual_model_id: str,
    trajectory_path: Path,
    trajectory_final_event_hash: str,
    timestamp_utc: str | None = None,
    campaign_snapshot: Mapping[str, Any] | None = None,
    execution_freeze_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind terminal model intent to runtime-selected scientific artifacts."""

    del campaign_snapshot  # v2 commitments replace the v1.1 campaign snapshot.
    submission = _submission(session)
    if requested_model_id != actual_model_id:
        raise TrustedFinalError("exact model identity is required")
    frozen_execution = (
        validate_execution_freeze_binding(execution_freeze_binding)
        if execution_freeze_binding is not None
        else None
    )
    if (
        actual_model_id != "deterministic-baseline/2.0"
        and frozen_execution is None
    ):
        raise TrustedFinalError("formal run lacks the Gate-8 execution binding")
    events = load_jsonl(trajectory_path)
    if events[-1]["event"] != "run_complete":
        raise TrustedFinalError("trajectory is not successfully terminal")
    if events[-1]["event_hash"] != trajectory_final_event_hash:
        raise TrustedFinalError("trajectory head binding changed")
    state = reconstruct_batch_state(session.ledger)
    if not state["eligible"] or not state["terminal_accepted"]:
        raise TrustedFinalError("independently reconstructed process is ineligible")
    profile_id = state["profile_ids"][-1]
    profile_path = session.profiles_root / f"{profile_id}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    predictions = profile.get("predicted_energies")
    if (
        not isinstance(predictions, list)
        or len(predictions) != 72
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in predictions
        )
    ):
        raise TrustedFinalError("terminal profile is not 72 finite points")
    if profile.get("profile_hash") != session.profile(profile_id)["profile_hash"]:
        raise TrustedFinalError("terminal profile binding changed")

    attempts: list[dict[str, Any]] = []
    for angle, records in sorted(session.attempt_summaries.items()):
        for record in records:
            attempts.append(
                {
                    "attempt_id": record["attempt_id"],
                    "angle_deg": angle,
                    "attempt_number": record["attempt_number"],
                    "profile_eligible": record["profile_eligible"],
                    "raw_attempt_manifest_sha256": record["provenance"][
                        "raw_attempt_manifest_sha256"
                    ],
                }
            )
    trajectory_sha = sha256_file(trajectory_path)
    profile_artifact_sha = sha256_file(profile_path)
    commitment_payload = {
        "schema_version": "2.0",
        "task_id": TASK_ID,
        "case_id": CASE_ID,
        "protocol": PROTOCOL,
        "requested_model_id": requested_model_id,
        "actual_model_id": actual_model_id,
        "trajectory_head_hash": trajectory_final_event_hash,
        "trajectory_sha256": trajectory_sha,
        "profile_id": profile_id,
        "profile_hash": profile["profile_hash"],
        "profile_artifact_sha256": profile_artifact_sha,
        "batch_ledger": session.ledger,
        "attempt_manifest_hashes": [
            item["raw_attempt_manifest_sha256"] for item in attempts
        ],
        "execution_freeze_binding": frozen_execution,
    }
    execution_commitment = hashlib.sha256(
        canonical_json_bytes(commitment_payload)
    ).hexdigest().upper()
    runtime = {
        "task_id": TASK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "case_id": CASE_ID,
        "protocol": PROTOCOL,
        "status": "PASS",
        "actual_model_identity": actual_model_id,
        "model_identity": {
            "requested": requested_model_id,
            "actual": actual_model_id,
            "exact_match": True,
            "fallback_used": False,
            "model_override_used": False,
        },
        "profile_id": profile_id,
        "profile_hash": profile["profile_hash"],
        "profile": {
            "profile_id": profile_id,
            "profile_hash": profile["profile_hash"],
            "artifact_sha256": profile_artifact_sha,
            "grid_angles_deg": profile["grid_angles_deg"],
            "predicted_energy_kcal_mol": predictions,
            "all_finite": True,
        },
        "trajectory_hash": trajectory_final_event_hash,
        "trajectory": {
            "head_hash": trajectory_final_event_hash,
            "file_sha256": trajectory_sha,
            "event_count": len(events),
        },
        "unique_angles": state["unique_angle_count"],
        "valid_unique_angles": state["valid_unique_angle_count"],
        "feedback_rounds": state["feedback_rounds"],
        "profile_fit_rounds": state["profile_fit_rounds"],
        "batch_count": state["batch_count"],
        "process_recomputed": {
            key: value for key, value in state.items() if key != "ledger"
        },
        "attempt_validity": {
            "attempt_count": len(attempts),
            "eligible_attempt_count": sum(
                bool(item["profile_eligible"]) for item in attempts
            ),
            "attempts": attempts,
        },
        "scientific_invariants": {
            "method": "GFN2-xTB",
            "xtb_version": "6.7.1",
            "phase": "gas",
            "charge": 0,
            "uhf": 0,
            "threads": 1,
            "angle_grid_deg": 5,
            "reference_seed": 20260720,
            "reference_geometry_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "benchmark_case"
                / "reference.xyz"
            ),
        },
        "execution_commitment": execution_commitment,
        "execution_commitment_payload": commitment_payload,
        "execution_freeze_binding": frozen_execution,
        "evaluator_version": EVALUATOR_VERSION,
        "timestamp": timestamp_utc or _utc_now(),
    }
    envelope = {"model_submission": submission, "trusted_runtime": runtime}
    validate_final_envelope(envelope)
    return envelope


def validate_final_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "model_submission",
        "trusted_runtime",
    }:
        raise TrustedFinalError("final envelope must contain exactly two keys")
    submission = value["model_submission"]
    runtime = value["trusted_runtime"]
    if not isinstance(submission, Mapping) or not isinstance(runtime, Mapping):
        raise TrustedFinalError("final envelope values must be objects")
    expected_submission = {
        "case_id",
        "final_status",
        "confidence",
        "rationale",
        "finalize_action_hash",
    }
    if set(submission) != expected_submission:
        raise TrustedFinalError("model submission is not closed")
    if (
        submission.get("case_id") != CASE_ID
        or submission.get("final_status") != "completed"
        or submission.get("confidence") not in CONFIDENCE
        or not isinstance(submission.get("rationale"), str)
        or not 1 <= len(submission["rationale"]) <= 400
        or not _is_sha256(submission.get("finalize_action_hash"))
    ):
        raise TrustedFinalError("model submission is invalid")
    if (
        runtime.get("task_id") != TASK_ID
        or runtime.get("benchmark_version") != BENCHMARK_VERSION
        or runtime.get("case_id") != CASE_ID
        or runtime.get("protocol") != PROTOCOL
        or runtime.get("status") != "PASS"
        or runtime.get("evaluator_version") != EVALUATOR_VERSION
    ):
        raise TrustedFinalError("trusted runtime identity changed")
    identity = runtime.get("model_identity")
    if (
        not isinstance(identity, Mapping)
        or set(identity)
        != {
            "requested",
            "actual",
            "exact_match",
            "fallback_used",
            "model_override_used",
        }
        or not isinstance(identity.get("requested"), str)
        or not identity["requested"]
        or identity.get("actual") != identity.get("requested")
        or runtime.get("actual_model_identity") != identity.get("actual")
        or identity.get("exact_match") is not True
    ):
        raise TrustedFinalError("exact identity is not bound")
    if (
        identity.get("fallback_used") is not False
        or identity.get("model_override_used") is not False
    ):
        raise TrustedFinalError("fallback or model override is forbidden")
    freeze_binding = runtime.get("execution_freeze_binding")
    if identity.get("actual") != "deterministic-baseline/2.0":
        try:
            validated_binding = validate_execution_freeze_binding(
                freeze_binding
            )
        except Exception as exc:
            raise TrustedFinalError(
                "formal execution freeze binding is invalid"
            ) from exc
        payload = runtime.get("execution_commitment_payload")
        if (
            not isinstance(payload, Mapping)
            or payload.get("execution_freeze_binding") != validated_binding
        ):
            raise TrustedFinalError(
                "formal execution freeze binding is not commitment-bound"
            )
    profile = runtime.get("profile")
    predictions = profile.get("predicted_energy_kcal_mol") if isinstance(profile, Mapping) else None
    if (
        not isinstance(profile, Mapping)
        or profile.get("profile_id") != runtime.get("profile_id")
        or profile.get("profile_hash") != runtime.get("profile_hash")
        or not _is_sha256(profile.get("profile_hash"))
        or not _is_sha256(profile.get("artifact_sha256"))
        or profile.get("grid_angles_deg") != list(range(0, 360, 5))
        or profile.get("all_finite") is not True
        or not isinstance(predictions, list)
        or len(predictions) != 72
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in predictions
        )
    ):
        raise TrustedFinalError("trusted 72-point profile is invalid")
    for field in ("execution_commitment", "profile_hash", "trajectory_hash"):
        digest = runtime.get(field)
        if not _is_sha256(digest):
            raise TrustedFinalError(f"trusted {field} is invalid")
    trajectory = runtime.get("trajectory")
    if (
        not isinstance(trajectory, Mapping)
        or trajectory.get("head_hash") != runtime.get("trajectory_hash")
        or not _is_sha256(trajectory.get("file_sha256"))
        or isinstance(trajectory.get("event_count"), bool)
        or not isinstance(trajectory.get("event_count"), int)
        or trajectory["event_count"] < 1
    ):
        raise TrustedFinalError("trusted trajectory binding is invalid")
    payload = runtime.get("execution_commitment_payload")
    if (
        not isinstance(payload, Mapping)
        or runtime.get("execution_commitment")
        != hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
        or payload.get("task_id") != TASK_ID
        or payload.get("case_id") != CASE_ID
        or payload.get("protocol") != PROTOCOL
        or payload.get("requested_model_id") != identity.get("requested")
        or payload.get("actual_model_id") != identity.get("actual")
        or payload.get("trajectory_head_hash")
        != runtime.get("trajectory_hash")
        or payload.get("trajectory_sha256") != trajectory.get("file_sha256")
        or payload.get("profile_id") != runtime.get("profile_id")
        or payload.get("profile_hash") != runtime.get("profile_hash")
        or payload.get("profile_artifact_sha256")
        != profile.get("artifact_sha256")
    ):
        raise TrustedFinalError("trusted execution commitment is invalid")
    return {
        "model_submission": dict(submission),
        "trusted_runtime": dict(runtime),
    }


def write_final_envelope(path: Path, envelope: Mapping[str, Any]) -> None:
    validate_final_envelope(envelope)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(envelope))


__all__ = [
    "BENCHMARK_VERSION",
    "CASE_ID",
    "EVALUATOR_VERSION",
    "PROTOCOL",
    "TASK_ID",
    "TrustedFinalError",
    "build_final_envelope",
    "validate_final_envelope",
    "write_final_envelope",
]

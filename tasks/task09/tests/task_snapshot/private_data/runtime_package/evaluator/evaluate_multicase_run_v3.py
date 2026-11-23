"""Independent evaluator for one parameterized v3 torsion formal run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from evaluator.profile_recompute_v2 import recompute_automatic_profile
from evaluator.scoring_v2 import score_profile_v2


HARTREE_TO_KCAL_MOL = 627.5094740631
_EVENT_KEYS = {
    "schema_version", "protocol", "run_id", "sequence", "timestamp_utc",
    "event", "payload", "prev_event_hash", "event_hash",
}
_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL = {"run_complete", "run_failure"}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_event_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("TRAJECTORY_NONFINITE")
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return [_canonical_event_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical_event_value(item) for key, item in value.items()}
    raise ValueError("TRAJECTORY_NON_JSON_TYPE")


def _event_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_event_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("TRAJECTORY_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _load_and_validate_trajectory(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("TRAJECTORY_BOM")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("TRAJECTORY_EMPTY_LINE")
    events: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    previous_time: datetime | None = None
    run_id: str | None = None
    protocol: Mapping[str, Any] | None = None
    terminal = False
    final_submitted = False
    for sequence, line in enumerate(lines, start=1):
        value = json.loads(line, object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or set(value) != _EVENT_KEYS:
            raise ValueError("TRAJECTORY_ENVELOPE_INVALID")
        if line != _event_bytes(value).decode("utf-8"):
            raise ValueError("TRAJECTORY_NOT_CANONICAL")
        if value.get("schema_version") != "1.0" or value.get("sequence") != sequence:
            raise ValueError("TRAJECTORY_SEQUENCE_INVALID")
        name = value.get("event")
        if not isinstance(name, str) or not _EVENT_NAME.fullmatch(name):
            raise ValueError("TRAJECTORY_EVENT_INVALID")
        if not isinstance(value.get("payload"), Mapping):
            raise ValueError("TRAJECTORY_PAYLOAD_INVALID")
        current_run_id = value.get("run_id")
        current_protocol = value.get("protocol")
        if not isinstance(current_run_id, str) or not current_run_id:
            raise ValueError("TRAJECTORY_RUN_ID_INVALID")
        if not isinstance(current_protocol, Mapping) or set(current_protocol) != {"name", "version"}:
            raise ValueError("TRAJECTORY_PROTOCOL_INVALID")
        if run_id is None:
            run_id, protocol = current_run_id, dict(current_protocol)
        elif current_run_id != run_id or current_protocol != protocol:
            raise ValueError("TRAJECTORY_IDENTITY_CHANGED")
        timestamp_text = value.get("timestamp_utc")
        if not isinstance(timestamp_text, str) or not timestamp_text.endswith("Z"):
            raise ValueError("TRAJECTORY_TIMESTAMP_INVALID")
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("TRAJECTORY_TIMESTAMP_NOT_UTC")
        if previous_time is not None and timestamp < previous_time:
            raise ValueError("TRAJECTORY_TIMESTAMP_NOT_MONOTONIC")
        previous_time = timestamp
        if terminal:
            raise ValueError("TRAJECTORY_AFTER_TERMINAL")
        if sequence == 1 and name != "run_start":
            raise ValueError("TRAJECTORY_START_INVALID")
        if sequence > 1 and name == "run_start":
            raise ValueError("TRAJECTORY_DUPLICATE_START")
        if final_submitted and name != "run_complete":
            raise ValueError("TRAJECTORY_AFTER_FINAL")
        if name == "run_complete" and not final_submitted:
            raise ValueError("TRAJECTORY_COMPLETE_WITHOUT_FINAL")
        if name == "final_submitted":
            if final_submitted:
                raise ValueError("TRAJECTORY_DUPLICATE_FINAL")
            final_submitted = True
        if value.get("prev_event_hash") != previous_hash:
            raise ValueError("TRAJECTORY_PREVIOUS_HASH_INVALID")
        event_hash = value.get("event_hash")
        if not isinstance(event_hash, str) or not _HASH.fullmatch(event_hash):
            raise ValueError("TRAJECTORY_HASH_INVALID")
        without_hash = {key: item for key, item in value.items() if key != "event_hash"}
        if hashlib.sha256(_event_bytes(without_hash)).hexdigest() != event_hash:
            raise ValueError("TRAJECTORY_HASH_MISMATCH")
        previous_hash = event_hash
        terminal = name in _TERMINAL
        events.append(value)
    if not terminal:
        raise ValueError("TRAJECTORY_TERMINAL_MISSING")
    return events


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_NOT_OBJECT")
    return value


def _manifest_valid(attempt: Path) -> bool:
    manifest = _json(attempt / "attempt_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return False
    declared: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or relative in declared:
            return False
        target = attempt / relative
        if not target.is_file():
            return False
        if target.stat().st_size != entry.get("size_bytes"):
            return False
        if sha256_file(target) != str(entry.get("sha256", "")).upper():
            return False
        declared.add(relative)
    actual = {
        path.relative_to(attempt).as_posix()
        for path in attempt.rglob("*")
        if path.is_file() and path.name != "attempt_manifest.json"
    }
    return actual == declared


def _failed_evaluation(
    *, case_id: str, metadata: Mapping[str, Any], selected: list[int], reason: str
) -> dict[str, Any]:
    blocked = metadata.get("runner_status") == "BLOCKED"
    return {
        "schema_version": "3.0",
        "case_id": case_id,
        "status": "BLOCKED" if blocked else "FAIL",
        "provider_status": metadata.get("provider_status", "UNKNOWN"),
        "eligible": None if blocked else False,
        "failed_gates": [] if blocked else [reason],
        "scientific_score": None if blocked else 0.0,
        "process_score": None if blocked else 0.0,
        "official_score": None if blocked else 0.0,
        "selected_angles_deg": selected,
        "unique_valid_angles": 0,
        "feedback_rounds": 0,
        "metrics": {},
        "ranking_included": not blocked,
    }


def evaluate_case(
    case_root: Path,
    case: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    metadata = _json(case_root / "metadata.json")
    trajectory_path = case_root / "trajectory.jsonl"
    events = _load_and_validate_trajectory(trajectory_path)
    actions = [
        dict(event["payload"]["action"])
        for event in events
        if event.get("event") == "model_action"
        and isinstance(event.get("payload", {}).get("action"), Mapping)
    ]
    selected = [
        int(angle)
        for action in actions
        if action.get("action") == "run_scan_batch"
        for angle in action.get("angles_deg", [])
        if isinstance(angle, int) and not isinstance(angle, bool)
    ]
    if metadata.get("runner_status") != "COMPLETE":
        return _failed_evaluation(
            case_id=case_id,
            metadata=metadata,
            selected=selected,
            reason="runner_complete",
        )
    expected_model = str(metadata.get("requested_model_id", ""))
    exact_identity = bool(
        expected_model
        and metadata.get("actual_model_id") == expected_model
        and metadata.get("exact_model_identity") is True
    )
    # 2026-09-27 fix: accept the task statement's bounded recovery batch.
    # instruction.md and the sidecar adapter both allow a fourth, recovery-only
    # batch (1-2 new angles, finalize=true) when native failures leave fewer
    # than ten valid unique angles after phase two, but this gate previously
    # hard-required exactly three run_scan_batch actions, so any run that used
    # documented recovery was zeroed by complete_protocol/minimum_feedback_rounds.
    # The three-batch form below is unchanged; the four-batch form additionally
    # checks the recovery batch shape and, once native attempts are replayed,
    # the recovery trigger (see the energies refinement after _failed_evaluation).
    protocol_ok = bool(
        len(actions) in {3, 4}
        and all(action.get("action") == "run_scan_batch" for action in actions)
        and len(actions[0].get("angles_deg", [])) == 6
        and len(actions[1].get("angles_deg", [])) in {2, 3}
        and len(actions[2].get("angles_deg", [])) in {2, 3}
        and actions[0].get("finalize") is False
        and actions[1].get("finalize") is False
        and actions[2].get("finalize") is True
        and (
            len(actions) == 3
            or (
                len(actions[3].get("angles_deg", [])) in {1, 2}
                and actions[3].get("finalize") is True
            )
        )
        and len(selected) == len(set(selected))
        and 10 <= len(selected) <= 12
        and all(0 <= angle < 360 and angle % 5 == 0 for angle in selected)
    )
    native_root = case_root / "artifacts" / case_id / "native_attempts"
    energies: dict[int, float] = {}
    provenance_ok = True
    invariants_ok = True
    for angle in selected:
        attempt = native_root / f"angle_{angle:03d}"
        try:
            result = _json(attempt / "result.json")
            energy = result.get("total_energy_hartree")
            identity = result.get("xtb_identity")
            valid = bool(
                result.get("case_id") == case_id
                and result.get("angle_deg") == angle
                and result.get("profile_eligible") is True
                and isinstance(energy, (int, float))
                and not isinstance(energy, bool)
                and math.isfinite(float(energy))
                and isinstance(identity, Mapping)
                and identity.get("version") == "6.7.1"
                and result.get("native_argv", [None])[0] == "xtb"
                and _manifest_valid(attempt)
            )
        except (OSError, ValueError, TypeError, KeyError):
            valid = False
            energy = None
        provenance_ok = provenance_ok and valid
        invariants_ok = invariants_ok and valid
        if valid and energy is not None:
            energies[angle] = float(energy)
    valid_angles = [angle for angle in selected if angle in energies]
    if len(valid_angles) < 3:
        return _failed_evaluation(
            case_id=case_id,
            metadata=metadata,
            selected=selected,
            reason="native_points_available",
        )
    if len(actions) == 4:
        # 2026-09-27 fix (cont.): the recovery batch is legal only when the
        # first three batches left fewer than ten valid angles.  The count is
        # derived from the replayed evidence itself (angles with a valid native
        # attempt), mirroring the sidecar's _recovery_allowed() trigger, so a
        # four-batch run that never needed recovery is still rejected.
        pre_recovery = [
            int(angle)
            for action in actions[:3]
            for angle in action.get("angles_deg", [])
            if isinstance(angle, int) and not isinstance(angle, bool)
        ]
        protocol_ok = protocol_ok and len({
            angle for angle in pre_recovery if angle in energies
        }) < 10
    minimum = min(energies.values())
    relative = [(energies[angle] - minimum) * HARTREE_TO_KCAL_MOL for angle in valid_angles]
    profile = recompute_automatic_profile(valid_angles, relative)
    final_answer = _json(case_root / "final_answer.json")
    stored_curve = final_answer.get("predicted_energies_kcal_mol")
    profile_match = bool(
        isinstance(stored_curve, list)
        and len(stored_curve) == 72
        and np.allclose(
            np.asarray(stored_curve, dtype=float),
            np.asarray(profile["predicted_energies"], dtype=float),
            rtol=0.0,
            atol=1.0e-9,
        )
    )
    oracle = _json(project_root / str(case["oracle_path"]))
    oracle_values = oracle.get("relative_energies_kcal_mol")
    if not isinstance(oracle_values, list) or len(oracle_values) != 72:
        raise ValueError("ORACLE_PROFILE_INVALID")
    gates = {
        "complete_protocol": protocol_ok,
        "minimum_valid_angles": len(valid_angles) >= 10,
        # 2026-09-27 fix: a recovery (four-batch) run has three feedback-driven
        # rounds, which still satisfies the public minimum of two; hard-requiring
        # exactly three actions zeroed the documented recovery form.
        "minimum_feedback_rounds": len(actions) in {3, 4},
        "native_points_hash_bound": provenance_ok,
        "scientific_invariants_preserved": invariants_ok,
        "profile_recomputed_independently": profile_match,
        "exact_model_identity": exact_identity,
        "trajectory_integrity": True,
    }
    facts = {
        "adaptive_sampling": {
            "coverage_fraction": min(1.0, len(set(selected)) / 12.0),
            # 2026-09-27 fix: feedback rounds are used in both the three-batch and
            # the documented recovery (four-batch) form; other counts are unchanged.
            "feedback_fraction": 1.0 if len(actions) in {3, 4} else 0.0,
            "refinement_fraction": 1.0 if protocol_ok else 0.0,
            "efficiency_fraction": 1.0 if len(selected) <= 12 else 0.0,
        },
        "process_and_reproducibility": {
            "trajectory_integrity": True,
            "artifact_provenance": provenance_ok,
            "exact_identity": exact_identity,
            "scientific_invariants": invariants_ok,
            "replay_consistency": True,
        },
    }
    scored = score_profile_v2(
        predicted_energies_kcal_mol=profile["predicted_energies"],
        oracle_energies_kcal_mol=oracle_values,
        facts=facts,
        eligibility_gates=gates,
    )
    scientific = (
        scored["categories"]["curve_accuracy"]["score"]
        + scored["categories"]["extrema_and_barriers"]["score"]
    )
    process = (
        scored["categories"]["adaptive_sampling"]["score"]
        + scored["categories"]["process_and_reproducibility"]["score"]
    )
    return {
        "schema_version": "3.0",
        "case_id": case_id,
        "status": "PASS" if scored["eligible"] else "FAIL",
        "provider_status": metadata.get("provider_status"),
        "eligible": scored["eligible"],
        "failed_gates": scored["eligibility"]["failed_gates"],
        "scientific_score": round(scientific, 12),
        "process_score": round(process, 12),
        "official_score": scored["official_score"],
        "ungated_score": scored["ungated_total_score"],
        "selected_angles_deg": selected,
        "unique_valid_angles": len(valid_angles),
        "feedback_rounds": 2 if len(actions) == 3 else max(0, len(actions) - 1),
        "profile_fits": len(actions),
        "metrics": scored["metrics"],
        "ranking_included": True,
        "inputs": {
            "trajectory_sha256": sha256_file(trajectory_path),
            "final_answer_sha256": sha256_file(case_root / "final_answer.json"),
            "oracle_sha256": sha256_file(project_root / str(case["oracle_path"])),
        },
    }


__all__ = ["canonical_bytes", "evaluate_case"]

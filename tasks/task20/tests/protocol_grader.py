#!/usr/bin/env python3
"""Independent organizer-evidence replay grader for Task 20."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path


MECHANISMS = {"coking", "sintering", "sulfur_poisoning", "pore_blockage", "external_mass_transfer"}
OXIDATION_ATMOSPHERES = {"2% o2/n2", "dilute oxygen in nitrogen"}
REDUCTION_ATMOSPHERES = {"10% h2/n2", "dilute hydrogen in nitrogen"}
REQUIRED = {"primary_mechanisms", "contributions", "confidence", "claims", "regeneration_plan", "predicted_performance", "uncertainty"}
EVIDENCE_SCHEMA_VERSION = "chembench-organizer-evidence/1.0"
TASK_ID = "task-20-catalyst-deactivation-diagnosis"
CASE_ID = "PDH-DEACT-H01"
EVIDENCE_ID_PREFIX = "T20-EVIDENCE-"
ATTESTATION_PREFIX = "hmac-sha256:"


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _available(evidence, reference):
    ids = {item for item in reference.get("valid_evidence_ids", []) if item.startswith("INIT_")}
    for event in evidence.get("tool_events", []):
        response = event.get("response") if isinstance(event, dict) else None
        result = response.get("result") if isinstance(response, dict) and response.get("ok") is True else None
        if isinstance(result, dict) and isinstance(result.get("measurement_id"), str):
            ids.add(result["measurement_id"])
    return ids


def _interval_fraction(predicted, target):
    if not isinstance(predicted, list) or len(predicted) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in predicted):
        return 0.0
    lo, hi = predicted
    if lo > hi or not 0 <= lo <= hi <= 1:
        return 0.0
    overlap = max(0.0, min(hi, target[1]) - max(lo, target[0]))
    union = max(hi, target[1]) - min(lo, target[0])
    return overlap / union if union > 0 else 1.0 if predicted == target else 0.0


def _f1(submitted, expected):
    if not submitted or not expected:
        return 0.0
    overlap = len(submitted & expected)
    if not overlap:
        return 0.0
    precision = overlap / len(submitted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _claim_qualities(candidate, available, reference):
    submitted = {}
    for claim in candidate.get("claims", []):
        if not isinstance(claim, dict):
            continue
        relation = claim.get("relation")
        targets = {item for item in claim.get("mechanism_ids", []) if item in MECHANISMS}
        for evidence_id in claim.get("evidence_ids", []):
            if evidence_id in available:
                submitted.setdefault((evidence_id, relation), set()).update(targets)
    qualities = {}
    for evidence_id in reference.get("valid_evidence_ids", []):
        best = 0.0
        if evidence_id in available:
            for relation, expected in reference.get("claim_truth", {}).get(evidence_id, {}).items():
                best = max(best, _f1(submitted.get((evidence_id, relation), set()), set(expected)))
        qualities[evidence_id] = best
    return qualities


def _regeneration_score(plan, reference):
    if not isinstance(plan, list):
        return 0.0
    oxidation = 0.0
    reduction = 0.0
    all_controls = set()
    for action in plan:
        if not isinstance(action, dict):
            continue
        action_id = action.get("action_id")
        temp = action.get("temperature_C")
        duration = action.get("duration_h")
        ramp = action.get("ramp_C_min")
        oxygen = action.get("oxygen_percent")
        atmosphere = str(action.get("atmosphere", "")).strip().lower()
        action_controls = {
            item for item in action.get("safety_controls", [])
            if isinstance(item, str) and item
        }
        all_controls.update(action_controls)
        if action_id == "controlled_oxidation" and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (temp, duration, ramp, oxygen)):
            safe = reference["safe_regeneration"]
            required = set(safe["required_oxidation_controls"])
            if safe["oxidation_temperature_C"][0] <= temp <= safe["oxidation_temperature_C"][1] and safe["oxygen_percent"][0] <= oxygen <= safe["oxygen_percent"][1] and 0.1 <= duration <= 6 and 0 < ramp <= safe["ramp_C_min_max"] and atmosphere in OXIDATION_ATMOSPHERES and required <= action_controls:
                oxidation = 8.0
        if action_id == "reduction" and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (temp, duration, ramp)):
            safe = reference["safe_regeneration"]
            required = set(safe["required_reduction_controls"])
            if safe["reduction_temperature_C"][0] <= temp <= safe["reduction_temperature_C"][1] and oxygen == 0 and 0.1 <= duration <= 6 and 0 < ramp <= safe["ramp_C_min_max"] and atmosphere in REDUCTION_ATMOSPHERES and required <= action_controls:
                reduction = 4.0
    safety = 3.0 if set(reference["safe_regeneration"]["required_controls"]) <= all_controls else 0.0
    return min(15.0, oxidation + reduction + safety)


def grade_evidence(evidence, reference):
    candidate = evidence.get("candidate") if isinstance(evidence, dict) else None
    version = reference["grader_version"]
    if not isinstance(candidate, dict) or not REQUIRED.issubset(candidate):
        return {"score": 0.0, "subscores": {}, "failure_codes": ["INVALID_SUBMISSION"], "evidence": {}, "grader_version": version}
    if not _finite(candidate):
        return {"score": 0.0, "subscores": {}, "failure_codes": ["NONFINITE_VALUE"], "evidence": {}, "grader_version": version}
    confidence = candidate.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        return {"score": 0.0, "subscores": {}, "failure_codes": ["INVALID_CONFIDENCE"], "evidence": {}, "grader_version": version}
    submitted_primary = candidate.get("primary_mechanisms")
    if not isinstance(submitted_primary, list) or len(submitted_primary) != 2 or len(set(submitted_primary)) != 2 or any(item not in MECHANISMS for item in submitted_primary):
        return {"score": 0.0, "subscores": {}, "failure_codes": ["INVALID_PRIMARY_MECHANISMS"], "evidence": {}, "grader_version": version}
    contributions = candidate.get("contributions")
    if not isinstance(contributions, dict) or set(contributions) != MECHANISMS or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1 for v in contributions.values()):
        return {"score": 0.0, "subscores": {}, "failure_codes": ["INVALID_CONTRIBUTIONS"], "evidence": {}, "grader_version": version}
    if abs(sum(contributions.values()) - 1.0) > 1e-6:
        return {"score": 0.0, "subscores": {}, "failure_codes": ["INVALID_CONTRIBUTION_SUM"], "evidence": {}, "grader_version": version}

    available = _available(evidence, reference)
    qualities = _claim_qualities(candidate, available, reference)
    mechanism_supported = {
        mechanism: any(
            quality >= 0.8
            and mechanism in reference.get("claim_truth", {}).get(evidence_id, {}).get("supports", [])
            for evidence_id, quality in qualities.items()
        )
        for mechanism in MECHANISMS
    }

    truth_primary = reference["primary_mechanisms"]
    overlap = len(set(submitted_primary[:2]) & set(truth_primary))
    raw_ranking = 12.0 * overlap / 2.0 + (8.0 if submitted_primary[:2] == truth_primary else 4.0 if set(submitted_primary[:2]) == set(truth_primary) else 0.0)
    primary_support = sum(mechanism_supported.get(mechanism, False) for mechanism in truth_primary) / len(truth_primary)
    ranking = raw_ranking * primary_support

    tolerances = reference.get("contribution_identifiability_tolerance", {})
    mae = sum(
        max(0.0, abs(contributions[key] - reference["reference_contributions"][key]) - float(tolerances.get(key, 0.0)))
        for key in MECHANISMS
    ) / len(MECHANISMS)
    raw_contribution_score = 20.0 * max(0.0, 1.0 - mae / 0.25)
    contribution_support = sum(mechanism_supported.values()) / len(MECHANISMS)
    diagnostic_coherence = overlap / len(truth_primary)
    contribution_score = raw_contribution_score * diagnostic_coherence

    credited = {evidence_id for evidence_id, quality in qualities.items() if quality > 0}
    raw_evidence_score = min(20.0, 5.0 * sum(sorted(qualities.values(), reverse=True)[:4]))
    evidence_score = raw_evidence_score * diagnostic_coherence

    used = []
    for event in evidence.get("tool_events", []):
        response = event.get("response") if isinstance(event, dict) else None
        if not isinstance(response, dict) or response.get("ok") is not True:
            continue
        tool = event.get("tool") or response.get("tool")
        if tool in reference["diagnostic_utility"] and tool not in used:
            used.append(tool)
    best = sum(sorted(reference["diagnostic_utility"].values(), reverse=True)[:4])
    diagnostic_score = 15.0 * sum(reference["diagnostic_utility"][tool] for tool in used) / best if best else 0.0

    regeneration_score = _regeneration_score(candidate.get("regeneration_plan"), reference)
    performance = candidate.get("predicted_performance") if isinstance(candidate.get("predicted_performance"), dict) else {}
    prediction_score = 2.5 * _interval_fraction(performance.get("conversion"), reference["target_performance"]["conversion"]) + 2.5 * _interval_fraction(performance.get("selectivity"), reference["target_performance"]["selectivity"])
    uncertainty = candidate.get("uncertainty") if isinstance(candidate.get("uncertainty"), dict) else {}
    alternatives = uncertainty.get("alternatives") if isinstance(uncertainty.get("alternatives"), list) else []
    limitations = uncertainty.get("limitations") if isinstance(uncertainty.get("limitations"), list) else []
    exact_primary = submitted_primary[:2] == truth_primary
    confidence_credit = 2.0 if (exact_primary and confidence >= 0.5) or (not exact_primary and confidence <= 0.5) else 0.0
    alternative_credit = 1.5 if any(item in MECHANISMS and item not in submitted_primary for item in alternatives) else 0.0
    limitation_credit = 1.5 if any(isinstance(item, str) and item.strip() for item in limitations) else 0.0
    uncertainty_score = confidence_credit + alternative_credit + limitation_credit

    subscores = {
        "mechanism_ranking": round(ranking, 4),
        "contribution_accuracy": round(contribution_score, 4),
        "evidence_discrimination": round(evidence_score, 4),
        "diagnostic_efficiency": round(diagnostic_score, 4),
        "regeneration_and_safety": round(regeneration_score, 4),
        "performance_prediction": round(prediction_score, 4),
        "confidence_and_uncertainty": round(uncertainty_score, 4),
    }
    failures = [] if credited else ["NO_REPLAYABLE_DIAGNOSTIC_EVIDENCE"]
    return {"score": round(sum(subscores.values()), 4), "subscores": subscores, "failure_codes": failures, "evidence": {"available_evidence_ids": sorted(available), "credited_evidence_ids": sorted(credited), "replayed_tool_count": len(used), "raw_mechanism_ranking": round(raw_ranking, 4), "primary_evidence_support": round(primary_support, 4), "diagnostic_coherence": round(diagnostic_coherence, 4), "raw_contribution_accuracy": round(raw_contribution_score, 4), "raw_evidence_discrimination": round(raw_evidence_score, 4), "contribution_evidence_support": round(contribution_support, 4), "mechanism_evidence_coverage": mechanism_supported}, "grader_version": version}


def _find(root):
    if root.is_file():
        return root
    paths = sorted(root.glob("**/case-1-evidence.json"))
    if len(paths) != 1:
        raise RuntimeError("organizer evidence must contain exactly one case")
    return paths[0]


def _attestation_payload(evidence):
    required = {
        "schema_version", "task_id", "case_id", "session_id", "evidence_id",
        "candidate", "candidate_sha256", "tool_events", "attestation",
    }
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("verifier artifact fields are missing")
    if evidence["schema_version"] != EVIDENCE_SCHEMA_VERSION or evidence["task_id"] != TASK_ID or evidence["case_id"] != CASE_ID:
        raise ValueError("verifier artifact identity mismatch")
    if not isinstance(evidence["session_id"], str) or re.fullmatch(r"[A-Za-z0-9_-]{32}", evidence["session_id"]) is None:
        raise ValueError("verifier artifact session provenance is invalid")
    evidence_id = evidence["evidence_id"]
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("verifier artifact evidence ID is missing")
    candidate = evidence["candidate"]
    if not isinstance(candidate, dict):
        raise ValueError("verifier artifact candidate is malformed")
    canonical = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    candidate_sha256 = hashlib.sha256(canonical).hexdigest()
    if evidence["candidate_sha256"] != candidate_sha256:
        raise ValueError("verifier artifact candidate hash mismatch")
    if evidence_id != EVIDENCE_ID_PREFIX + candidate_sha256[:12]:
        raise ValueError("verifier artifact evidence ID provenance mismatch")
    events = evidence["tool_events"]
    if not isinstance(events, list) or any(
        not isinstance(event, dict)
        or set(event) != {"tool", "arguments", "response"}
        or not isinstance(event["response"], dict)
        for event in events
    ):
        raise ValueError("verifier artifact tool events are malformed")
    for event in events:
        response = event["response"]
        if response.get("ok") is not True and response.get("ok") is not False:
            raise ValueError("verifier artifact tool response status is malformed")
        if response.get("ok") is True and (
            not isinstance(event["tool"], str)
            or not event["tool"]
            or not isinstance(event["arguments"], dict)
        ):
            raise ValueError("verifier artifact successful tool event is malformed")
    return {
        "schema_version": evidence["schema_version"],
        "task_id": evidence["task_id"],
        "case_id": evidence["case_id"],
        "session_id": evidence["session_id"],
        "evidence_id": evidence_id,
        "candidate": candidate,
        "candidate_sha256": candidate_sha256,
        "tool_events": events,
    }


def _validate_attestation(evidence, reference):
    payload = _attestation_payload(evidence)
    key = reference.get("attestation_key") if isinstance(reference, dict) else None
    if not isinstance(key, str) or len(key) < 32:
        raise ValueError("private attestation key is missing")
    attestation = evidence.get("attestation")
    if not isinstance(attestation, str) or not attestation:
        raise ValueError("verifier artifact attestation is missing")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = ATTESTATION_PREFIX + hmac.new(
        key.encode("utf-8"), canonical, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(attestation, expected):
        raise ValueError("verifier artifact attestation mismatch")


def _validate_verifier_bridge(evidence):
    required = {"public_case", "submission"}
    if not isinstance(evidence, dict) or not required.issubset(evidence):
        raise ValueError("verifier bridge fields are missing")
    evidence_id = evidence["evidence_id"]
    public_case = evidence["public_case"]
    submission = evidence["submission"]
    if not isinstance(public_case, dict) or public_case.get("case_id") != CASE_ID:
        raise ValueError("verifier artifact public case mismatch")
    if (
        not isinstance(submission, dict)
        or submission.get("ok") is not True
        or submission.get("case_id") != CASE_ID
        or submission.get("evidence_id") != evidence_id
        or submission.get("attestation") != evidence.get("attestation")
    ):
        raise ValueError("verifier artifact submission mismatch")


def main():
    task_root = Path("/task")
    if not (task_root / "private_data/reference.json").is_file():
        task_root = Path(__file__).parent / "task_snapshot"
    reference = json.loads((task_root / "private_data/reference.json").read_text())
    evidence_root = Path(os.environ.get("CHEMBENCH_ORGANIZER_EVIDENCE", "/organizer_evidence"))
    logs = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
    try:
        evidence_path = _find(evidence_root)
        evidence = json.loads(evidence_path.read_text())
        _validate_attestation(evidence, reference)
        if evidence_path.name == "case_result.json":
            _validate_verifier_bridge(evidence)
        result = grade_evidence(evidence, reference)
    except Exception as exc:
        result = {"score": 0.0, "subscores": {}, "failure_codes": ["ORGANIZER_EVIDENCE_ERROR"], "evidence": {"error_type": type(exc).__name__}, "grader_version": reference["grader_version"]}
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "grader-result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (logs / "reward.txt").write_text(str(max(0.0, min(1.0, result["score"] / 100.0))))
    print(json.dumps({"score": result["score"], "failure_codes": result["failure_codes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

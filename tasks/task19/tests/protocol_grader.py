#!/usr/bin/env python3
"""Pure deterministic organizer-evidence grader for Task 19 v2."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path


GRADER_VERSION = "task19-protocol-grader/3.1.0"
EVIDENCE_SCHEMA_VERSION = "chembench-organizer-evidence/1.0"
TASK_ID = "task-19-active-impurity-elucidation"
CASE_ID = "LID-DUAL-OX-H01"
EVIDENCE_ID_PREFIX = "T19-EVIDENCE-"
PUBLIC_ROLES = {"aromatic_oxidation_branch", "amine_oxidation_branch", "sidechain_oxidation_branch"}
ATTESTATION_PREFIX = "hmac-sha256:"
REQUIRED_CANDIDATE_FIELDS = {
    "pair_hypothesis_id",
    "members",
    "claims",
    "pathways",
    "adaptive_rationale",
    "excluded_pair_hypotheses",
    "uncertainty",
}
SUBSCORE_KEYS = (
    "identity_and_structure",
    "evidence_consistency",
    "degradation_pathway",
    "experiment_efficiency",
    "uncertainty_and_alternatives",
)


def _finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _candidate_shape_is_valid(candidate: dict) -> bool:
    if set(candidate) != REQUIRED_CANDIDATE_FIELDS:
        return False
    if not isinstance(candidate.get("pair_hypothesis_id"), str) or not candidate["pair_hypothesis_id"]:
        return False

    members = candidate.get("members")
    if not isinstance(members, list) or len(members) != 2 or not all(isinstance(member, dict) for member in members):
        return False
    member_ids = []
    for member in members:
        if set(member) != {"candidate_id", "role", "abundance_interval", "structural_features"}:
            return False
        candidate_id = member.get("candidate_id")
        role = member.get("role")
        interval = member.get("abundance_interval")
        features = member.get("structural_features")
        if not isinstance(candidate_id, str) or not candidate_id or role not in PUBLIC_ROLES:
            return False
        if not isinstance(interval, list) or len(interval) != 2 or not all(_number(value) for value in interval):
            return False
        if not 0 <= interval[0] <= interval[1] <= 1:
            return False
        if not isinstance(features, list) or not features or not all(isinstance(item, str) and item for item in features):
            return False
        member_ids.append(candidate_id)
    if len(set(member_ids)) != 2:
        return False

    claims = candidate.get("claims")
    if not isinstance(claims, list):
        return False
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"claim_id", "evidence_ids", "relation", "candidate_ids"}:
            return False
        evidence_ids = claim.get("evidence_ids")
        candidate_ids = claim.get("candidate_ids")
        if not isinstance(claim.get("claim_id"), str) or not claim["claim_id"]:
            return False
        if claim.get("relation") not in {"supports", "contradicts", "neutral"}:
            return False
        if not isinstance(evidence_ids, list) or not evidence_ids or not all(isinstance(item, str) and item for item in evidence_ids):
            return False
        if len(set(evidence_ids)) != len(evidence_ids):
            return False
        if not isinstance(candidate_ids, list) or not candidate_ids or not all(isinstance(item, str) and item for item in candidate_ids):
            return False
        if len(set(candidate_ids)) != len(candidate_ids):
            return False

    pathways = candidate.get("pathways")
    if not isinstance(pathways, list) or len(pathways) != 2 or not all(isinstance(pathway, dict) for pathway in pathways):
        return False
    pathway_ids = []
    for pathway in pathways:
        if set(pathway) != {"candidate_id", "events"}:
            return False
        candidate_id = pathway.get("candidate_id")
        events = pathway.get("events")
        if not isinstance(candidate_id, str) or not candidate_id:
            return False
        if not isinstance(events, list) or not events or not all(isinstance(event, str) and event for event in events):
            return False
        pathway_ids.append(candidate_id)
    if len(set(pathway_ids)) != 2 or set(pathway_ids) != set(member_ids):
        return False

    if not isinstance(candidate.get("adaptive_rationale"), str) or not candidate["adaptive_rationale"].strip():
        return False
    excluded = candidate.get("excluded_pair_hypotheses")
    if not isinstance(excluded, list) or not all(isinstance(item, str) and item for item in excluded):
        return False
    if len(set(excluded)) != len(excluded):
        return False

    uncertainty = candidate.get("uncertainty")
    if not isinstance(uncertainty, dict) or set(uncertainty) != {
        "remaining_pair_hypotheses",
        "limitations",
        "budget_external_confirmation",
    }:
        return False
    remaining = uncertainty.get("remaining_pair_hypotheses")
    limitations = uncertainty.get("limitations")
    confirmation = uncertainty.get("budget_external_confirmation")
    if not isinstance(remaining, list) or not all(isinstance(item, str) and item for item in remaining):
        return False
    if len(set(remaining)) != len(remaining):
        return False
    if not isinstance(limitations, list) or not all(isinstance(item, str) and item for item in limitations):
        return False
    return isinstance(confirmation, str) and bool(confirmation.strip())


def _f1(submitted: set[str], expected: set[str]) -> float:
    if not submitted or not expected:
        return 0.0
    overlap = len(submitted & expected)
    if not overlap:
        return 0.0
    precision = overlap / len(submitted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _interval_overlap(submitted: list[float], expected: list[float]) -> float:
    left = max(submitted[0], expected[0])
    right = min(submitted[1], expected[1])
    intersection = max(0.0, right - left)
    union = max(submitted[1], expected[1]) - min(submitted[0], expected[0])
    if union == 0:
        return 1.0 if submitted == expected else 0.0
    return intersection / union


def score_joint_identity(candidate: dict, reference: dict) -> float:
    """Score pair ID, member/role sets, member features and abundance overlap (max 25)."""
    submitted_members = {
        member["candidate_id"]: member for member in candidate.get("members", []) if isinstance(member, dict)
    }
    expected_members = {
        member["candidate_id"]: member for member in reference.get("members", []) if isinstance(member, dict)
    }
    expected_ids = set(expected_members)
    submitted_ids = set(submitted_members)

    pair_credit = 5.0 if candidate.get("pair_hypothesis_id") == reference.get("correct_pair_hypothesis_id") else 0.0
    member_credit = 8.0 * len(submitted_ids & expected_ids) / len(expected_ids) if expected_ids else 0.0
    expected_roles = {(candidate_id, member.get("role")) for candidate_id, member in expected_members.items()}
    submitted_roles = {(candidate_id, member.get("role")) for candidate_id, member in submitted_members.items()}
    role_credit = 4.0 * len(submitted_roles & expected_roles) / len(expected_roles) if expected_roles else 0.0

    feature_credit = 0.0
    interval_credit = 0.0
    per_member_feature_weight = 4.0 / len(expected_ids) if expected_ids else 0.0
    per_member_interval_weight = 4.0 / len(expected_ids) if expected_ids else 0.0
    for candidate_id in submitted_ids & expected_ids:
        submitted = submitted_members[candidate_id]
        expected = expected_members[candidate_id]
        feature_credit += per_member_feature_weight * _f1(
            set(submitted.get("structural_features", [])), set(expected.get("structural_features", []))
        )
        interval_credit += per_member_interval_weight * _interval_overlap(
            submitted.get("abundance_interval", []), expected.get("abundance_interval", [])
        )
    return min(25.0, pair_credit + member_credit + role_credit + feature_credit + interval_credit)


def _canonical(arguments: dict) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _successful_events(events: list[dict], reference: dict) -> list[dict]:
    valid_ids = set(reference.get("valid_evidence_ids", []))
    information_gain = reference.get("experiment_information_gain", {})
    successful = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        tool = event.get("tool")
        arguments = event.get("arguments")
        response = event.get("response")
        if not isinstance(tool, str) or not isinstance(arguments, dict) or not isinstance(response, dict):
            continue
        if response.get("ok") is not True or response.get("tool") != tool:
            continue
        result = response.get("result")
        measurement_id = result.get("measurement_id") if isinstance(result, dict) else None
        selection_key = f"{tool}|{_canonical(arguments)}"
        if selection_key not in information_gain or measurement_id not in valid_ids:
            continue
        successful.append(
            {
                "tool": tool,
                "measurement_id": measurement_id,
                "selection_key": selection_key,
            }
        )
    return successful


def _available_evidence(evidence: dict, reference: dict) -> set[str]:
    available = {
        evidence_id
        for evidence_id in reference.get("valid_evidence_ids", [])
        if isinstance(evidence_id, str) and evidence_id.startswith("INIT_")
    }
    available.update(
        event["measurement_id"] for event in _successful_events(evidence.get("tool_events", []), reference)
    )
    return available


def _claim_quality_by_evidence(candidate: dict, available: set[str], reference: dict) -> dict[str, float]:
    submitted_targets: dict[tuple[str, str], set[str]] = {}
    for claim in candidate.get("claims", []):
        if not isinstance(claim, dict):
            continue
        relation = claim.get("relation")
        targets = {item for item in claim.get("candidate_ids", []) if isinstance(item, str)}
        for evidence_id in claim.get("evidence_ids", []):
            if evidence_id in available:
                submitted_targets.setdefault((evidence_id, relation), set()).update(targets)

    qualities = {}
    truth = reference.get("claim_truth", {})
    for evidence_id in reference.get("valid_evidence_ids", []):
        if evidence_id not in available:
            qualities[evidence_id] = 0.0
            continue
        best = 0.0
        relations = truth.get(evidence_id, {})
        if isinstance(relations, dict):
            for relation, expected_targets in relations.items():
                submitted = submitted_targets.get((evidence_id, relation), set())
                expected = {item for item in expected_targets if isinstance(item, str)}
                best = max(best, _f1(submitted, expected))
        qualities[evidence_id] = best
    return qualities


def score_claims(candidate: dict, evidence: dict, reference: dict) -> tuple[float, set[str]]:
    """Score relation-correct target-set F1 for replayable organizer evidence (max 25)."""
    available = _available_evidence(evidence, reference)
    qualities = _claim_quality_by_evidence(candidate, available, reference)
    score = 25.0 * sum(qualities.get(evidence_id, 0.0) for evidence_id in available) / len(available) if available else 0.0
    credited = {evidence_id for evidence_id, quality in qualities.items() if quality > 0}
    return min(25.0, score), credited


def _lcs(left: list, right: list) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, 1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def score_pathways(candidate: dict, reference: dict) -> tuple[float, float]:
    """Score both candidate-specific ordered pathways by normalized LCS (max 20)."""
    submitted = {
        pathway.get("candidate_id"): pathway.get("events", [])
        for pathway in candidate.get("pathways", [])
        if isinstance(pathway, dict)
    }
    fractions = []
    for candidate_id, expected in reference.get("pathways", {}).items():
        observed = submitted.get(candidate_id, [])
        fractions.append(_lcs(observed, expected) / len(expected) if expected else 1.0)
    completeness = sum(fractions) / len(fractions) if fractions else 0.0
    return 20.0 * completeness, completeness


def _requirement_support(
    candidate: dict, available: set[str], reference: dict, requirement_key: str
) -> tuple[float, dict[str, bool]]:
    """Return member coverage for correctly interpreted, replayable evidence bundles."""
    qualities = _claim_quality_by_evidence(candidate, available, reference)
    coverage = {}
    for candidate_id, alternatives in reference.get(requirement_key, {}).items():
        coverage[candidate_id] = any(
            isinstance(bundle, list)
            and bool(bundle)
            and all(qualities.get(evidence_id, 0.0) >= 0.8 for evidence_id in bundle)
            for bundle in alternatives
        )
    fraction = sum(coverage.values()) / len(coverage) if coverage else 1.0
    return fraction, coverage


def score_adaptive_experiments(candidate: dict, events: list[dict], reference: dict) -> float:
    """Score successful information gain plus complementary and evidence-grounded adaptation (max 20)."""
    successful = _successful_events(events, reference)
    unique = []
    seen = set()
    for event in successful:
        if event["selection_key"] not in seen:
            seen.add(event["selection_key"])
            unique.append(event)
    if not unique:
        return 0.0

    information_gain = reference.get("experiment_information_gain", {})
    oracle_keys = reference.get("oracle_experiment_keys", [])
    best_four = sum(information_gain[key] for key in oracle_keys if key in information_gain)
    if not best_four:
        best_four = sum(sorted(information_gain.values(), reverse=True)[:4])
    observed_gain = sum(information_gain[event["selection_key"]] for event in unique)
    information_credit = 15.0 * min(1.0, observed_gain / best_four) if best_four else 0.0

    complementary_pairs = {
        frozenset(pair)
        for pair in reference.get("complementary_tool_families", [])
        if isinstance(pair, list) and len(pair) == 2
    }
    selected_families = {event["tool"] for event in unique}
    achieved_pairs = {pair for pair in complementary_pairs if pair <= selected_families}
    oracle_families = {key.split("|", 1)[0] for key in oracle_keys}
    oracle_pairs = {pair for pair in complementary_pairs if pair <= oracle_families}
    denominator_pairs = oracle_pairs or complementary_pairs
    achieved_pairs = {pair for pair in denominator_pairs if pair <= selected_families}
    complement_credit = 2.5 * len(achieved_pairs) / len(denominator_pairs) if denominator_pairs else 0.0

    measurement_ids = {event["measurement_id"] for event in unique}
    qualities = _claim_quality_by_evidence(candidate, measurement_ids, reference)
    credited_ids = {evidence_id for evidence_id, quality in qualities.items() if quality > 0}
    adaptive = False
    for later_index, later in enumerate(unique):
        for earlier in unique[:later_index]:
            if (
                frozenset((earlier["tool"], later["tool"])) in complementary_pairs
                and earlier["measurement_id"] in credited_ids
                and later["measurement_id"] in credited_ids
            ):
                adaptive = True
                break
        if adaptive:
            break
    adaptive_credit = 2.5 if adaptive and candidate.get("adaptive_rationale", "").strip() else 0.0
    return min(20.0, information_credit + complement_credit + adaptive_credit)


def score_uncertainty(candidate: dict, reference: dict) -> float:
    """Score pair/alternative consistency and explicit residual uncertainty (max 10)."""
    truth = reference.get("correct_pair_hypothesis_id")
    selected = candidate.get("pair_hypothesis_id")
    uncertainty = candidate.get("uncertainty", {})
    remaining = set(uncertainty.get("remaining_pair_hypotheses", []))
    excluded = set(candidate.get("excluded_pair_hypotheses", []))

    if selected == truth:
        hypothesis_credit = 4.0
    elif truth in remaining and truth not in excluded:
        hypothesis_credit = 3.0
    else:
        hypothesis_credit = 0.0
    consistency_credit = 2.0 if selected not in remaining and selected not in excluded else 0.0
    truth_safeguard_credit = 2.0 if truth not in excluded else 0.0
    limitation_credit = 1.0 if uncertainty.get("limitations") else 0.0
    confirmation_credit = 1.0 if uncertainty.get("budget_external_confirmation", "").strip() else 0.0
    return min(
        10.0,
        hypothesis_credit
        + consistency_credit
        + truth_safeguard_credit
        + limitation_credit
        + confirmation_credit,
    )


def _empty_subscores() -> dict[str, float]:
    return {key: 0.0 for key in SUBSCORE_KEYS}


def _failure_result(code: str, evidence_details: dict | None = None) -> dict:
    return {
        "score": 0.0,
        "subscores": _empty_subscores(),
        "failure_codes": [code],
        "evidence": evidence_details or {},
        "grader_version": GRADER_VERSION,
    }


def grade_evidence(evidence: dict, reference: dict) -> dict:
    candidate = evidence.get("candidate") if isinstance(evidence, dict) else None
    if not isinstance(candidate, dict):
        return _failure_result("INVALID_SUBMISSION")
    if not _finite(candidate):
        return _failure_result("NONFINITE_VALUE")
    if not _candidate_shape_is_valid(candidate):
        return _failure_result("INVALID_SUBMISSION")

    raw_identity_score = score_joint_identity(candidate, reference)
    evidence_score, credited_ids = score_claims(candidate, evidence, reference)
    raw_pathway_score, pathway_completeness = score_pathways(candidate, reference)
    experiment_score = score_adaptive_experiments(candidate, evidence.get("tool_events", []), reference)
    uncertainty_score = score_uncertainty(candidate, reference)
    available = _available_evidence(evidence, reference)
    identity_support, identity_coverage = _requirement_support(
        candidate, available, reference, "identity_evidence_requirements"
    )
    pathway_support, pathway_coverage = _requirement_support(
        candidate, available, reference, "pathway_evidence_requirements"
    )
    identity_score = raw_identity_score * (0.25 + 0.75 * identity_support)
    pathway_score = raw_pathway_score * pathway_support

    subscores = {
        "identity_and_structure": round(identity_score, 4),
        "evidence_consistency": round(evidence_score, 4),
        "degradation_pathway": round(pathway_score, 4),
        "experiment_efficiency": round(experiment_score, 4),
        "uncertainty_and_alternatives": round(uncertainty_score, 4),
    }
    pre_pathway_base = identity_score + evidence_score + experiment_score
    pathway_factor = 0.5 + 0.5 * pathway_completeness
    post_pathway_base = pre_pathway_base * pathway_factor
    pre_joint_identity_gate_score = post_pathway_base + pathway_score + uncertainty_score
    expected_member_ids = {member["candidate_id"] for member in reference.get("members", [])}
    submitted_member_ids = {member["candidate_id"] for member in candidate.get("members", [])}
    correct_member_fraction = (
        len(expected_member_ids & submitted_member_ids) / len(expected_member_ids)
        if expected_member_ids
        else 0.0
    )
    full_joint_identity = (
        correct_member_fraction == 1.0
        and candidate.get("pair_hypothesis_id") == reference.get("correct_pair_hypothesis_id")
    )
    joint_identity_score_ceiling = 30.0 + 38.0 * correct_member_fraction
    if full_joint_identity:
        joint_identity_score_ceiling += 32.0
    post_joint_identity_gate_score = min(
        pre_joint_identity_gate_score, joint_identity_score_ceiling
    )
    score = max(0.0, min(100.0, post_joint_identity_gate_score))

    claim_qualities = _claim_quality_by_evidence(candidate, available, reference)
    successful = _successful_events(evidence.get("tool_events", []), reference)
    successful_keys = sorted({event["selection_key"] for event in successful})
    failures = [] if credited_ids else ["NO_REPLAYABLE_EVIDENCE"]
    return {
        "score": round(score, 4),
        "subscores": subscores,
        "failure_codes": failures,
        "evidence": {
            "available_evidence_ids": sorted(available),
            "credited_evidence_ids": sorted(credited_ids),
            "valid_claim_count": sum(quality > 0 for quality in claim_qualities.values()),
            "replayed_tool_count": len(successful_keys),
            "successful_experiment_keys": successful_keys,
            "pathway_completeness": round(pathway_completeness, 4),
            "raw_identity_score": round(raw_identity_score, 4),
            "identity_evidence_support": round(identity_support, 4),
            "identity_evidence_coverage": identity_coverage,
            "raw_pathway_score": round(raw_pathway_score, 4),
            "pathway_evidence_support": round(pathway_support, 4),
            "pathway_evidence_coverage": pathway_coverage,
            "pathway_completeness_factor": round(pathway_factor, 4),
            "pre_pathway_base_score": round(pre_pathway_base, 4),
            "post_pathway_base_score": round(post_pathway_base, 4),
            "correct_member_fraction": round(correct_member_fraction, 4),
            "full_joint_identity": full_joint_identity,
            "pre_joint_identity_gate_score": round(pre_joint_identity_gate_score, 4),
            "joint_identity_score_ceiling": round(joint_identity_score_ceiling, 4),
            "joint_identity_gate_applied": post_joint_identity_gate_score
            < pre_joint_identity_gate_score,
            "post_joint_identity_gate_score": round(post_joint_identity_gate_score, 4),
        },
        "grader_version": GRADER_VERSION,
    }


def _find_evidence(root: Path) -> Path:
    if root.is_file():
        return root
    candidates = sorted(root.glob("**/case-1-evidence.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one organizer evidence file, found {len(candidates)}")
    return candidates[0]


def _attestation_payload(evidence: dict) -> dict:
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


def _validate_attestation(evidence: dict, reference: dict) -> None:
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


def _validate_verifier_bridge(evidence: dict) -> None:
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


def main() -> int:
    task_root = Path("/task")
    if not (task_root / "private_data/reference.json").is_file():
        task_root = Path(__file__).parent / "task_snapshot"
    evidence_root = Path(os.environ.get("CHEMBENCH_ORGANIZER_EVIDENCE", "/organizer_evidence"))
    logs = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
    try:
        reference = json.loads((task_root / "private_data/reference.json").read_text(encoding="utf-8"))
        evidence_path = _find_evidence(evidence_root)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        _validate_attestation(evidence, reference)
        if evidence_path.name == "case_result.json":
            _validate_verifier_bridge(evidence)
        result = grade_evidence(evidence, reference)
    except Exception as exc:
        result = _failure_result("ORGANIZER_EVIDENCE_ERROR", {"error_type": type(exc).__name__})
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "grader-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (logs / "reward.txt").write_text(
        str(max(0.0, min(1.0, result["score"] / 100.0))), encoding="utf-8"
    )
    print(json.dumps({"score": result["score"], "failure_codes": result["failure_codes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

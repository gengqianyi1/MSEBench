"""Private deterministic sidecar adapter for Task 19 v3. Never mount this file publicly."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path


PUBLIC_FIELDS = {
    "energy_resolved_msms": {"energy_eV": {30, 45, 60}},
    "selective_derivatization": {"reagent": {"d2o", "dansyl_chloride"}},
    "correlation_window": {"window": {"aromatic", "sidechain", "cross_component"}},
    "condition_perturbation": {"condition": {"dark", "radical_scavenger", "pH4"}},
}
EVIDENCE_SCHEMA_VERSION = "chembench-organizer-evidence/1.0"
TASK_ID = "task-19-active-impurity-elucidation"
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


def _canonical(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _attestation_events(tool_events) -> list[dict]:
    if not isinstance(tool_events, list):
        raise ValueError("tool_events_malformed")
    normalized = []
    for event in tool_events:
        if not isinstance(event, dict) or set(event) != {"tool", "arguments", "response"}:
            raise ValueError("tool_event_malformed")
        response = event.get("response")
        if not isinstance(response, dict) or (
            response.get("ok") is not True and response.get("ok") is not False
        ):
            raise ValueError("tool_response_malformed")
        tool = event.get("tool")
        arguments = event.get("arguments")
        if response.get("ok") is True and (
            not isinstance(tool, str) or not tool or not isinstance(arguments, dict)
        ):
            raise ValueError("successful_tool_event_malformed")
        normalized.append({"tool": tool, "arguments": arguments, "response": response})
    return normalized


def _attest(key: str, payload: dict) -> str:
    if not isinstance(key, str) or len(key) < 32:
        raise ValueError("attestation_key_missing")
    digest = hmac.new(key.encode("utf-8"), _canonical(payload).encode("utf-8"), hashlib.sha256)
    return ATTESTATION_PREFIX + digest.hexdigest()


def _valid_members(candidate: dict, candidate_ids: set[str], expected_pair: set[str], features_by_candidate: dict[str, set[str]]) -> bool:
    members = candidate.get("members")
    if not isinstance(members, list) or len(members) != 2 or not all(isinstance(member, dict) for member in members):
        return False
    member_ids = [member.get("candidate_id") for member in members]
    if not all(isinstance(member_id, str) for member_id in member_ids):
        return False
    if set(member_ids) != expected_pair or len(set(member_ids)) != 2 or not set(member_ids) <= candidate_ids:
        return False
    for member in members:
        if set(member) != {"candidate_id", "role", "abundance_interval", "structural_features"}:
            return False
        interval = member.get("abundance_interval")
        if member.get("role") not in {"aromatic_oxidation_branch", "amine_oxidation_branch", "sidechain_oxidation_branch"}:
            return False
        features = member.get("structural_features")
        if not isinstance(features, list) or not features or not all(isinstance(item, str) for item in features):
            return False
        if not set(features) <= features_by_candidate[member["candidate_id"]]:
            return False
        if not isinstance(interval, list) or len(interval) != 2 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in interval
        ):
            return False
        if not 0 <= interval[0] <= interval[1] <= 1:
            return False
    return True


class Adapter:
    def __init__(self, snapshot):
        self.private = Path(snapshot) / "private_data"
        self.case = json.loads((self.private / "suite.json").read_text(encoding="utf-8"))
        self.responses = json.loads((self.private / "experiment_data.json").read_text(encoding="utf-8"))
        reference = json.loads((self.private / "reference.json").read_text(encoding="utf-8"))
        oracle_keys = reference.get("oracle_experiment_keys")
        information_gain = reference.get("experiment_information_gain")
        if (
            not isinstance(oracle_keys, list)
            or len(oracle_keys) != 4
            or len(set(oracle_keys)) != 4
            or not isinstance(information_gain, dict)
            or any(key not in information_gain for key in oracle_keys)
        ):
            raise ValueError("oracle_experiment_keys must contain four unique known experiment keys")
        self.attestation_key = reference.get("attestation_key")
        self.case_count = 1
        self.candidate_ids = set(self.case["constraints"]["candidate_ids"])
        self.pairs = {
            pair["pair_hypothesis_id"]: set(pair["candidate_ids"])
            for pair in self.case["task"]["pair_hypotheses"]
        }
        self.features_by_candidate = {
            item["candidate_id"]: set(item["feature_ids"])
            for item in self.case["task"]["single_candidates"]
        }
        self.pathway_event_ids = set(self.case["task"]["pathway_event_ids"])
        self.relations = set(self.case["constraints"]["relations"])
        self.evidence_ids = {
            item["evidence_id"] for item in self.case["task"]["initial_evidence"]
        } | {
            result["measurement_id"] for result in self.responses.values()
        }

    def public_case(self, index):
        if index != 0:
            raise IndexError(index)
        return {
            key: self.case[key]
            for key in ("case_id", "task", "constraints", "budget", "output", "public_tools")
        }

    def start_session(self, evidence_dir, token):
        root = Path(evidence_dir) / token if evidence_dir else None
        if root:
            root.mkdir(parents=True, exist_ok=True)
            (root / "session.json").write_text(
                json.dumps(
                    {
                        "task_id": TASK_ID,
                        "case_id": self.case["case_id"],
                        "session_id": token,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return {"evidence_root": str(root) if root else None, "session_id": token}

    @staticmethod
    def _error(tool, code):
        return {"ok": False, "tool": tool if isinstance(tool, str) else "", "error": {"code": code}}

    def call_tool(self, index, name, arguments, state=None):
        if index != 0 or name not in PUBLIC_FIELDS:
            return self._error(name, "unknown_tool")
        if not isinstance(arguments, dict):
            return self._error(name, "invalid_arguments")
        schema = PUBLIC_FIELDS[name]
        if set(arguments) != set(schema) or any(
            isinstance(arguments[key], bool)
            or not isinstance(arguments[key], (str, int))
            or arguments[key] not in choices
            for key, choices in schema.items()
        ):
            return self._error(name, "invalid_arguments")
        key = f"{name}|{_canonical(arguments)}"
        if key not in self.responses:
            return self._error(name, "measurement_unavailable")
        call_id = "T18-" + hashlib.sha256(key.encode()).hexdigest()[:12]
        return {"ok": True, "tool": name, "call_id": call_id, "result": self.responses[key]}

    def _valid_candidate(self, candidate) -> bool:
        if not isinstance(candidate, dict) or set(candidate) != REQUIRED_CANDIDATE_FIELDS:
            return False
        pair_id = candidate.get("pair_hypothesis_id")
        if pair_id not in self.pairs or not _valid_members(
            candidate, self.candidate_ids, self.pairs[pair_id], self.features_by_candidate
        ):
            return False
        if not self._valid_claims(candidate["claims"]):
            return False
        if not self._valid_pathways(candidate["pathways"], self.pairs[pair_id]):
            return False
        if not isinstance(candidate["adaptive_rationale"], str) or not candidate["adaptive_rationale"].strip():
            return False
        excluded = candidate["excluded_pair_hypotheses"]
        if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
            return False
        if len(set(excluded)) != len(excluded) or not all(item in self.pairs for item in excluded):
            return False
        return self._valid_uncertainty(candidate["uncertainty"])

    def _valid_claims(self, claims) -> bool:
        if not isinstance(claims, list):
            return False
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"claim_id", "evidence_ids", "relation", "candidate_ids"}:
                return False
            evidence_ids = claim["evidence_ids"]
            candidate_ids = claim["candidate_ids"]
            if not isinstance(claim["claim_id"], str) or not claim["claim_id"]:
                return False
            if not isinstance(evidence_ids, list) or not evidence_ids or not all(isinstance(item, str) for item in evidence_ids):
                return False
            if len(set(evidence_ids)) != len(evidence_ids):
                return False
            if claim["relation"] not in self.relations:
                return False
            if not isinstance(candidate_ids, list) or not candidate_ids or not all(isinstance(item, str) for item in candidate_ids):
                return False
            if len(set(candidate_ids)) != len(candidate_ids) or not all(item in self.candidate_ids for item in candidate_ids):
                return False
        return True

    def _valid_pathways(self, pathways, expected_pair: set[str]) -> bool:
        if not isinstance(pathways, list) or len(pathways) != 2 or not all(isinstance(pathway, dict) for pathway in pathways):
            return False
        pathway_candidates = [pathway.get("candidate_id") for pathway in pathways]
        if not all(isinstance(candidate_id, str) for candidate_id in pathway_candidates):
            return False
        if set(pathway_candidates) != expected_pair or len(set(pathway_candidates)) != 2:
            return False
        for pathway in pathways:
            if set(pathway) != {"candidate_id", "events"}:
                return False
            events = pathway.get("events")
            if not isinstance(events, list) or not events or not all(
                isinstance(event, str) and event in self.pathway_event_ids for event in events
            ):
                return False
        return True

    def _valid_uncertainty(self, uncertainty) -> bool:
        if not isinstance(uncertainty, dict) or set(uncertainty) != {
            "remaining_pair_hypotheses", "limitations", "budget_external_confirmation"
        }:
            return False
        remaining = uncertainty["remaining_pair_hypotheses"]
        limitations = uncertainty["limitations"]
        confirmation = uncertainty["budget_external_confirmation"]
        return (
            isinstance(remaining, list)
            and all(isinstance(item, str) for item in remaining)
            and len(set(remaining)) == len(remaining)
            and all(item in self.pairs for item in remaining)
            and isinstance(limitations, list)
            and all(isinstance(item, str) and item for item in limitations)
            and isinstance(confirmation, str)
            and bool(confirmation.strip())
        )

    def submit(self, index, candidate, tool_events, state=None):
        case_id = self.case["case_id"]
        if index != 0 or not self._valid_candidate(candidate):
            return {
                "ok": False,
                "case_id": case_id,
                "status": "invalid",
                "feedback": {"schema_valid": False},
                "cases_remaining": 0,
            }
        candidate_hash = hashlib.sha256(_canonical(candidate).encode()).hexdigest()
        evidence_id = "T19-EVIDENCE-" + candidate_hash[:12]
        events = _attestation_events(tool_events)
        session_id = (state or {}).get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id_missing")
        payload = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "task_id": TASK_ID,
            "case_id": case_id,
            "session_id": session_id,
            "evidence_id": evidence_id,
            "candidate": candidate,
            "candidate_sha256": candidate_hash,
            "tool_events": events,
        }
        attestation = _attest(self.attestation_key, payload)
        root_value = (state or {}).get("evidence_root")
        if root_value:
            evidence = {**payload, "attestation": attestation}
            Path(root_value, "case-1-evidence.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
            )
        measurement_count = sum(
            1 for event in events if isinstance(event, dict) and (event.get("response") or {}).get("ok") is True
        )
        return {
            "ok": True,
            "case_id": case_id,
            "status": "submitted",
            "feedback": {"schema_valid": True, "accepted_measurements": measurement_count},
            "cases_remaining": 0,
            "evidence_id": evidence_id,
            "attestation": attestation,
        }


def create_adapter(snapshot):
    return Adapter(snapshot)

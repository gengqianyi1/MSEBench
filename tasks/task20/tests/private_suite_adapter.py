"""Private sidecar adapter for Task 20. Never mount this file in the Agent workspace."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path


SCHEMAS = {
    "tpo": {"sample": {"spent"}},
    "xps": {"region": {"Pt4f", "S2p"}},
    "bet": {"sample": {"fresh", "spent"}},
    "pulse_chemisorption": {"gas": {"CO", "H2"}},
    "particle_size_test": {"mode": {"tem"}},
    "condition_perturbation": {"change": {"flow_double", "temperature_minus_20C"}},
}
REQUIRED = {"primary_mechanisms", "contributions", "confidence", "claims", "regeneration_plan", "predicted_performance", "uncertainty"}
MECHANISMS = {"coking", "sintering", "sulfur_poisoning", "pore_blockage", "external_mass_transfer"}
EVIDENCE_SCHEMA_VERSION = "chembench-organizer-evidence/1.0"
TASK_ID = "task-20-catalyst-deactivation-diagnosis"
ATTESTATION_PREFIX = "hmac-sha256:"


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _attestation_events(tool_events):
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


def _attest(key, payload):
    if not isinstance(key, str) or len(key) < 32:
        raise ValueError("attestation_key_missing")
    digest = hmac.new(key.encode("utf-8"), _canonical(payload).encode("utf-8"), hashlib.sha256)
    return ATTESTATION_PREFIX + digest.hexdigest()


class Adapter:
    def __init__(self, snapshot):
        private = Path(snapshot) / "private_data"
        self.case = json.loads((private / "suite.json").read_text(encoding="utf-8"))
        self.responses = json.loads((private / "diagnostic_data.json").read_text(encoding="utf-8"))
        reference = json.loads((private / "reference.json").read_text(encoding="utf-8"))
        self.attestation_key = reference.get("attestation_key")
        self.case_count = 1

    def public_case(self, index):
        if index != 0:
            raise IndexError(index)
        return {key: self.case[key] for key in ("case_id", "task", "constraints", "budget", "output", "public_tools")}

    def start_session(self, evidence_dir, token):
        root = Path(evidence_dir) / token if evidence_dir else None
        if root:
            root.mkdir(parents=True, exist_ok=True)
            (root / "session.json").write_text(json.dumps({"task_id": TASK_ID, "case_id": self.case["case_id"], "session_id": token}, indent=2), encoding="utf-8")
        return {"evidence_root": str(root) if root else None, "session_id": token}

    @staticmethod
    def _error(tool, code):
        return {"ok": False, "tool": tool if isinstance(tool, str) else "", "error": {"code": code}}

    def call_tool(self, index, name, arguments, state=None):
        if index != 0 or name not in SCHEMAS:
            return self._error(name, "unknown_tool")
        if not isinstance(arguments, dict) or set(arguments) != set(SCHEMAS[name]):
            return self._error(name, "invalid_arguments")
        if any(
            not isinstance(arguments[key], str) or arguments[key] not in choices
            for key, choices in SCHEMAS[name].items()
        ):
            return self._error(name, "invalid_arguments")
        key = f"{name}|{_canonical(arguments)}"
        if key not in self.responses:
            return self._error(name, "measurement_unavailable")
        return {"ok": True, "tool": name, "call_id": "T20-" + hashlib.sha256(key.encode()).hexdigest()[:12], "result": self.responses[key]}

    @staticmethod
    def _valid_candidate(candidate):
        if not isinstance(candidate, dict) or set(candidate) != REQUIRED:
            return False
        primary = candidate.get("primary_mechanisms")
        if not isinstance(primary, list) or len(primary) != 2 or len(set(primary)) != 2 or any(item not in MECHANISMS for item in primary):
            return False
        contributions = candidate.get("contributions")
        if not isinstance(contributions, dict) or set(contributions) != MECHANISMS:
            return False
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1 for value in contributions.values()):
            return False
        if abs(sum(contributions.values()) - 1.0) > 1e-6:
            return False
        confidence = candidate.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            return False
        for claim in candidate.get("claims", []):
            if not isinstance(claim, dict) or set(claim) != {"claim_id", "evidence_ids", "relation", "mechanism_ids"}:
                return False
            if not isinstance(claim["claim_id"], str) or not claim["claim_id"] or claim["relation"] not in {"supports", "contradicts", "neutral"}:
                return False
            if not isinstance(claim["evidence_ids"], list) or not claim["evidence_ids"] or any(not isinstance(item, str) or not item for item in claim["evidence_ids"]):
                return False
            if not isinstance(claim["mechanism_ids"], list) or not claim["mechanism_ids"] or any(item not in MECHANISMS for item in claim["mechanism_ids"]):
                return False
        for action in candidate.get("regeneration_plan", []):
            if not isinstance(action, dict) or set(action) != {"action_id", "temperature_C", "atmosphere", "oxygen_percent", "duration_h", "ramp_C_min", "safety_controls"}:
                return False
            if not isinstance(action["action_id"], str) or not action["action_id"] or not isinstance(action["atmosphere"], str) or not action["atmosphere"]:
                return False
            if any(not isinstance(action[key], (int, float)) or isinstance(action[key], bool) for key in ("temperature_C", "oxygen_percent", "duration_h", "ramp_C_min")):
                return False
            controls = action["safety_controls"]
            if not isinstance(controls, list) or not controls or not all(isinstance(item, str) and item for item in controls):
                return False
        performance = candidate.get("predicted_performance")
        if not isinstance(performance, dict) or set(performance) != {"conversion", "selectivity"}:
            return False
        for interval in performance.values():
            if not isinstance(interval, list) or len(interval) != 2 or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in interval) or not 0 <= interval[0] <= interval[1] <= 1:
                return False
        uncertainty = candidate.get("uncertainty")
        if not isinstance(uncertainty, dict) or set(uncertainty) != {"alternatives", "limitations"}:
            return False
        return all(isinstance(values, list) and all(isinstance(item, str) for item in values) for values in uncertainty.values())

    def submit(self, index, candidate, tool_events, state=None):
        case_id = self.case["case_id"]
        if index != 0 or not self._valid_candidate(candidate):
            return {"ok": False, "case_id": case_id, "status": "invalid", "feedback": {"schema_valid": False}, "cases_remaining": 0}
        candidate_hash = hashlib.sha256(_canonical(candidate).encode()).hexdigest()
        evidence_id = "T20-EVIDENCE-" + candidate_hash[:12]
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
            value = {**payload, "attestation": attestation}
            Path(root_value, "case-1-evidence.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        accepted = sum(1 for event in events if event["response"].get("ok") is True)
        return {"ok": True, "case_id": case_id, "status": "submitted", "feedback": {"schema_valid": True, "accepted_measurements": accepted}, "cases_remaining": 0, "evidence_id": evidence_id, "attestation": attestation}


def create_adapter(snapshot):
    return Adapter(snapshot)

"""Private stateful adapter for the frozen five-case QED suite.

The adapter exposes only public case fields and public RDKit results. The
complete action/result stream and terminal candidate are written to organizer
evidence so the protocol grader never trusts model-writable tool results.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


class Adapter:
    def __init__(self, snapshot: Path):
        self.runtime = Path(snapshot) / "private_data" / "runtime_package"
        sys.path.insert(0, str(self.runtime))
        from framework.agent_runner import TOOL_DEFINITIONS
        from tools.rdkit_tools import TOOL_REGISTRY

        self.tools = TOOL_REGISTRY
        self.public_tools = TOOL_DEFINITIONS
        suite = yaml.safe_load(
            (self.runtime / "organizer_private" / "suite.yaml").read_text(encoding="utf-8")
        )
        self.paths = [self.runtime / relative for relative in suite["cases"]]
        self.configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in self.paths]
        self.case_count = len(self.configs)

    def start_session(self, evidence_dir: Path | None, token: str) -> dict[str, Any]:
        return {
            "evidence_dir": Path(evidence_dir) if evidence_dir else None,
            "token": token,
        }

    def public_case(self, index: int) -> dict[str, Any]:
        cfg = self.configs[index]
        return {
            "case_id": cfg["case"]["id"],
            "task": cfg["task"],
            "constraints": cfg["constraints"],
            "budget": cfg["budget"],
            "output": cfg["output"],
            "public_tools": self.public_tools,
        }

    def call_tool(
        self,
        index: int,
        name: Any,
        arguments: Any,
        _state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if name not in self.tools or not isinstance(arguments, dict):
            return {"ok": False, "error": "invalid_tool"}
        function = self.tools[name]
        kwargs = dict(arguments)
        if "config" in inspect.signature(function).parameters:
            kwargs["config"] = self.configs[index]
        try:
            result = function(**kwargs)
        except (TypeError, ValueError):
            return {"ok": False, "tool": name, "error": "invalid_arguments"}
        return {"ok": True, "tool": name, "result": result}

    def submit(
        self,
        index: int,
        candidate: Any,
        tool_events: list[dict[str, Any]],
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self.configs[index]
        case_id = cfg["case"]["id"]
        remaining = self.case_count - index - 1
        required = {"name", "smiles", "rationale"}
        if (
            not isinstance(candidate, dict)
            or set(candidate) != required
            or not all(isinstance(candidate.get(key), str) and candidate[key].strip() for key in required)
        ):
            return {
                "ok": False,
                "case_id": case_id,
                "status": "invalid",
                "feedback": {"valid": False},
                "cases_remaining": remaining,
            }

        result = self.tools["evaluate_candidate"](candidate["smiles"], cfg)
        feedback = {
            key: result.get(key)
            for key in (
                "valid",
                "canonical_smiles",
                "qed",
                "properties",
                "constraint_flags",
                "violations",
                "feasible",
            )
            if key in result
        }
        token = state["token"] if state else "missing-session"
        evidence_id = f"{token}/{case_id}"
        evidence_dir = state.get("evidence_dir") if state else None
        if evidence_dir is not None:
            target = Path(evidence_dir) / token / case_id
            if target.exists():
                raise ValueError("organizer evidence already exists")
            target.mkdir(parents=True)
            payload = {
                "schema_version": "task07-organizer-evidence/1.0",
                "session_id": token,
                "case_index": index + 1,
                "case_id": case_id,
                "candidate": candidate,
                "tool_events": tool_events,
                "submission_feedback": feedback,
                "provider_status": "HARBOR_PROTOCOL",
                "requested_model_id": os.getenv("CHEMBENCH_MODEL_ID"),
                "actual_model_id": os.getenv("CHEMBENCH_ACTUAL_MODEL_ID"),
            }
            payload["exact_model_identity"] = bool(
                payload["requested_model_id"]
                and payload["actual_model_id"] == payload["requested_model_id"]
            )
            (target / "evidence.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        return {
            "ok": True,
            "case_id": case_id,
            "status": "submitted",
            "feedback": feedback,
            "cases_remaining": remaining,
            "evidence_id": evidence_id,
        }


def create_adapter(snapshot: Path) -> Adapter:
    return Adapter(snapshot)

"""Organizer-owned workspace protocol and accounting, independent of scientific scoring."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema


def load_adapter(path, snapshot):
    spec = importlib.util.spec_from_file_location("workspace_private_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_adapter(snapshot)


class Protocol:
    def __init__(self, adapter, evidence_root):
        self.adapter = adapter
        self.root = Path(evidence_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.case = adapter.public_case(0)
        if set(self.case) - {
            "case_id",
            "task",
            "constraints",
            "budget",
            "output",
            "public_tools",
        }:
            raise RuntimeError("PRIVATE_CASE_CONTRACT_VIOLATION")
        self.sessions = {}
        self.lock = threading.RLock()
        self.tools = {}
        for item in self.case["public_tools"]:
            item = item.get("function", item)
            self.tools[item["name"]] = item.get(
                "parameters", item.get("input_schema", {})
            )

    def phase(self, s, *, between_turns=False):
        budget = self.case["budget"]
        elapsed = time.monotonic() - s["started"]
        scientific = budget.get("scientific_turns", budget["max_agent_turns"])
        reason = None
        if between_turns and s["scientific_turns"] >= scientific:
            reason = "scientific_turn_limit"
        elif elapsed >= budget.get("wall_time_seconds", 10800):
            reason = "scientific_wall_time_limit"
        elif s["calls"] >= budget["max_tool_calls"]:
            reason = "tool_budget_exhausted"
        elif hasattr(
            self.adapter, "compute_budget_exhausted"
        ) and self.adapter.compute_budget_exhausted(s["adapter_state"]):
            reason = "compute_budget_exhausted"
        if (
            reason
            and budget.get("finalization_turns", 0)
            and s["phase"] == "scientific"
        ):
            s["phase"] = "finalization"
            s["trigger"] = reason
            s["finalization_started"] = time.monotonic()
        return s["phase"]

    def metrics(self, s, reason):
        return {
            "agent_turns_used": s["turns"],
            "scientific_turns_used": s["scientific_turns"],
            "finalization_turns_used": s["finalization_turns"],
            "tool_calls_used": s["calls"],
            "termination_reason": reason,
        }

    def finalize(self, s, candidate, reason):
        if s["terminal"]:
            return s["terminal"]
        s["adapter_state"]["organizer_accounting"] = {
            **self.metrics(s, reason),
            "provider_attempts": s["turns"],
            "elapsed_seconds": time.monotonic() - s["started"],
            "rejected_requests": s["rejected"],
            "finalization_tool_calls": s["finalization_calls"],
            "exhaustion_trigger": s["trigger"],
            "phase": s["phase"],
        }
        result = self.adapter.submit(0, candidate, s["events"], s["adapter_state"])
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError("ADAPTER_SUBMISSION_FAILED")
        case_root = self.root / result["evidence_id"]
        if self.root.resolve() not in case_root.resolve().parents:
            raise RuntimeError("EVIDENCE_PATH_ESCAPE")
        # Runtime atomic writes may be mode 0600 and owned by container root.
        # The organizer must be able to transfer the sealed snapshot; private access
        # is controlled by the enclosing host trial directory (0700).
        owner = self.root.stat()
        for path in [case_root, *case_root.rglob("*")]:
            if path.is_symlink():
                raise RuntimeError("ORGANIZER_EVIDENCE_SYMLINK")
            path.chmod(0o755 if path.is_dir() else 0o644)
            # Windows source-level tests have no geteuid(); Linux containers keep
            # the ownership normalization used by the organizer evidence mount.
            if getattr(os, "geteuid", lambda: -1)() == 0:
                os.chown(path, owner.st_uid, owner.st_gid)
        output = {}
        roots = [
            case_root / "run" / "accepted_submission",
            case_root / "run" / "submission",
            case_root / "submission",
        ]
        for name in self.case["output"]["required_files"]:
            for root in roots:
                p = root / name
                if (
                    p.is_file()
                    and not p.is_symlink()
                    and root.resolve() in p.resolve().parents
                ):
                    output[name] = base64.b64encode(p.read_bytes()).decode()
                    break
        artifact = {
            "case_id": self.case["case_id"],
            "public_case": self.case,
            "submission": result,
            "agent_metrics": self.metrics(s, reason),
            "organizer_metadata": {
                "attempt_id": s["attempt_id"],
                "requested_model_id": s["requested_model_id"],
                "actual_model_ids": s["actual_model_ids"],
                "provider_attempts": s["turns"],
                "rejected_requests": s["rejected"],
                "elapsed_seconds": time.monotonic() - s["started"],
            },
        }
        (case_root / "participant_record.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        (case_root / "participant_events.json").write_text(
            json.dumps(s["events"], indent=2), encoding="utf-8"
        )
        (self.root / "current.json").write_text(
            json.dumps(artifact, indent=2), encoding="utf-8"
        )
        s["terminal"] = {
            "ok": True,
            "terminal": True,
            "evidence_id": result["evidence_id"],
            "termination_reason": reason,
            "public_artifacts": output,
        }
        return s["terminal"]

    def dispatch(self, route, body):
        with self.lock:
            if route == "/session":
                if self.sessions:
                    return {"ok": False, "error": "TRIAL_SESSION_ALREADY_STARTED"}
                if not all(
                    isinstance(body.get(k), str) and body[k].strip()
                    for k in ["attempt_id", "requested_model_id"]
                ):
                    return {"ok": False, "error": "MISSING_ORGANIZER_METADATA"}
                sid = secrets.token_urlsafe(24)
                s = {
                    "turns": 0,
                    "scientific_turns": 0,
                    "finalization_turns": 0,
                    "finalization_calls": 0,
                    "phase": "scientific",
                    "trigger": None,
                    "finalization_started": None,
                    "calls": 0,
                    "events": [],
                    "rejected": 0,
                    "actual_model_ids": [],
                    "attempt_id": body["attempt_id"],
                    "requested_model_id": body["requested_model_id"],
                    "started": time.monotonic(),
                    "terminal": None,
                    "adapter_state": self.adapter.start_session(self.root, sid),
                }
                self.sessions[sid] = s
                return {"ok": True, "session_id": sid, "case": self.case}
            s = self.sessions.get(body.get("session_id"))
            if s is None:
                return {"ok": False, "error": "UNKNOWN_SESSION"}
            if s["terminal"]:
                return {"ok": False, "error": "SESSION_TERMINAL"}
            budget = self.case["budget"]
            if route == "/turn":
                phase = self.phase(s, between_turns=True)
                elapsed = time.monotonic() - s["started"]
                if (
                    s["turns"] >= budget["max_agent_turns"]
                    or (
                        phase == "finalization"
                        and s["finalization_turns"]
                        >= budget.get("finalization_turns", 0)
                    )
                    or (
                        phase == "finalization"
                        and time.monotonic() - s["finalization_started"]
                        >= budget.get("finalization_wall_time_seconds", 1800)
                    )
                    or (
                        phase == "scientific"
                        and elapsed >= budget.get("wall_time_seconds", 10800)
                    )
                ):
                    return {"ok": False, "error": "TURN_OR_WALL_BUDGET_EXHAUSTED"}
                s["turns"] += 1
                s[phase + "_turns"] += 1
                wall_remaining = (
                    budget.get("finalization_wall_time_seconds", 1800)
                    - (time.monotonic() - s["finalization_started"])
                    if phase == "finalization"
                    else budget.get("wall_time_seconds", 10800) - elapsed
                )
                return {
                    "ok": True,
                    "turn": s["turns"],
                    "phase": phase,
                    "exhaustion_trigger": s["trigger"],
                    "turns_remaining": (
                        budget["max_agent_turns"] - s["turns"]
                        if phase == "scientific"
                        else budget["finalization_turns"] - s["finalization_turns"]
                    ),
                    "tool_calls_remaining": max(
                        0, budget["max_tool_calls"] - s["calls"]
                    ),
                    "wall_seconds_remaining": max(0, int(wall_remaining)),
                }
            if route == "/provider":
                actual = body.get("actual_model_id")
                if not isinstance(actual, str) or not actual:
                    return {"ok": False, "error": "INVALID_PROVIDER_MODEL_ID"}
                s["actual_model_ids"].append(actual)
                s["events"].append(
                    {
                        "turn": s["turns"],
                        "provider_actual_model_id": actual,
                        "assistant_content": body.get("assistant_content"),
                    }
                )
                return {"ok": True}
            if route == "/finalize":
                reason = body.get("reason")
                if reason not in {"agent_turn_limit", "provider_or_agent_failure"}:
                    return {"ok": False, "error": "INVALID_TERMINATION_REASON"}
                return self.finalize(s, {"status": "partial"}, reason)
            if route == "/tool":
                body = {
                    "action": {
                        "tool_calls": [
                            {
                                "tool": body.get("tool"),
                                "arguments": body.get("arguments"),
                            }
                        ]
                    }
                }
                route = "/actions"
            elif route == "/submit":
                body = {
                    "action": {
                        "tool_calls": [
                            {"tool": "submit", "arguments": body.get("candidate")}
                        ]
                    }
                }
                route = "/actions"
            if route != "/actions" or s["turns"] == 0:
                return {"ok": False, "error": "INVALID_PROTOCOL_TRANSITION"}
            action = body.get("action")
            calls = action.get("tool_calls") if isinstance(action, dict) else None
            if not isinstance(calls, list) or not calls or len(calls) > 1024:
                s["events"].append(
                    {"turn": s["turns"], "error": "INVALID_TOOL_REQUEST"}
                )
                return {"ok": False, "error": "INVALID_TOOL_REQUEST"}
            results = []
            for call in calls:
                current_phase = self.phase(s)
                deadline = (
                    s["finalization_started"]
                    + budget.get("finalization_wall_time_seconds", 1800)
                    if current_phase == "finalization"
                    else s["started"] + budget.get("wall_time_seconds", 10800)
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.finalize(
                        s, {"status": "partial"}, current_phase + "_wall_time_limit"
                    )
                s["adapter_state"]["remaining_wall_seconds"] = max(1, int(remaining))
                name = call.get("tool") if isinstance(call, dict) else None
                arguments = call.get("arguments") if isinstance(call, dict) else None
                if s["calls"] >= budget["max_tool_calls"]:
                    # Terminal finalization preserves the accepted evidence; it cannot run a tool.
                    if name == "submit":
                        return self.finalize(
                            s, {"status": "partial"}, "tool_budget_exhausted"
                        )
                    response = {"ok": False, "error": "TOOL_BUDGET_EXHAUSTED"}
                else:
                    s["calls"] += 1
                    if current_phase == "finalization":
                        s["finalization_calls"] += 1
                    try:
                        if not isinstance(name, str) or name not in self.tools:
                            raise ValueError("INVALID_TOOL_REQUEST")
                        jsonschema.validate(arguments, self.tools[name])
                        if current_phase == "finalization" and name == "xtb_run":
                            raise ValueError("FINALIZATION_COMPUTE_DISABLED")
                        if name == "submit":
                            s["events"].append(
                                {
                                    "turn": s["turns"],
                                    "tool": name,
                                    "arguments": arguments,
                                }
                            )
                            return self.finalize(s, arguments, "submitted")
                        response = self.adapter.call_tool(
                            0, name, arguments, s["adapter_state"]
                        )
                        if not isinstance(response, dict) or set(response) - {
                            "ok",
                            "tool",
                            "call_id",
                            "result",
                            "error",
                        }:
                            raise RuntimeError("PRIVATE_RESPONSE_CONTRACT_VIOLATION")
                    except jsonschema.ValidationError:
                        response = {"ok": False, "error": "INVALID_TOOL_ARGUMENTS"}
                    except ValueError as exc:
                        response = {"ok": False, "error": str(exc)}
                if (
                    name == "budget_status"
                    and response.get("ok")
                    and isinstance(response.get("result"), dict)
                ):
                    response["result"].update(self.metrics(s, "running"))
                    response["result"]["tool_calls_remaining"] = max(
                        0, budget["max_tool_calls"] - s["calls"]
                    )
                if not response.get("ok"):
                    s["rejected"] += 1
                s["events"].append(
                    {
                        "turn": s["turns"],
                        "tool": name,
                        "arguments": arguments,
                        "response": response,
                    }
                )
                results.append(response)
            return {"ok": True, "results": results}


def handler(protocol):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                200, {"ok": True, "task_id": protocol.case["task"]["id"]}
            ) if self.path == "/health" else self.reply(404, {"ok": False})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4_194_304:
                    raise ValueError("INVALID_REQUEST_SIZE")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise TypeError("INVALID_REQUEST_BODY")
                self.reply(200, protocol.dispatch(self.path, body))
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"ok": False, "error": type(exc).__name__})

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    protocol = Protocol(load_adapter(args.adapter, args.snapshot), args.evidence_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler(protocol))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

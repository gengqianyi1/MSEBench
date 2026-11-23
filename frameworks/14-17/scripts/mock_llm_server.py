#!/usr/bin/env python3
"""Deterministic stand-in for an OpenAI-compatible chat-completions endpoint, for tasks 14-17.

WHY THIS EXISTS
---------------
It drives the REAL Harbor LLM path -- ChemBenchAgent in the runner process, `exec` into the task
container, the 16/17 private sidecar, the per-round setup scripts and the separate verifiers --
without a model account. Every earlier agent check used a fake environment on the author machine and
passed while the real Harbor LLM path was broken on all four tasks (organizer DeepSeek run,
2026-09-16: 14/15 lost their round files, 16/17 could not reach the sidecar).

It replays each round's shipped solution candidate, so a correct pipeline reproduces the Oracle reward.

SCENARIOS
---------
  normal         a tool call first where the protocol has tools, then the round's solution candidate
  large          as normal, every reply prefixed with ~150 KB of prose, like a long-reasoning model;
                 the agent's saved conversation then exceeds Linux's 128 KB single-argument limit
  invalid-round  round 2 answers with prose and no JSON: a genuine model failure, which must be
                 scored (incomplete -> reward 0), not crash the trial

  python scripts/mock_llm_server.py --package . --task <task-dir> --scenario normal --port-file F
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCENARIOS = ("normal", "large", "invalid-round")
# task -> (tool the model calls first, argument key); task 14's protocol has no tools
TOOL_PROTOCOL = {
    "task-15-lipophilic-sulfonamide-edit-budget": ("validate_structure", "smiles"),
    "task-16-tablet-microenvironment-cqa": ("validate_formulation", "formulation"),
    "task-17-spherical-agglomeration": ("check_process_legality", "process"),
}
# ~150 KB and no braces, so the agent's JSON extraction still finds exactly the object that follows
PAD = "Working notes: weighing each descriptor window against the edit budget before committing. " * 1650


def round_number(messages: list) -> int:
    """The round being asked about: the highest round the latest round-naming user message mentions.

    Rounds 2-5 of tasks 14 and 15 name the previous round first ("feedback for your round 1
    candidate ..."), so the first mention is the wrong one.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            numbers = [int(n) for n in re.findall(r"round\s+(\d+)", str(message.get("content", "")), re.I)]
            if numbers:
                return max(numbers)
    return 1


def reply(package: Path, task: str, scenario: str, messages: list) -> tuple[str, int, str]:
    n = round_number(messages)
    solution = package / "tasks" / task / "steps" / f"round-{n}" / "solution" / "candidate.json"
    candidate = json.loads(solution.read_text(encoding="utf-8"))["candidate"]
    last = messages[-1] if messages else {}
    after_tools = last.get("role") == "user" and str(last.get("content", "")).lstrip().startswith('{"tool_')
    if scenario == "invalid-round" and n == 2:
        body, kind = "I need more information before I can propose a candidate this round.", "no_json"
    elif task in TOOL_PROTOCOL and not after_tools:
        tool, key = TOOL_PROTOCOL[task]
        body, kind = json.dumps({"tool_calls": [{"tool": tool, "arguments": {key: candidate[key]}}]}), "tool_calls"
    else:
        body, kind = json.dumps({"candidate": candidate}), "candidate"
    if scenario == "large":
        body = PAD + "\n" + body
    return body, n, kind


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        server = self.server
        try:
            request = json.loads(raw)
            messages = request.get("messages") or []
            text, n, kind = reply(server.package, server.task, server.scenario, messages)
        except Exception as exc:  # noqa: BLE001 -- report, do not hang the agent
            body = json.dumps({"error": {"message": f"mock server: {type(exc).__name__}: {exc}"}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(json.dumps({"t": round(time.time(), 3), "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            return
        response = {
            "id": f"mock-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": request.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(raw) // 4, "completion_tokens": len(text) // 4,
                      "total_tokens": (len(raw) + len(text)) // 4},
        }
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(json.dumps({"t": round(time.time(), 3), "task": server.task, "scenario": server.scenario,
                          "round": n, "reply": kind, "messages": len(messages),
                          "request_bytes": len(raw), "reply_bytes": len(text)}), flush=True)

    def log_message(self, *_args) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--scenario", choices=SCENARIOS, default="normal")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--port-file", type=Path)
    args = ap.parse_args()
    if not (args.package / "tasks" / args.task / "task.toml").is_file():
        raise SystemExit(f"no such task: {args.task}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.package, server.task, server.scenario = args.package.resolve(), args.task, args.scenario
    port = server.server_address[1]
    if args.port_file:
        args.port_file.write_text(str(port), encoding="utf-8")
    print(json.dumps({"listening": f"http://{args.host}:{port}/v1", "task": args.task, "scenario": args.scenario}),
          flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

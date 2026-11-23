#!/usr/bin/env python3
"""Regression: a sidecar submit error must return JSON and retain artifacts.

Uses a synthetic adapter only; it never starts Task 01 or Task 02.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from agents.chembench_agent import MSEBenchAgent

ROOT = Path(__file__).resolve().parents[1]


def post(url: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="chembench-sidecar-regression-") as raw:
        base = Path(raw)
        adapter = base / "adapter.py"
        adapter.write_text('''from pathlib import Path
import tempfile

class A:
    case_count = 1
    def start_session(self, evidence, token):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name) / "workspace"
        root.mkdir()
        (root / "result.json").write_text('{"ok":true}')
        return {"temp": temp, "root": root}
    def public_case(self, index):
        return {"case_id":"synthetic","task":{},"constraints":{},"budget":{},"output":{},"public_tools":[]}
    def export_paths(self, state):
        return ["result.json"]
    def call_tool(self, *args):
        return {"ok":True,"tool":"noop","call_id":"x","result":{},"budget":{}}
    def submit(self, *args):
        raise OSError("synthetic export failure")

def create_adapter(snapshot):
    return A()
''', encoding="utf-8")
        evidence = base / "evidence"
        port = 18991
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "templates/sidecar/hidden_suite_server.py"), "--adapter", str(adapter), "--snapshot", str(base), "--evidence-dir", str(evidence), "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            for _ in range(40):
                try:
                    status, session = post(f"http://127.0.0.1:{port}/session", {})
                    if status == 200:
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("sidecar did not start")
            status, result = post(f"http://127.0.0.1:{port}/submit", {"session_id": session["session_id"], "candidate": {}})
            exported = evidence / session["session_id"] / "last_submission" / "workspace" / "result.json"
            assert status == 500, (status, result)
            assert result.get("error") == "sidecar_internal_error", result
            assert exported.read_text(encoding="utf-8") == '{"ok":true}'
            # Exercise the agent-side atomic import with a local environment
            # shim.  This is deliberately synthetic, not a Task 01/02 run.
            workspace = base / "harbor_workspace"
            workspace.mkdir()
            class Result:
                def __init__(self, stdout: str, return_code: int):
                    self.stdout, self.stderr, self.return_code = stdout, "", return_code
            class LocalEnvironment:
                async def exec(self, command, cwd=None, timeout_sec=None):
                    match = re.search(r"base64\.b64decode\('([^']+)'\)", command)
                    if not match:
                        return Result("", 1)
                    source = base64.b64decode(match.group(1)).decode()
                    source = source.replace("Path('/workspace')", f"Path({str(workspace)!r})")
                    source = source.replace("Path('/organizer_evidence')", f"Path({str(evidence)!r})")
                    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
                    return Result(completed.stdout, completed.returncode)
            agent = object.__new__(MSEBenchAgent)
            imported = asyncio.run(agent._import_sidecar_artifacts(LocalEnvironment(), f"{session['session_id']}/last_submission", ["result.json"]))
            assert imported.get("ok") is True, imported
            assert (workspace / "result.json").read_text(encoding="utf-8") == '{"ok":true}'
            print(json.dumps({"status": "PASS", "http_status": status, "artifact_snapshot": str(exported), "artifact_import": imported}))
        finally:
            process.terminate()
            process.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen verifier image against production-built organizer evidence.

This is deliberately a release command as well as the implementation behind the
Docker-gated pytest coverage.  It exercises the same path as Harbor's separate
verifier: an image-owned ``/tests/test.sh`` receives a read-only declared
workspace, including Agent ``protocol_state.json`` and the signed
``/workspace/cases/case-1/case_result.json``. The frozen entrypoint explicitly
grades that case-result path; unrelated declared-workspace metadata is inert.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from task_paths import task_path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCKER_COMMAND_TIMEOUT_SECONDS = 1200
TASKS = {
    "task-19-active-impurity-elucidation": {
        "case_id": "LID-DUAL-OX-H01",
        "calls": (
            ("energy_resolved_msms", {"energy_eV": 45}),
            ("correlation_window", {"window": "aromatic"}),
            ("correlation_window", {"window": "sidechain"}),
            ("condition_perturbation", {"condition": "pH4"}),
        ),
    },
    "task-20-catalyst-deactivation-diagnosis": {
        "case_id": "PDH-DEACT-H01",
        "calls": (
            ("tpo", {"sample": "spent"}),
            ("bet", {"sample": "spent"}),
            ("xps", {"region": "S2p"}),
            ("particle_size_test", {"mode": "tem"}),
        ),
    },
}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _agent_class():
    """Load the production Agent without requiring the Harbor test runtime."""
    harbor = types.ModuleType("harbor")
    harbor_agents = types.ModuleType("harbor.agents")
    harbor_base = types.ModuleType("harbor.agents.base")
    harbor_base.BaseAgent = object
    sys.modules.setdefault("harbor", harbor)
    sys.modules.setdefault("harbor.agents", harbor_agents)
    sys.modules.setdefault("harbor.agents.base", harbor_base)
    from agents.chembench_agent import ChemBenchAgent

    return ChemBenchAgent


class _WorkspaceEnvironment:
    """Minimal Harbor workspace adapter for the Agent's post-submit artifact write."""

    def __init__(self, root: Path):
        self.root = root
        self.case_result = None
        self.protocol_state = None
        self._write_buffers = {}

    async def exec(self, command, **_kwargs):
        if "case_result.json" in command or "open('/workspace/protocol_state.json'" in command:
            match = re.search(r"base64\.b64decode\('([^']+)'\)", command)
            if match is None:
                raise RuntimeError("Agent artifact command is not base64-encoded")
            chunk = base64.b64decode(match.group(1))
            key = "case_result" if "case_result.json" in command else "protocol_state"
            prior = self._write_buffers.get(key, b"") if "'ab'" in command else b""
            payload = prior + chunk
            self._write_buffers[key] = payload
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            if "case_result.json" in command:
                path = self.root / "cases" / "case-1" / "case_result.json"
                self.case_result = value
            else:
                path = self.root / "protocol_state.json"
                self.protocol_state = value
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@contextmanager
def _running_sidecar(tmp_path: Path, task_name: str):
    task = task_path(task_name)
    server_module = _module(
        "docker_entrypoint_sidecar_" + task_name.replace("-", "_"),
        ROOT / "templates" / "sidecar" / "hidden_suite_server.py",
    )
    adapter = server_module._load(
        task / "tests" / "private_suite_adapter.py",
        task / "tests" / "task_snapshot",
    )
    state = server_module.State(adapter, tmp_path / "organizer-evidence")
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def produce_artifact(root: Path, task_name: str) -> dict:
    """Use the real private Adapter, HTTP sidecar, and production Agent builder."""
    task = task_path(task_name)
    details = TASKS[task_name]
    config = json.loads(
        (task / "environment" / "public" / ".chembench_agent.json").read_text(encoding="utf-8")
    )
    # This release check supplies deterministic Oracle model actions locally;
    # keep the production protocol path while preventing any provider/API call.
    config["use_native_tool_calls"] = False
    oracle = json.loads((task / "tests" / "fixtures" / "oracle.json").read_text(encoding="utf-8"))
    tools = (task / "environment" / "public" / "tools.json").read_text(encoding="utf-8")
    environment = _WorkspaceEnvironment(root / "workspace")
    context = SimpleNamespace(metadata={})
    responses = iter(
        (
            json.dumps(
                {
                    "tool_calls": [
                        {"function": {"name": "answer_lookup", "arguments": "not-json"}},
                        *[
                            {"tool": tool, "arguments": arguments}
                            for tool, arguments in details["calls"]
                        ],
                    ]
                }
            ),
            json.dumps({"candidate": oracle}),
        )
    )
    agent_class = _agent_class()
    agent = agent_class.__new__(agent_class)

    async def fixed_read(_environment, path, _limit=30000, **kwargs):
        if path == config.get('candidate_schema_path'):
            return (task / 'environment/public' / Path(path).name).read_text(encoding='utf-8')
        if path == "/workspace/protocol_state.json":
            state_path = environment.root / "protocol_state.json"
            return state_path.read_text(encoding="utf-8") if state_path.is_file() else ""
        if path == "/workspace/tools.json":
            return tools
        return ""

    def model_response(_messages, _temperature, _max_tokens, recovery):
        recovery.update({"finish_reason": "stop", "response_status": 200})
        return next(responses)

    agent._fixed_read = fixed_read
    agent._call_protocol_response = model_response
    agent._call = lambda messages, temperature, max_tokens: model_response(
        messages, temperature, max_tokens, {}
    )
    with _running_sidecar(root, task_name) as base:
        config["sidecar_url"] = base
        config["sidecar_url_env"] = "CHEMBENCH_DOCKER_ENTRYPOINT_SIDECAR"
        original = os.environ.pop(config["sidecar_url_env"], None)
        try:
            asyncio.run(agent._run_hidden_suite_case("case 1", environment, context, config))
        finally:
            if original is not None:
                os.environ[config["sidecar_url_env"]] = original
    if context.metadata.get("candidate_submitted") is not True or environment.case_result is None:
        raise RuntimeError("production Agent did not build an accepted verifier artifact")
    if environment.case_result.get("case_id") != details["case_id"]:
        raise RuntimeError("production artifact has the wrong case identity")
    return environment.case_result


class DockerDaemonUnavailable(RuntimeError):
    """The Docker CLI cannot reach any daemon, so local pytest may skip."""


def _daemon_unreachable(diagnostics: str) -> bool:
    text = diagnostics.casefold()
    if any(marker in text for marker in ("permission denied", "tls", "api version", "context", "configuration error")):
        return False
    return any(
        marker in text
        for marker in (
            "cannot connect to the docker daemon",
            "failed to connect to the docker api",
            "connection refused",
            "dockerdesktoplinuxengine; check if the path is correct",
            "dockerdesktoplinuxengine; open //./pipe",
            "dial unix /var/run/docker.sock: connect: no such file or directory",
        )
    )


def _docker(*arguments: str, timeout: int = DOCKER_COMMAND_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise DockerDaemonUnavailable("Docker CLI is unavailable")
    try:
        status = _docker("version", "--format", "{{.Server.Version}}", timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Docker daemon probe timed out; refusing to treat it as unavailable") from exc
    if status.returncode != 0 or not status.stdout.strip():
        diagnostics = status.stdout + status.stderr
        if _daemon_unreachable(diagnostics):
            raise DockerDaemonUnavailable("Docker daemon is unavailable: " + diagnostics.strip())
        raise RuntimeError("Docker daemon probe failed: " + diagnostics.strip())


def _image_tag(task_name: str) -> str:
    digest = hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:12]
    return f"chembench-frozen-verifier-{digest}:local"


def _build_image(task_name: str) -> str:
    tag = _image_tag(task_name)
    tests_dir = task_path(task_name) / "tests"
    result = _docker("build", "--tag", tag, str(tests_dir))
    if result.returncode != 0:
        raise RuntimeError("Docker verifier image build failed:\n" + result.stdout + result.stderr)
    return tag


def _entrypoint_command(
    image: str,
    workspace: Path,
    logs: Path,
    *,
    platform_name: str | None = None,
    uid: int | None = None,
    gid: int | None = None,
) -> list[str]:
    platform_name = platform_name or os.name
    command = ["run", "--rm", "--network", "none"]
    if platform_name == "posix":
        uid = os.getuid() if uid is None else uid
        gid = os.getgid() if gid is None else gid
        command.extend(("--user", f"{uid}:{gid}"))
    command.extend(
        (
            "--mount",
            f"type=bind,src={workspace.resolve()},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={logs.resolve()},dst=/logs",
            image,
            "/tests/test.sh",
        )
    )
    return command


def _run_entrypoint(image: str, workspace: Path, logs: Path) -> tuple[int, str, str]:
    logs.mkdir(parents=True, exist_ok=True)
    result = _docker(*_entrypoint_command(image, workspace, logs))
    return result.returncode, result.stdout, result.stderr


def _read_result(logs: Path, exit_code: int) -> dict:
    verifier_logs = logs / "verifier"
    result_path = verifier_logs / "grader-result.json"
    reward_path = verifier_logs / "reward.txt"
    if not result_path.is_file() or not reward_path.is_file():
        raise RuntimeError("frozen verifier did not write grader-result.json and reward.txt")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reward = float(reward_path.read_text(encoding="utf-8").strip())
    if not math.isfinite(reward):
        raise RuntimeError("frozen verifier reward is non-finite")
    return {"exit_code": exit_code, "reward": reward, **result}


def _write_artifact(workspace: Path, artifact: dict) -> None:
    case_dir = workspace / "cases" / "case-1"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_result.json").write_text(
        json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
    )


def _write_inert_workspace_metadata(workspace: Path, scenario: str) -> None:
    """Keep scenarios asymmetric without creating an alternate grader input."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "declared-workspace-metadata.json").write_text(
        json.dumps({"scenario": scenario, "not_grader_input": True}), encoding="utf-8"
    )


def _assert_valid(task_name: str, artifact: dict, observed: dict) -> None:
    grader = _module(
        "docker_entrypoint_grader_" + task_name.replace("-", "_"),
        task_path(task_name) / "tests" / "protocol_grader.py",
    )
    reference = json.loads(
        (
            task_path(task_name)
            / "tests"
            / "task_snapshot"
            / "private_data"
            / "reference.json"
        ).read_text(encoding="utf-8")
    )
    try:
        grader._validate_attestation(artifact, reference)
        grader._validate_verifier_bridge(artifact)
    except Exception as exc:
        raise RuntimeError(
            f"locally produced verifier bridge is invalid: {type(exc).__name__}: {exc}; "
            + json.dumps({"submission": artifact.get("submission"), "candidate": artifact.get("candidate")}, sort_keys=True)
        ) from exc
    expected = grader.grade_evidence(artifact, reference)
    if observed["exit_code"] != 0:
        raise RuntimeError("valid frozen verifier entrypoint exited nonzero")
    actual_result = {key: observed.get(key) for key in expected}
    if actual_result != expected:
        raise RuntimeError(
            "frozen verifier result differs from direct grading: "
            + json.dumps({"expected": expected, "actual": actual_result}, sort_keys=True)
        )
    if "ORGANIZER_EVIDENCE_ERROR" in observed["failure_codes"]:
        raise RuntimeError("valid attested artifact was rejected as organizer evidence")
    if not observed["reward"] > 0 or not math.isclose(observed["reward"], expected["score"] / 100.0):
        raise RuntimeError("valid frozen verifier reward is not finite nonzero direct parity")


def _assert_closed(label: str, observed: dict) -> None:
    if observed["reward"] != 0.0 or observed["score"] != 0.0:
        raise RuntimeError(f"{label} artifact did not fail closed with zero reward")
    if observed["failure_codes"] != ["ORGANIZER_EVIDENCE_ERROR"]:
        raise RuntimeError(f"{label} artifact did not record detailed organizer evidence error")
    if not isinstance(observed.get("evidence", {}).get("error_type"), str):
        raise RuntimeError(f"{label} artifact error result omitted error_type detail")


def run_task(task_name: str) -> dict:
    _require_docker()
    image = _build_image(task_name)
    with tempfile.TemporaryDirectory(prefix="chembench-verifier-entrypoint-") as raw:
        root = Path(raw)
        valid_workspace = root / "valid" / "workspace"
        artifact = produce_artifact(root / "valid", task_name)
        _write_artifact(valid_workspace, artifact)
        _write_inert_workspace_metadata(valid_workspace, "valid")
        valid_exit, _, _ = _run_entrypoint(image, valid_workspace, root / "valid" / "logs")
        valid = _read_result(root / "valid" / "logs", valid_exit)
        _assert_valid(task_name, artifact, valid)

        missing_workspace = root / "missing" / "workspace"
        _write_inert_workspace_metadata(missing_workspace, "missing")
        missing_exit, _, _ = _run_entrypoint(image, missing_workspace, root / "missing" / "logs")
        missing = _read_result(root / "missing" / "logs", missing_exit)
        _assert_closed("missing", missing)

        tampered_artifact = copy.deepcopy(artifact)
        tampered_artifact["attestation"] = "hmac-sha256:" + "0" * 64
        tampered_artifact["submission"]["attestation"] = tampered_artifact["attestation"]
        tampered_workspace = root / "tampered" / "workspace"
        _write_artifact(tampered_workspace, tampered_artifact)
        _write_inert_workspace_metadata(tampered_workspace, "tampered")
        tampered_exit, _, _ = _run_entrypoint(image, tampered_workspace, root / "tampered" / "logs")
        tampered = _read_result(root / "tampered" / "logs", tampered_exit)
        _assert_closed("tampered", tampered)

    return {"task": task_name, "valid": valid, "missing": missing, "tampered": tampered}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), action="append", help="Run one task; default runs both")
    parser.add_argument("--output", type=Path, help="Also write the machine-readable summary to this path")
    args = parser.parse_args()
    summaries = []
    failures = []
    for task_name in args.task or tuple(TASKS):
        try:
            summaries.append(run_task(task_name))
        except Exception as exc:
            failures.append({"task": task_name, "error": str(exc)})
    if failures:
        payload = {"results": summaries, "failures": failures}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        return 2
    payload = summaries[0] if len(summaries) == 1 else summaries
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

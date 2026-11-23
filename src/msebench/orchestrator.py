#!/usr/bin/env python3
"""Unified chembench-family orchestrator: drives harbor with the MERGED agent
(common/harness/agent.py) + the workspace_sidecar protocol plugin, reusing the
frozen task package for sidecar compose/verifier/task payloads.

Modes: smoke (scripted provider, offline) | oracle | invalid | model (real LLM).
Run inside the organizer image exactly like the package run.sh does:

  docker run --rm --network host \\
    -v /var/run/docker.sock:/var/run/docker.sock \\
    --mount type=bind,source=$INTEG,target=$INTEG \\
    --mount type=bind,source=$PKG,target=$PKG --workdir $INTEG \\
    --env PYTHONPATH=$INTEG:$PKG --env <llm env...> \\
    $ORGANIZER python $INTEG/common/harness/orchestrator.py \\
      --task 04 --mode smoke

Regression verdict mirrors scripts/run_workspace_harbor.py: exit 0 pass / 2 fail.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

INTEG = Path(os.environ.get("MSEBENCH_INTEG_ROOT",
                            Path(__file__).resolve().parents[2])).resolve()
AGENT_REF = "msebench.agent.unified:MSEBenchAgent"
PROTOCOL_PLUGIN = "msebench.protocols.workspace_sidecar"


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SENSITIVE_ENV = ("CHEMBENCH_API_KEY", "MSEBENCH_API_KEY")


def redact_command(command: list[str]) -> list[str]:
    redacted = []
    for part in command:
        if any(part.startswith(f"{name}=") for name in SENSITIVE_ENV):
            part = part.split("=", 1)[0] + "=***REDACTED***"
        redacted.append(part)
    return redacted


def append_runs_index(record: dict, out: Path, reward, passed: bool, args) -> None:
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": args.task,
        "mode": args.mode,
        "model": os.environ.get("CHEMBENCH_MODEL_ID", "organizer/scripted-fixture"),
        "score": reward,
        "status": "PASS" if passed else "FAIL",
        "evidence": str(out),
    }
    index = args.output_root / "experiments" / "runs.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if os.getuid() == 0:
        subprocess.run(["chmod", "ugo+rwX", str(index)], check=False)
        subprocess.run(["chown", "1000:1000", str(index)], check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["04", "05", "06", "18"], required=True)
    parser.add_argument("--mode", choices=["oracle", "smoke", "invalid", "model"], default="smoke")
    parser.add_argument("--package", type=Path,
                        default=INTEG / "tasks" / f"task{os.environ.get('MB_TASK','04')}")
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("MSEBENCH_RESULTS_ROOT", str(INTEG / "results"))) / "unified")
    args = parser.parse_args()

    args.package = (INTEG / "tasks" / f"task{args.task}").resolve()
    if importlib.metadata.version("harbor") != "0.21.0":
        raise RuntimeError("HARBOR_VERSION_MISMATCH")
    lock = json.loads((args.package / f"deploy/task{args.task}/images.lock.json").read_text())
    for record in lock["images"].values():
        actual = subprocess.check_output(
            ["docker", "image", "inspect", record["name"], "--format", "{{.Id}}"], text=True
        ).strip()
        if actual != record["image_id"]:
            raise RuntimeError("IMAGE_DIGEST_MISMATCH")

    mode = "valid" if args.mode == "smoke" else args.mode
    attempt = f"task{args.task}-{args.mode}-unified-{uuid.uuid4().hex[:12]}"
    out = (args.output_root / f"task{args.task}" / args.mode / attempt).resolve()
    out.mkdir(mode=0o700, parents=True)
    evidence = out / "organizer_evidence"
    evidence.mkdir(mode=0o755)

    task = next((args.package / "tasks").glob(f"task-{args.task}-*"))
    staged = out / task.name
    shutil.copytree(task, staged,
                    ignore=shutil.ignore_patterns("task_snapshot"))
    toml = staged / "task.toml"
    text = toml.read_text().replace(
        "[environment]\n",
        f'[environment]\ndocker_image = "{lock["images"]["environment"]["name"]}"\n',
    ).replace(
        "[verifier.environment]\n",
        f'[verifier.environment]\ndocker_image = "{lock["images"]["verifier"]["name"]}"\n',
    )
    toml.write_text(text)

    side_port, model_port = free_port(), free_port()
    env = dict(os.environ, SIDECAR_IMAGE=lock["images"]["sidecar"]["name"],
               SIDECAR_PORT=str(side_port), EVIDENCE_ROOT=str(evidence))
    compose = ["docker", "compose", "-p", attempt,
               "-f", str(args.package / f"deploy/task{args.task}/compose.sidecar.yaml")]
    cfg = json.loads((task / "environment/public/.chembench_agent.json").read_text())
    endpoint = f"http://127.0.0.1:{side_port}"
    provider = None
    try:
        subprocess.run(compose + ["up", "-d"], env=env, check=True)
        container_id = subprocess.check_output(
            compose + ["ps", "-q", "sidecar"], env=env, text=True).strip()
        networks = json.loads(subprocess.check_output(
            ["docker", "inspect", container_id, "--format",
             "{{json .NetworkSettings.Networks}}"], text=True))
        address = networks[attempt + "_isolated"]["IPAddress"]
        if not address:
            raise RuntimeError("MISSING_ISOLATED_SIDECAR_ADDRESS")
        endpoint = f"http://{address}:8080"
        for _ in range(60):
            try:
                with urllib.request.urlopen(endpoint + "/health", timeout=1) as r:
                    if json.load(r).get("task_id") == cfg["task_id"]:
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("SIDECAR_STARTUP_FAILED")

        agent = AGENT_REF
        agent_env = {cfg["sidecar_url_env"]: endpoint,
                     "CHEMBENCH_ATTEMPT_ID": attempt,
                     "MSEBENCH_PROTOCOL_MODULE": PROTOCOL_PLUGIN}
        if args.mode == "oracle":
            agent = "msebench.agents.sidecar_oracle:SidecarOracleAgent"
            golden_base = task / "tests/task_snapshot/organizer_private"
            golden = golden_base / "oracle_harbor/repetition_01"
            if not golden.exists():
                golden = golden_base / "oracle/repetition_01"
            if not golden.exists():
                raise RuntimeError("MISSING_ORACLE_FIXTURE")
            agent_env.update(CHEMBENCH_ORACLE_SIDECAR_URL=endpoint,
                             CHEMBENCH_ORACLE_EVIDENCE_ROOT=str(evidence),
                             CHEMBENCH_ORACLE_GOLDEN_ROOT=str(golden))
        elif mode in {"valid", "invalid"}:
            with open(out / "fixture_provider.log", "w") as provider_log:
                provider = subprocess.Popen(
                    [sys.executable, str(args.package / "scripts/participant_fixture.py"),
                     "--public", str(task / "environment/public"),
                     "--mode", mode, "--port", str(model_port)],
                    stdout=provider_log, stderr=subprocess.STDOUT)
            agent_env.update(CHEMBENCH_MODEL_ID="organizer/scripted-fixture",
                             CHEMBENCH_API_URL=f"http://127.0.0.1:{model_port}/v1/chat/completions")
        else:
            for name in ["CHEMBENCH_MODEL_ID", "CHEMBENCH_API_URL", "CHEMBENCH_API_KEY"]:
                if not os.environ.get(name):
                    raise RuntimeError(f"MISSING_MODEL_SETTING:{name}")
            for name in ["CHEMBENCH_MODEL_ID", "CHEMBENCH_API_URL", "CHEMBENCH_API_KEY",
                         "CHEMBENCH_MAX_TOKENS", "CHEMBENCH_REQUEST_TIMEOUT_SECONDS",
                         "CHEMBENCH_ENABLE_THINKING", "CHEMBENCH_REASONING_EFFORT",
                         "CHEMBENCH_STREAM", "CHEMBENCH_REQUEST_RETRIES",
                         "CHEMBENCH_TEMPERATURE", "CHEMBENCH_TOP_P"]:
                if os.environ.get(name):
                    agent_env[name] = os.environ[name]

        if os.environ.get("CHEMBENCH_LLM_CALL_LOG"):
            agent_env["CHEMBENCH_LLM_CALL_LOG"] = os.environ["CHEMBENCH_LLM_CALL_LOG"]
        command = ["harbor", "run", "-p", str(staged), "-a", agent,
                   "--job-name", attempt, "--jobs-dir", str(out / "jobs"),
                   "-n", "1", "-k", "1", "--max-retries", "0",
                   "--verifier", "verifiers.workspace_sidecar_verifier:WorkspaceSidecarVerifier"]
        for key, value in agent_env.items():
            command.extend(["--agent-env", f"{key}={value}"])
        for key, value in {"evidence_root": str(evidence),
                           "attempt_id": attempt,
                           "task_id": cfg["task_id"]}.items():
            command.extend(["--verifier-kwarg", f"{key}={value}"])
        record = {"attempt_id": attempt, "mode": args.mode, "agent_import_path": agent,
                   "images": lock["images"], "command": redact_command(command),
                  "orchestrator": "unified/common.harness"}
        (out / "deployment_record.json").write_text(json.dumps(record, indent=2))
        print(shlex.join(redact_command(command)), flush=True)
        run_env = dict(os.environ,
                       PYTHONPATH=str(INTEG / "src") + os.pathsep + str(args.package)
                       + os.pathsep + os.environ.get("PYTHONPATH", ""))
        with open(out / "harbor.log", "w") as log:
            result = subprocess.run(command, cwd=args.package, env=run_env,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        record["harbor_exit_code"] = result.returncode
        (out / "deployment_record.json").write_text(json.dumps(record, indent=2))
        print(f"Harbor exit code: {result.returncode}; artifacts: {out}", flush=True)
        job = json.loads((out / "jobs" / attempt / "result.json").read_text())
        rewards = list((out / "jobs" / attempt).rglob("reward.txt"))
        passed = (result.returncode == 0
                  and job["stats"]["n_completed_trials"] == 1
                  and job["stats"]["n_errored_trials"] == 0
                  and len(rewards) == 1)
        reward = float(rewards[0].read_text()) if len(rewards) == 1 else None
        passed = passed and reward is not None and math.isfinite(reward)
        if args.mode == "oracle":
            passed = passed and reward == 1.0
        elif args.mode == "invalid":
            passed = passed and reward == 0.0
        record.update(regression_passed=bool(passed), reward=reward)
        (out / "deployment_record.json").write_text(json.dumps(record, indent=2))
        append_runs_index(record, out, reward, bool(passed), args)
        print(json.dumps({"mode": args.mode, "reward": reward,
                          "regression_passed": bool(passed)}), flush=True)
        return 0 if passed else 2
    finally:
        if provider is not None:
            provider.terminate()
            provider.wait(timeout=10)
        with open(out / "sidecar.log", "w") as log:
            subprocess.run(compose + ["logs", "--no-color"], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=False)
        subprocess.run(compose + ["down", "--remove-orphans"], env=env, check=True)
        subprocess.run(["chmod", "-R", "ugo+rwX", str(out)], check=False)
        if os.getuid() == 0:
            subprocess.run(["chown", "-R", "1000:1000", str(out)], check=False)


if __name__ == "__main__":
    raise SystemExit(main())

"""One isolated Harbor trial. Fixture modes never contact an external model endpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["04", "05", "06", "18"], required=True)
    parser.add_argument(
        "--mode", choices=["oracle", "valid", "invalid", "model"], default="valid"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--common-agent", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("harbor") != "0.21.0":
        raise RuntimeError("HARBOR_VERSION_MISMATCH")
    lock = json.loads((ROOT / f"deploy/task{args.task}/images.lock.json").read_text())
    for record in lock["images"].values():
        actual = subprocess.check_output(
            ["docker", "image", "inspect", record["name"], "--format", "{{.Id}}"],
            text=True,
        ).strip()
        if actual != record["image_id"]:
            raise RuntimeError("IMAGE_DIGEST_MISMATCH")
    attempt = f"task{args.task}-{args.mode}-{uuid.uuid4().hex[:12]}"
    out = (
        args.output
        or ROOT
        / "validation/harbor_participant"
        / f"task{args.task}"
        / args.mode
        / attempt
    ).resolve()
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    evidence = out / "organizer_evidence"
    evidence.mkdir(mode=0o755)
    task = next((ROOT / "tasks").glob(f"task-{args.task}-*"))
    # Harbor task staging stays private to the organizer. Only environment/public enters the agent image.
    staged = out / task.name
    shutil.copytree(task, staged)
    toml = staged / "task.toml"
    text = toml.read_text().replace(
        "[environment]\n",
        f'[environment]\ndocker_image = "{lock["images"]["environment"]["name"]}"\n',
    )
    text = text.replace(
        "[verifier.environment]\n",
        f'[verifier.environment]\ndocker_image = "{lock["images"]["verifier"]["name"]}"\n',
    )
    toml.write_text(text)
    side_port, model_port = port(), port()
    env = dict(
        os.environ,
        SIDECAR_IMAGE=lock["images"]["sidecar"]["name"],
        SIDECAR_PORT=str(side_port),
        EVIDENCE_ROOT=str(evidence),
    )
    compose = [
        "docker",
        "compose",
        "-p",
        attempt,
        "-f",
        str(ROOT / f"deploy/task{args.task}/compose.sidecar.yaml"),
    ]
    cfg = json.loads((task / "environment/public/.chembench_agent.json").read_text())
    endpoint = f"http://127.0.0.1:{side_port}"
    provider = None
    try:
        subprocess.run(compose + ["up", "-d"], env=env, check=True)
        container_id = subprocess.check_output(
            compose + ["ps", "-q", "sidecar"], env=env, text=True
        ).strip()
        networks = json.loads(
            subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    container_id,
                    "--format",
                    "{{json .NetworkSettings.Networks}}",
                ],
                text=True,
            )
        )
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
        agent = (
            "agents.chembench_agent:MatBenchAgent"
            if args.common_agent
            else cfg["agent_import_path"]
        )
        agent_env = {cfg["sidecar_url_env"]: endpoint, "CHEMBENCH_ATTEMPT_ID": attempt}
        if args.mode == "oracle":
            agent = "agents.sidecar_oracle:SidecarOracleAgent"
            golden_base = task / "tests/task_snapshot/organizer_private"
            golden = golden_base / "oracle_harbor/repetition_01"
            if not golden.exists():
                golden = golden_base / "oracle/repetition_01"
            if not golden.exists():
                raise RuntimeError("MISSING_ORACLE_FIXTURE")
            agent_env.update(
                CHEMBENCH_ORACLE_SIDECAR_URL=endpoint,
                CHEMBENCH_ORACLE_EVIDENCE_ROOT=str(evidence),
                CHEMBENCH_ORACLE_GOLDEN_ROOT=str(golden),
            )
        elif args.mode in {"valid", "invalid"}:
            with open(out / "fixture_provider.log", "w") as provider_log:
                provider = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts/participant_fixture.py"),
                        "--public",
                        str(task / "environment/public"),
                        "--mode",
                        args.mode,
                        "--port",
                        str(model_port),
                    ],
                    stdout=provider_log,
                    stderr=subprocess.STDOUT,
                )
            agent_env.update(
                CHEMBENCH_MODEL_ID="organizer/scripted-fixture",
                CHEMBENCH_API_URL=f"http://127.0.0.1:{model_port}/v1/chat/completions",
            )
        else:
            for name in ["CHEMBENCH_MODEL_ID", "CHEMBENCH_API_URL"]:
                if not os.environ.get(name):
                    raise RuntimeError(f"MISSING_MODEL_SETTING:{name}")
                agent_env[name] = os.environ[name]
        command = [
            "harbor",
            "run",
            "-p",
            str(staged),
            "-a",
            agent,
            "--job-name",
            attempt,
            "--jobs-dir",
            str(out / "jobs"),
            "-n",
            "1",
            "-k",
            "1",
            "--max-retries",
            "0",
            "--verifier",
            "verifiers.workspace_sidecar_verifier:WorkspaceSidecarVerifier",
        ]
        for key, value in agent_env.items():
            command.extend(["--agent-env", f"{key}={value}"])
        for key, value in {
            "evidence_root": str(evidence),
            "attempt_id": attempt,
            "task_id": cfg["task_id"],
        }.items():
            command.extend(["--verifier-kwarg", f"{key}={value}"])
        record = {
            "attempt_id": attempt,
            "instance_id": cfg.get("instance_id", cfg["task_id"]),
            "scientific_problem_id": cfg.get(
                "scientific_problem_id", cfg["task_id"]
            ),
            "mode": args.mode,
            "harbor_version": "0.21.0",
            "agent_import_path": agent,
            "images": lock["images"],
            "sidecar_endpoint_env": cfg["sidecar_url_env"],
            "endpoint": endpoint,
            "private_evidence_transfer": "organizer host -> separate verifier /organizer_evidence",
            "participant_private_mounts": [],
            "network_mode": "no-network",
            "command": command,
            "external_model_api_calls": 0 if args.mode != "model" else None,
        }
        (out / "deployment_record.json").write_text(json.dumps(record, indent=2))
        print(shlex.join(command), flush=True)
        run_env = dict(
            os.environ,
            PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        with open(out / "harbor.log", "w") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=run_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["harbor_exit_code"] = result.returncode
        (out / "deployment_record.json").write_text(json.dumps(record, indent=2))
        print(f"Harbor exit code: {result.returncode}; artifacts: {out}", flush=True)
        job = json.loads((out / "jobs" / attempt / "result.json").read_text())
        rewards = list((out / "jobs" / attempt).rglob("reward.txt"))
        passed = (
            result.returncode == 0
            and job["stats"]["n_completed_trials"] == 1
            and job["stats"]["n_errored_trials"] == 0
            and len(rewards) == 1
        )
        reward = float(rewards[0].read_text()) if len(rewards) == 1 else None
        passed = passed and reward is not None and math.isfinite(reward)
        if args.mode == "oracle":
            passed = passed and reward == 1.0
        elif args.mode == "invalid":
            passed = passed and reward == 0.0
        record.update(regression_passed=bool(passed), reward=reward)
        (out / "deployment_record.json").write_text(json.dumps(record, indent=2))
        return 0 if passed else 2
    finally:
        if provider is not None:
            provider.terminate()
            provider.wait(timeout=10)
        with open(out / "sidecar.log", "w") as log:
            subprocess.run(
                compose + ["logs", "--no-color"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        subprocess.run(compose + ["down", "--remove-orphans"], env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main())

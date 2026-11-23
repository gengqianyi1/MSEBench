#!/usr/bin/env python3
"""Runner for revision6 tasks 10-13 (single_step budget protocol, chembench family).

Modes:
  oracle       replay packaged solution via `harbor run --agent oracle` (offline gate)
  model        real LLM via frozen source agent agents.chembench_agent:ChemBenchAgent
  install-only static validation + image presence check (no LLM)

Runs inside the organizer image (harbor 0.21 + docker CLI):
  docker run --rm --network host -v /var/run/docker.sock:/var/run/docker.sock \
    --mount type=bind,source=$INTEG,target=$INTEG --mount type=bind,source=$RUNS,target=$RUNS \
    --workdir $INTEG --env PYTHONPATH=$FRAMEWORK chembench/organizer:harbor-0.21.0-20260913 \
    python frameworks/10-13/run_task.py --task 13 --mode oracle --output-root $RUNS
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
import uuid
from pathlib import Path

INTEG = Path(os.environ.get("MSEBENCH_INTEG_ROOT", Path(__file__).resolve().parents[2])).resolve()
FRAMEWORK = INTEG / "frameworks/10-13/harbor_agent_harness"
TASK_IDS = {
    "10": "task-10-matbench-bandgap-pipeline",
    "11": "task-11-lithium-electrolyte-design",
    "12": "task-12-pybamm-fast-charge",
    "13": "task-13-retrosynthesis-route",
}
IMAGE_TAGS = {
    n: {"environment": f"chembench-{tid}-env:latest", "verifier": f"chembench-{tid}-tests:latest"}
    for n, tid in TASK_IDS.items()
}
EXPECTED_ORACLE_REWARD = {
    "10": 0.60914312,
    "11": 1.0,
    "12": 0.75,
    "13": 1.0,
}


def image_id(tag: str) -> str | None:
    proc = subprocess.run(["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or None


def load_and_verify_images(task: str) -> None:
    task_lock = INTEG / "images" / f"task{task}" / "images.lock.json"
    if task_lock.is_file():
        lock = json.loads(task_lock.read_text())
        for tar_name, entry in lock["images"].items():
            if image_id(entry["tag"]) != entry["image_id"]:
                tar = task_lock.parent / tar_name
                subprocess.run(["docker", "load", "-i", str(tar)], check=True)
    for tag in IMAGE_TAGS[task].values():
        if image_id(tag) is None:
            raise SystemExit(f"missing image: {tag}; run images/task{task}/ load first")


def static_validation() -> None:
    if not os.getuid() == 0 and os.getenv("SKIP_UID_CHECK"):
        return
    tasks_root = FRAMEWORK / "tasks"
    calls = [
        [sys.executable, str(FRAMEWORK / "scripts/validate_structure.py"), str(tasks_root)],
        [sys.executable, str(FRAMEWORK / "scripts/validate_handoff_10_13.py")],
    ]
    for proc_argv in calls:
        proc = subprocess.run(proc_argv, text=True, capture_output=True)
        if proc.returncode != 0:
            raise SystemExit(f"{script} failed:\n{proc.stdout}\n{proc.stderr}")
        print(proc.stdout.strip()[-300:])


def stage_task(task: str, out: Path) -> Path:
    task_dir = INTEG / "tasks" / f"task{task}"
    tid = TASK_IDS[task]
    staged = out / tid
    shutil.copytree(task_dir, staged, ignore=shutil.ignore_patterns("task.json", "runs"))
    tags = IMAGE_TAGS[task]
    toml = staged / "task.toml"
    text = toml.read_text(encoding="utf-8").replace(
        "[environment]\n",
        f'[environment]\ndocker_image = "{tags["environment"]}"\n',
    ).replace(
        "[verifier.environment]\n",
        f'[verifier.environment]\ndocker_image = "{tags["verifier"]}"\n',
    )
    toml.write_text(text, encoding="utf-8")
    return staged


SENSITIVE_ENV = ("CHEMBENCH_API_KEY", "MSEBENCH_API_KEY")


def redact_command(command: list[str]) -> list[str]:
    return [p.split("=", 1)[0] + "=***REDACTED***"
            if any(p.startswith(f"{n}=") for n in SENSITIVE_ENV) else p
            for p in command]


def agent_env_pairs() -> list[str]:
    pairs = []
    for name in ("CHEMBENCH_API_BASE", "CHEMBENCH_MODEL_ID", "CHEMBENCH_API_KEY",
                 "CHEMBENCH_MAX_TOKENS", "CHEMBENCH_REQUEST_TIMEOUT", "CHEMBENCH_REQUEST_RETRIES",
                 "CHEMBENCH_TEMPERATURE", "CHEMBENCH_TOP_P", "CHEMBENCH_STREAM",
                 "CHEMBENCH_ENABLE_THINKING", "CHEMBENCH_REASONING_EFFORT",
                 "CHEMBENCH_REQUEST_TIMEOUT_SECONDS", "CHEMBENCH_RESPONSE_FORMAT",
                 "CHEMBENCH_THINKING", "CHEMBENCH_USE_NATIVE_TOOL_CALLS",
                 "CHEMBENCH_LLM_CALL_LOG", "MSEBENCH_API_KEY"):
        value = os.environ.get(name)
        if value:
            pairs.append(f"{name}={value}")
    if not any(p.startswith("CHEMBENCH_API_KEY=") for p in pairs) and os.environ.get("MSEBENCH_API_KEY"):
        pairs.append(f"CHEMBENCH_API_KEY={os.environ['MSEBENCH_API_KEY']}")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_IDS))
    parser.add_argument("--mode", choices=["oracle", "model", "install-only"], required=True)
    parser.add_argument("--output-root", type=Path, default=INTEG / "results" / "revision6")
    args = parser.parse_args()

    load_and_verify_images(args.task)
    static_validation()
    if importlib.metadata.version("harbor") != "0.21.0":
        raise RuntimeError("HARBOR_VERSION_MISMATCH")
    if args.mode == "install-only":
        print(json.dumps({"mode": "install-only", "task": args.task, "status": "PASS",
                          "images": IMAGE_TAGS[args.task]}))
        return 0

    attempt = f"task{args.task}-{args.mode}-{uuid.uuid4().hex[:12]}"
    out = (args.output_root / f"task{args.task}" / args.mode / attempt).resolve()
    out.mkdir(mode=0o700, parents=True)
    staged = stage_task(args.task, out)
    jobs_dir = out / "jobs"
    jobs_dir.mkdir()

    command = ["harbor", "run", "--path", str(staged), "--jobs-dir", str(jobs_dir),
               "--job-name", attempt, "--yes"]
    if args.mode == "oracle":
        command += ["--agent", "oracle"]
    else:
        command += ["-a", "agents.chembench_agent:ChemBenchAgent"]
        model = os.environ.get("CHEMBENCH_MODEL_ID")
        if model:
            command += ["--model", model]
        for pair in agent_env_pairs():
            command += ["--agent-env", pair]
    run_env = dict(os.environ,
                   PYTHONPATH=str(FRAMEWORK) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    record = {"attempt_id": attempt, "mode": args.mode, "task": args.task,
              "images": IMAGE_TAGS[args.task], "command": redact_command(command),
              "runner": "revision6"}
    (out / "deployment_record.json").write_text(json.dumps(record, indent=2))
    print(shlex.join(redact_command(command)), flush=True)
    started = time.time()
    with open(out / "harbor.log", "w") as log:
        result = subprocess.run(command, cwd=FRAMEWORK, env=run_env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    record["harbor_exit_code"] = result.returncode
    record["elapsed_sec"] = round(time.time() - started, 1)

    rewards = list(jobs_dir.rglob("reward.txt"))
    reward = float(rewards[0].read_text()) if len(rewards) == 1 else None
    job_result = jobs_dir / attempt / "result.json"
    stats = {}
    if job_result.is_file():
        stats = (json.loads(job_result.read_text()).get("stats") or {})
    passed = (result.returncode == 0 and reward is not None and math.isfinite(reward)
              and stats.get("n_errored_trials", 1) == 0)
    if args.mode == "oracle":
        passed = passed and reward == EXPECTED_ORACLE_REWARD[args.task]
    record.update(regression_passed=bool(passed), reward=reward, stats=stats)
    (out / "deployment_record.json").write_text(json.dumps(record, indent=2))

    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "task": args.task, "mode": args.mode,
             "model": os.environ.get("CHEMBENCH_MODEL_ID", "oracle"),
             "score": reward, "status": "PASS" if passed else "FAIL", "evidence": str(out)}
    index = INTEG / "experiments" / "runs.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    subprocess.run(["chmod", "-R", "ugo+rwX", str(out)], check=False)
    if os.getuid() == 0:
        subprocess.run(["chown", "-R", "1000:1000", str(out)], check=False)
        subprocess.run(["chown", "1000:1000", str(index)], check=False)
    print(json.dumps({"mode": args.mode, "task": args.task, "reward": reward,
                      "regression_passed": bool(passed)}), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

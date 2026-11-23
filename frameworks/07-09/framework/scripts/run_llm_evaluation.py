#!/usr/bin/env python3
"""Run Task 01-03 with an arbitrary OpenAI-compatible LLM profile.

The profile contains model and provider settings, never an API key. The key is
read from the environment or a caller-provided `.env` file and passed to Harbor
through its secret/env-file mechanism.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIDECAR_SERVER = ROOT / "templates" / "sidecar" / "hidden_suite_server.py"

TASKS = {
    "task01": {
        "path": ROOT / "tasks" / "task-01-jarvis-2d-exfoliation-candidate",
        "sidecar_env": "CHEMBENCH_TASK01_SIDECAR_URL",
        "sidecar_url_key": "task01_sidecar_url",
        "image": "chembench/std-task01-verifier:2.1.0",
        "port": 8710,
        "container": "chembench-task01-llm-eval",
    },
    "task02": {
        "path": ROOT / "tasks" / "task-02-oligothiophene-smd-sapt",
        "sidecar_env": "CHEMBENCH_TASK02_SIDECAR_URL",
        "sidecar_url_key": "task02_sidecar_url",
        "image": "chembench/std-task02-verifier:2.1.1",
        "port": 8711,
        "container": "chembench-task02-llm-eval",
    },
    "task03": {
        "path": ROOT / "tasks" / "task-03-five-ring-xtb-dipro-marcus",
        "sidecar_env": None,
        "sidecar_url_key": None,
        "image": None,
        "port": None,
        "container": None,
    },
}

REFERENCE_LLM_CALLS = {"task01": 40, "task02": 60}


def _as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def _runner_python() -> str:
    """Return the interpreter running this launcher (or an explicit override)."""
    return os.environ.get("CHEMBENCH_RUNNER_PYTHON") or sys.executable


def _harbor_executable() -> str:
    """Locate Harbor without pinning the obsolete Windows audit virtualenv."""
    explicit = os.environ.get("CHEMBENCH_HARBOR_BIN")
    if explicit:
        return explicit
    legacy = ROOT / ".audit-venv" / "Scripts" / "harbor.exe"
    if legacy.is_file():
        return str(legacy)
    discovered = shutil.which("harbor")
    if discovered:
        return discovered
    raise RuntimeError(
        "Harbor executable was not found. Activate the Harbor virtual environment "
        "or set CHEMBENCH_HARBOR_BIN to its absolute path."
    )


def _load_profile(path: Path) -> dict:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("model profile must be a YAML object")
    if profile.get("schema_version") != "chembench-llm-profile/1.0":
        raise ValueError("unsupported model profile schema_version")
    required = ["name", "model_name", "api_base_url", "api_key_env", "tasks"]
    missing = [key for key in required if not profile.get(key)]
    if missing:
        raise ValueError(f"missing profile fields: {missing}")
    tasks = profile.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("profile.tasks must be a non-empty list")
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unsupported task: {task}")
    return profile


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_compose(path: Path, evidence_dir: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "main": {
                        "extra_hosts": ["host.docker.internal:host-gateway"],
                        "volumes": [
                            {
                                "type": "bind",
                                "source": _as_posix(evidence_dir),
                                "target": "/organizer_evidence",
                            }
                        ]
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _build_config(
    profile: dict,
    task_key: str,
    task: dict,
    output_dir: Path,
    jobs_dir: Path,
    extra_compose: Path | None,
    engine: str,
) -> Path:
    env = {
        "CHEMBENCH_API_BASE": profile["api_base_url"],
        "CHEMBENCH_API_FORMAT": str(profile.get("api_format") or "openai"),
        "CHEMBENCH_MODEL_ID": profile["model_name"],
        "CHEMBENCH_REQUEST_RETRIES": str(profile.get("request_retries", 5)),
        "CHEMBENCH_REQUEST_TIMEOUT": str(profile.get("request_timeout_sec", 300)),
        "CHEMBENCH_TEMPERATURE": str(profile.get("temperature", 0.0)),
        "CHEMBENCH_MAX_TOKENS": str(profile.get("max_tokens", 36000)),
        "CHEMBENCH_USE_NATIVE_TOOL_CALLS": "1" if profile.get("use_native_tool_calls", True) else "0",
        "CHEMBENCH_LLM_CALL_LOG": str((output_dir / f"{task_key}_llm_calls.jsonl").resolve()),
    }
    if task["sidecar_env"]:
        sidecar_url = str(profile.get(task["sidecar_url_key"], ""))
        if engine == "harbor" and ("127.0.0.1" in sidecar_url or "localhost" in sidecar_url):
            sidecar_url = f"http://host.docker.internal:{task['port']}"
        env[task["sidecar_env"]] = sidecar_url

    config = {
        "job_name": f"{profile['name']}-{task_key}",
        "jobs_dir": _as_posix(jobs_dir),
        "n_concurrent_trials": 1,
        "quiet": True,
        "tasks": [{"path": str(task["path"].relative_to(ROOT)).replace("\\", "/")}],
        "agents": [
            {
                "import_path": "agents.chembench_agent:MSEBenchAgent",
                "model_name": profile["model_name"],
                "env": env,
            }
        ],
        "environment": {
            "type": "docker",
            "force_build": False,
            "delete": False,
        },
    }
    if task_key == "task03":
        config["environment"].update(
            {
                "override_cpus": int(profile.get("task03_cpus", 4)),
                "override_memory_mb": int(profile.get("task03_memory_mb", 12288)),
                "env": {
                    "OMP_NUM_THREADS": str(profile.get("task03_omp_num_threads", "1")),
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            }
        )
    if extra_compose is not None:
        config["environment"]["extra_docker_compose"] = [_as_posix(extra_compose)]

    config_path = output_dir / f"{task_key}.harbor.yaml"
    _write_yaml(config_path, config)
    return config_path


def _ensure_env_file(
    api_key_env: str,
    provided_env_file: Path | None,
    output_dir: Path,
    *,
    allow_missing: bool = False,
) -> Path:
    if provided_env_file is not None:
        if not provided_env_file.is_file():
            raise FileNotFoundError(provided_env_file)
        return provided_env_file.resolve()
    key = os.environ.get(api_key_env)
    if key:
        path = output_dir / ".runtime.env"
        path.write_text(f"{api_key_env}={key}\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path
    if allow_missing:
        # Harbor still accepts an env file for an install-only trial, but no
        # model request is made and no credential should be fabricated.
        path = output_dir / ".install-only.env"
        path.write_text("# no model credential required for --install-only\n", encoding="utf-8")
        return path
    raise RuntimeError(f"environment variable {api_key_env} is required")


def _load_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _runtime_env(
    profile: dict,
    env_file: Path | None,
    output_dir: Path,
    *,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    api_key_env = profile["api_key_env"]
    key_value = os.environ.get(api_key_env)
    if key_value is None and env_file is not None:
        key_value = _load_env_values(env_file).get(api_key_env)
    if not key_value:
        raise RuntimeError(f"environment variable {api_key_env} is required")
    environment[api_key_env] = key_value
    environment["CHEMBENCH_API_KEY"] = key_value
    environment.update(
        {
            "CHEMBENCH_API_BASE": profile["api_base_url"],
        "CHEMBENCH_API_FORMAT": str(profile.get("api_format") or "openai"),
            "CHEMBENCH_MODEL_ID": profile["model_name"],
            "CHEMBENCH_REQUEST_RETRIES": str(profile.get("request_retries", 5)),
            "CHEMBENCH_REQUEST_TIMEOUT": str(profile.get("request_timeout_sec", 300)),
            "CHEMBENCH_TEMPERATURE": str(profile.get("temperature", 0.0)),
            "CHEMBENCH_MAX_TOKENS": str(profile.get("max_tokens", 36000)),
            "CHEMBENCH_USE_NATIVE_TOOL_CALLS": "1" if profile.get("use_native_tool_calls", True) else "0",
            "PYTHONPATH": str(ROOT),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    if extra:
        environment.update(extra)
    return environment


def _summarize_llm_calls(task_key: str, output_dir: Path) -> None:
    source = output_dir / f"{task_key}_llm_calls.jsonl"
    if not source.is_file():
        return
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    successful = [item for item in records if isinstance(item.get("response"), dict)]
    usage_totals: dict[str, int] = {}
    finish_reasons: list[str | None] = []
    for item in successful:
        response = item["response"]
        usage = response.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                usage_totals[key] = usage_totals.get(key, 0) + value
        choices = response.get("choices") or []
        finish_reasons.append(choices[0].get("finish_reason") if choices else None)
    summary = {
        "task": task_key,
        "record_file": source.name,
        "attempt_count": len(records),
        "successful_response_count": len(successful),
        "error_attempt_count": len(records) - len(successful),
        "elapsed_seconds_total": round(sum(float(item.get("elapsed_seconds") or 0) for item in records), 6),
        "finish_reasons": finish_reasons,
        "usage_totals": usage_totals,
    }
    (output_dir / f"{task_key}_llm_call_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_efficiency_metadata(task_key: str, profile: dict, output_dir: Path, run_dir: Path, tool_count: int) -> None:
    """Persist the billable LLM-call count for the private score wrapper."""
    source = output_dir / f"{task_key}_llm_calls.jsonl"
    actual = len([line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]) if source.is_file() else 0
    reference = int(profile.get(f"reference_llm_calls_{task_key}", profile.get("reference_llm_calls", REFERENCE_LLM_CALLS[task_key])))
    efficiency = 1.0 if actual <= reference else reference / actual
    (run_dir / "efficiency.json").write_text(
        json.dumps({
            "metric": "llm_request_efficiency",
            "reference_llm_calls": reference,
            "actual_llm_calls": actual,
            "tool_calls": int(tool_count),
            "efficiency": round(efficiency, 9),
            "formula": "min(1, reference_llm_calls / actual_llm_calls)",
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_direct_hidden_suite(task_key: str, task: dict, profile: dict, output_dir: Path, evidence_dir: Path, env_file: Path | None) -> None:
    workspace = output_dir / f"{task_key}_workspace"
    if workspace.exists():
        import shutil

        shutil.rmtree(workspace)
    environment = _runtime_env(
        profile,
        env_file,
        output_dir,
        extra={
            "CHEMBENCH_TASK_ID": task["path"].name,
            "CHEMBENCH_WORKSPACE": _as_posix(workspace),
            "CHEMBENCH_LLM_CALL_LOG": str((output_dir / f"{task_key}_llm_calls.jsonl").resolve()),
            task["sidecar_env"]: str(profile.get(task["sidecar_url_key"], "")),
        },
    )
    subprocess.run(
        [_runner_python(), str(ROOT / "scripts" / "run_live_hidden_suite.py")],
        cwd=ROOT,
        env=environment,
        text=True,
        check=True,
    )
    _summarize_llm_calls(task_key, output_dir)
    case_result = json.loads((workspace / "cases" / "case-1" / "case_result.json").read_text(encoding="utf-8"))
    submission = case_result.get("submission") or {}
    failure = case_result.get("failure")
    evidence_id = submission.get("evidence_id")
    if not evidence_id:
        status_path = output_dir / f"{task_key}_submission_status.json"
        status = submission.get("status") or failure or "missing_submission"
        if status == "invalid":
            score_basis = "not scored; sidecar rejected the submission because required artifacts were missing or invalid"
        elif status == "sidecar_infrastructure_failure":
            score_basis = "not scored; sidecar infrastructure failed before submission"
        else:
            score_basis = f"not scored; status={status}"
        status_path.write_text(
            json.dumps(
                {
                    "task": task_key,
                    "status": status,
                    "submission_ok": submission.get("ok"),
                    "feedback": submission.get("feedback"),
                    "tool_count": case_result.get("tool_count"),
                    "workspace": str(workspace),
                    "score": None,
                    "score_basis": score_basis,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"task": task_key, "score": None, "status": "invalid_submission", "status_file": str(status_path)}, indent=2))
        return
    evidence_workspace = evidence_dir / evidence_id / "workspace"
    if not evidence_workspace.is_dir():
        raise RuntimeError(f"evidence workspace not found: {evidence_workspace}")
    logs = output_dir / f"{task_key}_verifier_logs"
    run_dir = output_dir / f"{task_key}_verifier_run"
    logs.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_efficiency_metadata(task_key, profile, output_dir, run_dir, int(case_result.get("tool_count") or 0))
    _docker(
        "run",
        "--rm",
        "-v",
        f"{_as_posix(evidence_workspace)}:/workspace",
        "-v",
        f"{_as_posix(logs)}:/logs",
        "-v",
        f"{_as_posix(run_dir)}:/run",
        task["image"],
        "micromamba",
        "run",
        "-n",
        "chembench",
        "python",
        "/tests/run_legacy_grader.py",
    )
    reward = json.loads((logs / "verifier" / "reward.json").read_text(encoding="utf-8"))
    print(json.dumps({"task": task_key, "reward": reward}, indent=2))


def _run_direct_task03(task: dict, profile: dict, output_dir: Path, env_file: Path | None) -> None:
    workspace = output_dir / "task03_workspace"
    if workspace.exists():
        import shutil

        shutil.rmtree(workspace)
    environment = _runtime_env(
        profile,
        env_file,
        output_dir,
        extra={
            "CHEMBENCH_WORKSPACE": _as_posix(workspace),
            "CHEMBENCH_LLM_CALL_LOG": str((output_dir / "task03_llm_calls.jsonl").resolve()),
            "CHEMBENCH_TASK03_CPUS": str(profile.get("task03_cpus", "")),
            "CHEMBENCH_TASK03_MEMORY_MB": str(profile.get("task03_memory_mb", "")),
            "CHEMBENCH_TASK03_OMP_NUM_THREADS": str(profile.get("task03_omp_num_threads", 1)),
            "CHEMBENCH_TASK03_ROUND_TIMEOUT_SEC": str(profile.get("task03_round_timeout_sec", 1800)),
        },
    )
    subprocess.run(
        [_runner_python(), str(ROOT / "scripts" / "run_live_task03.py")],
        cwd=ROOT,
        env=environment,
        text=True,
        check=True,
    )
    _summarize_llm_calls("task03", output_dir)
    run_result = json.loads((workspace / "run_result.json").read_text(encoding="utf-8"))
    if run_result.get("failure") == "final_answer_missing":
        print(json.dumps({"task": "task03", "failure": "final_answer_missing", "workspace": str(workspace)}, indent=2))
        return
    logs = output_dir / "task03_verifier_logs"
    logs.mkdir(parents=True, exist_ok=True)
    _docker(
        "run",
        "--rm",
        "-e",
        "CHEMBENCH_WORKSPACE=/workspace",
        "-e",
        "CHEMBENCH_TASK_ROOT=/task",
        "-v",
        f"{_as_posix(workspace)}:/workspace",
        "-v",
        f"{_as_posix(logs)}:/logs",
        "chembench/std-task03-verifier:2.1.0",
        "micromamba",
        "run",
        "-n",
        "chembench",
        "python",
        "/tests/final_grader.py",
    )
    result = json.loads((workspace / "result.json").read_text(encoding="utf-8"))
    print(json.dumps({"task": "task03", "score": result.get("score"), "n_evaluated": result.get("checks", {}).get("n_evaluated")}, indent=2))


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run Docker with an actionable preflight error instead of a raw traceback."""
    try:
        result = subprocess.run(["docker", *args], text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker CLI is not installed or not on PATH; start Docker Desktop "
            "before running Task01/02 hidden-suite evaluation."
        ) from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if "dockerDesktopLinuxEngine" in detail or "docker daemon" in detail.lower() or "pipe" in detail.lower():
            raise RuntimeError(
                "Docker CLI is present but the Docker daemon is unavailable; "
                "start Docker Desktop and retry. " + detail
            )
        raise subprocess.CalledProcessError(result.returncode, ["docker", *args], result.stdout, result.stderr)
    return result


def _remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], text=True, capture_output=True)


def _start_sidecar(task_key: str, task: dict, evidence_dir: Path, rebuild: bool) -> None:
    _remove_container(task["container"])
    adapter_path = task["path"] / "tests" / "private_suite_adapter.py"
    if rebuild:
        tests_dir = task["path"] / "tests"
        print(f"Rebuilding {task['image']}")
        _docker("build", "-t", task["image"], str(tests_dir))
    host_port = task["port"]
    command = [
        "run",
        "-d",
        # The organizer-evidence directory is a host/WSL bind mount.  The
        # micromamba image defaults to an unprivileged UID that cannot create
        # session directories on all DrvFS mounts, which used to turn a valid
        # submission into a sidecar HTTP 500.  This sidecar is organizer-owned;
        # the participant still only sees the public HTTP tool contract.
        "--user",
        "0:0",
        "--name",
        task["container"],
        "-p",
        f"127.0.0.1:{host_port}:{host_port}",
        "-v",
        f"{_as_posix(SIDECAR_SERVER)}:/sidecar.py:ro",
        "-v",
        f"{_as_posix(adapter_path)}:/tests/private_suite_adapter.py:ro",
        "-v",
        f"{_as_posix(task['path'] / 'tests' / 'task_snapshot')}:/task:ro",
        "-v",
        f"{_as_posix(evidence_dir)}:/organizer_evidence",
        task["image"],
        "micromamba",
        "run",
        "-n",
        "chembench",
        "python",
        "/sidecar.py",
        "--adapter",
        "/tests/private_suite_adapter.py",
        "--snapshot",
        "/task",
        "--evidence-dir",
        "/organizer_evidence",
        "--host",
        "0.0.0.0",
        "--port",
        str(host_port),
    ]
    _docker(*command)


def _wait_for_sidecar(port: int, timeout_seconds: int = 120) -> None:
    import time
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(f"sidecar did not become healthy on port {port}")


def _run_harbor(config_path: Path, env_file: Path, extra_args: list[str]) -> None:
    context = subprocess.check_output(["docker", "context", "show"], text=True).strip()
    command = [
        _harbor_executable(),
        "job",
        "start",
        "--config",
        str(config_path),
        "--env-file",
        str(env_file),
        "--quiet",
        "--yes",
        *extra_args,
    ]
    environment = os.environ.copy()
    environment["DOCKER_CONTEXT"] = context
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["NO_COLOR"] = "1"
    subprocess.run(command, text=True, check=True, env=environment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--task",
        action="append",
        choices=sorted(TASKS),
        help="Run a named task from the profile; repeat to select a subset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Harbor-only environment/Agent setup regression; does not call the model or verifier.",
    )
    parser.add_argument("--rebuild-verifier-images", action="store_true")
    parser.add_argument(
        "--engine",
        choices=["direct", "harbor"],
        default="harbor",
        help="Use Harbor for official evaluation; direct is diagnostic-only.",
    )
    args = parser.parse_args()
    if args.install_only and args.engine != "harbor":
        parser.error("--install-only requires --engine harbor")

    profile = _load_profile(args.profile.resolve())
    selected_tasks = args.task or list(profile["tasks"])
    not_in_profile = [task for task in selected_tasks if task not in profile["tasks"]]
    if not_in_profile:
        parser.error(f"selected task(s) absent from profile: {not_in_profile}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(exist_ok=True)

    rebuild = args.rebuild_verifier_images or bool(profile.get("rebuild_verifier_images", False))
    sidecars: list[str] = []

    for task_key in selected_tasks:
        task = dict(TASKS[task_key])
        if task_key in ("task01", "task02"):
            task["port"] = int(profile.get(f"{task_key}_sidecar_port", task["port"]))
            task["container"] = str(profile.get(f"{task_key}_sidecar_container", task["container"]))
        extra_compose = None
        if task["sidecar_env"]:
            evidence_dir = output_dir / f"evidence-{task_key}"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            extra_compose = output_dir / f"{task_key}-evidence-compose.yaml"
            _write_compose(extra_compose, evidence_dir)
        elif task_key == "task03":
            rounds_dir = output_dir / "task03-rounds"
            rounds_dir.mkdir(parents=True, exist_ok=True)
            final_answer = output_dir / "task03-final-answer.json"
            if not final_answer.exists():
                final_answer.write_text("{}\n", encoding="utf-8")
            extra_compose = output_dir / "task03-workspace-compose.yaml"
            extra_compose.write_text(
                yaml.safe_dump(
                    {
                        "services": {
                            "main": {
                                "volumes": [
                                    {
                                        "type": "bind",
                                        "source": _as_posix(rounds_dir),
                                        "target": "/workspace/rounds",
                                    },
                                    {
                                        "type": "bind",
                                        "source": _as_posix(final_answer),
                                        "target": "/workspace/final_answer.json",
                                    },
                                ]
                            }
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

        config_path = _build_config(
            profile,
            task_key,
            task,
            output_dir,
            jobs_dir,
            extra_compose,
            args.engine,
        )
        print(f"Generated {config_path.relative_to(output_dir)}")

        if args.dry_run:
            continue

        sidecar_started = False
        if task["sidecar_env"] and not args.install_only:
            _start_sidecar(task_key, task, evidence_dir, rebuild)
            _wait_for_sidecar(task["port"])
            sidecar_started = True
            sidecars.append(task_key)

        print(f"Running {task_key}")
        try:
            if args.engine == "harbor":
                env_file = _ensure_env_file(
                    profile["api_key_env"],
                    args.env_file.resolve() if args.env_file else None,
                    output_dir,
                    allow_missing=args.install_only,
                )
                harbor_args = list(profile.get("harbor_extra_args") or [])
                if args.install_only:
                    harbor_args.append("--install-only")
                _run_harbor(
                    config_path,
                    env_file,
                    harbor_args,
                )
            elif task_key == "task03":
                _run_direct_task03(
                    task,
                    profile,
                    output_dir,
                    args.env_file.resolve() if args.env_file else None,
                )
            else:
                _run_direct_hidden_suite(
                    task_key,
                    task,
                    profile,
                    output_dir,
                    evidence_dir,
                    args.env_file.resolve() if args.env_file else None,
                )
        finally:
            if sidecar_started:
                _remove_container(task["container"])

    if args.dry_run:
        print("Dry run complete. No Docker or Harbor execution was performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

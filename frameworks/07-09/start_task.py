#!/usr/bin/env python3
"""Run Task 07, 08 or 09 from the unpacked organizer release."""
from __future__ import annotations

import argparse
import json
import hashlib
import tarfile
from functools import lru_cache
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import math
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parent
FRAMEWORK = ROOT / "framework"
MANIFEST = ROOT / "docker" / "image_manifest.json"
IMAGES = ROOT.parents[1] / "images"
RUNTIME_TAGS = (
    "gogost/gost:offline-pinned-amd64",
    "harbor-prebuilt:harbor-docker-egress-control-sidecar--b1e6760734a9e6f6",
    "harbor-prebuilt:harbor-docker-egress-control-sidecar--38d5fad35d5daec4",
)
RUNTIME_TARS = {
    "gogost/gost:offline-pinned-amd64": "shared_images/gost-offline-pinned-amd64.tar",
    "harbor-prebuilt:harbor-docker-egress-control-sidecar--b1e6760734a9e6f6": "shared_images/egress-control-sidecar-b1e6760734a9e6f6.tar",
    "harbor-prebuilt:harbor-docker-egress-control-sidecar--38d5fad35d5daec4": "shared_images/egress-control-sidecar-38d5fad35d5daec4.tar",
}


def task_tars(task_number: int) -> list[Path]:
    base = IMAGES / f"task{task_number:02d}"
    return [base / f"task{task_number:02d}-{kind}.tar" for kind in ("environment", "sidecar")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tar_integrity(tar_path: Path) -> None:
    lock = json.loads((tar_path.parent / "images.lock.json").read_text(encoding="utf-8"))
    entry = (lock.get("images") or {}).get(tar_path.name)
    if entry is None:
        raise SystemExit(f"no images.lock.json entry for {tar_path}")
    if sha256_file(tar_path) != entry["tar_sha256"]:
        raise SystemExit(f"image tar checksum mismatch: {tar_path.name}")


def tar_image_configs(tar_path: Path):
    with tarfile.open(tar_path) as archive:
        names = set(archive.getnames())
        tags = []
        if "index.json" in names:
            index = json.load(archive.extractfile("index.json"))
            for item in index.get("manifests", []):
                name = (item.get("annotations") or {}).get("io.containerd.image.name")
                if name:
                    tags.append(name.removeprefix("docker.io/").removeprefix("library/"))
        elif "manifest.json" in names:
            index = json.load(archive.extractfile("manifest.json"))
            for item in index:
                tags.extend(item.get("RepoTags") or [])
        for member in sorted(names):
            if member.startswith("blobs/sha256/"):
                raw = archive.extractfile(member).read()
            elif member.endswith(".json") and "/" not in member and member not in ("index.json", "manifest.json", "oci-layout"):
                # docker-save archives keep the config blob as a top-level <sha>.json
                raw = archive.extractfile(member).read()
            else:
                continue
            try:
                doc = json.loads(raw)
            except ValueError:
                continue
            if isinstance(doc, dict) and "rootfs" in doc and "architecture" in doc:
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                for tag in tags:
                    yield tag, (digest, doc)


def command(args: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=FRAMEWORK, env=env, text=True, check=True, capture_output=capture)


def image_exists(ref: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", ref], capture_output=True).returncode == 0


def expected_images(task_number: int) -> tuple[dict, dict]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = {row["role"]: row for row in data["images"] if row.get("task") == task_number}
    return rows["environment"], rows["sidecar"]


@lru_cache(maxsize=None)
def archive_configs(task_number: int) -> dict:
    tars = task_tars(task_number) + [IMAGES / rel for rel in RUNTIME_TARS.values()]
    configs = {}
    for tar_path in tars:
        if not tar_path.is_file():
            raise SystemExit(f"missing image tar: {tar_path}")
        verify_tar_integrity(tar_path)
        configs.update(dict(tar_image_configs(tar_path)))
    return configs


def verify_image(row: dict, task_number: int) -> None:
    tag = row["tag"]
    if not image_exists(tag):
        raise SystemExit(f"offline image/tag is absent: {tag}")
    info = json.loads(command(["docker", "image", "inspect", tag], capture=True).stdout)[0]
    config_digest, config = archive_configs(task_number)[tag]
    # Classic Docker returns config digest; containerd may return manifest digest.
    if info.get('Id') not in (config_digest, row['image_digest']):
        raise SystemExit(f'Local image identity mismatch for {tag}: {info.get("Id")}')
    if info.get('RootFS', {}).get('Layers') != config.get('rootfs', {}).get('diff_ids'):
        raise SystemExit(f'Local image filesystem differs from offline archive: {tag}')
    if info.get('Architecture') != config.get('architecture') or info.get('Os') != config.get('os'):
        raise SystemExit(f'Local image platform differs from offline archive: {tag}')


def load_images(task_number: int) -> tuple[str, str]:
    environment_row, sidecar_row = expected_images(task_number)
    environment_image, sidecar_image = environment_row["tag"], sidecar_row["tag"]
    if not image_exists(environment_image) or not image_exists(sidecar_image) or not all(image_exists(tag) for tag in RUNTIME_TAGS):
        tars = task_tars(task_number) + [IMAGES / rel for rel in RUNTIME_TARS.values()]
        for tar_path in tars:
            if not tar_path.is_file():
                raise SystemExit(f"missing image tar: {tar_path}")
            verify_tar_integrity(tar_path)
            command(["docker", "load", "-i", str(tar_path)])
    for ref in (environment_image, sidecar_image, *RUNTIME_TAGS):
        if not image_exists(ref):
            raise SystemExit(f"image/tag absent after offline load: {ref}")
    verify_image(environment_row, task_number)
    if not os.environ.get("CHEMBENCH_ALLOW_PATCHED_SIDECAR"):
        verify_image(sidecar_row, task_number)
    return environment_image, sidecar_image


def wait_health(port: int) -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise SystemExit(f"sidecar health timeout on 127.0.0.1:{port}")


def start_sidecar(task_number: int, port: int, model: str, image: str, evidence: Path) -> None:
    name = f"matbench-task{task_number:02d}-sidecar"
    # 2026-09-28 fix: never clobber a sidecar another live run is using — refuse instead of rm.
    running = subprocess.run(["docker", "ps", "-q", "--filter", f"name=^{name}$"],
                             capture_output=True, text=True).stdout.split()
    if running:
        raise SystemExit(
            f"{name} is already running (container {running[0]}); another task {task_number} run is in "
            f"progress on port {port}. Wait for it to finish (or stop it) before starting a second run.")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    # 2026-09-28 fix: resolve the host tests/ tree once; mount it at /tests (adapter) and
    # also overlay the server file onto its baked location so server-side fixes apply
    # without rebuilding the sidecar image.
    tests_dir = (next((FRAMEWORK / 'tasks').glob(f'task-{task_number:02d}-*')) / 'tests').resolve()
    command([
        "docker", "run", "-d", "--name", name, "--pull", "never",
        "-p", f"127.0.0.1:{port}:8717",
        "-e", f"CHEMBENCH_MODEL_ID={model}",
        "-e", f"CHEMBENCH_ACTUAL_MODEL_ID={model}",
        "-v", f"{evidence.resolve().as_posix()}:/organizer_evidence",
        "-v", f"{tests_dir.as_posix()}:/tests:rw",
        "-v", f"{(tests_dir / 'hidden_suite_server.py').as_posix()}:/opt/matbench/hidden_suite_server.py:ro",
        image,
    ])
    wait_health(port)


def _dump_sidecar_logs(name: str, log_path) -> None:
    """Persist the sidecar container's stdout/stderr (server prints traceback there) before removal."""
    if not log_path:
        return
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            subprocess.run(["docker", "logs", name], stdout=fh, stderr=subprocess.STDOUT)
    except Exception:
        pass


def install_sidecar_cleanup(task_number: int, log_path=None) -> None:
    """Remove the sidecar container on SIGTERM/SIGINT (timeout or Ctrl-C), not just on normal exit."""
    name = f"matbench-task{task_number:02d}-sidecar"

    def _handler(signum, _frame):
        _dump_sidecar_logs(name, log_path)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        # 2026-09-28 fix: harbor's own launcher containers (env-main / egress / verifier)
        # are not removed on SIGTERM; clean this task's leftovers too.
        ids = subprocess.run(["docker", "ps", "-q", "--filter", f"name=task{task_number:02d}__"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def harbor_executable() -> str:
    value = os.getenv("CHEMBENCH_HARBOR_BIN") or shutil.which("harbor")
    if not value:
        raise SystemExit("Harbor 0.21.0 is missing; install the package with: python -m pip install -e framework")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=(7, 8, 9))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-image-load", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "runtime_config.json").read_text(encoding="utf-8"))
    task_cfg = cfg["tasks"][str(args.task)]
    port = int(os.getenv(f"CHEMBENCH_TASK{args.task:02d}_SIDECAR_PORT", task_cfg["sidecar_port"]))
    sidecar_url = f"http://127.0.0.1:{port}"
    model = os.getenv("CHEMBENCH_MODEL_ID", cfg["model"])
    api_base = os.getenv("CHEMBENCH_API_BASE", cfg["api_base"])
    api_key = os.getenv("CHEMBENCH_API_KEY") or os.getenv(cfg["api_key_env"])
    if not args.preflight_only and not api_key:
        raise SystemExit(f"set {cfg['api_key_env']} or CHEMBENCH_API_KEY before a live run")
    if args.skip_image_load:
        environment_row, sidecar_row = expected_images(args.task)
        environment_image, sidecar_image = environment_row["tag"], sidecar_row["tag"]
        for ref in (environment_image, sidecar_image, *RUNTIME_TAGS):
            if not image_exists(ref):
                raise SystemExit(f"--skip-image-load requested but image/tag is absent: {ref}")
        verify_image(environment_row, args.task)
        if not os.environ.get("CHEMBENCH_ALLOW_PATCHED_SIDECAR"):
            verify_image(sidecar_row, args.task)
    else:
        environment_image, sidecar_image = load_images(args.task)
    task = next((FRAMEWORK / "tasks").glob(f"task-{args.task:02d}-*"))
    protocol = json.loads((task / "environment" / "public" / ".chembench_agent.json").read_text(encoding="utf-8"))
    if protocol["sidecar_url"] != sidecar_url:
        raise SystemExit(f"port contract mismatch: config={protocol['sidecar_url']} launcher={sidecar_url}")
    runtime = ROOT / "runtime" / f"task{args.task:02d}" / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    evidence, output = runtime / "evidence", runtime / "output"
    evidence.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    start_sidecar(args.task, port, model, sidecar_image, evidence)
    install_sidecar_cleanup(args.task, output / "sidecar.log")
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "task": args.task, "environment_image": environment_image,
                          "sidecar_image": sidecar_image, "sidecar_url": sidecar_url}))
        subprocess.run(["docker", "rm", "-f", f"matbench-task{args.task:02d}-sidecar"], capture_output=True)
        return 0
    sidecar_env = f"CHEMBENCH_TASK{args.task:02d}_SIDECAR_URL"
    llm_log = output / "llm_calls.jsonl"
    job = {
        "job_name": f"task{args.task:02d}-{int(time.time())}",
        "jobs_dir": (output / "jobs").resolve().as_posix(),
        "n_concurrent_trials": 1,
        "quiet": True,
        "tasks": [{"path": task.resolve().as_posix()}],
        "agents": [{"import_path": os.getenv("CHEMBENCH_AGENT_IMPORT_PATH", "agents.chembench_agent:MSEBenchAgent"), "model_name": model,
                    "env": {"CHEMBENCH_API_BASE": api_base, "CHEMBENCH_MODEL_ID": model,
                            sidecar_env: sidecar_url, "CHEMBENCH_LLM_CALL_LOG": llm_log.resolve().as_posix(),
                            "CHEMBENCH_REQUEST_TIMEOUT": os.getenv("CHEMBENCH_REQUEST_TIMEOUT", "1800"),
                            "CHEMBENCH_REQUEST_RETRIES": os.getenv("CHEMBENCH_REQUEST_RETRIES", "20"),
                            "CHEMBENCH_ENABLE_THINKING": os.getenv("CHEMBENCH_ENABLE_THINKING", ""),
                            "CHEMBENCH_STREAM": os.getenv("CHEMBENCH_STREAM", "true"),
                            "CHEMBENCH_REQUIRE_MODEL_IDENTITY": "1",
                            "CHEMBENCH_USE_NATIVE_TOOL_CALLS": "1"}}],
        "verifier": {"env": {"CHEMBENCH_ORACLE_VALIDATION": ""}},
        "environment": {"import_path": "agents.release_environment:OrganizerDockerEnvironment", "delete": True,
                        "kwargs": {"evidence_dir": evidence.resolve().as_posix()}},
    }
    for _name in ("CHEMBENCH_TEMPERATURE", "CHEMBENCH_TOP_P", "CHEMBENCH_MAX_TOKENS",
                  "CHEMBENCH_ENABLE_THINKING", "CHEMBENCH_REASONING_EFFORT",
                  "CHEMBENCH_MAX_AGENT_TURNS", "CHEMBENCH_TOKEN_BUDGET"):
        if os.getenv(_name):
            job["agents"][0]["env"][_name] = os.getenv(_name)
    job_path = output / "harbor.yaml"
    job_path.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env.update({"CHEMBENCH_API_KEY": api_key, "CHEMBENCH_API_BASE": api_base,
                "CHEMBENCH_MODEL_ID": model, sidecar_env: sidecar_url,
                "PYTHONPATH": str(FRAMEWORK) + os.pathsep + str(ROOT.parents[1] / "src"), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"})
    env.update(job['agents'][0]['env'])
    env.pop('CHEMBENCH_ORACLE_VALIDATION', None)
    try:
        result = subprocess.run([harbor_executable(), "run", "--config", str(job_path), "--yes"], cwd=FRAMEWORK, env=env)
        if result.returncode:
            return result.returncode
        audit = audit_live_job(output / 'jobs' / job['job_name'], llm_log, args.task)
        (output / 'run_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(json.dumps({**audit, 'output': str(output)}))
        if audit['status'] != 'PASS':
            return 1
        return 0
    finally:
        _dump_sidecar_logs(f"matbench-task{args.task:02d}-sidecar", output / "sidecar.log")
        subprocess.run(["docker", "rm", "-f", f"matbench-task{args.task:02d}-sidecar"], capture_output=True)


def audit_live_job(job_dir: Path, llm_log: Path, task: int) -> dict:
    errors = []
    files = list(job_dir.glob('*/result.json'))
    result = json.loads(files[0].read_text()) if len(files) == 1 else {}
    count = {7: 5, 8: 6, 9: 4}[task]
    steps = result.get('step_results') or []
    if not result.get('finished_at') or result.get('exception_info'):
        errors.append('Trial incomplete or failed')
    if [s.get('step_name') for s in steps] != [f'case-{n}' for n in range(1, count + 1)]:
        errors.append('Missing or out-of-order case results')
    if any(s.get('exception_info') for s in steps):
        errors.append('Case execution failed')
    grader_path = files[0].parent / 'steps' / f'case-{count}' / 'verifier' / 'grader-result.json' if files else None
    grader = json.loads(grader_path.read_text()) if grader_path and grader_path.is_file() else {}
    score = grader.get('score')
    reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    if not grader:
        errors.append('Final verifier report missing')
    if grader.get('failure_codes'):
        errors.append('Verifier rejected evidence: ' + str(grader['failure_codes']))
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        errors.append('Invalid final score')
    elif type(reward) not in (int, float) or not math.isfinite(reward) or not math.isclose(score / 100, reward, abs_tol=1e-12):
        errors.append('Score/reward mismatch')
    calls = [json.loads(line) for line in llm_log.read_text().splitlines() if line.strip()] if llm_log.is_file() else []
    successful = [r for r in calls if isinstance(r.get('response'), dict) and r['response'].get('choices')]
    if not successful:
        errors.append('No successful provider response recorded')
    return {'status': 'FAIL' if errors else 'PASS', 'task': task, 'mode': 'live_llm',
            'score': score, 'reward': reward, 'successful_provider_calls': len(successful),
            'errors': errors}


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tasks 16/17: does the agent container actually reach the private response sidecar?

`ChemBenchAgent`'s `private_response_round` branch POSTs to /session and /evaluate every round and
has no fallback -- an absent endpoint raises, so an evaluated-model run cannot even start. Until
2026-09-06 the repository had no sidecar Dockerfile, no compose file and nothing that set
`CHEMBENCH_TASK16/17_SIDECAR_URL`, so that endpoint did not exist in any Harbor configuration.

This check exercises the wiring the same way a real run would, and from the same place:

    docker compose up  (base 'main' from environment/Dockerfile  +  the task's
                        environment/docker-compose.yaml, which adds the sidecar)
      -> exec INSIDE the agent container
      -> GET  <url declared in environment/public/.chembench_agent.json>/health
      -> POST /session
      -> POST /evaluate   x5, then a 6th to prove the quota closes
      -> assert no reference/score/achievement-shaped key appears in any response

Deliberately model-free: it answers "can the agent talk to the sidecar, and does the sidecar keep
its response contract", which is the part that was missing. It does not evaluate an LLM.

The URL is READ OUT of the agent config rather than hardcoded -- if someone changes the port in
.chembench_agent.json and not in the compose file, this check must fail, not silently test its own
assumption.

`--project-directory <task>/environment` is passed for the same reason: that is what Harbor passes,
and `build.context` in a compose file resolves relative to the PROJECT DIRECTORY, not relative to
the file that declares it. The task compose file says `context: ../../..`, which is the package root
only when the project directory is `<task>/environment`. An invocation that omits the flag resolves
the context somewhere else entirely and fails with `lstat .../sidecar: no such file or directory`,
so this check would be testing a configuration Harbor never runs.

Both images are the ones shipped in docker/ (./chembench load-images) and are started BY NAME, the way
Harbor starts them: `main` from task.toml [environment].docker_image, the sidecar from the compose
image: with pull_policy: never. Nothing is built unless --build is given, which needs network -- the
organizer host has none (2026-09-14), so a check that silently built would test the wrong thing.

    python scripts/task16_17_sidecar_wiring_check.py
    python scripts/task16_17_sidecar_wiring_check.py --task 16 --keep
"""
from __future__ import annotations
import os as _os, sys as _sys  # noqa: E401
# Never write __pycache__ into tasks/: this script loads task code, and bytecode dropped there
# changes the task's Harbor checksum, so datasets/*/dataset.toml would stop matching the tree.
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {16: "task-16-tablet-microenvironment-cqa", 17: "task-17-spherical-agglomeration"}
# The response contract: the sidecar returns measurements, never anything that reveals how far the
# candidate is from the reference optimum. Same word list the framework's own sidecar check uses.
FORBIDDEN = ("reference", "achievement", "score", "oracle", "coefficient", "gradient", "sensitivity")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _forbidden_keys(value, hits=None) -> list[str]:
    hits = [] if hits is None else hits
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in str(key).lower() for word in FORBIDDEN):
                hits.append(str(key))
            _forbidden_keys(item, hits)
    elif isinstance(value, list):
        for item in value:
            _forbidden_keys(item, hits)
    return hits


def _declared_url(task_dir: Path) -> str:
    cfg = json.loads((task_dir / "environment/public/.chembench_agent.json").read_text(encoding="utf-8"))
    return str(cfg["sidecar_url"]).rstrip("/")


def _exec_json(project: list[str], url: str, path: str, payload: dict | None) -> dict:
    """Speak to the sidecar from inside the agent container, using only its stdlib."""
    script = (
        "import json,sys,urllib.request\n"
        f"u={url + path!r}\n"
        f"d={None if payload is None else json.dumps(payload)!r}\n"
        "req=urllib.request.Request(u,data=None if d is None else d.encode(),"
        "headers={'Content-Type':'application/json'},method='GET' if d is None else 'POST')\n"
        "print(urllib.request.urlopen(req,timeout=30).read().decode())\n"
    )
    proc = _run([*project, "exec", "-T", "main", "python", "-c", script])
    if proc.returncode != 0:
        raise RuntimeError(f"{path} failed inside the agent container: {proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def check(number: int, keep: bool, build: bool = False) -> dict:
    task = TASKS[number]
    task_dir = ROOT / "tasks" / task
    url = _declared_url(task_dir)
    compose = task_dir / "environment/docker-compose.yaml"
    if not compose.is_file():
        raise SystemExit(f"{compose} missing -- the sidecar service is not declared for task {number}")

    import tomllib
    import yaml
    image = (yaml.safe_load(compose.read_text(encoding="utf-8")) or {})["services"][f"task{number}-sidecar"]["image"]
    main_image = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))["environment"]["docker_image"]
    present = {ref: _run(["docker", "image", "inspect", ref]).returncode == 0 for ref in (main_image, image)}
    if not all(present.values()) and not build:
        missing = ", ".join(ref for ref, ok in present.items() if not ok)
        raise SystemExit(f"{missing}: not in the local Docker store. Load the shipped images first "
                         f"(./chembench load-images), or pass --build on a host with network access.")
    prebuilt = present[image]

    tmp = Path(tempfile.mkdtemp(prefix=f"chembench-wiring-{number}-"))
    # Stand-in for the base compose Harbor generates. With docker_image set Harbor uses its prebuilt
    # template (image: + sleep), so the stand-in starts the same image the same way.
    base = tmp / "base.yaml"
    main_build = ("" if present[main_image] else
                  f"    pull_policy: build\n    build:\n      context: {(task_dir / 'environment').as_posix()}\n")
    base.write_text(
        "services:\n"
        "  main:\n"
        f"    image: {main_image}\n{main_build}"
        "    command: [\"sh\", \"-c\", \"sleep infinity\"]\n", encoding="utf-8")
    project = ["docker", "compose", "-p", f"chembench-wiring-{number}",
               # Harbor passes --project-directory <task>/environment; build.context in the task
               # compose file is relative to it, so this flag is not optional.
               "--project-directory", str(task_dir / "environment"),
               "-f", str(base), "-f", str(compose)]
    result = {"task": number, "url": url, "image": image, "image_prebuilt": prebuilt}
    try:
        up = _run([*project, "up", "-d", "--wait", "--wait-timeout",
                   "180" if prebuilt else "1800"])
        if up.returncode != 0:
            raise RuntimeError(f"compose up failed: {up.stderr.strip()[-600:]}")

        health = _exec_json(project, url, "/health", None)
        session = _exec_json(project, url, "/session", {})
        token = session["session_id"]
        key = "formulation" if number == 16 else "process"
        design = json.loads((task_dir / "steps/round-5/solution/candidate.json")
                            .read_text(encoding="utf-8"))["candidate"][key]

        responses = [_exec_json(project, url, "/evaluate",
                                {"session_id": token, "candidate": {key: design}}) for _ in range(5)]
        over = _exec_json(project, url, "/evaluate", {"session_id": token, "candidate": {key: design}})

        problems = []
        if not health.get("ok"):
            problems.append(f"/health returned {health}")
        if "session_id" not in session:
            problems.append("/session returned no session_id")
        if not all(r.get("ok") for r in responses):
            problems.append(f"one of the five /evaluate calls failed: {responses}")
        if over.get("error", {}).get("code") != "quota_exceeded":
            problems.append(f"6th /evaluate should be quota_exceeded, got {over}")
        leaked = _forbidden_keys({"session": session, "responses": responses})
        if leaked:
            problems.append(f"private-shaped keys in sidecar responses: {sorted(set(leaked))}")

        result.update({"rounds_measured": len(responses), "quota_enforced": not problems,
                       "forbidden_key_hits": leaked, "problems": problems,
                       "status": "PASS" if not problems else "FAIL"})
        return result
    finally:
        if not keep:
            _run([*project, "down", "-v", "--remove-orphans"])
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=sorted(TASKS))
    ap.add_argument("--keep", action="store_true", help="leave the compose project running")
    ap.add_argument("--build", action="store_true",
                    help="build images that are not loaded (needs Docker Hub + conda-forge)")
    args = ap.parse_args()
    if shutil.which("docker") is None:
        raise SystemExit("docker not found")

    reports = [check(n, args.keep, args.build) for n in ([args.task] if args.task else sorted(TASKS))]
    for r in reports:
        built = "shipped image, started by name" if r["image_prebuilt"] else "image built with --build"
        print(f"  {r['status']:<4} task-{r['task']}  {r['url']}  "
              f"{r['rounds_measured']}/5 measured, quota closes, "
              f"{len(r['forbidden_key_hits'])} forbidden key(s)  [{built}]")
        for p in r["problems"]:
            print(f"       ! {p}")
    overall = "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL"
    print(f"\nstatus = {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

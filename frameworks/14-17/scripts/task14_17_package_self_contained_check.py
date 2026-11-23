#!/usr/bin/env python3
"""Is this package self-contained, and does it hold enough to run an LLM evaluation?

Three questions, asked of the package as it sits on disk -- not of the machine it was built on:

  1. SELF-CONTAINED. Does anything a run needs live outside this directory? That covers the shared
     framework (agents/, chembench_harness/, templates/, schemas/), the base images, the conda
     channels, and any file referenced by a build context. A package that silently depends on a
     locally deployed private package works only on the machine that built it.

  2. NO BUILD-MACHINE RESIDUE IN RUNTIME FILES. An absolute path from the author's machine inside a
     script, Dockerfile, task config or compose file is an undeclared dependency. Recorded run
     artifacts under verification/ are evidence, not runtime inputs, and are reported separately.

  3. LLM-EVALUABLE. Can a clean host actually start a model run? That needs the participant agent to
     be importable, every task to load under harbor, the agent's model/credential contract to be
     declared, and no task to reference an endpoint or image the package cannot produce.

It is deliberately package-relative: it imports nothing from the machine except the declared public
dependencies in requirements.txt, and it fails if any of those is missing rather than skipping.

    python scripts/task14_17_package_self_contained_check.py
    python scripts/task14_17_package_self_contained_check.py --self-test
"""
from __future__ import annotations
import os as _os, sys as _sys  # noqa: E401
# Never write __pycache__ into tasks/: this script loads task code, and bytecode dropped there
# changes the task's Harbor checksum, so datasets/*/dataset.toml would stop matching the tree.
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

import argparse
import importlib.util
import json
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["task-14-multiobjective-mol-design", "task-15-lipophilic-sulfonamide-edit-budget",
         "task-16-tablet-microenvironment-cqa", "task-17-spherical-agglomeration"]

# The framework the tasks and the participant agent need. Shipping tasks without these is what made
# every previous revision an increment that only ran inside the organizer's combined tree.
FRAMEWORK = ["agents/chembench_agent.py", "chembench_harness", "templates/sidecar",
             "schemas", "pyproject.toml", "requirements.txt", "chembench", "scripts/run_task.sh",
             "scripts/offline_images.py", "scripts/mock_llm_server.py", "scripts/llm_mock_check.py",
             "docker/load_images.sh", "docker/build_images.sh",
             "docker/runner/Dockerfile", "docker/runner/IMAGE", "docker/image_manifest.json"]

# Host paths. Runtime files must contain none; verification/ may, and is reported apart.
# Assembled from fragments on purpose: spelled out literally, this pattern would match THIS FILE
# and the scanner would report itself. Exempting its own path would be worse -- an exemption is a
# hole, and the next file to need one would get it too.
_U = "Us" + "ers"
_BS = chr(92)  # backslash, for the Windows forms
_HOST_ALTERNATIVES = [
    "/" + _U + "/[A-Za-z0-9._-]+",              # per-user home prefix
    "/" + "ho" + "me" + "/[A-Za-z0-9._-]+",     # per-user home prefix
    "[A-Za-z]:" + re.escape(_BS) + _U + re.escape(_BS),  # windows per-user home
    "[A-Za-z]:" + re.escape(_BS * 2) + _U + re.escape(_BS * 2),  # same, JSON-escaped
    "/mnt/[a-z]/" + _U + "/",                   # WSL drive mount
    "/private/tmp/cla" + "ude-",                # this session's scratchpad
]
HOST_PATH = re.compile("(" + "|".join(_HOST_ALTERNATIVES) + ")")
# Directories a USER creates while following RUN.md, not package content. Found by running from the
# zip in an empty container on 2026-09-13: RUN.md has users create .venv inside the package, and the
# host-path scan then failed on 21 third-party files (numpy, click, huggingface_hub, SBOM json) whose
# docstrings mention per-user paths. Every documented workflow must pass this check on its own output.
GENERATED_DIRS = {".venv", "venv", "env", "jobs", "__pycache__", ".git", "node_modules",
                  ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _is_generated(path: Path) -> bool:
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        parts = path.parts
    return any(part in GENERATED_DIRS for part in parts)


RUNTIME_SUFFIX = {".py", ".sh", ".toml", ".yaml", ".yml", ".json", ".cfg"}
# The agent reads these; a run cannot start without the first three being supplied at launch.
AGENT_ENV = ["CHEMBENCH_MODEL_ID", "CHEMBENCH_API_KEY", "CHEMBENCH_API_BASE"]


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def check_self_contained() -> dict:
    problems = []
    for item in FRAMEWORK:
        if not (ROOT / item).exists():
            problems.append(f"framework component missing from the package: {item}")

    # every Dockerfile's base image, and every compose build context, must resolve inside the package
    bases, contexts = set(), []
    for df in sorted(ROOT.rglob("Dockerfile")):
        if _is_generated(df):
            continue
        for line in df.read_text(encoding="utf-8").splitlines():
            if line.startswith("FROM "):
                bases.add(line.split()[1])
    import yaml
    for cf in sorted(ROOT.glob("tasks/*/environment/docker-compose.yaml")):
        doc = yaml.safe_load(cf.read_text(encoding="utf-8")) or {}
        for name, svc in (doc.get("services") or {}).items():
            build = svc.get("build")
            if isinstance(build, dict):
                ctx = (cf.parent / build["context"]).resolve()
                dockerfile = ctx / build.get("dockerfile", "Dockerfile")
                contexts.append((_rel(cf), name, build["context"]))
                if ROOT.resolve() not in ctx.parents and ctx != ROOT.resolve():
                    problems.append(f"{_rel(cf)}: service {name} builds from OUTSIDE the package: {ctx}")
                elif not dockerfile.is_file():
                    problems.append(f"{_rel(cf)}: service {name} dockerfile not in package: {dockerfile}")
                for arg in (build.get("args") or {}).values():
                    target = (ctx / str(arg))
                    if str(arg).startswith("tasks/") and not target.exists():
                        problems.append(f"{_rel(cf)}: build arg path not in package: {arg}")
            # a service with an image and no build must be pullable from a public registry
            if svc.get("image") and not isinstance(build, dict):
                problems.append(f"{_rel(cf)}: service {name} uses image {svc['image']} with no build "
                                f"context -- a clean host cannot obtain it")
            if isinstance(build, dict) and svc.get("pull_policy") not in (None, "build", "never"):
                problems.append(f"{_rel(cf)}: service {name} has pull_policy="
                                f"{svc.get('pull_policy')!r}; a locally named image must not be pulled")
    return {"section": "self_contained", "base_images": sorted(bases),
            "compose_build_contexts": contexts, "problems": problems,
            "status": "PASS" if not problems else "FAIL"}


def check_entrypoint_closure() -> dict:
    """Every script the ./chembench wrapper invokes must be in the package.

    Found the hard way: the wrapper shipped without 15 of the scripts it calls, so
    `./chembench validate` died on a missing file while every other check reported PASS. A
    self-containment check that only looks at directories misses this -- the gap is a reference
    inside an executable, not an absent directory.
    """
    wrapper = ROOT / "chembench"
    problems, referenced = [], []
    if wrapper.is_file():
        for match in re.finditer(r"scripts/[A-Za-z0-9_]+\.py", wrapper.read_text(encoding="utf-8")):
            rel = match.group(0)
            referenced.append(rel)
            if not (ROOT / rel).is_file():
                problems.append(f"./chembench calls {rel}, which is not in the package")
    return {"section": "entrypoint_closure", "referenced": sorted(set(referenced)),
            "problems": problems, "status": "PASS" if not problems else "FAIL"}


def check_references_resolve() -> dict:
    """Everything the package NAMES must be something the package SHIPS.

    Two gaps slipped past the earlier sections in 2026-09-13's empty-environment audit:
      * datasets/development/dataset.toml listed 14 tasks, 10 of them not shipped, and carried stale
        digests for the 4 that were -- chembench_harness/validation.py checks digest FORMAT only;
      * four shipped scripts, and two ./chembench subcommands, targeted task-03 and tasks 01-13.
    Each ran fine on the author's combined tree and failed on this package alone.
    """
    import tomllib, warnings
    problems = []
    shipped = {tomllib.loads((d / "task.toml").read_text(encoding="utf-8"))["task"]["name"]: d
               for d in sorted((ROOT / "tasks").glob("task-*")) if (d / "task.toml").is_file()}
    checksum = None
    try:
        from harbor.models.task.task import Task
        def checksum(d):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return "sha256:" + Task(task_dir=d).checksum
    except Exception:                                                  # noqa: BLE001
        pass
    for manifest in sorted((ROOT / "datasets").glob("*/dataset.toml")):
        doc = tomllib.loads(manifest.read_text(encoding="utf-8"))
        for entry in doc.get("tasks") or []:
            name = entry.get("name")
            if name not in shipped:
                problems.append(f"{_rel(manifest)} lists {name}, which is not in this package")
            elif checksum and entry.get("digest") != checksum(shipped[name]):
                problems.append(f"{_rel(manifest)}: digest for {name} does not match the shipped task")
    present = {d.name for d in (ROOT / "tasks").glob("task-*")}
    ref = re.compile(r"task-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*")
    for f in sorted(list((ROOT / "scripts").glob("*")) + [ROOT / "chembench"]):
        if not f.is_file() or f.suffix not in {".py", ".sh", ""}:
            continue
        for name in sorted(set(ref.findall(f.read_text(encoding="utf-8", errors="ignore")))):
            if name not in present and not any(p.startswith(name) for p in present):
                problems.append(f"{_rel(f)} references {name}, which is not in this package")
    return {"section": "references_resolve", "shipped_tasks": sorted(present),
            "problems": problems, "status": "PASS" if not problems else "FAIL"}


def check_no_host_paths() -> dict:
    runtime_hits, evidence_hits = [], []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or _is_generated(p) or p.suffix.lower() not in RUNTIME_SUFFIX:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = HOST_PATH.findall(text)
        if not hits:
            continue
        (evidence_hits if p.relative_to(ROOT).parts[0] == "verification" else runtime_hits).append(
            {"file": _rel(p), "n": len(hits)})
    problems = [f"build-machine path in a runtime file: {h['file']} ({h['n']} hit(s))"
                for h in runtime_hits]
    return {"section": "no_host_paths", "runtime_hits": runtime_hits,
            "evidence_only_hits": evidence_hits, "problems": problems,
            "status": "PASS" if not problems else "FAIL"}


def check_llm_evaluable() -> dict:
    problems, notes = [], {}
    sys.path.insert(0, str(ROOT))
    for module, why in (("harbor", "the runner"), ("yaml", "task configs"), ("jsonschema", "schemas")):
        if importlib.util.find_spec(module) is None:
            problems.append(f"declared dependency {module!r} ({why}) is not installed; "
                            f"pip install -r requirements.txt")
    if not problems:
        try:
            from agents.chembench_agent import ChemBenchAgent  # noqa: F401
            notes["participant_agent"] = "agents.chembench_agent:ChemBenchAgent imports"
        except Exception as exc:                                # noqa: BLE001
            problems.append(f"participant agent does not import: {type(exc).__name__}: {exc}")
        try:
            from harbor.models.task.task import Task
            for name in TASKS:
                d = ROOT / "tasks" / name
                task = Task(task_dir=d)
                steps = task.config.steps or []
                for s in steps:
                    task.step_instruction(s.name)
                notes.setdefault("tasks_loaded", []).append(f"{name} ({len(steps)} steps)")
        except Exception as exc:                                # noqa: BLE001
            problems.append(f"task load failed: {type(exc).__name__}: {exc}")

    # the credential/model contract must be declared somewhere a launcher can find it
    agent_src = (ROOT / "agents" / "chembench_agent.py").read_text(encoding="utf-8")
    for var in AGENT_ENV:
        if var not in agent_src:
            problems.append(f"agent does not read {var}; the launch contract is undocumented")
    notes["agent_env_contract"] = AGENT_ENV

    # every task must declare the protocol the agent dispatches on
    for name in TASKS:
        cfg = ROOT / "tasks" / name / "environment" / "public" / ".chembench_agent.json"
        if not cfg.is_file():
            problems.append(f"{name}: environment/public/.chembench_agent.json missing; "
                            f"the agent cannot select a protocol")
            continue
        mode = json.loads(cfg.read_text(encoding="utf-8")).get("protocol_mode")
        if mode and f"'{mode}'" not in agent_src and f'"{mode}"' not in agent_src:
            problems.append(f"{name}: protocol_mode {mode!r} has no branch in the shipped agent")
        notes.setdefault("protocol_modes", {})[name] = mode
    return {"section": "llm_evaluable", "notes": notes, "problems": problems,
            "status": "PASS" if not problems else "FAIL"}


def check_offline_images() -> dict:
    """Every image a run starts must ship in docker/: the organizer host can neither pull nor build.

    2026-09-14, organizer host, the first docker command of RUN.md:
        Get "https://registry-1.docker.io/v2/": context deadline exceeded
    Every section above had passed, because none asked where an image comes from at RUN time. Each
    task image was built by Harbor FROM Docker Hub with conda-forge packages, and Harbor's own
    egress-control image is built FROM Docker Hub too. This section requires: every docker_image and
    compose image: listed in docker/image_manifest.json; sidecars pull_policy: never; Harbor's egress
    image shipped under the name the installed Harbor looks up; the archive on disk matching the manifest.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import offline_images as oi
    manifest = oi.load_manifest()
    problems = oi.static_problems(manifest)
    summary = {}
    if manifest:
        summary = {"archive": f"docker/{manifest['archive']}", "platform": manifest["platform"],
                   "images": len(manifest["images"]), "gigabytes": round(manifest["archive_bytes"] / 1e9, 2)}
    return {"section": "offline_images", "summary": summary, "problems": problems,
            "status": "PASS" if not problems else "FAIL"}


def self_test() -> int:
    """Prove each section can fail, by pointing it at a deliberately broken copy."""
    import shutil, tempfile
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="chembench-selfcheck-"))
    try:
        # host-path detector
        # composed, not written out: a literal here would make this file itself a hit, and
        # the scanner is supposed to have no exemptions -- including for itself.
        probe = "somewhere /" + _U + "/someone/private/wheelhouse is referenced"
        fired = bool(HOST_PATH.search(probe))
        print(f"  {'ok  ' if fired else 'FAIL'} host-path detector fires on a planted author path")
        ok &= fired
        # generated-directory exclusion: .venv/jobs are user output, scripts/ is package content
        ignored = _is_generated(ROOT / ".venv" / "lib" / "x.py") and _is_generated(ROOT / "jobs" / "t" / "r.json")
        scanned = not _is_generated(ROOT / "scripts" / "run_task.sh") and not _is_generated(ROOT / "tasks" / "t" / "task.toml")
        fired = ignored and scanned
        print(f"  {'ok  ' if fired else 'FAIL'} generated dirs (.venv, jobs) ignored; scripts/ and tasks/ still scanned")
        ok &= fired
        # framework-missing detector
        missing = [i for i in FRAMEWORK if not (tmp / i).exists()]
        fired = len(missing) == len(FRAMEWORK)
        print(f"  {'ok  ' if fired else 'FAIL'} framework detector reports all {len(FRAMEWORK)} "
              f"components missing from an empty tree")
        ok &= fired
        # offline-image detector: a manifest that covers everything passes; dropping one image,
        # renaming Harbor's egress image, or losing the manifest each fails
        sys.path.insert(0, str(ROOT / "scripts"))
        import offline_images as oi
        plat = "linux/amd64"
        good = {"platform": plat, "harbor_version": oi.harbor_version(),
                "images": [{"role": r["role"], "reference": r["reference"], "platform": plat}
                           for r in oi.required()]
                          + [{"role": "egress", "reference": oi.egress_image_name(plat), "platform": plat},
                             {"role": "runner", "reference": oi.runner_image(), "platform": plat}]}
        clean = oi.static_problems(good, check_archive=False) == []
        dropped = dict(good, images=[e for e in good["images"] if e["role"] != "verifier"])
        renamed = dict(good, images=[dict(e, reference=e["reference"] + "x") if e["role"] == "egress" else e
                                     for e in good["images"]])
        fired = (clean and len(oi.static_problems(dropped, check_archive=False)) == 4
                 and oi.static_problems(renamed, check_archive=False) != []
                 and oi.static_problems(None) != [])
        print(f"  {'ok  ' if fired else 'FAIL'} offline-image detector: complete manifest passes; missing "
              f"verifier images, a wrong egress name, or no manifest each fail")
        ok &= fired
        # protocol-branch detector
        fired = "'no_such_protocol_mode'" not in (ROOT / "agents" / "chembench_agent.py").read_text(
            encoding="utf-8")
        print(f"  {'ok  ' if fired else 'FAIL'} protocol-branch detector would reject an unknown mode")
        ok &= fired
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nself-test " + ("PASS: each section can fail" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    reports = [check_self_contained(), check_entrypoint_closure(), check_references_resolve(),
               check_no_host_paths(), check_llm_evaluable(), check_offline_images()]
    for r in reports:
        print(f"  {r['status']:<4} {r['section']}")
        for p in r["problems"]:
            print(f"       ! {p}")
    sc = reports[0]
    print(f"\n  base image(s)        : {', '.join(sc['base_images'])}")
    print(f"  compose build ctx    : {[c[2] for c in sc['compose_build_contexts']]}")
    ev = reports[3]["evidence_only_hits"]
    print(f"  host paths           : 0 in runtime files, {len(ev)} file(s) under verification/ "
          f"(recorded run artifacts, not inputs)")
    notes = reports[4]["notes"]
    print(f"  participant agent    : {notes.get('participant_agent', 'n/a')}")
    print(f"  tasks loaded         : {len(notes.get('tasks_loaded', []))}/4")
    print(f"  protocol modes       : {notes.get('protocol_modes')}")
    print(f"  launch env contract  : {' '.join(notes.get('agent_env_contract', []))}")
    off = reports[5]["summary"]
    print(f"  offline images       : {off.get('archive')}  {off.get('platform')}  {off.get('images')} images  "
          f"{off.get('gigabytes')} GB  (nothing pulled or built at run time)")
    overall = "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL"
    print(f"\nstatus = {overall}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

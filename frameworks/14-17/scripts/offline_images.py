#!/usr/bin/env python3
"""Offline Docker images for tasks 14-17: what a run needs, building the archive, checking a daemon.

WHY THIS EXISTS
---------------
The organizer host has no route to Docker Hub. On 2026-09-14 the first command of RUN.md failed there:

    Get "https://registry-1.docker.io/v2/": context deadline exceeded

Every image this package used to build at run time starts FROM mambaorg/micromamba (Docker Hub) and
installs from conda-forge, so no task could have started on that host. Every image a run needs now
ships in docker/ and is started BY NAME; nothing is built or pulled at run time:

  role         the reference comes from                              used by
  environment  tasks/<t>/task.toml  [environment].docker_image       Harbor agent container
  verifier     tasks/<t>/task.toml  [verifier.environment].docker_image   separate verifier
  sidecar      tasks/<t>/environment/docker-compose.yaml  image:     16/17 private response service
  egress       Harbor's content-addressed egress-control image       every no-network trial on Linux;
                                                                     Harbor BUILDS it FROM gogost/gost
                                                                     (Docker Hub) when it is absent
  runner       docker/runner/IMAGE                                   the `docker run` runner (RUN.md)

Image identity is checked on the RootFS layer diff-IDs, not on the image ID: the same image has
different IDs in Docker's classic and containerd image stores, the diff-IDs are equal in both.

  offline_images.py required [--task T] [--role R] [--format json|refs]
  offline_images.py static                      manifest covers every reference; archive present
  offline_images.py check [--task T] [--no-egress] [--all-roles] [--quiet]     needs Docker
  offline_images.py build [--platform linux/amd64]     author side: build, save archive, manifest
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = ROOT / "docker"
MANIFEST = DOCKER_DIR / "image_manifest.json"
RUNNER_IMAGE_FILE = DOCKER_DIR / "runner" / "IMAGE"
ARCHIVE_PREFIX = "chembench_14_17_images_"


def runner_image() -> str:
    return RUNNER_IMAGE_FILE.read_text(encoding="utf-8").strip()


def archive_name(platform: str) -> str:
    return f"{ARCHIVE_PREFIX}{platform.replace('/', '-')}.tar.gz"


def required(task: str | None = None) -> list[dict]:
    """Every image reference the shipped tasks start, in task order."""
    refs, seen = [], False
    for tdir in sorted(p.parent for p in (ROOT / "tasks").glob("*/task.toml")):
        if task and tdir.name != task:
            continue
        seen = True
        rel = f"tasks/{tdir.name}"
        cfg = tomllib.loads((tdir / "task.toml").read_text(encoding="utf-8"))
        refs.append({"role": "environment", "task": tdir.name, "source": f"{rel}/task.toml [environment]",
                     "reference": (cfg.get("environment") or {}).get("docker_image"),
                     "recipe": {"context": f"{rel}/environment"}})
        verifier_env = (cfg.get("verifier") or {}).get("environment") or {}
        refs.append({"role": "verifier", "task": tdir.name, "source": f"{rel}/task.toml [verifier.environment]",
                     "reference": verifier_env.get("docker_image"), "recipe": {"context": f"{rel}/tests"}})
        compose = tdir / "environment" / "docker-compose.yaml"
        if compose.is_file():
            services = (yaml.safe_load(compose.read_text(encoding="utf-8")) or {}).get("services") or {}
            for name, svc in services.items():
                if not svc.get("image"):
                    continue
                build = svc.get("build") or {}
                ctx = (compose.parent / build.get("context", ".")).resolve()
                refs.append({"role": "sidecar", "task": tdir.name,
                             "source": f"{rel}/environment/docker-compose.yaml services.{name}",
                             "reference": svc["image"], "pull_policy": svc.get("pull_policy"),
                             "recipe": {"context": ctx.relative_to(ROOT.resolve()).as_posix() or ".",
                                        "dockerfile": build.get("dockerfile", "Dockerfile"),
                                        "args": {k: str(v) for k, v in (build.get("args") or {}).items()}}})
    if task and not seen:
        raise SystemExit(f"no such task: {task}")
    return refs


def harbor_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("harbor")
    except Exception:
        return None


def egress_image_name(platform: str) -> str:
    """The name Harbor looks up before it would build its egress-control sidecar (Harbor 0.21.0)."""
    from harbor.environments.docker import EGRESS_CONTROL_SIDECAR_CONTEXT_PATH as ctx
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.environments.docker.utils import _compute_image_name, docker_build_context_hash
    digest = docker_build_context_hash(context=ctx, dockerfile_path=ctx / "Dockerfile",
                                       build_args={}, platform=platform)
    return _compute_image_name(DockerEnvironment._EGRESS_CONTROL_SIDECAR_DOCKER_NAME, digest)


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def inspect(ref: str) -> dict | None:
    result = _docker("image", "inspect", ref)
    if result.returncode != 0:
        return None
    info = json.loads(result.stdout)[0]
    layers = (info.get("RootFS") or {}).get("Layers") or []
    return {"image_id": info.get("Id"), "platform": f"{info.get('Os')}/{info.get('Architecture')}",
            "layers_digest": "sha256:" + hashlib.sha256("\n".join(layers).encode()).hexdigest(),
            "layer_count": len(layers), "size_bytes": info.get("Size")}


def daemon_platform() -> str | None:
    result = _docker("version", "--format", "{{.Server.Os}}/{{.Server.Arch}}")
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict | None:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else None


# ---------------------------------------------------------------------------------------------------
def static_problems(manifest: dict | None, check_archive: bool = True) -> list[str]:
    """No Docker needed: does the shipped manifest cover everything the tasks will start?"""
    if manifest is None:
        return ["docker/image_manifest.json missing -- the tasks start prebuilt images that are not shipped"]
    problems = []
    listed = {e["reference"]: e for e in manifest.get("images", [])}
    for r in required():
        if not r["reference"]:
            problems.append(f"{r['source']}: no docker_image -- Harbor would BUILD this at run time "
                            f"(Docker Hub + conda-forge), which the organizer host cannot reach")
        elif r["reference"] not in listed:
            problems.append(f"{r['source']}: {r['reference']} is not in docker/image_manifest.json")
        if r["role"] == "sidecar" and r.get("pull_policy") != "never":
            problems.append(f"{r['source']}: pull_policy {r.get('pull_policy')!r}; must be 'never' so compose "
                            f"can neither pull nor rebuild the shipped image")
    for e in manifest.get("images", []):
        if e.get("platform") != manifest.get("platform"):
            problems.append(f"{e['reference']} is {e.get('platform')}, archive declares {manifest.get('platform')}")
    runners = [e["reference"] for e in manifest.get("images", []) if e["role"] == "runner"]
    if runners != [runner_image()]:
        problems.append(f"runner in manifest {runners} != docker/runner/IMAGE {runner_image()!r}")
    egress = [e["reference"] for e in manifest.get("images", []) if e["role"] == "egress"]
    try:
        expected = egress_image_name(manifest["platform"])
        if egress != [expected]:
            problems.append(f"egress image in manifest {egress} != {expected!r}, the name Harbor "
                            f"{harbor_version()} looks up on {manifest['platform']}; Harbor would build it "
                            f"FROM Docker Hub")
    except ImportError:
        problems.append("harbor is not importable here, so the egress image name cannot be confirmed")
    if manifest.get("harbor_version") != harbor_version():
        problems.append(f"manifest built with harbor {manifest.get('harbor_version')}, installed "
                        f"{harbor_version()} (requirements.txt pins the version the egress name depends on)")
    if check_archive:
        archive = DOCKER_DIR / str(manifest.get("archive"))
        sums = archive.with_name(archive.name + ".sha256")
        if not archive.is_file():
            problems.append(f"docker/{manifest.get('archive')} missing")
        elif archive.stat().st_size != manifest.get("archive_bytes"):
            problems.append(f"docker/{archive.name} is {archive.stat().st_size} bytes, manifest says "
                            f"{manifest.get('archive_bytes')}")
        if not sums.is_file() or sums.read_text(encoding="utf-8").split()[:1] != [manifest.get("archive_sha256")]:
            problems.append(f"docker/{sums.name} missing or disagrees with the manifest archive_sha256")
    return problems


def cmd_static() -> int:
    manifest = load_manifest()
    problems = static_problems(manifest)
    if manifest:
        print(f"  archive   : docker/{manifest['archive']}  {manifest['platform']}  "
              f"{manifest['archive_bytes'] / 1e9:.2f} GB  {len(manifest['images'])} images")
    for p in problems:
        print(f"  ! {p}")
    print(f"\nstatus = {'PASS' if not problems else 'FAIL'}")
    return 0 if not problems else 1


def cmd_check(task: str | None, no_egress: bool, all_roles: bool, quiet: bool) -> int:
    say = (lambda *_: None) if quiet else print
    manifest = load_manifest()
    if manifest is None:
        print("  ! docker/image_manifest.json missing")
        return 1
    problems = []
    daemon = daemon_platform()
    if daemon is None:
        print("  ! cannot reach a Docker daemon")
        return 1
    say(f"  daemon    : {daemon}   bundle: {manifest['platform']}")
    if daemon != manifest["platform"]:
        problems.append(f"this Docker daemon is {daemon} but the shipped images are {manifest['platform']}; "
                        f"rebuild on a networked {daemon} host with docker/build_images.sh --platform {daemon}")
    listed = {e["reference"]: e for e in manifest["images"]}
    wanted = [(r["role"], r["reference"]) for r in required(task)]
    if not no_egress:
        try:
            wanted.append(("egress", egress_image_name(daemon)))
        except ImportError:
            wanted += [("egress", e["reference"]) for e in manifest["images"] if e["role"] == "egress"]
    if all_roles:
        wanted.append(("runner", runner_image()))
    for role, ref in wanted:
        if not ref:
            problems.append(f"{role}: task declares no docker_image")
            continue
        info = inspect(ref)
        entry = listed.get(ref)
        if entry is None:
            problems.append(f"{role} {ref}: not in docker/image_manifest.json (Harbor {harbor_version()} on "
                            f"{daemon} would look for this name)")
        elif info is None:
            problems.append(f"MISSING {role} {ref} -- run ./chembench load-images; this host cannot pull or "
                            f"build it")
        elif info["layers_digest"] != entry["layers_digest"] and role == "egress":
            # Harbor names this image after the hash of its own build context, so every image under this
            # name was built from the same recipe. Another task package (task 18's, on 2026-09-16) ships
            # its own build of it; loading that one after ours only changed the layer timestamps, yet this
            # check used to call it stale and stop every 14-17 run until the organizer reloaded our archive.
            say(f"  ok   {role:<11} {ref}  (another build of the same Harbor recipe, e.g. from another package)")
        elif info["layers_digest"] != entry["layers_digest"]:
            problems.append(f"{role} {ref}: local layers differ from the shipped image (left over from an "
                            f"earlier revision?) -- docker image rm {ref} && ./chembench load-images")
        else:
            say(f"  ok   {role:<11} {ref}")
    for p in problems:
        print(f"  ! {p}")
    say(f"\nstatus = {'PASS' if not problems else 'FAIL'}")
    return 0 if not problems else 1


def cmd_build(platform: str) -> int:
    """Author side, on a host WITH network: build every image, save the archive, write the manifest."""
    import asyncio
    from harbor.environments.docker import EGRESS_CONTROL_SIDECAR_CONTEXT_PATH as ectx
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.environments.docker.utils import ensure_docker_image_built

    jobs = []
    for r in required():
        if not r["reference"]:
            raise SystemExit(f"{r['source']}: no image reference to build")
        recipe = r["recipe"]
        ctx = ROOT / recipe["context"]
        args = []
        if "dockerfile" in recipe:
            args += ["-f", str(ctx / recipe["dockerfile"])]
        for key, value in recipe.get("args", {}).items():
            args += ["--build-arg", f"{key}={value}"]
        jobs.append((r["reference"], [*args, str(ctx)]))
    runner_ctx = Path(tempfile.mkdtemp(prefix="chembench-runner-ctx-"))
    shutil.copy2(ROOT / "requirements.txt", runner_ctx / "requirements.txt")
    jobs.append((runner_image(), ["-f", str(DOCKER_DIR / "runner" / "Dockerfile"), str(runner_ctx)]))

    for ref, args in jobs:
        print(f"### build {ref}", flush=True)
        subprocess.run(["docker", "buildx", "build", "--platform", platform, "--load", "-t", ref, *args],
                       check=True)
    shutil.rmtree(runner_ctx, ignore_errors=True)
    egress = asyncio.run(ensure_docker_image_built(
        docker_name=DockerEnvironment._EGRESS_CONTROL_SIDECAR_DOCKER_NAME, docker_build_context=ectx,
        dockerfile_path=ectx / "Dockerfile", build_args={}, platform=platform, timeout_sec=3600))
    print(f"### egress {egress}", flush=True)

    refs = [ref for ref, _ in jobs] + [egress]
    for old in DOCKER_DIR.glob(f"{ARCHIVE_PREFIX}*"):
        old.unlink()
    archive = DOCKER_DIR / archive_name(platform)
    print(f"### save {len(refs)} images -> docker/{archive.name}", flush=True)
    cmd = ["docker", "save", "--platform", platform, *refs]
    if "--platform" not in _docker("save", "--help").stdout:
        cmd.remove("--platform"); cmd.remove(platform)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc, \
            gzip.GzipFile(filename="", mode="wb", compresslevel=6, mtime=0,
                          fileobj=archive.open("wb")) as out:
        for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
            out.write(chunk)
    if proc.returncode != 0:
        raise SystemExit("docker save failed")
    return write_manifest(archive, platform, egress)


def write_manifest(archive: Path, platform: str, egress: str) -> int:
    entries = []
    wanted = [(r["role"], r["task"], r["reference"], r["recipe"]) for r in required()]
    wanted += [("egress", None, egress, {"harbor_builtin": "harbor/environments/docker/"
                                                            "harbor-docker-egress-control-sidecar"}),
               ("runner", None, runner_image(), {"dockerfile": "docker/runner/Dockerfile",
                                                 "context": "requirements.txt"})]
    for role, task, ref, recipe in wanted:
        info = inspect(ref)
        if info is None:
            raise SystemExit(f"{ref} is not in the local Docker store")
        if info["platform"] != platform:
            raise SystemExit(f"{ref} is {info['platform']}, expected {platform}")
        entries.append({"role": role, "task": task, "reference": ref, **info, "recipe": recipe})
    digest = _sha256(archive)
    manifest = {
        "schema": "chembench-offline-images/1",
        "platform": platform,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harbor_version": harbor_version(),
        "archive": archive.name,
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "registry_push_performed": False,
        "private_material": "verifier and sidecar images contain tests/task_snapshot (private reference "
                            "data). Organizer use only; never push them to a public registry.",
        "identity": "layers_digest = sha256 over RootFS.Layers joined by newlines; equal across the classic "
                    "and containerd image stores, unlike image_id",
        "images": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"### manifest docker/{MANIFEST.name}: {len(entries)} images, archive sha256 {digest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rq = sub.add_parser("required")
    rq.add_argument("--task")
    rq.add_argument("--role", choices=["environment", "verifier", "sidecar"])
    rq.add_argument("--format", choices=["json", "refs"], default="json")
    sub.add_parser("static")
    ck = sub.add_parser("check")
    ck.add_argument("--task")
    ck.add_argument("--no-egress", action="store_true", help="skip Harbor's egress image (public-network runs)")
    ck.add_argument("--all-roles", action="store_true", help="also require the runner image")
    ck.add_argument("--quiet", action="store_true", help="print problems only")
    bd = sub.add_parser("build")
    bd.add_argument("--platform", default="linux/amd64")
    args = ap.parse_args()

    if args.cmd == "required":
        refs = [r for r in required(args.task) if not args.role or r["role"] == args.role]
        print("\n".join(r["reference"] or "" for r in refs) if args.format == "refs"
              else json.dumps(refs, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "static":
        return cmd_static()
    if args.cmd == "check":
        return cmd_check(args.task, args.no_egress, args.all_roles, args.quiet)
    return cmd_build(args.platform)


if __name__ == "__main__":
    sys.exit(main())

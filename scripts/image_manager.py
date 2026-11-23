#!/usr/bin/env python3
"""Task image manager — install MSEBench task Docker images from YAML manifests.

Reads images/task_images.yaml (aggregate) or images/taskNN/images.yaml (per task)
and drives docker to verify, load, or build each image.  Idempotent: `ensure`
skips images whose id already matches the local docker store.

Commands:
  list    [--task NN] [--json]      show task -> image mapping
  verify  [--task NN] [--all]       check tar presence + sha256 (default: all)
  load    [--task NN] [--all]       docker load from tar (sha256 verified first)
  build   [--task NN] [--all]       docker build from the matched Dockerfile recipe
  ensure  [--task NN] [--all]       skip if already present, else load, else build

Options:
  --role environment|sidecar|verifier|tests   restrict to one role
  --no-sha                                     skip (slow) tar sha256 verification
  --dry-run                                    print actions without running docker

Examples:
  python3 scripts/image_manager.py list --task 8
  python3 scripts/image_manager.py verify --task 8
  python3 scripts/image_manager.py ensure --task 8
  python3 scripts/image_manager.py ensure --all --no-sha
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
AGGREGATE = ROOT / "images" / "task_images.yaml"


def _load_aggregate() -> dict:
    if not AGGREGATE.is_file():
        raise SystemExit(f"missing {AGGREGATE}; run scripts/gen_task_images_yaml.py first")
    return yaml.safe_load(AGGREGATE.read_text(encoding="utf-8"))


def _iter_images(man: dict, task: str | None, role: str | None):
    """Yield (task_number, entry) across the aggregate manifest."""
    for family, tasks in (man.get("families") or {}).items():
        for num, entries in tasks.items():
            if task and num != task:
                continue
            for entry in entries:
                if role and entry.get("role") != role:
                    continue
                yield num, entry
    for entry in man.get("shared_images") or []:
        if role and entry.get("role") != role:
            continue
        yield "shared", entry


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _docker(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=capture, text=True)


def _local_image_id(tag: str) -> str | None:
    proc = _docker("image", "inspect", tag, "--format", "{{.Id}}", capture=True)
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def _drag(action: str, dry: bool):
    if dry:
        print(f"    [dry-run] {action}")
        return True
    print(f"    $ {action}")
    return None


def cmd_list(man: dict, args) -> int:
    rows = []
    for num, entry in _iter_images(man, args.task, args.role):
        rows.append({"task": num, "role": entry["role"], "tag": entry["tag"],
                     "tar": entry["tar"], "bytes": entry.get("bytes"),
                     "build": "yes" if entry.get("build") else None})
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    print(f"{'task':6} {'role':12} {'tag':52} {'MB':>6} {'tar':40}")
    for r in rows:
        mb = f"{r['bytes']/1e6:.0f}" if r.get("bytes") else "-"
        print(f"{r['task']:6} {r['role']:12} {r['tag'][:52]:52} {mb:>6} {r['tar'][:40]}")
    print(f"total {len(rows)} image entries")
    return 0


def cmd_verify(man: dict, args) -> int:
    bad = 0
    for num, entry in _iter_images(man, args.task, args.role):
        tar = ROOT / entry["tar"]
        label = f"{num} {entry['role']}"
        if not tar.is_file():
            print(f"[MISSING] {label}: {entry['tar']}")
            bad += 1
            continue
        if args.no_sha:
            print(f"[ok-tar]  {label}: {entry['tar']}")
            continue
        digest = _sha256(tar)
        if digest == entry.get("tar_sha256"):
            print(f"[ok]      {label}: sha256 verified")
        else:
            print(f"[BAD]     {label}: sha256 {digest[:16]} != {str(entry.get('tar_sha256'))[:16]}")
            bad += 1
    return 1 if bad else 0


def _load_one(num: str, entry: dict, args) -> bool:
    tar = ROOT / entry["tar"]
    if not tar.is_file():
        print(f"  [skip] {entry['tag']}: tar missing ({entry['tar']})")
        return False
    if not args.no_sha:
        digest = _sha256(tar)
        if digest != entry.get("tar_sha256"):
            print(f"  [fail] {entry['tag']}: tar sha256 mismatch; refusing to load")
            return False
    result = _drag(f"docker load -i {entry['tar']}", args.dry_run)
    if result is None:
        proc = _docker("load", "-i", str(tar), capture=True)
        if proc.returncode != 0:
            print(f"  [fail] {entry['tag']}: {proc.stderr.strip()[:200]}")
            return False
    print(f"  [loaded] {entry['tag']}")
    return True


def _build_one(num: str, entry: dict, args) -> bool:
    recipe = entry.get("build")
    if not recipe:
        print(f"  [skip] {entry['tag']}: no build recipe")
        return False
    if recipe.get("compose"):
        command = ["docker", "compose", "-f", str(ROOT / recipe["compose"]), "build"]
        if args.no_cache:
            command.append("--no-cache")
        if args.pull:
            command.append("--pull")
        command.append(recipe["service"])
        if args.dry_run:
            print(" ".join(command))
            return True
        return subprocess.run(command, cwd=ROOT).returncode == 0
    context = ROOT / recipe["context"]
    if not context.is_dir() or not (context / recipe['dockerfile']).is_file():
        print(f"  [fail] {entry['tag']}: missing build context or Dockerfile")
        return False
    flags = (['--no-cache'] if getattr(args, 'no_cache', False) else []) + (['--pull'] if getattr(args, 'pull', False) else [])
    command = f"docker build {' '.join(flags)} -t {entry['tag']} -f {recipe['context']}/{recipe['dockerfile']} {recipe['context']}"
    result = _drag(command, args.dry_run)
    if result is None:
        proc = _docker("build", *flags, "-t", entry["tag"], "-f", str(context / recipe["dockerfile"]), str(context),
                       capture=True)
        if proc.returncode != 0:
            print(f"  [fail] build {entry['tag']}: {proc.stderr.strip()[-200:]}")
            return False
    print(f"  [{'planned' if args.dry_run else 'built'}] {entry['tag']}")
    return True


def cmd_load(man: dict, args) -> int:
    failed = 0
    for num, entry in _iter_images(man, args.task, args.role):
        if not _load_one(num, entry, args):
            failed += 1
    return 1 if failed else 0


def cmd_build(man: dict, args) -> int:
    failed = 0
    for num, entry in _iter_images(man, args.task, args.role):
        if not _build_one(num, entry, args):
            failed += 1
    return 1 if failed else 0


def cmd_ensure(man: dict, args) -> int:
    failed = 0
    for num, entry in _iter_images(man, args.task, args.role):
        tag = entry["tag"]
        local = _local_image_id(tag)
        if local and local == entry.get("image_id"):
            print(f"[present] task{num} {entry['role']}: {tag}")
            continue
        if local:
            print(f"[stale]   task{num} {entry['role']}: local {local[:20]} != manifest {str(entry.get('image_id'))[:20]}")
        if _load_one(num, entry, args):
            continue
        if entry.get("build") and _build_one(num, entry, args):
            continue
        print(f"[FAIL]    task{num} {entry['role']}: {tag} — no tar and no usable build recipe")
        failed += 1
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("list", "verify", "load", "build", "ensure"))
    parser.add_argument("--task", help="two-digit task number, e.g. 8")
    parser.add_argument("--all", action="store_true", help="all tasks (default for verify/load/build/ensure)")
    parser.add_argument("--role", choices=("environment", "sidecar", "verifier", "tests"))
    parser.add_argument("--no-sha", action="store_true", help="skip tar sha256 verification")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild without Docker layer cache")
    parser.add_argument("--pull", action="store_true", help="Refresh public base image before building")
    parser.add_argument("--json", action="store_true", help="machine-readable output for list")
    args = parser.parse_args()

    man = _load_aggregate()
    if args.task and str(args.task).lower() == "all":
        args.task = None
        args.all = True
    if args.task:
        args.task = str(int(args.task))

    dispatch = {"list": cmd_list, "verify": cmd_verify, "load": cmd_load,
                "build": cmd_build, "ensure": cmd_ensure}
    return dispatch[args.command](man, args)


if __name__ == "__main__":
    raise SystemExit(main())

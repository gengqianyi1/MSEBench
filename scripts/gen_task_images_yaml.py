#!/usr/bin/env python3
"""Generate YAML image manifests for every task from the authoritative locks.

Sources of truth:
  images/manifest.json                    family -> task -> {tar: {tag, image_id, tar_sha256}}
  images/taskNN/images.lock.json          per-task authoritative lock
  images/shared_images/images.lock.json   shared runtime images
  tasks/taskNN/**/Dockerfile*             optional rebuild recipes (matched heuristically)

Outputs:
  images/task_images.yaml                 one aggregate manifest (all tasks + shared)
  images/taskNN/images.yaml               per-task manifest

Usage:
  python3 scripts/gen_task_images_yaml.py
  python3 scripts/image_manager.py list
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"

BUILD_PATTERNS = {
    # prefer the Dockerfile.rebuild (vendor-free) variant when present
    "environment": ["environment/Dockerfile.rebuild", "environment/Dockerfile",
                    "tasks/*/environment/Dockerfile.rebuild", "tasks/*/environment/Dockerfile"],
    "verifier": ["tests/Dockerfile.rebuild", "tests/Dockerfile",
                 "tasks/*/tests/Dockerfile.rebuild", "tasks/*/tests/Dockerfile",
                 "steps/case-1/tests/Dockerfile"],
    "tests": ["tests/Dockerfile.rebuild", "tests/Dockerfile",
              "tasks/*/tests/Dockerfile.rebuild", "tasks/*/tests/Dockerfile"],
    "sidecar": ["tests/Dockerfile.sidecar", "sidecar/Dockerfile", "tests/sidecar/Dockerfile",
                "tasks/*/tests/Dockerfile.sidecar", "tasks/*/sidecar/Dockerfile",
                "tasks/*/tests/sidecar/Dockerfile"],
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_build_recipe(num: str, role: str) -> dict | None:
    base = ROOT / "tasks" / f"task{int(num):02d}"
    local_sidecar = base / "environment/Dockerfile.sidecar"
    if role == "sidecar" and local_sidecar.is_file():
        return {"context": ".", "dockerfile": local_sidecar.relative_to(ROOT).as_posix()}
    if role == "sidecar" and int(num) in {16, 17}:
        return {
            "context": "frameworks/14-17/docker_build", "dockerfile": "dockerfiles/sidecar.Dockerfile",
            "compose": "frameworks/14-17/docker_build/docker-compose.build.yaml",
            "service": f"task{int(num)}-sidecar",
        }
    for pattern in BUILD_PATTERNS.get(role, []):
        for dockerfile in sorted(base.glob(pattern)):
            if dockerfile.is_file():
                if num in ('19', '20') and role == 'sidecar':
                    return {'context': base.relative_to(ROOT).as_posix(),
                            'dockerfile': dockerfile.relative_to(base).as_posix()}
                # sidecar/Dockerfile builds with the package root as context (see sidecar/README.md)
                if dockerfile.parent.name == "sidecar":
                    return {"context": dockerfile.parent.parent.relative_to(ROOT).as_posix(),
                            "dockerfile": dockerfile.relative_to(dockerfile.parent.parent).as_posix()}
                return {"context": dockerfile.parent.relative_to(ROOT).as_posix(), "dockerfile": dockerfile.name}
    return None


def _entry(tname: str, tarname: str, rec: dict, num: str | None, role: str, tar_prefix: str) -> dict:
    entry = {
        "role": role,
        "tag": rec["tag"],
        "image_id": rec["image_id"],
        "tar": f"{tar_prefix}/{tarname}",
        "bytes": rec.get("bytes"),
        "tar_sha256": rec.get("tar_sha256"),
    }
    if num is not None:
        recipe = _find_build_recipe(num, role)
        if recipe:
            entry["build"] = recipe
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default=str(IMAGES / "task_images.yaml"))
    args = parser.parse_args()

    manifest = _load(IMAGES / "manifest.json")
    aggregate = {
        "schema": "msebench.task-images/1.0",
        "generated_from": "images/manifest.json + images/taskNN/images.lock.json",
        "install": {
            "loader": "scripts/image_manager.py",
            "usage": "python3 scripts/image_manager.py ensure --all     # verify+load every task image",
        },
        "families": {},
        "shared_images": [],
    }
    count = 0

    for family, tasks in manifest["families"].items():
        if family == "shared":
            lock = _load(IMAGES / "shared_images" / "images.lock.json")
            for tarname, rec in lock["images"].items():
                aggregate["shared_images"].append(
                    _entry("shared_images", tarname, rec, None, tarname[:-4], "images/shared_images"))
                count += 1
            continue
        for tname in tasks:
            if not (ROOT / 'tasks' / tname).is_dir():
                continue
            num = str(int(tname.replace("task", "")))
            lock = _load(IMAGES / tname / "images.lock.json")
            entries = []
            for tarname, rec in lock["images"].items():
                role = tarname[:-4].replace(tname + "-", "")
                entries.append(_entry(tname, tarname, rec, num, role, f"images/{tname}"))
                count += 1
            aggregate["families"].setdefault(family, {})[num] = entries
            per_task = {
                "schema": "msebench.task-images/1.0",
                "task": num,
                "family": family,
                "images": entries,
                "install": aggregate["install"],
            }
            (IMAGES / tname / "images.yaml").write_text(
                yaml.safe_dump(per_task, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")

    Path(args.aggregate).write_text(
        yaml.safe_dump(aggregate, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")
    print(f"wrote {args.aggregate} ({count} image entries) + per-task images/taskNN/images.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

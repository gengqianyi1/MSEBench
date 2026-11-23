#!/usr/bin/env python3
"""Validate Docker build inputs without invoking a Docker daemon."""
from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "frameworks/14-17"
BUILD = FRAMEWORK / "docker_build"
SLUGS = {
    14: "task-14-multiobjective-mol-design",
    15: "task-15-lipophilic-sulfonamide-edit-budget",
    16: "task-16-tablet-microenvironment-cqa",
    17: "task-17-spherical-agglomeration",
}


def check_1417() -> int:
    compose = yaml.safe_load((BUILD / "docker-compose.build.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert len(services) == 12, f"expected 12 services, got {len(services)}"
    for number, slug in SLUGS.items():
        task = ROOT / "tasks" / f"task{number}"
        cfg = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        for role, expected in (
            ("environment", cfg["environment"]["docker_image"]),
            ("verifier", cfg["verifier"]["environment"]["docker_image"]),
        ):
            assert services[f"task{number}-{role}"]["image"] == expected
        if number >= 16:
            sidecar = yaml.safe_load((task / "environment/docker-compose.yaml").read_text(encoding="utf-8"))
            assert services[f"task{number}-sidecar"]["image"] == sidecar["services"][f"task{number}-sidecar"]["image"]
    assert services["harbor-runner"]["image"] == (FRAMEWORK / "docker/runner/IMAGE").read_text(encoding="utf-8").strip()

    for name, spec in services.items():
        build = spec["build"]
        context = (BUILD / build["context"]).resolve()
        dockerfile = context / build.get("dockerfile", "Dockerfile")
        assert dockerfile.is_file(), f"{name}: missing {dockerfile}"
        contexts = {}
        for key, raw in build.get("additional_contexts", {}).items():
            rel = raw.replace("${CHEMBENCH_PKG:-..}", "..")
            match = re.search(r"/tasks/(task-\d+-[^/]+)/(.+)$", rel)
            if match:
                number = next(n for n, slug in SLUGS.items() if slug == match.group(1))
                target = ROOT / "tasks" / f"task{number}" / match.group(2)
            else:
                target = BUILD / rel
            assert target.is_dir(), f"{name}: missing context {key}: {target}"
            contexts[key] = target
        for key in re.findall(r"^COPY\s+--from=([\w-]+)", dockerfile.read_text(encoding="utf-8"), re.M):
            assert key in contexts, f"{name}: COPY refers to missing context {key}"
    return len(services)


def check_1920() -> int:
    count = 0
    for number in (19, 20):
        manifest = yaml.safe_load((ROOT / f"images/task{number}/images.yaml").read_text(encoding="utf-8"))
        assert {item["role"] for item in manifest["images"]} == {"environment", "sidecar", "verifier"}
        for item in manifest["images"]:
            recipe = item["build"]
            assert "\\" not in recipe["context"] and "\\" not in recipe["dockerfile"]
            context = ROOT / recipe["context"]
            dockerfile = context / recipe["dockerfile"]
            assert dockerfile.is_file(), str(dockerfile)
            lines = dockerfile.read_text(encoding="utf-8").splitlines()
            assert "FROM python:3.11.9-slim" in lines, str(dockerfile)
            for line in lines:
                if line.startswith("COPY "):
                    for source in shlex.split(line)[1:-1]:
                        assert (context / source).exists(), f"{dockerfile}: missing COPY source {source}"
            count += 1
    return count


def check_inventory() -> tuple[int, int, list[str]]:
    aggregate = yaml.safe_load((ROOT / "images/task_images.yaml").read_text(encoding="utf-8"))
    entries = [(family, task, item) for family, tasks in aggregate["families"].items()
               for task, images in tasks.items() for item in images]
    without_recipe = []
    recipes = 0
    for family, task, item in entries:
        recipe = item.get("build")
        if not recipe:
            without_recipe.append(f"{family}/task{task}/{item['role']}")
            continue
        assert (ROOT / recipe["context"] / recipe["dockerfile"]).is_file(), item["tag"]
        recipes += 1
    return len(entries), recipes, without_recipe


if __name__ == "__main__":
    services, task1920 = check_1417(), check_1920()
    images, recipes, missing = check_inventory()
    print(f"PASS: {services} revision10 services, {task1920} task19/20 recipes; "
          f"aggregate {recipes}/{images} task image references have a recipe")
    if missing:
        print("No aggregate recipe: " + ", ".join(missing))

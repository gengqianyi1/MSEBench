#!/usr/bin/env python3
"""Validate author-controlled Oracle bundles for Tasks 01 and 02.

This is intentionally task-side release evidence, not a Harbor harness change.
Bundles live only under tests/private_data/oracle_bundle and must never be
copied into environment/public or an agent workspace.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tomllib
import yaml
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "task01": PACKAGE_ROOT / "tasks" / "task01" / "tasks" / "task-01-jarvis-2d-exfoliation-candidate",
    "task02": PACKAGE_ROOT / "tasks" / "task02" / "tasks" / "task-02-oligothiophene-smd-sapt",
}
REQUIRED_MANIFEST_FIELDS = ("task_name", "task_version", "sidecar_image_digest", "reward", "score", "artifacts")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(task_key: str, task: Path) -> list[str]:
    task_name = task.name
    task_toml = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    bundle = task / "tests" / "private_data" / "oracle_bundle"
    manifest_path = bundle / "manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"{task_name}: missing author Oracle bundle: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{task_name}: invalid manifest JSON: {exc}"]
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            errors.append(f"{task_name}: manifest missing {field}")
    if manifest.get("task_name") != task_toml["task"]["name"]:
        errors.append(f"{task_name}: manifest task_name does not match task.toml")
    if manifest.get("task_version") != task_toml["task"]["version"]:
        errors.append(f"{task_name}: manifest task_version does not match task.toml")
    digest = manifest.get("sidecar_image_digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
        errors.append(f"{task_name}: sidecar_image_digest must be a sha256 digest")
    if not isinstance(manifest.get("reward"), (int, float)) or not isinstance(manifest.get("score"), (int, float)):
        errors.append(f"{task_name}: score and reward must be numeric outputs from the private grader")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return errors + [f"{task_name}: artifacts must map filenames to sha256 digests"]
    for artifact in task_toml["metadata"]["required_artifacts"]:
        relative = artifact.lstrip("/")
        path = bundle / "workspace" / relative
        if not path.is_file() and not path.is_dir():
            errors.append(f"{task_name}: missing required artifact {relative}")
            continue
        if path.is_file():
            expected = artifacts.get(relative)
            if expected != sha256(path):
                errors.append(f"{task_name}: artifact hash mismatch or missing manifest entry: {relative}")
    workspace = bundle / "workspace"
    workspace_files = {
        path.relative_to(workspace).as_posix(): sha256(path)
        for path in workspace.rglob("*") if path.is_file()
    }
    normalized_artifacts = {str(key).replace("\\", "/"): value for key, value in artifacts.items()}
    missing_hashes = sorted(set(workspace_files) - set(normalized_artifacts))
    extra_hashes = sorted(set(normalized_artifacts) - set(workspace_files))
    mismatched_hashes = sorted(
        key for key in set(workspace_files) & set(normalized_artifacts)
        if workspace_files[key] != normalized_artifacts[key]
    )
    if missing_hashes:
        errors.append(f"{task_name}: workspace files absent from artifact map: {missing_hashes}")
    if extra_hashes:
        errors.append(f"{task_name}: artifact map paths absent from workspace: {extra_hashes}")
    if mismatched_hashes:
        errors.append(f"{task_name}: workspace hash mismatches: {mismatched_hashes}")
    evidence = bundle / "verifier" / "reward.json"
    details = bundle / "verifier" / "details.json"
    if not details.is_file():
        errors.append(f"{task_name}: missing verifier/details.json")
    if not evidence.is_file():
        errors.append(f"{task_name}: missing verifier/reward.json")
    else:
        try:
            observed = json.loads(evidence.read_text(encoding="utf-8"))["reward"]
            if observed != manifest.get("reward"):
                errors.append(f"{task_name}: manifest reward differs from verifier/reward.json")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"{task_name}: invalid verifier/reward.json: {exc}")
    deployment_path = task / "environment" / "public" / "sidecar_deployment.yaml"
    try:
        deployment = yaml.safe_load(deployment_path.read_text(encoding="utf-8"))
        declared = deployment["release"]["private_image_digest"]
        if declared != manifest.get("sidecar_image_digest"):
            errors.append(f"{task_name}: Oracle and deployment image digests differ")
        if not (deployment.get("deployment") or {}).get("evidence_volume_mount"):
            errors.append(f"{task_name}: deployment lacks evidence volume mount")
        if not (deployment.get("deployment") or {}).get("entrypoint"):
            errors.append(f"{task_name}: deployment lacks sidecar entrypoint")
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError) as exc:
        errors.append(f"{task_name}: invalid sidecar deployment evidence: {exc}")
    return errors


def main() -> int:
    errors = [error for key, task in TASKS.items() for error in check(key, task)]
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bind the outer Harness manifest to an immutable scientific submission."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from .path_io import iter_tree_files, path_is_file, read_text, sha256_file
from .schema_validation import validate_harness_manifest, validate_scientific_manifest


def submission_tree_sha256(submission_dir: Path) -> str:
    root = Path(submission_dir).resolve()
    digest = hashlib.sha256()
    for logical_relative, path in iter_tree_files(root):
        relative = logical_relative.as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def submission_payload_sha256(submission_dir: Path) -> str:
    """Hash path/hash records for every payload file except run_manifest.json."""
    root = Path(submission_dir).resolve()
    digest = hashlib.sha256()
    for logical_relative, path in iter_tree_files(root):
        relative = logical_relative.as_posix()
        if relative == "run_manifest.json":
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_cross_manifest_identity(outer: Mapping, inner: Mapping) -> None:
    fields = (
        "run_id", "task_id", "task_version", "grader_version",
        "environment_lock_sha256", "task_config_sha256", "public_inputs_sha256",
        "xtb_executable_sha256",
    )
    mismatches = [field for field in fields if outer.get(field) != inner.get(field)]
    if mismatches:
        raise ValueError(f"outer and inner manifest identity mismatch: {mismatches}")


def _load_inner(submission_dir: Path) -> tuple[Path, dict]:
    manifest_path = Path(submission_dir).resolve() / "run_manifest.json"
    if not path_is_file(manifest_path):
        raise FileNotFoundError("submission/run_manifest.json is required")
    document = json.loads(read_text(manifest_path, encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("scientific run manifest must be a JSON object")
    return manifest_path, document


def link_and_validate_submission(
    outer_manifest: Mapping,
    submission_dir: Path,
    schema_dir: Path,
) -> dict:
    manifest_path, inner = _load_inner(submission_dir)
    validate_scientific_manifest(inner, schema_dir)
    payload_sha256 = submission_payload_sha256(submission_dir)
    if inner.get("submission_sha256") != payload_sha256:
        raise ValueError("scientific submission payload SHA-256 mismatch")
    linked = deepcopy(dict(outer_manifest))
    _validate_cross_manifest_identity(linked, inner)
    linked.update(
        {
            "submission_manifest_path": "submission/run_manifest.json",
            "submission_manifest_sha256": sha256_file(manifest_path),
            "submission_sha256": submission_tree_sha256(submission_dir),
            "submission_payload_sha256": payload_sha256,
        }
    )
    validate_harness_manifest(linked, schema_dir)
    return linked


def verify_linked_submission(
    outer_manifest: Mapping,
    submission_dir: Path,
    schema_dir: Path,
) -> None:
    validate_harness_manifest(outer_manifest, schema_dir)
    if outer_manifest.get("submission_manifest_path") != "submission/run_manifest.json":
        raise ValueError("outer manifest points to the wrong scientific manifest")
    manifest_path, inner = _load_inner(submission_dir)
    validate_scientific_manifest(inner, schema_dir)
    payload_sha256 = submission_payload_sha256(submission_dir)
    if inner.get("submission_sha256") != payload_sha256:
        raise ValueError("scientific submission payload SHA-256 mismatch")
    if sha256_file(manifest_path) != outer_manifest.get("submission_manifest_sha256"):
        raise ValueError("scientific manifest SHA-256 mismatch")
    if submission_tree_sha256(submission_dir) != outer_manifest.get("submission_sha256"):
        raise ValueError("submission SHA-256 mismatch")
    if payload_sha256 != outer_manifest.get("submission_payload_sha256"):
        raise ValueError("outer submission payload SHA-256 mismatch")
    _validate_cross_manifest_identity(outer_manifest, inner)

"""Transactionally rebuild a formal trajectory's derived human view.

This command never changes trajectory.jsonl, final_answer.json, evaluation.json,
or the artifact workspace.  It exists for the narrow case where an older runner
rendered dialogue from pre-canonical in-memory values (for example 1.0) while
the immutable JSONL encoded the equivalent canonical value (1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.trajectory import render_dialogue
from scripts.runtime_support import (
    RuntimeIntegrityError,
    build_file_manifest,
    canonical_json_bytes,
    deterministic_evaluation,
    load_json,
    scan_tree,
    sha256_file,
    verify_file_manifest,
)


FORMAL_MODELS = ("kimi", "deepseek", "glm")
IMMUTABLE_NAMES = ("trajectory.jsonl", "final_answer.json", "evaluation.json")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _events(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeIntegrityError("trajectory event is not an object")
        values.append(value)
    if not values:
        raise RuntimeIntegrityError("trajectory is empty")
    return values


def _replace_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.canonicalizing.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeIntegrityError("canonicalization temporary path already exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def _manifest_with_overrides(root: Path, overrides: dict[str, bytes]) -> dict[str, Any]:
    manifest = build_file_manifest(root, excluded=("artifact_manifest.json",))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeIntegrityError("artifact manifest entries are unavailable")
    found: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeIntegrityError("artifact manifest entry is invalid")
        path = entry.get("path")
        if path in overrides:
            value = overrides[str(path)]
            entry["sha256"] = _sha256_bytes(value)
            entry["size_bytes"] = len(value)
            found.add(str(path))
    if found != set(overrides):
        raise RuntimeIntegrityError("derived artifact is absent from manifest inventory")
    return manifest


def canonicalize(logical_model: str) -> dict[str, Any]:
    if logical_model not in FORMAL_MODELS:
        raise RuntimeIntegrityError("unknown formal model")
    root = ROOT / "trajectories" / logical_model
    required = {
        *IMMUTABLE_NAMES,
        "dialogue.md",
        "metadata.json",
        "artifact_manifest.json",
        "artifacts",
    }
    if not root.is_dir() or any(not (root / name).exists() for name in required):
        raise RuntimeIntegrityError("formal trajectory is incomplete")
    manifest_path = root / "artifact_manifest.json"
    if not verify_file_manifest(root, manifest_path):
        raise RuntimeIntegrityError("existing formal artifact manifest is invalid")

    immutable_before = {name: sha256_file(root / name) for name in IMMUTABLE_NAMES}
    dialogue_path = root / "dialogue.md"
    metadata_path = root / "metadata.json"
    original = {
        "dialogue.md": dialogue_path.read_bytes(),
        "metadata.json": metadata_path.read_bytes(),
        "artifact_manifest.json": manifest_path.read_bytes(),
    }
    dialogue_bytes = render_dialogue(_events(root / "trajectory.jsonl")).encode("utf-8")
    if dialogue_bytes == original["dialogue.md"]:
        return {
            "schema_version": "1.0",
            "status": "PASS",
            "logical_model": logical_model,
            "changed": False,
            "dialogue_sha256": _sha256_bytes(dialogue_bytes),
            "immutable_sha256": immutable_before,
        }

    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("artifact_sha256"), dict):
        raise RuntimeIntegrityError("formal metadata artifact hashes are unavailable")
    metadata["artifact_sha256"].update(
        {
            "dialogue.md": _sha256_bytes(dialogue_bytes).lower(),
            "final_answer.json": immutable_before["final_answer.json"].lower(),
            "trajectory.jsonl": immutable_before["trajectory.jsonl"].lower(),
        }
    )
    metadata_bytes = canonical_json_bytes(metadata)
    manifest_bytes = canonical_json_bytes(
        _manifest_with_overrides(
            root,
            {"dialogue.md": dialogue_bytes, "metadata.json": metadata_bytes},
        )
    )

    try:
        _replace_bytes(dialogue_path, dialogue_bytes)
        _replace_bytes(metadata_path, metadata_bytes)
        _replace_bytes(manifest_path, manifest_bytes)
        immutable_after = {name: sha256_file(root / name) for name in IMMUTABLE_NAMES}
        if immutable_after != immutable_before:
            raise RuntimeIntegrityError("immutable formal evidence changed")
        replay, replay_bytes = deterministic_evaluation(
            final_answer=root / "final_answer.json",
            trajectory=root / "trajectory.jsonl",
            artifacts=root / "artifacts",
        )
        if replay_bytes != (root / "evaluation.json").read_bytes():
            raise RuntimeIntegrityError("evaluator replay changed during canonicalization")
        if not bool(replay.get("eligibility", {}).get("eligible")):
            raise RuntimeIntegrityError("formal trajectory is no longer evaluator-eligible")
        if not verify_file_manifest(root, manifest_path):
            raise RuntimeIntegrityError("rebuilt artifact manifest is invalid")
        if not scan_tree(root)["passed"]:
            raise RuntimeIntegrityError("rebuilt derived artifacts failed security scan")
        if dialogue_path.read_text(encoding="utf-8") != render_dialogue(
            _events(root / "trajectory.jsonl")
        ):
            raise RuntimeIntegrityError("rebuilt dialogue is not trajectory-derived")
    except BaseException:
        for name, value in original.items():
            _replace_bytes(root / name, value)
        raise

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "logical_model": logical_model,
        "changed": True,
        "old_dialogue_sha256": _sha256_bytes(original["dialogue.md"]),
        "dialogue_sha256": _sha256_bytes(dialogue_bytes),
        "immutable_sha256": immutable_before,
        "evaluation_byte_identical": True,
        "artifact_manifest_valid": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonicalize a derived formal dialogue")
    parser.add_argument("--model", required=True, choices=FORMAL_MODELS)
    args = parser.parse_args(argv)
    try:
        report = canonicalize(args.model)
    except (RuntimeIntegrityError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen validation and compact binding for the Gate-8 execution snapshot.

This module intentionally lives in the execution layer.  Formal runs must not
depend on a release-layer validator that can legitimately change after the
campaign.  The package builder independently implements the same inventory
contract and the tests require both implementations to agree byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping


SCHEMA_VERSION = "execution-commitment/2.0"
BINDING_SCHEMA_VERSION = "execution-freeze-binding/2.0"
IDENTITY = {
    "benchmark_version": "2.0.0",
    "case_id": "2_fluorobiphenyl_central_torsion_v1",
    "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
    "task_id": "adaptive_xtb_torsion_profile_agent_v2",
}
ROOT_FILES = frozenset({"TASK.md", "environment.yml", "requirements.txt"})
TOP_LEVEL = frozenset(
    {
        "baselines",
        "benchmark_case",
        "configs",
        "evaluator",
        "evaluator_private",
        "framework",
        "prompts",
        "tools",
    }
)
SCRIPT_EXCLUSIONS = frozenset(
    {
        "package_submission.py",
        "pre_freeze_package_rehearsal.py",
        "release_patch_rehearsal.py",
        "replay_trajectories.py",
        "secret_scan.py",
        "validate_package.py",
        "validate_project.py",
        "verify_old_evidence.py",
    }
)
CONFIG_EXCLUSIONS = frozenset(
    {"campaign_snapshot.json", "release.json", "release_commitment.json"}
)
TOP_LEVEL_WORK_EXCLUSIONS = frozenset(
    {".git", ".test_tmp", "audit_tmp", "outputs", "runs", "runs_failed", "staging"}
)
NESTED_CACHE_EXCLUSIONS = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
_HEX = frozenset("0123456789ABCDEF")


class ExecutionFreezeError(RuntimeError):
    """The execution commitment or its current-tree binding is invalid."""


def _canonical(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_NONCANONICAL") from exc
    return (text + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _link_or_reparse(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or (callable(junction) and junction())
    )


def _excluded(relative: PurePosixPath) -> bool:
    folded = tuple(part.casefold() for part in relative.parts)
    return bool(
        folded
        and (
            folded[0]
            in {part.casefold() for part in TOP_LEVEL_WORK_EXCLUSIONS}
            or set(folded)
            & {part.casefold() for part in NESTED_CACHE_EXCLUSIONS}
        )
    )


def _safe_relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_PATH_ESCAPE") from exc
    value = relative.as_posix()
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
    ):
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_PATH_UNSAFE")
    return value


def _collect(root: Path, base: Path) -> list[Path]:
    if not base.exists():
        return []
    if _link_or_reparse(base):
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_LINK_OR_REPARSE")
    if base.is_file():
        return [base]
    if not base.is_dir():
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_SPECIAL_FILE")
    result: list[Path] = []
    for candidate in base.rglob("*"):
        relative = candidate.relative_to(root)
        if _link_or_reparse(candidate):
            raise ExecutionFreezeError("EXECUTION_COMMITMENT_LINK_OR_REPARSE")
        if _excluded(PurePosixPath(relative.as_posix())):
            continue
        if candidate.is_file():
            result.append(candidate)
        elif candidate.exists() and not candidate.is_dir():
            raise ExecutionFreezeError("EXECUTION_COMMITMENT_SPECIAL_FILE")
    return result


def execution_file_entries(root: Path) -> list[dict[str, Any]]:
    """Return the frozen execution inventory without release-layer imports."""

    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for name in sorted(ROOT_FILES):
        path = resolved / name
        if path.is_file():
            files.append(path)
    for name in sorted(TOP_LEVEL):
        for path in _collect(resolved, resolved / name):
            relative = path.relative_to(resolved)
            if (
                relative.parts[0] == "configs"
                and relative.as_posix().split("/", 1)[-1] in CONFIG_EXCLUSIONS
            ):
                continue
            files.append(path)
    for path in _collect(resolved, resolved / "scripts"):
        if path.name not in SCRIPT_EXCLUSIONS:
            files.append(path)

    by_name: dict[str, Path] = {}
    folded: set[str] = set()
    for path in files:
        if not path.is_file() or _link_or_reparse(path):
            raise ExecutionFreezeError("EXECUTION_COMMITMENT_FILE_UNSAFE")
        name = _safe_relative(resolved, path)
        if name in by_name or name.casefold() in folded:
            raise ExecutionFreezeError("EXECUTION_COMMITMENT_CASE_COLLISION")
        by_name[name] = path
        folded.add(name.casefold())
    return [
        {
            "path": name,
            "size_bytes": by_name[name].stat().st_size,
            "sha256": _sha_file(by_name[name]),
        }
        for name in sorted(by_name)
    ]


def _load_closed(path: Path) -> tuple[dict[str, Any], bytes]:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicates.append(key)
            value[key] = item
        return value

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_FILE_INVALID") from exc
    if duplicates or not isinstance(value, dict) or raw != _canonical(value):
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_FILE_INVALID")
    return value, raw


def _validate_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_FILES")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    folded: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ExecutionFreezeError("EXECUTION_COMMITMENT_FILES")
        path = raw["path"]
        digest = raw["sha256"]
        size = raw["size_bytes"]
        pure = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or pure is None
            or pure.is_absolute()
            or pure.as_posix() != path
            or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
            or path in names
            or path.casefold() in folded
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise ExecutionFreezeError("EXECUTION_COMMITMENT_FILES")
        names.add(path)
        folded.add(path.casefold())
        result.append({"path": path, "size_bytes": size, "sha256": digest})
    if result != sorted(result, key=lambda item: item["path"]):
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_FILES")
    return result


def load_execution_freeze_binding(root: Path) -> dict[str, Any]:
    """Validate the root commitment against the current tree and bind it."""

    resolved = root.resolve(strict=True)
    payload, raw = _load_closed(resolved / "execution_commitment.json")
    if set(payload) != {
        "commitment_kind",
        "content_sha256",
        "file_count",
        "files",
        "identity",
        "schema_version",
        "total_bytes",
    }:
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_SCHEMA")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["commitment_kind"] != "execution"
        or payload["identity"] != IDENTITY
    ):
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_IDENTITY")
    entries = _validate_entries(payload["files"])
    current = execution_file_entries(resolved)
    hash_value = {
        "commitment_kind": "execution",
        "identity": IDENTITY,
        "files": entries,
    }
    digest = _sha_bytes(_canonical(hash_value))
    if (
        entries != current
        or payload["file_count"] != len(entries)
        or payload["total_bytes"]
        != sum(int(item["size_bytes"]) for item in entries)
        or payload["content_sha256"] != digest
    ):
        raise ExecutionFreezeError("EXECUTION_COMMITMENT_MISMATCH")
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "content_sha256": digest,
        "commitment_file_sha256": _sha_bytes(raw),
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
    }


def validate_execution_freeze_binding(value: Any) -> dict[str, Any]:
    """Validate the closed compact binding carried by a formal trajectory."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "content_sha256",
        "commitment_file_sha256",
        "file_count",
        "total_bytes",
    }:
        raise ExecutionFreezeError("EXECUTION_FREEZE_BINDING_SCHEMA")
    if value["schema_version"] != BINDING_SCHEMA_VERSION:
        raise ExecutionFreezeError("EXECUTION_FREEZE_BINDING_IDENTITY")
    for name in ("content_sha256", "commitment_file_sha256"):
        digest = value[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise ExecutionFreezeError("EXECUTION_FREEZE_BINDING_DIGEST")
    for name in ("file_count", "total_bytes"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ExecutionFreezeError("EXECUTION_FREEZE_BINDING_COUNT")
    return dict(value)


__all__ = [
    "BINDING_SCHEMA_VERSION",
    "ExecutionFreezeError",
    "execution_file_entries",
    "load_execution_freeze_binding",
    "validate_execution_freeze_binding",
]

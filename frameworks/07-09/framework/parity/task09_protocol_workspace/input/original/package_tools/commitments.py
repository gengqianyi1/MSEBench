"""Deterministic v2 execution and release commitments.

The execution commitment freezes only code and assets that can influence a
model run or its scientific evaluation.  Packaging, scanners, reports, tests,
and formal result files are deliberately excluded.  The release commitment
then freezes the release layer and binds both the immutable execution
commitment and every result artifact.

Neither commitment contains timestamps, host paths, or filesystem metadata
other than relative path, byte length, and SHA-256.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable, Mapping

from package_tools.provenance import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


EXECUTION_SCHEMA_VERSION = "execution-commitment/2.0"
RELEASE_SCHEMA_VERSION = "release-commitment/2.0"
V2_IDENTITY: dict[str, str] = {
    "benchmark_version": "2.0.0",
    "case_id": "2_fluorobiphenyl_central_torsion_v1",
    "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
    "task_id": "adaptive_xtb_torsion_profile_agent_v2",
}

EXECUTION_ROOT_FILES = frozenset({"TASK.md", "environment.yml", "requirements.txt"})
EXECUTION_TOP_LEVEL = frozenset(
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
RELEASE_ROOT_FILES = frozenset(
    {
        ".env.example",
        ".gitignore",
        "LICENSE",
        "LICENSES.md",
        "NOTICE",
        "README.md",
        "TASK.md",
        "environment.yml",
        "pytest.ini",
        "requirements.txt",
        "run_all.ps1",
        "run_all.sh",
    }
)
RELEASE_TOP_LEVEL = frozenset({"package_tools", "reports", "tests"})
RELEASE_SCRIPT_NAMES = frozenset(
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
EXECUTION_SCRIPT_EXCLUSIONS = RELEASE_SCRIPT_NAMES
EXECUTION_CONFIG_EXCLUSIONS = frozenset(
    {
        "campaign_snapshot.json",
        "release.json",
        "release_commitment.json",
    }
)
RELEASE_CONFIG_NAMES = frozenset({"release.json"})
FORBIDDEN_NAMES = frozenset({".env", "MANIFEST.json", "release_commitment.json"})
TOP_LEVEL_WORK_EXCLUSIONS = frozenset(
    {
        ".git",
        ".test_tmp",
        "audit_tmp",
        "outputs",
        "runs",
        "runs_failed",
        "staging",
    }
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


class CommitmentError(ValueError):
    """A commitment inventory, binding, or payload failed closed validation."""


def _is_link_or_reparse(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        or (callable(is_junction) and is_junction())
    )


def _safe_relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CommitmentError("COMMITMENT_PATH_ESCAPE") from exc
    posix = relative.as_posix()
    pure = PurePosixPath(posix)
    if (
        not posix
        or "\\" in posix
        or pure.is_absolute()
        or pure.as_posix() != posix
        or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
    ):
        raise CommitmentError("COMMITMENT_PATH_UNSAFE")
    return posix


def _excluded(relative: PurePosixPath) -> bool:
    folded = tuple(part.casefold() for part in relative.parts)
    if not folded:
        return False
    return bool(
        folded[0]
        in {part.casefold() for part in TOP_LEVEL_WORK_EXCLUSIONS}
        or set(folded)
        & {part.casefold() for part in NESTED_CACHE_EXCLUSIONS}
    )


def _collect_tree(root: Path, base: Path) -> list[Path]:
    resolved_root = root.resolve(strict=True)
    if not base.exists():
        return []
    if _is_link_or_reparse(base):
        raise CommitmentError("COMMITMENT_LINK_OR_REPARSE")
    if base.is_file():
        return [base]
    if not base.is_dir():
        raise CommitmentError("COMMITMENT_SPECIAL_FILE")
    files: list[Path] = []
    for candidate in base.rglob("*"):
        relative = candidate.relative_to(resolved_root)
        if _is_link_or_reparse(candidate):
            raise CommitmentError("COMMITMENT_LINK_OR_REPARSE")
        if _excluded(PurePosixPath(relative.as_posix())):
            continue
        if candidate.is_file():
            files.append(candidate)
        elif candidate.exists() and not candidate.is_dir():
            raise CommitmentError("COMMITMENT_SPECIAL_FILE")
    return files


def _dedupe_sort(root: Path, files: Iterable[Path]) -> list[Path]:
    resolved = root.resolve(strict=True)
    by_path: dict[str, Path] = {}
    casefolded: set[str] = set()
    for path in files:
        if not path.is_file() or _is_link_or_reparse(path):
            raise CommitmentError("COMMITMENT_FILE_UNSAFE")
        relative = _safe_relative(resolved, path)
        folded = relative.casefold()
        if relative in by_path or folded in casefolded:
            raise CommitmentError("COMMITMENT_CASE_COLLISION")
        name = PurePosixPath(relative).name
        if name.casefold() in {
            item.casefold() for item in FORBIDDEN_NAMES
        } or (
            name.casefold().startswith(".env")
            and relative != ".env.example"
        ):
            raise CommitmentError("COMMITMENT_FORBIDDEN_FILE")
        by_path[relative] = path
        casefolded.add(folded)
    return [by_path[name] for name in sorted(by_path)]


def discover_execution_files(root: Path) -> list[Path]:
    """Discover only execution/scientific files; exclude all release layers."""

    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for name in sorted(EXECUTION_ROOT_FILES):
        path = resolved / name
        if path.is_file():
            files.append(path)
    for name in sorted(EXECUTION_TOP_LEVEL):
        base = resolved / name
        for path in _collect_tree(resolved, base):
            relative = path.relative_to(resolved)
            if (
                relative.parts[0] == "configs"
                and relative.as_posix().split("/", 1)[-1]
                in EXECUTION_CONFIG_EXCLUSIONS
            ):
                continue
            files.append(path)
    scripts = resolved / "scripts"
    for path in _collect_tree(resolved, scripts):
        if path.name in EXECUTION_SCRIPT_EXCLUSIONS:
            continue
        files.append(path)
    return _dedupe_sort(resolved, files)


def discover_release_files(root: Path) -> list[Path]:
    """Discover release implementation, release tests, and packaged documents."""

    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for name in sorted(RELEASE_ROOT_FILES):
        path = resolved / name
        if path.is_file():
            files.append(path)
    for name in sorted(RELEASE_TOP_LEVEL):
        files.extend(_collect_tree(resolved, resolved / name))
    scripts = resolved / "scripts"
    for path in _collect_tree(resolved, scripts):
        if path.name in RELEASE_SCRIPT_NAMES:
            files.append(path)
    configs = resolved / "configs"
    for name in sorted(RELEASE_CONFIG_NAMES):
        path = configs / name
        if path.is_file():
            files.append(path)
    return _dedupe_sort(resolved, files)


def discover_result_files(root: Path) -> list[Path]:
    """Discover all immutable Baseline/formal trajectory result artifacts."""

    resolved = root.resolve(strict=True)
    trajectories = resolved / "trajectories"
    return _dedupe_sort(resolved, _collect_tree(resolved, trajectories))


def _file_entries(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    resolved = root.resolve(strict=True)
    return [
        {
            "path": _safe_relative(resolved, path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _dedupe_sort(resolved, files)
    ]


def _identity(value: Mapping[str, Any] | None) -> dict[str, str]:
    identity = dict(V2_IDENTITY if value is None else value)
    if identity != V2_IDENTITY:
        raise CommitmentError("COMMITMENT_IDENTITY_MISMATCH")
    return dict(V2_IDENTITY)


def _execution_hash_value(
    identity: Mapping[str, str],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "commitment_kind": "execution",
        "identity": dict(identity),
        "files": files,
    }


def build_execution_commitment(
    root: Path,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a timestamp-free execution commitment for the current tree."""

    resolved = root.resolve(strict=True)
    entries = _file_entries(resolved, discover_execution_files(resolved))
    frozen_identity = _identity(identity)
    content_hash = sha256_bytes(
        canonical_json_bytes(_execution_hash_value(frozen_identity, entries))
    )
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "commitment_kind": "execution",
        "identity": frozen_identity,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
        "files": entries,
        "content_sha256": content_hash,
    }


def _closed_mapping(
    value: Any,
    fields: frozenset[str],
    error: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise CommitmentError(error)
    return value


def _validate_entry_list(value: Any, error: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise CommitmentError(error)
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    folded_names: set[str] = set()
    for raw in value:
        item = _closed_mapping(
            raw,
            frozenset({"path", "sha256", "size_bytes"}),
            error,
        )
        path = item["path"]
        if not isinstance(path, str):
            raise CommitmentError(error)
        pure = PurePosixPath(path)
        if (
            not path
            or "\\" in path
            or pure.is_absolute()
            or pure.as_posix() != path
            or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
            or path in names
            or path.casefold() in folded_names
        ):
            raise CommitmentError(error)
        size = item["size_bytes"]
        digest = item["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789ABCDEF" for char in digest)
        ):
            raise CommitmentError(error)
        names.add(path)
        folded_names.add(path.casefold())
        normalized.append(
            {"path": path, "size_bytes": size, "sha256": digest}
        )
    if normalized != sorted(normalized, key=lambda item: str(item["path"])):
        raise CommitmentError(error)
    return normalized


def validate_execution_commitment(
    root: Path,
    payload: Any,
) -> dict[str, Any]:
    """Raise on any execution inventory or binding mismatch."""

    value = _closed_mapping(
        payload,
        frozenset(
            {
                "commitment_kind",
                "content_sha256",
                "file_count",
                "files",
                "identity",
                "schema_version",
                "total_bytes",
            }
        ),
        "EXECUTION_COMMITMENT_SCHEMA",
    )
    if (
        value["schema_version"] != EXECUTION_SCHEMA_VERSION
        or value["commitment_kind"] != "execution"
        or value["identity"] != V2_IDENTITY
    ):
        raise CommitmentError("EXECUTION_COMMITMENT_IDENTITY")
    entries = _validate_entry_list(
        value["files"], "EXECUTION_COMMITMENT_FILES"
    )
    expected = build_execution_commitment(root)
    if (
        value["file_count"] != len(entries)
        or value["total_bytes"]
        != sum(int(item["size_bytes"]) for item in entries)
        or entries != expected["files"]
        or value["content_sha256"] != expected["content_sha256"]
    ):
        raise CommitmentError("EXECUTION_COMMITMENT_MISMATCH")
    return {
        "passed": True,
        "content_sha256": expected["content_sha256"],
        "commitment_file_sha256": sha256_bytes(canonical_json_bytes(dict(value))),
        "file_count": expected["file_count"],
        "total_bytes": expected["total_bytes"],
    }


def verify_execution_commitment(root: Path, payload: Any) -> dict[str, Any]:
    try:
        return validate_execution_commitment(root, payload)
    except CommitmentError as exc:
        return {"passed": False, "error": str(exc)}


def _load_canonical_commitment(path: Path, error: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommitmentError(error) from exc
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise CommitmentError(error)
    return payload


def _resolve_execution_payload(
    root: Path,
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    path = root / "execution_commitment.json"
    if payload is None:
        value = _load_canonical_commitment(
            path, "EXECUTION_COMMITMENT_FILE_INVALID"
        )
    else:
        value = dict(payload)
        if path.is_file():
            file_value = _load_canonical_commitment(
                path, "EXECUTION_COMMITMENT_FILE_INVALID"
            )
            if file_value != value:
                raise CommitmentError("EXECUTION_COMMITMENT_FILE_MISMATCH")
    validation = validate_execution_commitment(root, value)
    return value, validation["commitment_file_sha256"]


def _release_hash_value(
    identity: Mapping[str, str],
    execution_binding: Mapping[str, Any],
    release_files: list[dict[str, Any]],
    result_files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "commitment_kind": "release",
        "identity": dict(identity),
        "execution_binding": dict(execution_binding),
        "release_files": release_files,
        "result_files": result_files,
    }


def build_release_commitment(
    root: Path,
    *,
    execution_payload: Mapping[str, Any] | None = None,
    result_files: Iterable[Path] | None = None,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a release commitment bound to execution and result artifacts."""

    resolved = root.resolve(strict=True)
    frozen_identity = _identity(identity)
    execution, execution_file_sha = _resolve_execution_payload(
        resolved, execution_payload
    )
    release_entries = _file_entries(
        resolved, discover_release_files(resolved)
    )
    result_entries = _file_entries(
        resolved,
        discover_result_files(resolved)
        if result_files is None
        else result_files,
    )
    execution_binding = {
        "content_sha256": execution["content_sha256"],
        "commitment_file_sha256": execution_file_sha,
        "file_count": execution["file_count"],
        "total_bytes": execution["total_bytes"],
    }
    content_hash = sha256_bytes(
        canonical_json_bytes(
            _release_hash_value(
                frozen_identity,
                execution_binding,
                release_entries,
                result_entries,
            )
        )
    )
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "commitment_kind": "release",
        "identity": frozen_identity,
        "execution_binding": execution_binding,
        "release_file_count": len(release_entries),
        "release_total_bytes": sum(
            int(item["size_bytes"]) for item in release_entries
        ),
        "release_files": release_entries,
        "result_file_count": len(result_entries),
        "result_total_bytes": sum(
            int(item["size_bytes"]) for item in result_entries
        ),
        "result_files": result_entries,
        "content_sha256": content_hash,
    }


def validate_release_commitment(
    root: Path,
    payload: Any,
) -> dict[str, Any]:
    """Raise when release code, execution, or any result file changed."""

    value = _closed_mapping(
        payload,
        frozenset(
            {
                "commitment_kind",
                "content_sha256",
                "execution_binding",
                "identity",
                "release_file_count",
                "release_files",
                "release_total_bytes",
                "result_file_count",
                "result_files",
                "result_total_bytes",
                "schema_version",
            }
        ),
        "RELEASE_COMMITMENT_SCHEMA",
    )
    if (
        value["schema_version"] != RELEASE_SCHEMA_VERSION
        or value["commitment_kind"] != "release"
        or value["identity"] != V2_IDENTITY
    ):
        raise CommitmentError("RELEASE_COMMITMENT_IDENTITY")
    _closed_mapping(
        value["execution_binding"],
        frozenset(
            {
                "commitment_file_sha256",
                "content_sha256",
                "file_count",
                "total_bytes",
            }
        ),
        "RELEASE_EXECUTION_BINDING_SCHEMA",
    )
    release_entries = _validate_entry_list(
        value["release_files"], "RELEASE_COMMITMENT_RELEASE_FILES"
    )
    result_entries = _validate_entry_list(
        value["result_files"], "RELEASE_COMMITMENT_RESULT_FILES"
    )
    expected = build_release_commitment(root)
    if (
        value["execution_binding"] != expected["execution_binding"]
        or value["release_file_count"] != len(release_entries)
        or value["release_total_bytes"]
        != sum(int(item["size_bytes"]) for item in release_entries)
        or value["result_file_count"] != len(result_entries)
        or value["result_total_bytes"]
        != sum(int(item["size_bytes"]) for item in result_entries)
        or release_entries != expected["release_files"]
        or result_entries != expected["result_files"]
        or value["content_sha256"] != expected["content_sha256"]
    ):
        raise CommitmentError("RELEASE_COMMITMENT_MISMATCH")
    return {
        "passed": True,
        "content_sha256": expected["content_sha256"],
        "commitment_file_sha256": sha256_bytes(canonical_json_bytes(dict(value))),
        "execution_content_sha256": expected["execution_binding"][
            "content_sha256"
        ],
        "release_file_count": expected["release_file_count"],
        "result_file_count": expected["result_file_count"],
    }


def verify_release_commitment(root: Path, payload: Any) -> dict[str, Any]:
    try:
        return validate_release_commitment(root, payload)
    except CommitmentError as exc:
        return {"passed": False, "error": str(exc)}


def write_execution_commitment(
    root: Path,
    destination: Path | None = None,
    *,
    exclusive: bool = True,
) -> dict[str, Any]:
    payload = build_execution_commitment(root)
    target = destination or (root / "execution_commitment.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    try:
        with target.open(mode) as handle:
            handle.write(canonical_json_bytes(payload))
    except FileExistsError as exc:
        raise CommitmentError("EXECUTION_COMMITMENT_ALREADY_EXISTS") from exc
    return payload


def write_release_commitment(
    root: Path,
    destination: Path | None = None,
    *,
    exclusive: bool = True,
) -> dict[str, Any]:
    payload = build_release_commitment(root)
    target = destination or (root / "release_commitment.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    try:
        with target.open(mode) as handle:
            handle.write(canonical_json_bytes(payload))
    except FileExistsError as exc:
        raise CommitmentError("RELEASE_COMMITMENT_ALREADY_EXISTS") from exc
    return payload


__all__ = [
    "CommitmentError",
    "EXECUTION_SCHEMA_VERSION",
    "RELEASE_SCHEMA_VERSION",
    "V2_IDENTITY",
    "build_execution_commitment",
    "build_release_commitment",
    "discover_execution_files",
    "discover_release_files",
    "discover_result_files",
    "validate_execution_commitment",
    "validate_release_commitment",
    "verify_execution_commitment",
    "verify_release_commitment",
    "write_execution_commitment",
    "write_release_commitment",
]

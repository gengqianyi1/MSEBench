"""Fail-closed construction of the exact files visible to a benchmark model."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import shutil
import stat
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class ModelVisibleBoundaryError(RuntimeError):
    """Raised when a file crosses the scorer/model trust boundary."""


ALLOWLIST_NAME = "model_visible.allowlist.json"
SURFACE_ROOTS = ("data", "submission_contract")
BASE_FILES = frozenset({"question.md", "final_answer.schema.json", "config/task.yaml"})
ALLOWED_EXTENSIONS = frozenset({".md", ".json", ".yaml", ".yml", ".csv", ".xyz"})
ARCHIVE_EXTENSIONS = frozenset({
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".whl",
})
FORBIDDEN_PARTS = frozenset({
    ".superpowers", "private_runtime", "private_data", "private_grader", "golden",
    "baseline", "runs", "tests", "validation", "__pycache__",
})
FORBIDDEN_NAMES = frozenset({
    "generate_golden.py", "build_median_reference.py", "verify_golden.py",
    "run_task04_histidine_preflight.py", "task-5-report.md",
})
FORBIDDEN_CONTENT = (
    b"private_runtime", b".superpowers", b"run_task04_histidine_preflight",
    b"generate_golden.py", b"classify_condition_route", b"classify_histidine_geometry",
    b"aggregate_histidine_reference", b"build_histidine_candidates", b"24+c",
    b"all twelve deterministic", b"product hessians were not assigned",
    b"neutral_npi_h", b"neutral_ntau_h", b"zwitterionic_npi_h", b"zwitterionic_ntau_h",
)
BASE64_TOKEN = re.compile(rb"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(junction and junction())


def _normalize_member(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ModelVisibleBoundaryError(f"forbidden model-visible path: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise ModelVisibleBoundaryError(f"non-NFC model-visible path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ModelVisibleBoundaryError(f"forbidden model-visible path: {value!r}")
    normalized = path.as_posix()
    lowered_parts = {part.casefold() for part in path.parts}
    if lowered_parts & FORBIDDEN_PARTS or path.name.casefold() in FORBIDDEN_NAMES:
        raise ModelVisibleBoundaryError(f"forbidden private path: {normalized}")
    if normalized not in BASE_FILES and path.parts[0] not in SURFACE_ROOTS:
        raise ModelVisibleBoundaryError(f"path is outside the exact model-visible surface: {normalized}")
    suffix = path.suffix.casefold()
    if suffix in ARCHIVE_EXTENSIONS or suffix not in ALLOWED_EXTENSIONS:
        raise ModelVisibleBoundaryError(f"archive or disallowed extension in model-visible path: {normalized}")
    return normalized


def _validate_content(path: str, payload: bytes) -> None:
    if payload.startswith((b"PK\x03\x04", b"\x1f\x8b", b"7z\xbc\xaf\x27\x1c", b"Rar!")):
        raise ModelVisibleBoundaryError(f"nested archive payload is forbidden: {path}")
    lowered = payload.lower()
    for marker in FORBIDDEN_CONTENT:
        if marker in lowered:
            raise ModelVisibleBoundaryError(
                f"forbidden private recipe marker {marker!r} in model-visible file {path}"
            )
    for token in BASE64_TOKEN.findall(payload):
        try:
            decoded = base64.b64decode(token, validate=True).lower()
        except (ValueError, binascii.Error):
            continue
        if any(marker in decoded for marker in FORBIDDEN_CONTENT):
            raise ModelVisibleBoundaryError(f"forbidden decoded private recipe in {path}")


def _load_allowlist(root: Path) -> list[dict[str, Any]]:
    path = root / ALLOWLIST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelVisibleBoundaryError(f"exact model-visible allowlist is missing or invalid: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "1.0.0":
        raise ModelVisibleBoundaryError("exact model-visible allowlist schema is invalid")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ModelVisibleBoundaryError("exact model-visible allowlist contains no files")
    result: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ModelVisibleBoundaryError("exact model-visible allowlist record is invalid")
        relative = _normalize_member(str(item.get("path") or ""))
        collision = unicodedata.normalize("NFC", relative).casefold()
        if collision in seen:
            raise ModelVisibleBoundaryError(f"casefold/NFC allowlist collision: {seen[collision]} vs {relative}")
        seen[collision] = relative
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ModelVisibleBoundaryError(f"invalid reviewed SHA-256 for {relative}")
        result.append({"path": relative, "sha256": digest})
    if not BASE_FILES.issubset({item["path"] for item in result}):
        raise ModelVisibleBoundaryError("exact model-visible allowlist omits a required base file")
    return sorted(result, key=lambda item: item["path"])


def _assert_safe_ancestors(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if _is_link_or_junction(current):
            raise ModelVisibleBoundaryError(f"symlink/junction ancestor is forbidden: {relative}")
    if not current.is_file():
        raise ModelVisibleBoundaryError(f"reviewed model-visible file is missing: {relative}")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ModelVisibleBoundaryError(f"reviewed model-visible file escapes its root: {relative}") from exc
    return current


def _surface_files(root: Path) -> set[str]:
    observed = set(BASE_FILES)
    for tree in SURFACE_ROOTS:
        tree_root = root / tree
        if not tree_root.is_dir() or _is_link_or_junction(tree_root):
            raise ModelVisibleBoundaryError(f"required model-visible tree is missing or unsafe: {tree}")
        for path in sorted(tree_root.rglob("*")):
            if _is_link_or_junction(path):
                raise ModelVisibleBoundaryError(f"symlink/junction is forbidden: {path}")
            if path.is_file():
                observed.add(_normalize_member(path.relative_to(root).as_posix()))
    return observed


def model_visible_manifest(task_dir: str | Path) -> list[dict[str, Any]]:
    """Return files authorized by the task-specific reviewed path+SHA allowlist."""

    original_root = Path(task_dir).absolute()
    if _is_link_or_junction(original_root):
        raise ModelVisibleBoundaryError("model-visible task root may not be a symlink or junction")
    root = original_root.resolve(strict=True)
    reviewed = _load_allowlist(root)
    reviewed_paths = {str(item["path"]) for item in reviewed}
    observed_paths = _surface_files(root)
    if observed_paths != reviewed_paths:
        raise ModelVisibleBoundaryError(
            f"unreviewed or missing model-visible paths; extra={sorted(observed_paths-reviewed_paths)}, "
            f"missing={sorted(reviewed_paths-observed_paths)}"
        )
    manifest: list[dict[str, Any]] = []
    for item in reviewed:
        relative = str(item["path"])
        path = _assert_safe_ancestors(root, relative)
        payload = path.read_bytes()
        digest = _digest(payload)
        if digest != item["sha256"]:
            raise ModelVisibleBoundaryError(f"reviewed model-visible SHA-256 mismatch: {relative}")
        _validate_content(relative, payload)
        manifest.append({"path": relative, "sha256": digest, "size": len(payload)})
    return manifest


def stage_model_workspace(task_dir: str | Path, workspace: str | Path) -> None:
    source = Path(task_dir).resolve(strict=True)
    destination = Path(workspace).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ModelVisibleBoundaryError(f"model workspace must be empty before staging: {destination}")
    for item in model_visible_manifest(source):
        relative = str(item["path"])
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.joinpath(*PurePosixPath(relative).parts), target)
        if _digest(target.read_bytes()) != item["sha256"]:
            raise ModelVisibleBoundaryError(f"staged model-visible hash mismatch: {relative}")


def build_model_visible_zip(task_dir: str | Path, archive: str | Path) -> list[dict[str, Any]]:
    root = Path(task_dir).resolve(strict=True)
    manifest = model_visible_manifest(root)
    target = Path(archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.comment = json.dumps(
            {"schema_version": "1.0.0", "files": manifest},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for item in manifest:
            relative = str(item["path"])
            payload = root.joinpath(*PurePosixPath(relative).parts).read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            handle.writestr(info, payload)
    if validate_model_visible_zip(target) != manifest:
        raise ModelVisibleBoundaryError("model-visible ZIP replay mismatch")
    return manifest


def validate_model_visible_zip(archive: str | Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    with zipfile.ZipFile(Path(archive)) as handle:
        for info in handle.infolist():
            if info.is_dir():
                raise ModelVisibleBoundaryError(f"nonregular ZIP directory entry is forbidden: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ModelVisibleBoundaryError(f"symlink ZIP member is forbidden: {info.filename}")
            if mode not in {0, stat.S_IFREG}:
                raise ModelVisibleBoundaryError(f"nonregular ZIP member is forbidden: {info.filename}")
            relative = _normalize_member(info.filename)
            collision = unicodedata.normalize("NFC", relative).casefold()
            if collision in seen:
                raise ModelVisibleBoundaryError(f"casefold/NFC ZIP collision: {seen[collision]} vs {relative}")
            seen[collision] = relative
            payload = handle.read(info)
            _validate_content(relative, payload)
            manifest.append({"path": relative, "sha256": _digest(payload), "size": len(payload)})
        try:
            authorization = json.loads(handle.comment.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelVisibleBoundaryError("ZIP lacks its exact reviewed manifest") from exc
        expected = authorization.get("files") if isinstance(authorization, Mapping) else None
        if authorization.get("schema_version") != "1.0.0" or not isinstance(expected, list):
            raise ModelVisibleBoundaryError("ZIP reviewed manifest is invalid")
    manifest = sorted(manifest, key=lambda item: str(item["path"]))
    if manifest != expected:
        raise ModelVisibleBoundaryError("ZIP members/hashes do not equal the exact reviewed allowlist")
    return manifest

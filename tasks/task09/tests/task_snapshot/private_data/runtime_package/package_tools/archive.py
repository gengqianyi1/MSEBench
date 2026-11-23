from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any

import yaml

from package_tools.manifest import (
    file_entries,
    sha256_file,
    validate_manifest,
    write_manifest,
)
from package_tools.policy import (
    EXPECTED_ARCHIVE_FILENAME,
    EXPECTED_BENCHMARK_VERSION,
    EXPECTED_CASE_ID,
    EXPECTED_PROTOCOL,
    EXPECTED_SUBMISSION_NAME,
    EXPECTED_TASK_ID,
    collect_allowlisted_files,
    collect_strict_release_tree,
    required_delivery_paths,
)
from package_tools.scanners import (
    scan_forbidden_names,
    scan_runtime_absolute_paths,
    scan_secrets,
)


def _is_link_or_reparse(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & 0x400
        or (callable(is_junction) and is_junction())
    )


def preflight_source(root: Path) -> dict[str, Any]:
    files = collect_allowlisted_files(root)
    relative = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(required_delivery_paths() - relative)
    links = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if _is_link_or_reparse(path)
    ]
    secrets = scan_secrets(root, files)
    absolute_failures, test_literals = scan_runtime_absolute_paths(root, files)
    forbidden = scan_forbidden_names(root, files)
    return {
        "passed": not (missing or links or secrets or absolute_failures or forbidden),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "missing_required": missing,
        "links_or_reparse_points": links,
        "secret_findings": secrets,
        "runtime_absolute_path_findings": absolute_failures,
        "expected_security_test_path_literals": test_literals,
        "forbidden_findings": forbidden,
        "files": files,
    }


def stage_allowlisted_files(root: Path, staging: Path) -> dict[str, Any]:
    if staging.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_STAGING")
    preflight = preflight_source(root)
    if not preflight["passed"]:
        raise RuntimeError("SOURCE_PREFLIGHT_FAILED")
    staging.mkdir(parents=True, exist_ok=False)
    source_files: list[Path] = preflight.pop("files")
    for source in source_files:
        relative = source.relative_to(root)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return preflight


def write_staged_manifest(staging: Path) -> dict[str, Any]:
    """Write MANIFEST.json only after every staged payload mutation is final."""

    if not staging.is_dir():
        raise FileNotFoundError("STAGING_MISSING")
    if (staging / "MANIFEST.json").exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_STAGED_MANIFEST")
    staging_resolved = staging.resolve(strict=True)

    task = yaml.safe_load(
        (staging_resolved / "configs" / "task.yaml").read_text(encoding="utf-8")
    )
    release = json.loads(
        (staging_resolved / "configs" / "release.json").read_text(encoding="utf-8")
    )
    task_identity = task.get("task", {}) if isinstance(task, dict) else {}
    if (
        not isinstance(task_identity, dict)
        or task_identity.get("id") != EXPECTED_TASK_ID
        or task_identity.get("benchmark_version")
        != EXPECTED_BENCHMARK_VERSION
        or task_identity.get("case_id") != EXPECTED_CASE_ID
        or task_identity.get("protocol") != EXPECTED_PROTOCOL
    ):
        raise RuntimeError("STAGED_TASK_IDENTITY_MISMATCH")
    if (
        not isinstance(release, dict)
        or release.get("schema_version") != "2.0"
        or release.get("task_id") != EXPECTED_TASK_ID
        or release.get("benchmark_version") != EXPECTED_BENCHMARK_VERSION
        or release.get("protocol") != EXPECTED_PROTOCOL
    ):
        raise RuntimeError("STAGED_RELEASE_IDENTITY_MISMATCH")
    release_status = release.get("status")
    if (
        release_status not in {"PASS", "FAIL", "BLOCKED"}
        or not isinstance(release.get("models"), dict)
        or release.get("release_ready") is not (release_status == "PASS")
    ):
        raise RuntimeError("STAGED_RELEASE_STATUS_INVALID")
    staged_files = collect_strict_release_tree(staging_resolved)
    staged_relative = {
        path.relative_to(staging_resolved).as_posix() for path in staged_files
    }
    missing = sorted(required_delivery_paths() - staged_relative)
    if missing:
        raise RuntimeError("STAGED_REQUIRED_DELIVERY_PATHS_MISSING")
    manifest = {
        "schema_version": "1.0",
        "submission_name": EXPECTED_SUBMISSION_NAME,
        "archive_filename": EXPECTED_ARCHIVE_FILENAME,
        "archive_layout": "flat_project_root",
        "task_id": EXPECTED_TASK_ID,
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
        "case_id": EXPECTED_CASE_ID,
        "protocol": EXPECTED_PROTOCOL,
        "models": release["models"],
        "release_status": release["status"],
        "file_count_excluding_manifest": len(staged_files),
        "total_bytes_excluding_manifest": sum(path.stat().st_size for path in staged_files),
        "excluded_categories": [
            ".env and API credentials",
            "configs/models.yaml",
            "runs and runs_failed",
            "outputs and staging",
            "raw provider responses and debug dumps",
            "caches and virtual environments",
            "old archives, audit_tmp, test temp and staging",
        ],
        "files": file_entries(staging_resolved, staged_files),
    }
    write_manifest(staging_resolved / "MANIFEST.json", manifest)
    return {
        "staged_file_count": len(staged_files) + 1,
        "manifest_entries": len(staged_files),
    }


def stage_project(root: Path, staging: Path) -> dict[str, Any]:
    """Backward-compatible one-step staging for callers needing no mutation."""

    preflight = stage_allowlisted_files(root, staging)
    manifest = write_staged_manifest(staging)
    return {**preflight, **manifest}


def build_flat_archive(staging: Path, archive_path: Path) -> dict[str, Any]:
    if archive_path.name != EXPECTED_ARCHIVE_FILENAME:
        raise ValueError(
            f"ARCHIVE_FILENAME_MUST_BE:{EXPECTED_ARCHIVE_FILENAME}"
        )
    if archive_path.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_ARCHIVE")
    staging_resolved = staging.resolve(strict=True)
    archive_resolved = archive_path.resolve(strict=False)
    try:
        archive_resolved.relative_to(staging_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("ARCHIVE_DESTINATION_INSIDE_STAGING")
    files = collect_strict_release_tree(staging_resolved)
    relative_files = {
        path.relative_to(staging_resolved).as_posix() for path in files
    }
    missing = sorted((required_delivery_paths() | {"MANIFEST.json"}) - relative_files)
    if missing:
        raise RuntimeError("STAGED_REQUIRED_DELIVERY_PATHS_MISSING")
    manifest_validation = validate_manifest(staging_resolved)
    if not manifest_validation["passed"]:
        raise RuntimeError("STAGED_MANIFEST_VALIDATION_FAILED")
    try:
        manifest_payload = json.loads(
            (staging_resolved / "MANIFEST.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("STAGED_MANIFEST_INVALID") from error
    if (
        manifest_payload.get("archive_filename") != EXPECTED_ARCHIVE_FILENAME
        or manifest_payload.get("submission_name") != EXPECTED_SUBMISSION_NAME
        or manifest_payload.get("archive_layout") != "flat_project_root"
        or manifest_payload.get("task_id") != EXPECTED_TASK_ID
        or manifest_payload.get("benchmark_version")
        != EXPECTED_BENCHMARK_VERSION
        or manifest_payload.get("case_id") != EXPECTED_CASE_ID
        or manifest_payload.get("protocol") != EXPECTED_PROTOCOL
    ):
        raise RuntimeError("STAGED_MANIFEST_RELEASE_IDENTITY_MISMATCH")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".partial")
    if partial.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_PARTIAL_ARCHIVE")
    try:
        with zipfile.ZipFile(
            partial,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=False,
        ) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(staging_resolved).as_posix())
        with zipfile.ZipFile(partial, "r") as archive:
            bad_crc = archive.testzip()
            if bad_crc is not None:
                raise RuntimeError("ZIP_CRC_FAILURE")
        os.replace(partial, archive_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "archive_path": archive_path,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "zip_file_count": len(files),
    }


def zip_member_is_regular(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    return file_type in (0, stat.S_IFREG)

from __future__ import annotations

import json
import shutil
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from package_tools.archive import zip_member_is_regular
from package_tools.manifest import sha256_file, validate_manifest
from package_tools.policy import PACKAGE_GATE_CHECKS
from package_tools.scanners import (
    scan_forbidden_names,
    scan_runtime_absolute_paths,
    scan_secrets,
)


MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_FILES = 10_000


def preflight_archive(archive_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    entries: list[dict[str, Any]] = []
    if not archive_path.is_file():
        return {"passed": False, "errors": ["ARCHIVE_MISSING"]}
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        errors.append("ARCHIVE_TOO_LARGE")
    seen: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_FILES:
                errors.append("ARCHIVE_FILE_COUNT_INVALID")
            for info in infos:
                name = info.filename
                if info.is_dir():
                    errors.append("DIRECTORY_ENTRIES_NOT_EXPECTED")
                    continue
                if not name or "\\" in name or name.startswith("/"):
                    errors.append("UNSAFE_MEMBER_NAME")
                    continue
                pure = PurePosixPath(name)
                if any(part in {"", ".", ".."} for part in pure.parts):
                    errors.append("UNSAFE_MEMBER_COMPONENT")
                if ":" in pure.parts[0] or any(":" in part for part in pure.parts):
                    errors.append("UNSAFE_MEMBER_COLON")
                if unicodedata.normalize("NFC", name) != name:
                    errors.append("NON_NFC_MEMBER")
                collision = name.casefold()
                if collision in seen:
                    errors.append("DUPLICATE_OR_COLLIDING_MEMBER")
                seen.add(collision)
                if not zip_member_is_regular(info):
                    errors.append("NON_REGULAR_MEMBER")
                if info.flag_bits & 0x1:
                    errors.append("ENCRYPTED_MEMBER")
                if info.file_size > MAX_FILE_BYTES:
                    errors.append("MEMBER_TOO_LARGE")
                total += info.file_size
                entries.append(
                    {
                        "name": name,
                        "size_bytes": info.file_size,
                        "crc": info.CRC,
                    }
                )
            if total > MAX_TOTAL_BYTES:
                errors.append("UNCOMPRESSED_ARCHIVE_TOO_LARGE")
            bad_crc = archive.testzip()
            if bad_crc is not None:
                errors.append("ZIP_CRC_FAILURE")
    except (OSError, zipfile.BadZipFile, RuntimeError):
        errors.append("ARCHIVE_UNREADABLE")
    required_root = {"README.md", "TASK.md", "run_all.ps1", "MANIFEST.json"}
    names = {entry["name"] for entry in entries}
    if not required_root.issubset(names):
        errors.append("ROOT_ENTRYPOINT_MISSING")
    if any(name.startswith("LiuChengchun_xTB_Failure_Diagnosis_Restart_Agent_Benchmark/") for name in names):
        errors.append("UNWANTED_WRAPPER_DIRECTORY")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "archive_size_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "archive_sha256": sha256_file(archive_path) if archive_path.is_file() else None,
        "entry_count": len(entries),
        "total_uncompressed_bytes": total,
        "entries": entries,
    }


def extract_safely(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError("REFUSING_TO_OVERWRITE_EXTRACTION")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            target = (root / PurePosixPath(info.filename)).resolve()
            target.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return root


def validate_release_commit_claim(root: Path) -> dict[str, Any]:
    """Check the final-state claim without trusting it as Gate 8 evidence."""

    errors: list[str] = []
    release_path = root / "configs" / "release.json"
    manifest_path = root / "MANIFEST.json"
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        release = {}
        errors.append("RELEASE_REGISTRY_INVALID")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        manifest_payload = {}
        errors.append("MANIFEST_INVALID")

    release_status = release.get("status") if isinstance(release, dict) else None
    manifest_status = (
        manifest_payload.get("release_status")
        if isinstance(manifest_payload, dict)
        else None
    )
    package_gate = (
        release.get("gates", {}).get("package", {})
        if isinstance(release, dict)
        else {}
    )
    checks = package_gate.get("checks", {}) if isinstance(package_gate, dict) else {}
    if release_status != "PASS":
        errors.append("RELEASE_STATUS_NOT_PASS")
    if manifest_status != "PASS":
        errors.append("MANIFEST_RELEASE_STATUS_NOT_PASS")
    if release_status != manifest_status:
        errors.append("RELEASE_MANIFEST_STATUS_MISMATCH")
    if not isinstance(package_gate, dict) or package_gate.get("status") != "PASS":
        errors.append("PACKAGE_GATE_STATUS_NOT_PASS")
    if not isinstance(package_gate, dict) or package_gate.get("passed") is not True:
        errors.append("PACKAGE_GATE_NOT_PASSED")
    missing_checks = [
        name
        for name in PACKAGE_GATE_CHECKS
        if not isinstance(checks, dict) or checks.get(name) is not True
    ]
    if missing_checks:
        errors.append("PACKAGE_GATE_CHECKS_INCOMPLETE")
    if isinstance(release, dict) and isinstance(manifest_payload, dict):
        if manifest_payload.get("models") != release.get("models"):
            errors.append("RELEASE_MANIFEST_MODELS_MISMATCH")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "release_status": release_status,
        "manifest_release_status": manifest_status,
        "package_gate_status": (
            package_gate.get("status") if isinstance(package_gate, dict) else None
        ),
        "package_gate_checks_complete": not missing_checks,
    }


def validate_extracted_static(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest = validate_manifest(root)
    release_commit = validate_release_commit_claim(root)
    secrets = scan_secrets(root, files)
    runtime_paths, expected_test_literals = scan_runtime_absolute_paths(root, files)
    forbidden = scan_forbidden_names(root, files)
    errors = []
    if not manifest["passed"]:
        errors.extend(manifest["errors"])
    if not release_commit["passed"]:
        errors.extend(release_commit["errors"])
    if secrets:
        errors.append("SECRET_FINDINGS")
    if runtime_paths:
        errors.append("RUNTIME_ABSOLUTE_PATH_FINDINGS")
    if forbidden:
        errors.append("FORBIDDEN_FILES")
    if (root / ".env").exists():
        errors.append("REAL_ENV_PRESENT")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "file_count": len(files),
        "manifest": manifest,
        "release_commit": release_commit,
        "secret_findings": secrets,
        "runtime_absolute_path_findings": runtime_paths,
        "expected_security_test_path_literals": expected_test_literals,
        "forbidden_findings": forbidden,
        "real_env_present": (root / ".env").exists(),
    }

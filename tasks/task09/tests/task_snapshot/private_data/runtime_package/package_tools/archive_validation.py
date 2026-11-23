from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from package_tools.archive import zip_member_is_regular
from package_tools.historical_integrity import (
    HistoricalIntegrityError,
    validate_historical_integrity_receipt,
)
from package_tools.manifest import SHA256_RE, sha256_file, validate_manifest
from package_tools.policy import (
    EXPECTED_ARCHIVE_FILENAME,
    EXPECTED_BENCHMARK_VERSION,
    EXPECTED_CASE_ID,
    EXPECTED_PROTOCOL,
    EXPECTED_SUBMISSION_NAME,
    EXPECTED_TASK_ID,
    PACKAGE_GATE_CHECKS,
    allowed_release_file,
    collect_strict_release_tree,
    forbidden_release_path_reason,
    required_delivery_paths,
)
from package_tools.scanners import (
    scan_forbidden_names,
    scan_runtime_absolute_paths,
    scan_secrets,
)


MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_FILES = 10_000
ALLOWED_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".xyz",
    ".inp", ".ps1", ".sh", ".gitignore", ".example", ".mol", ".ini",
    ".log",
}
ALLOWED_EXTENSIONLESS = {
    "LICENSE", "NOTICE", ".gitignore",
}


def _zip_member_sha256(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_archive_manifest(
    archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
) -> list[str]:
    errors: list[str] = []
    manifest_infos = [
        info for info in infos if not info.is_dir() and info.filename == "MANIFEST.json"
    ]
    if len(manifest_infos) != 1:
        return ["ARCHIVE_MANIFEST_MISSING_OR_DUPLICATE"]
    try:
        manifest = json.loads(archive.read(manifest_infos[0]).decode("utf-8"))
    except (KeyError, UnicodeError, ValueError, TypeError, RuntimeError):
        return ["ARCHIVE_MANIFEST_INVALID"]
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return ["ARCHIVE_MANIFEST_INVALID"]
    if manifest.get("archive_filename") != EXPECTED_ARCHIVE_FILENAME:
        errors.append("ARCHIVE_MANIFEST_FILENAME_MISMATCH")
    if manifest.get("submission_name") != EXPECTED_SUBMISSION_NAME:
        errors.append("ARCHIVE_MANIFEST_SUBMISSION_MISMATCH")
    if manifest.get("archive_layout") != "flat_project_root":
        errors.append("ARCHIVE_MANIFEST_LAYOUT_MISMATCH")
    if (
        manifest.get("task_id") != EXPECTED_TASK_ID
        or manifest.get("benchmark_version") != EXPECTED_BENCHMARK_VERSION
        or manifest.get("case_id") != EXPECTED_CASE_ID
        or manifest.get("protocol") != EXPECTED_PROTOCOL
    ):
        errors.append("ARCHIVE_MANIFEST_TASK_IDENTITY_MISMATCH")

    payload_infos = [
        info for info in infos if not info.is_dir() and info.filename != "MANIFEST.json"
    ]
    actual_by_name = {info.filename: info for info in payload_infos}
    listed: dict[str, dict[str, Any]] = {}
    listed_casefold: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append("ARCHIVE_MANIFEST_ENTRY_INVALID")
            continue
        path = item["path"]
        collision = path.casefold()
        if path in listed or collision in listed_casefold:
            errors.append("ARCHIVE_MANIFEST_DUPLICATE_PATH")
            continue
        listed[path] = item
        listed_casefold.add(collision)
        info = actual_by_name.get(path)
        if info is None:
            errors.append("ARCHIVE_MANIFEST_FILE_MISSING")
            continue
        size = item.get("size_bytes")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size != info.file_size
        ):
            errors.append("ARCHIVE_MANIFEST_SIZE_MISMATCH")
        expected_hash = item.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or SHA256_RE.fullmatch(expected_hash) is None
            or _zip_member_sha256(archive, info) != expected_hash.upper()
        ):
            errors.append("ARCHIVE_MANIFEST_HASH_MISMATCH")
    if set(listed) != set(actual_by_name):
        errors.append("ARCHIVE_MANIFEST_INVENTORY_MISMATCH")
    count = manifest.get("file_count_excluding_manifest")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(listed)
        or count != len(payload_infos)
    ):
        errors.append("ARCHIVE_MANIFEST_COUNT_MISMATCH")
    total_bytes = manifest.get("total_bytes_excluding_manifest")
    actual_total = sum(info.file_size for info in payload_infos)
    if (
        not isinstance(total_bytes, int)
        or isinstance(total_bytes, bool)
        or total_bytes != actual_total
    ):
        errors.append("ARCHIVE_MANIFEST_TOTAL_BYTES_MISMATCH")
    return errors


def preflight_archive(archive_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    entries: list[dict[str, Any]] = []
    missing_required_members: list[str] = []
    if not archive_path.is_file():
        return {"passed": False, "errors": ["ARCHIVE_MISSING"]}
    if archive_path.name != EXPECTED_ARCHIVE_FILENAME:
        errors.append("ARCHIVE_FILENAME_MISMATCH")
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
                member = PurePosixPath(name)
                forbidden_reason = forbidden_release_path_reason(member)
                if forbidden_reason is not None:
                    errors.append("FORBIDDEN_RELEASE_MEMBER")
                if not allowed_release_file(member, include_manifest=True):
                    errors.append("UNEXPECTED_RELEASE_MEMBER")
                if member.suffix.casefold() not in ALLOWED_SUFFIXES and member.name not in ALLOWED_EXTENSIONLESS:
                    errors.append("UNEXPECTED_EXTENSION")
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
            errors.extend(_validate_archive_manifest(archive, infos))
    except (OSError, zipfile.BadZipFile, RuntimeError):
        errors.append("ARCHIVE_UNREADABLE")
    required_root = {"README.md", "TASK.md", "run_all.ps1", "MANIFEST.json"}
    names = {entry["name"] for entry in entries}
    if not required_root.issubset(names):
        errors.append("ROOT_ENTRYPOINT_MISSING")
    wrapper_names = {
        "adaptive_xtb_torsion_profile_agent_v1",
        "adaptive_xtb_torsion_profile_agent_v1_1",
        "adaptive_xtb_torsion_profile_agent_v2",
        "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark",
        "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v1_1",
        "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v2",
    }
    if any(
        PurePosixPath(name).parts
        and PurePosixPath(name).parts[0].casefold() == wrapper.casefold()
        for name in names
        for wrapper in wrapper_names
    ):
        errors.append("UNWANTED_WRAPPER_DIRECTORY")
    missing_required_members = sorted(required_delivery_paths() - names)
    if missing_required_members:
        errors.append("REQUIRED_DELIVERY_MEMBER_MISSING")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "archive_size_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "archive_sha256": sha256_file(archive_path) if archive_path.is_file() else None,
        "entry_count": len(entries),
        "total_uncompressed_bytes": total,
        "missing_required_members": missing_required_members,
        "entries": entries,
    }


def extract_safely(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError("REFUSING_TO_OVERWRITE_EXTRACTION")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    archive_preflight = preflight_archive(archive_path)
    if not archive_preflight["passed"]:
        raise RuntimeError("ARCHIVE_PREFLIGHT_FAILED")
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
    historical_receipt_path = (
        root / "reports" / "HISTORICAL_RELEASE_INTEGRITY_POST.json"
    )
    try:
        historical_receipt = json.loads(
            historical_receipt_path.read_text(encoding="utf-8")
        )
        historical_validation = validate_historical_integrity_receipt(
            historical_receipt
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        HistoricalIntegrityError,
    ):
        historical_validation = {"passed": False}
        errors.append("HISTORICAL_INTEGRITY_RECEIPT_INVALID")

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
    allowed_release_statuses = {"PASS", "FAIL", "BLOCKED"}
    if release_status not in allowed_release_statuses:
        errors.append("RELEASE_STATUS_INVALID")
    if manifest_status not in allowed_release_statuses:
        errors.append("MANIFEST_RELEASE_STATUS_INVALID")
    if release_status != manifest_status:
        errors.append("RELEASE_MANIFEST_STATUS_MISMATCH")
    if not isinstance(release, dict) or (
        release.get("schema_version") != "2.0"
        or release.get("task_id") != EXPECTED_TASK_ID
        or release.get("benchmark_version") != EXPECTED_BENCHMARK_VERSION
        or release.get("protocol") != EXPECTED_PROTOCOL
    ):
        errors.append("RELEASE_TASK_IDENTITY_MISMATCH")
    expected_release_ready = release_status == "PASS"
    if (
        not isinstance(release, dict)
        or release.get("release_ready") is not expected_release_ready
    ):
        errors.append("RELEASE_READY_STATUS_MISMATCH")
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
    preservation_valid = historical_validation.get("passed") is True
    if not preservation_valid:
        errors.append("OLD_EVIDENCE_PRESERVATION_NOT_PROVEN")
    if isinstance(release, dict) and isinstance(manifest_payload, dict):
        if manifest_payload.get("models") != release.get("models"):
            errors.append("RELEASE_MANIFEST_MODELS_MISMATCH")
        if manifest_payload.get("archive_filename") != EXPECTED_ARCHIVE_FILENAME:
            errors.append("MANIFEST_ARCHIVE_FILENAME_MISMATCH")
        if manifest_payload.get("submission_name") != EXPECTED_SUBMISSION_NAME:
            errors.append("MANIFEST_SUBMISSION_NAME_MISMATCH")
        if manifest_payload.get("archive_layout") != "flat_project_root":
            errors.append("MANIFEST_ARCHIVE_LAYOUT_MISMATCH")
        if (
            manifest_payload.get("task_id") != EXPECTED_TASK_ID
            or manifest_payload.get("benchmark_version")
            != EXPECTED_BENCHMARK_VERSION
            or manifest_payload.get("case_id") != EXPECTED_CASE_ID
            or manifest_payload.get("protocol") != EXPECTED_PROTOCOL
        ):
            errors.append("MANIFEST_TASK_IDENTITY_MISMATCH")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "release_status": release_status,
        "manifest_release_status": manifest_status,
        "package_gate_status": (
            package_gate.get("status") if isinstance(package_gate, dict) else None
        ),
        "package_gate_checks_complete": not missing_checks,
        "old_evidence_preservation_valid": preservation_valid,
        "release_ready": release_status == "PASS",
    }


def validate_extracted_static(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    strict_tree_error = None
    try:
        strict_files = collect_strict_release_tree(root)
    except (OSError, RuntimeError, ValueError) as error:
        strict_files = []
        strict_tree_error = str(error).split(":", 1)[0]
    relative_files = {path.relative_to(root).as_posix() for path in strict_files}
    missing_required = sorted(
        (required_delivery_paths() | {"MANIFEST.json"}) - relative_files
    )
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
    if strict_tree_error is not None:
        errors.append("STRICT_RELEASE_TREE_INVALID")
    if missing_required:
        errors.append("REQUIRED_DELIVERY_PATHS_MISSING")
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
        "strict_release_tree_error": strict_tree_error,
        "missing_required": missing_required,
        "real_env_present": (root / ".env").exists(),
    }

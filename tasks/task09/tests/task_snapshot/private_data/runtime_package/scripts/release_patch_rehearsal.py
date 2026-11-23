"""Run an immutable second pre-freeze rehearsal after release-layer repairs.

This is intentionally not a final-package builder.  It never creates root
commitments, calls a provider or xTB, or writes outside the v2 project root.
The rehearsal ZIP remains under ``audit_tmp`` and omits outputs that do not
yet exist instead of synthesizing them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any
import unicodedata
import zipfile

from package_tools.archive import zip_member_is_regular
from package_tools.archive_validation import preflight_archive
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
from tools.science import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
FORMAL_LABELS = ("kimi", "deepseek", "glm")


def _safe_member(name: str) -> bool:
    pure = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and not pure.is_absolute()
        and pure.as_posix() == name
        and unicodedata.normalize("NFC", name) == name
        and all(
            part not in {"", ".", ".."} and ":" not in part
            for part in pure.parts
        )
    )


def _formal_files(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label in FORMAL_LABELS:
        run = root / "trajectories" / label
        for path in sorted(
            (item for item in run.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        ):
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def _run(
    name: str,
    python_args: list[str],
    cwd: Path,
    log_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    interpreter_root = Path(sys.executable).resolve().parent
    conda_dll_dir = interpreter_root / "Library" / "bin"
    if conda_dll_dir.is_dir():
        environment["PATH"] = (
            str(conda_dll_dir)
            + os.pathsep
            + environment.get("PATH", "")
        )
    completed = subprocess.run(
        [sys.executable, "-s", *python_args],
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        timeout=600,
        shell=False,
    )
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    return {
        "argv": ["<VALIDATED_PYTHON>", "-s", *python_args],
        "shell": False,
        "return_code": completed.returncode,
        "stdout": {
            "path": stdout_path.relative_to(work_root).as_posix(),
            "size_bytes": stdout_path.stat().st_size,
            "sha256": sha256_file(stdout_path),
        },
        "stderr": {
            "path": stderr_path.relative_to(work_root).as_posix(),
            "size_bytes": stderr_path.stat().st_size,
            "sha256": sha256_file(stderr_path),
        },
    }


def _extract_rehearsal(
    archive_path: Path,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("REHEARSAL_EXTRACTION_ALREADY_EXISTS")
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve(strict=True)
    errors: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        for info in infos:
            name = info.filename
            if info.is_dir():
                errors.append("DIRECTORY_ENTRY")
                continue
            if not _safe_member(name):
                errors.append("UNSAFE_MEMBER")
                continue
            folded = name.casefold()
            if folded in seen:
                errors.append("DUPLICATE_MEMBER")
                continue
            seen.add(folded)
            if not zip_member_is_regular(info):
                errors.append("NON_REGULAR_MEMBER")
                continue
            if info.flag_bits & 0x1:
                errors.append("ENCRYPTED_MEMBER")
                continue
            target = root.joinpath(*PurePosixPath(name).parts).resolve(
                strict=False
            )
            try:
                target.relative_to(root)
            except ValueError:
                errors.append("EXTRACTION_ESCAPE")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            total_bytes += target.stat().st_size
        crc_failure = archive.testzip()
        if crc_failure is not None:
            errors.append("ZIP_CRC_FAILURE")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "member_count": len(seen),
        "total_uncompressed_bytes": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        default="03",
        help="Two-digit immutable rehearsal sequence (default: 03).",
    )
    args = parser.parse_args()
    if re.fullmatch(r"[0-9]{2}", args.sequence) is None:
        raise ValueError("REHEARSAL_SEQUENCE_MUST_BE_TWO_DIGITS")
    work = (
        ROOT
        / "audit_tmp"
        / f"pre_freeze_package_rehearsal_release_patch_{args.sequence}"
    )
    report_path = (
        ROOT
        / "reports"
        / f"pre_freeze_package_rehearsal_release_patch_{args.sequence}.json"
    )
    if work.exists() or work.is_symlink():
        raise FileExistsError("RELEASE_PATCH_REHEARSAL_WORK_ALREADY_EXISTS")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError("RELEASE_PATCH_REHEARSAL_REPORT_ALREADY_EXISTS")
    if (ROOT / "execution_commitment.json").exists() or (
        ROOT / "release_commitment.json"
    ).exists():
        raise RuntimeError("PREFREEZE_REHEARSAL_REFUSES_ROOT_COMMITMENTS")
    final_archive = ROOT.parent / EXPECTED_ARCHIVE_FILENAME
    if final_archive.exists() or final_archive.is_symlink():
        raise RuntimeError("PREFREEZE_REHEARSAL_REFUSES_FINAL_ARCHIVE")

    formal_before = _formal_files(ROOT)
    source_files = collect_allowlisted_files(ROOT)
    source_secret = scan_secrets(ROOT, source_files)
    source_absolute, expected_test_literals = scan_runtime_absolute_paths(
        ROOT, source_files
    )
    source_forbidden = scan_forbidden_names(ROOT, source_files)

    work.mkdir(parents=True, exist_ok=False)
    stage = work / "staged_flat_root"
    extracted = work / "clean_extract"
    logs = work / "command_logs"
    archive_path = work / EXPECTED_ARCHIVE_FILENAME
    stage.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    for source in source_files:
        target = stage / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    staged_payload = collect_strict_release_tree(stage)
    staged_relative = {
        path.relative_to(stage).as_posix() for path in staged_payload
    }
    missing_final = sorted(required_delivery_paths() - staged_relative)
    release_registry = json.loads(
        (stage / "configs" / "release.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(release_registry, dict)
        or release_registry.get("schema_version") != "2.0"
        or release_registry.get("task_id") != EXPECTED_TASK_ID
        or release_registry.get("benchmark_version")
        != EXPECTED_BENCHMARK_VERSION
        or release_registry.get("protocol") != EXPECTED_PROTOCOL
        or release_registry.get("status") not in {"PASS", "FAIL", "BLOCKED"}
        or not isinstance(release_registry.get("models"), dict)
    ):
        raise RuntimeError("REHEARSAL_RELEASE_REGISTRY_INVALID")
    manifest_payload = {
        "schema_version": "2.0-release-patch-rehearsal",
        "submission_name": EXPECTED_SUBMISSION_NAME,
        "archive_filename": EXPECTED_ARCHIVE_FILENAME,
        "archive_layout": "flat_project_root",
        "task_id": EXPECTED_TASK_ID,
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
        "case_id": EXPECTED_CASE_ID,
        "protocol": EXPECTED_PROTOCOL,
        "models": release_registry["models"],
        "release_status": release_registry["status"],
        "file_count_excluding_manifest": len(staged_payload),
        "total_bytes_excluding_manifest": sum(
            path.stat().st_size for path in staged_payload
        ),
        "files": file_entries(stage, staged_payload),
    }
    write_manifest(stage / "MANIFEST.json", manifest_payload)
    staged_manifest = validate_manifest(stage)
    staged_files = collect_strict_release_tree(stage)
    with zipfile.ZipFile(
        archive_path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=False,
    ) as archive:
        for path in staged_files:
            archive.write(path, path.relative_to(stage).as_posix())

    production_preflight = preflight_archive(archive_path)
    expected_preflight_only_missing = (
        production_preflight.get("errors")
        == ["REQUIRED_DELIVERY_MEMBER_MISSING"]
        and production_preflight.get("missing_required_members")
        == missing_final
    )
    extraction = _extract_rehearsal(archive_path, extracted)
    extracted_files = collect_strict_release_tree(extracted)
    extracted_manifest = validate_manifest(extracted)
    extracted_secret = scan_secrets(extracted, extracted_files)
    extracted_absolute, extracted_test_literals = scan_runtime_absolute_paths(
        extracted, extracted_files
    )
    extracted_forbidden = scan_forbidden_names(extracted, extracted_files)

    commands = {
        "source_validation": _run(
            "source_validation",
            ["-m", "scripts.validate_project"],
            extracted,
            logs,
            work,
        ),
        "pytest": _run(
            "pytest",
            ["-m", "pytest", "-q"],
            extracted,
            logs,
            work,
        ),
        "baseline_replay": _run(
            "baseline_replay",
            [
                "-m",
                "scripts.replay_trajectories",
                "baseline",
                "--output",
                ".test_tmp/release_patch_replay.json",
            ],
            extracted,
            logs,
            work,
        ),
        "secret_scan": _run(
            "secret_scan",
            ["-m", "scripts.secret_scan"],
            extracted,
            logs,
            work,
        ),
    }
    formal_after = _formal_files(ROOT)
    no_final_outputs_synthesized = all(
        path.startswith(
            (
                "trajectories/kimi/",
                "trajectories/deepseek/",
                "trajectories/glm/",
            )
        )
        or path.startswith("reports/")
        or path
        in {
            "execution_commitment.json",
            "release_commitment.json",
        }
        for path in missing_final
    )
    checks = {
        "source_allowlist_collection": True,
        "source_secret_scan": not source_secret,
        "source_runtime_absolute_path_scan": not source_absolute,
        "source_forbidden_name_scan": not source_forbidden,
        "staged_manifest_exact_coverage": staged_manifest["passed"],
        "production_zip_preflight_only_expected_missing_outputs": (
            expected_preflight_only_missing
        ),
        "safe_clean_extraction": extraction["passed"],
        "clean_extract_manifest": extracted_manifest["passed"],
        "clean_extract_secret_scan": not extracted_secret,
        "clean_extract_runtime_absolute_path_scan": not extracted_absolute,
        "clean_extract_forbidden_name_scan": not extracted_forbidden,
        "clean_extract_source_validation": (
            commands["source_validation"]["return_code"] == 0
        ),
        "clean_extract_full_pytest": commands["pytest"]["return_code"] == 0,
        "clean_extract_baseline_replay": (
            commands["baseline_replay"]["return_code"] == 0
        ),
        "clean_extract_secret_scan_command": (
            commands["secret_scan"]["return_code"] == 0
        ),
        "formal_results_unchanged": formal_before == formal_after,
        "formal_results_not_synthesized": no_final_outputs_synthesized,
        "root_commitments_absent": not (
            (ROOT / "execution_commitment.json").exists()
            or (ROOT / "release_commitment.json").exists()
        ),
        "parent_final_zip_absent": not final_archive.exists(),
        "provider_calls_absent": True,
        "native_xtb_scan_calls_absent": True,
        "xtb_version_probe_allowed": True,
    }
    report = {
        "schema_version": "2.0",
        "scope": (
            f"pre_freeze_release_patch_rehearsal_{args.sequence}"
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source": {
            "file_count": len(source_files),
            "total_bytes": sum(path.stat().st_size for path in source_files),
            "secret_findings": source_secret,
            "absolute_path_findings": source_absolute,
            "forbidden_findings": source_forbidden,
            "expected_security_test_path_literals": expected_test_literals,
        },
        "stage": {
            "file_count_excluding_manifest": len(staged_payload),
            "manifest_sha256": sha256_file(stage / "MANIFEST.json"),
            "missing_final_outputs_not_synthesized": missing_final,
        },
        "archive": {
            "location": "<PROJECT_AUDIT_TMP>",
            "filename": EXPECTED_ARCHIVE_FILENAME,
            "size_bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "entry_count": len(staged_files),
            "production_preflight_errors": production_preflight.get("errors"),
        },
        "clean_extract": {
            **extraction,
            "file_count": len(extracted_files),
            "manifest": extracted_manifest,
            "secret_findings": extracted_secret,
            "absolute_path_findings": extracted_absolute,
            "forbidden_findings": extracted_forbidden,
            "expected_security_test_path_literals": extracted_test_literals,
        },
        "commands": commands,
        "formal_file_count_before": len(formal_before),
        "formal_file_count_after": len(formal_after),
        "provider_call_count": 0,
        "native_xtb_scan_call_count": 0,
        "xtb_version_probe_note": (
            "source validation and scientific tests may execute only "
            "the allowlisted `xtb --version` identity probe"
        ),
    }
    with report_path.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

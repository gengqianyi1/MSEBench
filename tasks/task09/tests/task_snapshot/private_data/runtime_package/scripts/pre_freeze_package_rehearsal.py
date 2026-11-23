"""Exercise release mechanics before provider execution without fake results."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from package_tools.manifest import (
    file_entries,
    sha256_file,
    validate_manifest,
    write_manifest,
)
from package_tools.policy import (
    EXPECTED_ARCHIVE_FILENAME,
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


def _safe_member(name: str) -> bool:
    pure = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and not pure.is_absolute()
        and pure.as_posix() == name
        and all(part not in {"", ".", ".."} and ":" not in part for part in pure.parts)
    )


def main() -> int:
    work = ROOT / "audit_tmp" / "pre_freeze_package_rehearsal"
    stage = work / "staged_flat_root"
    extracted = work / "clean_extract"
    archive_path = work / EXPECTED_ARCHIVE_FILENAME
    report_path = ROOT / "reports" / "pre_freeze_package_rehearsal.json"
    if work.exists() or report_path.exists():
        raise RuntimeError("pre-freeze rehearsal is immutable and already exists")
    files = collect_allowlisted_files(ROOT)
    secret_findings = scan_secrets(ROOT, files)
    absolute_findings, expected_test_literals = scan_runtime_absolute_paths(
        ROOT, files
    )
    forbidden = scan_forbidden_names(ROOT, files)
    work.mkdir(parents=True, exist_ok=False)
    stage.mkdir(parents=True, exist_ok=False)
    for source in files:
        target = stage / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    staged_payload = [
        path
        for path in collect_strict_release_tree(stage)
        if path.name != "MANIFEST.json"
    ]
    manifest = {
        "schema_version": "2.0-rehearsal",
        "submission_name": (
            "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v2"
        ),
        "archive_filename": EXPECTED_ARCHIVE_FILENAME,
        "archive_layout": "flat_project_root",
        "task_id": "adaptive_xtb_torsion_profile_agent_v2",
        "case_id": "2_fluorobiphenyl_central_torsion_v1",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "release_status": "PREFREEZE_REHEARSAL_ONLY",
        "file_count_excluding_manifest": len(staged_payload),
        "total_bytes_excluding_manifest": sum(
            path.stat().st_size for path in staged_payload
        ),
        "files": file_entries(stage, staged_payload),
    }
    write_manifest(stage / "MANIFEST.json", manifest)
    manifest_validation = validate_manifest(stage)
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
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        archive_checks = {
            "crc": archive.testzip() is None,
            "all_members_safe": all(_safe_member(name) for name in names),
            "no_duplicate_members": len(names)
            == len({name.casefold() for name in names}),
            "flat_project_root": "README.md" in names
            and not any(name.startswith("adaptive_xtb_torsion_profile_agent") for name in names),
        }
        extracted.mkdir(parents=True, exist_ok=False)
        for info in infos:
            destination = extracted.joinpath(*PurePosixPath(info.filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
    extracted_files = collect_strict_release_tree(extracted)
    extracted_secret = scan_secrets(extracted, extracted_files)
    extracted_absolute, _ = scan_runtime_absolute_paths(
        extracted, extracted_files
    )
    extracted_forbidden = scan_forbidden_names(extracted, extracted_files)
    extracted_manifest = validate_manifest(extracted)
    present = {
        path.relative_to(ROOT).as_posix() for path in files
    }
    missing_final_only = sorted(required_delivery_paths() - present)
    checks = {
        "source_allowlist_collection": True,
        "source_secret_scan": not secret_findings,
        "source_runtime_absolute_path_scan": not absolute_findings,
        "source_forbidden_name_scan": not forbidden,
        "staged_manifest_exact_coverage": manifest_validation["passed"],
        "zip_crc": archive_checks["crc"],
        "zip_members_safe": archive_checks["all_members_safe"],
        "zip_duplicate_members_absent": archive_checks["no_duplicate_members"],
        "zip_flat_root": archive_checks["flat_project_root"],
        "clean_extract_manifest": extracted_manifest["passed"],
        "clean_extract_secret_scan": not extracted_secret,
        "clean_extract_absolute_path_scan": not extracted_absolute,
        "clean_extract_forbidden_name_scan": not extracted_forbidden,
        "formal_results_not_fabricated": all(
            path.startswith(("trajectories/kimi/", "trajectories/deepseek/", "trajectories/glm/"))
            or path.startswith("reports/")
            or path in {
                "configs/release.json",
                "execution_commitment.json",
                "release_commitment.json",
            }
            for path in missing_final_only
        ),
    }
    report = {
        "schema_version": "2.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "release_mechanics_before_execution_freeze",
        "checks": checks,
        "source_file_count": len(files),
        "staged_file_count": len(staged_files),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "expected_security_test_path_literals": expected_test_literals,
        "missing_final_outputs_not_synthesized": missing_final_only,
        "secret_findings": secret_findings,
        "absolute_path_findings": absolute_findings,
        "forbidden_findings": forbidden,
    }
    with report_path.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

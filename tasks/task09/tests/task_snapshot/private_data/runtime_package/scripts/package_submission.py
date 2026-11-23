"""Build and validate the final protocol-2.0 flat-root release archive.

The candidate archive remains below the project audit directory until every
clean-delivery check passes.  Only then is it atomically moved to the frozen
parent-directory filename.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from package_tools.archive import build_flat_archive, stage_project
from package_tools.archive_validation import (
    extract_safely,
    preflight_archive,
    validate_extracted_static,
)
from package_tools.commitments import (
    CommitmentError,
    validate_execution_commitment,
    validate_release_commitment,
)
from package_tools.policy import EXPECTED_ARCHIVE_FILENAME
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _strict_json(path: Path) -> dict[str, Any]:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=hook,
        parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
    )
    if duplicates or not isinstance(value, dict):
        raise ValueError(f"invalid closed JSON object: {path.name}")
    return value


def _validated_commitments(root: Path) -> dict[str, Any]:
    execution = _strict_json(root / "execution_commitment.json")
    release = _strict_json(root / "release_commitment.json")
    execution_validation = validate_execution_commitment(root, execution)
    release_validation = validate_release_commitment(root, release)
    return {
        "execution": execution_validation,
        "release": release_validation,
    }


def _run(argv: list[str], cwd: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Conda packages and xTB resolve their native DLLs from Library/bin.
    interpreter_root = Path(sys.executable).resolve().parent
    conda_dll_dir = interpreter_root / "Library" / "bin"
    if conda_dll_dir.is_dir():
        environment["PATH"] = (
            str(conda_dll_dir)
            + os.pathsep
            + environment.get("PATH", "")
        )
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        shell=False,
    )
    return {
        "argv": argv,
        "shell": False,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-30:],
        "stderr_tail": completed.stderr.splitlines()[-30:],
    }


def _inside_project(value: Path, *, field: str) -> Path:
    candidate = value if value.is_absolute() else ROOT / value
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{field} must remain inside the project") from exc
    if not relative.parts:
        raise ValueError(f"{field} cannot be the project root")
    return resolved


def _final_archive(value: Path) -> Path:
    candidate = value if value.is_absolute() else ROOT.parent / value
    resolved = candidate.resolve(strict=False)
    if (
        resolved.name != EXPECTED_ARCHIVE_FILENAME
        or resolved.parent != ROOT.parent.resolve(strict=True)
    ):
        raise ValueError(
            "final archive must use the frozen v2 filename in the parent directory"
        )
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError("refusing to overwrite the final archive")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT.parent / EXPECTED_ARCHIVE_FILENAME,
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / "audit_tmp" / "final_package_build_v2",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "audit_tmp" / "final_package_receipt_v2.json",
    )
    args = parser.parse_args()

    final_archive = _final_archive(args.archive)
    work = _inside_project(args.work_root, field="package work root")
    receipt_path = _inside_project(args.receipt, field="package receipt")
    if work.exists() or work.is_symlink():
        raise FileExistsError("package work root already exists")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("package receipt already exists")

    try:
        commitments = _validated_commitments(ROOT)
    except (CommitmentError, OSError, UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError("ROOT_COMMITMENT_VALIDATION_FAILED") from exc

    release = _strict_json(ROOT / "configs" / "release.json")
    if (
        release.get("status") != "PASS"
        or release.get("release_ready") is not True
        or release.get("formal_campaign_execution_status") != "PASS"
    ):
        raise RuntimeError("RELEASE_REGISTRY_NOT_READY")

    work.mkdir(parents=True, exist_ok=False)
    stage = work / "release_tree"
    extraction = work / "clean_extract"
    candidate_dir = work / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    candidate_archive = candidate_dir / EXPECTED_ARCHIVE_FILENAME

    stage_result = stage_project(ROOT, stage)
    archive_result = build_flat_archive(stage, candidate_archive)
    archive_result["archive_path"] = "<VALIDATED_CANDIDATE_ARCHIVE>"
    zip_preflight = preflight_archive(candidate_archive)
    extract_safely(candidate_archive, extraction)
    static = validate_extracted_static(extraction)

    source_validation = _run(
        [sys.executable, "-s", "-m", "scripts.validate_project", "--delivery"],
        extraction,
    )
    unit_tests = _run(
        [sys.executable, "-s", "-m", "pytest", "-q", "tests/unit"],
        extraction,
    )
    integration_tests = _run(
        [sys.executable, "-s", "-m", "pytest", "-q", "tests/integration"],
        extraction,
    )
    security_tests = _run(
        [sys.executable, "-s", "-m", "pytest", "-q", "tests/security"],
        extraction,
    )
    e2e_tests = _run(
        [sys.executable, "-s", "-m", "pytest", "-q", "tests/e2e"],
        extraction,
    )
    replay = _run(
        [
            sys.executable,
            "-s",
            "-m",
            "scripts.replay_trajectories",
            "baseline",
            "kimi",
            "deepseek",
            "glm",
            "--output",
            ".test_tmp/replay_delivery.json",
        ],
        extraction,
    )
    secret_scan = _run(
        [sys.executable, "-s", "-m", "scripts.secret_scan"],
        extraction,
    )
    extracted_commitments: dict[str, Any]
    try:
        extracted_commitments = _validated_commitments(extraction)
        extracted_commitments_ok = True
    except (CommitmentError, OSError, UnicodeError, ValueError, TypeError) as exc:
        extracted_commitments = {"error": type(exc).__name__}
        extracted_commitments_ok = False

    checks = {
        "root_commitments": (
            commitments["execution"].get("passed") is True
            and commitments["release"].get("passed") is True
        ),
        "stage_preflight": stage_result.get("passed") is True,
        "flat_zip_preflight": zip_preflight.get("passed") is True,
        "clean_extract_static": static.get("passed") is True,
        "extracted_commitments": extracted_commitments_ok,
        "source_validation": source_validation["return_code"] == 0,
        "unit_tests": unit_tests["return_code"] == 0,
        "integration_tests": integration_tests["return_code"] == 0,
        "security_tests": security_tests["return_code"] == 0,
        "e2e_tests": e2e_tests["return_code"] == 0,
        "offline_replay": replay["return_code"] == 0,
        "secret_scan": secret_scan["return_code"] == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status == "PASS":
        os.replace(candidate_archive, final_archive)
        final_archive_record = {
            "filename": final_archive.name,
            "size_bytes": final_archive.stat().st_size,
            "sha256": sha256_file(final_archive),
            "flat_project_root": True,
        }
    else:
        final_archive_record = {
            "filename": final_archive.name,
            "created": False,
            "candidate_retained_inside_audit_work_root": (
                candidate_archive.is_file()
            ),
        }

    receipt = {
        "schema_version": "2.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "mode": "FINAL",
        "status": status,
        "checks": checks,
        "root_commitments": commitments,
        "stage": stage_result,
        "archive": archive_result,
        "zip_preflight": zip_preflight,
        "clean_extract_static": static,
        "extracted_commitments": extracted_commitments,
        "source_validation": source_validation,
        "unit_tests": unit_tests,
        "integration_tests": integration_tests,
        "security_tests": security_tests,
        "e2e_tests": e2e_tests,
        "offline_replay": replay,
        "secret_scan": secret_scan,
        "final_archive": final_archive_record,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as handle:
        handle.write(canonical_json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

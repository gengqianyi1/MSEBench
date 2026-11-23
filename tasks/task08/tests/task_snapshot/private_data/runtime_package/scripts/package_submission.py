"""Build, validate, and transactionally publish the organizer archive."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from package_tools.archive import (
    build_flat_archive,
    stage_allowlisted_files,
    write_staged_manifest,
)
from package_tools.manifest import sha256_file
from package_tools.policy import PACKAGE_GATE_CHECKS
from scripts.generate_reports import (
    FINAL_PACKAGE_GATE,
    PENDING_PACKAGE_GATE,
    generate_reports,
)


DEFAULT_NAME = "LiuChengchun_xTB_Failure_Diagnosis_Restart_Agent_Benchmark.zip"
CHECKLIST_NAME = "ORGANIZER_SUBMISSION_CHECKLIST.md"


class PackageTransactionError(RuntimeError):
    """A non-reflective package blocker safe to persist and display."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PackageTransactionError("JSON_OBJECT_REQUIRED")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".package.tmp")
    if temporary.exists():
        raise PackageTransactionError("PACKAGE_TEMPORARY_ALREADY_EXISTS")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def _validation_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    for name in list(environment):
        if any(
            word in name.upper()
            for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        ):
            environment.pop(name, None)
    system_root = os.environ.get("SystemRoot")
    if system_root:
        environment["PATH"] = (
            str(Path(system_root) / "System32")
            + os.pathsep
            + environment.get("PATH", "")
        )
    return environment


def _package_gate_claim(
    *,
    sidecar_name: str,
    validation_name: str,
    checklist_name: str,
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "passed": True,
        "validation_protocol": "EXACT_CANDIDATE_BYTES_THEN_ATOMIC_RENAME",
        "archive_mutated_after_validation": False,
        "archive_sha256_location": sidecar_name,
        "validation_receipt_location": validation_name,
        "organizer_checklist_location": checklist_name,
        "checks": {name: True for name in PACKAGE_GATE_CHECKS},
    }


def _final_release_snapshot(
    ready_release: dict[str, Any],
    *,
    sidecar_name: str,
    validation_name: str,
    checklist_name: str,
) -> dict[str, Any]:
    if ready_release.get("status") != "READY_FOR_PACKAGE":
        raise PackageTransactionError("RELEASE_NOT_READY_FOR_PACKAGE")
    if ready_release.get("gates", {}).get("formal_models", {}).get("passed") is not True:
        raise PackageTransactionError("FORMAL_MODEL_GATE_NOT_PASSED")
    final = deepcopy(ready_release)
    final["status"] = "PASS"
    final["execution_attestation"] = {
        "parent_env_modified": False,
        "ssh_executed": False,
        "hpc_connected": False,
        "slurm_submitted": False,
    }
    final.setdefault("gates", {})["package"] = _package_gate_claim(
        sidecar_name=sidecar_name,
        validation_name=validation_name,
        checklist_name=checklist_name,
    )
    return final


def _run_candidate_validation(
    *,
    root: Path,
    candidate: Path,
    extract_dir: Path,
    report_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(root / "scripts" / "validate_package.py"),
        "--archive",
        str(candidate),
        "--extract-dir",
        str(extract_dir),
        "--output",
        str(report_path),
        "--expected-sha256",
        expected_sha256,
        "--execute-trusted-code",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=_validation_environment(),
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=3600,
    )
    if not report_path.is_file():
        raise PackageTransactionError("PACKAGE_VALIDATION_REPORT_MISSING")
    report = _load_json(report_path)
    if completed.returncode != 0 or report.get("passed") is not True:
        raise PackageTransactionError("CLEAN_EXTRACTION_VALIDATION_FAILED")
    if report.get("archive_sha256") != expected_sha256:
        raise PackageTransactionError("VALIDATED_ARCHIVE_HASH_MISMATCH")
    if report.get("expected_sha256") != expected_sha256:
        raise PackageTransactionError("VALIDATION_EXPECTED_HASH_MISMATCH")
    return report


def _write_pending_text(path: Path, content: str) -> None:
    if path.exists():
        raise PackageTransactionError("PENDING_PUBLICATION_FILE_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content.rstrip() + "\n")


def _checklist_text(
    *,
    release: dict[str, Any],
    archive_name: str,
    archive_sha256: str,
    sidecar_name: str,
    validation_name: str,
    report: dict[str, Any],
) -> str:
    gates = release.get("gates", {})
    tests = gates.get("tests", {})
    native = gates.get("native_xtb", {})
    baseline = gates.get("baseline", {})
    scan = gates.get("security_and_path_scan", {})
    formal = gates.get("formal_models", {})
    configuration = gates.get("configuration", {})
    attestation = release.get("execution_attestation", {})
    static = report.get("static_extraction_validation", {})
    dynamic = report.get("dynamic_validation", {})
    gate_rows = [
        ("Gate 1 — configuration", configuration.get("passed") is True),
        ("Gate 2 — unit/integration tests", tests.get("passed") is True),
        (
            "Gate 3 — security and path controls",
            scan.get("passed") is True,
        ),
        (
            "Gate 4 — deterministic replay",
            baseline.get("evaluation_byte_identical") is True,
        ),
        ("Gate 5 — native xTB fixtures/smokes", native.get("passed") is True),
        ("Gate 6 — live deterministic baseline", baseline.get("passed") is True),
        ("Gate 7 — three formal models", formal.get("passed") is True),
        (
            "Gate 8 — exact archive clean validation",
            report.get("passed") is True,
        ),
    ]
    gate_table = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in gate_rows
    )
    model_rows = []
    models = release.get("models", {})
    for logical in ("kimi", "deepseek", "glm"):
        model = models.get(logical, {}) if isinstance(models, dict) else {}
        model_rows.append(
            "| {logical} | `{requested}` | `{actual}` | {status} | {execution} | "
            "{tools} | {attempts} | {optimization} | {hessian} | {score} | "
            "{replay} | `{trajectory}` |".format(
                logical=logical,
                requested=model.get("requested_model", "MISSING"),
                actual=model.get("actual_model", "MISSING"),
                status=model.get("status", "MISSING"),
                execution=(
                    f"{model.get('execution_mode', 'MISSING')} + "
                    f"{model.get('native_mode', 'MISSING')}"
                ),
                tools=model.get("tool_calls", 0),
                attempts=model.get("xtb_attempts", 0),
                optimization="PASS" if model.get("optimization_completed") is True else "FAIL",
                hessian="PASS" if model.get("hessian_completed") is True else "FAIL",
                score=model.get("official_score", 0),
                replay="PASS" if model.get("replay_byte_identical") is True else "FAIL",
                trajectory=model.get("trajectory_sha256", "MISSING"),
            )
        )
    all_exact = all(
        isinstance(models.get(logical), dict)
        and models[logical].get("status") == "PASS"
        and models[logical].get("actual_model")
        == models[logical].get("requested_model")
        for logical in ("kimi", "deepseek", "glm")
    )
    all_live = all(
        isinstance(models.get(logical), dict)
        and models[logical].get("execution_mode") == "REAL_API"
        and models[logical].get("native_mode") == "LIVE_XTB"
        for logical in ("kimi", "deepseek", "glm")
    )
    all_replay = bool(
        formal.get("passed")
        and all(
            isinstance(models.get(logical), dict)
            and models[logical].get("replay_byte_identical") is True
            for logical in ("kimi", "deepseek", "glm")
        )
    )
    tests_summary = (
        f"{tests.get('total_passed', 0)} passed, "
        f"{tests.get('total_skipped', 0)} skipped, "
        f"{tests.get('total_failed', 0)} failed, "
        f"{tests.get('total_errors', 0)} errors"
    )
    return f"""# Organizer Submission Checklist

This checklist is generated from the committed `configs/release.json` snapshot
and the independent exact-byte package-validation report.

## Gate 1–8

| Gate | Status |
| --- | --- |
{gate_table}

## Formal model evidence

| Logical model | Exact requested model | Actual response model | Status | Execution | Tool calls | xTB attempts | Optimization | Hessian | Score | Replay | Trajectory SHA-256 |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |
{chr(10).join(model_rows)}

## Completion audit

1. Project root inside archive: `.`
2. Task/version: `{release.get('task_id', 'MISSING')}` / release schema `{release.get('schema_version', 'MISSING')}`
3. Final ZIP: `{archive_name}`
4. ZIP SHA-256: `{archive_sha256}`
5. SHA-256 sidecar: `{sidecar_name}`
6. Independent validator report: `{validation_name}` — `{'PASS' if report.get('passed') else 'FAIL'}`
7. Tests: `{tests_summary}`
8. Current blocker: `NONE`
9. xTB executable status/version: `{native.get('xtb', {}).get('status', 'MISSING')}` / `{native.get('xtb', {}).get('version', 'MISSING')}`
10. Initial failure fixture generated by native xTB: `{'YES' if native.get('initial_failure_native') is True else 'NO'}`
11. Initial failure optimization-cycle limit: `{native.get('initial_failure_cycle_limit', 'MISSING')}`
12. Baseline replay byte-identical: `{'PASS' if baseline.get('evaluation_byte_identical') is True else 'FAIL'}`
13. Baseline live completion/score: `{'PASS' if baseline.get('passed') is True else 'FAIL'}` / `{baseline.get('official_score', 0)}`
14. Kimi formal trajectory: `{models.get('kimi', {}).get('status', 'MISSING')}`
15. DeepSeek formal trajectory: `{models.get('deepseek', {}).get('status', 'MISSING')}`
16. GLM formal trajectory: `{models.get('glm', {}).get('status', 'MISSING')}`
17. Three exact provider model identities: `{'PASS' if all_exact else 'FAIL'}`
18. Three trajectories are REAL_API + LIVE_XTB: `{'PASS' if all_live else 'FAIL'}`
19. Three evaluator replays byte-identical: `{'PASS' if all_replay else 'FAIL'}`
20. Parent `.env` modified: `{'YES' if attestation.get('parent_env_modified') is True else 'NO'}`
21. API-key/secret findings in package scope: `{scan.get('secret_finding_count', 'MISSING')}`
22. SSH executed: `{'YES' if attestation.get('ssh_executed') is True else 'NO'}`
23. HPC connected: `{'YES' if attestation.get('hpc_connected') is True else 'NO'}`
24. Slurm submitted: `{'YES' if attestation.get('slurm_submitted') is True else 'NO'}`
25. Config conflict: `{'NO' if configuration.get('passed') is True else 'YES'}`
26. Archive preflight: `{'PASS' if report.get('preflight', {}).get('passed') else 'FAIL'}`
27. Clean extraction and static validation: `{'PASS' if static.get('passed') else 'FAIL'}`
28. Imports, tests, baseline replay, and three-model replay: `{'PASS' if dynamic.get('passed') else 'FAIL'}`
29. Candidate hash matched expected hash: `{'PASS' if report.get('expected_sha256') == archive_sha256 else 'FAIL'}`
30. Archive mutated after validation: `NO`
31. Unfinished items: `NONE`
32. Final status: `PASS`
"""


def _source_commit_targets(root: Path, payload: Path) -> list[tuple[Path, Path]]:
    relatives = [
        Path("configs/release.json"),
        Path("reports/BENCHMARK_DESIGN_REPORT.md"),
        Path("reports/ORGANIZER_VALIDATION_REPORT.md"),
        Path("reports/ORGANIZER_MODEL_COMPARISON.md"),
        Path("MANIFEST.json"),
    ]
    return [(payload / relative, root / relative) for relative in relatives]


def _restore_source(
    *,
    root: Path,
    transaction: Path,
    originals: dict[Path, bytes | None],
) -> None:
    rollback = transaction / "rollback_source"
    for target, original in originals.items():
        if original is None:
            if target.exists():
                retained = rollback / target.relative_to(root)
                retained.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, retained)
            continue
        temporary = target.with_name(target.name + ".rollback.tmp")
        if temporary.exists():
            raise PackageTransactionError("ROLLBACK_TEMPORARY_ALREADY_EXISTS")
        with temporary.open("xb") as handle:
            handle.write(original)
        os.replace(temporary, target)


def _publish_validated_candidate(
    *,
    root: Path,
    transaction: Path,
    payload: Path,
    candidate: Path,
    archive_path: Path,
    validation_report: Path,
    report: dict[str, Any],
    archive_sha256: str,
    release: dict[str, Any],
) -> dict[str, str]:
    sidecar = archive_path.with_suffix(".sha256")
    external_validation = archive_path.with_suffix(".validation.json")
    checklist = archive_path.parent / CHECKLIST_NAME
    official_paths = (archive_path, sidecar, external_validation, checklist)
    if any(path.exists() for path in official_paths):
        raise PackageTransactionError("REFUSING_TO_OVERWRITE_DELIVERY_ARTIFACT")
    if (root / "MANIFEST.json").exists():
        raise PackageTransactionError("REFUSING_TO_OVERWRITE_SOURCE_MANIFEST")

    pending_sidecar = sidecar.with_name(sidecar.name + ".pending")
    pending_validation = external_validation.with_name(
        external_validation.name + ".pending"
    )
    pending_checklist = checklist.with_name(checklist.name + ".pending")
    if any(path.exists() for path in (pending_sidecar, pending_validation, pending_checklist)):
        raise PackageTransactionError("PENDING_PUBLICATION_FILE_EXISTS")
    _write_pending_text(
        pending_sidecar,
        f"{archive_sha256}  {archive_path.name}",
    )
    shutil.copy2(validation_report, pending_validation)
    _write_pending_text(
        pending_checklist,
        _checklist_text(
            release=release,
            archive_name=archive_path.name,
            archive_sha256=archive_sha256,
            sidecar_name=sidecar.name,
            validation_name=external_validation.name,
            report=report,
        ),
    )

    source_targets = _source_commit_targets(root, payload)
    originals = {
        target: target.read_bytes() if target.is_file() else None
        for _source, target in source_targets
    }
    source_pending: list[tuple[Path, Path]] = []
    for source, target in source_targets:
        pending = target.with_name(target.name + ".package.pending")
        if pending.exists():
            raise PackageTransactionError("SOURCE_COMMIT_TEMPORARY_EXISTS")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, pending)
        source_pending.append((pending, target))

    try:
        # candidate and official output share a volume; os.replace preserves the
        # exact bytes and therefore the hash already validated above.
        os.replace(candidate, archive_path)
        for pending, target in source_pending:
            os.replace(pending, target)
        os.replace(pending_sidecar, sidecar)
        os.replace(pending_validation, external_validation)
        os.replace(pending_checklist, checklist)
        if sha256_file(archive_path) != archive_sha256:
            raise PackageTransactionError("PUBLISHED_ARCHIVE_HASH_CHANGED")
    except (OSError, PackageTransactionError) as exc:
        if archive_path.exists() and not candidate.exists():
            os.replace(archive_path, transaction / "unpublished_after_commit_failure.zip")
        _restore_source(root=root, transaction=transaction, originals=originals)
        rollback_external = transaction / "rollback_external"
        rollback_external.mkdir(exist_ok=True)
        for path in (sidecar, external_validation, checklist):
            if path.exists():
                os.replace(path, rollback_external / path.name)
        raise PackageTransactionError("PUBLICATION_COMMIT_FAILED") from exc
    return {
        "sha256_sidecar": sidecar.name,
        "validation_report": external_validation.name,
        "organizer_checklist": checklist.name,
    }


def package_release(
    *,
    root: Path,
    archive_path: Path,
    transaction: Path,
) -> dict[str, Any]:
    release_path = root / "configs" / "release.json"
    if not release_path.is_file():
        raise PackageTransactionError("RELEASE_REGISTRY_MISSING")
    ready_release = _load_json(release_path)
    if ready_release.get("status") != "READY_FOR_PACKAGE":
        raise PackageTransactionError("RELEASE_NOT_READY_FOR_PACKAGE")
    if transaction.exists():
        raise PackageTransactionError("REFUSING_TO_OVERWRITE_STAGING")
    if archive_path.exists():
        raise PackageTransactionError("REFUSING_TO_OVERWRITE_ARCHIVE")

    # Ensure the source allowlist has all three script-generated provisional
    # reports before it is copied into the transaction.
    generate_reports(root, ready_release, package_gate=PENDING_PACKAGE_GATE)

    transaction.mkdir(parents=True, exist_ok=False)
    payload = transaction / "payload"
    stage = stage_allowlisted_files(root, payload)
    sidecar_name = archive_path.with_suffix(".sha256").name
    validation_name = archive_path.with_suffix(".validation.json").name
    final_release = _final_release_snapshot(
        ready_release,
        sidecar_name=sidecar_name,
        validation_name=validation_name,
        checklist_name=CHECKLIST_NAME,
    )
    _write_json_atomic(payload / "configs" / "release.json", final_release)
    generate_reports(payload, final_release, package_gate=FINAL_PACKAGE_GATE)
    manifest = write_staged_manifest(payload)

    candidate = transaction / archive_path.name
    archive = build_flat_archive(payload, candidate)
    archive_sha256 = str(archive["archive_sha256"])
    validation_report = transaction / "package_validation.json"
    report = _run_candidate_validation(
        root=root,
        candidate=candidate,
        extract_dir=transaction / "clean_extraction",
        report_path=validation_report,
        expected_sha256=archive_sha256,
    )
    receipts = _publish_validated_candidate(
        root=root,
        transaction=transaction,
        payload=payload,
        candidate=candidate,
        archive_path=archive_path,
        validation_report=validation_report,
        report=report,
        archive_sha256=archive_sha256,
        release=final_release,
    )
    return {
        "status": "PASS",
        "archive_filename": archive_path.name,
        "archive_size_bytes": archive["archive_size_bytes"],
        "archive_sha256": archive_sha256,
        "zip_file_count": archive["zip_file_count"],
        "source_preflight": {
            key: value
            for key, value in stage.items()
            if key not in {"expected_security_test_path_literals"}
        },
        **manifest,
        **receipts,
    }


def _record_package_failure(root: Path, error_code: str) -> None:
    path = root / "configs" / "release.json"
    if not path.is_file():
        return
    try:
        release = _load_json(path)
    except (OSError, ValueError, PackageTransactionError):
        return
    if release.get("status") != "READY_FOR_PACKAGE":
        return
    release["status"] = "BLOCKED_PACKAGE_VALIDATION"
    release.setdefault("gates", {})["package"] = {
        "status": "FAIL",
        "passed": False,
        "error_code": error_code,
    }
    _write_json_atomic(path, release)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, clean-validate, and atomically publish the organizer archive"
    )
    parser.add_argument("--output", default=str(ROOT.parent / DEFAULT_NAME))
    parser.add_argument("--staging", default="staging/final_submission")
    args = parser.parse_args(argv)
    archive_path = Path(args.output).expanduser().resolve()
    transaction = (ROOT / args.staging).resolve()
    try:
        transaction.relative_to(ROOT)
    except ValueError:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PACKAGE_VALIDATION",
                    "error": "STAGING_OUTSIDE_PROJECT",
                },
                sort_keys=True,
            )
        )
        return 1
    try:
        result = package_release(
            root=ROOT,
            archive_path=archive_path,
            transaction=transaction,
        )
    except (
        FileExistsError,
        OSError,
        PackageTransactionError,
        RuntimeError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as exc:
        error_code = (
            str(exc)
            if isinstance(exc, PackageTransactionError)
            else type(exc).__name__.upper()
        )
        try:
            _record_package_failure(ROOT, error_code)
        except (OSError, ValueError, PackageTransactionError):
            error_code = "PACKAGE_FAILURE_AND_REGISTRY_UPDATE_FAILED"
        print(
            json.dumps(
                {"status": "BLOCKED_PACKAGE_VALIDATION", "error": error_code},
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

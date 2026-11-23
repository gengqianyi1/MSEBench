from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from package_tools.archive import build_flat_archive, write_staged_manifest
from package_tools.archive_validation import (
    extract_safely,
    preflight_archive,
    validate_release_commit_claim,
)
from package_tools.historical_integrity import (
    HISTORICAL_INTEGRITY_SCHEMA_VERSION,
    TREE_ALGORITHM,
    V1_1_CAMPAIGN_BASELINE,
    V1_1_PROJECT_BASELINE,
    V1_ARCHIVE_BASELINE,
    V1_PROJECT_BASELINE,
    validate_historical_integrity_receipt,
)
from package_tools.manifest import file_entries, validate_manifest, write_manifest
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
    required_delivery_paths,
)
from package_tools.scanners import (
    scan_forbidden_names,
    scan_runtime_absolute_paths,
    scan_secrets,
)


def _zip(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)
    return path


def _root_members() -> list[tuple[str, bytes]]:
    return [
        ("README.md", b"readme\n"),
        ("TASK.md", b"task\n"),
        ("run_all.ps1", b"exit 0\n"),
        ("MANIFEST.json", b"{}\n"),
    ]


def _historical_receipt() -> dict:
    return {
        "schema_version": HISTORICAL_INTEGRITY_SCHEMA_VERSION,
        "status": "PASS",
        "algorithm": TREE_ALGORITHM,
        "historical_write_operations": False,
        "v1": {**V1_PROJECT_BASELINE, "matches_pre_work": True},
        "v1_archive": {
            **V1_ARCHIVE_BASELINE,
            "matches_pre_work": True,
        },
        "v1_1": {
            **V1_1_PROJECT_BASELINE,
            "matches_pre_work": True,
            "campaign": {
                **V1_1_CAMPAIGN_BASELINE,
                "matches_pre_work": True,
            },
        },
    }


def _write_minimal_required_delivery_tree(root: Path) -> None:
    for relative in sorted(required_delivery_paths()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "configs/task.yaml":
            content = (
                'schema_version: "2.0"\n'
                "task:\n"
                f"  id: {EXPECTED_TASK_ID}\n"
                f'  benchmark_version: "{EXPECTED_BENCHMARK_VERSION}"\n'
                f"  case_id: {EXPECTED_CASE_ID}\n"
                f"  protocol: {EXPECTED_PROTOCOL}\n"
            )
        elif relative == "configs/release.json":
            content = json.dumps(
                {
                    "schema_version": "2.0",
                    "status": "BLOCKED",
                    "release_ready": False,
                    "task_id": EXPECTED_TASK_ID,
                    "benchmark_version": EXPECTED_BENCHMARK_VERSION,
                    "protocol": EXPECTED_PROTOCOL,
                    "models": {},
                }
            )
        elif relative == "reports/HISTORICAL_RELEASE_INTEGRITY_POST.json":
            content = json.dumps(_historical_receipt())
        elif path.suffix in {".json", ".jsonl"}:
            content = "{}\n"
        else:
            content = "release fixture\n"
        path.write_text(content, encoding="utf-8", newline="\n")


@pytest.mark.parametrize(
    "member,expected",
    [
        ("../escape.txt", "UNSAFE_MEMBER_COMPONENT"),
        ("C:/escape.txt", "UNSAFE_MEMBER_COLON"),
        ("nested/project.exe", "UNEXPECTED_EXTENSION"),
    ],
)
def test_release_zip_rejects_zip_slip_drive_and_unexpected_extension(
    tmp_path: Path, member: str, expected: str
) -> None:
    archive = _zip(tmp_path / "bad.zip", _root_members() + [(member, b"x")])
    result = preflight_archive(archive)
    assert result["passed"] is False
    assert expected in result["errors"]


def test_release_zip_rejects_casefold_duplicate(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "duplicate.zip", _root_members() + [("readme.md", b"collision")])
    result = preflight_archive(archive)
    assert result["passed"] is False
    assert "DUPLICATE_OR_COLLIDING_MEMBER" in result["errors"]


def test_release_zip_rejects_nested_project_root(tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "nested.zip",
        [(f"wrapper/{name}", content) for name, content in _root_members()],
    )
    result = preflight_archive(archive)
    assert result["passed"] is False
    assert "ROOT_ENTRYPOINT_MISSING" in result["errors"]


@pytest.mark.parametrize(
    "wrapper",
    [
        "adaptive_xtb_torsion_profile_agent_v1",
        "adaptive_xtb_torsion_profile_agent_v1_1",
        "adaptive_xtb_torsion_profile_agent_v2",
        "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark",
        "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v1_1",
        "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v2",
    ],
)
def test_release_zip_explicitly_rejects_old_and_new_wrapper_roots(
    tmp_path: Path, wrapper: str
) -> None:
    archive = _zip(
        tmp_path / EXPECTED_ARCHIVE_FILENAME,
        [(f"{wrapper}/{name}", content) for name, content in _root_members()],
    )
    result = preflight_archive(archive)
    assert result["passed"] is False
    assert "UNWANTED_WRAPPER_DIRECTORY" in result["errors"]


@pytest.mark.parametrize(
    "member",
    [
        ".env",
        ".env.local",
        "reports/.env.example",
        "reports/raw_provider_response.json",
        "reports/provider_raw-response.json",
        "reports/raw_provider/response.json",
        "reports/debug_dump/data.json",
        "reports/debug_dump.json",
        "tests/__pycache__/module.py",
        "reports/.pytest_cache/state.json",
        "trajectories/kimi/artifacts/stdout.log",
        "trajectories/kimi/artifacts/charges",
    ],
)
def test_release_zip_rejects_env_raw_provider_debug_and_cache_members(
    tmp_path: Path, member: str
) -> None:
    archive = _zip(
        tmp_path / EXPECTED_ARCHIVE_FILENAME,
        _root_members() + [(member, b"must not ship")],
    )
    result = preflight_archive(archive)
    assert result["passed"] is False
    assert "FORBIDDEN_RELEASE_MEMBER" in result["errors"]


@pytest.mark.parametrize("name", ["stdout.log", "stderr.log"])
def test_only_projected_attempt_logs_are_releasable(name: str) -> None:
    assert allowed_release_file(
        Path(
            "trajectories/kimi/artifacts/scan_summaries/"
            f"batch_01_angle_090_attempt_01/{name}"
        )
    )
    assert not allowed_release_file(
        Path(f"trajectories/kimi/artifacts/{name}")
    )
    assert not allowed_release_file(
        Path(
            "trajectories/kimi/artifacts/scan_summaries/"
            f"not-an-attempt/{name}"
        )
    )


@pytest.mark.parametrize("name", ["stdout.log", "stderr.log"])
def test_archive_extension_gate_accepts_only_exact_projected_attempt_logs(
    tmp_path: Path, name: str
) -> None:
    exact = (
        "trajectories/kimi/artifacts/scan_summaries/"
        f"batch_01_angle_090_attempt_01/{name}"
    )
    accepted = preflight_archive(
        _zip(
            tmp_path / EXPECTED_ARCHIVE_FILENAME,
            _root_members() + [(exact, b"projected native output\n")],
        )
    )
    assert "UNEXPECTED_EXTENSION" not in accepted["errors"]
    assert "UNEXPECTED_RELEASE_MEMBER" not in accepted["errors"]
    assert "FORBIDDEN_RELEASE_MEMBER" not in accepted["errors"]

    rejected = preflight_archive(
        _zip(
            tmp_path / ("bad-" + EXPECTED_ARCHIVE_FILENAME),
            _root_members() + [(f"reports/{name}", b"not allowlisted\n")],
        )
    )
    assert "UNEXPECTED_EXTENSION" not in rejected["errors"]
    assert "UNEXPECTED_RELEASE_MEMBER" in rejected["errors"]


def test_safe_extract_refuses_nonempty_destination(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "safe.zip", _root_members())
    destination = tmp_path / "extract"
    destination.mkdir()
    (destination / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        extract_safely(archive, destination)


def test_manifest_exact_inventory_and_tamper_detection(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    files = [tmp_path / "README.md"]
    payload = {
        "file_count_excluding_manifest": 1,
        "total_bytes_excluding_manifest": (tmp_path / "README.md").stat().st_size,
        "files": file_entries(tmp_path, files),
    }
    write_manifest(tmp_path / "MANIFEST.json", payload)
    assert validate_manifest(tmp_path)["passed"] is True
    (tmp_path / "README.md").write_text("tampered\n", encoding="utf-8")
    result = validate_manifest(tmp_path)
    assert result["passed"] is False
    assert "MANIFEST_HASH_MISMATCH" in result["errors"]


def test_manifest_rejects_total_bytes_mismatch_and_malformed_entries(
    tmp_path: Path,
) -> None:
    payload = {
        "file_count_excluding_manifest": 1,
        "total_bytes_excluding_manifest": 999,
        "files": [{"path": "../escape.txt", "size_bytes": "bad", "sha256": None}],
    }
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = validate_manifest(tmp_path)
    assert result["passed"] is False
    assert "MANIFEST_UNSAFE_PATH" in result["errors"]
    assert "MANIFEST_TOTAL_BYTES_MISMATCH" in result["errors"]


def test_archive_preflight_recomputes_manifest_exact_coverage(
    tmp_path: Path,
) -> None:
    payload_members = [
        ("README.md", b"readme\n"),
        ("TASK.md", b"task\n"),
        ("run_all.ps1", b"exit 0\n"),
        ("reports/surprise.json", b"{}\n"),
    ]
    listed_members = payload_members[:-1]
    manifest = {
        "schema_version": "1.0",
        "submission_name": EXPECTED_SUBMISSION_NAME,
        "archive_filename": EXPECTED_ARCHIVE_FILENAME,
        "archive_layout": "flat_project_root",
        "file_count_excluding_manifest": len(listed_members),
        "total_bytes_excluding_manifest": sum(len(data) for _, data in listed_members),
        "files": [
            {
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest().upper(),
            }
            for name, data in listed_members
        ],
    }
    archive = _zip(
        tmp_path / EXPECTED_ARCHIVE_FILENAME,
        payload_members
        + [("MANIFEST.json", json.dumps(manifest).encode("utf-8"))],
    )
    result = preflight_archive(archive)
    assert result["passed"] is False
    assert "ARCHIVE_MANIFEST_INVENTORY_MISMATCH" in result["errors"]
    assert "ARCHIVE_MANIFEST_COUNT_MISMATCH" in result["errors"]
    assert "ARCHIVE_MANIFEST_TOTAL_BYTES_MISMATCH" in result["errors"]


def test_archive_builder_refuses_noncanonical_v2_filename(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(ValueError, match="ARCHIVE_FILENAME_MUST_BE"):
        build_flat_archive(staging, tmp_path / "wrong-name.zip")


def test_package_primitives_build_flat_exact_manifest_archive(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "release_tree"
    staging.mkdir()
    _write_minimal_required_delivery_tree(staging)
    manifest_result = write_staged_manifest(staging)
    archive = tmp_path / EXPECTED_ARCHIVE_FILENAME
    archive_result = build_flat_archive(staging, archive)

    assert manifest_result["manifest_entries"] == len(required_delivery_paths())
    assert archive_result["archive_path"] == archive
    result = preflight_archive(archive)
    assert result["passed"] is True, result["errors"]
    assert result["missing_required_members"] == []
    with zipfile.ZipFile(archive, "r") as handle:
        names = handle.namelist()
    assert "README.md" in names
    assert all(
        not name.startswith("adaptive_xtb_torsion_profile_agent_v2/")
        for name in names
    )


def test_manifest_builder_rejects_stale_task_identity(tmp_path: Path) -> None:
    staging = tmp_path / "release_tree"
    staging.mkdir()
    _write_minimal_required_delivery_tree(staging)
    task_path = staging / "configs" / "task.yaml"
    task_path.write_text(
        task_path.read_text(encoding="utf-8").replace(
            EXPECTED_TASK_ID,
            "adaptive_xtb_torsion_profile_agent_v1_1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="STAGED_TASK_IDENTITY_MISMATCH"):
        write_staged_manifest(staging)


@pytest.mark.parametrize(
    "relative",
    [
        Path("reports/raw_provider_response.json"),
        Path("reports/debug_dump.json"),
        Path("tests/__pycache__/module.py"),
        Path(".env"),
    ],
)
def test_strict_release_tree_rejects_sensitive_or_cache_artifacts(
    tmp_path: Path, relative: Path
) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not releasable\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        collect_strict_release_tree(tmp_path)


def test_required_delivery_paths_include_v2_integrity_and_freeze_evidence() -> None:
    required = required_delivery_paths()
    assert "reports/HISTORICAL_RELEASE_INTEGRITY.md" in required
    assert "reports/HISTORICAL_RELEASE_INTEGRITY_POST.json" in required
    assert "reports/EXECUTION_FREEZE_REPORT.md" in required
    assert "reports/PROVIDER_PREFLIGHT.json" in required
    assert "reports/PROVIDER_PREFLIGHT_RECHARGE_01.json" in required
    assert "reports/kimi_formal_validation_recharge_01.json" in required
    assert "reports/deepseek_formal_validation_recharge_01.json" in required
    assert "reports/glm_formal_validation_recharge_01.json" in required
    assert "reports/replay_validation_recharge_01.json" in required
    assert "reports/gate1_scientific_smoke.json" in required
    assert "reports/oracle_independent_validation.json" in required
    assert "reports/pre_freeze_package_rehearsal.json" in required
    assert "reports/RELEASE_PATCH_CHANGELOG.md" in required
    assert "package_tools/historical_integrity.py" in required
    assert "scripts/release_patch_rehearsal.py" in required
    assert "scripts/verify_old_evidence.py" in required
    assert "execution_commitment.json" in required
    assert "release_commitment.json" in required


def test_secret_and_forbidden_file_scanners(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text(
        "Authorization: " + "Bea" + "rer " + "abcdefghijklmnopqrstuvwxyz",
        encoding="utf-8",
    )
    cache = tmp_path / "__pycache__" / "module.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"bytecode")
    assert scan_secrets(tmp_path, [secret])
    assert scan_forbidden_names(tmp_path, [cache]) == ["__pycache__/module.pyc"]


def test_secret_scanner_decodes_quoted_and_nested_json_assignments(
    tmp_path: Path,
) -> None:
    credential = "plain" + "credentialvalue123456"
    direct = tmp_path / "direct.json"
    nested = tmp_path / "nested.json"
    direct.write_text(
        json.dumps({"GJLD_API_KEY": credential}),
        encoding="utf-8",
    )
    nested.write_text(
        json.dumps(
            {
                "safe_message": json.dumps(
                    {
                        "authorization": (
                            "Bea" + "rer " + credential
                        )
                    }
                )
            }
        ),
        encoding="utf-8",
    )

    direct_findings = scan_secrets(tmp_path, [direct])
    nested_findings = scan_secrets(tmp_path, [nested])
    assert any(
        item["rule"] == "recursive_json_secret"
        for item in direct_findings
    )
    assert any(
        item["rule"] == "recursive_json_secret"
        for item in nested_findings
    )


def test_absolute_paths_are_tolerated_only_as_security_test_literals(
    tmp_path: Path,
) -> None:
    report = tmp_path / "reports" / "receipt.json"
    test_source = tmp_path / "tests" / "security" / "test_path.py"
    report.parent.mkdir(parents=True)
    test_source.parent.mkdir(parents=True)
    absolute = "C" + ":" + "\\host\\private\\attempt"
    report.write_text(json.dumps({"workdir": absolute}), encoding="utf-8")
    test_source.write_text(f'PATH = r"{absolute}"\n', encoding="utf-8")

    failures, literals = scan_runtime_absolute_paths(
        tmp_path, [report, test_source]
    )

    assert failures == [
        {"path": "reports/receipt.json", "line": 1, "rule": "windows_absolute"}
    ]
    assert literals == [
        {
            "path": "tests/security/test_path.py",
            "line": 1,
            "rule": "windows_absolute",
        }
    ]


@pytest.mark.parametrize(
    ("status", "release_ready"),
    [("PASS", True), ("FAIL", False), ("BLOCKED", False)],
)
def test_release_commit_accepts_closed_statuses_but_only_pass_is_ready(
    tmp_path: Path, status: str, release_ready: bool
) -> None:
    release = {
        "schema_version": "2.0",
        "status": status,
        "release_ready": release_ready,
        "task_id": EXPECTED_TASK_ID,
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
        "protocol": EXPECTED_PROTOCOL,
        "models": {},
        "gates": {
            "package": {
                "status": "PASS",
                "passed": True,
                "checks": {name: True for name in PACKAGE_GATE_CHECKS},
            }
        },
    }
    manifest = {
        "release_status": status,
        "models": {},
        "archive_filename": EXPECTED_ARCHIVE_FILENAME,
        "submission_name": EXPECTED_SUBMISSION_NAME,
        "archive_layout": "flat_project_root",
        "task_id": EXPECTED_TASK_ID,
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
        "case_id": EXPECTED_CASE_ID,
        "protocol": EXPECTED_PROTOCOL,
    }
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "release.json").write_text(
        json.dumps(release), encoding="utf-8"
    )
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "HISTORICAL_RELEASE_INTEGRITY_POST.json").write_text(
        json.dumps(_historical_receipt()),
        encoding="utf-8",
    )

    result = validate_release_commit_claim(tmp_path)

    assert result["passed"] is True
    assert result["release_ready"] is release_ready
    assert result["old_evidence_preservation_valid"] is True


def test_release_commit_rejects_post_preservation_mismatch(
    tmp_path: Path,
) -> None:
    release = {
        "schema_version": "2.0",
        "status": "FAIL",
        "release_ready": False,
        "task_id": EXPECTED_TASK_ID,
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
        "protocol": EXPECTED_PROTOCOL,
        "models": {},
        "gates": {
            "package": {
                "status": "PASS",
                "passed": True,
                "checks": {name: True for name in PACKAGE_GATE_CHECKS},
            }
        },
    }
    manifest = {
        "release_status": "FAIL",
        "models": {},
        "archive_filename": EXPECTED_ARCHIVE_FILENAME,
        "submission_name": EXPECTED_SUBMISSION_NAME,
        "archive_layout": "flat_project_root",
        "task_id": EXPECTED_TASK_ID,
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
        "case_id": EXPECTED_CASE_ID,
        "protocol": EXPECTED_PROTOCOL,
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "configs" / "release.json").write_text(
        json.dumps(release),
        encoding="utf-8",
    )
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    tampered = _historical_receipt()
    tampered["v1_1"]["campaign"]["content_commitment_sha256"] = "0" * 64
    (tmp_path / "reports" / "HISTORICAL_RELEASE_INTEGRITY_POST.json").write_text(
        json.dumps(tampered),
        encoding="utf-8",
    )
    result = validate_release_commit_claim(tmp_path)
    assert result["passed"] is False
    assert result["old_evidence_preservation_valid"] is False
    assert "OLD_EVIDENCE_PRESERVATION_NOT_PROVEN" in result["errors"]


def test_historical_receipt_contract_covers_v1_and_v1_1() -> None:
    result = validate_historical_integrity_receipt(_historical_receipt())
    assert result["passed"] is True
    assert result["v1_tree_sha256"] == V1_PROJECT_BASELINE["tree_sha256"]
    assert result["v1_archive_sha256"] == V1_ARCHIVE_BASELINE["sha256"]
    assert result["v1_1_tree_sha256"] == V1_1_PROJECT_BASELINE["tree_sha256"]
    assert result["v1_1_campaign_commitment"] == V1_1_CAMPAIGN_BASELINE[
        "content_commitment_sha256"
    ]

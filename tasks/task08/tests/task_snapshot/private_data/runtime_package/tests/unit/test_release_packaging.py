from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from package_tools.archive_validation import validate_extracted_static
from package_tools.manifest import file_entries, sha256_file, write_manifest
from package_tools.policy import PACKAGE_GATE_CHECKS
import scripts.package_submission as package_submission
import scripts.validate_project as validate_project


def _passing_states() -> dict[str, dict]:
    return {
        "config": {"passed": True},
        "tests": {"passed": True},
        "native": {"passed": True, "xtb": {"status": "AVAILABLE"}},
        "baseline": {"passed": True},
        "scan": {
            "passed": True,
            "secret_finding_count": 0,
            "runtime_absolute_path_finding_count": 0,
        },
        "formal": {
            "passed": True,
            "identity_mismatch_detected": False,
        },
    }


def _status(stage: str = "final", **changes: dict) -> str:
    states = _passing_states()
    states.update(changes)
    return validate_project._release_status(stage, **states)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"config": {"passed": False}}, "BLOCKED_CONFIG_CONFLICT"),
        (
            {"native": {"passed": False, "xtb": {"status": "UNAVAILABLE"}}},
            "BLOCKED_XTB_UNAVAILABLE",
        ),
        (
            {"native": {"passed": False, "xtb": {"status": "AVAILABLE"}}},
            "BLOCKED_INITIAL_FIXTURE",
        ),
        ({"tests": {"passed": False}}, "BLOCKED_TEST_FAILURE"),
        ({"baseline": {"passed": False}}, "BLOCKED_BASELINE_FAILURE"),
        (
            {"scan": {"passed": False, "secret_finding_count": 1}},
            "BLOCKED_SECRET_FINDING",
        ),
        (
            {"scan": {"passed": False, "secret_finding_count": 0}},
            "BLOCKED_PACKAGE_VALIDATION",
        ),
        (
            {"formal": {"passed": False, "identity_mismatch_detected": True}},
            "BLOCKED_MODEL_IDENTITY",
        ),
        (
            {"formal": {"passed": False, "identity_mismatch_detected": False}},
            "BLOCKED_MODEL_TRAJECTORIES",
        ),
    ],
)
def test_release_blocker_mapping(changes: dict, expected: str) -> None:
    assert _status(**changes) == expected


def test_release_status_success_states_and_priority() -> None:
    assert _status("pre-model") == "PRE_MODEL_GATES_PASS"
    assert _status("final") == "READY_FOR_PACKAGE"
    assert _status(
        config={"passed": False},
        native={"passed": False, "xtb": {"status": "UNAVAILABLE"}},
        tests={"passed": False},
    ) == "BLOCKED_CONFIG_CONFLICT"
    assert validate_project.BLOCKER_PRIORITY == (
        "BLOCKED_CONFIG_CONFLICT",
        "BLOCKED_XTB_UNAVAILABLE",
        "BLOCKED_INITIAL_FIXTURE",
        "BLOCKED_TEST_FAILURE",
        "BLOCKED_BASELINE_FAILURE",
        "BLOCKED_SECRET_FINDING",
        "BLOCKED_PACKAGE_VALIDATION",
        "BLOCKED_MODEL_IDENTITY",
        "BLOCKED_MODEL_TRAJECTORIES",
    )


def test_final_project_validation_stops_at_ready_for_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tests = {
        "passed": True,
        "groups": {},
        "total_passed": 1,
        "total_skipped": 0,
        "total_failed": 0,
        "total_errors": 0,
    }
    monkeypatch.setattr(validate_project, "validate_configuration", lambda: {"passed": True})
    monkeypatch.setattr(validate_project, "_test_gates", lambda: tests)
    monkeypatch.setattr(
        validate_project,
        "_native_gate",
        lambda: {"passed": True, "xtb": {"status": "AVAILABLE"}},
    )
    monkeypatch.setattr(validate_project, "_baseline_gate", lambda: {"passed": True})
    monkeypatch.setattr(
        validate_project,
        "_package_scope_scan",
        lambda: {"passed": True, "secret_finding_count": 0},
    )
    monkeypatch.setattr(
        validate_project,
        "_formal_gate",
        lambda: (
            {"passed": True, "identity_mismatch_detected": False},
            {"kimi": {}, "deepseek": {}, "glm": {}},
        ),
    )
    release = validate_project.validate("final", write_release=False)
    assert release["status"] == "READY_FOR_PACKAGE"
    assert release["gates"]["package"] == {"status": "NOT_RUN", "passed": False}


def _ready_release_value() -> dict:
    models = {}
    exact_models = {
        "kimi": "Pro/moonshotai/Kimi-K2.6",
        "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
        "glm": "zai-org/GLM-5.2",
    }
    for logical, model_name in exact_models.items():
        models[logical] = {
            "requested_model": model_name,
            "actual_model": model_name,
            "status": "PASS",
            "execution_mode": "REAL_API",
            "native_mode": "LIVE_XTB",
            "tool_calls": 12,
            "xtb_attempts": 2,
            "optimization_completed": True,
            "hessian_completed": True,
            "official_score": 100,
            "replay_byte_identical": True,
            "trajectory_sha256": "A" * 64,
        }
    return {
        "schema_version": "1.0",
        "task_id": "xtb_failure_diagnosis_restart_agent_v1",
        "status": "READY_FOR_PACKAGE",
        "gates": {
            "configuration": {"passed": True},
            "tests": {
                "passed": True,
                "total_passed": 100,
                "total_skipped": 2,
                "total_failed": 0,
                "total_errors": 0,
            },
            "native_xtb": {
                "passed": True,
                "initial_failure_native": True,
                "initial_failure_cycle_limit": 1,
                "xtb": {"status": "AVAILABLE", "version": "6.7.1"},
            },
            "baseline": {
                "passed": True,
                "official_score": 100,
                "evaluation_byte_identical": True,
            },
            "security_and_path_scan": {
                "passed": True,
                "secret_finding_count": 0,
            },
            "formal_models": {"passed": True},
            "package": {"status": "NOT_RUN", "passed": False},
        },
        "models": models,
    }


def _final_release() -> dict:
    return package_submission._final_release_snapshot(
        _ready_release_value(),
        sidecar_name="submission.sha256",
        validation_name="submission.validation.json",
        checklist_name="ORGANIZER_SUBMISSION_CHECKLIST.md",
    )


def _write_static_tree(root: Path, release: dict) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "release.json").write_text(
        json.dumps(release, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    files = [root / "configs" / "release.json"]
    write_manifest(
        root / "MANIFEST.json",
        {
            "release_status": release["status"],
            "models": release["models"],
            "file_count_excluding_manifest": len(files),
            "files": file_entries(root, files),
        },
    )


def test_static_validation_requires_consistent_committed_pass(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    _write_static_tree(root, _final_release())
    result = validate_extracted_static(root)
    assert result["passed"] is True
    assert result["release_commit"]["package_gate_checks_complete"] is True

    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_status"] = "READY_FOR_PACKAGE"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = validate_extracted_static(root)
    assert result["passed"] is False
    assert "RELEASE_MANIFEST_STATUS_MISMATCH" in result["errors"]


def _stub_transaction_build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_passes: bool,
) -> None:
    report_names = (
        "BENCHMARK_DESIGN_REPORT.md",
        "ORGANIZER_VALIDATION_REPORT.md",
        "ORGANIZER_MODEL_COMPARISON.md",
    )

    def reports(root: Path, release: dict, *, package_gate: str) -> dict:
        directory = root / "reports"
        directory.mkdir(exist_ok=True)
        for name in report_names:
            (directory / name).write_text(
                f"{release['status']} {package_gate}\n",
                encoding="utf-8",
                newline="\n",
            )
        return {"report_count": 3}

    def stage(root: Path, payload: Path) -> dict:
        (payload / "configs").mkdir(parents=True)
        shutil.copy2(root / "configs" / "release.json", payload / "configs" / "release.json")
        shutil.copytree(root / "reports", payload / "reports")
        return {"passed": True, "file_count": 4}

    def manifest(payload: Path) -> dict:
        (payload / "MANIFEST.json").write_text(
            "{}\n", encoding="utf-8", newline="\n"
        )
        return {"staged_file_count": 5, "manifest_entries": 4}

    def archive(_payload: Path, candidate: Path) -> dict:
        candidate.write_bytes(b"exact validated candidate bytes")
        return {
            "archive_size_bytes": candidate.stat().st_size,
            "archive_sha256": sha256_file(candidate),
            "zip_file_count": 5,
        }

    def validate(
        *,
        candidate: Path,
        report_path: Path,
        expected_sha256: str,
        **_kwargs: object,
    ) -> dict:
        if not validation_passes:
            raise package_submission.PackageTransactionError(
                "CLEAN_EXTRACTION_VALIDATION_FAILED"
            )
        report = {
            "passed": True,
            "archive_sha256": sha256_file(candidate),
            "expected_sha256": expected_sha256,
            "preflight": {"passed": True},
            "static_extraction_validation": {"passed": True},
            "dynamic_validation": {"passed": True},
        }
        report_path.write_text(
            json.dumps(report, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return report

    monkeypatch.setattr(package_submission, "generate_reports", reports)
    monkeypatch.setattr(package_submission, "stage_allowlisted_files", stage)
    monkeypatch.setattr(package_submission, "write_staged_manifest", manifest)
    monkeypatch.setattr(package_submission, "build_flat_archive", archive)
    monkeypatch.setattr(package_submission, "_run_candidate_validation", validate)


def _ready_source(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "release.json").write_text(
        json.dumps(_ready_release_value(), sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_failed_candidate_is_not_published_and_source_is_not_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _ready_source(root)
    _stub_transaction_build(monkeypatch, validation_passes=False)
    archive = tmp_path / package_submission.DEFAULT_NAME
    transaction = root / "staging" / "failure"

    with pytest.raises(
        package_submission.PackageTransactionError,
        match="CLEAN_EXTRACTION_VALIDATION_FAILED",
    ):
        package_submission.package_release(
            root=root,
            archive_path=archive,
            transaction=transaction,
        )
    source = json.loads((root / "configs" / "release.json").read_text(encoding="utf-8"))
    staged = json.loads(
        (transaction / "payload" / "configs" / "release.json").read_text(encoding="utf-8")
    )
    assert source["status"] == "READY_FOR_PACKAGE"
    assert staged["status"] == "PASS"
    assert not archive.exists()
    assert (transaction / archive.name).is_file()


def test_validated_candidate_exact_bytes_are_atomically_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _ready_source(root)
    _stub_transaction_build(monkeypatch, validation_passes=True)
    archive = tmp_path / package_submission.DEFAULT_NAME
    transaction = root / "staging" / "success"

    result = package_submission.package_release(
        root=root,
        archive_path=archive,
        transaction=transaction,
    )
    release = json.loads((root / "configs" / "release.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert archive.read_bytes() == b"exact validated candidate bytes"
    assert sha256_file(archive) == result["archive_sha256"]
    assert release["status"] == "PASS"
    assert release["gates"]["package"]["passed"] is True
    assert set(release["gates"]["package"]["checks"]) == set(PACKAGE_GATE_CHECKS)
    assert archive.with_suffix(".sha256").is_file()
    assert archive.with_suffix(".validation.json").is_file()
    checklist_path = archive.parent / package_submission.CHECKLIST_NAME
    assert checklist_path.is_file()
    assert not (transaction / archive.name).exists()
    checklist = checklist_path.read_text(encoding="utf-8")
    for gate in range(1, 9):
        assert f"Gate {gate}" in checklist
    assert "Pro/moonshotai/Kimi-K2.6" in checklist
    assert "deepseek-ai/DeepSeek-V4-Pro" in checklist
    assert "zai-org/GLM-5.2" in checklist
    assert "Three trajectories are REAL_API + LIVE_XTB: `PASS`" in checklist
    assert "SSH executed: `NO`" in checklist
    assert "HPC connected: `NO`" in checklist
    assert "Slurm submitted: `NO`" in checklist
    assert "Unfinished items: `NONE`" in checklist
    assert "Final status: `PASS`" in checklist

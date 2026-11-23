from __future__ import annotations

import copy
from pathlib import Path

import pytest

from framework.execution_freeze import (
    ExecutionFreezeError,
    execution_file_entries,
    load_execution_freeze_binding,
)
from package_tools.commitments import (
    CommitmentError,
    build_execution_commitment,
    build_release_commitment,
    discover_execution_files,
    discover_release_files,
    discover_result_files,
    validate_execution_commitment,
    validate_release_commitment,
    verify_execution_commitment,
    verify_release_commitment,
    write_execution_commitment,
    write_release_commitment,
)
from package_tools.provenance import canonical_json_bytes, sha256_bytes


def _write(root: Path, relative: str, text: str) -> Path:
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _tree(root: Path) -> None:
    for relative, content in {
        "TASK.md": "task v2\n",
        "environment.yml": "name: v2\n",
        "requirements.txt": "numpy\n",
        "README.md": "release docs\n",
        "LICENSE": "license\n",
        "LICENSES.md": "licenses\n",
        "NOTICE": "notice\n",
        ".env.example": "GJLD_API_KEY=PLACEHOLDER\n",
        ".gitignore": ".env\n",
        "pytest.ini": "[pytest]\n",
        "run_all.ps1": "exit 0\n",
        "run_all.sh": "exit 0\n",
        "baselines/baseline_agent.py": "POLICY = 'v2'\n",
        "benchmark_case/reference.xyz": "1\nreference\nH 0 0 0\n",
        "configs/task.yaml": "task: {id: v2}\n",
        "configs/tool_contracts.json": "{}\n",
        "configs/release.json": '{"status":"BLOCKED"}\n',
        "evaluator/evaluate.py": "VERSION = '2.0'\n",
        "evaluator_private/oracle_profile.json": '{"points":[]}\n',
        "framework/agent_runner.py": "RUNNER = 2\n",
        "framework/provider_retry.py": "ATTEMPTS = 4\n",
        "prompts/agent_prompt.md": "Return JSON.\n",
        "tools/science.py": "METHOD = 'GFN2-xTB'\n",
        "tools/tool_registry.py": "TOOLS = ('run_scan_batch',)\n",
        "scripts/run_model.py": "RUN = True\n",
        "scripts/package_submission.py": "PACKAGE = True\n",
        "scripts/pre_freeze_package_rehearsal.py": "REHEARSAL = 1\n",
        "scripts/release_patch_rehearsal.py": "REHEARSAL = 2\n",
        "scripts/replay_trajectories.py": "REPLAY = True\n",
        "scripts/secret_scan.py": "SCAN = True\n",
        "package_tools/scanners.py": "SCANNER = 2\n",
        "package_tools/provenance.py": "PROJECTION = 2\n",
        "reports/INHERITANCE_AUDIT.md": "audit\n",
        "tests/security/test_package.py": "def test_package(): pass\n",
        "trajectories/baseline/trajectory.jsonl": '{"event":"end"}\n',
        "trajectories/baseline/final_answer.json": "{}\n",
        "trajectories/baseline/evaluation.json": "{}\n",
        "trajectories/baseline/run_manifest.json": "{}\n",
        "trajectories/baseline/usage_report.json": "{}\n",
    }.items():
        _write(root, relative, content)


def _relative(root: Path, paths: list[Path]) -> set[str]:
    return {path.relative_to(root).as_posix() for path in paths}


def _freeze_execution(root: Path) -> dict:
    return write_execution_commitment(root)


def test_execution_and_release_discovery_are_separate(tmp_path: Path) -> None:
    _tree(tmp_path)
    execution = _relative(tmp_path, discover_execution_files(tmp_path))
    release = _relative(tmp_path, discover_release_files(tmp_path))
    results = _relative(tmp_path, discover_result_files(tmp_path))

    assert {
        "framework/agent_runner.py",
        "tools/science.py",
        "evaluator/evaluate.py",
        "evaluator_private/oracle_profile.json",
        "scripts/run_model.py",
        "benchmark_case/reference.xyz",
    }.issubset(execution)
    assert "package_tools/provenance.py" not in execution
    assert "scripts/package_submission.py" not in execution
    assert "scripts/pre_freeze_package_rehearsal.py" not in execution
    assert "scripts/release_patch_rehearsal.py" not in execution
    assert "scripts/replay_trajectories.py" not in execution
    assert "reports/INHERITANCE_AUDIT.md" not in execution
    assert not any(path.startswith("tests/") for path in execution)
    assert not any(path.startswith("trajectories/") for path in execution)

    assert {
        "package_tools/provenance.py",
        "package_tools/scanners.py",
        "scripts/package_submission.py",
        "scripts/pre_freeze_package_rehearsal.py",
        "scripts/release_patch_rehearsal.py",
        "scripts/replay_trajectories.py",
        "scripts/secret_scan.py",
        "reports/INHERITANCE_AUDIT.md",
        "tests/security/test_package.py",
        "README.md",
        "configs/release.json",
    }.issubset(release)
    assert "framework/agent_runner.py" not in release
    assert "scripts/run_model.py" not in release
    assert not any(path.startswith("trajectories/") for path in release)

    assert len(results) == 5
    assert all(path.startswith("trajectories/baseline/") for path in results)


def test_execution_commitment_is_deterministic_and_exact(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    first = build_execution_commitment(tmp_path)
    second = build_execution_commitment(tmp_path)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    result = validate_execution_commitment(tmp_path, first)
    assert result["passed"] is True
    assert result["content_sha256"] == first["content_sha256"]
    assert result["commitment_file_sha256"] == sha256_bytes(
        canonical_json_bytes(first)
    )


def test_frozen_runner_and_release_builder_agree_on_execution_inventory(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    package_entries = build_execution_commitment(tmp_path)["files"]
    assert execution_file_entries(tmp_path) == package_entries
    payload = write_execution_commitment(tmp_path)
    binding = load_execution_freeze_binding(tmp_path)
    assert binding["content_sha256"] == payload["content_sha256"]
    assert binding["file_count"] == payload["file_count"]
    assert binding["total_bytes"] == payload["total_bytes"]

    _write(tmp_path, "framework/agent_runner.py", "RUNNER = 3\n")
    with pytest.raises(
        ExecutionFreezeError, match="EXECUTION_COMMITMENT_MISMATCH"
    ):
        load_execution_freeze_binding(tmp_path)


def test_release_patch_does_not_change_execution_commitment(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    execution = _freeze_execution(tmp_path)
    before = build_release_commitment(tmp_path)
    _write(tmp_path, "package_tools/scanners.py", "SCANNER = 3\n")

    assert build_execution_commitment(tmp_path) == execution
    with pytest.raises(CommitmentError, match="RELEASE_COMMITMENT_MISMATCH"):
        validate_release_commitment(tmp_path, before)
    after = build_release_commitment(tmp_path)
    assert (
        after["execution_binding"]["content_sha256"]
        == before["execution_binding"]["content_sha256"]
        == execution["content_sha256"]
    )
    assert after["content_sha256"] != before["content_sha256"]


def test_release_commitment_binds_every_result_hash(tmp_path: Path) -> None:
    _tree(tmp_path)
    execution = _freeze_execution(tmp_path)
    release = build_release_commitment(tmp_path)
    assert release["execution_binding"]["content_sha256"] == execution[
        "content_sha256"
    ]
    assert release["result_file_count"] == 5
    assert {
        item["path"] for item in release["result_files"]
    } == _relative(tmp_path, discover_result_files(tmp_path))
    validate_release_commitment(tmp_path, release)

    _write(
        tmp_path,
        "trajectories/baseline/evaluation.json",
        '{"score":100}\n',
    )
    assert build_execution_commitment(tmp_path) == execution
    with pytest.raises(CommitmentError, match="RELEASE_COMMITMENT_MISMATCH"):
        validate_release_commitment(tmp_path, release)


def test_execution_change_invalidates_execution_and_release_binding(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    execution = _freeze_execution(tmp_path)
    release = build_release_commitment(tmp_path)
    _write(tmp_path, "framework/agent_runner.py", "RUNNER = 999\n")

    with pytest.raises(CommitmentError, match="EXECUTION_COMMITMENT_MISMATCH"):
        validate_execution_commitment(tmp_path, execution)
    with pytest.raises(CommitmentError, match="EXECUTION_COMMITMENT_MISMATCH"):
        validate_release_commitment(tmp_path, release)


def test_commitment_closed_schema_and_verifier_fail_closed(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    execution = build_execution_commitment(tmp_path)
    unknown = copy.deepcopy(execution)
    unknown["timestamp"] = "not permitted"
    assert verify_execution_commitment(tmp_path, unknown) == {
        "passed": False,
        "error": "EXECUTION_COMMITMENT_SCHEMA",
    }

    _freeze_execution(tmp_path)
    release = build_release_commitment(tmp_path)
    release_unknown = copy.deepcopy(release)
    release_unknown["host_path"] = "redacted"
    assert verify_release_commitment(tmp_path, release_unknown) == {
        "passed": False,
        "error": "RELEASE_COMMITMENT_SCHEMA",
    }


def test_noncanonical_execution_file_is_rejected_by_release_builder(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    execution = build_execution_commitment(tmp_path)
    (tmp_path / "execution_commitment.json").write_text(
        "{}",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        CommitmentError, match="EXECUTION_COMMITMENT_FILE_INVALID"
    ):
        build_release_commitment(tmp_path, execution_payload=execution)


def test_write_commitments_are_canonical_exclusive_and_self_validating(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)
    execution = write_execution_commitment(tmp_path)
    assert (tmp_path / "execution_commitment.json").read_bytes() == (
        canonical_json_bytes(execution)
    )
    with pytest.raises(
        CommitmentError, match="EXECUTION_COMMITMENT_ALREADY_EXISTS"
    ):
        write_execution_commitment(tmp_path)

    release = write_release_commitment(tmp_path)
    assert (tmp_path / "release_commitment.json").read_bytes() == (
        canonical_json_bytes(release)
    )
    assert validate_release_commitment(tmp_path, release)["passed"] is True
    with pytest.raises(
        CommitmentError, match="RELEASE_COMMITMENT_ALREADY_EXISTS"
    ):
        write_release_commitment(tmp_path)

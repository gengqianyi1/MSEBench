from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import tools.file_tools as file_tools_module
from tools.file_tools import WorkspaceSandbox
from tools.tool_registry import ToolRegistry
from tools.xtb_runner import XtbAttemptManager


def _registry(tmp_path: Path) -> ToolRegistry:
    project = tmp_path / "project"
    case = project / "benchmark_case"
    configs = project / "configs"
    case.mkdir(parents=True)
    configs.mkdir()
    source = Path(__file__).resolve().parents[2]
    shutil.copyfile(source / "configs/task.yaml", configs / "task.yaml")
    shutil.copyfile(source / "configs/tool_contracts.json", configs / "tool_contracts.json")
    xyz = case / "input.xyz"
    xyz.write_text("1\natom\nH 0 0 0\n", encoding="utf-8")
    job = {
        "input_geometry": {"path": "benchmark_case/input.xyz"},
        "method": "GFN2-xTB",
        "charge": 0,
        "uhf": 0,
        "solvation": None,
        "required_phases": ["geometry_optimization", "hessian"],
    }
    (case / "job_spec.json").write_text(json.dumps(job), encoding="utf-8")
    manifest = {
        "artifacts": [
            {
                "path": "input.xyz",
                "sha256": hashlib.sha256(xyz.read_bytes()).hexdigest(),
                "current": True,
                "restart_allowed": True,
            }
        ]
    }
    (case / "case_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (case / ".env").write_text("DO_NOT_READ=this-value\n", encoding="utf-8")
    return ToolRegistry(project)


def test_registry_rejects_path_escape_and_env_reads(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for index, path in enumerate(("../configs/task.yaml", "C:\\secret.txt", ".env")):
        result = registry.dispatch(
            "read_text_range",
            {"path": path, "start_line": 1, "end_line": 1},
            call_id=f"read_{index}",
        )
        assert result["ok"] is False
    listed = registry.dispatch("list_case_files", {}, call_id="list")
    assert listed["ok"] is True
    assert ".env" not in {entry["path"] for entry in listed["result"]["files"]}


def test_registry_rejects_unknown_extra_duplicate_and_attempt_injection(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    unknown = registry.dispatch("shell", {}, call_id="unknown")
    assert unknown["ok"] is False
    assert unknown["error"]["code"] == "UNKNOWN_TOOL"

    extra = registry.dispatch("list_case_files", {"command": "whoami"}, call_id="extra")
    assert extra["ok"] is False
    assert extra["error"]["code"] == "SCHEMA_VALIDATION_FAILED"

    first = registry.dispatch("list_case_files", {}, call_id="duplicate")
    second = registry.dispatch("list_case_files", {}, call_id="duplicate")
    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"]["code"] == "DUPLICATE_CALL_ID"

    injection = registry.dispatch(
        "run_xtb", {"attempt_id": "attempt_001 & whoami"}, call_id="inject"
    )
    assert injection["ok"] is False
    assert injection["error"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert registry.trusted_state["tool_call_count"] == 5


def test_initial_and_noncanonical_attempt_ids_cannot_be_executed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    for index, attempt_id in enumerate(
        ("attempt_000_initial_failure", "attempt_001/../attempt_000", "attempt_01")
    ):
        result = registry.dispatch(
            "run_xtb", {"attempt_id": attempt_id}, call_id=f"immutable_{index}"
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "SCHEMA_VALIDATION_FAILED"


def test_malformed_tool_contract_fails_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    case = project / "benchmark_case"
    configs = project / "configs"
    case.mkdir(parents=True)
    configs.mkdir()
    source = Path(__file__).resolve().parents[2]
    shutil.copyfile(source / "configs/task.yaml", configs / "task.yaml")
    (configs / "tool_contracts.json").write_text(
        '{"schema_version":"1.0","tools":[{"type":"function","function":{"name":7}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tool contract"):
        ToolRegistry(project)


def test_reparse_detection_is_fail_closed_without_os_link_privileges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "case"
    nested = root / "junction_like"
    nested.mkdir(parents=True)
    (nested / "outside.txt").write_text("must not be traversed", encoding="utf-8")
    sandbox = WorkspaceSandbox(root)
    real_check = file_tools_module._is_reparse_or_symlink

    def simulated_reparse(path: Path) -> bool:
        return path.name == "junction_like" or real_check(path)

    monkeypatch.setattr(file_tools_module, "_is_reparse_or_symlink", simulated_reparse)
    with pytest.raises(Exception) as caught:
        sandbox.resolve("junction_like/outside.txt")
    assert getattr(caught.value, "code", None) == "LINK_ESCAPE_REJECTED"
    listing = sandbox.list_case_files()
    assert listing["files"] == []


def test_native_environment_is_an_allowlist_and_excludes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GJLD_API_KEY", "must-never-reach-native-process")
    monkeypatch.setenv("PYTHONPATH", "C:\\untrusted\\python")
    monkeypatch.setenv("UNRELATED_SECRET", "must-never-reach-native-process")
    environment = XtbAttemptManager._safe_environment(1)
    assert "GJLD_API_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert "UNRELATED_SECRET" not in environment
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"

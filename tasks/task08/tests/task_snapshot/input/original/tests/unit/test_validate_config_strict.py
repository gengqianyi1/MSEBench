from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import yaml

from scripts.validate_config import _json, _yaml, validate_configuration


def test_yaml_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: '1.0'\nschema_version: '2.0'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        _yaml(path)


def test_json_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"1.0","schema_version":"2.0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _json(path)


def test_gate_reports_duplicate_task_yaml_as_config_conflict(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "task.yaml").write_text(
        "schema_version: '1.0'\nschema_version: '2.0'\n",
        encoding="utf-8",
    )
    result = validate_configuration(tmp_path)
    assert result["passed"] is False
    assert result["status"] == "BLOCKED_CONFIG_CONFLICT"
    assert result["errors"] == ["TASK_CONFIG_INVALID"]


def test_missing_public_budget_field_is_a_config_conflict(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2]
    shutil.copytree(source / "configs", tmp_path / "configs")
    shutil.copytree(source / "benchmark_case", tmp_path / "benchmark_case")
    task_path = tmp_path / "configs" / "task.yaml"
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    del task["budget"]["max_text_read_lines_per_call"]
    task_path.write_text(
        yaml.safe_dump(task, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    result = validate_configuration(tmp_path)
    assert result["passed"] is False
    assert "BUDGET_SCHEMA_MISMATCH" in result["errors"]


def test_non_mapping_section_fails_closed_without_validator_crash(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "task.yaml").write_text(
        "schema_version: '1.0'\ntask: []\nbudget: {}\nruntime: {}\nmodels: {}\n"
        "tools: {}\nevaluator: {}\nscoring: {}\nsecurity: {}\ndeterminism: {}\n"
        "packaging: {}\n",
        encoding="utf-8",
    )
    result = validate_configuration(tmp_path)
    assert result == {
        "schema_version": "1.0",
        "passed": False,
        "status": "BLOCKED_CONFIG_CONFLICT",
        "errors": ["TASK_SCHEMA_MISMATCH"],
    }

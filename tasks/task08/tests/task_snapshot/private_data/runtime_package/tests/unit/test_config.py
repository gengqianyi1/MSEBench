from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest
import yaml

from framework import ConfigurationError, TaskConfig, ToolContractSet
from scripts.validate_config import validate_configuration


ROOT = Path(__file__).resolve().parents[2]


def test_configuration_gate_passes() -> None:
    report = validate_configuration(ROOT)
    assert report["passed"] is True
    assert report["status"] == "PASS"
    assert report["errors"] == []


def test_scoring_weights_total_exactly_100() -> None:
    config = yaml.safe_load((ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    assert config["scoring"]["total_points"] == 100
    assert sum(config["scoring"]["categories"].values()) == 100


def test_job_spec_is_scientific_authority() -> None:
    config_text = (ROOT / "configs" / "task.yaml").read_text(encoding="utf-8")
    job = json.loads((ROOT / "benchmark_case" / "job_spec.json").read_text(encoding="utf-8"))
    assert job["method"] == "GFN2-xTB"
    assert job["molecule"]["charge"] == 0
    assert job["molecule"]["uhf"] == 0
    assert job["solvation"] is None
    assert job["required_phases"] == ["geometry_optimization", "hessian"]
    assert "GFN2-xTB" not in config_text


def test_model_registry_has_three_exact_nonfallback_models() -> None:
    models = yaml.safe_load((ROOT / "configs" / "models.example.yaml").read_text(encoding="utf-8"))
    assert {name: item["model"] for name, item in models["models"].items()} == {
        "kimi": "Pro/moonshotai/Kimi-K2.6",
        "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
        "glm": "zai-org/GLM-5.2",
    }
    assert models["provider_defaults"]["sdk_max_retries"] == 0
    assert models["provider_defaults"]["api_key_env"] == "GJLD_API_KEY"


def test_budget_schema_is_closed_and_lossless() -> None:
    raw = yaml.safe_load((ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    parsed = TaskConfig.from_mapping(raw)
    assert parsed.budget.to_dict() == raw["budget"]

    unknown = copy.deepcopy(raw)
    unknown["budget"]["unpublished_limit"] = 1
    with pytest.raises(ConfigurationError, match="published fields"):
        TaskConfig.from_mapping(unknown)

    malformed = copy.deepcopy(raw)
    malformed["budget"]["require_read_text_evidence"] = 1
    with pytest.raises(ConfigurationError, match="must be a boolean"):
        TaskConfig.from_mapping(malformed)


def test_task_yaml_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "task.yaml").read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(source + "\nbudget:\n  max_tool_calls: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate keys"):
        TaskConfig.load(duplicate)


def test_final_answer_artifact_contract_is_closed_and_not_model_facing() -> None:
    contracts = ToolContractSet.load(ROOT / "configs" / "tool_contracts.json")
    artifact = contracts.final_answer_artifact_schema
    assert artifact["additionalProperties"] is False
    assert artifact["properties"]["trusted_runtime"]["additionalProperties"] is False
    submit = contracts.parameters_by_name["submit_final"]
    assert submit["additionalProperties"] is False
    assert "trusted_runtime" not in submit["properties"]
    assert "model_submission" not in submit["properties"]


def test_prepare_restart_contract_exposes_phase_specific_cycle_rule() -> None:
    contracts = json.loads(
        (ROOT / "configs" / "tool_contracts.json").read_text(encoding="utf-8")
    )
    prepare = next(
        item["function"] for item in contracts["tools"]
        if item["function"]["name"] == "prepare_restart"
    )
    assert "hessian requires optimization_cycle_limit null exactly" in prepare["description"]
    cycle = prepare["parameters"]["properties"]["optimization_cycle_limit"]
    assert "must be null for hessian" in cycle["description"]

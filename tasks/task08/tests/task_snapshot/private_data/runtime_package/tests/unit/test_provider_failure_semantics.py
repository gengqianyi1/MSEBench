"""Regression coverage for transient provider failure handling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evaluator.evaluate_suite import evaluate_suite
from framework import ModelConfig, TaskConfig
from scripts.run_suite_models_v2 import _case_status


ROOT = Path(__file__).resolve().parents[2]


def test_terminal_provider_failure_is_blocked_not_scientific_zero() -> None:
    result = SimpleNamespace(
        complete=False,
        status="FAILED",
        metadata={
            "actual_model": "Pro/moonshotai/Kimi-K2.6",
            "case_id": "case",
            "failure": {"status": "RUN_FAILED", "type": "ProviderError"},
            "counts": {},
            "duration_seconds": 1.0,
            "token_usage": {},
        },
    )
    status = _case_status(
        logical_model="kimi",
        requested_model="Pro/moonshotai/Kimi-K2.6",
        result=result,
        evaluation={
            "eligibility": {"eligible": False},
            "raw_score_before_scientific_cap": 5,
            "official_score": 0,
        },
        replay_match=True,
        scan={
            "passed": True,
            "secret_finding_count": 0,
            "runtime_absolute_path_finding_count": 0,
        },
    )
    assert status["run_status"] == "BLOCKED"
    assert status["provider_status"] == "UNAVAILABLE_AFTER_RETRIES"
    assert status["raw_score"] is None
    assert status["official_score"] is None


def test_suite_excludes_provider_blocked_case_from_numeric_mean(tmp_path: Path) -> None:
    suite = {
        "public_case": {
            "case_id": "case",
            "config": "configs/case.yaml",
            "workspace": "case",
        },
        "hidden_cases": [],
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    case_root = tmp_path / "runs" / "case"
    case_root.mkdir(parents=True)
    (case_root / "status.json").write_text(
        json.dumps({"run_status": "BLOCKED"}), encoding="utf-8"
    )
    aggregate = evaluate_suite(suite_path, tmp_path / "runs")
    assert aggregate["coverage"] == {
        "pass": 0,
        "fail": 0,
        "blocked": 1,
        "scored": 0,
    }
    assert aggregate["macro_official_score_excluding_blocked"] is None
    assert aggregate["ranking_eligible"] is False


def test_provider_retry_budget_is_bounded_and_refreshed() -> None:
    task = TaskConfig.load(ROOT / "configs" / "task.yaml")
    suite = json.loads(
        (ROOT / "organizer_private" / "suite.json").read_text(encoding="utf-8")
    )
    case_configs = [suite["public_case"]["config"]] + [
        item["config"] for item in suite["hidden_cases"]
    ]
    case_tasks = [TaskConfig.load(ROOT / relative) for relative in case_configs]
    models = [
        ModelConfig.load_registry(ROOT / "configs" / "models.example.yaml", label)
        for label in ("deepseek", "kimi", "glm")
    ]
    assert task.budget.max_provider_retries == 5
    assert all(case.budget.max_provider_retries == 5 for case in case_tasks)
    assert all(model.max_retries == 5 for model in models)

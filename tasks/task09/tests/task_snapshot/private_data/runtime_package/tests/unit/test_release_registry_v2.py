from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from framework.schemas import (
    BENCHMARK_VERSION,
    FORMAL_CAMPAIGN_ID,
    FORMAL_MODELS,
    FORMAL_VALIDATION_REPORTS,
    PROTOCOL,
    TASK_ID,
)
from package_tools.policy import PACKAGE_GATE_CHECKS
from scripts.validate_project import _release_registry_matches


def _formal() -> dict[str, dict]:
    return {
        "kimi": {
            "actual_model_identity": FORMAL_MODELS["kimi"],
            "eligible": True,
            "exact_identity": True,
            "legal_failure_envelope": False,
            "run_status": "PASS",
            "total_score": 89.0,
        },
        "deepseek": {
            "actual_model_identity": FORMAL_MODELS["deepseek"],
            "eligible": True,
            "exact_identity": True,
            "legal_failure_envelope": False,
            "run_status": "PASS",
            "total_score": 88.0,
        },
        "glm": {
            "actual_model_identity": None,
            "eligible": False,
            "exact_identity": False,
            "failure_reason": "NETWORK_CONNECTION_FAILED",
            "legal_failure_envelope": True,
            "run_status": "BLOCKED",
            "total_score": 0.0,
        },
    }


def _release(tmp_path: Path) -> dict:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "glm-blocker.json").write_text("{}\n", encoding="utf-8")
    for path in FORMAL_VALIDATION_REPORTS.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    models = {}
    for label, result in _formal().items():
        item = {
            "actual_model": result["actual_model_identity"],
            "eligible": result["eligible"],
            "exact_identity": result["exact_identity"],
            "formal_trajectory_started": True,
            "legal_failure_envelope": result["legal_failure_envelope"],
            "requested_model": FORMAL_MODELS[label],
            "score": result["total_score"],
            "status": result["run_status"],
            "terminal": True,
            "validation_report": FORMAL_VALIDATION_REPORTS[label],
        }
        if label == "glm":
            item["model_blocker_report"] = "reports/glm-blocker.json"
            item["safe_status"] = "NETWORK_CONNECTION_FAILED"
        models[label] = item
    return {
        "benchmark_framework_status": "PASS",
        "benchmark_version": BENCHMARK_VERSION,
        "blocker_report": None,
        "final_zip_created": True,
        "formal_campaign_execution_status": "PASS",
        "formal_campaign_id": FORMAL_CAMPAIGN_ID,
        "gates": {
            "package": {
                "checks": {name: True for name in PACKAGE_GATE_CHECKS},
                "passed": True,
                "status": "PASS",
            }
        },
        "models": models,
        "package_submission_invoked": True,
        "protocol": PROTOCOL,
        "release_ready": True,
        "schema_version": "2.0",
        "status": "PASS",
        "task_id": TASK_ID,
    }


def test_release_registry_accepts_individual_legal_blocked_model(
    tmp_path: Path,
) -> None:
    assert _release_registry_matches(_release(tmp_path), _formal(), tmp_path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("formal_campaign_execution_status",), "BLOCKED"),
        (("models", "glm", "status"), "PASS"),
        (("models", "glm", "legal_failure_envelope"), False),
        (("models", "glm", "actual_model"), FORMAL_MODELS["glm"]),
        (("models", "kimi", "score"), 0.0),
        (("gates", "package", "passed"), False),
    ],
)
def test_release_registry_rejects_claims_that_disagree_with_replay(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    release = _release(tmp_path)
    cursor = release
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    assert not _release_registry_matches(release, _formal(), tmp_path)

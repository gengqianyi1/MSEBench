"""Single, validated source of release identity for ChemBench 04--06."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_IDS = ("task_04", "task_05", "task_06")
TASK_BUDGET_FIELDS = {
    "max_xtb_jobs",
    "max_scan_points",
    "max_agent_turns",
    "finalization_turns",
    "max_tool_calls",
    "wall_time_seconds",
    "finalization_wall_time_seconds",
}
TASK04_COMPUTE_BUDGET_FIELDS = {
    "main_credits",
    "recovery_credits",
    "max_jobs",
    "max_recovery_chains",
    "max_recovery_credits_per_chain",
    "costs",
    "scan_allowed",
}
TASK04_COMPUTE_COST_FIELDS = {
    "other",
    "single_point",
    "optimization",
    "hessian",
    "optimization_hessian",
}


@dataclass(frozen=True)
class ReleaseContract:
    document: dict[str, Any]

    @property
    def grader_version(self) -> str:
        return str(self.document["grader_version"])

    @property
    def package_release(self) -> str:
        return str(self.document["package_release"])

    @property
    def task_status(self) -> str:
        return str(self.document["task_status"])

    @property
    def task_ids(self) -> tuple[str, ...]:
        return TASK_IDS

    @property
    def public_tools(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.goal_driven_contract()["public_tools"])

    @property
    def finalization_notice(self) -> str:
        return str(self.goal_driven_contract()["finalization_notice"])

    @property
    def flash_validation_order(self) -> tuple[str, ...]:
        return tuple(str(task_id) for task_id in self.goal_driven_contract()["flash_validation_order"])

    def task_version(self, task_id: str) -> str:
        try:
            return str(self.document["task_versions"][task_id])
        except KeyError as exc:
            raise ValueError(f"task is absent from release contract: {task_id}") from exc

    def task_grader_version(self, task_id: str) -> str:
        try:
            return str(self.document["task_grader_versions"][task_id])
        except KeyError as exc:
            raise ValueError(f"task is absent from release contract: {task_id}") from exc

    def task_package_release(self, task_id: str) -> str:
        try:
            return str(self.document["task_package_releases"][task_id])
        except KeyError as exc:
            raise ValueError(f"task is absent from release contract: {task_id}") from exc

    def task_budget(self, task_id: str) -> dict[str, int]:
        try:
            budget = self.document["task_budgets"][task_id]
        except KeyError as exc:
            raise ValueError(f"task is absent from release contract: {task_id}") from exc
        return {str(key): int(value) for key, value in budget.items()}

    def task_compute_budget(self, task_id: str) -> dict[str, Any]:
        try:
            budget = self.document["task_compute_budgets"][task_id]
        except KeyError as exc:
            raise ValueError(f"task is absent from release contract: {task_id}") from exc
        return {
            "main_credits": int(budget["main_credits"]),
            "recovery_credits": int(budget["recovery_credits"]),
            "max_jobs": int(budget["max_jobs"]),
            "max_recovery_chains": int(budget["max_recovery_chains"]),
            "max_recovery_credits_per_chain": int(budget["max_recovery_credits_per_chain"]),
            "costs": {str(key): int(value) for key, value in budget["costs"].items()},
            "scan_allowed": bool(budget["scan_allowed"]),
        }

    def goal_driven_contract(self) -> dict[str, Any]:
        return dict(self.document["goal_driven_contract"])


def load_release_contract(bundle_root: Path) -> ReleaseContract:
    path = Path(bundle_root) / "release_contract.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "task_versions", "grader_version", "package_release",
        "task_grader_versions", "task_package_releases",
        "task_status", "formal_models", "formal_task_count", "formal_run_count",
        "golden_repetitions_per_task", "golden_run_count", "hash_fields",
        "task_budgets", "task_compute_budgets", "goal_driven_contract",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"release contract missing fields: {missing}")
    if set(document["task_versions"]) != set(TASK_IDS):
        raise ValueError("release contract must define exactly task_04, task_05 and task_06")
    if set(document["task_grader_versions"]) != set(TASK_IDS):
        raise ValueError("release contract grader versions must define exactly task_04, task_05 and task_06")
    if set(document["task_package_releases"]) != set(TASK_IDS):
        raise ValueError("release contract package releases must define exactly task_04, task_05 and task_06")
    if document["grader_version"] != document["task_grader_versions"]["task_04"]:
        raise ValueError("root grader version must match Task 04")
    if document["package_release"] != document["task_package_releases"]["task_04"]:
        raise ValueError("root package release must match Task 04")
    if set(document["task_budgets"]) != set(TASK_IDS):
        raise ValueError("release contract budgets must define exactly task_04, task_05 and task_06")
    for task_id, budget in document["task_budgets"].items():
        if set(budget) != TASK_BUDGET_FIELDS:
            raise ValueError(f"release contract budget fields are invalid for {task_id}")
        if any(not isinstance(value, int) or value < 0 for value in budget.values()):
            raise ValueError(f"release contract budget values are invalid for {task_id}")
    compute_budgets = document["task_compute_budgets"]
    if set(compute_budgets) != {"task_04"}:
        raise ValueError("release contract compute budgets must define exactly task_04")
    compute_budget = compute_budgets["task_04"]
    if set(compute_budget) != TASK04_COMPUTE_BUDGET_FIELDS:
        raise ValueError("release contract compute budget fields are invalid for task_04")
    if set(compute_budget["costs"]) != TASK04_COMPUTE_COST_FIELDS:
        raise ValueError("release contract compute cost fields are invalid for task_04")
    numeric_fields = TASK04_COMPUTE_BUDGET_FIELDS - {"costs", "scan_allowed"}
    if any(not isinstance(compute_budget[field], int) or compute_budget[field] < 0 for field in numeric_fields):
        raise ValueError("release contract compute budget values are invalid for task_04")
    if any(not isinstance(value, int) or value <= 0 for value in compute_budget["costs"].values()):
        raise ValueError("release contract compute cost values are invalid for task_04")
    if compute_budget["scan_allowed"] is not False:
        raise ValueError("Task 04 scans must be disabled")
    if compute_budget["max_jobs"] != document["task_budgets"]["task_04"]["max_xtb_jobs"]:
        raise ValueError("Task 04 compute max_jobs must equal max_xtb_jobs")
    goal = document["goal_driven_contract"]
    if goal.get("public_tools") != ["read_file", "write_file", "run_python", "xtb_run", "xtb_parse"]:
        raise ValueError("release contract must expose exactly the five generic tools")
    if goal.get("hints") != {"enabled": False, "requestable": False}:
        raise ValueError("release contract must disable hints")
    if goal.get("budget_exhaustion_is_zero") is not False:
        raise ValueError("budget exhaustion must permit partial scientific scoring")
    if goal.get("flash_validation_order") != list(TASK_IDS):
        raise ValueError("Flash validation order must be task_04, task_05, task_06")
    return ReleaseContract(document)


def validate_task_metadata(contract: ReleaseContract, *, task_id: str, version: str, status: str) -> None:
    expected_version = contract.task_version(task_id)
    if version != expected_version:
        raise ValueError(f"task version drift: {task_id} has {version!r}, expected {expected_version!r}")
    if status != contract.task_status:
        raise ValueError(f"task status drift: {task_id} has {status!r}, expected {contract.task_status!r}")

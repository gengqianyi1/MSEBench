"""Configuration models and validation for the ChemBench harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .schema_validation import validate_document


ALLOWED_SCORING_TYPES = {"hard_rules", "custom", "hybrid"}


def _safe_relative_path(value: str, field_name: str) -> str:
    """Reject absolute paths and parent traversal in task-controlled paths."""
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a safe relative path: {value!r}")
    return str(path)


@dataclass
class TaskConfig:
    """Single versioned task configuration loaded from ``task.yaml``."""

    task_id: str
    title: str
    domain: str
    author: str
    version: str

    max_iterations: int = 3
    max_tool_calls: int = 20
    timeout_per_iteration: int = 600
    total_timeout_seconds: int = 3600
    python_path: str = "python"
    final_answer_required: bool = True
    stop_on_status_complete: bool = True

    required_tools: list[str] = field(default_factory=list)
    tools_registry_path: str = "tools.json"

    required_artifacts: list[str] = field(default_factory=list)
    optional_artifacts: list[str] = field(default_factory=list)
    final_answer_file: str = "final_answer.json"

    rubric_path: str = "rubric.json"
    scoring_type: str = "hard_rules"
    grader_path: str | None = None
    grader_entrypoint: str = "grade"
    grader_version: str = "unknown"
    scientific_evaluator_version: str = "unknown"
    package_release: str = "unknown"
    score_semantics: dict[str, str] = field(default_factory=dict)
    private_data_path: str | None = None
    private_required_files: list[str] = field(default_factory=list)
    submission_root: str = "submission"
    scientific_manifest_path: str = "submission/run_manifest.json"
    max_score: float = 100.0
    pass_score: float = 50.0

    env_overrides: dict[str, str] = field(default_factory=dict)
    network_allowed: bool = False
    environment_file: str | None = None
    execution_mode: str = "agent"
    runner_script: str | None = None
    runner_timeout_seconds: int | None = None
    runtime_checks: dict[str, Any] = field(default_factory=dict)
    budget_limits: dict[str, int | float] = field(default_factory=dict)
    manifest_schemas: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "TaskConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        required = ("task_id", "title")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"Missing required task.yaml fields: {', '.join(missing)}")
        if not data.get("runtime_checks"):
            raise ValueError("Missing required task.yaml field: runtime_checks")

        schema_dir = Path(__file__).resolve().parent.parent / "schemas"
        validate_document(data, schema_dir / "task.schema.json")

        cfg = cls(
            task_id=str(data["task_id"]),
            title=str(data["title"]),
            domain=str(data.get("domain", "general")),
            author=str(data.get("author", "unknown")),
            version=str(data.get("version", "1.0.0")),
            max_iterations=int(data.get("max_iterations", 3)),
            max_tool_calls=int(data.get("max_tool_calls", 20)),
            timeout_per_iteration=int(data.get("timeout_per_iteration", 600)),
            total_timeout_seconds=int(data.get("total_timeout_seconds", 3600)),
            python_path=str(data.get("python_path", "python")),
            final_answer_required=bool(data.get("final_answer_required", True)),
            stop_on_status_complete=bool(data.get("stop_on_status_complete", True)),
            required_tools=list(data.get("required_tools", [])),
            tools_registry_path=_safe_relative_path(data.get("tools_registry_path", "tools.json"), "tools_registry_path"),
            required_artifacts=[_safe_relative_path(item, "required_artifacts") for item in data.get("required_artifacts", [])],
            optional_artifacts=[_safe_relative_path(item, "optional_artifacts") for item in data.get("optional_artifacts", [])],
            final_answer_file=_safe_relative_path(data.get("final_answer_file", "final_answer.json"), "final_answer_file"),
            rubric_path=_safe_relative_path(data.get("rubric_path", "rubric.json"), "rubric_path"),
            scoring_type=str(data.get("scoring_type", "hard_rules")),
            grader_path=_safe_relative_path(data["grader_path"], "grader_path") if data.get("grader_path") else None,
            grader_entrypoint=str(data.get("grader_entrypoint", "grade")),
            grader_version=str(data.get("grader_version", "unknown")),
            scientific_evaluator_version=str(data.get("scientific_evaluator_version", "unknown")),
            package_release=str(data.get("package_release", "unknown")),
            score_semantics={str(k): str(v) for k, v in data.get("score_semantics", {}).items()},
            private_data_path=_safe_relative_path(data["private_data_path"], "private_data_path") if data.get("private_data_path") else None,
            private_required_files=[_safe_relative_path(item, "private_required_files") for item in data.get("private_required_files", [])],
            submission_root=_safe_relative_path(data.get("submission_root", "submission"), "submission_root"),
            scientific_manifest_path=_safe_relative_path(data.get("scientific_manifest_path", "submission/run_manifest.json"), "scientific_manifest_path"),
            max_score=float(data.get("max_score", 100.0)),
            pass_score=float(data.get("pass_score", data.get("max_score", 100.0) * 0.5)),
            env_overrides={str(k): str(v) for k, v in data.get("env_overrides", {}).items()},
            network_allowed=bool(data.get("network_allowed", False)),
            environment_file=_safe_relative_path(data["environment_file"], "environment_file") if data.get("environment_file") else None,
            execution_mode=str(data.get("execution_mode", "agent")),
            runner_script=_safe_relative_path(data["runner_script"], "runner_script") if data.get("runner_script") else None,
            runner_timeout_seconds=int(data["runner_timeout_seconds"]) if data.get("runner_timeout_seconds") else None,
            runtime_checks=dict(data.get("runtime_checks", {})),
            budget_limits={str(k): v for k, v in data.get("budget_limits", {}).items()},
            manifest_schemas={
                str(k): _safe_relative_path(v, f"manifest_schemas.{k}")
                for k, v in data.get("manifest_schemas", {}).items()
            },
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.scoring_type not in ALLOWED_SCORING_TYPES:
            raise ValueError(f"Unsupported scoring_type: {self.scoring_type}")
        if self.scoring_type in {"custom", "hybrid"} and not self.grader_path:
            raise ValueError(f"scoring_type={self.scoring_type} requires grader_path")
        if self.max_iterations < 1 or self.max_tool_calls < 0:
            raise ValueError("max_iterations must be >= 1 and max_tool_calls must be >= 0")
        if self.timeout_per_iteration < 1 or self.total_timeout_seconds < 1:
            raise ValueError("timeouts must be positive")
        if self.max_score <= 0 or not 0 <= self.pass_score <= self.max_score:
            raise ValueError("max_score/pass_score are inconsistent")
        if self.execution_mode not in {"agent", "native_runner"}:
            raise ValueError(f"Unsupported execution_mode: {self.execution_mode}")
        if self.execution_mode == "native_runner" and not self.runner_script:
            raise ValueError("execution_mode=native_runner requires runner_script")
        if not self.runtime_checks:
            raise ValueError("runtime_checks must be declared")
        required_budget_limits = {
            "agent_turns",
            "finalization_turns",
            "api_attempts",
            "tool_calls",
            "xtb_jobs",
            "xtb_scan_points",
            "wall_seconds",
            "finalization_wall_seconds",
            "requested_tokens_per_turn",
        }
        if set(self.budget_limits) != required_budget_limits:
            raise ValueError("budget_limits must declare all cumulative budget dimensions")
        required_manifest_schemas = {
            "harness",
            "scientific_submission",
            "budget_ledger",
            "runtime_checks",
            "api_event",
            "score",
        }
        if set(self.manifest_schemas) != required_manifest_schemas:
            raise ValueError("manifest_schemas must declare manifests, runtime evidence, and score schemas")
        if not self.package_release or self.package_release == "unknown":
            raise ValueError("package_release must be declared")
        if not self.scientific_evaluator_version or self.scientific_evaluator_version == "unknown":
            raise ValueError("scientific_evaluator_version must be declared")
        expected_score_semantics = {"task_local_score", "cross_task_comparison_score", "aggregation_field"}
        if set(self.score_semantics) != expected_score_semantics:
            raise ValueError("score_semantics must define task_local_score, cross_task_comparison_score, and aggregation_field")
        if self.score_semantics.get("aggregation_field") != "comparison_score":
            raise ValueError("cross-task aggregation must use comparison_score")
        if self.final_answer_required and self.final_answer_file not in self.required_artifacts:
            raise ValueError("final_answer_file must be listed in required_artifacts")
        overlap = set(self.required_artifacts) & set(self.optional_artifacts)
        if overlap:
            raise ValueError(f"required_artifacts and optional_artifacts overlap: {sorted(overlap)}")
        if self.scientific_manifest_path != f"{self.submission_root}/run_manifest.json":
            raise ValueError("scientific_manifest_path must point to <submission_root>/run_manifest.json")


@dataclass
class GlobalConfig:
    """Global runner configuration. Secrets are resolved from environment variables."""

    llm_api: dict[str, Any]
    execution: dict[str, Any]
    software: dict[str, Any]
    safety: dict[str, Any]
    evaluation: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "GlobalConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        llm_api = dict(data.get("llm_api", {}))
        native_selector = llm_api.get("model_config_file")
        if native_selector:
            selector_path = Path(str(native_selector))
            if not selector_path.is_absolute():
                selector_path = Path(path).resolve().parent / selector_path
            llm_api["model_config_file"] = str(selector_path.resolve())
        env_name = llm_api.get("api_key_env")
        if env_name:
            llm_api["api_key"] = os.environ.get(str(env_name), "")
        if not native_selector and not llm_api.get("api_key"):
            raise ValueError(
                "LLM API key is missing. Set the environment variable named by "
                "llm_api.api_key_env (recommended) or provide llm_api.api_key."
            )
        if not native_selector and not llm_api.get("url"):
            raise ValueError("llm_api.url is required")
        return cls(
            llm_api=llm_api,
            execution=dict(data.get("execution", {})),
            software=dict(data.get("software", {})),
            safety=dict(data.get("safety", {})),
            evaluation=dict(data.get("evaluation", {})),
        )

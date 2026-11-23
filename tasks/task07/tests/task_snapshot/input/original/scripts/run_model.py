#!/usr/bin/env python3
"""Run one configured OpenAI-compatible model through the benchmark agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv


# ``python scripts/run_model.py`` places scripts/, not the repository root, at
# sys.path[0].  Add the root before importing the local framework package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework import AgentRunner, ConfigurationError  # noqa: E402
from framework.schemas import (  # noqa: E402
    SILICONFLOW_API_KEY_ENV,
    TRUSTED_PROVIDER_BASE_URLS,
)


SILICONFLOW_LOGICAL_MODELS = {
    "glm": "zai-org/GLM-5.2",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "kimi": "Pro/moonshotai/Kimi-K2.6",
}


def _load_runtime_environment() -> None:
    """Load only the nearest ancestor .env and never expose credential values."""

    explicit = os.environ.get("ANT_BENCHMARK_ENV_FILE", "").strip()
    if explicit:
        env_path = Path(explicit).expanduser().resolve()
    else:
        env_path = next(
            (
                candidate / ".env"
                for candidate in (PROJECT_ROOT, *PROJECT_ROOT.parents)
                if (candidate / ".env").is_file()
            ),
            PROJECT_ROOT.parent / ".env",
        )
    load_dotenv(dotenv_path=env_path, override=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError("PyYAML is required to read model configuration") from exc
    if not path.is_file():
        raise ConfigurationError(
            f"configuration file not found: {path}. Copy configs/models.example.yaml "
            "to configs/models.yaml and configure environment-variable names only."
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration {path} must contain an object")
    return value


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _model_config(path: Path, logical_name: str) -> dict[str, Any]:
    document = _load_yaml(path)
    models = document.get("models")
    if not isinstance(models, dict) or logical_name not in models:
        available = ", ".join(sorted(models)) if isinstance(models, dict) else "none"
        raise ConfigurationError(
            f"model {logical_name!r} is not configured in {path.name}; available: {available}"
        )
    raw = models[logical_name]
    if not isinstance(raw, dict):
        raise ConfigurationError(f"model entry {logical_name!r} must be an object")
    if raw.get("enabled") is False:
        raise ConfigurationError(f"model entry {logical_name!r} is disabled")
    selected = dict(raw)
    raw_provider = selected.get("provider")
    if isinstance(raw_provider, str):
        # Normalize before selecting provider-specific policy.  Otherwise a
        # spelling such as `` SiliconFlow `` could bypass the exact logical-ID,
        # endpoint, and credential checks and be normalized only later.
        selected["provider"] = raw_provider.strip().lower()
    if selected.get("provider") == "siliconflow":
        expected_model = SILICONFLOW_LOGICAL_MODELS.get(logical_name)
        if expected_model is None:
            raise ConfigurationError(
                "siliconflow logical model must be exactly one of: "
                + ", ".join(sorted(SILICONFLOW_LOGICAL_MODELS))
            )
        if selected.get("model") != expected_model:
            raise ConfigurationError(
                f"siliconflow model {logical_name!r} must be exactly {expected_model!r}"
            )
        if selected.get("model_env"):
            raise ConfigurationError(
                "siliconflow exact model IDs cannot be overridden through model_env"
            )
        if selected.get("base_url") != TRUSTED_PROVIDER_BASE_URLS["siliconflow"]:
            raise ConfigurationError("siliconflow must use the official base_url")
        if selected.get("api_key_env") != SILICONFLOW_API_KEY_ENV:
            raise ConfigurationError(
                f"siliconflow api_key_env must be exactly {SILICONFLOW_API_KEY_ENV!r}"
            )
    model_env = selected.get("model_env")
    if isinstance(model_env, str) and model_env:
        override = os.environ.get(model_env)
        if override:
            selected["model"] = override
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real model with the bounded RDKit tool-calling agent."
    )
    parser.add_argument("--model", required=True, help="Logical key under models: in the YAML file.")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--task-config", default="configs/task.yaml")
    parser.add_argument("--prompt", default="prompts/agent_prompt.md")
    parser.add_argument(
        "--output",
        help="Exact run directory; takes precedence over --output-root.",
    )
    parser.add_argument(
        "--output-root",
        default="trajectories",
        help="Parent directory used as <root>/<model> when --output is omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _load_runtime_environment()
        models_path = _project_path(args.models_config)
        task_path = _project_path(args.task_config)
        prompt_path = _project_path(args.prompt)
        output_path = (
            _project_path(args.output)
            if args.output
            else _project_path(args.output_root) / args.model
        )
        selected_model = _model_config(models_path, args.model)
        runner = AgentRunner(
            task_config=task_path,
            model_config=selected_model,
            output_dir=output_path,
            prompt=prompt_path,
            run_label=args.model,
            model_config_source=models_path,
        )
        result = runner.run()
    except Exception as exc:
        print(f"run setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary = {
        "status": result.status,
        "logical_model": args.model,
        "requested_model": result.metadata.get("requested_model"),
        "model": result.metadata.get("model"),
        "output": str(output_path),
        "failure": result.metadata.get("failure"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

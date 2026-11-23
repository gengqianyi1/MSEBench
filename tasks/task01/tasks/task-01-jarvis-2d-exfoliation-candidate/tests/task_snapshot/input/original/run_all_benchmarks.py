#!/usr/bin/env python3
"""Run configured benchmark models sequentially without storing API credentials in source files."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BENCHMARK_DIR / "configs"
API_CONFIG_PATH = CONFIGS_DIR / "api_config.json"

MODELS = [
    {
        "name": "Vendor3/qwen3.5-plus",
        "api_key_env": "GPUGEEK_API_KEY",
        "base_url": "https://api.example.com/v1",
    },
    {
        "name": "Vendor2/kimi-k2.6",
        "api_key_env": "GPUGEEK_API_KEY",
        "base_url": "https://api.example.com/v1",
    },
    {
        "name": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.example.com/v1",
    },
]


def write_api_config(model: dict) -> None:
    api_key = os.environ.get(model["api_key_env"])
    if not api_key:
        raise RuntimeError(f"missing environment variable: {model['api_key_env']}")
    config = {
        "provider": "openai_compatible",
        "model": model["name"],
        "api_key": api_key,
        "base_url": model["base_url"],
        "temperature": 0.2,
        "max_tokens": 32000,
        "timeout": 300,
        "max_retries": 2,
        "retry_delay_seconds": 2,
    }
    API_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_benchmark(model_name: str) -> int:
    print(f"\n{'=' * 60}\nRunning benchmark for: {model_name}\n{'=' * 60}\n")
    process = subprocess.run(
        [sys.executable, "src/main.py", "--config", "configs/default.yaml"],
        cwd=BENCHMARK_DIR,
        timeout=7200,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    return int(process.returncode)


def main() -> None:
    for model in MODELS:
        write_api_config(model)
        return_code = run_benchmark(model["name"])
        print(f"{model['name']} finished with return code {return_code}")


if __name__ == "__main__":
    main()

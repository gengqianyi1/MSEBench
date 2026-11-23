"""Unified config: reads configs/llm_config.json, applies env overrides.

All tasks read LLM parameters from this single source; no per-task duplication.
"""
from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "llm_config.json"

ENV_MAP = {
    "api_base": ("MSEBENCH_API_BASE", "CHEMBENCH_API_BASE"),
    "model": ("MSEBENCH_MODEL_ID", "CHEMBENCH_MODEL_ID"),
    "temperature": ("MSEBENCH_TEMPERATURE",),
    "top_p": ("MSEBENCH_TOP_P", "CHEMBENCH_TOP_P"),
    "max_tokens": ("MSEBENCH_MAX_TOKENS",),
    "stream": ("MSEBENCH_STREAM",),
    "enable_thinking": ("MSEBENCH_ENABLE_THINKING",),
    "reasoning_effort": ("MSEBENCH_REASONING_EFFORT",),
    "request_timeout": ("MSEBENCH_REQUEST_TIMEOUT",),
    "request_retries": ("MSEBENCH_REQUEST_RETRIES",),
    "connection_close": ("MSEBENCH_CONNECTION_CLOSE",),
}

_INT_KEYS = {"max_tokens", "request_timeout", "request_retries"}
_FLOAT_KEYS = {"temperature", "top_p", "frequency_penalty", "presence_penalty"}
_BOOL_KEYS = {"stream", "enable_thinking", "connection_close"}


def _coerce(key, value):
    if key in _INT_KEYS:
        return int(value)
    if key in _FLOAT_KEYS:
        return float(value)
    if key in _BOOL_KEYS:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return value


def load_profile(profile=None, config_path=None):
    configured_path = config_path or os.getenv("MSEBENCH_CONFIG")
    if configured_path:
        path = Path(configured_path)
    elif CONFIG_PATH.is_file():
        path = CONFIG_PATH
    else:
        path = files("msebench").joinpath("default_profile.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    name = profile or os.getenv("MSEBENCH_LLM_PROFILE") or data.get("default_profile")
    if name not in data["profiles"]:
        raise SystemExit(f"unknown llm profile: {name}")
    cfg = dict(data["profiles"][name])
    cfg["profile"] = name
    for key, envs in ENV_MAP.items():
        for env in envs:
            value = os.getenv(env)
            if value is not None and value != "":
                cfg[key] = _coerce(key, value)
                break
    return cfg


def api_key(cfg):
    name = cfg.get("api_key_env", "MSEBENCH_API_KEY")
    key = os.getenv(name, "")
    if name == "MSEBENCH_API_KEY":
        return key or os.getenv("MATBENCH_API_KEY", "")
    return key


def _env_with_legacy(name, default=None):
    import os as _os
    return _os.getenv(name) or _os.getenv(name.replace('MSEBENCH_', 'MATBENCH_')) or default


def env_dict(cfg):
    """Map unified config to CHEMBENCH_*/MSEBENCH_* env for the native harnesses."""
    key = api_key(cfg)
    out = {
        "CHEMBENCH_API_BASE": cfg["api_base"],
        "CHEMBENCH_API_URL": cfg["api_base"].rstrip("/") + "/chat/completions",
        "CHEMBENCH_MODEL_ID": cfg["model"],
        "CHEMBENCH_API_FORMAT": str(cfg.get("api_format") or "openai"),
        "CHEMBENCH_API_KEY": key,
        "CHEMBENCH_STREAM": "true" if cfg.get("stream") else "false",
        "CHEMBENCH_MAX_TOKENS": str(cfg.get("max_tokens")),
        "CHEMBENCH_TEMPERATURE": str(cfg.get("temperature")),
        "CHEMBENCH_REQUEST_TIMEOUT": str(cfg.get("request_timeout", 1800)),
        "CHEMBENCH_REQUEST_TIMEOUT_SECONDS": str(cfg.get("request_timeout", 1800)),
        "CHEMBENCH_REQUEST_RETRIES": str(cfg.get("request_retries", 10)),
        "CHEMBENCH_CONNECTION_CLOSE": "True" if cfg.get("connection_close") else "False",
    }
    # Some providers reject unsupported options even when sent as "None".
    if cfg.get("enable_thinking") is not None:
        out["CHEMBENCH_ENABLE_THINKING"] = str(cfg.get("enable_thinking"))
    if cfg.get("reasoning_effort"):
        out["CHEMBENCH_REASONING_EFFORT"] = str(cfg.get("reasoning_effort"))
    if cfg.get("top_p") is not None:
        out["CHEMBENCH_TOP_P"] = str(cfg.get("top_p"))
    # 2026-09-23 (P1-3): thinking profiles get a relaxed turn ceiling; the
    # completion-token budget (agent-side, default 2M) gates the cost. The
    # task-declared budgets stay authoritative for non-thinking runs.
    if cfg.get("enable_thinking"):
        out["CHEMBENCH_MAX_AGENT_TURNS"] = os.getenv("MSEBENCH_MAX_AGENT_TURNS", "40")
    return out

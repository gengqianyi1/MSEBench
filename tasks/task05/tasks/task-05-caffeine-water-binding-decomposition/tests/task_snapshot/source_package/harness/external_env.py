"""Resolve an external credential file without exposing its path to model workspaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def resolve_external_env_path(
    explicit: Path | None, environment: Mapping[str, str]
) -> Path:
    """Prefer an explicit path, otherwise use WSLENV's translated path."""

    if explicit is not None:
        return Path(explicit)
    translated = str(environment.get("CHEMBENCH_ENV_FILE") or "").strip()
    if not translated:
        raise ValueError(
            "--env-file or the WSL-translated CHEMBENCH_ENV_FILE environment variable is required"
        )
    return Path(translated)


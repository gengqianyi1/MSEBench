"""Offline reference baselines, lazily imported for clean ``python -m`` use."""

from __future__ import annotations

from typing import Any

__all__ = ["run_baseline", "write_final_answer"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import baseline_agent

        return getattr(baseline_agent, name)
    raise AttributeError(name)

"""Single source of truth for the runtime protocol vocabulary and coverage.

The canonical *runtime* vocabulary is ``agents/chembench_agent.py``'s
``protocol_mode`` dispatch keys.  ``harness_contract.yaml``'s ``protocol.type``
values are a coarse declarative taxonomy; ``CONTRACT_TYPE_TO_MODES`` bridges the
two so coverage can be audited and gated in exactly one place.

This module is intentionally dependency-light (stdlib + PyYAML) so it can run on
the Windows host as well as inside the WSL/Harbor environment.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents" / "chembench_agent.py"

# Coarse declarative `protocol.type` -> allowed runtime `protocol_mode` values.
# A `protocol_mode` may be shared by several contract types (e.g. hidden_suite_case
# backs controlled_tool, staged_campaign and controlled_suite).
CONTRACT_TYPE_TO_MODES: dict[str, set[str]] = {
    "single_step": {"generic"},
    "multi_round": {
        "task03_design_round",
        "candidate_only_round",
        "task15_diagnostic_round",
        "private_response_round",
    },
    "controlled_tool": {"hidden_suite_case", "workspace_sidecar_case"},
    "controlled_suite": {"hidden_suite_case"},
    "staged_campaign": {"hidden_suite_case"},
    "adaptive_campaign": {"adaptive_hidden_suite_case"},
}

# `implementation_status` values are constrained by
# schemas/harness-contract.schema.json: {implemented, partial, author_input_required}.
STATUS_IMPLEMENTED = "implemented"
STATUS_AUTHOR_REQUIRED = "author_input_required"


def supported_protocol_modes(agent: Path = AGENT) -> set[str]:
    source = agent.read_text(encoding="utf-8")
    return set(re.findall(r"protocol_mode'\)\s*==\s*'([^']+)'", source))


def agent_implements(mode: str, supported: set[str]) -> bool:
    # "generic" is the intentional JSON/Python fallback at the bottom of
    # ChemBenchAgent.run(); it is a real implementation, not a gap.
    return mode == "generic" or mode in supported


def task_protocol_mode(task_dir: Path) -> str:
    cfg_path = task_dir / "environment" / "public" / ".chembench_agent.json"
    if not cfg_path.is_file():
        return "generic"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg.get("protocol_mode") or "generic"


def task_release_eligible(task_dir: Path) -> bool:
    manifest = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    return bool((manifest.get("metadata") or {}).get("release_eligible", False))


def load_contract(task_dir: Path) -> dict:
    path = task_dir / "harness_contract.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def contract_type(contract: dict) -> str:
    return ((contract.get("protocol") or {}).get("type")) or ""


def contract_implementation_status(contract: dict) -> str:
    return ((contract.get("protocol") or {}).get("implementation_status")) or ""


def derived_implementation_status(mode: str, supported: set[str]) -> str:
    return STATUS_IMPLEMENTED if agent_implements(mode, supported) else STATUS_AUTHOR_REQUIRED


def contract_paths(task_dir: Path) -> list[Path]:
    """Both copies of the generated contract that release tooling must keep in sync."""
    paths = [task_dir / "harness_contract.yaml"]
    public = task_dir / "environment" / "public" / "harness_contract.yaml"
    if public.is_file():
        paths.append(public)
    return paths

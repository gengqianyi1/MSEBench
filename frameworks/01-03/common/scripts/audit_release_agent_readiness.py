#!/usr/bin/env python3
"""Audit whether release-eligible tasks can be driven by MSEBenchAgent.

This is deliberately a *release gate*, not a score checker.  It prevents a
task from being selected for an official Agent batch merely because its
``task.toml`` says ``release_eligible = true`` while its public protocol has
no implementation in the common Agent.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
AGENT = ROOT / "agents" / "chembench_agent.py"


def supported_protocols() -> set[str]:
    source = AGENT.read_text(encoding="utf-8")
    return set(re.findall(r"protocol_mode'\)\s*==\s*'([^']+)'", source))


def task_record(task_dir: Path, supported: set[str]) -> dict[str, object]:
    manifest = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    task_metadata = manifest.get("task", {})
    public = task_dir / "environment" / "public"
    agent_config_path = public / ".chembench_agent.json"
    deployment_path = public / "sidecar_deployment.yaml"
    config: dict[str, object] = {}
    config_error = None
    if agent_config_path.is_file():
        try:
            config = json.loads(agent_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            config_error = str(exc)
    mode = config.get("protocol_mode") if isinstance(config, dict) else None
    release = bool(metadata.get("release_eligible", False))
    issues: list[str] = []
    if not agent_config_path.is_file():
        issues.append("missing .chembench_agent.json")
    elif config_error:
        issues.append(f"invalid .chembench_agent.json: {config_error}")
    elif isinstance(mode, str) and mode not in supported:
        issues.append(f"unsupported protocol_mode: {mode}")
    elif not isinstance(mode, str):
        # Generic mode is a supported intentional fallback, but flag it in the
        # machine-readable result so a release reviewer can see it.
        mode = "generic"
    deployment = deployment_path.read_text(encoding="utf-8") if deployment_path.is_file() else ""
    if release and "AUTHOR_REQUIRED_SHA256" in deployment:
        issues.append("release task retains AUTHOR_REQUIRED_SHA256 sidecar image")
    return {
        "directory": task_dir.name,
        "task_name": task_metadata.get("name"),
        "release_eligible": release,
        "protocol_mode": mode,
        "agent_config": str(agent_config_path.relative_to(ROOT)) if agent_config_path.is_file() else None,
        "sidecar_deployment": str(deployment_path.relative_to(ROOT)) if deployment_path.is_file() else None,
        "issues": issues,
        "ready_for_common_agent": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-unreleased", action="store_true", help="emit all tasks, not only release tasks")
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args()
    supported = supported_protocols()
    records = [task_record(path, supported) for path in sorted(TASKS.glob("task-*")) if (path / "task.toml").is_file()]
    shown = records if args.include_unreleased else [row for row in records if row["release_eligible"]]
    report = {
        "schema_version": "chembench-agent-readiness/1.0",
        "agent": str(AGENT.relative_to(ROOT)),
        "supported_protocol_modes": sorted(supported),
        "tasks": shown,
        "release_gate_pass": all(bool(row["ready_for_common_agent"]) for row in shown),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["release_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

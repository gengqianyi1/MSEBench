"""Bind one public task configuration to the generic disposable runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from runtime.runner import RunState
from runtime.staging import ModelWorkspace
from runtime.tools import ToolRuntime


@dataclass(frozen=True)
class TaskRuntimeConfig:
    main_compute_credits: int
    recovery_compute_credits: int
    operation_costs: dict[str, int]
    allowed_operations: frozenset[str]
    required_outputs: tuple[str, ...]
    default_charge: int
    default_uhf: int
    xtb_executable: Path
    timeout_seconds: int
    max_scan_points: int | None


def load_task_config(task_yaml: Path) -> TaskRuntimeConfig:
    raw = yaml.safe_load(Path(task_yaml).read_text(encoding="utf-8"))
    costs = {
        "sp": int(raw["costs"]["sp"]),
        "opt": int(raw["costs"]["opt"]),
        "hess": int(raw["costs"]["hessian"]),
    }
    allowed = frozenset(costs)
    if raw["scan_allowed"]:
        allowed = allowed | {"scan"}
        costs["scan"] = int(raw["costs"]["scan"])
    return TaskRuntimeConfig(
        main_compute_credits=int(raw["main_compute_credits"]),
        recovery_compute_credits=int(raw["recovery_compute_credits"]),
        operation_costs=costs,
        allowed_operations=allowed,
        required_outputs=tuple(str(item) for item in raw["required_outputs"]),
        default_charge=int(raw["physical_settings"]["charge"]),
        default_uhf=int(raw["physical_settings"]["uhf"]),
        xtb_executable=Path(raw["runtime"]["xtb_executable"]),
        timeout_seconds=int(raw["runtime"]["timeout_seconds"]),
        max_scan_points=(
            None
            if raw.get("max_scan_points") is None
            else int(raw["max_scan_points"])
        ),
    )


def build_task_runtime(
    workspace: ModelWorkspace,
    task_yaml: Path,
    *,
    xtb_executable: Path | None = None,
    python_executable: Path,
    runtime_paths: tuple[Path, ...],
    bubblewrap_executable: Path = Path("/usr/bin/bwrap"),
    resume_existing: bool = False,
) -> ToolRuntime:
    config = load_task_config(task_yaml)
    run_dir = workspace.public_root.parents[1]
    state_factory = RunState.from_existing if resume_existing else RunState
    state = state_factory(
        run_dir,
        config.main_compute_credits,
        config.recovery_compute_credits,
        xtb_executable or config.xtb_executable,
        config.timeout_seconds,
        operation_costs=config.operation_costs,
        allowed_operations=config.allowed_operations,
        max_scan_points=config.max_scan_points,
    )
    return ToolRuntime(
        workspace,
        state,
        config,
        python_executable=python_executable,
        runtime_paths=runtime_paths,
        bubblewrap_executable=bubblewrap_executable,
    )

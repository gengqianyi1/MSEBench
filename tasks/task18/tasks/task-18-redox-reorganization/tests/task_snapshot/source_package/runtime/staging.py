"""Create the three directories visible to one model run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelWorkspace:
    public_root: Path
    workspace_root: Path
    submission_root: Path


def stage_model_workspace(public_dir: Path, run_dir: Path) -> ModelWorkspace:
    public_source = Path(public_dir).resolve(strict=True)
    if not public_source.is_dir():
        raise NotADirectoryError(public_source)

    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    model_root = run_root / "model"
    model_root.mkdir()
    public_root = model_root / "public"
    workspace_root = model_root / "workspace"
    submission_root = model_root / "submission"

    shutil.copytree(public_source, public_root)
    workspace_root.mkdir()
    submission_root.mkdir()
    return ModelWorkspace(public_root, workspace_root, submission_root)

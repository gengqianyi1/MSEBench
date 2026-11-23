"""Validation boundary for model-visible task packages."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


class PublicSurfaceViolation(ValueError):
    """Raised when a public task exposes non-public benchmark information."""


def validate_public_task_surface(task_root: Path) -> None:
    """Validate one model-visible task directory."""

    root = Path(task_root).resolve()
    if not root.is_dir():
        raise PublicSurfaceViolation(f"public task directory does not exist: {root}")

    forbidden_files = {"identity_rule.yaml", "selected_torsions.csv"}
    for path in root.rglob("*"):
        if path.is_file() and path.name.casefold() in forbidden_files:
            raise PublicSurfaceViolation(f"forbidden public file: {path.relative_to(root).as_posix()}")

    forbidden_keys = {
        "required_stages",
        "hessian_allocation",
        "retention_rule",
        "retain_top",
        "pose_family",
        "forward_schedule",
        "reverse_schedule",
        "merge_threshold",
        "fourier_order",
        "refinement_points",
        "deduplication",
    }

    def visit(value: object, *, source: Path) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold()
                if normalized in forbidden_keys:
                    raise PublicSurfaceViolation(
                        f"forbidden route key {key!r} in {source.relative_to(root).as_posix()}"
                    )
                visit(child, source=source)
        elif isinstance(value, list):
            for child in value:
                visit(child, source=source)

    for suffix in ("*.yaml", "*.yml", "*.json"):
        for path in root.rglob(suffix):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, yaml.YAMLError) as exc:
                raise PublicSurfaceViolation(f"unreadable public data file: {path}") from exc
            visit(data, source=path)

    reference_disclosure = re.compile(
        r"(?:reference\s+(?:barrier|energy|value)|golden\s+value)[^\n]{0,48}?[-+]?\d",
        re.IGNORECASE,
    )
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if reference_disclosure.search(text):
            raise PublicSurfaceViolation(
                f"forbidden reference disclosure in {path.relative_to(root).as_posix()}"
            )

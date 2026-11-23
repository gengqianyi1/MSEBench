#!/usr/bin/env python3
"""Write Harbor 0.21-compatible dataset manifests with exact task package digests."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pathspec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import harness_coverage as hc  # noqa: E402


DEFAULT_IGNORES = ["__pycache__/", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~"]


def collect_files(task_dir: Path) -> list[Path]:
    candidates = [task_dir / name for name in ("task.toml", "instruction.md", "README.md")]
    for name in ("environment", "tests", "solution", "steps"):
        directory = task_dir / name
        if directory.exists():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    ignore = task_dir / ".gitignore"
    lines = ignore.read_text(encoding="utf-8").splitlines() if ignore.exists() else DEFAULT_IGNORES
    spec = pathspec.PathSpec.from_lines("gitignore", lines)
    return sorted(
        {path.resolve() for path in candidates if path.is_file() and not spec.match_file(path.relative_to(task_dir).as_posix())},
        key=lambda path: path.relative_to(task_dir).as_posix(),
    )


def digest(task_dir: Path) -> str:
    outer = hashlib.sha256()
    for path in collect_files(task_dir):
        relative = path.relative_to(task_dir).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        outer.update(f"{relative}\0{file_hash}\n".encode())
    return outer.hexdigest()


def dataset_text(name: str, version: str, description: str, tasks: list[tuple[str, str]]) -> str:
    lines = [
        'schema_version = "1.0"',
        "",
        "[dataset]",
        f'name = "{name}"',
        f'version = "{version}"',
        f'description = "{description}"',
    ]
    for task_name, task_digest in tasks:
        lines.extend(["", "[[tasks]]", f'name = "{task_name}"', f'digest = "sha256:{task_digest}"'])
    return "\n".join(lines) + "\n"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    supported = hc.supported_protocol_modes()
    refs = []
    for task_dir in sorted((root / "tasks").glob("task-*")):
        # Development admits only tasks the common agent can actually drive.
        # Tasks whose protocol_mode has no agent branch (e.g. workspace_sidecar_case)
        # are excluded until their adapter lands or the author migrates the protocol.
        mode = hc.task_protocol_mode(task_dir)
        if not hc.agent_implements(mode, supported):
            continue
        short_name = task_dir.name
        refs.append((f"chembench/{short_name}", digest(task_dir)))
    manifests = {
        "development": dataset_text(
            "chembench/chembench-development", "0.2.0",
            "Canonical development tasks; local execution uses ./chembench run-development.", refs,
        ),
        "parity": dataset_text(
            "chembench/chembench-parity", "0.2.0",
            "Tasks admitted only after build and Oracle gates.", [],
        ),
        "release": dataset_text(
            "chembench/chembench-release", "0.2.0",
            "Tasks admitted only after every release gate.", [],
        ),
    }
    for name, text in manifests.items():
        path = root / "datasets" / name / "dataset.toml"
        path.write_text(text, encoding="utf-8", newline="\n")
        print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build physically separated participant and organizer source bundles.

The participant bundle is intentionally not a local grading bundle: it contains task instructions
and public environments, while private verifier sources are expected to be deployed as access-
controlled images by the benchmark operator.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import zipfile


ROOT_PUBLIC_FILES = {
    "README.md", "pyproject.toml", "TASK_INVENTORY.json", "chembench",
}
ROOT_PUBLIC_DIRS = {
    "msebench_harness", "schemas", "templates",
}
TASK_PUBLIC_FILES = {
    "task.toml", "instruction.md", "README.md", "MIGRATION.md", "harness_contract.yaml",
}
FORBIDDEN_PARTS = {
    "private_data", "private_grader", "replay_corpus", "legacy_upstream",
    "organizer_private", ".audit-venv", ".harbor-wheel", "__pycache__",
    ".pytest_cache", "docs/archive",
}
FORBIDDEN_NAMES = {
    "expected.json", "reference_target.yaml", "oracle.npz", "final_report.json",
    "archived_scores.md",
}
SECRET_PATTERN = re.compile(
    rb"(?i)(?:api[_-]?key|authorization)\s*[\"']?\s*[:=]\s*[\"']?"
    rb"(?!YOUR_|PUT_YOUR_|<|\$\{|os\.getenv|env:)([A-Za-z0-9_./+\-=]{20,})"
)
ORGANIZER_RUNTIME_ROOT_PATTERNS = (
    "dist",
    "eval_runs",
    "fresh_workspaces",
    "jobs",
    "jobs_*",
    "llm_eval",
    "parity",
    "runs",
    "runs_*",
)


def _iter_files(path: Path):
    if path.is_file():
        if "__pycache__" not in path.parts and path.suffix != ".pyc":
            yield path
    elif path.is_dir():
        yield from (
            item for item in path.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
        )


def participant_files(root: Path) -> list[Path]:
    selected: set[Path] = set()
    for name in ROOT_PUBLIC_FILES:
        path = root / name
        if path.is_file(): selected.add(path)
    for name in ROOT_PUBLIC_DIRS:
        selected.update(_iter_files(root / name))
    for dataset_name in ("development", "parity", "release"):
        selected.update(_iter_files(root / "datasets" / dataset_name))
    for task in sorted((root / "tasks").glob("task-*")):
        for name in TASK_PUBLIC_FILES:
            path = task / name
            if path.is_file(): selected.add(path)
        selected.update(_iter_files(task / "environment"))
        # Multi-step participant material includes instructions and public setup hooks, never
        # Oracle solutions or verifier scripts.
        steps = task / "steps"
        if steps.is_dir():
            for step in steps.iterdir():
                instruction = step / "instruction.md"
                if instruction.is_file(): selected.add(instruction)
                selected.update(_iter_files(step / "workdir"))
    selected = {
        path for path in selected
        if path.name.lower() not in FORBIDDEN_NAMES
        and not any(part in path.relative_to(root).as_posix().lower() for part in FORBIDDEN_PARTS)
    }
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def organizer_files(root: Path, output: Path) -> list[Path]:
    ignored = {".audit-venv", ".harbor-wheel", "__pycache__", ".pytest_cache"}

    def runtime_root(path: Path) -> bool:
        relative = path.relative_to(root)
        if not relative.parts:
            return False
        first = relative.parts[0].lower()
        return any(
            first == pattern or (pattern.endswith("*") and first.startswith(pattern[:-1]))
            for pattern in ORGANIZER_RUNTIME_ROOT_PATTERNS
        )

    return sorted(
        (
            path for path in root.rglob("*")
            if path.is_file()
            and not ignored.intersection(part.lower() for part in path.relative_to(root).parts)
            and not runtime_root(path)
            and path != output
            and output not in path.parents
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def audit_participant(root: Path, files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        relative = PurePosixPath(path.relative_to(root).as_posix())
        relative_text = relative.as_posix().lower()
        lowered = {part.lower() for part in relative.parts}
        wrapped = f"/{relative_text}/"
        if lowered & FORBIDDEN_PARTS or any(f"/{part.strip('/')}/" in wrapped for part in FORBIDDEN_PARTS):
            errors.append(f"forbidden path: {relative}")
        if relative.name.lower() in FORBIDDEN_NAMES:
            errors.append(f"forbidden file: {relative}")
        if relative.name.startswith("#U") or "审核意见" in relative.name or "reviewer_report" in relative_text:
            errors.append(f"review artifact exposed to participant: {relative}")
        if path.stat().st_size <= 5_000_000:
            data = path.read_bytes()
            match = SECRET_PATTERN.search(data)
            if match:
                errors.append(f"possible embedded secret: {relative}")
    return errors


def write_zip(root: Path, output: Path, files: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output-dir", default="dist")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output_dir).resolve()
    participant = participant_files(root)
    errors = audit_participant(root, participant)
    if errors:
        for error in errors: print(f"ERROR {error}")
        return 1
    participant_zip = output / "chembench-participant-public.zip"
    organizer_zip = output / "chembench-organizer-source.zip"
    write_zip(root, participant_zip, participant)
    write_zip(root, organizer_zip, organizer_files(root, output))
    print(f"participant_files={len(participant)} path={participant_zip}")
    print(f"organizer_path={organizer_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

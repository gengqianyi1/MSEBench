"""Assemble a MSEBench source checkout and checksum-locked Hugging Face resources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


def safe_path(root: Path, name: str) -> Path:
    relative = Path(name)
    path = root / relative
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or path.resolve() != path:
        raise ValueError(f"Unsafe workspace file: {name}")
    return path


def prepare(code: Path, output: Path, *, source: Path | None = None, repo: str | None = None,
            revision: str | None = None, tasks: set[str] | None = None, images: bool = False) -> None:
    code, output = code.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a new workspace output directory")
    manifest_bytes = (code / "data_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("release_mode") != "local-evaluation":
        raise ValueError("Expected a local-evaluation data manifest")
    if source:
        source = source.resolve()
        if (source / "data_manifest.json").read_bytes() != manifest_bytes:
            raise ValueError("Code and data manifests differ")
    elif not (re.fullmatch(r"[\w.-]+/[\w.-]+", repo or "", flags=re.ASCII)
              and re.fullmatch(r"[0-9a-f]{40}", revision or "")):
        raise ValueError("Provide --source or --repo owner/dataset and a full --revision commit SHA")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".msebench-prepare-") as temp:
        workspace = (Path(temp) / "workspace").resolve()
        if any(path.is_symlink() for path in code.rglob("*")):
            raise ValueError("Source candidate must contain regular files")
        shutil.copytree(code, workspace, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", ".git", ".venv"))
        count = 0
        for record in manifest["files"]:
            name = record["path"]
            if tasks and record["task"].startswith("task") and record["task"] not in tasks:
                continue
            if name.endswith(".tar") and not images:
                continue
            target = safe_path(workspace, name)
            if target.exists():
                raise ValueError(f"Code/data path collision: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source:
                handle = safe_path(source, name).open("rb")
            else:
                url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(name)}"
                handle = urlopen(url, timeout=120)
            checksum, size = hashlib.sha256(), 0
            with handle, target.open("xb") as destination:
                while chunk := handle.read(1024 * 1024):
                    size += len(chunk)
                    if size > record["bytes"]:
                        raise ValueError(f"Resource exceeds declared size: {name}")
                    checksum.update(chunk)
                    destination.write(chunk)
            if size != record["bytes"] or checksum.hexdigest() != record["sha256"]:
                raise ValueError(f"Resource checksum mismatch: {name}")
            target.chmod(0o755 if record.get("executable") else 0o644)
            count += 1
        for name, link in manifest["workspace_links"].items():
            path = safe_path(workspace, name)
            if Path(link).is_absolute():
                raise ValueError(f"Absolute workspace link: {name}")
            target = (path.parent / link).resolve()
            if not target.is_relative_to(workspace):
                raise ValueError(f"External workspace link: {name}")
            # This link intentionally points to a newly created run-output directory.
            if name == "frameworks/07-09/runtime":
                target.mkdir(parents=True, exist_ok=True)
            if target.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(link, target_is_directory=target.is_dir())
        record = {
            "version": manifest["version"], "release_mode": "local-evaluation",
            "tasks": sorted(tasks) if tasks else [f"task{i:02}" for i in range(1, 21)] + ["textqa"],
            "images_downloaded": images, "resource_files": count, "data_revision": revision,
        }
        (workspace / "workspace.json").write_text(json.dumps(record, indent=2) + "\n")
        workspace.rename(output)
    print(f"Assembled {count} verified resources into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--source", type=Path, help="Local Hugging Face candidate directory")
    location.add_argument("--repo")
    parser.add_argument("--revision")
    parser.add_argument("--task", help="Comma-separated task numbers or textqa, default all")
    parser.add_argument("--images", action="store_true", help="Include prebuilt Docker archives")
    args = parser.parse_args()
    try:
        tasks = None
        if args.task:
            tasks = {f"task{int(n):02}" if n.isdigit() and 1 <= int(n) <= 20 else n for n in args.task.split(",")}
            if not tasks or not tasks <= {f"task{n:02}" for n in range(1, 21)} | {"textqa"}:
                raise ValueError("Select task numbers 1–20 or textqa")
        prepare(args.code, args.output, source=args.source, repo=args.repo, revision=args.revision,
                tasks=tasks, images=args.images)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Workspace preparation failed: {exc}\n")


if __name__ == "__main__":
    main()

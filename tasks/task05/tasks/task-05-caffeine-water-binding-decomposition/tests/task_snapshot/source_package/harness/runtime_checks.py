"""Task-declared, read-only runtime verification for xTB benchmarks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import psutil

from .schema_validation import validate_runtime_checks
from .path_io import iter_tree_files, path_is_dir, path_is_file, sha256_file


VERSION_PATTERN = re.compile(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)")


def sha256_tree(path: Path) -> str:
    """Hash relative names, sizes and contents of a directory deterministically."""
    root = Path(path)
    digest = hashlib.sha256()
    if not path_is_dir(root):
        return digest.hexdigest()
    for logical_relative, file_path in iter_tree_files(root):
        relative = logical_relative.as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def evidence_tree_sha256(files_sha256: Mapping[str, str]) -> str:
    """Frozen evidence tree (spec 5.1).

    Canonical relative POSIX paths are sorted; each record is
    path + NUL + sha256 + LF; the concatenation is SHA-256.
    """
    digest = hashlib.sha256()
    for raw_path in sorted(files_sha256.keys()):
        normalized = str(raw_path).replace("\\", "/")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(files_sha256[raw_path]).lower().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _module_version(module_name: str, module: Any) -> str:
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    distribution_name = "PyYAML" if module_name == "yaml" else module_name.split(".", 1)[0]
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "stdlib-or-unknown"


def _check_module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"name": name, "status": "passed", "version": _module_version(name, module)}
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _check_binary(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = str(spec["name"])
    expected = str(spec["version"])
    resolved = shutil.which(name)
    if resolved is None:
        return {
            "name": name,
            "path": name,
            "expected_version": expected,
            "actual_version": None,
            "sha256": None,
            "status": "failed",
            "error": "binary not found on PATH",
        }
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        match = VERSION_PATTERN.search(combined)
        actual = match.group(1) if match else None
        ok = completed.returncode == 0 and actual == expected
        item = {
            "name": name,
            "path": str(Path(resolved).resolve()),
            "expected_version": expected,
            "actual_version": actual,
            "sha256": sha256_file(Path(resolved)),
            "status": "passed" if ok else "failed",
        }
        if not ok:
            item["error"] = f"expected {expected}, observed {actual or 'unparseable'}"
        return item
    except Exception as exc:
        return {
            "name": name,
            "path": str(Path(resolved).resolve()),
            "expected_version": expected,
            "actual_version": None,
            "sha256": sha256_file(Path(resolved)),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _safe_task_path(task_dir: Path, relative: str) -> Path:
    root = Path(task_dir).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"runtime check path escapes task directory: {relative}")
    return resolved


def run_runtime_checks(
    task_dir: Path,
    declaration: Mapping[str, Any],
    schema_dir: Path,
) -> dict[str, Any]:
    """Execute only checks named by the task and return schema-valid evidence."""
    task_root = Path(task_dir).resolve()
    modules = [_check_module(str(name)) for name in declaration.get("python_modules", [])]
    binaries = [_check_binary(spec) for spec in declaration.get("binaries", [])]

    lock_spec = declaration["environment_lock"]
    lock_path = _safe_task_path(task_root, str(lock_spec["path"]))
    expected_lock_hash = str(lock_spec["sha256"]).lower()
    actual_lock_hash = sha256_file(lock_path) if path_is_file(lock_path) else None
    lock_ok = actual_lock_hash == expected_lock_hash
    environment_lock: dict[str, Any] = {
        "path": str(lock_spec["path"]).replace("\\", "/"),
        "expected_sha256": expected_lock_hash,
        "actual_sha256": actual_lock_hash,
        "status": "passed" if lock_ok else "failed",
    }
    if not lock_ok:
        environment_lock["error"] = "environment lock is missing or its SHA-256 does not match"

    all_passed = (
        bool(modules)
        and bool(binaries)
        and all(item["status"] == "passed" for item in modules)
        and all(item["status"] == "passed" for item in binaries)
        and lock_ok
    )
    result = {
        "schema_version": "1.0.0",
        "status": "passed" if all_passed else "failed",
        "python": {"executable": str(Path(sys.executable).resolve()), "version": platform.python_version()},
        "modules": modules,
        "binaries": binaries,
        "environment_lock": environment_lock,
        "system": {
            "platform": platform.system().lower(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        },
        "hashes": {
            "task_config_sha256": sha256_file(task_root / "task.yaml"),
            "public_inputs_sha256": sha256_tree(task_root / "input" / "original"),
        },
    }
    validate_runtime_checks(result, schema_dir)
    return result

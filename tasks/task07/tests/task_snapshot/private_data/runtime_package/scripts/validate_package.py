"""Independently inspect, permanently extract, and optionally execute a ZIP.

Static archive, manifest, source, and security checks always run before any
code from the package can execute.  ``--execute-trusted-code`` is deliberately
required for dynamic validation.  The reduced environment and invalid proxy
settings lower accidental credential and network exposure; they are not an OS
sandbox and must not be presented as one.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import yaml


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT_NAME = "qed_mol_design_benchmark"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_FILE_COUNT = 10_000
MAX_JSONL_LINES = 100_000
FILE_ATTRIBUTE_REPARSE_POINT = 0x400

ROOT_FILES = frozenset(
    {
        ".env.example",
        ".gitignore",
        "LICENSE",
        "LICENSES.md",
        "NOTICE",
        "README.md",
        "TASK.md",
        "environment.yml",
        "requirements.txt",
        "pytest.ini",
        "run_all.ps1",
        "run_all.sh",
    }
)
OPTIONAL_ROOT_FILES: frozenset[str] = frozenset()
EXACT_FILES = frozenset(
    {
        "configs/model_transition.json",
        "configs/models.example.yaml",
        "configs/task.yaml",
        "prompts/agent_prompt.md",
        "scripts/package_submission.py",
        "scripts/run_all.py",
        "scripts/run_model.py",
        "scripts/validate_package.py",
    }
)
RECURSIVE_DIRS = frozenset(
    {"baselines", "evaluator", "framework", "reports", "review", "tests", "tools", "trajectories"}
)
REQUIRED_STRUCTURE_FILES = frozenset(
    ROOT_FILES
    | EXACT_FILES
    | {
        "baselines/baseline_agent.py",
        "evaluator/evaluate.py",
        "evaluator/scoring.py",
        "framework/agent_runner.py",
        "framework/schemas.py",
        "framework/siliconflow_adapter.py",
        "tools/rdkit_tools.py",
    }
)
REQUIRED_REPORTS = frozenset(
    {
        "reports/initial_audit.md",
        "reports/recovery_status.md",
        "reports/file_migration_report.md",
        "reports/baseline_report.md",
        "reports/test_report.md",
        "reports/model_comparison.md",
        "reports/package_validation_report.md",
        "reports/blockers.md",
        "reports/final_delivery_report.md",
        "reports/remediation_plan.md",
        "reports/remediation_test_report.md",
        "reports/security_remediation_report.md",
        "reports/final_remediation_report.md",
        "reports/PROJECT_HANDOFF_20260718.md",
        "reports/TRAJECTORY_HANDOFF_20260718.md",
        "reports/model_transition_report.md",
        "reports/legacy_model_failures.md",
        "reports/final_release_test_report.md",
        "reports/final_security_report.md",
    }
)
REQUIRED_DIRECTORIES = frozenset(
    {"baselines", "configs", "evaluator", "framework", "prompts", "reports", "review", "scripts", "tests", "tools", "trajectories"}
)
REQUIRED_TRAJECTORY_FILES = frozenset(
    {"trajectory.jsonl", "dialogue.md", "metadata.json", "final_answer.json", "evaluation.json"}
)
REQUIRED_HASHED_SOURCES = frozenset(
    {
        "framework/agent_runner.py",
        "framework/schemas.py",
        "framework/siliconflow_adapter.py",
        "tools/rdkit_tools.py",
        "evaluator/evaluate.py",
    }
)
CANONICAL_DEPENDENCY_IDENTIFIERS: Mapping[str, str] = {
    "transport": "framework.siliconflow_adapter.SiliconFlowChatTransport",
    "tool_registry": "tools.rdkit_tools.TOOL_REGISTRY",
    "evaluator": "evaluator.evaluate.evaluate_submission",
}
CANONICAL_DEPENDENCY_ATTESTATIONS: Mapping[str, str] = {
    "transport": "class_source_instance_client_verified",
    "tool_registry": "identity_source_verified",
    "evaluator": "identity_source_verified",
}
FORBIDDEN_PARTS = frozenset(
    {
        ".env",
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".vscode",
        "__pycache__",
        "answers",
        "credentials",
        "debug",
        "debug_dumps",
        "env",
        "outputs",
        "raw_logs",
        "recovery_backup",
        "runs",
        "secrets",
        "submission_packages",
        "temporary_validation",
        "validation_outputs",
        "venv",
    }
)
FORBIDDEN_FILE_NAMES = frozenset(
    {
        "api_config.json",
        "credentials.json",
        "credentials.yaml",
        "credentials.yml",
        "debug.json",
        "debug_dump.json",
        "models.yaml",
        "raw_response.json",
        "raw_responses.json",
        "request_dump.json",
        "response_dump.json",
        "secrets.json",
        ".env",
        ".coverage",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {".bak", ".dump", ".log", ".pkl", ".pyc", ".pyo", ".tmp", ".trace", ".zip"}
)
ALLOWED_SUFFIXES = frozenset(
    {".json", ".jsonl", ".md", ".ps1", ".py", ".sh", ".txt", ".yaml", ".yml"}
)
SPECIAL_ALLOWED_NAMES = frozenset(
    {".env.example", ".gitignore", "LICENSE", "NOTICE", "requirements.txt", "pytest.ini"}
)
SILICONFLOW_ENDPOINT = "https://api.siliconflow.cn/v1"
EXPECTED_TRAJECTORY_MODELS: Mapping[str, str] = {
    "glm": "zai-org/GLM-5.2",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "kimi": "Pro/moonshotai/Kimi-K2.6",
}
MODEL_TRANSITION_FILE = "configs/model_transition.json"
FROZEN_KIMI_MODEL = "Pro/moonshotai/Kimi-K2.6"
FROZEN_KIMI_ARTIFACT_SHA256: Mapping[str, str] = {
    "dialogue.md": "0a57cd0030065bdac9bc998018c50749dc18de9dac12cd3ae7e5faec581e1eaa",
    "evaluation.json": "78c961d181028ddb8dbab0f9a9afa965936ce8c3c124d3e94c853d3e58f7cc79",
    "final_answer.json": "fd2da5a0e910245e621923b2fbfa808c5eea390bb8dbc5673e45622bf64952de",
    "metadata.json": "cfd7ff6d77dbc6f24d451b8583cdc3892f221cdc9c47c845be30e7e68df1a660",
    "trajectory.jsonl": "34a540b4dda7a8924d3728bbbf650838c9c7556606bc7d3c0f3817f7c142ca79",
}
APPROVED_MODEL_TRANSITIONS: Mapping[str, tuple[str, str]] = {
    "glm": ("THUDM/GLM-4-9B-0414", "zai-org/GLM-5.2"),
    "deepseek": (
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "deepseek-ai/DeepSeek-V4-Pro",
    ),
    "kimi": (FROZEN_KIMI_MODEL, FROZEN_KIMI_MODEL),
}
APPROVED_KIMI_SOURCE_TRANSITIONS: Mapping[str, tuple[str, str]] = {
    "configs/models.example.yaml": (
        "df20ebae2e6d23bdcb17462c0b7d3576c39cf02bb952146a071594b56c821f12",
        "b3348f519bf123ae7e43c35cd89c3dec73fedcd3ac72ed1c3e988966623981b0",
    ),
    "framework/schemas.py": (
        "06049f41379b146d03d397dacd76d4f3882a5fc157c8a4025c4c9a361758ae65",
        "95da9fcc9410ea29734713e4537bb81149efa01015467ed220c3e7dbe9218199",
    ),
}

_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("github_token", re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer", re.compile(r"(?i)\bauthorization\s*[:=]\s*['\"]?bearer\s+([A-Za-z0-9._~+/=-]{12,})")),
    ("cookie", re.compile(r"(?i)\b(?:set-)?cookie\s*[:=]\s*['\"]?([^\s'\";,]{12,})")),
    (
        "configured_api_key_assignment",
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])GJLD_API_KEY\s*[:=]\s*['\"]?([^\s'\",;}{]{12,})"
        ),
    ),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|client[_-]?secret|password|passwd)"
            r"\s*[:=]\s*['\"]?([^\s'\",;}{]{12,})"
        ),
    ),
)
_PLACEHOLDER_WORDS = (
    "changeme",
    "dummy",
    "example",
    "fake",
    "placeholder",
    "redacted",
    "replace",
    "test",
    "your_",
    "your-",
)
_WINDOWS_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/](?:[^\s'\"`<>|?*\[\]{}(),;]+[\\/]?)+)"
)
_POSIX_ABSOLUTE = re.compile(
    r"(?<![:A-Za-z0-9_])(/(?:home|Users|tmp|var/tmp|workspace|mnt/[A-Za-z0-9._-]+)(?:/[^\s'\"`<>|?*\[\]{}(),;]+)+)"
)
_SENSITIVE_ENV_NAME = re.compile(
    r"(?i)(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|COOKIE|CREDENTIAL|AUTH|SESSION|PROXY)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derived_workspace_root(project_root: Path = SCRIPT_ROOT) -> Path:
    resolved = project_root.resolve()
    if len(resolved.parents) < 2:
        raise ValueError("Cannot derive a workspace root from the validator location")
    return resolved.parents[1]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_write_path(value: str | os.PathLike[str], workspace_root: Path) -> Path:
    workspace = workspace_root.expanduser().resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    if candidate == workspace or not _is_relative_to(candidate, workspace):
        raise ValueError(f"Write path must be below workspace root {workspace}: {candidate}")
    return candidate


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        stat_result = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & FILE_ATTRIBUTE_REPARSE_POINT
        or (callable(is_junction) and is_junction())
    )


def _strict_json_text(text: str, description: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{description} contains duplicate key {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{description} contains invalid JSON constant {value}")
        ),
    )


def _read_utf8(path: Path, *, limit: int = MAX_FILE_BYTES) -> str:
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError(f"file exceeds {limit} bytes")
    if b"\x00" in raw:
        raise ValueError("file contains NUL bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("file is not valid UTF-8") from exc


def _strict_json_file(path: Path) -> Any:
    return _strict_json_text(_read_utf8(path), path.as_posix())


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().strip("'\"<>[](){}").casefold()
    return (
        not lowered
        or any(word in lowered for word in _PLACEHOLDER_WORDS)
        or "${" in value
        or "$env:" in lowered
        or lowered.startswith(("os.", "self.", "config.", "metadata.", "environment.", "getenv"))
        or "validate_credentials" in lowered
        or lowered.endswith("_api_key")
        or lowered.endswith("_token")
        or lowered.endswith("_secret")
    )


def is_package_relative_allowed(relative: PurePosixPath) -> bool:
    parts = relative.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return False
    lowered = tuple(part.casefold() for part in parts)
    if any(part in FORBIDDEN_PARTS for part in lowered):
        return False
    name = parts[-1]
    lowered_name = name.casefold()
    lowered_stem = Path(name).stem.casefold().lstrip(".")
    if lowered_name in FORBIDDEN_FILE_NAMES:
        return False
    if lowered_name.startswith(".env") and lowered_name != ".env.example":
        return False
    if lowered_stem.startswith(
        (
            "credential",
            "debug",
            "raw_log",
            "raw_response",
            "request_dump",
            "response_dump",
            "secret",
        )
    ):
        return False
    if Path(name).suffix.casefold() in FORBIDDEN_SUFFIXES:
        return False
    if any(part.startswith(".") for part in parts[:-1]):
        return False
    text = relative.as_posix()
    if text in ROOT_FILES or text in OPTIONAL_ROOT_FILES or text in EXACT_FILES or text == "MANIFEST.json":
        return True
    if parts[0] == "trajectories":
        if text == "trajectories/README.md":
            return True
        return (
            len(parts) == 3
            and parts[1] in EXPECTED_TRAJECTORY_MODELS
            and parts[2] in REQUIRED_TRAJECTORY_FILES
        )
    return (
        len(parts) >= 2
        and parts[0] in RECURSIVE_DIRS
        and (name in SPECIAL_ALLOWED_NAMES or Path(name).suffix.casefold() in ALLOWED_SUFFIXES)
    )


def _zip_member_kind(info: zipfile.ZipInfo) -> str:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    dos_attributes = info.external_attr & 0xFFFF
    if dos_attributes & FILE_ATTRIBUTE_REPARSE_POINT:
        return "reparse"
    if file_type == stat.S_IFLNK:
        return "symlink"
    if file_type and file_type not in (stat.S_IFREG, stat.S_IFDIR):
        return "special"
    return "directory" if info.is_dir() else "file"


def _validate_zip_name(name: str, *, is_directory: bool) -> tuple[str | None, str | None]:
    if not name or "\x00" in name:
        return None, "empty or NUL-containing member name"
    if "\\" in name:
        return None, "backslashes are forbidden in ZIP member names"
    if name.startswith("/") or name.startswith("//"):
        return None, "absolute ZIP member path"
    trimmed = name[:-1] if is_directory and name.endswith("/") else name
    raw_parts = trimmed.split("/")
    if not raw_parts or any(part in ("", ".", "..") for part in raw_parts):
        return None, "ZIP member contains an empty, dot, or parent component"
    if any(part[-1:] in {" ", "."} for part in raw_parts):
        return None, "ZIP member component has a Windows-ambiguous trailing space or dot"
    if any(ord(character) < 32 for part in raw_parts for character in part):
        return None, "ZIP member contains a control character"
    pure = PurePosixPath(*raw_parts)
    if ":" in pure.parts[0] or any(":" in part for part in pure.parts):
        return None, "drive or alternate-stream syntax is forbidden"
    normalized = unicodedata.normalize("NFC", pure.as_posix())
    if normalized != pure.as_posix():
        return None, "ZIP member name is not NFC-normalized"
    return normalized, None


def preflight_archive(archive_path: Path) -> dict[str, Any]:
    """Inspect all central-directory metadata and the embedded manifest."""

    errors: list[str] = []
    members: list[dict[str, Any]] = []
    normalized_names: dict[str, str] = {}
    roots: set[str] = set()
    total_size = 0
    manifest_member: str | None = None
    if not archive_path.is_file():
        return {"passed": False, "errors": ["archive does not exist or is not a regular file"]}
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        return {"passed": False, "errors": [f"archive exceeds {MAX_ARCHIVE_BYTES} bytes"]}
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if not infos:
                errors.append("archive is empty")
            if len(infos) > MAX_FILE_COUNT:
                errors.append(f"archive has more than {MAX_FILE_COUNT} entries")
            for info in infos:
                kind = _zip_member_kind(info)
                normalized, name_error = _validate_zip_name(
                    info.filename, is_directory=info.is_dir()
                )
                if name_error:
                    errors.append(f"unsafe member {info.filename!r}: {name_error}")
                    continue
                assert normalized is not None
                collision_key = normalized.rstrip("/").casefold()
                previous = normalized_names.get(collision_key)
                if previous is not None:
                    errors.append(
                        f"duplicate/colliding ZIP entries: {previous!r} and {info.filename!r}"
                    )
                else:
                    normalized_names[collision_key] = info.filename
                pure = PurePosixPath(normalized)
                roots.add(pure.parts[0])
                if kind in {"symlink", "reparse", "special"}:
                    errors.append(f"forbidden {kind} ZIP entry: {info.filename}")
                if info.flag_bits & 0x1:
                    errors.append(f"encrypted ZIP entry is forbidden: {info.filename}")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    errors.append(f"unsupported compression method: {info.filename}")
                if info.file_size < 0 or info.file_size > MAX_FILE_BYTES:
                    errors.append(f"entry exceeds per-file size limit: {info.filename}")
                total_size += max(0, info.file_size)
                if total_size > MAX_TOTAL_BYTES:
                    errors.append(f"archive exceeds {MAX_TOTAL_BYTES} uncompressed bytes")
                if not info.is_dir():
                    if len(pure.parts) < 2:
                        errors.append(f"file is not below one package root: {info.filename}")
                    else:
                        relative = PurePosixPath(*pure.parts[1:])
                        if not is_package_relative_allowed(relative):
                            errors.append(f"member is outside the package allowlist: {relative.as_posix()}")
                        if relative.as_posix() == "MANIFEST.json":
                            if manifest_member is not None:
                                errors.append("archive contains more than one manifest")
                            manifest_member = info.filename
                members.append(
                    {
                        "name": info.filename,
                        "kind": kind,
                        "compressed_size": info.compress_size,
                        "uncompressed_size": info.file_size,
                    }
                )
            if len(roots) != 1:
                errors.append(f"archive must contain exactly one root directory; found {sorted(roots)}")
            elif next(iter(roots)) != PACKAGE_ROOT_NAME:
                errors.append(
                    f"archive root must be exactly {PACKAGE_ROOT_NAME!r}; found {next(iter(roots))!r}"
                )
            if manifest_member is None:
                errors.append("archive does not contain root/MANIFEST.json")
            elif not errors:
                manifest_info = archive.getinfo(manifest_member)
                if manifest_info.file_size > MAX_FILE_BYTES:
                    errors.append("embedded manifest is too large")
                else:
                    try:
                        manifest_text = archive.read(manifest_member).decode("utf-8")
                        manifest = _strict_json_text(manifest_text, "embedded MANIFEST.json")
                        if not isinstance(manifest, dict):
                            errors.append("embedded manifest is not a JSON object")
                    except Exception as exc:
                        errors.append(f"embedded manifest parse failed: {type(exc).__name__}: {exc}")
            if not errors:
                corrupt = archive.testzip()
                if corrupt is not None:
                    errors.append(f"CRC check failed for {corrupt}")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        errors.append(f"archive cannot be read safely: {type(exc).__name__}: {exc}")
    return {
        "passed": not errors,
        "errors": errors,
        "entry_count": len(members),
        "total_uncompressed_bytes": total_size,
        "root_name": next(iter(roots)) if len(roots) == 1 else None,
        "manifest_member": manifest_member,
        "members": members,
    }


def extract_archive_safely(archive_path: Path, extract_dir: Path, preflight: Mapping[str, Any]) -> Path:
    if not preflight.get("passed"):
        raise ValueError("archive preflight did not pass")
    if extract_dir.exists():
        if not extract_dir.is_dir() or _is_reparse_or_symlink(extract_dir):
            raise ValueError("extract directory is not a safe regular directory")
        if any(extract_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty extract directory: {extract_dir}")
    else:
        extract_dir.mkdir(parents=True, exist_ok=False)
    extract_resolved = extract_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            normalized, error = _validate_zip_name(info.filename, is_directory=info.is_dir())
            if error or normalized is None:
                raise ValueError(f"unsafe member changed after preflight: {info.filename!r}")
            destination = (extract_dir / Path(*PurePosixPath(normalized).parts)).resolve()
            if not _is_relative_to(destination, extract_resolved):
                raise ValueError(f"member escapes extraction root: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("xb") as target:
                copied = shutil.copyfileobj(source, target, length=1024 * 1024)
                del copied
            if destination.stat().st_size != info.file_size:
                raise ValueError(f"extracted size mismatch: {info.filename}")
    root_name = preflight.get("root_name")
    root = extract_dir / str(root_name)
    if not root.is_dir() or _is_reparse_or_symlink(root):
        raise ValueError("extracted package root is missing or unsafe")
    return root


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def check_structure(root: Path) -> dict[str, Any]:
    missing_files = sorted(relative for relative in REQUIRED_STRUCTURE_FILES if not (root / relative).is_file())
    missing_directories = sorted(relative for relative in REQUIRED_DIRECTORIES if not (root / relative).is_dir())
    missing_reports = sorted(relative for relative in REQUIRED_REPORTS if not (root / relative).is_file())
    tests = sorted(path.relative_to(root).as_posix() for path in (root / "tests").glob("test_*.py")) if (root / "tests").is_dir() else []
    errors: list[str] = []
    if missing_files:
        errors.append(f"missing required files: {missing_files}")
    if missing_directories:
        errors.append(f"missing required directories: {missing_directories}")
    if missing_reports:
        errors.append(f"missing required reports: {missing_reports}")
    if not tests:
        errors.append("tests directory contains no test_*.py files")
    return {
        "passed": not errors,
        "errors": errors,
        "missing_files": missing_files,
        "missing_directories": missing_directories,
        "missing_reports": missing_reports,
        "test_files": tests,
        "project_license_present": (root / "LICENSE").is_file(),
        "project_notice_present": (root / "NOTICE").is_file(),
    }


def check_manifest(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest: Mapping[str, Any] = {}
    try:
        loaded = _strict_json_file(root / "MANIFEST.json")
        if not isinstance(loaded, dict):
            raise ValueError("manifest root must be an object")
        manifest = loaded
    except Exception as exc:
        return {"passed": False, "errors": [f"manifest parse failed: {type(exc).__name__}: {exc}"]}
    files = manifest.get("files")
    if not isinstance(files, list):
        return {"passed": False, "errors": ["manifest.files must be a list"]}
    expected: dict[str, Mapping[str, Any]] = {}
    normalized: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            errors.append(f"manifest.files[{index}] is not an object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            errors.append(f"manifest.files[{index}].path is invalid")
            continue
        raw_parts = relative.split("/")
        pure = PurePosixPath(relative)
        if (
            "\\" in relative
            or pure.is_absolute()
            or not raw_parts
            or any(part in ("", ".", "..") for part in raw_parts)
            or any(part[-1:] in {" ", "."} for part in raw_parts)
            or any(":" in part or any(ord(character) < 32 for character in part) for part in raw_parts)
            or unicodedata.normalize("NFC", relative) != relative
        ):
            errors.append(f"unsafe manifest path: {relative}")
            continue
        key = unicodedata.normalize("NFC", pure.as_posix()).casefold()
        if key in normalized:
            errors.append(f"duplicate/colliding manifest path: {relative}")
            continue
        normalized.add(key)
        if not is_package_relative_allowed(pure) or relative == "MANIFEST.json":
            errors.append(f"manifest path is outside allowlist: {relative}")
        expected[relative] = entry
        target = (root / Path(*pure.parts)).resolve()
        if not _is_relative_to(target, root.resolve()) or not target.is_file():
            errors.append(f"manifested file is missing or unsafe: {relative}")
            continue
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0 or size > MAX_FILE_BYTES:
            errors.append(f"invalid manifest size: {relative}")
        elif target.stat().st_size != size:
            errors.append(f"size mismatch: {relative}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"invalid manifest SHA-256: {relative}")
        elif sha256_file(target) != digest:
            errors.append(f"hash mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in _iter_files(root)
        if path.name != "MANIFEST.json"
    }
    extra = sorted(actual - set(expected))
    absent = sorted(set(expected) - actual)
    if extra:
        errors.append(f"unmanifested files: {extra}")
    if absent:
        errors.append(f"manifest-only files: {absent}")
    if manifest.get("file_count_excluding_manifest") != len(expected):
        errors.append("manifest file_count_excluding_manifest is incorrect")
    if manifest.get("total_size_excluding_manifest") != sum(path.stat().st_size for path in _iter_files(root) if path.name != "MANIFEST.json"):
        errors.append("manifest total_size_excluding_manifest is incorrect")
    if manifest.get("delivery_status") not in {"BLOCKED", "READY_FOR_REVIEW"}:
        errors.append("manifest delivery_status is invalid")
    if manifest.get("schema_version") != "2.0":
        errors.append("manifest schema_version must be 2.0")
    if manifest.get("benchmark") != "constrained_qed_optimization_v1":
        errors.append("manifest benchmark identity is invalid")
    blocking_reasons = manifest.get("blocking_reasons")
    if not isinstance(blocking_reasons, list) or not all(
        isinstance(reason, str) and reason for reason in blocking_reasons
    ):
        errors.append("manifest blocking_reasons must be a list of non-empty strings")
    elif manifest.get("delivery_status") == "READY_FOR_REVIEW" and blocking_reasons:
        errors.append("READY_FOR_REVIEW manifest must not contain blocking reasons")
    elif manifest.get("delivery_status") == "BLOCKED" and not blocking_reasons:
        errors.append("BLOCKED manifest must contain at least one blocking reason")
    if not isinstance(manifest.get("readiness_checks"), Mapping):
        errors.append("manifest readiness_checks must be an object")
    return {
        "passed": not errors,
        "errors": errors,
        "delivery_status": manifest.get("delivery_status"),
        "blocking_reasons": manifest.get("blocking_reasons"),
        "file_count": len(expected),
        "manifest": dict(manifest),
    }


def check_utf8(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    total = 0
    count = 0
    for path in _iter_files(root):
        count += 1
        total += path.stat().st_size
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds size limit: {relative}")
            continue
        try:
            _read_utf8(path)
        except Exception as exc:
            errors.append(f"{relative}: {exc}")
    if count > MAX_FILE_COUNT:
        errors.append(f"file count exceeds {MAX_FILE_COUNT}")
    if total > MAX_TOTAL_BYTES:
        errors.append(f"total size exceeds {MAX_TOTAL_BYTES}")
    return {"passed": not errors, "errors": errors, "file_count": count, "total_bytes": total}


def check_security(root: Path) -> dict[str, Any]:
    forbidden: list[str] = []
    secret_findings: list[dict[str, Any]] = []
    reparse_findings: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        lowered = tuple(part.casefold() for part in relative.parts)
        if any(part in FORBIDDEN_PARTS for part in lowered):
            forbidden.append(relative.as_posix())
        if path.name.casefold() in FORBIDDEN_FILE_NAMES or path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            forbidden.append(relative.as_posix())
        if _is_reparse_or_symlink(path):
            reparse_findings.append(relative.as_posix())
        if not path.is_file():
            continue
        try:
            text = _read_utf8(path)
        except Exception:
            continue
        for rule_name, pattern in _SECRET_RULES:
            for match in pattern.finditer(text):
                captured = match.group(1) if match.lastindex else match.group(0)
                if rule_name in {
                    "bearer",
                    "cookie",
                    "configured_api_key_assignment",
                    "assigned_secret",
                } and _looks_like_placeholder(captured):
                    continue
                secret_findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": text.count("\n", 0, match.start()) + 1,
                        "rule": rule_name,
                    }
                )
    return {
        "passed": not forbidden and not secret_findings and not reparse_findings,
        "forbidden_paths": sorted(set(forbidden)),
        "secret_findings": secret_findings,
        "symlink_or_reparse_findings": reparse_findings,
    }


def check_absolute_paths(root: Path) -> dict[str, Any]:
    runtime_findings: list[dict[str, Any]] = []
    report_evidence: list[dict[str, Any]] = []
    informational: list[dict[str, Any]] = []
    runtime_suffixes = {".py", ".json", ".jsonl", ".yaml", ".yml", ".ps1", ".sh"}
    runtime_names = {"requirements.txt", ".env.example", "pytest.ini"}
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            text = _read_utf8(path)
        except Exception:
            continue
        matches: list[tuple[int, str]] = []
        for pattern in (_WINDOWS_ABSOLUTE, _POSIX_ABSOLUTE):
            for match in pattern.finditer(text):
                matches.append((text.count("\n", 0, match.start()) + 1, match.group(1)[:240]))
        for line, evidence in matches:
            record = {"path": relative, "line": line, "evidence": evidence}
            if relative.startswith("reports/"):
                report_evidence.append(record)
            elif path.suffix.casefold() in runtime_suffixes or path.name in runtime_names:
                runtime_findings.append(record)
            else:
                informational.append(record)
    return {
        "passed": not runtime_findings,
        "runtime_failures": runtime_findings,
        "report_delivery_path_evidence": report_evidence,
        "other_informational_findings": informational,
        "policy": "Absolute delivery paths in reports are evidence only; runtime code/config findings fail validation.",
    }


def check_environment_files(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    parsed: dict[str, str] = {}
    for relative in ("environment.yml", "configs/task.yaml", "configs/models.example.yaml"):
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        try:
            value = yaml.safe_load(_read_utf8(path))
            if not isinstance(value, dict):
                raise ValueError("top level is not a mapping")
            parsed[relative] = "mapping"
        except Exception as exc:
            errors.append(f"invalid safe YAML {relative}: {type(exc).__name__}: {exc}")
    return {"passed": not errors, "errors": errors, "parsed": parsed}


def check_python_static(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    parsed: list[str] = []
    for path in _iter_files(root):
        if path.suffix.casefold() != ".py":
            continue
        relative = path.relative_to(root).as_posix()
        try:
            ast.parse(_read_utf8(path), filename=relative)
            parsed.append(relative)
        except Exception as exc:
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    return {"passed": not errors, "errors": errors, "parsed_file_count": len(parsed)}


def check_shell_scripts_static(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    ps_path = root / "run_all.ps1"
    sh_path = root / "run_all.sh"
    try:
        ps_text = _read_utf8(ps_path)
        ps_ok = "scripts\\run_all.py" in ps_text or "scripts/run_all.py" in ps_text
        if not ps_ok:
            errors.append("run_all.ps1 does not invoke scripts/run_all.py")
        details["run_all.ps1"] = {"static_reference_ok": ps_ok}
    except Exception as exc:
        errors.append(f"run_all.ps1 static read failed: {exc}")
    try:
        sh_text = _read_utf8(sh_path)
        sh_ok = sh_text.startswith("#!") and "set -eu" in sh_text and "scripts/run_all.py" in sh_text
        if not sh_ok:
            errors.append("run_all.sh lacks the expected safe launcher structure")
        details["run_all.sh"] = {"static_reference_ok": sh_ok}
    except Exception as exc:
        errors.append(f"run_all.sh static read failed: {exc}")
    return {"passed": not errors, "errors": errors, "scripts": details}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check_model_identity_consistency(
    metadata: Mapping[str, Any], expected_directory: str | None = None
) -> dict[str, Any]:
    """Check the SiliconFlow request/response identity recorded by the runner."""

    errors: list[str] = []
    required_top_level = (
        "provider",
        "model",
        "endpoint",
        "timestamp",
        "temperature",
        "max_tokens",
    )
    missing_top_level = [name for name in required_top_level if name not in metadata]
    if missing_top_level:
        errors.append(f"required top-level metadata is missing: {missing_top_level}")

    provider = metadata.get("provider")
    transport = metadata.get("transport")
    base_url = metadata.get("base_url")
    endpoint = metadata.get("endpoint")
    request_endpoint = metadata.get("request_endpoint")
    requested_model = metadata.get("requested_model")
    requested_model_id = metadata.get("requested_model_id")
    top_level_model = metadata.get("model")
    actual_model = metadata.get("actual_model_id")
    model_config = metadata.get("model_config")

    expected_requested = EXPECTED_TRAJECTORY_MODELS.get(expected_directory or "")
    if expected_directory is not None and expected_requested is None:
        errors.append(f"unexpected trajectory directory: {expected_directory!r}")
    if expected_directory is None:
        expected_requested = (
            requested_model
            if isinstance(requested_model, str)
            and requested_model in EXPECTED_TRAJECTORY_MODELS.values()
            else None
        )
    if expected_requested is None:
        errors.append("requested model is not one of the three exact benchmark model IDs")

    temperature = metadata.get("temperature")
    max_tokens = metadata.get("max_tokens")
    numeric_fields_valid = bool(
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and math.isfinite(float(temperature))
        and isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens > 0
    )
    if not numeric_fields_valid:
        errors.append("top-level temperature/max_tokens are missing or invalid")

    timestamp = metadata.get("timestamp")
    timestamp_valid = False
    if isinstance(timestamp, str) and timestamp == timestamp.strip() and timestamp:
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            timestamp_valid = (
                parsed_timestamp.tzinfo is not None
                and parsed_timestamp.utcoffset() is not None
            )
        except ValueError:
            pass
    if not timestamp_valid:
        errors.append("metadata.timestamp must be an ISO-8601 timestamp with a timezone")

    model_config_valid = bool(
        isinstance(model_config, Mapping)
        and model_config.get("provider") == "siliconflow"
        and model_config.get("transport") == "openai_compatible"
        and model_config.get("base_url") == SILICONFLOW_ENDPOINT
        and model_config.get("model") == expected_requested
        and model_config.get("api_key_env") == "GJLD_API_KEY"
        and model_config.get("temperature") == temperature
        and model_config.get("max_tokens") == max_tokens
    )
    if not model_config_valid:
        errors.append("metadata.model_config is missing or inconsistent with top-level fields")

    trusted_endpoint = bool(
        provider == "siliconflow"
        and transport == "openai_compatible"
        and base_url == SILICONFLOW_ENDPOINT
        and endpoint == SILICONFLOW_ENDPOINT
        and request_endpoint == SILICONFLOW_ENDPOINT + "/chat/completions"
        and requested_model == expected_requested
        and requested_model_id == expected_requested
        and model_config_valid
    )
    if not trusted_endpoint:
        errors.append(
            "provider, endpoint, and requested model are not the exact SiliconFlow benchmark mapping"
        )

    exact_actual_model = bool(
        isinstance(expected_requested, str)
        and actual_model == expected_requested
        and top_level_model == actual_model
    )
    if not exact_actual_model:
        errors.append(
            "top-level model and actual_model_id must equal the exact requested SiliconFlow model ID"
        )

    responses = metadata.get("provider_response_metadata")
    response_models = (
        [
            item.get("model")
            for item in responses
            if isinstance(item, Mapping) and isinstance(item.get("model"), str)
        ]
        if isinstance(responses, list)
        else []
    )
    response_model_verified = bool(
        exact_actual_model
        and isinstance(responses, list)
        and responses
        and len(response_models) == len(responses)
        and all(value == actual_model for value in response_models)
    )
    if not response_model_verified:
        errors.append("provider response model IDs do not verify actual_model_id")

    observed = metadata.get("observed_model_ids")
    observed_models_verified = bool(
        isinstance(observed, list)
        and observed
        and all(isinstance(value, str) and value == actual_model for value in observed)
    )
    if not observed_models_verified:
        errors.append("observed_model_ids do not verify actual_model_id")

    run_status_complete = metadata.get("status") == "COMPLETE"
    real_run_flags_valid = (
        metadata.get("execution_mode") == "REAL_API"
        and metadata.get("is_real_model_run") is True
        and metadata.get("failure") is None
    )
    if not run_status_complete:
        errors.append("runner status is not COMPLETE")
    if not real_run_flags_valid:
        errors.append("runner real-execution flags are invalid")
    host = urlparse(base_url).hostname if isinstance(base_url, str) else None
    return {
        "passed": not errors,
        "errors": errors,
        "provider": provider,
        "endpoint_host": host,
        "expected_directory": expected_directory,
        "expected_requested_model": expected_requested,
        "trusted_endpoint": trusted_endpoint,
        "timestamp_valid": timestamp_valid,
        "model_config_valid": model_config_valid,
        "response_model_verified": response_model_verified,
        "observed_models_verified": observed_models_verified,
        "actual_model_id": actual_model,
        "run_status_complete": run_status_complete,
        "real_run_flags_valid": real_run_flags_valid,
        "self_recorded_consistency_only": True,
        "cryptographic_provider_proof": False,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    text = _read_utf8(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line_number > MAX_JSONL_LINES:
            raise ValueError(f"JSONL exceeds {MAX_JSONL_LINES} records")
        if not line.strip():
            raise ValueError(f"blank JSONL record at line {line_number}")
        value = _strict_json_text(line, f"{path.name} line {line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        records.append(value)
    if not records:
        raise ValueError("trajectory JSONL is empty")
    return records


def _last_provider_final_candidate(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    provider_messages = [
        event.get("message")
        for event in events
        if event.get("event") == "message" and event.get("source") == "provider"
    ]
    if not provider_messages:
        return None
    message = provider_messages[-1]
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        value = _strict_json_text(content, "final provider message")
    except Exception:
        return None
    if isinstance(value, Mapping) and isinstance(value.get("final_candidate"), Mapping):
        return value["final_candidate"]
    return None


def check_complete_run_lifecycle(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require an unambiguous start, real activity, and final COMPLETE terminal."""

    errors: list[str] = []
    event_types = [event.get("event") for event in events]
    if not events or event_types[0] != "run_start":
        errors.append("first trajectory event must be run_start")
    if event_types.count("run_start") != 1:
        errors.append("trajectory must contain exactly one run_start")
    if event_types.count("run_complete") != 1:
        errors.append("COMPLETE trajectory must contain exactly one run_complete")
    if "run_failure" in event_types:
        errors.append("COMPLETE trajectory must not contain run_failure")
    if not events or event_types[-1] != "run_complete":
        errors.append("run_complete must be the final event with nothing afterward")
    elif events[-1].get("status") != "COMPLETE":
        errors.append("run_complete.status must be COMPLETE")
    request_indices = [index for index, event in enumerate(events) if event.get("event") == "provider_request"]
    response_indices = [index for index, event in enumerate(events) if event.get("event") == "provider_response"]
    provider_message_indices = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "message"
        and event.get("source") == "provider"
        and isinstance(event.get("message"), Mapping)
        and event["message"].get("role") == "assistant"
    ]
    tool_indices = [index for index, event in enumerate(events) if event.get("event") == "tool_call"]
    if not request_indices or not response_indices or not provider_message_indices or not tool_indices:
        errors.append("trajectory lacks provider request/response/message or tool activity")
    else:
        first_tool = min(tool_indices)
        pre_tool_chain = any(
            request < response < message < first_tool
            for request in request_indices
            for response in response_indices
            for message in provider_message_indices
        )
        if not pre_tool_chain:
            errors.append("trajectory has no ordered provider request/response/message before tool activity")
        if max(provider_message_indices) <= max(tool_indices):
            errors.append("the final provider message does not follow all tool activity")
    return {
        "passed": not errors,
        "errors": errors,
        "first_event": event_types[0] if event_types else None,
        "last_event": event_types[-1] if event_types else None,
        "terminal_event_count": event_types.count("run_complete") + event_types.count("run_failure"),
    }


def check_run_attestation(
    metadata: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Cross-check self-recorded source/dependency attestations across artifacts."""

    errors: list[str] = []
    integrity = metadata.get("integrity")
    hashes = metadata.get("hashes")
    if not isinstance(integrity, Mapping):
        errors.append("metadata.integrity is missing")
        integrity = {}
    start_snapshot = integrity.get("start_snapshot")
    end_snapshot = integrity.get("end_snapshot")
    if (
        integrity.get("verified") is not True
        or integrity.get("changed") != []
        or integrity.get("error") is not None
        or not isinstance(start_snapshot, Mapping)
        or not isinstance(end_snapshot, Mapping)
        or hashes != start_snapshot
        or start_snapshot != end_snapshot
    ):
        errors.append("source/config integrity snapshots are absent, changed, or inconsistent")
    start_event = events[0] if events else {}
    terminal_event = events[-1] if events else {}
    model_config = metadata.get("model_config")
    configured_model = (
        model_config.get("model") if isinstance(model_config, Mapping) else None
    )
    requested_model = metadata.get("requested_model")
    if requested_model is None:
        requested_model = configured_model
    if (
        start_event.get("event") != "run_start"
        or start_event.get("status") != "RUNNING"
        or start_event.get("task_id") != metadata.get("task_id")
        or start_event.get("provider") != metadata.get("provider")
        or start_event.get("transport") != metadata.get("transport")
        or start_event.get("model") != requested_model
        or start_event.get("requested_model") != requested_model
        or start_event.get("endpoint") != metadata.get("endpoint")
        or start_event.get("request_endpoint") != metadata.get("request_endpoint")
        or start_event.get("execution_mode") != metadata.get("execution_mode")
        or start_event.get("dependency_mode") != metadata.get("dependency_mode")
        or start_event.get("canonical_dependencies") != metadata.get("canonical_dependencies")
        or start_event.get("dependency_provenance") != metadata.get("dependency_provenance")
        or start_event.get("integrity_start_snapshot") != start_snapshot
    ):
        errors.append("run_start attestation does not match metadata")
    if (
        terminal_event.get("event") != "run_complete"
        or terminal_event.get("status") != "COMPLETE"
        or terminal_event.get("terminal_status") != "COMPLETE"
        or terminal_event.get("dependency_mode") != metadata.get("dependency_mode")
        or terminal_event.get("canonical_dependencies") != metadata.get("canonical_dependencies")
        or terminal_event.get("integrity_verified") is not True
    ):
        errors.append("run_complete attestation does not match metadata")
    return {
        "passed": not errors,
        "errors": errors,
        "integrity_verified": integrity.get("verified"),
        "self_recorded_consistency_only": True,
        "cryptographic_proof": False,
    }


def _is_successful_tool_event(event: Mapping[str, Any]) -> bool:
    result = event.get("result")
    return bool(
        event.get("event") == "tool_call"
        and event.get("status") == "SUCCESS"
        and isinstance(result, Mapping)
        and result.get("ok") is True
    )


def assess_closed_loop_rounds(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require two request-anchored provider/tool feedback rounds."""

    errors: list[str] = []
    request_entries = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "provider_request"
    ]
    request_turns = [event.get("turn") for _, event in request_entries]
    request_turns_valid = bool(
        request_turns
        and all(type(turn) is int and turn > 0 for turn in request_turns)
    )
    monotonic_request_turns = bool(
        request_turns_valid
        and all(
            later >= earlier
            for earlier, later in zip(request_turns, request_turns[1:])
        )
    )
    distinct_request_turns = (
        sorted(set(request_turns)) if request_turns_valid else []
    )
    if not request_turns_valid:
        errors.append("provider_request events must have positive integer turn values")
    if request_turns_valid and not monotonic_request_turns:
        errors.append("provider_request turns are not monotonic")
    if len(distinct_request_turns) < 2:
        errors.append("fewer than two distinct provider_request turns were recorded")

    paired_tool_rounds: list[int] = []
    feedback_verified_rounds: list[int] = []
    candidates_by_round: dict[int, set[str]] = {}
    prior_linked_call_ids: set[str] = set()
    for position, (request_index, request) in enumerate(request_entries):
        turn = request.get("turn")
        if type(turn) is not int or turn <= 0:
            continue
        segment_end = (
            request_entries[position + 1][0]
            if position + 1 < len(request_entries)
            else len(events)
        )
        response_entries = [
            (index, events[index])
            for index in range(request_index + 1, segment_end)
            if events[index].get("event") == "provider_response"
        ]
        mismatched_response_turns = [
            response.get("turn")
            for _, response in response_entries
            if response.get("turn") != turn
        ]
        if mismatched_response_turns:
            errors.append(
                f"provider_request turn {turn} contains mismatched provider_response turns: "
                f"{mismatched_response_turns}"
            )
        matching_responses = [
            (index, response)
            for index, response in response_entries
            if response.get("turn") == turn
        ]
        if not matching_responses:
            continue
        if len(matching_responses) != 1:
            errors.append(
                f"provider_request turn {turn} has multiple matching provider_response events"
            )
            continue
        response_index = matching_responses[0][0]
        assistant_entries = [
            (index, events[index])
            for index in range(response_index + 1, segment_end)
            if events[index].get("event") == "message"
            and events[index].get("source") == "provider"
            and isinstance(events[index].get("message"), Mapping)
            and events[index]["message"].get("role") == "assistant"
        ]
        segment_tool_entries = [
            (index, events[index])
            for index in range(request_index + 1, segment_end)
            if events[index].get("event") == "tool_call"
        ]
        if not assistant_entries:
            if segment_tool_entries:
                errors.append(
                    f"provider_request turn {turn} has tool activity without an ordered provider assistant message"
                )
            continue
        if len(assistant_entries) != 1:
            errors.append(
                f"provider_request turn {turn} has multiple provider assistant messages"
            )
            continue
        assistant_index, assistant_event = assistant_entries[0]
        if any(index < assistant_index for index, _ in segment_tool_entries):
            errors.append(
                f"provider_request turn {turn} has tool activity before its provider assistant message"
            )
        assistant_message = assistant_event["message"]
        declared_tool_calls = assistant_message.get("tool_calls")
        declared_call_ids = {
            call.get("id")
            for call in declared_tool_calls or []
            if isinstance(call, Mapping)
            and isinstance(call.get("id"), str)
            and call.get("id")
        } if isinstance(declared_tool_calls, list) else set()
        after_assistant_tools = [
            (index, event)
            for index, event in segment_tool_entries
            if index > assistant_index
        ]
        unlinked_tools = [
            event.get("call_id")
            for _, event in after_assistant_tools
            if event.get("call_id") not in declared_call_ids
        ]
        if unlinked_tools:
            errors.append(
                f"provider_request turn {turn} has tool events not declared by its assistant message"
            )
        linked_tools = [
            event
            for _, event in after_assistant_tools
            if isinstance(event.get("call_id"), str)
            and event.get("call_id") in declared_call_ids
        ]
        successful_linked_tools = [
            event for event in linked_tools if _is_successful_tool_event(event)
        ]
        if not successful_linked_tools:
            continue
        if turn not in paired_tool_rounds:
            paired_tool_rounds.append(turn)

        request_messages = request.get("messages")
        prior_feedback_messages = [
            event.get("message")
            for event in events[:request_index]
            if event.get("event") == "message"
            and event.get("source") == "rdkit_tool"
            and isinstance(event.get("message"), Mapping)
            and event["message"].get("role") == "tool"
            and event["message"].get("tool_call_id") in prior_linked_call_ids
        ]
        feedback_verified = not prior_linked_call_ids or bool(
            isinstance(request_messages, list)
            and any(
                isinstance(message, Mapping)
                and any(message == prior for prior in prior_feedback_messages)
                for message in request_messages
            )
        )
        if feedback_verified:
            feedback_verified_rounds.append(turn)
        else:
            errors.append(
                f"provider_request turn {turn} does not carry a prior recorded tool-role feedback message"
            )

        candidates = candidates_by_round.setdefault(turn, set())
        for event in successful_linked_tools:
            result = event.get("result")
            if (
                event.get("tool") == "evaluate_candidate"
                and isinstance(result, Mapping)
                and result.get("valid") is True
                and isinstance(result.get("canonical_smiles"), str)
                and result.get("canonical_smiles")
            ):
                candidates.add(result["canonical_smiles"])
        prior_linked_call_ids.update(
            event["call_id"]
            for event in successful_linked_tools
            if isinstance(event.get("call_id"), str)
        )
    novel_feedback_rounds: list[int] = []
    seen_candidates: set[str] = set()
    for turn in paired_tool_rounds:
        novel = candidates_by_round[turn] - seen_candidates
        if novel:
            novel_feedback_rounds.append(turn)
            seen_candidates.update(novel)
    if len(paired_tool_rounds) < 2:
        errors.append(
            "real run has fewer than two paired request/response/assistant/tool rounds"
        )
    if len(novel_feedback_rounds) < 2:
        errors.append("new successful canonical candidates do not span two feedback rounds")
    return {
        "passed": not errors,
        "errors": errors,
        "provider_request_turns": request_turns,
        "provider_request_turns_monotonic": monotonic_request_turns,
        "paired_tool_rounds": paired_tool_rounds,
        "feedback_verified_rounds": feedback_verified_rounds,
        "closed_loop_tool_round_count": len(paired_tool_rounds),
        "candidate_feedback_round_count": len(novel_feedback_rounds),
        "candidate_rounds": {
            str(turn): sorted(values) for turn, values in sorted(candidates_by_round.items())
        },
        "self_recorded_process_consistency_only": True,
    }


def _expected_model_transition_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "transition_id": "qed-final-model-transition-20260718",
        "provider": "siliconflow",
        "endpoint": SILICONFLOW_ENDPOINT,
        "preserved_trajectory": {
            "directory": "trajectories/kimi",
            "model": FROZEN_KIMI_MODEL,
            "artifacts_sha256": dict(FROZEN_KIMI_ARTIFACT_SHA256),
        },
        "model_replacements": {
            name: {"previous": values[0], "current": values[1]}
            for name, values in APPROVED_MODEL_TRANSITIONS.items()
        },
        "approved_source_transitions": {
            path: {"previous_sha256": values[0], "current_sha256": values[1]}
            for path, values in APPROVED_KIMI_SOURCE_TRANSITIONS.items()
        },
    }


def _approved_kimi_historical_hashes(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    expected_directory: str | None,
) -> tuple[dict[str, str], list[str]]:
    """Prove the frozen Kimi evidence predates only the two approved model-ID edits."""

    if expected_directory != "kimi":
        return {}, []
    identity_fields = (
        metadata.get("provider") == "siliconflow",
        metadata.get("model") == FROZEN_KIMI_MODEL,
        metadata.get("requested_model") == FROZEN_KIMI_MODEL,
        metadata.get("requested_model_id") == FROZEN_KIMI_MODEL,
        metadata.get("actual_model_id") == FROZEN_KIMI_MODEL,
        metadata.get("endpoint") == SILICONFLOW_ENDPOINT,
        metadata.get("request_endpoint") == f"{SILICONFLOW_ENDPOINT}/chat/completions",
    )
    if not all(identity_fields):
        return {}, ["frozen Kimi transition identity/endpoint mismatch"]

    try:
        manifest = _strict_json_file(root / MODEL_TRANSITION_FILE)
    except Exception as exc:
        return {}, [f"model transition manifest cannot be verified: {type(exc).__name__}: {exc}"]
    if manifest != _expected_model_transition_manifest():
        return {}, ["model transition manifest differs from the exact approved transition"]

    errors: list[str] = []
    kimi_root = root / "trajectories" / "kimi"
    for filename, expected_hash in FROZEN_KIMI_ARTIFACT_SHA256.items():
        target = kimi_root / filename
        if not target.is_file() or sha256_file(target) != expected_hash:
            errors.append(f"frozen Kimi artifact hash mismatch: {filename}")

    hashes = metadata.get("hashes")
    sources = hashes.get("sources") if isinstance(hashes, Mapping) else None
    configs = hashes.get("configs") if isinstance(hashes, Mapping) else None
    if not isinstance(sources, Mapping) or not isinstance(configs, Mapping):
        return {}, errors + ["frozen Kimi transition requires source and config hashes"]
    model_config_historical = APPROVED_KIMI_SOURCE_TRANSITIONS[
        "configs/models.example.yaml"
    ][0]
    if configs.get("model_config_source_file") != model_config_historical:
        errors.append("frozen Kimi model-config source hash does not match its historical evidence")

    approved: dict[str, str] = {}
    reversible = tuple(
        (current, previous)
        for name, (previous, current) in APPROVED_MODEL_TRANSITIONS.items()
        if name != "kimi"
    )
    for relative, (historical_hash, current_hash) in APPROVED_KIMI_SOURCE_TRANSITIONS.items():
        target = root / Path(*PurePosixPath(relative).parts)
        if not target.is_file() or sha256_file(target) != current_hash:
            errors.append(f"approved transition current hash mismatch: {relative}")
            continue
        text = _read_utf8(target)
        reconstructed = text
        token_error = False
        for current_token, historical_token in reversible:
            if text.count(current_token) != 1 or historical_token in text:
                errors.append(f"approved transition token occurrence mismatch: {relative}")
                token_error = True
            reconstructed = reconstructed.replace(current_token, historical_token)
        reconstructed_hash = _sha256_text(reconstructed)
        if token_error or reconstructed_hash != historical_hash:
            errors.append(f"approved transition is not byte-reversible: {relative}")
            continue
        if sources.get(relative) != historical_hash:
            errors.append(f"frozen Kimi metadata does not attest the historical source: {relative}")
            continue
        approved[relative] = historical_hash
    if errors or set(approved) != set(APPROVED_KIMI_SOURCE_TRANSITIONS):
        return {}, errors
    return approved, []


def _check_source_hashes(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    expected_directory: str | None = None,
) -> list[str]:
    errors: list[str] = []
    approved_historical, transition_errors = _approved_kimi_historical_hashes(
        root, metadata, expected_directory=expected_directory
    )
    errors.extend(transition_errors)
    hashes = metadata.get("hashes")
    if not isinstance(hashes, Mapping):
        return ["metadata.hashes is missing"]
    sources = hashes.get("sources")
    configs = hashes.get("configs")
    if not isinstance(sources, Mapping):
        errors.append("metadata.hashes.sources is missing")
    else:
        missing = sorted(REQUIRED_HASHED_SOURCES - set(sources))
        if missing:
            errors.append(f"required source hashes are missing: {missing}")
        for relative, expected in sources.items():
            if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                errors.append(f"invalid source hash entry: {relative!r}")
                continue
            pure = PurePosixPath(relative)
            target = (root / Path(*pure.parts)).resolve()
            if pure.is_absolute() or ".." in pure.parts or not _is_relative_to(target, root.resolve()) or not target.is_file():
                errors.append(f"hashed source path is missing or unsafe: {relative}")
            elif sha256_file(target) != expected and approved_historical.get(relative) != expected:
                errors.append(f"source hash mismatch: {relative}")
    if not isinstance(configs, Mapping):
        errors.append("metadata.hashes.configs is missing")
        return errors
    try:
        task = yaml.safe_load(_read_utf8(root / "configs" / "task.yaml"))
        prompt = _read_utf8(root / "prompts" / "agent_prompt.md")
        model = metadata.get("model_config")
        expected_config_hashes = {
            "task_config": _sha256_text(_canonical_json(task)),
            "model_config_sanitized": _sha256_text(_canonical_json(model)),
            "agent_prompt": _sha256_text(prompt),
        }
        for name, expected in expected_config_hashes.items():
            if configs.get(name) != expected:
                errors.append(f"config/source evidence hash mismatch: {name}")
    except Exception as exc:
        errors.append(f"could not reproduce config hashes: {type(exc).__name__}: {exc}")
    return errors


def assess_process_eligibility(
    *,
    metadata: Mapping[str, Any],
    final_answer: Mapping[str, Any],
    stored_evaluation: Mapping[str, Any],
    accepted_tool_events: Sequence[Mapping[str, Any]],
    successful_tool_events: Sequence[Mapping[str, Any]],
    minimum_unique_candidates: int,
) -> dict[str, Any]:
    """Apply READY process gates without requiring chemical score eligibility."""

    errors: list[str] = []
    run_metadata = final_answer.get("run_metadata")
    process_evidence = stored_evaluation.get("process_evidence")
    process_checks = stored_evaluation.get("process_checks")
    canonical_candidates = {
        event["result"].get("canonical_smiles")
        for event in successful_tool_events
        if event.get("tool") == "evaluate_candidate"
        and isinstance(event.get("result"), Mapping)
        and event["result"].get("ok") is True
        and event["result"].get("valid") is True
        and isinstance(event["result"].get("canonical_smiles"), str)
        and event["result"].get("canonical_smiles")
    }
    listed_candidates: set[str] | None = None
    if not isinstance(run_metadata, Mapping):
        errors.append("final_answer.run_metadata is missing")
    else:
        if run_metadata.get("budget_exceeded") is not False:
            errors.append("final_answer.run_metadata.budget_exceeded must be false")
        if run_metadata.get("tool_call_count") != len(accepted_tool_events):
            errors.append("final_answer tool count does not match accepted trajectory events")
        listed = run_metadata.get("unique_evaluated_candidates")
        if isinstance(listed, list) and all(isinstance(value, str) for value in listed):
            listed_candidates = set(listed)
        if listed_candidates != canonical_candidates:
            errors.append("final_answer unique canonical candidates do not match successful events")
    if metadata.get("budget_exceeded") is not False:
        errors.append("metadata.budget_exceeded must be false")
    if len(canonical_candidates) < minimum_unique_candidates:
        errors.append(
            f"only {len(canonical_candidates)} unique successful canonical candidates; "
            f"at least {minimum_unique_candidates} required"
        )
    if (
        stored_evaluation.get("status") != "ok"
        or stored_evaluation.get("valid_submission") is not True
        or stored_evaluation.get("process_eligible") is not True
    ):
        errors.append("stored evaluation is not a valid process-eligible submission")
    evidence_candidates = process_evidence.get("unique_evaluated_candidates") if isinstance(process_evidence, Mapping) else None
    evidence_candidate_set = (
        set(evidence_candidates)
        if isinstance(evidence_candidates, list) and all(isinstance(value, str) for value in evidence_candidates)
        else None
    )
    if (
        not isinstance(process_evidence, Mapping)
        or process_evidence.get("valid") is not True
        or process_evidence.get("budget_exceeded") is not False
        or process_evidence.get("tool_call_count") != len(accepted_tool_events)
        or evidence_candidate_set != canonical_candidates
    ):
        errors.append("stored process evidence does not match successful canonical events")
    if (
        not isinstance(process_checks, Mapping)
        or not process_checks
        or not all(value is True for value in process_checks.values())
    ):
        errors.append("stored process checks are incomplete or failed")
    return {
        "passed": not errors,
        "errors": errors,
        "unique_successful_canonical_candidate_count": len(canonical_candidates),
        "minimum_unique_candidates": minimum_unique_candidates,
        "chemical_score_eligibility_required": False,
    }


def _is_finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def check_positive_objective_evidence(
    evaluation: Mapping[str, Any], task_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a strict measured QED improvement and a positive gated score."""

    errors: list[str] = []
    task = task_config.get("task") if isinstance(task_config, Mapping) else None
    objective = task.get("objective") if isinstance(task, Mapping) else None
    strict_improvement_required = bool(
        isinstance(objective, Mapping)
        and objective.get("require_strict_improvement") is True
    )
    if not strict_improvement_required:
        errors.append("task.objective.require_strict_improvement must be true")

    required_true = (
        "valid_submission",
        "eligible",
        "hard_constraints_passed",
        "objective_passed",
        "process_eligible",
    )
    if evaluation.get("status") != "ok":
        errors.append("evaluation.status must be 'ok'")
    for name in required_true:
        if evaluation.get(name) is not True:
            errors.append(f"evaluation.{name} must be true")
    if evaluation.get("errors") != []:
        errors.append("evaluation.errors must be an empty list")

    score = evaluation.get("score")
    total_score = evaluation.get("total_score")
    positive_score = bool(
        _is_finite_number(score)
        and _is_finite_number(total_score)
        and float(score) > 0.0
        and float(total_score) > 0.0
        and math.isclose(float(score), float(total_score), rel_tol=0.0, abs_tol=1e-9)
    )
    if not positive_score:
        errors.append("evaluation score and total_score must be equal, finite, and positive")

    metrics = evaluation.get("metrics")
    start_qed = metrics.get("start_qed") if isinstance(metrics, Mapping) else None
    final_qed = metrics.get("final_qed") if isinstance(metrics, Mapping) else None
    qed_delta = metrics.get("qed_delta") if isinstance(metrics, Mapping) else None
    strict_delta = bool(
        _is_finite_number(start_qed)
        and _is_finite_number(final_qed)
        and _is_finite_number(qed_delta)
        and float(final_qed) > float(start_qed)
        and float(qed_delta) > 0.0
        and math.isclose(
            float(qed_delta),
            float(final_qed) - float(start_qed),
            rel_tol=0.0,
            abs_tol=1e-5,
        )
    )
    if not strict_delta:
        errors.append("evaluation metrics do not prove a positive strict QED improvement")

    breakdown = evaluation.get("score_breakdown")
    improvement_score = (
        breakdown.get("qed_improvement") if isinstance(breakdown, Mapping) else None
    )
    positive_improvement_score = bool(
        _is_finite_number(improvement_score) and float(improvement_score) > 0.0
    )
    if not positive_improvement_score:
        errors.append("score_breakdown.qed_improvement must be finite and positive")

    return {
        "passed": not errors,
        "errors": errors,
        "strict_improvement_required": strict_improvement_required,
        "strict_qed_improvement": strict_delta,
        "positive_improvement_score": positive_improvement_score,
        "positive_total_score": positive_score,
        "start_qed": start_qed,
        "final_qed": final_qed,
        "qed_delta": qed_delta,
        "score": score,
    }


def check_dependency_consistency(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Check self-recorded default dependency provenance; this is not cryptographic proof."""

    errors: list[str] = []
    provenance = metadata.get("dependency_provenance")
    if metadata.get("dependency_mode") != "CANONICAL":
        errors.append("metadata.dependency_mode is not CANONICAL")
    if metadata.get("canonical_dependencies") is not True:
        errors.append("metadata.canonical_dependencies is not true")
    if not isinstance(provenance, Mapping):
        errors.append("metadata.dependency_provenance is missing")
        provenance = {}
    for name, identifier in CANONICAL_DEPENDENCY_IDENTIFIERS.items():
        item = provenance.get(name)
        required_attestation = CANONICAL_DEPENDENCY_ATTESTATIONS[name]
        if not isinstance(item, Mapping):
            errors.append(f"dependency provenance is missing {name}")
            continue
        if (
            item.get("mode") != "default"
            or item.get("canonical") is not True
            or item.get("identifier") != identifier
            or item.get("attestation") != required_attestation
        ):
            errors.append(
                f"dependency {name} is not the strongly attested canonical non-injected default"
            )
    return {
        "passed": not errors,
        "errors": errors,
        "self_recorded_consistency_only": True,
        "cryptographic_provider_proof": False,
    }


def check_trajectories(root: Path) -> dict[str, Any]:
    trajectory_root = root / "trajectories"
    results: dict[str, Any] = {}
    task_config: Mapping[str, Any] = {}
    try:
        loaded_task_config = yaml.safe_load(_read_utf8(root / "configs" / "task.yaml"))
        if isinstance(loaded_task_config, Mapping):
            task_config = loaded_task_config
        minimum_tool_calls = int(((task_config or {}).get("budget") or {}).get("min_tool_calls_for_eligibility", 1))
        minimum_unique_candidates = int(
            ((task_config or {}).get("budget") or {}).get(
                "min_unique_evaluated_candidates_for_eligibility", 1
            )
        )
    except Exception:
        minimum_tool_calls = 1
        minimum_unique_candidates = 1
    directories = (
        sorted(
            (item for item in trajectory_root.iterdir() if item.is_dir()),
            key=lambda item: item.name.casefold(),
        )
        if trajectory_root.is_dir()
        else []
    )
    actual_directories = {directory.name for directory in directories}
    expected_directories = set(EXPECTED_TRAJECTORY_MODELS)
    missing_directories = sorted(expected_directories - actual_directories)
    unexpected_directories = sorted(actual_directories - expected_directories)
    passing_identities: list[str] = []
    for directory in directories:
        errors: list[str] = []
        entries = list(directory.iterdir())
        present = {path.name for path in entries if path.is_file()}
        missing = sorted(REQUIRED_TRAJECTORY_FILES - present)
        unexpected = sorted(
            path.name
            for path in entries
            if not path.is_file() or path.name not in REQUIRED_TRAJECTORY_FILES
        )
        if missing:
            errors.append(f"missing required artifacts: {missing}")
        if unexpected:
            errors.append(f"unexpected trajectory artifacts: {unexpected}")
        metadata: Mapping[str, Any] = {}
        final_answer: Mapping[str, Any] = {}
        stored_evaluation: Mapping[str, Any] = {}
        events: list[dict[str, Any]] = []
        if not missing:
            try:
                loaded_metadata = _strict_json_file(directory / "metadata.json")
                loaded_final = _strict_json_file(directory / "final_answer.json")
                loaded_evaluation = _strict_json_file(directory / "evaluation.json")
                if not all(isinstance(value, dict) for value in (loaded_metadata, loaded_final, loaded_evaluation)):
                    raise ValueError("metadata, final answer, and evaluation must be objects")
                metadata = loaded_metadata
                final_answer = loaded_final
                stored_evaluation = loaded_evaluation
                events = _load_jsonl(directory / "trajectory.jsonl")
            except Exception as exc:
                errors.append(f"artifact parse failed: {type(exc).__name__}: {exc}")

        counts = metadata.get("counts") if isinstance(metadata, Mapping) else None
        response_metadata = metadata.get("provider_response_metadata") if isinstance(metadata, Mapping) else None
        endpoint_host: str | None = None
        actual_model: str | None = None
        identity_check: Mapping[str, Any] = {}
        accepted_tool_events = [
            event for event in events if event.get("event") == "tool_call" and event.get("status") != "REJECTED"
        ]
        successful_tool_events = [
            event
            for event in accepted_tool_events
            if _is_successful_tool_event(event)
        ]
        provider_requests = [event for event in events if event.get("event") == "provider_request"]
        provider_responses = [event for event in events if event.get("event") == "provider_response"]
        if metadata:
            if metadata.get("status") != "COMPLETE":
                errors.append("metadata.status is not COMPLETE")
            if metadata.get("execution_mode") != "REAL_API" or metadata.get("is_real_model_run") is not True:
                errors.append("metadata is inconsistent with a REAL_API run")
            if metadata.get("failure") is not None:
                errors.append("metadata contains a failure")
            identity_check = check_model_identity_consistency(
                metadata, expected_directory=directory.name
            )
            errors.extend(identity_check.get("errors", []))
            endpoint_host = identity_check.get("endpoint_host")
            identity_value = identity_check.get("actual_model_id")
            actual_model = identity_value if isinstance(identity_value, str) else None
            if not isinstance(response_metadata, list) or not response_metadata:
                errors.append("provider_response_metadata is empty")
            if not isinstance(counts, Mapping):
                errors.append("metadata.counts is missing")
            else:
                if counts.get("tool_calls") != len(accepted_tool_events):
                    errors.append("metadata tool count does not match JSONL events")
                if counts.get("provider_responses") != len(response_metadata or []):
                    errors.append("metadata provider response count is inconsistent")
        dependency_check = check_dependency_consistency(metadata)
        errors.extend(dependency_check["errors"])
        sequences = [event.get("sequence") for event in events]
        if sequences and sequences != list(range(1, len(events) + 1)):
            errors.append("trajectory sequence numbers are not contiguous from 1")
        lifecycle = check_complete_run_lifecycle(events)
        errors.extend(lifecycle["errors"])
        run_attestation = check_run_attestation(metadata, events)
        errors.extend(run_attestation["errors"])
        closed_loop = assess_closed_loop_rounds(events)
        errors.extend(closed_loop["errors"])
        if not provider_requests or not provider_responses:
            errors.append("trajectory lacks provider request/response events")
        if len(accepted_tool_events) < minimum_tool_calls:
            errors.append(
                f"trajectory has {len(accepted_tool_events)} tool events; at least {minimum_tool_calls} required"
            )
        process_gate = assess_process_eligibility(
            metadata=metadata,
            final_answer=final_answer,
            stored_evaluation=stored_evaluation,
            accepted_tool_events=accepted_tool_events,
            successful_tool_events=successful_tool_events,
            minimum_unique_candidates=minimum_unique_candidates,
        )
        errors.extend(process_gate["errors"])
        objective_gate = check_positive_objective_evidence(
            stored_evaluation, task_config
        )
        errors.extend(objective_gate["errors"])
        final_candidate = final_answer.get("final_candidate") if isinstance(final_answer, Mapping) else None
        run_metadata = final_answer.get("run_metadata") if isinstance(final_answer, Mapping) else None
        if not isinstance(final_candidate, Mapping) or not isinstance(run_metadata, Mapping):
            errors.append("final_answer lacks final_candidate/run_metadata")
        else:
            if run_metadata.get("tool_call_count") != len(accepted_tool_events):
                errors.append("final_answer tool count does not match JSONL")
            if run_metadata.get("final_candidate_evaluated") is not True:
                errors.append("final answer does not claim measured final candidate")
            process_candidate = _last_provider_final_candidate(events)
            if process_candidate is None or dict(process_candidate) != dict(final_candidate):
                errors.append("final_answer candidate differs from the final provider process message")
            final_smiles = final_candidate.get("smiles")
            evaluated_final = stored_evaluation.get("final_candidate") if isinstance(stored_evaluation, Mapping) else None
            final_canonical = evaluated_final.get("canonical_smiles") if isinstance(evaluated_final, Mapping) else None
            unique_evaluated = run_metadata.get("unique_evaluated_candidates")
            canonical_results = {
                event["result"].get("canonical_smiles")
                for event in successful_tool_events
                if event.get("tool") == "evaluate_candidate"
                and isinstance(event.get("result"), Mapping)
                and event["result"].get("ok") is True
                and event["result"].get("valid") is True
                and isinstance(event["result"].get("canonical_smiles"), str)
            }
            measured = any(
                event.get("tool") == "evaluate_candidate"
                and isinstance(event.get("arguments"), Mapping)
                and event["arguments"].get("smiles") == final_smiles
                for event in successful_tool_events
            ) or bool(
                isinstance(final_canonical, str)
                and final_canonical in canonical_results
                and isinstance(unique_evaluated, list)
                and final_canonical in unique_evaluated
            )
            if not measured:
                errors.append("no successful evaluate_candidate event canonically measures the final molecule")
            try:
                if not isinstance(final_smiles, str) or final_smiles not in _read_utf8(directory / "dialogue.md"):
                    errors.append("dialogue.md does not contain the final SMILES")
            except Exception as exc:
                errors.append(f"dialogue.md cannot be checked: {exc}")
        errors.extend(
            _check_source_hashes(
                root, metadata, expected_directory=directory.name
            )
            if metadata
            else []
        )
        passed = not errors
        if passed and actual_model:
            passing_identities.append(actual_model.casefold())
        results[directory.name] = {
            "passed": passed,
            "errors": errors,
            "actual_model": actual_model,
            "endpoint_host": endpoint_host,
            "event_count": len(events),
            "tool_event_count": len(accepted_tool_events),
            "successful_tool_event_count": len(successful_tool_events),
            "directory": directory.relative_to(root).as_posix(),
            "identity_consistency": dict(identity_check),
            "process_eligibility_gate": process_gate,
            "positive_objective_gate": objective_gate,
            "lifecycle": lifecycle,
            "run_attestation": run_attestation,
            "closed_loop": closed_loop,
            "dependency_consistency": dependency_check,
        }
    passing_count = sum(
        1
        for name in expected_directories
        if name in results and results[name]["passed"]
    )
    distinct_count = len(set(passing_identities))
    errors: list[str] = []
    if missing_directories:
        errors.append(f"missing required trajectory directories: {missing_directories}")
    if unexpected_directories:
        errors.append(f"unexpected trajectory directories: {unexpected_directories}")
    if passing_count != len(expected_directories):
        errors.append(
            f"only {passing_count} exact mapped trajectories passed; "
            f"{len(expected_directories)} required"
        )
    if distinct_count != len(expected_directories):
        errors.append(
            f"only {distinct_count} distinct compatible actual model identifiers passed; "
            f"{len(expected_directories)} required"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "expected_directory_models": dict(EXPECTED_TRAJECTORY_MODELS),
        "missing_directories": missing_directories,
        "unexpected_directories": unexpected_directories,
        "passing_trajectory_count": passing_count,
        "distinct_actual_model_count": distinct_count,
        "minimum_tool_calls": minimum_tool_calls,
        "minimum_unique_evaluated_candidates": minimum_unique_candidates,
        "models": results,
    }


def derive_static_delivery_status(
    structure: Mapping[str, Any], trajectories: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "required_structure_and_reports": bool(structure.get("passed")),
        "project_license": bool(structure.get("project_license_present")),
        "project_notice": bool(structure.get("project_notice_present")),
        "test_files": bool(structure.get("test_files")),
        "test_report": "reports/test_report.md" not in set(structure.get("missing_reports", [])),
        "three_trusted_real_trajectories": bool(trajectories.get("passed")),
    }
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "delivery_status": "READY_FOR_REVIEW" if not blocking else "BLOCKED",
        "blocking_reasons": blocking,
        "checks": checks,
    }


def _minimal_environment(temp_root: Path, executable_directories: Sequence[Path] = ()) -> tuple[dict[str, str], list[str]]:
    stripped = sorted(name for name in os.environ if _SENSITIVE_ENV_NAME.search(name))
    environment: dict[str, str] = {}
    for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "OS", "NUMBER_OF_PROCESSORS"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    directories: list[Path] = [Path(sys.executable).resolve().parent]
    directories.extend(executable_directories)
    for candidate in (
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "DLLs",
        Path(sys.prefix) / "Scripts",
    ):
        if candidate.is_dir():
            directories.append(candidate.resolve())
    system_root = Path(environment.get("SYSTEMROOT", environment.get("WINDIR", "")))
    if str(system_root) and system_root.is_dir():
        directories.append(system_root / "System32")
    unique: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        value = str(directory)
        key = value.casefold()
        if key not in seen and directory.is_dir():
            seen.add(key)
            unique.append(value)
    home = temp_root / "home"
    appdata = temp_root / "appdata"
    localappdata = temp_root / "localappdata"
    for directory in (temp_root, home, appdata, localappdata):
        directory.mkdir(parents=True, exist_ok=True)
    invalid_proxy = "http://127.0.0.1:9"
    environment.update(
        {
            "PATH": os.pathsep.join(unique),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "HOME": str(home),
            "USERPROFILE": str(home),
            "APPDATA": str(appdata),
            "LOCALAPPDATA": str(localappdata),
            "HTTP_PROXY": invalid_proxy,
            "HTTPS_PROXY": invalid_proxy,
            "ALL_PROXY": invalid_proxy,
            "http_proxy": invalid_proxy,
            "https_proxy": invalid_proxy,
            "all_proxy": invalid_proxy,
            "NO_PROXY": "",
            "no_proxy": "",
            "PIP_NO_INDEX": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    # Keep user-site auto-discovery disabled, but explicitly expose the one
    # trusted dependency location needed by this benchmark when RDKit happens
    # to be installed there rather than in the interpreter prefix.
    rdkit_spec = importlib.util.find_spec("rdkit")
    if rdkit_spec is not None:
        locations = list(rdkit_spec.submodule_search_locations or [])
        if locations:
            environment["PYTHONPATH"] = str(Path(locations[0]).resolve().parent)
    return environment, stripped


def run_command(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    *,
    timeout: int = 600,
) -> dict[str, Any]:
    display = [str(item) for item in command]
    try:
        completed = subprocess.run(
            display,
            cwd=cwd,
            env=dict(environment),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return {
            "passed": completed.returncode == 0,
            "command": display,
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-8000:],
            "stderr_tail": completed.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "command": display,
            "cwd": str(cwd),
            "exit_code": None,
            "error": f"command exceeded {timeout} seconds",
            "stdout_tail": str(exc.stdout or "")[-8000:],
            "stderr_tail": str(exc.stderr or "")[-8000:],
        }
    except OSError as exc:
        return {
            "passed": False,
            "command": display,
            "cwd": str(cwd),
            "exit_code": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _import_command() -> list[str]:
    code = (
        "import importlib,pathlib; "
        "r=pathlib.Path('.').resolve(); mods=[]; "
        "roots=('baselines','evaluator','framework','scripts','tools'); "
        "files=[p for d in roots for p in (r/d).rglob('*.py')]; "
        "mods=[('.'.join(p.relative_to(r).with_suffix('').parts[:-1]) if p.stem=='__init__' else '.'.join(p.relative_to(r).with_suffix('').parts)) for p in files]; "
        "[importlib.import_module(m) for m in sorted(set(mods)) if m]; "
        "print('imported',len(set(mods)),'modules')"
    )
    return [sys.executable, "-c", code]


def run_dynamic_validation(root: Path, trajectory_check: Mapping[str, Any]) -> dict[str, Any]:
    # Keep all temporary state within the caller's permanent extraction while
    # remaining outside the project root.  Several confinement tests need a
    # genuinely external fixture, and putting TEMP inside ``root`` weakens that
    # check even though it is still local to the extracted package.
    work = root.parent / ".validation_work"
    work.mkdir(parents=True, exist_ok=False)
    temp_root = work / "temp"
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    shell = shutil.which("sh")
    executable_dirs = [Path(value).resolve().parent for value in (powershell, shell) if value]
    environment, stripped = _minimal_environment(temp_root, executable_dirs)
    commands: dict[str, Any] = {}
    commands["complete_import"] = run_command(_import_command(), root, environment)

    baseline_dir = work / "baseline"
    commands["baseline"] = run_command(
        [sys.executable, "baselines/baseline_agent.py", "--config", "configs/task.yaml", "--output", str(baseline_dir)],
        root,
        environment,
    )
    baseline_submission = baseline_dir / "final_answer.json"
    baseline_trajectory = baseline_dir / "trajectory.jsonl"
    if baseline_submission.is_file() and baseline_trajectory.is_file():
        commands["baseline_evaluator"] = run_command(
            [
                sys.executable,
                "evaluator/evaluate.py",
                "--submission",
                str(baseline_submission),
                "--trajectory",
                str(baseline_trajectory),
                "--config",
                "configs/task.yaml",
                "--output",
                str(baseline_dir / "evaluation.json"),
            ],
            root,
            environment,
        )
    else:
        commands["baseline_evaluator"] = {
            "passed": False,
            "error": "baseline did not create both final_answer.json and trajectory.jsonl",
            "exit_code": None,
        }
    commands["pytest"] = run_command(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        root,
        environment,
        timeout=900,
    )
    if powershell:
        commands["run_all.ps1"] = run_command(
            [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(root / "run_all.ps1"), "tests"],
            root,
            environment,
            timeout=900,
        )
    else:
        commands["run_all.ps1"] = {
            "passed": True,
            "executed": False,
            "static_check_used": True,
            "exit_code": None,
            "reason": "neither pwsh nor powershell is available; the required static PowerShell check passed before execution",
        }
    if shell:
        commands["run_all.sh"] = run_command(
            [shell, str(root / "run_all.sh"), "tests"], root, environment, timeout=900
        )
    else:
        commands["run_all.sh"] = {
            "passed": True,
            "executed": False,
            "static_check_used": True,
            "reason": "no sh executable is available; the required static shell check passed before execution",
        }

    re_evaluations: dict[str, Any] = {}
    evaluation_dir = work / "trajectory_evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    try:
        loaded_task_config = yaml.safe_load(_read_utf8(root / "configs" / "task.yaml"))
        task_config: Mapping[str, Any] = (
            loaded_task_config if isinstance(loaded_task_config, Mapping) else {}
        )
    except Exception:
        task_config = {}
    model_results = trajectory_check.get("models")
    if isinstance(model_results, Mapping):
        for name, model_result in sorted(model_results.items()):
            if not isinstance(model_result, Mapping) or not model_result.get("passed"):
                continue
            trajectory_dir = root / str(model_result["directory"])
            generated = evaluation_dir / f"{name}.evaluation.json"
            command_result = run_command(
                [
                    sys.executable,
                    "evaluator/evaluate.py",
                    "--submission",
                    str(trajectory_dir / "final_answer.json"),
                    "--trajectory",
                    str(trajectory_dir / "trajectory.jsonl"),
                    "--config",
                    "configs/task.yaml",
                    "--output",
                    str(generated),
                ],
                root,
                environment,
            )
            byte_identical = False
            json_equal = False
            generated_process_eligible = False
            generated_valid_submission = False
            positive_objective_gate: dict[str, Any] = {
                "passed": False,
                "errors": ["trajectory evaluator replay did not produce a checked JSON object"],
            }
            compare_error: str | None = None
            if command_result.get("passed") and generated.is_file():
                stored = trajectory_dir / "evaluation.json"
                byte_identical = generated.read_bytes() == stored.read_bytes()
                try:
                    generated_json = _strict_json_file(generated)
                    stored_json = _strict_json_file(stored)
                    json_equal = generated_json == stored_json
                    generated_process_eligible = bool(
                        isinstance(generated_json, Mapping)
                        and generated_json.get("process_eligible") is True
                    )
                    generated_valid_submission = bool(
                        isinstance(generated_json, Mapping)
                        and generated_json.get("valid_submission") is True
                    )
                    if isinstance(generated_json, Mapping):
                        positive_objective_gate = check_positive_objective_evidence(
                            generated_json, task_config
                        )
                except Exception as exc:
                    compare_error = f"{type(exc).__name__}: {exc}"
            passed = bool(
                command_result.get("passed")
                and json_equal
                and generated_process_eligible
                and generated_valid_submission
                and positive_objective_gate.get("passed")
            )
            re_evaluations[str(name)] = {
                "passed": passed,
                "command_result": command_result,
                "byte_identical": byte_identical,
                "json_object_equal": json_equal,
                "generated_process_eligible": generated_process_eligible,
                "generated_valid_submission": generated_valid_submission,
                "positive_objective_gate": positive_objective_gate,
                "comparison_error": compare_error,
                "policy": "JSON object equality is required; byte identity is also reported.",
            }
    trajectory_re_evaluation_passed = bool(
        set(re_evaluations) == set(EXPECTED_TRAJECTORY_MODELS)
        and all(result.get("passed") for result in re_evaluations.values())
    )
    local_dynamic_checks_passed = all(result.get("passed") for result in commands.values())
    return {
        "passed": local_dynamic_checks_passed and trajectory_re_evaluation_passed,
        "local_dynamic_checks_passed": local_dynamic_checks_passed,
        "commands": commands,
        "trajectory_re_evaluations": re_evaluations,
        "trajectory_re_evaluation_passed": trajectory_re_evaluation_passed,
        "environment_policy": {
            "inherited_sensitive_names_removed": stripped,
            "temp_and_home_within_extraction": True,
            "invalid_loopback_proxies_set": True,
            "os_sandbox": False,
            "warning": "Environment cleaning and invalid proxies reduce accidental access but cannot replace an OS sandbox.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely validate a benchmark ZIP in a permanent extraction directory")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--extract-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace-root", default=str(derived_workspace_root()))
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--execute-trusted-code",
        action="store_true",
        help="After all static checks pass, execute package code with a reduced environment (not an OS sandbox)",
    )
    return parser


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    archive_path = Path(args.archive).expanduser().resolve()
    extract_dir = resolve_write_path(args.extract_dir, workspace_root)
    output_path = resolve_write_path(args.output, workspace_root)
    if _is_relative_to(output_path, extract_dir):
        raise ValueError("Validation report must be outside --extract-dir")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {output_path}")
    if extract_dir.exists() and (
        not extract_dir.is_dir() or _is_reparse_or_symlink(extract_dir) or any(extract_dir.iterdir())
    ):
        report = {
            "schema_version": "2.0",
            "passed": False,
            "archive": str(archive_path),
            "extract_dir": str(extract_dir),
            "fatal_error": f"Refusing to overwrite non-empty or unsafe extract directory: {extract_dir}",
        }
        _write_report(output_path, report)
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 1

    actual_hash = sha256_file(archive_path) if archive_path.is_file() else None
    expected = args.expected_sha256.casefold() if args.expected_sha256 else None
    identity_errors: list[str] = []
    if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
        identity_errors.append("--expected-sha256 must be exactly 64 hexadecimal characters")
    if expected is not None and actual_hash != expected:
        identity_errors.append("archive SHA-256 does not match --expected-sha256")
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "archive_sha256": actual_hash,
        "expected_sha256": expected,
        "workspace_root": str(workspace_root),
        "extract_dir": str(extract_dir),
        "execute_trusted_code_requested": bool(args.execute_trusted_code),
        "security_boundary": {
            "static_checks_before_execution": True,
            "os_sandbox": False,
            "cryptographic_provider_authentication": False,
            "trajectory_metadata_role": "self-recorded consistency evidence",
            "warning": "This validator reduces environment and network exposure but does not provide an OS sandbox.",
        },
        "checks": {
            "archive_identity": {"passed": not identity_errors, "errors": identity_errors}
        },
    }
    root: Path | None = None
    if not identity_errors:
        preflight = preflight_archive(archive_path)
        report["checks"]["archive_preflight"] = preflight
        if preflight.get("passed"):
            try:
                root = extract_archive_safely(archive_path, extract_dir, preflight)
                report["checks"]["permanent_extraction"] = {
                    "passed": True,
                    "root": str(root),
                    "extract_dir_retained": True,
                }
            except Exception as exc:
                report["checks"]["permanent_extraction"] = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    if root is not None:
        structure = check_structure(root)
        manifest = check_manifest(root)
        utf8 = check_utf8(root)
        security = check_security(root)
        absolute_paths = check_absolute_paths(root)
        environment_files = check_environment_files(root)
        python_static = check_python_static(root)
        shell_static = check_shell_scripts_static(root)
        trajectories = check_trajectories(root)
        static_delivery = derive_static_delivery_status(structure, trajectories)
        manifest_status = manifest.get("delivery_status")
        status_integrity = {
            "passed": manifest_status == static_delivery["delivery_status"],
            "manifest_status": manifest_status,
            "independently_derived_static_status": static_delivery["delivery_status"],
            "blocking_reasons": static_delivery["blocking_reasons"],
        }
        report["checks"].update(
            {
                "structure_and_reports": structure,
                "manifest": {key: value for key, value in manifest.items() if key != "manifest"},
                "utf8_and_size_limits": utf8,
                "security": security,
                "absolute_paths": absolute_paths,
                "environment_files": environment_files,
                "python_static": python_static,
                "shell_static": shell_static,
                "trajectories": trajectories,
                "delivery_status_integrity": status_integrity,
            }
        )
        safety_gate_names = (
            "manifest",
            "utf8_and_size_limits",
            "security",
            "absolute_paths",
            "environment_files",
            "python_static",
            "shell_static",
        )
        safety_gate_passed = all(report["checks"][name].get("passed") for name in safety_gate_names)
        authorized = bool(args.execute_trusted_code and safety_gate_passed)
        report["checks"]["execution_authorization"] = {
            "passed": authorized,
            "flag_present": bool(args.execute_trusted_code),
            "all_pre_execution_safety_checks_passed": safety_gate_passed,
            "safety_gate_checks": list(safety_gate_names),
            "readiness_blockers_do_not_prevent_safe_local_tests": True,
            "reason": (
                "authorized after pre-execution safety checks"
                if authorized
                else "package code was not executed: explicit flag absent or a pre-execution safety check failed"
            ),
        }
        if authorized:
            report["dynamic_validation"] = run_dynamic_validation(root, trajectories)
        else:
            report["dynamic_validation"] = {
                "passed": False,
                "executed": False,
                "reason": report["checks"]["execution_authorization"]["reason"],
            }

    check_values = list(report["checks"].values())
    report["passed"] = bool(
        root is not None
        and check_values
        and all(isinstance(value, Mapping) and value.get("passed") for value in check_values)
        and report.get("dynamic_validation", {}).get("passed")
        and report.get("checks", {}).get("manifest", {}).get("delivery_status") == "READY_FOR_REVIEW"
    )
    _write_report(output_path, report)
    # ASCII escaping keeps JSON printable even on legacy Windows code pages.
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

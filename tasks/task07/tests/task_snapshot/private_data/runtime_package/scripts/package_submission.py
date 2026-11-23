"""Create a review archive from a small, explicit source allowlist.

The packager intentionally treats a delivery status as an observation, not as
caller supplied metadata.  A partial (``BLOCKED``) archive is still useful for
review, but it must never claim that missing evidence exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import uuid
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT_NAME = "qed_mol_design_benchmark"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_FILE_COUNT = 10_000
FILE_ATTRIBUTE_REPARSE_POINT = 0x400

ROOT_FILES = (
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
)

OPTIONAL_ROOT_FILES: tuple[str, ...] = ()

EXACT_FILES = (
    "configs/model_transition.json",
    "configs/models.example.yaml",
    "configs/task.yaml",
    "prompts/agent_prompt.md",
    "scripts/package_submission.py",
    "scripts/run_all.py",
    "scripts/run_model.py",
    "scripts/validate_package.py",
)

REQUIRED_CORE_FILES = frozenset(
    (*ROOT_FILES, *EXACT_FILES, "framework/siliconflow_adapter.py")
)

RECURSIVE_DIRS = (
    "baselines",
    "evaluator",
    "framework",
    "reports",
    "review",
    "tests",
    "tools",
    "trajectories",
)

REQUIRED_REPORTS = (
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
)

REQUIRED_TRAJECTORY_FILES = frozenset(
    {
        "trajectory.jsonl",
        "dialogue.md",
        "metadata.json",
        "final_answer.json",
        "evaluation.json",
    }
)

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

_DIAGNOSTIC_WINDOWS_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s'\"`<>|?*]+"
)
_DIAGNOSTIC_POSIX_ABSOLUTE = re.compile(
    r"(?<![:A-Za-z0-9_])/(?:home|Users|tmp|var/tmp|workspace|mnt/[A-Za-z0-9._-]+)(?:/[^\s'\"`<>|?*]+)+"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_workspace_root(project_root: Path = ROOT) -> Path:
    """Return the workspace containing the canonical two-level project tree."""

    resolved = project_root.resolve()
    if len(resolved.parents) < 2:
        raise ValueError("Cannot derive a workspace root from the project location")
    return resolved.parents[1]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_write_path(value: str | os.PathLike[str], workspace_root: Path) -> Path:
    """Resolve a caller-selected output and prove that it stays in its workspace."""

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
        return True
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    is_junction = getattr(path, "is_junction", None)
    return bool(
        path.is_symlink()
        or attributes & FILE_ATTRIBUTE_REPARSE_POINT
        or (callable(is_junction) and is_junction())
    )


def _assert_safe_source(path: Path, project_root: Path) -> None:
    resolved_root = project_root.resolve()
    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Allowlisted source is outside the project: {path}") from exc
    cursor = project_root
    for part in relative.parts:
        cursor = cursor / part
        if _is_reparse_or_symlink(cursor):
            raise ValueError(f"Symlink, junction, or reparse point is forbidden: {relative.as_posix()}")
    resolved = path.resolve()
    if not _is_relative_to(resolved, resolved_root):
        raise ValueError(f"Source resolves outside the project: {relative.as_posix()}")


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
    if text in ROOT_FILES or text in OPTIONAL_ROOT_FILES or text in EXACT_FILES:
        return True
    if parts[0] == "trajectories":
        if text == "trajectories/README.md":
            return True
        return (
            len(parts) == 3
            and parts[1] in EXPECTED_TRAJECTORY_MODELS
            and parts[2] in REQUIRED_TRAJECTORY_FILES
        )
    if len(parts) >= 2 and parts[0] in RECURSIVE_DIRS:
        return name in SPECIAL_ALLOWED_NAMES or Path(name).suffix.casefold() in ALLOWED_SUFFIXES
    return False


def iter_source_files(project_root: Path = ROOT) -> Iterable[Path]:
    """Yield existing allowlisted regular files without following links."""

    candidates: list[Path] = []
    for relative in ROOT_FILES + OPTIONAL_ROOT_FILES + EXACT_FILES:
        path = project_root / relative
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Allowlisted path is not a regular file: {relative}")
            candidates.append(path)

    for directory in RECURSIVE_DIRS:
        root = project_root / directory
        if not root.exists():
            continue
        if not root.is_dir():
            raise ValueError(f"Allowlisted path is not a directory: {directory}")
        _assert_safe_source(root, project_root)
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            safe_directories: list[str] = []
            for name in sorted(directory_names):
                child = current_path / name
                relative = child.relative_to(project_root)
                if name.casefold() in FORBIDDEN_PARTS or name.startswith("."):
                    continue
                _assert_safe_source(child, project_root)
                safe_directories.append(name)
            directory_names[:] = safe_directories
            for name in sorted(file_names):
                child = current_path / name
                relative = PurePosixPath(child.relative_to(project_root).as_posix())
                if is_package_relative_allowed(relative):
                    candidates.append(child)

    seen_resolved: set[Path] = set()
    seen_names: set[str] = set()
    for path in sorted(candidates, key=lambda item: item.relative_to(project_root).as_posix()):
        _assert_safe_source(path, project_root)
        resolved = path.resolve()
        relative_text = path.relative_to(project_root).as_posix()
        normalized_name = unicodedata.normalize("NFC", relative_text).casefold()
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate package path after normalization: {relative_text}")
        if resolved in seen_resolved:
            raise ValueError(f"One source file is allowlisted under multiple names: {relative_text}")
        seen_names.add(normalized_name)
        seen_resolved.add(resolved)
        yield path


def _read_utf8(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"Allowlisted file exceeds {MAX_FILE_BYTES} bytes: {path}")
    if b"\x00" in raw:
        raise ValueError(f"NUL bytes are forbidden in packaged text: {path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Allowlisted file is not valid UTF-8: {path}") from exc


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


def _sanitize_diagnostic_tail(value: Any, *, limit: int = 4000) -> str:
    """Keep useful subprocess diagnostics without embedding machine paths."""

    text = str(value or "")[-limit:]
    text = _DIAGNOSTIC_WINDOWS_ABSOLUTE.sub("<absolute-path>", text)
    text = _DIAGNOSTIC_POSIX_ABSOLUTE.sub("<absolute-path>", text)
    return text


def scan_for_secrets(
    paths: Sequence[Path], project_root: Path = ROOT
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        text = _read_utf8(path)
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
                findings.append(
                    {
                        "path": path.relative_to(project_root).as_posix(),
                        "line": text.count("\n", 0, match.start()) + 1,
                        "rule": rule_name,
                    }
                )
    return findings


def scan_absolute_paths(
    paths: Sequence[Path], project_root: Path = ROOT
) -> dict[str, Any]:
    runtime_failures: list[dict[str, Any]] = []
    report_evidence: list[dict[str, Any]] = []
    informational: list[dict[str, Any]] = []
    runtime_suffixes = {".py", ".json", ".jsonl", ".yaml", ".yml", ".ps1", ".sh"}
    runtime_names = {"requirements.txt", ".env.example", "pytest.ini"}
    for path in paths:
        relative = path.relative_to(project_root).as_posix()
        text = _read_utf8(path)
        matches: list[tuple[int, str]] = []
        for pattern in (_DIAGNOSTIC_WINDOWS_ABSOLUTE, _DIAGNOSTIC_POSIX_ABSOLUTE):
            for match in pattern.finditer(text):
                matches.append(
                    (text.count("\n", 0, match.start()) + 1, match.group(0)[:240])
                )
        for line, evidence in matches:
            record = {"path": relative, "line": line, "evidence": evidence}
            if relative.startswith("reports/"):
                report_evidence.append(record)
            elif path.suffix.casefold() in runtime_suffixes or path.name in runtime_names:
                runtime_failures.append(record)
            else:
                informational.append(record)
    return {
        "passed": not runtime_failures,
        "runtime_failures": runtime_failures,
        "report_delivery_path_evidence": report_evidence,
        "other_informational_findings": informational,
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path) -> Any:
    return json.loads(
        _read_utf8(path),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    project_root: Path,
    metadata: Mapping[str, Any],
    *,
    expected_directory: str | None,
) -> tuple[dict[str, str], list[str]]:
    """Prove the frozen Kimi evidence predates only the two approved model-ID edits."""

    if expected_directory != "kimi":
        return {}, []
    errors: list[str] = []
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
        manifest = _strict_json(project_root / MODEL_TRANSITION_FILE)
    except Exception as exc:
        return {}, [f"model transition manifest cannot be verified: {type(exc).__name__}: {exc}"]
    if manifest != _expected_model_transition_manifest():
        return {}, ["model transition manifest differs from the exact approved transition"]

    kimi_root = project_root / "trajectories" / "kimi"
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
        target = project_root / Path(*PurePosixPath(relative).parts)
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
        reconstructed_hash = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
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


def _verify_hash_evidence(
    project_root: Path,
    metadata: Mapping[str, Any],
    *,
    expected_directory: str | None = None,
) -> list[str]:
    errors: list[str] = []
    approved_historical, transition_errors = _approved_kimi_historical_hashes(
        project_root, metadata, expected_directory=expected_directory
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
            target = (project_root / Path(*pure.parts)).resolve()
            if pure.is_absolute() or ".." in pure.parts or not _is_relative_to(target, project_root.resolve()) or not target.is_file():
                errors.append(f"hashed source path is missing or unsafe: {relative}")
            elif sha256_file(target) != expected and approved_historical.get(relative) != expected:
                errors.append(f"source hash mismatch: {relative}")
    if not isinstance(configs, Mapping):
        errors.append("metadata.hashes.configs is missing")
        return errors
    try:
        task = yaml.safe_load(_read_utf8(project_root / "configs" / "task.yaml"))
        prompt = _read_utf8(project_root / "prompts" / "agent_prompt.md")
        expected_hashes = {
            "task_config": hashlib.sha256(_canonical_json(task).encode("utf-8")).hexdigest(),
            "model_config_sanitized": hashlib.sha256(
                _canonical_json(metadata.get("model_config")).encode("utf-8")
            ).hexdigest(),
            "agent_prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        for name, expected in expected_hashes.items():
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
    listed_candidate_set: set[str] | None = None
    if not isinstance(run_metadata, Mapping):
        errors.append("final_answer.run_metadata is missing")
    else:
        if run_metadata.get("budget_exceeded") is not False:
            errors.append("final_answer.run_metadata.budget_exceeded must be false")
        if run_metadata.get("tool_call_count") != len(accepted_tool_events):
            errors.append("final_answer tool count does not match accepted trajectory events")
        listed = run_metadata.get("unique_evaluated_candidates")
        if isinstance(listed, list) and all(isinstance(value, str) for value in listed):
            listed_candidate_set = set(listed)
        if listed_candidate_set != canonical_candidates:
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


def _last_provider_final_candidate(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    provider_messages = [
        event.get("message")
        for event in events
        if event.get("event") == "message" and event.get("source") == "provider"
    ]
    if not provider_messages:
        return None
    message = provider_messages[-1]
    if (
        not isinstance(message, Mapping)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        return None
    try:
        value = json.loads(
            message["content"],
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except Exception:
        return None
    if isinstance(value, Mapping) and isinstance(value.get("final_candidate"), Mapping):
        return value["final_candidate"]
    return None


def check_complete_run_lifecycle(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        if not any(
            request < response < message < first_tool
            for request in request_indices
            for response in response_indices
            for message in provider_message_indices
        ):
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
    return {
        "passed": not errors,
        "errors": errors,
        "provider": provider,
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


def assess_trajectories(
    project_root: Path = ROOT, *, workspace_root: Path | None = None
) -> dict[str, Any]:
    trajectories_root = project_root / "trajectories"
    results: dict[str, Any] = {}
    passing_actual_models: list[str] = []
    if trajectories_root.is_dir():
        directories = sorted(
            (item for item in trajectories_root.iterdir() if item.is_dir()),
            key=lambda item: item.name.casefold(),
        )
    else:
        directories = []
    actual_directories = {directory.name for directory in directories}
    expected_directories = set(EXPECTED_TRAJECTORY_MODELS)
    missing_directories = sorted(expected_directories - actual_directories)
    unexpected_directories = sorted(actual_directories - expected_directories)
    layout_errors: list[str] = []
    if missing_directories:
        layout_errors.append(f"missing required trajectory directories: {missing_directories}")
    if unexpected_directories:
        layout_errors.append(f"unexpected trajectory directories: {unexpected_directories}")
    task_config: Mapping[str, Any] = {}
    try:
        loaded_task_config = yaml.safe_load(
            _read_utf8(project_root / "configs" / "task.yaml")
        )
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
        event_count = 0
        tool_event_count = 0
        provider_response_count = 0
        if not missing:
            try:
                loaded_metadata = _strict_json(directory / "metadata.json")
                loaded_final = _strict_json(directory / "final_answer.json")
                loaded_evaluation = _strict_json(directory / "evaluation.json")
                if not isinstance(loaded_metadata, dict) or not isinstance(loaded_final, dict) or not isinstance(loaded_evaluation, dict):
                    raise ValueError("metadata, final answer, and evaluation must be JSON objects")
                metadata = loaded_metadata
                final_answer = loaded_final
                stored_evaluation = loaded_evaluation
                for line_number, line in enumerate(
                    _read_utf8(directory / "trajectory.jsonl").splitlines(), 1
                ):
                    if not line.strip():
                        raise ValueError(f"blank JSONL record at line {line_number}")
                    event = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")),
                    )
                    if not isinstance(event, dict):
                        raise ValueError(f"JSONL line {line_number} is not an object")
                    events.append(event)
                    event_count += 1
                    if event.get("event") == "tool_call" and event.get("status") != "REJECTED":
                        tool_event_count += 1
                    if event.get("event") == "provider_response":
                        provider_response_count += 1
            except Exception as exc:
                errors.append(f"artifact parse failed: {type(exc).__name__}: {exc}")

        counts = metadata.get("counts") if isinstance(metadata, Mapping) else None
        responses = metadata.get("provider_response_metadata") if isinstance(metadata, Mapping) else None
        identity_check: Mapping[str, Any] = {}
        if metadata:
            if metadata.get("status") != "COMPLETE":
                errors.append("metadata.status is not COMPLETE")
            if metadata.get("execution_mode") != "REAL_API" or metadata.get("is_real_model_run") is not True:
                errors.append("metadata is inconsistent with a REAL_API run")
            if metadata.get("failure") is not None:
                errors.append("metadata contains a run failure")
            identity_check = check_model_identity_consistency(
                metadata, expected_directory=directory.name
            )
            errors.extend(identity_check.get("errors", []))
            if not isinstance(responses, list) or not responses:
                errors.append("provider response metadata is empty")
            if not isinstance(counts, Mapping) or counts.get("tool_calls") != tool_event_count:
                errors.append("metadata tool count does not match trajectory events")
            if provider_response_count < 1:
                errors.append("trajectory contains no provider response event")
            errors.extend(
                _verify_hash_evidence(
                    project_root, metadata, expected_directory=directory.name
                )
            )
        dependency_check = check_dependency_consistency(metadata)
        errors.extend(dependency_check["errors"])
        sequences = [event.get("sequence") for event in events]
        if sequences and sequences != list(range(1, len(events) + 1)):
            errors.append("trajectory sequence numbers are not contiguous from 1")
        run_metadata = final_answer.get("run_metadata") if isinstance(final_answer, Mapping) else None
        if final_answer and (
            not isinstance(run_metadata, Mapping)
            or run_metadata.get("tool_call_count") != tool_event_count
            or run_metadata.get("final_candidate_evaluated") is not True
        ):
            errors.append("final_answer process metadata is inconsistent with the trajectory")
        lifecycle = check_complete_run_lifecycle(events)
        errors.extend(lifecycle["errors"])
        run_attestation = check_run_attestation(metadata, events)
        errors.extend(run_attestation["errors"])
        closed_loop = assess_closed_loop_rounds(events)
        errors.extend(closed_loop["errors"])
        if final_answer:
            process_candidate = _last_provider_final_candidate(events)
            final_candidate = final_answer.get("final_candidate")
            if not isinstance(final_candidate, Mapping) or process_candidate is None or dict(process_candidate) != dict(final_candidate):
                errors.append("final answer differs from the final provider process message")
            final_smiles = final_candidate.get("smiles") if isinstance(final_candidate, Mapping) else None
            try:
                if not isinstance(final_smiles, str) or final_smiles not in _read_utf8(directory / "dialogue.md"):
                    errors.append("dialogue.md does not contain the final SMILES")
            except Exception as exc:
                errors.append(f"dialogue.md cannot be checked: {type(exc).__name__}: {exc}")
        if tool_event_count < minimum_tool_calls:
            errors.append(f"trajectory has fewer than {minimum_tool_calls} required tool calls")
        accepted_tool_events = [
            event for event in events if event.get("event") == "tool_call" and event.get("status") != "REJECTED"
        ]
        successful_tool_events = [
            event
            for event in accepted_tool_events
            if _is_successful_tool_event(event)
        ]
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
        replay: dict[str, Any]
        if errors:
            replay = {
                "passed": False,
                "executed": False,
                "reason": "static trajectory consistency checks failed before evaluator replay",
            }
        else:
            replay = replay_trajectory_evaluation(
                project_root,
                directory,
                workspace_root=workspace_root or project_root.resolve().parent,
            )
            errors.extend(replay.get("errors", []))
        passed = not errors
        actual_identity = identity_check.get("actual_model_id") if identity_check else None
        if passed and isinstance(actual_identity, str):
            passing_actual_models.append(actual_identity.casefold())
        results[directory.name] = {
            "passed": passed,
            "errors": errors,
            "event_count": event_count,
            "tool_event_count": tool_event_count,
            "identity_consistency": dict(identity_check),
            "process_eligibility_gate": process_gate,
            "positive_objective_gate": objective_gate,
            "lifecycle": lifecycle,
            "run_attestation": run_attestation,
            "closed_loop": closed_loop,
            "dependency_consistency": dependency_check,
            "evaluation_replay": replay,
        }

    passing_count = sum(
        1
        for name in expected_directories
        if name in results and results[name]["passed"]
    )
    distinct_models = len(set(passing_actual_models))
    all_expected_passed = all(
        name in results and results[name]["passed"] for name in expected_directories
    )
    return {
        "passed": not layout_errors and all_expected_passed and distinct_models == 3,
        "errors": layout_errors,
        "expected_directory_models": dict(EXPECTED_TRAJECTORY_MODELS),
        "missing_directories": missing_directories,
        "unexpected_directories": unexpected_directories,
        "passing_trajectory_count": passing_count,
        "distinct_actual_model_count": distinct_models,
        "models": results,
    }


def _clean_test_environment(temp_dir: Path) -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    python_dir = str(Path(sys.executable).resolve().parent)
    path_parts = [python_dir]
    for candidate in (Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "Scripts"):
        if candidate.is_dir():
            path_parts.append(str(candidate.resolve()))
    environment.update(
        {
            "PATH": os.pathsep.join(path_parts),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
        }
    )
    rdkit_spec = importlib.util.find_spec("rdkit")
    if rdkit_spec is not None:
        locations = list(rdkit_spec.submodule_search_locations or [])
        if locations:
            environment["PYTHONPATH"] = str(Path(locations[0]).resolve().parent)
    return environment


def replay_trajectory_evaluation(
    project_root: Path,
    trajectory_dir: Path,
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """Re-run the checked-in evaluator without placing temp files outside the workspace."""

    workspace = workspace_root.expanduser().resolve()
    temp_parent = resolve_write_path(".package_evaluation_temp", workspace)
    if temp_parent.exists() and (
        not temp_parent.is_dir() or _is_reparse_or_symlink(temp_parent)
    ):
        return {
            "passed": False,
            "executed": False,
            "errors": ["workspace evaluator temporary parent is unsafe"],
        }
    temp_parent.mkdir(parents=True, exist_ok=True)
    run_temp = temp_parent / uuid.uuid4().hex
    run_temp.mkdir(exist_ok=False)
    generated = run_temp / "evaluation.json"
    relative_dir = trajectory_dir.relative_to(project_root).as_posix()
    command = [
        sys.executable,
        "evaluator/evaluate.py",
        "--submission",
        f"{relative_dir}/final_answer.json",
        "--trajectory",
        f"{relative_dir}/trajectory.jsonl",
        "--config",
        "configs/task.yaml",
        "--output",
        str(generated),
    ]
    result: dict[str, Any] = {
        "passed": False,
        "executed": True,
        "command": [
            Path(sys.executable).name,
            *command[1:-1],
            "<workspace-temporary-evaluation.json>",
        ],
        "exit_code": None,
        "json_object_equal": False,
        "byte_identical": False,
        "errors": [],
    }
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=_clean_test_environment(run_temp),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=120,
            check=False,
        )
        result["exit_code"] = completed.returncode
        if completed.returncode != 0:
            result["errors"].append("evaluator replay command returned a nonzero exit code")
            return result
        if not generated.is_file():
            result["errors"].append("evaluator replay did not create an evaluation file")
            return result
        stored_path = trajectory_dir / "evaluation.json"
        generated_json = _strict_json(generated)
        stored_json = _strict_json(stored_path)
        loaded_task_config = yaml.safe_load(
            _read_utf8(project_root / "configs" / "task.yaml")
        )
        task_config = (
            loaded_task_config if isinstance(loaded_task_config, Mapping) else {}
        )
        result["json_object_equal"] = generated_json == stored_json
        result["byte_identical"] = generated.read_bytes() == stored_path.read_bytes()
        result["generated_sha256"] = sha256_file(generated)
        result["stored_sha256"] = sha256_file(stored_path)
        result["generated_valid_submission"] = bool(
            isinstance(generated_json, Mapping)
            and generated_json.get("valid_submission") is True
        )
        result["generated_process_eligible"] = bool(
            isinstance(generated_json, Mapping)
            and generated_json.get("process_eligible") is True
        )
        result["positive_objective_gate"] = (
            check_positive_objective_evidence(generated_json, task_config)
            if isinstance(generated_json, Mapping)
            else {
                "passed": False,
                "errors": ["replayed evaluation is not a JSON object"],
            }
        )
        if not result["json_object_equal"]:
            result["errors"].append("replayed evaluation JSON differs from the stored evaluation")
        if not result["generated_valid_submission"]:
            result["errors"].append("replayed evaluation is not a valid submission")
        if not result["generated_process_eligible"]:
            result["errors"].append("replayed evaluation is not process eligible")
        if not result["positive_objective_gate"].get("passed"):
            result["errors"].append(
                "replayed evaluation does not prove strict positive objective improvement"
            )
            result["errors"].extend(result["positive_objective_gate"].get("errors", []))
        result["passed"] = not result["errors"]
        return result
    except subprocess.TimeoutExpired:
        result["errors"].append("evaluator replay exceeded the 120 second timeout")
        return result
    except Exception as exc:
        result["errors"].append(f"evaluator replay failed with {type(exc).__name__}")
        return result
    finally:
        shutil.rmtree(run_temp, ignore_errors=True)
        try:
            temp_parent.rmdir()
        except OSError:
            pass


def run_test_readiness(project_root: Path, workspace_root: Path) -> dict[str, Any]:
    temp_dir = workspace_root / ".package_test_temp" / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=_clean_test_environment(temp_dir),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=600,
            check=False,
        )
        return {
            "passed": completed.returncode == 0,
            "command": [Path(sys.executable).name, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
            "exit_code": completed.returncode,
            "stdout_tail": _sanitize_diagnostic_tail(completed.stdout),
            "stderr_tail": _sanitize_diagnostic_tail(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "command": command,
            "exit_code": None,
            "error": "pytest exceeded the 600 second readiness timeout",
            "stdout_tail": _sanitize_diagnostic_tail(exc.stdout),
            "stderr_tail": _sanitize_diagnostic_tail(exc.stderr),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        parent = temp_dir.parent
        try:
            parent.rmdir()
        except OSError:
            pass


def derive_readiness(
    project_root: Path = ROOT,
    *,
    test_result: Mapping[str, Any] | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    missing_core = sorted(
        relative
        for relative in REQUIRED_CORE_FILES
        if not (project_root / relative).is_file()
    )
    missing_reports = sorted(relative for relative in REQUIRED_REPORTS if not (project_root / relative).is_file())
    trajectory_result = assess_trajectories(
        project_root, workspace_root=workspace_root
    )
    tests_present = (project_root / "tests").is_dir() and any((project_root / "tests").glob("test_*.py"))
    test_report_present = (project_root / "reports" / "test_report.md").is_file()
    actual_tests_passed = bool(test_result and test_result.get("passed"))
    checks = {
        "core_files": {"passed": not missing_core, "missing": missing_core},
        "required_reports": {"passed": not missing_reports, "missing": missing_reports},
        "project_license": {"passed": (project_root / "LICENSE").is_file(), "required": "LICENSE"},
        "project_notice": {"passed": (project_root / "NOTICE").is_file(), "required": "NOTICE"},
        "test_evidence": {
            "passed": tests_present and test_report_present and actual_tests_passed,
            "tests_present": tests_present,
            "test_report_present": test_report_present,
            "actual_test_run": dict(test_result or {"passed": False, "reason": "not executed"}),
        },
        "real_trajectories": trajectory_result,
    }
    blocking_reasons: list[str] = []
    for name, result in checks.items():
        if not result.get("passed"):
            blocking_reasons.append(f"{name} evidence is incomplete or invalid")
    status = "READY_FOR_REVIEW" if not blocking_reasons else "BLOCKED"
    return {"delivery_status": status, "blocking_reasons": blocking_reasons, "checks": checks}


def validate_source_set(paths: Sequence[Path], project_root: Path = ROOT) -> dict[str, Any]:
    if len(paths) > MAX_FILE_COUNT:
        raise ValueError(f"Allowlisted source contains more than {MAX_FILE_COUNT} files")
    total = 0
    for path in paths:
        _read_utf8(path)
        total += path.stat().st_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f"Allowlisted source exceeds {MAX_TOTAL_BYTES} total bytes")
    secret_findings = scan_for_secrets(paths, project_root)
    if secret_findings:
        raise ValueError(
            "Secret scan failed; no archive was created: "
            + json.dumps(secret_findings, ensure_ascii=False)
        )
    absolute_paths = scan_absolute_paths(paths, project_root)
    if not absolute_paths["passed"]:
        raise ValueError(
            "Runtime absolute-path scan failed; no archive was created: "
            + json.dumps(absolute_paths["runtime_failures"], ensure_ascii=False)
        )
    return {
        "file_count": len(paths),
        "total_bytes": total,
        "secret_findings": [],
        "absolute_paths": absolute_paths,
    }


def build_manifest(staging_root: Path, readiness: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(staging_root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            entries.append(
                {
                    "path": path.relative_to(staging_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "2.0",
        "benchmark": "constrained_qed_optimization_v1",
        "delivery_status": readiness["delivery_status"],
        "blocking_reasons": list(readiness.get("blocking_reasons", [])),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count_excluding_manifest": len(entries),
        "total_size_excluding_manifest": sum(entry["size_bytes"] for entry in entries),
        "readiness_checks": readiness.get("checks", {}),
        "files": entries,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _check_canonical_manifest_target(path: Path) -> None:
    """Refuse to replace a manifest that this packager cannot identify."""

    if not path.exists():
        return
    if not path.is_file() or _is_reparse_or_symlink(path):
        raise FileExistsError(f"Canonical manifest target is not a safe regular file: {path}")
    try:
        value = _strict_json(path)
    except Exception as exc:
        raise FileExistsError(
            f"Refusing to overwrite an unrecognized canonical manifest: {path}"
        ) from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != "2.0" or value.get("benchmark") != "constrained_qed_optimization_v1":
        raise FileExistsError(
            f"Refusing to overwrite an unrecognized canonical manifest: {path}"
        )


def _publish_canonical_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if not path.exists():
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return
    _check_canonical_manifest_target(path)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _copy_sources(paths: Sequence[Path], project_root: Path, staging_root: Path) -> None:
    for source in paths:
        relative = source.relative_to(project_root)
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination, follow_symlinks=False)


def _create_archive(staging_parent: Path, staging_root: Path, archive_path: Path) -> None:
    partial = archive_path.parent / f".{archive_path.name}.{uuid.uuid4().hex}.partial"
    try:
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            seen: set[str] = set()
            for path in sorted(staging_root.rglob("*")):
                if not path.is_file():
                    continue
                member = path.relative_to(staging_parent).as_posix()
                normalized = unicodedata.normalize("NFC", member).casefold()
                if normalized in seen:
                    raise ValueError(f"Duplicate ZIP member after normalization: {member}")
                seen.add(normalized)
                archive.write(path, member)
        if archive_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing archive: {archive_path}")
        partial.rename(archive_path)
    finally:
        if partial.exists():
            partial.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a security-checked allowlisted benchmark archive")
    parser.add_argument("--workspace-root", default=str(derived_workspace_root()))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--staging-dir", default=None, help="Parent for the persistent clean staging tree")
    parser.add_argument("--filename", default=f"qed_mol_design_agent_benchmark_v3_{date.today():%Y%m%d}.zip")
    parser.add_argument(
        "--status",
        choices=["BLOCKED", "READY_FOR_REVIEW"],
        default=None,
        help="Deprecated status request; recorded but never allowed to override evidence",
    )
    parser.add_argument(
        "--skip-test-run",
        action="store_true",
        help="Do not execute pytest; this necessarily derives BLOCKED status",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    if not workspace_root.is_dir():
        raise FileNotFoundError(f"Workspace root does not exist: {workspace_root}")
    output_dir = resolve_write_path(args.output_dir or "outputs", workspace_root)
    staging_parent = resolve_write_path(args.staging_dir or "submission_packages", workspace_root)
    filename_path = Path(args.filename)
    if filename_path.name != args.filename or filename_path.suffix.casefold() != ".zip":
        raise ValueError("--filename must be a plain .zip filename without directory components")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_parent.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / filename_path.name
    report_path = archive_path.with_suffix(".package.json")
    staging_root = staging_parent / archive_path.stem / PACKAGE_ROOT_NAME
    staging_container = staging_root.parent
    for target in (archive_path, report_path, staging_container):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    canonical_manifest = ROOT / "MANIFEST.json"
    _check_canonical_manifest_target(canonical_manifest)

    sources = list(iter_source_files(ROOT))
    source_validation = validate_source_set(sources, ROOT)
    test_result: Mapping[str, Any]
    if args.skip_test_run:
        test_result = {"passed": False, "reason": "caller explicitly skipped the current pytest run"}
    else:
        test_result = run_test_readiness(ROOT, workspace_root)
    readiness = derive_readiness(
        ROOT, test_result=test_result, workspace_root=workspace_root
    )

    staging_root.mkdir(parents=True, exist_ok=False)
    _copy_sources(sources, ROOT, staging_root)
    manifest = build_manifest(staging_root, readiness)
    _write_json_exclusive(staging_root / "MANIFEST.json", manifest)
    _create_archive(staging_container, staging_root, archive_path)
    _publish_canonical_manifest(canonical_manifest, manifest)

    report: dict[str, Any] = {
        "schema_version": "2.0",
        "delivery_status": readiness["delivery_status"],
        "caller_requested_status": args.status,
        "caller_status_can_override_evidence": False,
        "archive": str(archive_path),
        "staging_root": str(staging_root),
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "manifest_file_count": manifest["file_count_excluding_manifest"],
        "source_validation": source_validation,
        "readiness": readiness,
    }
    _write_json_exclusive(report_path, report)
    # ASCII escaping keeps JSON printable even on legacy Windows code pages.
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

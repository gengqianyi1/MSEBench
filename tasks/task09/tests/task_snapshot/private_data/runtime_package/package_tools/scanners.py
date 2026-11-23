from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from package_tools.policy import forbidden_release_path_reason


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    (
        "github_token",
        re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    (
        "bearer",
        re.compile(
            r"(?i)\bauthorization\s*[:=]\s*['\"]?bearer\s+[A-Za-z0-9._~+/=-]{12,}"
        ),
    ),
    (
        "configured_key",
        re.compile(r"(?i)\bGJLD_API_KEY\s*[:=]\s*['\"]?([^\s'\",;}{]{8,})"),
    ),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd)"
            r"\s*[:=]\s*['\"]?([^\s'\",;}{]{12,})"
        ),
    ),
)
PLACEHOLDER_WORDS = (
    "dummy",
    "example",
    "placeholder",
    "redacted",
    "replace",
    "test",
    "your_",
    "your-",
)
WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s'\"<>|?*]+")
POSIX_ABSOLUTE = re.compile(
    r"(?<![:A-Za-z0-9_])/(?:home|Users|tmp|workspace|mnt)/[^\s'\"<>|?*]+"
)
SENSITIVE_JSON_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|secret|authorization|credential|password|passwd)"
    r"(?:$|[_-])"
)


def _text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _placeholder(value: str) -> bool:
    lowered = value.strip().strip("'\"<>[](){}").casefold()
    template_marker = "$" + "{"
    return (
        not lowered
        or any(word in lowered for word in PLACEHOLDER_WORDS)
        or template_marker in value
        or "$env:" in lowered
        or lowered.startswith(("load_", "os.", "self.", "config.", "environment."))
        or lowered.endswith(("_api_key", "_token", "_secret", "_credential"))
    )


def _recursive_json_secret(value: Any, *, depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                SENSITIVE_JSON_KEY.search(str(key))
                and isinstance(child, str)
                and len(child.strip()) >= 8
                and not _placeholder(child)
            ):
                return True
            if _recursive_json_secret(child, depth=depth + 1):
                return True
        return False
    if isinstance(value, list):
        return any(
            _recursive_json_secret(item, depth=depth + 1)
            for item in value
        )
    if isinstance(value, str):
        if any(
            pattern.search(value)
            and not _placeholder(
                pattern.search(value).group(1)
                if pattern.search(value).lastindex
                else pattern.search(value).group(0)
            )
            for _rule, pattern in SECRET_PATTERNS
        ):
            return True
        stripped = value.strip()
        if stripped.startswith(("{", "[", '"')) and stripped.endswith(
            ("}", "]", '"')
        ):
            try:
                decoded = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return False
            if decoded != value:
                return _recursive_json_secret(
                    decoded, depth=depth + 1
                )
    return False


def scan_secrets(root: Path, files: Iterable[Path]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path in files:
        content = _text(path)
        if content is None:
            continue
        relative = path.relative_to(root).as_posix()
        for rule, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(content):
                captured = match.group(1) if match.lastindex else match.group(0)
                if _placeholder(captured):
                    continue
                findings.append(
                    {
                        "path": relative,
                        "line": content.count("\n", 0, match.start()) + 1,
                        "rule": rule,
                    }
                )
        stripped = content.strip()
        if stripped.startswith(("{", "[", '"')):
            try:
                decoded = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                decoded = None
            if (
                decoded is not None
                and _recursive_json_secret(decoded)
            ):
                findings.append(
                    {
                        "path": relative,
                        "line": 1,
                        "rule": "recursive_json_secret",
                    }
                )
    return findings


def scan_runtime_absolute_paths(
    root: Path, files: Iterable[Path]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    failures: list[dict[str, object]] = []
    expected_test_literals: list[dict[str, object]] = []
    for path in files:
        content = _text(path)
        if content is None:
            continue
        relative = path.relative_to(root).as_posix()
        for rule, pattern in (
            ("windows_absolute", WINDOWS_ABSOLUTE),
            ("posix_absolute", POSIX_ABSOLUTE),
        ):
            for match in pattern.finditer(content):
                item = {
                    "path": relative,
                    "line": content.count("\n", 0, match.start()) + 1,
                    "rule": rule,
                }
                if relative.startswith("tests/"):
                    expected_test_literals.append(item)
                else:
                    failures.append(item)
    return failures, expected_test_literals


def scan_forbidden_names(root: Path, files: Iterable[Path]) -> list[str]:
    findings: list[str] = []
    for path in files:
        relative = path.relative_to(root)
        if forbidden_release_path_reason(relative) is not None:
            findings.append(relative.as_posix())
    return sorted(set(findings))

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


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
        or lowered.endswith(("_api_key", "_token", "_secret"))
    )


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
        lowered = tuple(part.casefold() for part in relative.parts)
        if ".env" in lowered or "__pycache__" in lowered:
            findings.append(relative.as_posix())
        if path.suffix.casefold() in {".pyc", ".pyo", ".zip"}:
            findings.append(relative.as_posix())
    return sorted(set(findings))

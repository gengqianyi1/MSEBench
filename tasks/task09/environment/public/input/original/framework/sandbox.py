"""Fail-closed filesystem and native-command boundaries for public tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
from typing import Any


SHELL_ENABLED = False
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_DEVICE = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$", re.IGNORECASE
)
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class SandboxViolation(PermissionError):
    """An input attempted to escape the benchmark's declared boundary."""


def _is_reparse_or_symlink(path: Path) -> bool:
    """Detect POSIX symlinks and Windows junction/reparse points."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SandboxViolation("cannot verify path metadata") from exc
    if stat.S_ISLNK(metadata.st_mode) or path.is_symlink():
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & _REPARSE_ATTRIBUTE)


def _relative_parts(value: str | os.PathLike[str]) -> tuple[str, ...]:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise SandboxViolation("path must be text or path-like") from exc
    if not isinstance(raw, str) or not raw:
        raise SandboxViolation("path must be non-empty text")
    if _CONTROL_CHARACTERS.search(raw):
        raise SandboxViolation("control characters are forbidden in paths")
    if raw.startswith(("\\\\?\\", "\\\\.\\", "//?/", "//./")):
        raise SandboxViolation("Windows device paths are forbidden")

    windows = PureWindowsPath(raw)
    posix = PurePosixPath(raw)
    if windows.drive or windows.root or posix.is_absolute():
        raise SandboxViolation("absolute, drive, rooted, and UNC paths are forbidden")
    # A colon in any relative component is a Windows alternate data stream (or
    # drive-relative syntax) and is rejected on every host for replay parity.
    if ":" in raw:
        raise SandboxViolation("alternate data streams and drive syntax are forbidden")

    # Split both separator families even when replaying on a non-Windows host.
    parts = tuple(re.split(r"[\\/]", raw))
    if any(part in {"", ".", ".."} for part in parts):
        raise SandboxViolation("empty, dot, and parent path components are forbidden")
    for part in parts:
        if part.endswith((" ", ".")):
            raise SandboxViolation("Windows-normalized trailing dots/spaces are forbidden")
        if _WINDOWS_DEVICE.fullmatch(part):
            raise SandboxViolation("Windows device names are forbidden")
    return parts


def _normalized_case(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _same_or_descendant(path: Path, root: Path) -> bool:
    candidate_text = _normalized_case(path)
    root_text = _normalized_case(root)
    try:
        return os.path.commonpath((candidate_text, root_text)) == root_text
    except (OSError, ValueError):
        return False


class PathSandbox:
    """Resolve only strict relative paths beneath a non-reparse workspace root."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        forbidden_roots: Sequence[str | os.PathLike[str]] = (),
    ) -> None:
        candidate_root = Path(root)
        if not candidate_root.is_absolute():
            raise SandboxViolation("sandbox root must be an absolute trusted path")
        try:
            resolved_root = candidate_root.resolve(strict=True)
        except OSError as exc:
            raise SandboxViolation("sandbox root does not exist") from exc
        if not resolved_root.is_dir():
            raise SandboxViolation("sandbox root must be a directory")
        if _is_reparse_or_symlink(candidate_root) or _is_reparse_or_symlink(resolved_root):
            raise SandboxViolation("sandbox root must not be a symlink or reparse point")
        self.root = resolved_root

        normalized_forbidden: list[Path] = []
        for raw_forbidden in forbidden_roots:
            forbidden = Path(raw_forbidden)
            if not forbidden.is_absolute():
                raise SandboxViolation("forbidden roots must be absolute trusted paths")
            try:
                normalized = forbidden.resolve(strict=False)
            except OSError as exc:
                raise SandboxViolation("cannot normalize a forbidden root") from exc
            normalized_forbidden.append(normalized)
        self.forbidden_roots = tuple(normalized_forbidden)

    def _check_components(self, parts: Sequence[str]) -> Path:
        current = self.root
        if _is_reparse_or_symlink(current):
            raise SandboxViolation("sandbox root became a reparse point")
        for part in parts:
            current = current / part
            if current.exists() or current.is_symlink():
                if _is_reparse_or_symlink(current):
                    raise SandboxViolation("symlink, junction, or reparse traversal is forbidden")
        return current

    def _check_forbidden(self, candidate: Path) -> None:
        for forbidden in self.forbidden_roots:
            # Also reject a parent-directory handle that would permit listing a
            # forbidden child (oracle, source, evaluator, or .env).
            if _same_or_descendant(candidate, forbidden) or _same_or_descendant(
                forbidden, candidate
            ):
                raise SandboxViolation("path intersects a forbidden root")

    def resolve(
        self,
        relative_path: str | os.PathLike[str],
        *,
        must_exist: bool = False,
        expect_file: bool | None = None,
    ) -> Path:
        parts = _relative_parts(relative_path)
        lexical_candidate = self._check_components(parts)
        try:
            candidate = lexical_candidate.resolve(strict=False)
        except OSError as exc:
            raise SandboxViolation("path cannot be safely resolved") from exc
        if not _same_or_descendant(candidate, self.root):
            raise SandboxViolation("path escapes the sandbox root")
        self._check_forbidden(candidate)
        if must_exist and not candidate.exists():
            raise SandboxViolation("requested path does not exist")
        if candidate.exists():
            if _is_reparse_or_symlink(candidate):
                raise SandboxViolation("resolved path is a symlink or reparse point")
            if expect_file is True and not candidate.is_file():
                raise SandboxViolation("requested path is not a regular file")
            if expect_file is False and not candidate.is_dir():
                raise SandboxViolation("requested path is not a directory")
        return candidate

    def resolve_for_read(self, relative_path: str | os.PathLike[str]) -> Path:
        return self.resolve(relative_path, must_exist=True, expect_file=True)

    def resolve_for_write(self, relative_path: str | os.PathLike[str]) -> Path:
        destination = self.resolve(relative_path, must_exist=False)
        parent = destination.parent
        if not parent.exists() or not parent.is_dir():
            raise SandboxViolation("write destination parent must already exist")
        if _is_reparse_or_symlink(parent):
            raise SandboxViolation("write destination parent is a reparse point")
        return destination


def resolve_relative_path(
    root: str | os.PathLike[str],
    relative_path: str | os.PathLike[str],
    *,
    forbidden_roots: Sequence[str | os.PathLike[str]] = (),
    must_exist: bool = False,
) -> Path:
    """One-shot convenience wrapper around :class:`PathSandbox`."""

    return PathSandbox(root, forbidden_roots=forbidden_roots).resolve(
        relative_path, must_exist=must_exist
    )


class AllowlistedCommandRunner:
    """Run an exact executable allowlist via argv with ``shell=False`` fixed."""

    def __init__(self, executables: Mapping[str, str | os.PathLike[str]]) -> None:
        if not isinstance(executables, Mapping) or not executables:
            raise SandboxViolation("executable allowlist must be a non-empty mapping")
        normalized: dict[str, Path] = {}
        for name, raw_path in executables.items():
            if not isinstance(name, str) or not name or not re.fullmatch(
                r"[A-Za-z0-9_.-]+", name
            ):
                raise SandboxViolation("executable logical name is invalid")
            path = Path(raw_path)
            if not path.is_absolute():
                raise SandboxViolation("allowlisted executable path must be absolute")
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise SandboxViolation("allowlisted executable does not exist") from exc
            if not resolved.is_file() or _is_reparse_or_symlink(path):
                raise SandboxViolation("allowlisted executable must be a regular non-reparse file")
            normalized[name] = resolved
        self.executables = normalized

    def argv(self, executable: str, arguments: Sequence[str]) -> list[str]:
        if executable not in self.executables:
            raise SandboxViolation("executable is not allowlisted")
        if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
            raise SandboxViolation("command arguments must be an argv sequence")
        argv = [str(self.executables[executable])]
        for argument in arguments:
            if not isinstance(argument, str) or "\x00" in argument:
                raise SandboxViolation("command arguments must be NUL-free strings")
            argv.append(argument)
        return argv

    def run(
        self,
        executable: str,
        arguments: Sequence[str],
        *,
        cwd: str | os.PathLike[str],
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        argv = self.argv(executable, arguments)
        workdir = Path(cwd)
        if not workdir.is_absolute():
            raise SandboxViolation("native working directory must be absolute")
        try:
            resolved_workdir = workdir.resolve(strict=True)
        except OSError as exc:
            raise SandboxViolation("native working directory does not exist") from exc
        if not resolved_workdir.is_dir() or _is_reparse_or_symlink(workdir):
            raise SandboxViolation("native working directory is unsafe")
        if not isinstance(env, Mapping) or any(
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(value, str)
            or "\x00" in value
            for key, value in env.items()
        ):
            raise SandboxViolation("native environment must contain valid strings")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math_is_finite_positive(timeout_seconds)
        ):
            raise SandboxViolation("native timeout must be finite and positive")
        return subprocess.run(
            argv,
            cwd=str(resolved_workdir),
            env=dict(env),
            timeout=float(timeout_seconds),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


def math_is_finite_positive(value: int | float) -> bool:
    # Kept local to avoid accepting Decimal or provider-defined numeric types.
    number = float(value)
    return number > 0.0 and number != float("inf") and number == number


__all__ = [
    "AllowlistedCommandRunner",
    "PathSandbox",
    "SHELL_ENABLED",
    "SandboxViolation",
    "resolve_relative_path",
]

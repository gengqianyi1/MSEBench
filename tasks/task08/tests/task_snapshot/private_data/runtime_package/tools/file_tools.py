"""Read-only, workspace-confined file tools for the xTB benchmark.

The model-facing tools never receive a host path.  Every path is parsed as a
canonical relative path and every existing component is checked for symbolic
links and Windows reparse points before it is opened.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any


FILE_ATTRIBUTE_REPARSE_POINT = 0x400
DEFAULT_MAX_LINES = 160
DEFAULT_MAX_BYTES = 65_536
DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
_SAFE_TEXT_SUFFIXES = frozenset(
    {".json", ".log", ".out", ".err", ".txt", ".xyz", ".trj", ".inp", ".yaml", ".yml", ".md"}
)
_FORBIDDEN_NAMES = frozenset(
    {".env", "credentials.json", "secrets.json", "api_config.json"}
)


class ToolInputError(ValueError):
    """A safe, model-displayable tool input error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(stat.S_ISLNK(info.st_mode) or attributes & FILE_ATTRIBUTE_REPARSE_POINT)


def _strict_json_bytes(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolInputError("INVALID_UTF8", "file is not valid UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ToolInputError("INVALID_JSON", "file is not strict JSON") from exc


class WorkspaceSandbox:
    """Resolve and read files without allowing workspace escape."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        forbidden_roots: tuple[str, ...] = (),
    ) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise ValueError("sandbox root must be an existing directory")
        if _is_reparse_or_symlink(root_path):
            raise ValueError("sandbox root must not be a symlink or reparse point")
        self.root = root_path.resolve(strict=True)
        self.max_lines = int(max_lines)
        self.max_bytes = int(max_bytes)
        self.max_file_bytes = int(max_file_bytes)
        self.forbidden_roots = frozenset(value.casefold() for value in forbidden_roots)
        if self.max_lines < 1 or self.max_bytes < 1 or self.max_file_bytes < 1:
            raise ValueError("sandbox read limits must be positive")

    @staticmethod
    def _parts(raw_path: str) -> tuple[str, ...]:
        if not isinstance(raw_path, str) or not raw_path or len(raw_path) > 300:
            raise ToolInputError("INVALID_PATH", "path must be a non-empty bounded string")
        if "\x00" in raw_path or any(ord(character) < 32 for character in raw_path):
            raise ToolInputError("INVALID_PATH", "path contains a control character")
        windows = PureWindowsPath(raw_path)
        if windows.is_absolute() or windows.drive or raw_path.startswith(("/", "\\")):
            raise ToolInputError("ABSOLUTE_PATH_REJECTED", "absolute paths are not allowed")
        normalized = raw_path.replace("\\", "/")
        pure = PurePosixPath(normalized)
        parts = pure.parts
        if not parts or any(part in ("", ".", "..") for part in parts):
            code = "PARENT_PATH_REJECTED" if ".." in parts else "INVALID_PATH"
            raise ToolInputError(code, "path must be canonical and must not contain parent segments")
        if any(":" in part for part in parts):
            raise ToolInputError("INVALID_PATH", "drive and alternate-stream syntax is not allowed")
        if parts[-1].casefold() in _FORBIDDEN_NAMES or parts[-1].casefold().startswith(".env"):
            raise ToolInputError("FORBIDDEN_PATH", "credential-bearing files are not readable")
        return tuple(parts)

    def resolve(
        self,
        raw_path: str,
        *,
        must_exist: bool = True,
        require_file: bool = True,
    ) -> Path:
        parts = self._parts(raw_path)
        if parts[0].casefold() in self.forbidden_roots:
            raise ToolInputError("FORBIDDEN_PATH", "the requested root is not model-readable")
        cursor = self.root
        for part in parts:
            cursor = cursor / part
            if cursor.exists() or cursor.is_symlink():
                if _is_reparse_or_symlink(cursor):
                    raise ToolInputError(
                        "LINK_ESCAPE_REJECTED",
                        "symlink and reparse-point traversal is not allowed",
                    )
        try:
            resolved = cursor.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise ToolInputError("NOT_FOUND", "requested path does not exist") from exc
        if not _is_relative_to(resolved, self.root):
            raise ToolInputError("WORKSPACE_ESCAPE_REJECTED", "path escapes the case workspace")
        if must_exist and require_file and not resolved.is_file():
            raise ToolInputError("NOT_A_FILE", "requested path is not a regular file")
        return resolved

    def relative(self, path: Path) -> str:
        resolved = path.resolve(strict=False)
        if not _is_relative_to(resolved, self.root):
            raise ToolInputError("WORKSPACE_ESCAPE_REJECTED", "path escapes the case workspace")
        return resolved.relative_to(self.root).as_posix()

    def _assert_readable_text(self, path: Path) -> int:
        if path.suffix.casefold() not in _SAFE_TEXT_SUFFIXES:
            raise ToolInputError("UNSUPPORTED_TEXT_TYPE", "file type is not approved for text reading")
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise ToolInputError("FILE_TOO_LARGE", "file exceeds the bounded readable-file size")
        return size

    def _walk_files(self) -> Iterator[Path]:
        pending = [self.root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda item: item.name.casefold(), reverse=True):
                    path = Path(entry.path)
                    if entry.is_symlink() or _is_reparse_or_symlink(path):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        yield path

    def list_case_files(
        self,
        provenance_resolver: Callable[[str, int | None], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """List visible files with model-useful, neutral provenance facts.

        ``provenance_resolver`` is deliberately injected instead of importing
        :mod:`tools.provenance` here (which would create a dependency cycle).
        The registry supplies its hash-bound ``ProvenanceInspector``.  Direct
        sandbox users still receive conservative metadata: attempt ownership
        is derived from the canonical path and every other provenance claim is
        ``UNMANIFESTED``/not-current.
        """

        files: list[dict[str, Any]] = []
        for path in sorted(self._walk_files(), key=lambda item: self.relative(item)):
            relative = self.relative(path)
            if Path(relative).name.casefold() in _FORBIDDEN_NAMES or Path(relative).name.casefold().startswith(".env"):
                continue
            parts = PurePosixPath(relative).parts
            source_attempt = (
                parts[1]
                if len(parts) >= 2
                and parts[0] == "attempts"
                and re.fullmatch(
                    r"attempt_(?:000_initial_failure|(?!000)[0-9]{3})", parts[1]
                )
                else None
            )
            provenance_state = "UNMANIFESTED"
            current = False
            if provenance_resolver is not None:
                try:
                    provenance = provenance_resolver(relative, None)
                except (OSError, UnicodeError, ToolInputError, ValueError, TypeError, KeyError, IndexError):
                    # Listing must stay available when one artifact has invalid
                    # provenance or malformed XYZ content.  Fail closed for the
                    # provenance attributes instead of hiding the file.
                    provenance = {}
                if isinstance(provenance, Mapping):
                    state = provenance.get("source_kind")
                    if isinstance(state, str) and state:
                        provenance_state = state
                    current = provenance.get("current") is True
                    origin = provenance.get("origin")
                    if isinstance(origin, str):
                        match = re.search(
                            r"(?:^|[/])(?P<attempt>attempt_(?:000_initial_failure|(?!000)[0-9]{3}))(?:$|[:/])",
                            origin,
                        )
                        if match:
                            source_attempt = match.group("attempt")
            size = path.stat().st_size
            readable = bool(
                path.suffix.casefold() in _SAFE_TEXT_SUFFIXES
                and size <= self.max_file_bytes
            )
            files.append(
                {
                    "path": relative,
                    "relative_path": relative,
                    "size_bytes": size,
                    "sha256": sha256_file(path),
                    "kind": path.suffix.casefold().lstrip(".") or "file",
                    "file_type": path.suffix.casefold().lstrip(".") or "file",
                    "source_attempt": source_attempt,
                    "provenance_state": provenance_state,
                    "current": current,
                    "readable": readable,
                }
            )
        return {"files": files, "file_count": len(files)}

    @staticmethod
    def _decode_selected(data: bytes) -> str:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ToolInputError("INVALID_UTF8", "selected text is not valid UTF-8") from exc

    def read_text_range(self, raw_path: str, start_line: int, end_line: int) -> dict[str, Any]:
        if isinstance(start_line, bool) or isinstance(end_line, bool):
            raise ToolInputError("INVALID_RANGE", "line numbers must be integers")
        if start_line < 1 or end_line < start_line:
            raise ToolInputError("INVALID_RANGE", "line range must be positive and ordered")
        if end_line - start_line + 1 > self.max_lines:
            raise ToolInputError("LINE_LIMIT_EXCEEDED", "requested range exceeds the line limit")
        path = self.resolve(raw_path)
        file_size = self._assert_readable_text(path)
        selected: list[bytes] = []
        last_line = 0
        with path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                last_line = number
                if number < start_line:
                    continue
                if number > end_line:
                    break
                selected.append(line)
                if sum(len(item) for item in selected) > self.max_bytes:
                    raise ToolInputError("BYTE_LIMIT_EXCEEDED", "selected range exceeds the byte limit")
        raw = b"".join(selected)
        actual_end = start_line + len(selected) - 1 if selected else start_line - 1
        return {
            "path": self.relative(path),
            "requested_start_line": start_line,
            "requested_end_line": end_line,
            "line_start": start_line if selected else None,
            "line_end": actual_end if selected else None,
            "line_count": len(selected),
            "file_line_count_at_least": last_line,
            "file_size_bytes": file_size,
            "content": self._decode_selected(raw),
            "content_sha256": sha256_bytes(raw),
            "file_sha256": sha256_file(path),
            "truncated_by_eof": actual_end < end_line,
        }

    def tail_text(self, raw_path: str, line_count: int) -> dict[str, Any]:
        if isinstance(line_count, bool) or not isinstance(line_count, int) or not 1 <= line_count <= self.max_lines:
            raise ToolInputError("LINE_LIMIT_EXCEEDED", "tail line count is outside the allowed range")
        path = self.resolve(raw_path)
        file_size = self._assert_readable_text(path)
        tail: deque[tuple[int, bytes]] = deque(maxlen=line_count)
        total_lines = 0
        with path.open("rb") as handle:
            for total_lines, line in enumerate(handle, 1):
                tail.append((total_lines, line))
        raw = b"".join(line for _, line in tail)
        if len(raw) > self.max_bytes:
            raise ToolInputError("BYTE_LIMIT_EXCEEDED", "tail selection exceeds the byte limit")
        start = tail[0][0] if tail else None
        end = tail[-1][0] if tail else None
        return {
            "path": self.relative(path),
            "line_start": start,
            "line_end": end,
            "line_count": len(tail),
            "file_line_count": total_lines,
            "file_size_bytes": file_size,
            "content": self._decode_selected(raw),
            "content_sha256": sha256_bytes(raw),
            "file_sha256": sha256_file(path),
            "citation_eligible": False,
            "citation_instruction": (
                "Navigation only. Call read_text_range with this path and the "
                "actual line_start/line_end, then cite only its evidence_reference."
            ),
        }

    def read_json(self, raw_path: str) -> Any:
        path = self.resolve(raw_path)
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise ToolInputError("FILE_TOO_LARGE", "JSON file exceeds the readable-file size")
        return _strict_json_bytes(path.read_bytes())


def inspect_scheduler_record(sandbox: WorkspaceSandbox, path: str) -> dict[str, Any]:
    """Report execution-layer facts without declaring scientific success."""

    resolved = sandbox.resolve(path)
    sandbox._assert_readable_text(resolved)
    raw = resolved.read_bytes()
    if len(raw) > sandbox.max_bytes:
        raise ToolInputError("BYTE_LIMIT_EXCEEDED", "scheduler record exceeds the per-call byte limit")
    text = WorkspaceSandbox._decode_selected(raw)
    parsed: Any = None
    if resolved.suffix.casefold() == ".json":
        parsed = _strict_json_bytes(raw)
    state: Any = None
    exit_code: Any = None
    timed_out: Any = None
    if isinstance(parsed, Mapping):
        for key in ("scheduler_state", "state", "status", "job_state"):
            if key in parsed:
                state = parsed[key]
                break
        for key in ("exit_code", "return_code", "returncode"):
            if key in parsed:
                exit_code = parsed[key]
                break
        value = parsed.get("timed_out")
        timed_out = value if type(value) is bool else None
    else:
        state_match = re.search(r"(?im)^\s*(?:scheduler_state|state|status|job_state)\s*[:=]\s*([^\r\n]+)", text)
        exit_match = re.search(r"(?im)^\s*(?:exit_code|return_code)\s*[:=]\s*(-?\d+)", text)
        timeout_value = re.search(
            r"(?im)^\s*(?:timed_out|timeout)\s*[:=]\s*(true|false|yes|no|1|0)\s*$",
            text,
        )
        state = state_match.group(1).strip() if state_match else None
        exit_code = int(exit_match.group(1)) if exit_match else None
        if timeout_value:
            timed_out = timeout_value.group(1).casefold() in {"true", "yes", "1"}
        else:
            timed_out = bool(re.search(r"(?i)time(?:d)?\s*out|walltime.*exceed", text))
    state_text = "" if state is None else str(state).strip().upper()
    # Normalize only execution-layer states.  The ordering is intentional:
    # structured state fields outrank incidental prose in notes/comments.
    timeout_evidence = bool(
        re.search(r"\b(?:TIMEOUT|TIMED[ _-]?OUT|TIME[ _-]?LIMIT)\b", state_text)
        or re.search(r"(?i)\btimed\s+out\b|walltime.*(?:exceed|reach)|time\s+limit.*(?:exceed|reach)", text)
    )
    if timed_out is True or timeout_evidence:
        canonical_state = "TIMEOUT"
    elif re.search(r"\b(?:OUT[ _-]?OF[ _-]?MEMORY|OOM|MEMORY[ _-]?LIMIT)\b", state_text):
        canonical_state = "OUT_OF_MEMORY"
    elif re.search(r"\bNODE[ _-]?FAIL(?:ED|URE)?\b", state_text):
        canonical_state = "NODE_FAIL"
    elif re.search(r"\bPREEMPT(?:ED|ION)?\b", state_text):
        canonical_state = "PREEMPTED"
    elif re.search(r"\bCANCEL(?:LED|ED|LATION)?\b", state_text):
        canonical_state = "CANCELLED"
    elif re.search(r"\b(?:FAILED|FAILURE|ERROR|ABORTED)\b", state_text):
        canonical_state = "FAILED"
    elif re.search(r"\b(?:COMPLETED|COMPLETE|DONE|SUCCESS|SUCCEEDED)\b", state_text):
        canonical_state = "COMPLETED"
    elif isinstance(exit_code, int) and not isinstance(exit_code, bool):
        canonical_state = "COMPLETED" if exit_code == 0 else "FAILED"
    else:
        canonical_state = "UNKNOWN"
    return {
        "path": sandbox.relative(resolved),
        "record_sha256": sha256_bytes(raw),
        "state": state,
        "scheduler_state": state,
        "canonical_state": canonical_state,
        "exit_code": exit_code,
        "timed_out": timed_out is True,
        "execution_record_present": True,
        "scientific_success_inference": False,
        "scientific_completion": "NOT_ASSESSED",
        "neutrality_note": (
            "execution-layer scheduler/process state is not scientific convergence evidence"
        ),
    }


__all__ = [
    "ToolInputError",
    "WorkspaceSandbox",
    "inspect_scheduler_record",
    "sha256_bytes",
    "sha256_file",
]

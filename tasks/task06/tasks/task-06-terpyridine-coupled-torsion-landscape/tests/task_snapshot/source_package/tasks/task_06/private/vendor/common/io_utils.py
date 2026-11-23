"""Small, dependency-light I/O and sandbox helpers used across the benchmark."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux is the supported benchmark host
    fcntl = None  # type: ignore[assignment]


_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|secret|password|passwd|credential|authorization|cookie)", re.I)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def native_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path that remains usable beyond Win32 MAX_PATH."""

    absolute = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute.lstrip("\\"))
    return Path("\\\\?\\" + absolute)


def require_linux_execution(component: str) -> None:
    """Reject execution on hosts that cannot provide the Linux sandbox contract."""

    if sys.platform != "linux":
        raise RuntimeError(
            f"{component} execution requires Linux (Ubuntu/WSL2 is supported); "
            f"this host is {sys.platform}. Parsing and static evaluation remain portable."
        )


def bubblewrap_command(
    argv: list[str],
    *,
    cwd: str | Path,
    writable_roots: list[str | Path],
    readonly_roots: list[str | Path] | None = None,
    masked_files: Mapping[str | Path, str | Path] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build a fail-closed, networkless bubblewrap command.

    The host root is deliberately *not* mounted.  Only runtime directories and
    explicitly listed roots are visible, preventing accidental access to sibling
    evaluator, credential, or project directories.
    """

    require_linux_execution("sandboxed subprocess")
    executable = shutil.which("bwrap") or shutil.which("bubblewrap")
    if executable is None:
        raise RuntimeError("bubblewrap is required for benchmark execution but was not found on PATH")

    command = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]

    mounted: set[str] = set()

    def add(path_value: str | Path, mode: str) -> None:
        requested = Path(path_value)
        destination = requested.absolute()
        source = requested.resolve(strict=True)
        if destination == Path(destination.anchor) or source == Path(source.anchor):
            raise ValueError("sandbox policy forbids mounting the host filesystem root")
        target = str(destination)
        if target in mounted:
            return
        command.extend([mode, str(source), target])
        mounted.add(target)

    # Runtime-only system mounts.  Do not bind /, /home, /mnt, or /etc broadly.
    for candidate in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(candidate).exists():
            add(candidate, "--ro-bind")
    for candidate in ("/etc/ld.so.cache", "/etc/localtime"):
        if Path(candidate).exists():
            add(candidate, "--ro-bind")
    for root in readonly_roots or []:
        add(root, "--ro-bind")
    for root in writable_roots:
        add(root, "--bind")
    masked_targets: list[str] = []
    for target_value, source_value in (masked_files or {}).items():
        target = Path(target_value).absolute()
        source = Path(source_value).resolve(strict=True)
        if not source.is_file():
            raise ValueError("sandbox mask source must be a file")
        if target == Path(target.anchor):
            raise ValueError("sandbox policy forbids masking the host filesystem root")
        command.extend(["--ro-bind", str(source), str(target)])
        masked_targets.append(str(target))

    working_directory = Path(cwd).resolve(strict=True)
    if not any(working_directory == Path(root).resolve() or working_directory.is_relative_to(Path(root).resolve()) for root in writable_roots):
        raise ValueError("sandbox cwd must be inside a writable root")
    command.extend(["--chdir", str(working_directory), "--", *argv])
    return command, {
        "engine": "bubblewrap",
        "executable": executable,
        "network_namespace": "unshared",
        "host_root_mounted": False,
        "writable_roots": [str(Path(root).resolve()) for root in writable_roots],
        "readonly_roots": sorted(mounted - {str(Path(root).resolve()) for root in writable_roots}),
        "masked_files": sorted(masked_targets),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with native_path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, data: bytes) -> None:
    target = native_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str) -> None:
    _atomic_replace(Path(path), text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, payload)


def append_jsonl(path: str | Path, value: Any) -> None:
    target = native_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_yaml(path: str | Path, *, default: Any = None) -> Any:
    target = native_path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return default if value is None else value


def redact(value: Any) -> Any:
    """Recursively redact credential-like fields and bearer strings."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            result[text_key] = "<redacted>" if _SECRET_KEY.search(text_key) else redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        return _BEARER.sub("Bearer <redacted>", value)
    return value


def sanitize_environment(base_environment: Mapping[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if base_environment is None else base_environment)
    result: dict[str, str] = {}
    for key, value in source.items():
        if _SECRET_KEY.search(key):
            continue
        result[str(key)] = str(value)
    # xTB needs HOME for temporary/config resolution. Avoid inheriting a missing value.
    result.setdefault("HOME", str(Path.home()))
    result.setdefault("PATH", os.environ.get("PATH", ""))
    return result


def safe_resolve(root: str | Path, value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path inside *root*, rejecting traversal and symlink escapes."""

    root_path = Path(os.path.realpath(os.path.abspath(os.fspath(root))))
    if not native_path(root_path).is_dir():
        raise FileNotFoundError(root_path)
    raw = str(value)
    if "\x00" in raw:
        raise ValueError("path contains NUL")
    input_path = Path(raw)
    if input_path.is_absolute():
        candidate = input_path
    else:
        candidate = root_path / input_path
    resolved = Path(os.path.realpath(os.path.abspath(os.fspath(candidate))))
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"path escapes sandbox root: {value}") from exc
    if must_exist and not native_path(resolved).exists():
        raise FileNotFoundError(resolved)
    return resolved


def relative_posix(path: str | Path, root: str | Path) -> str:
    resolved_path = Path(path).resolve(strict=False)
    resolved_root = Path(root).resolve(strict=True)
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside root: {resolved_path}") from exc


def validate_namespace(value: str) -> str:
    if not _NAMESPACE.fullmatch(value):
        raise ValueError("namespace must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    return value


def file_hash_manifest(root: str | Path) -> dict[str, str]:
    logical_base = Path(os.path.realpath(os.path.abspath(os.fspath(root))))
    base = native_path(logical_base)
    if not base.is_dir():
        raise FileNotFoundError(logical_base)
    result: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in evidence trees: {path}")
        if path.is_file():
            result[path.relative_to(base).as_posix()] = sha256_file(path)
    return result


class FileLock:
    """Advisory process lock backed by ``flock`` on supported Unix hosts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: Any | None = None

    def __enter__(self) -> "FileLock":
        if fcntl is None:  # pragma: no cover
            raise RuntimeError("FileLock requires fcntl on the benchmark host")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

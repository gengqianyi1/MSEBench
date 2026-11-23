"""Cross-platform filesystem I/O with stable logical paths.

Windows extended paths are an implementation detail: callers receive relative
logical paths for manifests and ZIP members, while all filesystem operations
use the native path returned by :func:`native_path`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator


def native_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute filesystem path usable beyond Win32 ``MAX_PATH``."""
    target = os.path.abspath(os.fspath(path))
    if os.name != "nt" or target.startswith("\\\\?\\"):
        return Path(target)
    if target.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + target.lstrip("\\"))
    return Path("\\\\?\\" + target)


def path_is_file(path: str | os.PathLike[str]) -> bool:
    return native_path(path).is_file()


def path_is_dir(path: str | os.PathLike[str]) -> bool:
    return native_path(path).is_dir()


def read_bytes(path: str | os.PathLike[str]) -> bytes:
    return native_path(path).read_bytes()


def read_text(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> str:
    return native_path(path).read_text(encoding=encoding)


def file_size(path: str | os.PathLike[str]) -> int:
    return native_path(path).stat().st_size


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with native_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_tree_files(root: str | os.PathLike[str]) -> Iterator[tuple[Path, Path]]:
    """Yield ``(logical_relative_path, native_file_path)`` in stable order."""
    io_root = native_path(root)
    if not io_root.is_dir():
        return
    paths = sorted(io_root.rglob("*"), key=lambda item: item.relative_to(io_root).as_posix())
    for path in paths:
        if path.is_file():
            yield Path(path.relative_to(io_root).as_posix()), path


def iter_directories(root: str | os.PathLike[str]) -> Iterator[tuple[Path, Path]]:
    """Yield immediate child directories as logical/native path pairs."""
    io_root = native_path(root)
    if not io_root.is_dir():
        return
    for path in sorted(io_root.iterdir(), key=lambda item: item.name):
        if path.is_dir():
            yield Path(path.name), path


__all__ = [
    "file_size",
    "iter_directories",
    "iter_tree_files",
    "native_path",
    "path_is_dir",
    "path_is_file",
    "read_bytes",
    "read_text",
    "sha256_file",
]

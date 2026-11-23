from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_entries(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files, key=lambda value: value.relative_to(root).as_posix())
    ]


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def validate_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "MANIFEST.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"passed": False, "errors": ["MANIFEST_MISSING"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"passed": False, "errors": ["MANIFEST_INVALID"]}
    if not isinstance(manifest, dict):
        return {"passed": False, "errors": ["MANIFEST_INVALID"]}
    manifest_items = manifest.get("files")
    if not isinstance(manifest_items, list):
        return {"passed": False, "errors": ["MANIFEST_FILES_INVALID"]}
    listed: dict[str, dict[str, Any]] = {}
    seen_casefold: set[str] = set()
    for item in manifest_items:
        if not isinstance(item, dict):
            errors.append("MANIFEST_ENTRY_INVALID")
            continue
        relative_value = item.get("path")
        if not isinstance(relative_value, str):
            errors.append("MANIFEST_PATH_INVALID")
            continue
        relative = relative_value
        pure = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or pure.is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
            or pure.as_posix() != relative
            or relative == "MANIFEST.json"
        ):
            errors.append("MANIFEST_UNSAFE_PATH")
            continue
        collision = relative.casefold()
        if relative in listed or collision in seen_casefold:
            errors.append("MANIFEST_DUPLICATE_PATH")
            continue
        seen_casefold.add(collision)
        listed[relative] = item
        path = root.joinpath(*pure.parts)
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append("MANIFEST_UNSAFE_PATH")
            continue
        if not path.is_file():
            errors.append("MANIFEST_FILE_MISSING")
            continue
        size_value = item.get("size_bytes")
        if not isinstance(size_value, int) or isinstance(size_value, bool) or size_value < 0:
            errors.append("MANIFEST_SIZE_INVALID")
        elif path.stat().st_size != size_value:
            errors.append("MANIFEST_SIZE_MISMATCH")
        hash_value = item.get("sha256")
        if not isinstance(hash_value, str) or SHA256_RE.fullmatch(hash_value) is None:
            errors.append("MANIFEST_HASH_INVALID")
        elif sha256_file(path) != hash_value.upper():
            errors.append("MANIFEST_HASH_MISMATCH")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(listed) != actual:
        errors.append("MANIFEST_INVENTORY_MISMATCH")
    count_value = manifest.get("file_count_excluding_manifest")
    if (
        not isinstance(count_value, int)
        or isinstance(count_value, bool)
        or count_value != len(listed)
    ):
        errors.append("MANIFEST_COUNT_MISMATCH")
    total_value = manifest.get("total_bytes_excluding_manifest")
    actual_total = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if (
        not isinstance(total_value, int)
        or isinstance(total_value, bool)
        or total_value != actual_total
    ):
        errors.append("MANIFEST_TOTAL_BYTES_MISMATCH")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "file_count": len(listed),
    }

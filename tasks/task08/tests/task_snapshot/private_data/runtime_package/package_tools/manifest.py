from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


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
    listed: dict[str, dict[str, Any]] = {}
    for item in manifest.get("files", []):
        relative = str(item.get("path", ""))
        if relative in listed:
            errors.append("MANIFEST_DUPLICATE_PATH")
            continue
        listed[relative] = item
        path = root / Path(relative)
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append("MANIFEST_UNSAFE_PATH")
            continue
        if not path.is_file():
            errors.append("MANIFEST_FILE_MISSING")
            continue
        if path.stat().st_size != int(item.get("size_bytes", -1)):
            errors.append("MANIFEST_SIZE_MISMATCH")
        if sha256_file(path) != str(item.get("sha256", "")).upper():
            errors.append("MANIFEST_HASH_MISMATCH")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if set(listed) != actual:
        errors.append("MANIFEST_INVENTORY_MISMATCH")
    if int(manifest.get("file_count_excluding_manifest", -1)) != len(listed):
        errors.append("MANIFEST_COUNT_MISMATCH")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "file_count": len(listed),
    }

#!/usr/bin/env python3
"""Create the Tasks 07-09 delivery manifest and portable ZIP."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT.parent / "MatBench_Tasks07-09_Organizer_20260911_v3.zip"
MANIFEST = ROOT / "delivery_manifest.sha256"
BLOCKED_PARTS = {"__pycache__", ".pytest_cache", ".test_tmp"}
BLOCKED_PREFIXES = {
    "framework/release_evidence/task08/migration_regression_sixth_review/",
}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if path == MANIFEST or path.suffix == ".pyc":
        return False
    if any(part in BLOCKED_PARTS for part in path.relative_to(ROOT).parts):
        return False
    return not any(relative.startswith(prefix) for prefix in BLOCKED_PREFIXES)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def files() -> list[Path]:
    return sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and included(path)),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def main() -> int:
    selected = files()
    MANIFEST.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in selected),
        encoding="ascii",
        newline="\n",
    )
    selected.append(MANIFEST)
    selected.sort(key=lambda path: path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True
    ) as archive:
        for path in selected:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative)
            info.create_system = 3
            info.date_time = (2026, 9, 11, 0, 0, 0)
            executable = path.suffix in {".sh", ".py"} or path.name in {
                "conda", "test.sh", "solve.sh"
            }
            info.external_attr = ((0o100755 if executable else 0o100644) << 16)
            info.compress_type = zipfile.ZIP_STORED if path.suffix == ".tar" else zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    print(f"files={len(selected)}")
    print(f"zip={OUTPUT}")
    print(f"zip_sha256={sha256(OUTPUT)}")
    print(f"zip_bytes={OUTPUT.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the trusted-author handoff ZIP plus reproducible manifest and checksum."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "chembench-author-handoff.zip"
MANIFEST = ROOT / "AUTHOR_PACKAGE_MANIFEST.json"
CHECKSUM = ROOT / "dist" / "chembench-author-handoff.zip.sha256"

ROOT_FILES = {
    "README.md", "MIGRATION_HANDOFF.md", "MODIFICATION_REPORT.md",
    "AUDIT_REPORT_20260827.md", "AUTHOR_HANDOFF_PACKAGE.md",
    "AUTHOR_REVIEW_20260829.md", "TASK01_03_LLM_EVALUATION_READINESS_20260830.md",
    "AUTHOR_FILE_MATRIX.md", "AUTHOR_ACCEPTANCE_CHECKLIST.md",
    "AUTHOR_CONFIRMATION_TEMPLATE.md", "AUTHOR_PACKAGE_MANIFEST.json",
    "TASK_INVENTORY.json", "pyproject.toml", "chembench",
}
ROOT_DIRS = {"tasks", "scripts", "agents", "msebench_harness", "schemas", "templates", "datasets", "docs", "adapter"}
EXCLUDED_PARTS = {".audit-venv", ".harbor-wheel", "__pycache__", ".pytest_cache", "parity", "dist", "legacy_upstream", ".git"}
SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?i)authorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9_./+\-=]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def selected_files() -> list[Path]:
    files: set[Path] = set()
    for name in ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            files.add(path)
    for name in ROOT_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts) and path.suffix != ".pyc":
                files.add(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(files: list[Path]) -> None:
    errors: list[str] = []
    for path in files:
        rel = PurePosixPath(path.relative_to(ROOT).as_posix())
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"unsafe path: {rel}")
        if path.stat().st_size <= 5_000_000:
            data = path.read_bytes()
            if any(pattern.search(data) for pattern in SECRET_PATTERNS):
                errors.append(f"possible embedded secret: {rel}")
    if errors:
        raise SystemExit("\n".join(f"ERROR {e}" for e in errors))


def main() -> int:
    initial = selected_files()
    audit(initial)
    manifest = {
        "package_type": "trusted-author-private-development-handoff",
        "warning": "Not a participant or leaderboard release; may contain organizer-private source.",
        "file_count": len(initial) + (0 if MANIFEST in initial else 1),
        "files": [{"path": p.relative_to(ROOT).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)} for p in initial if p != MANIFEST],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = selected_files()
    audit(files)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    digest = sha256(OUTPUT)
    CHECKSUM.write_text(f"{digest}  {OUTPUT.name}\n", encoding="ascii")
    print(f"files={len(files)} path={OUTPUT}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

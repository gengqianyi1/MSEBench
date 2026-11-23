from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,})"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    manifest_path = ROOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    version = str(manifest.get("benchmark_version") or manifest.get("version") or "")
    if version != "1.3":
        errors.append(f"manifest release version is {version!r}, expected '1.3'")

    forbidden_public_files = {
        "scripts/batch_v2.ps1",
        "scripts/batch_marcus.py",
        "scripts/validate_designs.py",
        "configs/api_config_codex_scripted.json",
        "configs/codex_scripted.yaml",
        "examples/satisfied_molecules.csv",
        "examples/satisfied_molecules.json",
    }
    manifest_files = set(manifest.get("files", []))
    for relative in sorted(forbidden_public_files):
        if relative in manifest_files or (ROOT / relative).exists():
            errors.append(f"forbidden oracle/replay asset in public package: {relative}")

    for relative in manifest.get("files", []):
        if not (ROOT / relative).is_file():
            errors.append(f"missing manifest file: {relative}")
    for relative, expected in (manifest.get("sha256") or {}).items():
        if relative == "MANIFEST.json":
            continue  # skip self-hash (circular by design)
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing hashed file: {relative}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or "runs" in path.relative_to(ROOT).parts:
            continue
        try:
            if path.suffix == ".json":
                text = path.read_text(encoding="utf-8-sig")
                json.loads(text)
                if "scripted_responses" in text:
                    errors.append(f"possible scripted replay content: {path.relative_to(ROOT)}")
            elif path.suffix in {".yaml", ".yml"}:
                text = path.read_text(encoding="utf-8-sig")
                yaml.safe_load(text)
                if "scripted_responses" in text:
                    errors.append(f"possible scripted replay content: {path.relative_to(ROOT)}")
            elif path.suffix in {".md", ".py", ".ps1", ".cff"}:
                text = path.read_text(encoding="utf-8-sig")
                if SECRET_PATTERN.search(text):
                    errors.append(f"possible embedded credential: {path.relative_to(ROOT)}")
                if path.name != "validate_package.py" and path.suffix in {".py", ".ps1"} and (
                    "scripted_responses" in text or "Batch run all 34 SMILES" in text
                ):
                    errors.append(f"possible oracle/replay content: {path.relative_to(ROOT)}")
        except Exception as exc:
            errors.append(f"parse failure {path.relative_to(ROOT)}: {exc}")

    report = {"release": version, "errors": errors, "passed": not errors}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

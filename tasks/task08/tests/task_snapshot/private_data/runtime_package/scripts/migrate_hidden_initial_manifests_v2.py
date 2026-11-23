"""One-time deterministic migration for v2 initial-attempt manifest path scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
    }


for case_root in sorted((ROOT / "organizer_private" / "cases").iterdir()):
    attempt = case_root / "attempts" / "attempt_000_initial_failure"
    artifact_path = attempt / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for entry in artifact["entries"]:
        entry["path"] = Path(entry["path"]).name
    write_json(artifact_path, artifact)

    manifest_path = case_root / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record(case_root, path)
        for path in sorted(case_root.rglob("*"), key=lambda item: item.relative_to(case_root).as_posix())
        if path.is_file() and path.name != "case_manifest.json"
    ]
    write_json(manifest_path, manifest)

    config_path = ROOT / "organizer_configs" / f"{case_root.name}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["task"]["case_manifest"]["sha256"] = digest(manifest_path)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

print("MIGRATED=5")

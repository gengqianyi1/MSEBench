"""Preserve raw native bytes and publish a path-redacted model-readable log."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_PROGRAM = re.compile(
    rb"(?im)^(\s*program call\s*:\s*)[A-Za-z]:[\\/][^\r\n ]*xtb\.exe(\s+)"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def record(root: Path, path: Path, *, short: bool = False) -> dict[str, object]:
    return {
        "path": path.name if short else path.relative_to(root).as_posix(),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
    }


for case_root in sorted((ROOT / "organizer_private" / "cases").iterdir()):
    attempt = case_root / "attempts" / "attempt_000_initial_failure"
    stdout = attempt / "stdout.log"
    raw_path = attempt / "stdout.raw.bin"
    raw = stdout.read_bytes()
    if raw_path.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE:{raw_path}")
    raw_path.write_bytes(raw)
    redacted, substitutions = ABSOLUTE_PROGRAM.subn(rb"\1<XTB_EXECUTABLE>\2", raw)
    stdout.write_bytes(redacted)

    command_path = attempt / "command.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))
    command["stdout_raw_path"] = "stdout.raw.bin"
    command["stdout_raw_sha256"] = digest(raw_path)
    command["stdout_sha256"] = digest(stdout)
    command["stdout_persistence_policy"] = {
        "raw_bytes_preserved": True,
        "model_readable_path_redaction": substitutions > 0,
        "replacement": "<XTB_EXECUTABLE>",
    }
    write_json(command_path, command)

    artifact_path = attempt / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["entries"] = [
        {**record(case_root, path, short=True), "role": "initial_failure_artifact"}
        for path in sorted(attempt.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    write_json(artifact_path, artifact)

    manifest_path = case_root / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["failure_injection"]["disclosure"] = (
        "Native stdout bytes are preserved as stdout.raw.bin; stdout.log only redacts the host executable path."
    )
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
        encoding="utf-8", newline="\n",
    )

print("PATH_REDACTION_MIGRATED=5")

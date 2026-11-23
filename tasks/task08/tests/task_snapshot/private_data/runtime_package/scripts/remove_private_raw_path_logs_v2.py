"""Remove generated raw logs that contain host paths; retain their SHA-256 receipts."""

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
        encoding="utf-8", newline="\n",
    )


def record(root: Path, path: Path, *, short: bool = False) -> dict[str, object]:
    return {
        "path": path.name if short else path.relative_to(root).as_posix(),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
    }


for case_root in sorted((ROOT / "organizer_private" / "cases").iterdir()):
    attempt = (case_root / "attempts" / "attempt_000_initial_failure").resolve(strict=True)
    raw = (attempt / "stdout.raw.bin").resolve(strict=True)
    if raw.parent != attempt or not raw.is_file():
        raise RuntimeError("RAW_LOG_TARGET_INVALID")
    command_path = attempt / "command.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))
    if command.get("stdout_raw_sha256") != digest(raw):
        raise RuntimeError("RAW_LOG_HASH_MISMATCH")
    raw.unlink()
    command["stdout_raw_path"] = None
    command["stdout_persistence_policy"] = {
        "raw_sha256_retained": True,
        "raw_bytes_omitted_from_release_for_host_path_privacy": True,
        "model_readable_path_redaction": command["stdout_persistence_policy"]["model_readable_path_redaction"],
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
        "stdout.log redacts only the host executable path; the pre-redaction SHA-256 is retained in command.json."
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

print("REMOVED_PRIVATE_RAW_LOGS=5")

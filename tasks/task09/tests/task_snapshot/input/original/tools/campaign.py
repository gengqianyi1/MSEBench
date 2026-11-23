"""Fail-closed campaign snapshot discovery and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.science import canonical_json_bytes, sha256_file


CAMPAIGN_IDENTITY = {
    "benchmark_version": "1.1.0",
    "case_id": "2_fluorobiphenyl_central_torsion_v1",
    "formal_campaign_id": "formal_campaign_v1_1_rc1",
    "protocol": "adaptive_torsion_sampling_tool_loop/1.1",
    "task_id": "adaptive_xtb_torsion_profile_agent_v1_1",
}
INCLUDED_TOP_LEVEL = {
    "baselines",
    "benchmark_case",
    "evaluator",
    "framework",
    "package_tools",
    "prompts",
    "scripts",
    "tests",
    "tools",
}
INCLUDED_ROOT_FILES = {
    ".env.example",
    "environment.yml",
    "pytest.ini",
    "requirements.txt",
    "TASK.md",
}
EXCLUDED_PATHS = {
    "configs/campaign_snapshot.json",
    "configs/release.json",
}
INCLUDED_CONFIGS = {
    "configs/models.example.yaml",
    "configs/task.yaml",
    "configs/tool_contracts.json",
}
INCLUDED_PRIVATE_ASSETS = {
    "evaluator_private/oracle_commitment.json",
    "evaluator_private/oracle_profile.json",
    "evaluator_private/oracle_provenance.json",
}
ALLOWED_SUFFIXES = {
    ".example",
    ".inp",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".mol",
    ".ps1",
    ".py",
    ".sh",
    ".txt",
    ".xyz",
    ".yaml",
    ".yml",
}
ALLOWED_EXTENSIONLESS = {
    ".xtboptok",
    "charges",
    "energy",
    "gradient",
    "wbo",
    "xtbrestart",
}


class CampaignSnapshotError(RuntimeError):
    pass


def discover_campaign_files(root: Path) -> list[Path]:
    resolved = root.resolve(strict=True)
    files: list[Path] = []
    for path in resolved.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(resolved)
        posix = relative.as_posix()
        if "__pycache__" in relative.parts or posix in EXCLUDED_PATHS:
            continue
        if len(relative.parts) == 1:
            if relative.name in INCLUDED_ROOT_FILES:
                files.append(path)
            continue
        if relative.parts[0] == "configs":
            if posix in INCLUDED_CONFIGS:
                files.append(path)
            continue
        if relative.parts[0] == "evaluator_private":
            if posix in INCLUDED_PRIVATE_ASSETS:
                files.append(path)
            continue
        if relative.parts[0] not in INCLUDED_TOP_LEVEL:
            continue
        if (
            relative.suffix.casefold() in ALLOWED_SUFFIXES
            or relative.name in ALLOWED_EXTENSIONLESS
        ):
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(resolved).as_posix())


def campaign_file_entries(root: Path) -> list[dict[str, object]]:
    resolved = root.resolve(strict=True)
    return [
        {
            "path": path.relative_to(resolved).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in discover_campaign_files(resolved)
    ]


def content_commitment(entries: list[dict[str, object]]) -> str:
    value = {"identity": CAMPAIGN_IDENTITY, "files": entries}
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def validate_campaign_snapshot(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    snapshot_path = resolved / "configs" / "campaign_snapshot.json"
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise CampaignSnapshotError("CAMPAIGN_SNAPSHOT_INVALID") from exc
    for field, expected in CAMPAIGN_IDENTITY.items():
        if payload.get(field) != expected:
            raise CampaignSnapshotError("CAMPAIGN_IDENTITY_MISMATCH")
    current = campaign_file_entries(resolved)
    if payload.get("files") != current:
        raise CampaignSnapshotError("CAMPAIGN_FILE_INVENTORY_MISMATCH")
    if payload.get("file_count") != len(current):
        raise CampaignSnapshotError("CAMPAIGN_FILE_COUNT_MISMATCH")
    if payload.get("total_bytes") != sum(
        int(item["size_bytes"]) for item in current
    ):
        raise CampaignSnapshotError("CAMPAIGN_TOTAL_BYTES_MISMATCH")
    expected_commitment = content_commitment(current)
    if payload.get("content_commitment_sha256") != expected_commitment:
        raise CampaignSnapshotError("CAMPAIGN_CONTENT_COMMITMENT_MISMATCH")
    return {
        **CAMPAIGN_IDENTITY,
        "file_count": len(current),
        "content_commitment_sha256": expected_commitment,
        "snapshot_file_sha256": sha256_file(snapshot_path),
    }


__all__ = [
    "CAMPAIGN_IDENTITY",
    "CampaignSnapshotError",
    "campaign_file_entries",
    "content_commitment",
    "discover_campaign_files",
    "validate_campaign_snapshot",
]

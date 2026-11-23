from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.evaluate import EvaluationError, _campaign_snapshot
from tools.campaign import (
    CAMPAIGN_IDENTITY,
    CampaignSnapshotError,
    campaign_file_entries,
    content_commitment,
    discover_campaign_files,
    validate_campaign_snapshot,
)
from tools.science import canonical_json_bytes


def _campaign_tree(root: Path) -> None:
    (root / "configs").mkdir()
    (root / "tools").mkdir()
    (root / "evaluator_private").mkdir()
    (root / "TASK.md").write_text("task\n", encoding="utf-8")
    (root / ".env.example").write_text("KEY=PLACEHOLDER\n", encoding="utf-8")
    (root / "configs" / "task.yaml").write_text("task: {}\n", encoding="utf-8")
    (root / "configs" / "models.example.yaml").write_text(
        "models: {}\n", encoding="utf-8"
    )
    (root / "configs" / "tool_contracts.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (root / "configs" / "release.json").write_text(
        '{"status":"BLOCKED"}\n', encoding="utf-8"
    )
    (root / "tools" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name in (
        "oracle_commitment.json",
        "oracle_profile.json",
        "oracle_provenance.json",
    ):
        (root / "evaluator_private" / name).write_text("{}\n", encoding="utf-8")


def _write_snapshot(root: Path) -> dict:
    entries = campaign_file_entries(root)
    payload = {
        "schema_version": "1.0",
        **CAMPAIGN_IDENTITY,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
        "content_commitment_sha256": content_commitment(entries),
        "files": entries,
    }
    (root / "configs" / "campaign_snapshot.json").write_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def test_campaign_discovery_is_allowlisted_and_excludes_mutable_release(
    tmp_path: Path,
) -> None:
    _campaign_tree(tmp_path)
    relative = {
        path.relative_to(tmp_path).as_posix()
        for path in discover_campaign_files(tmp_path)
    }
    assert "configs/release.json" not in relative
    assert "configs/campaign_snapshot.json" not in relative
    assert "tools/runtime.py" in relative
    assert relative.issuperset(
        {
            "TASK.md",
            "configs/task.yaml",
            "evaluator_private/oracle_profile.json",
        }
    )


def test_campaign_snapshot_verifies_exact_inventory(tmp_path: Path) -> None:
    _campaign_tree(tmp_path)
    payload = _write_snapshot(tmp_path)
    result = validate_campaign_snapshot(tmp_path)
    assert result["file_count"] == payload["file_count"]
    assert (
        result["content_commitment_sha256"]
        == payload["content_commitment_sha256"]
    )


def test_campaign_snapshot_detects_source_tamper(tmp_path: Path) -> None:
    _campaign_tree(tmp_path)
    _write_snapshot(tmp_path)
    (tmp_path / "tools" / "runtime.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(
        CampaignSnapshotError, match="CAMPAIGN_FILE_INVENTORY_MISMATCH"
    ):
        validate_campaign_snapshot(tmp_path)


def test_campaign_snapshot_detects_added_source_file(tmp_path: Path) -> None:
    _campaign_tree(tmp_path)
    _write_snapshot(tmp_path)
    (tmp_path / "tools" / "later.py").write_text("NEW = True\n", encoding="utf-8")
    with pytest.raises(
        CampaignSnapshotError, match="CAMPAIGN_FILE_INVENTORY_MISMATCH"
    ):
        validate_campaign_snapshot(tmp_path)


def test_independent_evaluator_recomputes_same_campaign_commitment(
    tmp_path: Path,
) -> None:
    _campaign_tree(tmp_path)
    payload = _write_snapshot(tmp_path)
    result = _campaign_snapshot(tmp_path)
    assert result["content_commitment_sha256"] == payload[
        "content_commitment_sha256"
    ]
    assert result["file_count"] == payload["file_count"]


def test_independent_evaluator_rejects_campaign_source_tamper(
    tmp_path: Path,
) -> None:
    _campaign_tree(tmp_path)
    _write_snapshot(tmp_path)
    (tmp_path / "tools" / "runtime.py").write_text(
        "VALUE = 'tampered'\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="file inventory mismatch"):
        _campaign_snapshot(tmp_path)

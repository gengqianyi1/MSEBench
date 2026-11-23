from __future__ import annotations

from pathlib import Path

from tools import ToolRegistry


ROOT = Path(__file__).resolve().parents[2]


def _call(registry: ToolRegistry, sequence: int, name: str, arguments: dict) -> dict:
    envelope = registry.dispatch(name, arguments, call_id=f"e2e-{sequence:02d}")
    assert envelope["ok"] is True, envelope.get("error", {}).get("code")
    return envelope["result"]


def test_frozen_failure_can_be_diagnosed_through_public_tools_only() -> None:
    registry = ToolRegistry(ROOT)
    listing = _call(registry, 1, "list_case_files", {})
    assert listing["file_count"] >= 10

    read = _call(
        registry,
        2,
        "read_text_range",
        {
            "path": "attempts/attempt_000_initial_failure/stdout.log",
            "start_line": 284,
            "end_line": 284,
        },
    )
    assert "FAILED TO CONVERGE" in read["content"]

    parsed = _call(
        registry,
        3,
        "parse_xtb_log",
        {"path": "attempts/attempt_000_initial_failure/stdout.log"},
    )
    assert parsed["latest_segment"]["optimization_failure_observed"] is True

    scheduler = _call(
        registry,
        4,
        "inspect_scheduler_record",
        {"path": "attempts/attempt_000_initial_failure/scheduler.json"},
    )
    assert scheduler["state"] == "COMPLETED"
    assert scheduler["scientific_success_inference"] is False

    xyz = _call(
        registry,
        5,
        "validate_xyz",
        {"path": "attempts/attempt_000_initial_failure/xtbopt.log"},
    )
    assert xyz["valid"] is True
    assert xyz["complete_frame_count"] >= 1

    provenance = _call(
        registry,
        6,
        "inspect_artifact_provenance",
        {
            "path": "attempts/attempt_000_initial_failure/xtbopt.log",
            "frame_index": xyz["selected_frame_index"],
        },
    )
    assert provenance["source_kind"] == "CURRENT_TRAJECTORY_FRAME"
    assert provenance["restart_allowed"] is True
    assert provenance["success_evidence_allowed"] is False


def test_legacy_trap_is_rejected_as_restart_source() -> None:
    registry = ToolRegistry(ROOT)
    provenance = _call(
        registry,
        1,
        "inspect_artifact_provenance",
        {"path": "legacy/xtbopt.xyz", "frame_index": 0},
    )
    assert provenance["source_kind"] == "LEGACY_ARTIFACT"
    assert provenance["current"] is False
    assert provenance["restart_allowed"] is False
    assert provenance["success_evidence_allowed"] is False

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.file_tools import ToolInputError, WorkspaceSandbox
from tools.xtb_log_parser import parse_xtb_log_text
from tools.xyz_validator import validate_xyz_text


def test_bounded_text_reads_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_bytes(b"one\ntwo\nthree\nfour\n")
    sandbox = WorkspaceSandbox(tmp_path, max_lines=3, max_bytes=32)

    result = sandbox.read_text_range("run.log", 2, 3)
    assert result["content"] == "two\nthree\n"
    assert result["line_start"] == 2
    assert result["line_end"] == 3
    assert len(result["content_sha256"]) == 64

    tail = sandbox.tail_text("run.log", 2)
    assert tail["content"] == "three\nfour\n"
    assert (tail["line_start"], tail["line_end"]) == (3, 4)
    assert tail["citation_eligible"] is False
    assert "read_text_range" in tail["citation_instruction"]

    with pytest.raises(ToolInputError, match="line limit"):
        sandbox.read_text_range("run.log", 1, 4)


@pytest.mark.parametrize(
    "path",
    ["../outside.log", "folder/../outside.log", "C:\\outside.log", "/outside.log", "\\\\server\\share\\x"],
)
def test_workspace_rejects_escape_paths(tmp_path: Path, path: str) -> None:
    sandbox = WorkspaceSandbox(tmp_path)
    with pytest.raises(ToolInputError):
        sandbox.resolve(path)


def test_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.log"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    sandbox = WorkspaceSandbox(tmp_path)
    with pytest.raises(ToolInputError) as caught:
        sandbox.resolve("link.log")
    assert caught.value.code == "LINK_ESCAPE_REJECTED"
    assert sandbox.list_case_files()["file_count"] == 0


def test_multisegment_xtb_log_keeps_segment_facts_separate() -> None:
    text = """$ xtb input.xyz --opt
GEOMETRY OPTIMIZATION FAILED TO CONVERGE
abnormal termination
$ xtb input.xyz --opt --cycles 100
GEOMETRY OPTIMIZATION CONVERGED AFTER 42 ITERATIONS
normal termination of xtb
"""
    result = parse_xtb_log_text(text, path="attempt.log")
    assert result["segment_count"] == 2
    assert result["segments"][0]["optimization_failure_observed"] is True
    assert result["segments"][0]["optimization_converged_observed"] is False
    assert result["latest_segment"]["optimization_converged_observed"] is True
    assert result["latest_segment"]["normal_termination_observed"] is True


def test_xyz_multiframe_truncation_order_and_geometry() -> None:
    valid = """2
first
H 0 0 0
H 0 0 0.74
2
second
H 0 0 0.01
H 0 0 0.75
"""
    result = validate_xyz_text(valid, expected_symbols=("H", "H"))
    assert result["valid"] is True
    assert result["frame_count"] == 2
    assert result["selected_frame_index"] == 1

    truncated = validate_xyz_text("2\nbroken\nH 0 0 0\n")
    assert truncated["valid"] is False
    assert truncated["truncated"] is True

    reordered = validate_xyz_text("2\nchanged\nO 0 0 0\nH 0 0 1\n", expected_symbols=("H", "O"))
    assert reordered["valid"] is False
    assert reordered["atom_order_consistent"] is False

    collision = validate_xyz_text("2\ncollision\nH 0 0 0\nH 0 0 0.1\n")
    assert collision["valid"] is False
    assert any("implausibly close" in error for error in collision["errors"])

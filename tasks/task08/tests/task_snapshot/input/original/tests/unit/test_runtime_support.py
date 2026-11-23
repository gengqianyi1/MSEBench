from __future__ import annotations

from pathlib import Path

import pytest

from scripts.runtime_support import (
    RuntimeIntegrityError,
    build_file_manifest,
    clone_frozen_case,
    pre_model_source_fingerprint,
    verify_case_template,
    verify_file_manifest,
    write_json_exclusive,
)
from scripts.run_model import _formal_prompt


def test_formal_prompt_contains_the_agent_visible_task_and_protocol_guard() -> None:
    prompt = _formal_prompt()
    assert "# xTB Failure Diagnosis and Restart Agent Benchmark v2" in prompt
    assert "uses six equally weighted cases" in prompt
    assert "cycle limits are 2–200" in prompt
    assert "Only successful `read_text_range` calls are citation-eligible" in prompt
    assert "Parser and tail\nresults locate passages" in prompt
    assert "Determine their relationship" in prompt
    assert "Process style receives no standalone points" in prompt
    assert "exactly `null` for a Hessian" in prompt


def test_pre_model_fingerprint_binds_source_but_excludes_generated_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "trajectories").mkdir()
    (tmp_path / "README.md").write_text("source one\n", encoding="utf-8")
    (tmp_path / "configs" / "task.yaml").write_text("task: one\n", encoding="utf-8")
    (tmp_path / "configs" / "release.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "reports" / "BENCHMARK_DESIGN_REPORT.md").write_text(
        "generated one\n", encoding="utf-8"
    )
    (tmp_path / "trajectories" / "README.md").write_text(
        "generated one\n", encoding="utf-8"
    )
    first = pre_model_source_fingerprint(tmp_path)
    (tmp_path / "configs" / "release.json").write_text('{"changed":true}\n', encoding="utf-8")
    (tmp_path / "reports" / "BENCHMARK_DESIGN_REPORT.md").write_text(
        "generated two\n", encoding="utf-8"
    )
    (tmp_path / "trajectories" / "README.md").write_text(
        "generated two\n", encoding="utf-8"
    )
    assert pre_model_source_fingerprint(tmp_path) == first
    (tmp_path / "README.md").write_text("source two\n", encoding="utf-8")
    assert pre_model_source_fingerprint(tmp_path) != first


def test_frozen_case_manifest_is_complete_and_hash_bound() -> None:
    manifest = verify_case_template()
    assert manifest["source_type"] == "NATIVE_XTB"
    assert manifest["initial_failure"]["verified_not_converged"] is True


def test_clone_is_verified_and_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "workspace"
    clone_frozen_case(target)
    assert verify_case_template(target)["case_id"] == "aspirin_gfn2_opt_hess_restart_v1"
    with pytest.raises(RuntimeIntegrityError, match="already exists"):
        clone_frozen_case(target)


def test_file_manifest_round_trip_and_mutation_detection(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "evidence.txt").write_text(
        "immutable\n", encoding="utf-8", newline="\n"
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    write_json_exclusive(
        manifest_path,
        build_file_manifest(tmp_path, excluded=("artifact_manifest.json",)),
    )
    assert verify_file_manifest(tmp_path, manifest_path)
    (tmp_path / "nested" / "evidence.txt").write_text(
        "changed\n", encoding="utf-8", newline="\n"
    )
    assert not verify_file_manifest(tmp_path, manifest_path)

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "benchmark_case"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_initial_failure_is_native_and_cycle_limited() -> None:
    config = yaml.safe_load((ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((CASE / "case_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (CASE / "attempts" / "attempt_000_initial_failure" / "execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    stdout = (
        CASE / "attempts" / "attempt_000_initial_failure" / "stdout.log"
    ).read_text(encoding="utf-8")
    assert manifest["source_type"] == "NATIVE_XTB"
    assert manifest["initial_failure"]["optimization_cycle_limit"] == 1
    assert config["runtime"]["xtb"]["initial_failure_cycle_limit"] == 1
    assert summary["process_exit_code"] == 0
    assert summary["optimization_converged"] is False
    assert summary["complete_trajectory_frames"] >= 1
    assert re.search(r"max\.\s+optcycles\s+1", stdout, re.IGNORECASE)
    assert "FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 1 ITERATIONS" in stdout


def test_scheduler_completed_is_not_scientific_success() -> None:
    attempt = CASE / "attempts" / "attempt_000_initial_failure"
    scheduler = json.loads((attempt / "scheduler.json").read_text(encoding="utf-8"))
    summary = json.loads((attempt / "execution_summary.json").read_text(encoding="utf-8"))
    assert scheduler["state"] == "COMPLETED"
    assert scheduler["scope"] == "execution_layer_only"
    assert scheduler["scientific_success_inference_allowed"] is False
    assert summary["scientific_status"] == "OPTIMIZATION_NOT_CONVERGED"


def test_legacy_artifact_is_real_but_not_current_or_restartable() -> None:
    legacy = json.loads((CASE / "legacy" / "artifact_manifest.json").read_text(encoding="utf-8"))
    initial_xyz = CASE / "attempts" / "attempt_000_initial_failure" / "xtbopt.xyz"
    legacy_xyz = CASE / "legacy" / "xtbopt.xyz"
    assert legacy["source_type"] == "NATIVE_XTB"
    assert legacy["native_run_receipt"]["optimization_converged"] is True
    assert legacy["provenance_state"] == "LEGACY_INDEPENDENT_SUCCESS"
    assert legacy["allowed_as_current_success"] is False
    assert legacy["allowed_as_restart_source"] is False
    assert sha256(legacy_xyz) == legacy["files"][0]["sha256"]
    assert sha256(initial_xyz) != sha256(legacy_xyz)


def test_case_manifest_is_complete_and_hash_valid() -> None:
    manifest = json.loads((CASE / "case_manifest.json").read_text(encoding="utf-8"))
    listed = {item["path"]: item for item in manifest["files"]}
    actual = {
        path.relative_to(CASE).as_posix()
        for path in CASE.rglob("*")
        if path.is_file() and path.name != "case_manifest.json"
    }
    assert set(listed) == actual
    for relative, item in listed.items():
        path = CASE / relative
        assert path.stat().st_size == item["size_bytes"]
        assert sha256(path) == item["sha256"]


def test_case_artifacts_do_not_contain_local_absolute_paths() -> None:
    windows = re.compile(rb"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
    posix = re.compile(rb"(?<![:A-Za-z0-9_])/(?:home|Users|tmp|workspace|mnt)/")
    findings = []
    for path in CASE.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            if windows.search(data) or posix.search(data):
                findings.append(path.relative_to(CASE).as_posix())
    assert findings == []

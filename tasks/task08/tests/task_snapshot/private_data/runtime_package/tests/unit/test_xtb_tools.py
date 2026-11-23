from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tools.tool_registry import ToolRegistry


XYZ = """2
hydrogen
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(
    tmp_path: Path,
    process_runner: object,
    *,
    include_initial_failure_trajectory: bool = False,
) -> ToolRegistry:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "benchmark_case").mkdir()
    source_root = Path(__file__).resolve().parents[2]
    shutil.copyfile(source_root / "configs/task.yaml", project / "configs/task.yaml")
    shutil.copyfile(source_root / "configs/tool_contracts.json", project / "configs/tool_contracts.json")
    input_path = project / "benchmark_case/input.xyz"
    input_path.write_text(XYZ, encoding="utf-8", newline="\n")
    job = {
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "input_geometry": {"path": "benchmark_case/input.xyz"},
        "scientific_intent": {
            "method": "GFN2-xTB",
            "charge": 0,
            "uhf": 0,
            "solvation": None,
            "required_phases": ["geometry_optimization", "hessian"],
        },
    }
    (project / "benchmark_case/job_spec.json").write_text(
        json.dumps(job), encoding="utf-8"
    )
    artifacts = [
            {
                "path": "input.xyz",
                "sha256": _sha(input_path),
                "origin": "ORIGINAL_INPUT",
                "current": True,
                "restart_allowed": True,
                "success_evidence_allowed": False,
            }
        ]
    manifest: dict[str, object] = {"artifacts": artifacts}
    if include_initial_failure_trajectory:
        initial = project / "benchmark_case/attempts/attempt_000_initial_failure"
        initial.mkdir(parents=True)
        trajectory = initial / "xtbopt.log"
        trajectory.write_text(XYZ, encoding="utf-8", newline="\n")
        artifacts.append(
            {
                "path": "attempts/attempt_000_initial_failure/xtbopt.log",
                "sha256": _sha(trajectory),
            }
        )
        manifest["initial_failure"] = {
            "attempt_id": "attempt_000_initial_failure",
            "optimization_cycle_limit": 1,
            "verified_not_converged": True,
        }
    (project / "benchmark_case/case_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return ToolRegistry(project, process_runner=process_runner)


class FakeXtb:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        assert kwargs["shell"] is False
        assert isinstance(argv, list)
        cwd = Path(kwargs["cwd"])
        stdout = kwargs["stdout"]
        stderr = kwargs["stderr"]
        assert hasattr(stdout, "write") and hasattr(stderr, "write")
        if "--hess" in argv:
            stdout.write(
                b"$ xtb input.xyz --hess\nHessian calculation completed\n"
                b"vibrational frequencies\nnormal termination of xtb\n"
            )
            (cwd / "hessian").write_text("1.0 0.0\n", encoding="utf-8")
            (cwd / "vibspectrum").write_text("frequency\n", encoding="utf-8")
        else:
            stdout.write(
                b"$ xtb input.xyz --opt\nGEOMETRY OPTIMIZATION CONVERGED AFTER 4 ITERATIONS\n"
                b"normal termination of xtb\n"
            )
            (cwd / "xtbopt.xyz").write_text(XYZ, encoding="utf-8", newline="\n")
        stderr.write(b"")
        self.calls.append({"argv": argv, **kwargs})
        return SimpleNamespace(returncode=0)


def _prepare_args(phase: str, source: str, policy: str) -> dict[str, object]:
    return {
        "phase": phase,
        "source_geometry": {"path": source, "frame_index": 0},
        "optimization_cycle_limit": 100 if phase == "geometry_optimization" else None,
        "thread_count": 1,
        "walltime_seconds": 180,
        "restart_policy": policy,
        "rationale": "Use a hash-bound, validated geometry and preserve the job specification.",
        "preserved_invariants": [
            "method",
            "charge",
            "uhf",
            "solvation",
            "atom_order",
            "composition",
            "required_phases",
        ],
    }


def test_prepare_run_validate_two_phase_lifecycle(tmp_path: Path) -> None:
    fake = FakeXtb()
    registry = _project(tmp_path, fake)

    prepared = registry.dispatch(
        "prepare_restart",
        _prepare_args("geometry_optimization", "input.xyz", "original_input"),
        call_id="call_prepare_opt",
    )
    assert prepared["ok"] is True
    attempt_1 = prepared["result"]["attempt_id"]
    assert prepared["result"]["shell"] is False
    assert prepared["result"]["argv"][0] == "conda"
    assert prepared["result"]["argv"] == (
        prepared["result"]["runtime_wrapper_argv"]
        + prepared["result"]["scientific_argv"]
    )
    assert "--cycles" not in prepared["result"]["argv"]
    assert prepared["result"]["argv"][:6] == [
        "conda",
        "run",
        "-n",
            "chembench",
        "--no-capture-output",
        "xtb",
    ]
    attempt_root = registry.sandbox.root / "attempts" / attempt_1
    assert (attempt_root / "xcontrol.inp").read_text(encoding="utf-8") == (
        "$opt\n  maxcycle=100\n$end\n"
    )

    executed = registry.dispatch("run_xtb", {"attempt_id": attempt_1}, call_id="call_run_opt")
    assert executed["ok"] is True
    assert executed["result"]["native_xtb_executed"] is True
    assert executed["result"]["execution_summary_path"].endswith("/execution_summary.json")
    assert executed["result"]["artifact_manifest_path"].endswith("/artifact_manifest.json")
    assert executed["result"]["artifact_hashes"]
    execution_manifest_path = attempt_root / "artifact_manifest.json"
    assert execution_manifest_path.is_file()
    execution_manifest_before_validation = execution_manifest_path.read_bytes()
    execution_manifest = json.loads(execution_manifest_before_validation)
    assert execution_manifest["attempt_id"] == attempt_1
    assert execution_manifest["entries"]
    for entry in execution_manifest["entries"]:
        manifested = registry.sandbox.root / Path(entry["path"])
        assert manifested.is_file()
        assert _sha(manifested) == entry["sha256"]
        assert manifested.stat().st_size == entry["size_bytes"]
    prevalidation_listing = registry.dispatch(
        "list_case_files", {}, call_id="list_prevalidation_artifacts"
    )
    listed_by_path = {
        item["path"]: item for item in prevalidation_listing["result"]["files"]
    }
    prevalidation_output = listed_by_path[f"attempts/{attempt_1}/stdout.txt"]
    assert prevalidation_output["source_attempt"] == attempt_1
    assert prevalidation_output["provenance_state"] == "ATTEMPT_ARTIFACT_UNVALIDATED"
    assert prevalidation_output["current"] is True
    validated = registry.dispatch(
        "validate_attempt", {"attempt_id": attempt_1}, call_id="call_validate_opt"
    )
    assert validated["ok"] is True
    assert validated["result"]["validation_passed"] is True
    assert validated["result"]["optimization_completed"] is True
    assert validated["result"]["classification"] == "VALIDATED_SUCCESS"
    assert validated["result"]["phase_classification"] == "OPTIMIZATION_CONVERGED"
    assert execution_manifest_path.read_bytes() == execution_manifest_before_validation
    required_attempt_files = {
        "attempt.json",
        "prepared_command.json",
        "input.xyz",
        "input_manifest.json",
        "stdout.txt",
        "stderr.txt",
        "execution_summary.json",
        "artifact_manifest.json",
        "validation.json",
    }
    assert required_attempt_files.issubset(
        {path.name for path in attempt_root.iterdir() if path.is_file()}
    )
    assert _sha(attempt_root / "attempt.json") == _sha(attempt_root / "prepared.json")
    assert _sha(attempt_root / "prepared_command.json") == _sha(attempt_root / "prepared.json")
    assert _sha(attempt_root / "input_manifest.json") == _sha(attempt_root / "provenance.json")
    assert _sha(attempt_root / "execution_summary.json") == _sha(attempt_root / "run_record.json")
    assert _sha(attempt_root / "stdout.txt") == _sha(attempt_root / "stdout.log")
    assert _sha(attempt_root / "stderr.txt") == _sha(attempt_root / "stderr.log")

    optimized = validated["result"]["artifacts"]["optimized_geometry"]["path"]
    provenance = registry.dispatch(
        "inspect_artifact_provenance",
        {"path": optimized, "frame_index": 0},
        call_id="call_provenance",
    )
    assert provenance["ok"] is True
    assert provenance["result"]["restart_allowed"] is True

    prepared_hess = registry.dispatch(
        "prepare_restart",
        _prepare_args("hessian", optimized, "validated_optimized_geometry"),
        call_id="call_prepare_hess",
    )
    assert prepared_hess["ok"] is True
    attempt_2 = prepared_hess["result"]["attempt_id"]
    registry.dispatch("run_xtb", {"attempt_id": attempt_2}, call_id="call_run_hess")
    validated_hess = registry.dispatch(
        "validate_attempt", {"attempt_id": attempt_2}, call_id="call_validate_hess"
    )
    assert validated_hess["ok"] is True
    assert validated_hess["result"]["validation_passed"] is True
    assert validated_hess["result"]["hessian_completed"] is True
    assert validated_hess["result"]["classification"] == "VALIDATED_SUCCESS"
    assert validated_hess["result"]["phase_classification"] == "HESSIAN_COMPLETED"
    assert "hessian" in validated_hess["result"]["artifacts"]
    assert "vibrational_spectrum" in validated_hess["result"]["artifacts"]

    repeat = registry.dispatch("run_xtb", {"attempt_id": attempt_2}, call_id="call_repeat")
    assert repeat["ok"] is False
    assert repeat["error"]["code"] == "ATTEMPT_ALREADY_RUN"
    assert registry.trusted_state["attempts"][attempt_2]["hessian_completed"] is True
    assert len(fake.calls) == 2


def test_identical_restart_is_rejected_without_overwrite(tmp_path: Path) -> None:
    registry = _project(tmp_path, FakeXtb())
    arguments = _prepare_args("geometry_optimization", "input.xyz", "original_input")
    first = registry.dispatch("prepare_restart", arguments, call_id="first")
    second = registry.dispatch("prepare_restart", arguments, call_id="second")
    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"]["code"] == "DUPLICATE_ATTEMPT"
    attempts = list((registry.sandbox.root / "attempts").glob("attempt_[0-9][0-9][0-9]"))
    assert len(attempts) == 1


def test_prepare_requires_the_complete_public_invariant_set(tmp_path: Path) -> None:
    registry = _project(tmp_path, FakeXtb())
    arguments = _prepare_args("geometry_optimization", "input.xyz", "original_input")
    arguments["preserved_invariants"] = [
        "method",
        "charge",
        "uhf",
        "solvation",
        "atom_order",
    ]
    rejected = registry.dispatch("prepare_restart", arguments, call_id="five_invariants")
    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "SCHEMA_VALIDATION_FAILED"


def test_registry_dispatches_contract_text_range_and_tail_arguments(tmp_path: Path) -> None:
    registry = _project(tmp_path, FakeXtb())
    ranged = registry.dispatch(
        "read_text_range",
        {"path": "input.xyz", "start_line": 2, "end_line": 4},
        call_id="range_contract",
    )
    assert ranged["ok"] is True, ranged
    assert ranged["result"]["content"] == (
        "hydrogen\nH 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.740000\n"
    )
    assert ranged["result"]["line_start"] == 2
    assert ranged["result"]["line_end"] == 4

    tailed = registry.dispatch(
        "tail_text",
        {"path": "input.xyz", "line_count": 2},
        call_id="tail_contract",
    )
    assert tailed["ok"] is True, tailed
    assert tailed["result"]["line_start"] == 3
    assert tailed["result"]["line_end"] == 4


def test_list_case_files_exposes_hash_bound_provenance_and_binary_readability(
    tmp_path: Path,
) -> None:
    registry = _project(tmp_path, FakeXtb())
    binary = registry.sandbox.root / "native.raw.bin"
    binary.write_bytes(b"\x00\xff\x10")

    listed = registry.dispatch("list_case_files", {}, call_id="list_observability")
    assert listed["ok"] is True, listed
    by_path = {item["path"]: item for item in listed["result"]["files"]}
    source = by_path["input.xyz"]
    assert source["relative_path"] == "input.xyz"
    assert source["provenance_state"] == "ORIGINAL_INPUT"
    assert source["current"] is True
    assert source["readable"] is True
    assert source["source_attempt"] is None
    assert by_path["native.raw.bin"]["readable"] is False
    assert by_path["native.raw.bin"]["provenance_state"] == "UNMANIFESTED"


class ProcessFailureXtb:
    def __call__(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        kwargs["stdout"].write(b"$ xtb input.xyz --opt\nfatal error: aborted\n")
        kwargs["stderr"].write(b"abnormal termination of xtb\n")
        return SimpleNamespace(returncode=2)


class IncompleteOptimizationXtb:
    def __call__(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        kwargs["stdout"].write(
            b"$ xtb input.xyz --opt\nmaximum optimization cycles reached\n"
        )
        kwargs["stderr"].write(b"normal termination of xtb\n")
        return SimpleNamespace(returncode=0)


class InvalidArtifactXtb:
    def __call__(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        cwd = Path(kwargs["cwd"])
        kwargs["stdout"].write(
            b"$ xtb input.xyz --opt\nGEOMETRY OPTIMIZATION CONVERGED AFTER 2 ITERATIONS\n"
        )
        kwargs["stderr"].write(b"normal termination of xtb\n")
        (cwd / "xtbopt.xyz").write_text(
            "2\ncollision\nH 0 0 0\nH 0 0 0.1\n", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (ProcessFailureXtb(), "PROCESS_FAILED"),
        (IncompleteOptimizationXtb(), "OPTIMIZATION_NOT_CONVERGED"),
        (InvalidArtifactXtb(), "INVALID_ARTIFACT"),
    ],
)
def test_validate_attempt_fail_closed_classification_priority(
    tmp_path: Path, runner: object, expected: str
) -> None:
    registry = _project(tmp_path, runner)
    prepared = registry.dispatch(
        "prepare_restart",
        _prepare_args("geometry_optimization", "input.xyz", "original_input"),
        call_id="prepare_classification",
    )
    assert prepared["ok"] is True, prepared
    attempt_id = prepared["result"]["attempt_id"]
    executed = registry.dispatch(
        "run_xtb", {"attempt_id": attempt_id}, call_id="run_classification"
    )
    assert executed["ok"] is True, executed
    validated = registry.dispatch(
        "validate_attempt", {"attempt_id": attempt_id}, call_id="validate_classification"
    )
    assert validated["ok"] is True, validated
    result = validated["result"]
    assert result["state"] == "VALIDATED"
    assert result["classification"] == expected
    assert result["phase_classification"] in {
        "OPTIMIZATION_NOT_CONVERGED",
        "OPTIMIZATION_CONVERGED",
    }
    assert result["classification_priority"] == [
        "INVALID_ARTIFACT",
        "PROCESS_FAILED",
        "OPTIMIZATION_NOT_CONVERGED",
        "PROGRAM_COMPLETED_TASK_INCOMPLETE",
        "SCIENTIFIC_WARNING",
        "VALIDATED_SUCCESS",
    ]


def test_tail_text_cannot_be_used_as_final_citation(tmp_path: Path) -> None:
    registry = _project(tmp_path, FakeXtb())
    tailed = registry.dispatch(
        "tail_text",
        {"path": "input.xyz", "line_count": 2},
        call_id="tail_navigation_only",
    )
    assert tailed["ok"] is True
    result = tailed["result"]
    answer = {
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "final_status": "FAILED_WITHIN_BUDGET",
        "diagnosis": {
            "primary_category": "UNKNOWN",
            "primary_cause_code": "UNKNOWN",
                "scheduler_assessment": "UNKNOWN",
                "conflict_resolution": "UNRESOLVED",
            "legacy_disposition": "NOT_INSPECTED",
            "secondary_findings": [],
            "confidence": 0,
            "evidence": [
                {
                    "tool_call_id": "tail_navigation_only",
                    "path": result["path"],
                    "line_start": result["line_start"],
                    "line_end": result["line_end"],
                    "content_sha256": result["content_sha256"],
                    "interpretation": "A tail result is useful for navigation only.",
                }
            ],
        },
        "restart_summary": {
                "starting_geometry_source_kind": "NONE",
                "starting_geometry_source": {
                    "attempt_id": "attempt_000_initial_failure",
                    "path": "initial_failure/input.xyz",
                    "frame_index": 0,
                    "file_sha256": "a" * 64,
                    "frame_sha256": "a" * 64,
                },
            "changes": [],
            "preserved_invariants": [],
        },
        "result_summary": {
            "optimization_completed": False,
            "hessian_completed": False,
            "final_attempt_id": "",
            "warnings": [],
        },
        "rationale": "No trusted direct-range evidence was submitted.",
    }
    submitted = registry.dispatch("submit_final", answer, call_id="submit_tail_evidence")
    assert submitted["ok"] is False
    assert submitted["error"]["code"] == "UNTRUSTED_EVIDENCE"


def test_last_complete_initial_failure_frame_is_restart_only(tmp_path: Path) -> None:
    registry = _project(
        tmp_path,
        FakeXtb(),
        include_initial_failure_trajectory=True,
    )
    source = "attempts/attempt_000_initial_failure/xtbopt.log"
    inspected = registry.dispatch(
        "inspect_artifact_provenance",
        {"path": source, "frame_index": 0},
        call_id="inspect_initial_trajectory",
    )
    assert inspected["ok"] is True
    assert inspected["result"]["source_kind"] == "CURRENT_TRAJECTORY_FRAME"
    assert inspected["result"]["current"] is True
    assert inspected["result"]["restart_allowed"] is True
    assert inspected["result"]["success_evidence_allowed"] is False

    prepared = registry.dispatch(
        "prepare_restart",
        _prepare_args("geometry_optimization", source, "last_complete_current_frame"),
        call_id="prepare_from_initial_trajectory",
    )
    assert prepared["ok"] is True, prepared
    assert prepared["result"]["source_geometry"]["provenance_state"] == (
        "CURRENT_TRAJECTORY_FRAME"
    )


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_XTB") != "1",
    reason="set RUN_LIVE_XTB=1 to execute the installed native xTB runtime",
)
def test_live_native_xtb_prepare_run_validate(tmp_path: Path) -> None:
    """Opt-in smoke test: this is native execution, never a mock or replay."""

    registry = _project(tmp_path, None)
    prepared = registry.dispatch(
        "prepare_restart",
        _prepare_args("geometry_optimization", "input.xyz", "original_input"),
        call_id="live_prepare",
    )
    assert prepared["ok"] is True, prepared
    attempt_id = prepared["result"]["attempt_id"]
    executed = registry.dispatch(
        "run_xtb", {"attempt_id": attempt_id}, call_id="live_run"
    )
    assert executed["ok"] is True, executed
    assert executed["result"]["native_xtb_executed"] is True
    assert executed["result"]["process_exit_code"] == 0
    assert executed["result"]["runtime_identity"]["verified"] is True
    assert executed["result"]["runtime_identity"]["version"] == "6.7.1"
    validated = registry.dispatch(
        "validate_attempt", {"attempt_id": attempt_id}, call_id="live_validate"
    )
    assert validated["ok"] is True, validated
    assert validated["result"]["validation_passed"] is True, validated
    assert validated["result"]["optimization_converged"] is True

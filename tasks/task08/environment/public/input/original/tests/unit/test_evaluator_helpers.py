"""Deterministic evaluator fixtures used by unit and security tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from evaluator.trusted_validation import canonical_json_bytes
from framework.trusted_final import build_final_answer_envelope


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value, newline=True, indent=2))


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / Path(relative)
    return {
        "path": relative.replace("\\", "/"),
        "sha256": _sha(path.read_bytes()),
        "size_bytes": path.stat().st_size,
    }


def _xyz() -> str:
    return "2\naspirin test geometry\nC 0.0 0.0 0.0\nO 1.2 0.0 0.0\n"


def _create_attempt(
    case_root: Path,
    attempt_id: str,
    *,
    phase: str,
    source_path: str,
    parent_attempt_id: str | None,
) -> dict[str, Any]:
    relative_dir = f"attempts/{attempt_id}"
    directory = case_root / relative_dir
    directory.mkdir(parents=True)
    source = case_root / source_path
    input_path = f"{relative_dir}/input.xyz"
    _write_text(case_root / input_path, _xyz())
    source_hash = _sha(source.read_bytes())
    input_hash = _sha((case_root / input_path).read_bytes())
    is_opt = phase == "geometry_optimization"
    argv = [
        "conda",
        "run",
        "-n",
        "chembench",
        "--no-capture-output",
        "xtb",
        "input.xyz",
        "--gfn",
        "2",
        "--chrg",
        "0",
        "--uhf",
        "0",
    ]
    if is_opt:
        argv.extend(["--opt", "normal", "--input", "xcontrol.inp", "--parallel", "1"])
    else:
        argv.extend(["--hess", "--parallel", "1"])
    control_path = f"{relative_dir}/xcontrol.inp" if is_opt else None
    if control_path is not None:
        _write_text(case_root / control_path, "$opt\n  maxcycle=100\n$end\n")
    prepared = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "phase": phase,
        "state": "PREPARED",
        "request": {
            "optimization_cycle_limit": 100 if is_opt else None,
            "thread_count": 1,
            "walltime_seconds": 180,
            "restart_policy": (
                "last_complete_current_frame" if is_opt else "validated_optimized_geometry"
            ),
            "rationale": "Use trusted current geometry and preserve the scientific intent.",
            "preserved_invariants": [
                "method",
                "charge",
                "uhf",
                "solvation",
                "atom_order",
                "composition",
                "required_phases",
            ],
        },
        "source_geometry": {
            "path": source_path,
            "frame_index": 0 if is_opt else None,
            "file_sha256": source_hash,
            "frame_sha256": source_hash,
        },
        "input_geometry": {"path": input_path, "sha256": input_hash},
        "control_input": (
            {
                "path": control_path,
                "sha256": _sha((case_root / control_path).read_bytes()),
                "optimization_cycle_limit": 100,
            }
            if control_path is not None
            else None
        ),
        "scientific_settings": {
            "method": "GFN2-xTB",
            "charge": 0,
            "uhf": 0,
            "solvation": None,
            "required_phases": ["geometry_optimization", "hessian"],
        },
        "execution": {
            "environment_manager": "conda",
            "conda_environment": "chembench",
            "executable_alias": "xtb",
            "argv": argv,
            "runtime_wrapper_argv": argv[:5],
            "scientific_argv": argv[5:],
            "shell": False,
            "timeout_seconds": 180,
            "expected_version": "6.7.1",
            "identity_policy": "xtb_portable_identity/2.0",
            "required_help_tokens": ["--opt", "--hess", "--gfn", "--chrg", "--uhf"],
        },
        "fingerprint": _sha(canonical_json_bytes({"phase": phase, "argv": argv})),
        "created_at": "2026-07-19T00:00:10Z",
    }
    prepared_path = f"{relative_dir}/prepared.json"
    _write_json(case_root / prepared_path, prepared)
    attempt_path = f"{relative_dir}/attempt.json"
    prepared_command_path = f"{relative_dir}/prepared_command.json"
    (case_root / attempt_path).write_bytes((case_root / prepared_path).read_bytes())
    (case_root / prepared_command_path).write_bytes((case_root / prepared_path).read_bytes())
    provenance = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "source": {
            "path": source_path,
            "frame_index": prepared["source_geometry"]["frame_index"],
            "file_sha256": source_hash,
        },
        "input": {"path": input_path, "sha256": input_hash},
        "chain": [source_path, input_path],
        "parent_attempt_id": parent_attempt_id,
        "currentness": "PREPARED_INPUT",
        "restart_allowed": True,
        "success_evidence_allowed": False,
    }
    provenance_path = f"{relative_dir}/provenance.json"
    _write_json(case_root / provenance_path, provenance)
    input_manifest_path = f"{relative_dir}/input_manifest.json"
    (case_root / input_manifest_path).write_bytes((case_root / provenance_path).read_bytes())

    stdout_path = f"{relative_dir}/stdout.log"
    stderr_path = f"{relative_dir}/stderr.log"
    if is_opt:
        _write_text(
            case_root / stdout_path,
            "xTB 6.7.1\nGEOMETRY OPTIMIZATION CONVERGED AFTER 7 ITERATIONS\n",
        )
        output_path = f"{relative_dir}/xtbopt.xyz"
        _write_text(case_root / output_path, _xyz())
        output_roles = {
            "optimized_geometry": _file_record(case_root, output_path)
        }
    else:
        _write_text(
            case_root / stdout_path,
            "xTB 6.7.1\nHessian calculation\nvibrational frequencies\n",
        )
        hessian_path = f"{relative_dir}/hessian"
        spectrum_path = f"{relative_dir}/vibspectrum"
        _write_text(case_root / hessian_path, "1.0 0.0\n0.0 1.0\n")
        _write_text(case_root / spectrum_path, "1 100.0\n2 200.0\n")
        output_roles = {
            "hessian": _file_record(case_root, hessian_path),
            "vibrational_spectrum": _file_record(case_root, spectrum_path),
        }
    _write_text(case_root / stderr_path, "normal termination of xtb\n")
    stdout_canonical_path = f"{relative_dir}/stdout.txt"
    stderr_canonical_path = f"{relative_dir}/stderr.txt"
    (case_root / stdout_canonical_path).write_bytes((case_root / stdout_path).read_bytes())
    (case_root / stderr_canonical_path).write_bytes((case_root / stderr_path).read_bytes())
    run_files = {
        relative: {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
        for relative, record in {
            stdout_path: _file_record(case_root, stdout_path),
            stderr_path: _file_record(case_root, stderr_path),
            **{entry["path"]: entry for entry in output_roles.values()},
        }.items()
    }
    run_record = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "phase": phase,
        "state": "EXECUTED",
        "native_xtb_executed": True,
        "shell": False,
        "argv": argv,
        "scientific_argv": argv[5:],
        "runtime_wrapper_argv": argv[:5],
        "conda_environment": "chembench",
        "started_at": "2026-07-19T00:00:20Z",
        "completed_at": "2026-07-19T00:00:21Z",
        "duration_seconds": 1,
        "timeout_seconds": 180,
        "timed_out": False,
        "exit_code": 0,
        "exception_type": None,
        "files": run_files,
    }
    run_path = f"{relative_dir}/run_record.json"
    _write_json(case_root / run_path, run_record)
    execution_summary_path = f"{relative_dir}/execution_summary.json"
    (case_root / execution_summary_path).write_bytes((case_root / run_path).read_bytes())
    validation = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "phase": phase,
        "state": "VALIDATED",
        "validation_passed": True,
        "process_valid": True,
        "native_xtb_executed": True,
        "optimization_completed": is_opt,
        "hessian_completed": not is_opt,
        "normal_termination_observed": True,
        "invariants_preserved": True,
        "source_current": True,
        "artifacts_current": True,
        "artifacts": output_roles,
        "warnings": [],
        "errors": [],
        "validated_at": "2026-07-19T00:00:22Z",
    }
    validation_path = f"{relative_dir}/validation.json"
    _write_json(case_root / validation_path, validation)
    manifest_entries = [
        _file_record(case_root, path)
        for path in sorted(
            [
                input_path,
                *([control_path] if control_path is not None else []),
                attempt_path,
                prepared_command_path,
                prepared_path,
                input_manifest_path,
                provenance_path,
                stdout_canonical_path,
                stdout_path,
                stderr_canonical_path,
                stderr_path,
                execution_summary_path,
                run_path,
                validation_path,
                *[entry["path"] for entry in output_roles.values()],
            ]
        )
    ]
    for entry in manifest_entries:
        entry["role"] = "attempt_artifact"
    manifest = {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "entries": manifest_entries,
        "created_at": "2026-07-19T00:00:23Z",
    }
    manifest_path = f"{relative_dir}/artifact_manifest.json"
    _write_json(case_root / manifest_path, manifest)
    return {
        "prepared": prepared,
        "prepared_path": prepared_command_path,
        "provenance_path": input_manifest_path,
        "run_record": run_record,
        "run_path": execution_summary_path,
        "validation": validation,
        "validation_path": validation_path,
        "manifest_path": manifest_path,
        "source_hash": source_hash,
    }


def _event(
    events: list[dict[str, Any]],
    name: str,
    payload: dict[str, Any],
    *,
    run_id: str = "fixture-run-001",
) -> None:
    previous = events[-1]["event_hash"] if events else "0" * 64
    value = {
        "schema_version": "1.0",
        "protocol": {
            "name": "xtb_failure_recovery_tool_loop",
            "version": "1.0",
        },
        "run_id": run_id,
        "sequence": len(events) + 1,
        "timestamp_utc": f"2026-07-19T00:00:{len(events):02d}Z",
        "event": name,
        "payload": payload,
        "prev_event_hash": previous,
    }
    value["event_hash"] = _sha(canonical_json_bytes(value)).lower()
    events.append(value)


def _tool_pair(
    events: list[dict[str, Any]],
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    result: Any,
) -> None:
    _event(events, "tool_call", {"call_id": call_id, "name": name, "arguments": arguments})
    _event(events, "tool_result", {"call_id": call_id, "name": name, "ok": True, "result": result})


def build_complete_fixture(root: Path) -> dict[str, Path]:
    """Build a small self-contained, fully eligible evaluator fixture."""

    project = root / "project"
    config_dir = project / "configs"
    case_root = project / "benchmark_case"
    config_dir.mkdir(parents=True)
    initial_dir = case_root / "attempts" / "attempt_000_initial_failure"
    initial_dir.mkdir(parents=True)
    _write_text(case_root / "input" / "start.xyz", _xyz())
    initial_stdout = (
        "xTB 6.7.1\n"
        "GEOMETRY OPTIMIZATION FAILED: maximum optimization cycles reached\n"
        "last complete structure follows\n"
    )
    _write_text(initial_dir / "stdout.log", initial_stdout)
    _write_text(initial_dir / "xtbopt.log", _xyz())
    _write_json(initial_dir / "scheduler.json", {"schema_version": "1.0", "state": "COMPLETED"})
    _write_text(case_root / "legacy" / "xtbopt.xyz", _xyz())

    job_spec = {
        "schema_version": "1.0",
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "molecule": {
            "name": "aspirin",
            "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "charge": 0,
            "uhf": 0,
            "atom_count": 2,
            "element_sequence": ["C", "O"],
        },
        "method": "GFN2-xTB",
        "solvation": None,
        "required_phases": ["geometry_optimization", "hessian"],
        "optimization": {"level": "normal", "require_convergence": True},
        "hessian": {"required": True},
        "allowed_mutations": [
            "optimization_cycle_limit",
            "safe_restart_source",
            "thread_count",
            "walltime",
        ],
        "forbidden_mutations": [
            "molecule_composition",
            "atom_order",
            "charge",
            "uhf",
            "method",
            "solvation",
            "required_phases",
        ],
    }
    job_path = case_root / "job_spec.json"
    _write_json(job_path, job_spec)
    frozen_files = [
        _file_record(case_root, path.relative_to(case_root).as_posix())
        for path in sorted(
            case_root.rglob("*"),
            key=lambda value: value.relative_to(case_root).as_posix(),
        )
        if path.is_file()
    ]
    case_manifest = {
        "schema_version": "1.0",
        "case_id": job_spec["case_id"],
        "source_type": "NATIVE_XTB",
        "files": frozen_files,
    }
    manifest_path = case_root / "case_manifest.json"
    _write_json(manifest_path, case_manifest)

    opt = _create_attempt(
        case_root,
        "attempt_001",
        phase="geometry_optimization",
        source_path="attempts/attempt_000_initial_failure/xtbopt.log",
        parent_attempt_id="attempt_000_initial_failure",
    )
    hess = _create_attempt(
        case_root,
        "attempt_002",
        phase="hessian",
        source_path="attempts/attempt_001/xtbopt.xyz",
        parent_attempt_id="attempt_001",
    )

    models = yaml.safe_load((PROJECT_ROOT / "configs" / "models.example.yaml").read_text(encoding="utf-8"))
    (config_dir / "models.example.yaml").write_text(
        yaml.safe_dump(models, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )
    contracts = json.loads((PROJECT_ROOT / "configs" / "tool_contracts.json").read_text(encoding="utf-8"))
    _write_json(config_dir / "tool_contracts.json", contracts)
    task = yaml.safe_load((PROJECT_ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    task["task"]["job_spec"]["sha256"] = _sha(job_path.read_bytes())
    task["task"]["case_manifest"]["sha256"] = _sha(manifest_path.read_bytes())
    task["runtime"]["xtb"]["initial_failure_cycle_limit"] = 1
    config_path = config_dir / "task.yaml"
    config_path.write_text(
        yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )

    evidence_content = "GEOMETRY OPTIMIZATION FAILED: maximum optimization cycles reached"
    model_submission = {
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "final_status": "RECOVERED",
        "diagnosis": {
            "primary_category": "OPTIMIZATION_FAILURE",
            "primary_cause_code": "OPTIMIZATION_MAX_CYCLES",
            "scheduler_assessment": "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE",
            "conflict_resolution": "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION",
            "legacy_disposition": "NOT_CURRENT_INFORMATION_ONLY",
            "secondary_findings": ["The scheduler state is execution-layer evidence only."],
            "confidence": 0.99,
            "evidence": [
                {
                    "tool_call_id": "call_read_log",
                    "path": "attempts/attempt_000_initial_failure/stdout.log",
                    "line_start": 2,
                    "line_end": 2,
                    "content_sha256": _sha((evidence_content + "\n").encode("utf-8")),
                    "interpretation": "The optimization exhausted its cycle budget.",
                }
            ],
        },
        "restart_summary": {
            "starting_geometry_source_kind": "CURRENT_TRAJECTORY_FRAME",
            "starting_geometry_source": {
                "attempt_id": "attempt_000_initial_failure",
                "path": opt["prepared"]["source_geometry"]["path"],
                "frame_index": opt["prepared"]["source_geometry"]["frame_index"],
                "file_sha256": opt["prepared"]["source_geometry"]["file_sha256"],
                "frame_sha256": opt["prepared"]["source_geometry"]["frame_sha256"],
            },
            "changes": ["Raised the optimization cycle limit from 1 to 100."],
            "preserved_invariants": [
                "method",
                "charge",
                "uhf",
                "solvation",
                "atom_order",
                "composition",
                "required_phases",
            ],
        },
        "result_summary": {
            "optimization_completed": True,
            "hessian_completed": True,
            "final_attempt_id": "attempt_002",
            "warnings": [],
        },
        "rationale": "The current failed trajectory was safely restarted and both required phases validated.",
    }
    final_path = project / "final_answer.json"

    events: list[dict[str, Any]] = []
    _event(
        events,
        "run_start",
        {
            "run_type": "FORMAL_MODEL",
            "logical_model": "kimi",
            "requested_model": "Pro/moonshotai/Kimi-K2.6",
            "provider": "siliconflow",
            "transport": "openai_compatible",
            "execution_mode": "REAL_API",
            "native_mode": "LIVE_XTB",
        },
    )
    _event(events, "provider_request", {"request_id": "req-1", "model": "Pro/moonshotai/Kimi-K2.6"})
    _event(
        events,
        "provider_response_metadata",
        {
            "request_id": "req-1",
            "actual_model": "Pro/moonshotai/Kimi-K2.6",
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        },
    )
    _tool_pair(events, "call_scheduler", "inspect_scheduler_record", {"path": "attempts/attempt_000_initial_failure/scheduler.json"}, {"state": "COMPLETED"})
    _tool_pair(
        events,
        "call_read_log",
        "read_text_range",
        {"path": "attempts/attempt_000_initial_failure/stdout.log", "start_line": 1, "end_line": 3},
        {"content": initial_stdout.rstrip("\n"), "line_count": 3, "byte_count": len(initial_stdout.encode("utf-8"))},
    )
    _tool_pair(events, "call_xyz", "validate_xyz", {"path": "attempts/attempt_000_initial_failure/xtbopt.log", "frame_index": 0}, {"valid": True, "complete_frames": 1})
    _tool_pair(events, "call_prov", "inspect_artifact_provenance", {"path": "attempts/attempt_000_initial_failure/xtbopt.log", "frame_index": 0}, {"current": True, "restart_allowed": True})

    for call_prefix, record in (("opt", opt), ("hess", hess)):
        prepared = record["prepared"]
        attempt_id = prepared["attempt_id"]
        phase = prepared["phase"]
        prepare_args = {
            "phase": phase,
            "source_geometry": {
                "path": prepared["source_geometry"]["path"],
                "frame_index": prepared["source_geometry"]["frame_index"],
            },
            "optimization_cycle_limit": prepared["request"]["optimization_cycle_limit"],
            "thread_count": 1,
            "walltime_seconds": 180,
            "restart_policy": prepared["request"]["restart_policy"],
            "rationale": prepared["request"]["rationale"],
            "preserved_invariants": prepared["request"]["preserved_invariants"],
        }
        _tool_pair(events, f"call_prepare_{call_prefix}", "prepare_restart", prepare_args, {"attempt_id": attempt_id, "state": "PREPARED"})
        _event(
            events,
            "attempt_prepared",
            {
                "attempt_id": attempt_id,
                "phase": phase,
                "state": "PREPARED",
                "source_geometry": {
                    "path": prepared["source_geometry"]["path"],
                    "frame_index": prepared["source_geometry"]["frame_index"],
                    "sha256": prepared["source_geometry"]["file_sha256"],
                    "provenance_state": "CURRENT_VALIDATED",
                },
                "prepared_command_path": record["prepared_path"],
                "input_manifest_path": record["provenance_path"],
            },
        )
        _tool_pair(events, f"call_run_{call_prefix}", "run_xtb", {"attempt_id": attempt_id}, {"attempt_id": attempt_id, "native_xtb_executed": True})
        _event(
            events,
            "native_xtb_execution",
            {
                "attempt_id": attempt_id,
                "phase": phase,
                "source_type": "NATIVE_XTB",
                "argv": record["run_record"]["argv"],
                "shell": False,
                "process_exit_code": 0,
                "timed_out": False,
                "execution_summary_path": record["run_path"],
                "artifact_manifest_path": record["manifest_path"],
                "artifact_hashes": {
                    path: details["sha256"]
                    for path, details in record["run_record"]["files"].items()
                },
            },
        )
        _tool_pair(events, f"call_validate_{call_prefix}", "validate_attempt", {"attempt_id": attempt_id}, record["validation"])
        _event(
            events,
            "attempt_validated",
            {
                "attempt_id": attempt_id,
                "phase": phase,
                "classification": "SUCCESS",
                "valid": True,
                "optimization_converged": phase == "geometry_optimization",
                "hessian_completed": phase == "hessian",
                "current_artifacts": True,
                "scientific_invariants_preserved": True,
                "source_provenance": "CURRENT_VALIDATED",
                "validation_path": record["validation_path"],
            },
        )
        if call_prefix == "opt":
            _event(events, "provider_request", {"request_id": "req-2", "model": "Pro/moonshotai/Kimi-K2.6"})
            _event(
                events,
                "provider_response_metadata",
                {
                    "request_id": "req-2",
                    "actual_model": "Pro/moonshotai/Kimi-K2.6",
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 20},
                },
            )

    _tool_pair(events, "call_submit", "submit_final", model_submission, {"accepted": True})
    _event(
        events,
        "budget_state",
        {
            "tool_calls": sum(event["event"] == "tool_call" for event in events),
            "successful_tool_calls": sum(
                event["event"] == "tool_result" and event["payload"].get("ok") is True
                for event in events
            ),
            "agent_turns": sum(
                event["event"] == "provider_response_metadata" for event in events
            ),
            "execution_attempts": sum(
                event["event"] == "attempt_prepared" for event in events
            ),
            "provider_retries": sum(event["event"] == "provider_retry" for event in events),
            "elapsed_seconds": 30,
        },
    )
    _event(events, "final_submitted", {"final_answer": model_submission})
    _event(events, "run_complete", {"status": "RECOVERED"})
    trajectory_path = project / "trajectory.jsonl"
    trajectory_path.write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in events))
    final_answer = build_final_answer_envelope(
        model_submission=model_submission,
        events=events,
        trajectory_path=trajectory_path,
        artifact_root=case_root,
        genesis_hash=str(task["determinism"]["trajectory_genesis_hash"]),
    )
    _write_json(final_path, final_answer)
    return {
        "project": project,
        "config": config_path,
        "case": case_root,
        "final": final_path,
        "trajectory": trajectory_path,
    }


def rehash_events(path: Path, mutator: Any) -> None:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutator(events)
    previous = "0" * 64
    for index, event in enumerate(events, 1):
        event["sequence"] = index
        event["prev_event_hash"] = previous
        event.pop("event_hash", None)
        event["event_hash"] = _sha(canonical_json_bytes(event)).lower()
        previous = event["event_hash"]
    path.write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in events))


def refresh_final_answer_envelope(paths: Mapping[str, Path]) -> None:
    """Rebuild runner-owned fields after an intentional valid trace mutation."""

    final_value = json.loads(paths["final"].read_text(encoding="utf-8"))
    model_submission = final_value.get("model_submission", final_value)
    events = [
        json.loads(line)
        for line in paths["trajectory"].read_text(encoding="utf-8").splitlines()
    ]
    task = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    envelope = build_final_answer_envelope(
        model_submission=model_submission,
        events=events,
        trajectory_path=paths["trajectory"],
        artifact_root=paths["case"],
        genesis_hash=str(task["determinism"]["trajectory_genesis_hash"]),
    )
    _write_json(paths["final"], envelope)

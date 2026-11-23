"""Deterministic baseline for the frozen xTB recovery case.

Live mode uses the public ``ToolRegistry`` exactly as a model run does, but it
does not contact a model provider.  Replay mode is deliberately smaller: it
only invokes the independent evaluator twice over already-frozen files and
requires the two serialized results (and the stored evaluation) to match.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate import evaluate_files, evaluation_bytes, write_evaluation
from framework import (
    ToolContractSet,
    TrajectoryRecorder,
    build_final_answer_envelope,
    render_dialogue,
    verify_hash_chain,
)
from scripts.runtime_support import clone_frozen_case
from tools.tool_registry import ToolRegistry


CASE_SOURCE = ROOT / "benchmark_case"
DEFAULT_LIVE = ROOT / "baselines" / "live"
DEFAULT_REPLAY_FIXTURE = ROOT / "baselines" / "replay_fixture"
TASK_CONFIG = ROOT / "configs" / "task.yaml"
INITIAL_ATTEMPT = "attempts/attempt_000_initial_failure"
INITIAL_STDOUT = f"{INITIAL_ATTEMPT}/stdout.log"
INITIAL_TRAJECTORY = f"{INITIAL_ATTEMPT}/xtbopt.log"
INITIAL_SCHEDULER = f"{INITIAL_ATTEMPT}/scheduler.json"
LEGACY_GEOMETRY = "legacy/xtbopt.xyz"
INVARIANTS = [
    "method",
    "charge",
    "uhf",
    "solvation",
    "atom_order",
    "composition",
    "required_phases",
]


class BaselineError(RuntimeError):
    """Non-reflective baseline protocol failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_task() -> dict[str, Any]:
    value = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BaselineError("TASK_CONFIG_INVALID")
    return value


def _special_event_payload(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the runner's frozen public special-event projection."""

    if name == "prepare_restart":
        source = result.get("source_geometry")
        source = source if isinstance(source, Mapping) else {}
        return {
            "attempt_id": result.get("attempt_id"),
            "phase": result.get("phase"),
            "state": result.get("state"),
            "source_geometry": {
                "path": source.get("path"),
                "frame_index": source.get("frame_index"),
                "sha256": source.get("sha256", source.get("file_sha256")),
                "provenance_state": source.get(
                    "provenance_state", result.get("source_provenance")
                ),
            },
            "prepared_command_path": result.get(
                "prepared_command_path", result.get("prepared_record")
            ),
            "input_manifest_path": result.get(
                "input_manifest_path", result.get("provenance_record")
            ),
        }
    if name == "run_xtb":
        hashes = result.get("artifact_hashes")
        if not isinstance(hashes, Mapping):
            files = result.get("files")
            hashes = {
                str(path): metadata.get("sha256")
                for path, metadata in files.items()
                if isinstance(path, str)
                and isinstance(metadata, Mapping)
                and isinstance(metadata.get("sha256"), str)
            } if isinstance(files, Mapping) else {}
        return {
            "attempt_id": result.get("attempt_id"),
            "phase": result.get("phase"),
            "source_type": "NATIVE_XTB",
            "argv": result.get("argv"),
            "shell": result.get("shell"),
            "process_exit_code": result.get(
                "process_exit_code", result.get("exit_code")
            ),
            "timed_out": result.get("timed_out"),
            "execution_summary_path": result.get(
                "execution_summary_path", result.get("run_record")
            ),
            "artifact_manifest_path": result.get("artifact_manifest_path"),
            "artifact_hashes": dict(hashes),
        }
    if name == "validate_attempt":
        source_provenance = result.get("source_provenance")
        if not isinstance(source_provenance, (str, Mapping)):
            source_provenance = {"current": result.get("source_current") is True}
        return {
            "attempt_id": result.get("attempt_id"),
            "phase": result.get("phase"),
            "classification": result.get("classification", result.get("state")),
            "valid": result.get("valid", result.get("validation_passed")),
            "optimization_converged": result.get(
                "optimization_converged", result.get("optimization_completed")
            ),
            "hessian_completed": result.get("hessian_completed"),
            "current_artifacts": result.get(
                "current_artifacts", result.get("artifacts_current")
            ),
            "scientific_invariants_preserved": result.get(
                "scientific_invariants_preserved", result.get("invariants_preserved")
            ),
            "source_provenance": source_provenance,
            "validation_path": result.get(
                "validation_path", result.get("validation_record")
            ),
        }
    raise BaselineError("UNKNOWN_SPECIAL_EVENT")


class BaselineSession:
    """Small deterministic policy layer over the exact public tool registry."""

    def __init__(
        self,
        registry: ToolRegistry,
        recorder: TrajectoryRecorder,
        *,
        started: float,
    ) -> None:
        self.registry = registry
        self.recorder = recorder
        self.started = started
        self.tool_calls = 0
        self.successful_tool_calls = 0
        self.execution_attempts = 0
        self.native_executions = 0
        self.results: dict[str, dict[str, Any]] = {}

    def _budget(self) -> None:
        self.recorder.append(
            "budget_state",
            {
                "tool_calls": self.tool_calls,
                "successful_tool_calls": self.successful_tool_calls,
                "agent_turns": 0,
                "execution_attempts": self.execution_attempts,
                "provider_retries": 0,
                "elapsed_seconds": round(max(0.0, time.monotonic() - self.started), 6),
            },
        )

    def call(self, name: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        self.tool_calls += 1
        call_id = f"baseline-{self.tool_calls:03d}-{name}"
        safe_arguments = json.loads(
            json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
        self.recorder.append(
            "tool_call", {"call_id": call_id, "name": name, "arguments": safe_arguments}
        )
        envelope = self.registry.dispatch(name, safe_arguments, call_id)
        if not isinstance(envelope, Mapping) or envelope.get("ok") is not True:
            error = envelope.get("error") if isinstance(envelope, Mapping) else {}
            code = error.get("code") if isinstance(error, Mapping) else None
            safe_code = code if isinstance(code, str) else "TOOL_REPORTED_FAILURE"
            self.recorder.append(
                "tool_result",
                {
                    "call_id": call_id,
                    "name": name,
                    "ok": False,
                    "error": {"type": safe_code, "status": "REJECTED"},
                },
            )
            self._budget()
            raise BaselineError(safe_code)
        result = envelope.get("result")
        if not isinstance(result, Mapping):
            raise BaselineError("TOOL_RESULT_INVALID")
        safe_result = json.loads(
            json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
        self.successful_tool_calls += 1
        if name == "prepare_restart":
            self.execution_attempts += 1
        elif name == "run_xtb":
            self.native_executions += 1
        self.recorder.append(
            "tool_result",
            {
                "call_id": call_id,
                "name": name,
                "ok": True,
                "result": safe_result,
            },
        )
        if name in {"prepare_restart", "run_xtb", "validate_attempt"}:
            special = {
                "prepare_restart": "attempt_prepared",
                "run_xtb": "native_xtb_execution",
                "validate_attempt": "attempt_validated",
            }[name]
            self.recorder.append(special, _special_event_payload(name, safe_result))
        self._budget()
        self.results[call_id] = safe_result
        return call_id, safe_result


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise BaselineError(code)


def _build_final_answer(
    decisive_call_id: str,
    decisive_read: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    optimization_attempt: str,
    hessian_attempt: str,
) -> dict[str, Any]:
    xyz_validation = source_provenance.get("xyz_validation")
    if not isinstance(xyz_validation, Mapping):
        raise BaselineError("SOURCE_XYZ_PROVENANCE_MISSING")
    return {
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "final_status": "RECOVERED",
        "diagnosis": {
            "primary_category": "OPTIMIZATION_FAILURE",
            "primary_cause_code": "OPTIMIZATION_MAX_CYCLES",
            "scheduler_assessment": "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE",
            "conflict_resolution": "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION",
            "legacy_disposition": "NOT_CURRENT_REJECTED",
            "secondary_findings": [
                "The wrapper completed, but the initial native xTB optimization did not converge.",
                "The independent legacy success is neither current evidence nor an authorized restart source.",
            ],
            "confidence": 1.0,
            "evidence": [
                {
                    "tool_call_id": decisive_call_id,
                    "path": decisive_read["path"],
                    "line_start": decisive_read["line_start"],
                    "line_end": decisive_read["line_end"],
                    "content_sha256": decisive_read["content_sha256"],
                    "interpretation": (
                        "The bounded native stdout excerpt records max. optcycles = 1 and "
                        "FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 1 ITERATIONS."
                    ),
                }
            ],
        },
        "restart_summary": {
            "starting_geometry_source_kind": "CURRENT_TRAJECTORY_FRAME",
            "starting_geometry_source": {
                "attempt_id": str(source_provenance["origin"]),
                "path": str(source_provenance["path"]),
                "frame_index": int(source_provenance["frame_index"]),
                "file_sha256": str(source_provenance["sha256"]),
                "frame_sha256": str(xyz_validation["selected_frame_sha256"]),
            },
            "changes": [
                "Raised only the geometry-optimization cycle limit from 1 to 100.",
                "Ran the required Hessian from the independently validated optimized geometry.",
            ],
            "preserved_invariants": list(INVARIANTS),
        },
        "result_summary": {
            "optimization_completed": True,
            "hessian_completed": True,
            "final_attempt_id": hessian_attempt,
            "warnings": [],
        },
        "rationale": (
            f"Attempt {optimization_attempt} repaired the artificial one-cycle limit using the last "
            f"complete current trajectory frame. After its independent validation, {hessian_attempt} "
            "completed and validated the required native xTB Hessian without changing method, charge, "
            "spin, solvation, atom order, composition, or required phases."
        ),
    }


def _run_policy(session: BaselineSession) -> dict[str, Any]:
    session.call("list_case_files", {})

    _, job_read = session.call(
        "read_text_range", {"path": "job_spec.json", "start_line": 1, "end_line": 80}
    )
    try:
        job_spec = json.loads(str(job_read["content"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BaselineError("JOB_SPEC_READ_INVALID") from exc
    _require(job_spec.get("case_id") == "aspirin_gfn2_opt_hess_restart_v1", "CASE_ID_MISMATCH")
    _require(job_spec.get("method") == "GFN2-xTB", "METHOD_MISMATCH")

    decisive_call, decisive = session.call(
        "read_text_range",
        {"path": INITIAL_STDOUT, "start_line": 239, "end_line": 284},
    )
    decisive_text = str(decisive.get("content", ""))
    _require("max. optcycles" in decisive_text, "MAX_CYCLE_EVIDENCE_MISSING")
    _require("FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 1 ITERATIONS" in decisive_text,
             "CONVERGENCE_EVIDENCE_MISSING")

    _, parsed = session.call("parse_xtb_log", {"path": INITIAL_STDOUT})
    latest = parsed.get("latest_segment")
    latest = latest if isinstance(latest, Mapping) else {}
    facts = latest.get("facts")
    _require(
        isinstance(facts, list)
        and any(isinstance(item, Mapping) and item.get("code") == "OPTIMIZATION_MAX_CYCLES" for item in facts),
        "PARSED_CAUSE_MISMATCH",
    )

    _, scheduler = session.call("inspect_scheduler_record", {"path": INITIAL_SCHEDULER})
    scheduler_state = scheduler.get("scheduler_state", scheduler.get("state"))
    _require(str(scheduler_state).upper() == "COMPLETED", "SCHEDULER_STATE_MISMATCH")

    _, current_xyz = session.call("validate_xyz", {"path": INITIAL_TRAJECTORY})
    _require(current_xyz.get("valid") is True, "CURRENT_FRAME_INVALID")
    _require(current_xyz.get("selected_frame_index") == 0, "CURRENT_LAST_FRAME_MISMATCH")

    _, current_provenance = session.call(
        "inspect_artifact_provenance", {"path": INITIAL_TRAJECTORY, "frame_index": 0}
    )
    _require(current_provenance.get("current") is True, "CURRENT_PROVENANCE_REJECTED")
    _require(current_provenance.get("restart_allowed") is True, "CURRENT_RESTART_REJECTED")

    _, legacy = session.call(
        "inspect_artifact_provenance", {"path": LEGACY_GEOMETRY, "frame_index": 0}
    )
    _require(legacy.get("current") is False, "LEGACY_CURRENTNESS_INVALID")
    _require(legacy.get("restart_allowed") is False, "LEGACY_RESTART_POLICY_INVALID")

    _, prepared_opt = session.call(
        "prepare_restart",
        {
            "phase": "geometry_optimization",
            "source_geometry": {"path": INITIAL_TRAJECTORY, "frame_index": 0},
            "optimization_cycle_limit": 100,
            "thread_count": 1,
            "walltime_seconds": 180,
            "restart_policy": "last_complete_current_frame",
            "rationale": "Repair the artificial one-cycle limit using the last complete current trajectory frame.",
            "preserved_invariants": list(INVARIANTS),
        },
    )
    optimization_attempt = str(prepared_opt.get("attempt_id", ""))
    _require(bool(optimization_attempt), "OPTIMIZATION_ATTEMPT_ID_MISSING")
    session.call("run_xtb", {"attempt_id": optimization_attempt})
    _, validated_opt = session.call("validate_attempt", {"attempt_id": optimization_attempt})
    _require(validated_opt.get("optimization_completed") is True, "OPTIMIZATION_NOT_VALIDATED")
    _require(validated_opt.get("validation_passed") is True, "OPTIMIZATION_VALIDATION_FAILED")

    optimized_geometry = f"attempts/{optimization_attempt}/xtbopt.xyz"
    _, prepared_hessian = session.call(
        "prepare_restart",
        {
            "phase": "hessian",
            "source_geometry": {"path": optimized_geometry, "frame_index": 0},
            "optimization_cycle_limit": None,
            "thread_count": 1,
            "walltime_seconds": 180,
            "restart_policy": "validated_optimized_geometry",
            "rationale": "Run the required Hessian only after independent optimization validation.",
            "preserved_invariants": list(INVARIANTS),
        },
    )
    hessian_attempt = str(prepared_hessian.get("attempt_id", ""))
    _require(bool(hessian_attempt), "HESSIAN_ATTEMPT_ID_MISSING")
    session.call("run_xtb", {"attempt_id": hessian_attempt})
    _, validated_hessian = session.call("validate_attempt", {"attempt_id": hessian_attempt})
    _require(validated_hessian.get("hessian_completed") is True, "HESSIAN_NOT_VALIDATED")
    _require(validated_hessian.get("validation_passed") is True, "HESSIAN_VALIDATION_FAILED")

    final_answer = _build_final_answer(
        decisive_call,
        decisive,
        current_provenance,
        optimization_attempt,
        hessian_attempt,
    )
    _, submitted = session.call("submit_final", final_answer)
    _require(submitted.get("status") == "SUBMITTED", "FINAL_NOT_SUBMITTED")
    _require(session.registry.frozen, "FINAL_DID_NOT_FREEZE")
    return final_answer


def _manifest(output: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*"), key=lambda item: item.relative_to(output).as_posix()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema_version": "1.0",
        "source_type": "BASELINE_LIVE_XTB",
        "immutable": True,
        "files": files,
    }


def run_live(
    output: Path = DEFAULT_LIVE,
    *,
    registry_factory: Callable[..., ToolRegistry] = ToolRegistry,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_BASELINE_OUTPUT")
    if not output.is_relative_to(ROOT / "baselines"):
        raise BaselineError("OUTPUT_MUST_BE_INSIDE_BASELINES")
    if not CASE_SOURCE.is_dir():
        raise FileNotFoundError("FROZEN_BENCHMARK_CASE_MISSING")
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    clone_frozen_case(workspace)

    task = _load_task()
    task_block = task.get("task")
    task_block = task_block if isinstance(task_block, Mapping) else {}
    protocol = task_block.get("protocol")
    determinism = task.get("determinism")
    determinism = determinism if isinstance(determinism, Mapping) else {}
    started = time.monotonic()
    recorder = TrajectoryRecorder(
        protocol=protocol,
        run_id="baseline-live-aspirin-gfn2-opt-hess-restart-v1",
        genesis_hash=str(determinism.get("trajectory_genesis_hash", "0" * 64)),
        timestamp_fn=_utc_now,
    )
    recorder.append(
        "run_start",
        {
            "task_id": "xtb_failure_diagnosis_restart_agent_v1",
            "case_id": "aspirin_gfn2_opt_hess_restart_v1",
            "protocol": dict(protocol) if isinstance(protocol, Mapping) else {},
            "run_type": "BASELINE",
            "logical_model": "deterministic_baseline",
            "requested_model": "none",
            "provider": "none",
            "transport": "local_deterministic_policy",
            "execution_mode": "BASELINE",
            "native_mode": "LIVE_XTB",
        },
    )
    registry = registry_factory(ROOT, case_root=workspace.relative_to(ROOT).as_posix())
    session = BaselineSession(registry, recorder, started=started)
    try:
        model_submission = _run_policy(session)
        recorder.append(
            "final_submitted", {"final_answer": model_submission}
        )
        recorder.append(
            "run_complete",
            {
                "status": "COMPLETE",
                "terminal_status": "COMPLETE",
                "actual_model": None,
                "tool_calls": session.tool_calls,
                "successful_tool_calls": session.successful_tool_calls,
            },
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if not recorder.terminal and not recorder.final_submitted:
            recorder.append(
                "run_failure",
                {
                    "status": "FAILED",
                    "terminal_status": "FAILED",
                    "error": {"type": type(exc).__name__, "status": "BASELINE_FAILED"},
                    "tool_calls": session.tool_calls,
                    "successful_tool_calls": session.successful_tool_calls,
                },
            )
            recorder.write(output / "trajectory.jsonl")
        raise

    recorder.write(output / "trajectory.jsonl")
    final_answer = build_final_answer_envelope(
        model_submission=model_submission,
        events=recorder.events,
        trajectory_path=output / "trajectory.jsonl",
        artifact_root=workspace,
        genesis_hash=str(determinism.get("trajectory_genesis_hash", "0" * 64)),
    )
    contracts = ToolContractSet.load(ROOT / "configs" / "tool_contracts.json")
    final_answer = contracts.validate_final_answer_artifact(final_answer)
    _write_json_exclusive(output / "final_answer.json", final_answer)
    with (output / "dialogue.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(render_dialogue(recorder.events))

    evaluation = evaluate_files(
        final_answer_path=output / "final_answer.json",
        trajectory_path=output / "trajectory.jsonl",
        artifacts_path=workspace,
        config_path=TASK_CONFIG,
    )
    write_evaluation(output / "evaluation.json", evaluation)
    hashes = {
        name: _sha256_file(output / name)
        for name in ("final_answer.json", "trajectory.jsonl", "dialogue.md", "evaluation.json")
    }
    metadata = {
        "schema_version": "1.0",
        "status": "COMPLETE",
        "run_id": "baseline-live-aspirin-gfn2-opt-hess-restart-v1",
        "task_id": "xtb_failure_diagnosis_restart_agent_v1",
        "case_id": "aspirin_gfn2_opt_hess_restart_v1",
        "protocol": dict(protocol) if isinstance(protocol, Mapping) else {},
        "run_type": "BASELINE",
        "execution_mode": "BASELINE",
        "native_mode": "LIVE_XTB",
        "logical_model": "deterministic_baseline",
        "requested_model": "none",
        "actual_model": None,
        "provider": "none",
        "transport": "local_deterministic_policy",
        "counts": {
            "agent_turns": 0,
            "provider_attempts": 0,
            "provider_failures": 0,
            "provider_retries": 0,
            "tool_calls": session.tool_calls,
            "successful_tool_calls": session.successful_tool_calls,
            "failed_tool_calls": 0,
            "execution_attempts": session.execution_attempts,
            "native_executions": session.native_executions,
        },
        "submitted": True,
        "submission_frozen": True,
        "hash_chain": {
            "algorithm": "sha256",
            "genesis_hash": recorder.genesis_hash,
            "head_hash": recorder.head_hash,
            "event_count": len(recorder.events),
            "valid": verify_hash_chain(
                recorder.events, genesis_hash=recorder.genesis_hash, require_terminal=True
            ),
        },
        "artifact_sha256": hashes,
        "request_ids": [],
        "hidden_reasoning_persisted": False,
        "failure": None,
        "artifacts": [
            "trajectory.jsonl",
            "dialogue.md",
            "metadata.json",
            "final_answer.json",
            "evaluation.json",
            "artifact_manifest.json",
            "workspace",
        ],
    }
    _write_json_exclusive(output / "metadata.json", metadata)
    _write_json_exclusive(output / "artifact_manifest.json", _manifest(output))
    return {
        "output": output,
        "official_score": evaluation.get("official_score"),
        "eligible": evaluation.get("eligibility", {}).get("eligible"),
        "tool_calls": session.tool_calls,
        "native_executions": session.native_executions,
    }


def _default_replay_source() -> Path:
    if (DEFAULT_REPLAY_FIXTURE / "trajectory.jsonl").is_file():
        return DEFAULT_REPLAY_FIXTURE
    return DEFAULT_LIVE


def run_replay(
    source: Path | None = None,
    *,
    evaluator_fn: Callable[..., Mapping[str, Any]] = evaluate_files,
) -> dict[str, Any]:
    source = (source or _default_replay_source()).expanduser().resolve()
    required = {
        "final_answer.json": source / "final_answer.json",
        "trajectory.jsonl": source / "trajectory.jsonl",
        "evaluation.json": source / "evaluation.json",
        "workspace": source / "workspace",
    }
    if not all(path.is_file() if name != "workspace" else path.is_dir() for name, path in required.items()):
        raise FileNotFoundError("FROZEN_BASELINE_REPLAY_SOURCE_INCOMPLETE")
    arguments = {
        "final_answer_path": required["final_answer.json"],
        "trajectory_path": required["trajectory.jsonl"],
        "artifacts_path": required["workspace"],
        "config_path": TASK_CONFIG,
    }
    first = evaluator_fn(**arguments)
    second = evaluator_fn(**arguments)
    first_bytes = evaluation_bytes(first)
    second_bytes = evaluation_bytes(second)
    if first_bytes != second_bytes:
        raise BaselineError("REPLAY_EVALUATION_NOT_BYTE_IDENTICAL")
    stored = required["evaluation.json"].read_bytes()
    if stored != first_bytes:
        raise BaselineError("FROZEN_EVALUATION_MISMATCH")
    return {
        "source": source,
        "evaluation_sha256": hashlib.sha256(first_bytes).hexdigest().upper(),
        "byte_identical": True,
        "eligible": first.get("eligibility", {}).get("eligible"),
        "official_score": first.get("official_score"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or replay the deterministic xTB baseline")
    parser.add_argument("--mode", required=True, choices=["live", "replay"])
    parser.add_argument("--output", help="New live output directory; existing targets are never overwritten")
    parser.add_argument("--source", help="Frozen baseline source for replay mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "live":
        if args.source:
            raise BaselineError("SOURCE_IS_REPLAY_ONLY")
        result = run_live(Path(args.output) if args.output else DEFAULT_LIVE)
    else:
        if args.output:
            raise BaselineError("OUTPUT_IS_LIVE_ONLY")
        result = run_replay(Path(args.source) if args.source else None)
    printable = {key: (value.as_posix() if isinstance(value, Path) else value) for key, value in result.items()}
    print(json.dumps(printable, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

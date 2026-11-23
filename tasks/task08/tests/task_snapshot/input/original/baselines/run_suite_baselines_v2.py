"""Run weak and deterministic-evidence baselines over the six-case v2 suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.baseline_agent import BaselineSession
from evaluator.evaluate import evaluate_files, write_evaluation
from framework import ToolContractSet, TrajectoryRecorder, build_final_answer_envelope
from tools.tool_registry import ToolRegistry


INVARIANTS = [
    "method", "charge", "uhf", "solvation", "atom_order", "composition", "required_phases"
]
INITIAL = "attempts/attempt_000_initial_failure"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_suite() -> list[dict[str, str]]:
    suite = json.loads((ROOT / "organizer_private" / "suite.json").read_text(encoding="utf-8"))
    return [suite["public_case"], *suite["hidden_cases"]]


def fact_codes(parsed: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    latest = parsed.get("latest_segment")
    latest = latest if isinstance(latest, Mapping) else {}
    facts = latest.get("facts")
    facts = facts if isinstance(facts, list) else []
    return {
        str(item.get("code")): item
        for item in facts
        if isinstance(item, Mapping) and isinstance(item.get("code"), str)
    }


def diagnosis(
    stdout_parsed: Mapping[str, Any],
    stderr_parsed: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> tuple[dict[str, str], str, Mapping[str, Any]]:
    stdout = fact_codes(stdout_parsed)
    stderr = fact_codes(stderr_parsed)
    combined = {**stderr, **stdout}
    state = str(scheduler.get("canonical_state", "UNKNOWN"))
    if scheduler.get("timed_out") is True or state == "TIMEOUT":
        values = ("RESOURCE_FAILURE", "TIMEOUT", "INCOMPLETE_OUTPUT_OVERRIDES_PROCESS_STATUS")
        evidence_code = next(iter(stdout), next(iter(stderr), ""))
    elif "UNSUPPORTED_OPTION" in combined:
        values = ("INPUT_FAILURE", "UNSUPPORTED_OPTION", "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION")
        evidence_code = "UNSUPPORTED_OPTION"
    elif "COORDINATE_FORMAT_ERROR" in combined:
        values = ("INPUT_FAILURE", "COORDINATE_FORMAT_ERROR", "INPUT_REJECTION_PREVENTED_SCIENTIFIC_RUN")
        evidence_code = "COORDINATE_FORMAT_ERROR"
    elif "SCC_NON_CONVERGENCE" in combined:
        values = ("SCC_FAILURE", "SCC_NON_CONVERGENCE", "INCOMPLETE_OUTPUT_OVERRIDES_PROCESS_STATUS")
        evidence_code = "SCC_NON_CONVERGENCE"
    elif "OPTIMIZATION_MAX_CYCLES" in combined:
        values = ("OPTIMIZATION_FAILURE", "OPTIMIZATION_MAX_CYCLES", "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION")
        evidence_code = "OPTIMIZATION_MAX_CYCLES"
    else:
        values = ("PROGRAM_FAILURE", "WRONG_PHASE_OR_MISSING_OUTPUT", "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION")
        evidence_code = "NORMAL_TERMINATION"
    scheduler_assessment = (
        "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE"
        if state == "COMPLETED" else "EXECUTION_FAILED"
    )
    fact = stdout.get(evidence_code) or stderr.get(evidence_code) or next(
        iter(stdout.values()), next(iter(stderr.values()), {})
    )
    return {
        "primary_category": values[0],
        "primary_cause_code": values[1],
        "scheduler_assessment": scheduler_assessment,
        "conflict_resolution": values[2],
    }, (f"{INITIAL}/stdout.log" if evidence_code in stdout else f"{INITIAL}/stderr.log"), fact


def deliberately_wrong_diagnosis(correct: Mapping[str, str]) -> dict[str, str]:
    """Return a schema-valid but scientifically wrong fixed-policy diagnosis."""

    if correct.get("primary_cause_code") != "UNSUPPORTED_OPTION":
        category, cause = "INPUT_FAILURE", "UNSUPPORTED_OPTION"
    else:
        category, cause = "OPTIMIZATION_FAILURE", "OPTIMIZATION_MAX_CYCLES"
    assessment = (
        "EXECUTION_FAILED"
        if correct.get("scheduler_assessment") == "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE"
        else "EXECUTION_COMPLETED_SCIENCE_INCOMPLETE"
    )
    return {
        "primary_category": category,
        "primary_cause_code": cause,
        "scheduler_assessment": assessment,
        "conflict_resolution": "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION",
    }


def evidence_from_fact(
    session: BaselineSession, path: str, fact: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    line = fact.get("line_start")
    line = line if type(line) is int and line > 0 else 1
    call_id, read = session.call(
        "read_text_range", {"path": path, "start_line": line, "end_line": line}
    )
    return call_id, read


def final_answer(
    *, case_id: str, diag: Mapping[str, str], evidence_call: str,
    evidence_read: Mapping[str, Any], legacy_disposition: str,
    source: Mapping[str, Any] | None, hessian_attempt: str,
    recovered: bool,
) -> dict[str, Any]:
    source_value = source or {
        "path": f"{INITIAL}/input.xyz", "frame_index": 0,
        "file_sha256": "0" * 64, "frame_sha256": "0" * 64,
    }
    return {
        "case_id": case_id,
        "final_status": "RECOVERED" if recovered else "FAILED_WITHIN_BUDGET",
        "diagnosis": {
            **diag,
            "legacy_disposition": legacy_disposition,
            "secondary_findings": [],
            "confidence": 1.0 if recovered else 0.6,
            "evidence": [{
                "tool_call_id": evidence_call,
                "path": evidence_read["path"],
                "line_start": evidence_read["line_start"],
                "line_end": evidence_read["line_end"],
                "content_sha256": evidence_read["content_sha256"],
                "interpretation": "The cited native initial-attempt record supports the classified failure.",
            }],
        },
        "restart_summary": {
            "starting_geometry_source_kind": "ORIGINAL_INPUT" if recovered else "NONE",
            "starting_geometry_source": {
                "attempt_id": "attempt_000_initial_failure",
                "path": source_value["path"],
                "frame_index": int(source_value.get("frame_index", 0)),
                "file_sha256": source_value.get("file_sha256", source_value.get("sha256", "0" * 64)),
                "frame_sha256": source_value.get("frame_sha256", "0" * 64),
            },
            "changes": (["Used the valid original input and reran both required phases."] if recovered else []),
            "preserved_invariants": list(INVARIANTS) if recovered else [],
        },
        "result_summary": {
            "optimization_completed": recovered,
            "hessian_completed": recovered,
            "final_attempt_id": hessian_attempt if recovered else "",
            "warnings": [],
        },
        "rationale": (
            "The deterministic policy classified native evidence, then independently validated optimization and Hessian."
            if recovered else "This non-recovery calibration policy submits a bounded diagnostic outcome."
        ),
    }


def run_case(mode: str, case: Mapping[str, str], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE:{output}")
    output.mkdir(parents=True)
    workspace = output / "workspace"
    shutil.copytree(ROOT / case["workspace"], workspace)
    config_path = ROOT / case["config"]
    task = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = task["task"]["protocol"]
    recorder = TrajectoryRecorder(
        protocol=protocol,
        run_id=f"baseline-{mode}-{case['case_id']}",
        genesis_hash=task["determinism"]["trajectory_genesis_hash"],
        timestamp_fn=utc_now,
    )
    recorder.append("run_start", {
        "task_id": task["task"]["id"], "case_id": case["case_id"],
        "protocol": protocol, "run_type": "BASELINE", "logical_model": f"{mode}_baseline",
        "requested_model": "none", "provider": "none", "transport": "local_deterministic_policy",
        "execution_mode": "BASELINE", "native_mode": "LIVE_XTB",
    })
    registry = ToolRegistry(
        ROOT,
        task_config_path=case["config"],
        case_root=workspace.relative_to(ROOT).as_posix(),
    )
    session = BaselineSession(registry, recorder, started=time.monotonic())
    session.call("list_case_files", {})
    _, parsed_out = session.call("parse_xtb_log", {"path": f"{INITIAL}/stdout.log"})
    _, parsed_err = session.call("parse_xtb_log", {"path": f"{INITIAL}/stderr.log"})
    _, scheduler = session.call("inspect_scheduler_record", {"path": f"{INITIAL}/scheduler.json"})
    diag, evidence_path, fact = diagnosis(parsed_out, parsed_err, scheduler)
    if mode == "adversarial_wrong_science":
        diag = deliberately_wrong_diagnosis(diag)
    evidence_call, evidence_read = evidence_from_fact(session, evidence_path, fact)
    files = session.results[next(iter(session.results))].get("files", [])
    legacy_present = any(
        isinstance(item, Mapping) and str(item.get("path", "")).startswith("legacy/")
        for item in files if isinstance(files, list)
    )
    legacy_disposition = "NOT_CURRENT_INFORMATION_ONLY" if legacy_present else "NOT_INSPECTED"
    source: Mapping[str, Any] | None = None
    hessian_attempt = ""
    if mode == "deterministic_evidence":
        source_path = f"{INITIAL}/input.xyz"
        session.call("validate_xyz", {"path": source_path, "frame_index": 0})
        session.call("inspect_artifact_provenance", {"path": source_path, "frame_index": 0})
        opt_attempt = ""
        opt_validation: Mapping[str, Any] = {}
        for native_try in range(2):
            _, prepared_opt = session.call("prepare_restart", {
                "phase": "geometry_optimization",
                "source_geometry": {"path": source_path, "frame_index": 0},
                "optimization_cycle_limit": 100, "thread_count": 1, "walltime_seconds": 180 + native_try,
                "restart_policy": "original_input",
                "rationale": (
                    "Recover from the valid original geometry."
                    if native_try == 0
                    else "Retry once after a transient native process failure; scientific inputs are unchanged."
                ),
                "preserved_invariants": list(INVARIANTS),
            })
            source = prepared_opt["source_geometry"]
            opt_attempt = prepared_opt["attempt_id"]
            session.call("run_xtb", {"attempt_id": opt_attempt})
            _, opt_validation = session.call("validate_attempt", {"attempt_id": opt_attempt})
            if opt_validation.get("optimization_completed") is True:
                break
        if opt_validation.get("optimization_completed") is not True:
            raise RuntimeError("BASELINE_OPTIMIZATION_FAILED_AFTER_BOUNDED_RETRY")
        _, prepared_hess = session.call("prepare_restart", {
            "phase": "hessian",
            "source_geometry": {"path": f"attempts/{opt_attempt}/xtbopt.xyz", "frame_index": 0},
            "optimization_cycle_limit": None, "thread_count": 1, "walltime_seconds": 180,
            "restart_policy": "validated_optimized_geometry", "rationale": "Complete required Hessian.",
            "preserved_invariants": list(INVARIANTS),
        })
        hessian_attempt = prepared_hess["attempt_id"]
        session.call("run_xtb", {"attempt_id": hessian_attempt})
        _, hess_validation = session.call("validate_attempt", {"attempt_id": hessian_attempt})
        if hess_validation.get("hessian_completed") is not True:
            raise RuntimeError("BASELINE_HESSIAN_FAILED")
    answer = final_answer(
        case_id=case["case_id"], diag=diag, evidence_call=evidence_call,
        evidence_read=evidence_read, legacy_disposition=legacy_disposition,
        source=source, hessian_attempt=hessian_attempt,
        recovered=mode == "deterministic_evidence",
    )
    session.call("submit_final", answer)
    recorder.append("final_submitted", {"final_answer": answer})
    recorder.append("run_complete", {
        "status": "COMPLETE", "terminal_status": "COMPLETE", "actual_model": None,
        "tool_calls": session.tool_calls, "successful_tool_calls": session.successful_tool_calls,
    })
    recorder.write(output / "trajectory.jsonl")
    envelope = build_final_answer_envelope(
        model_submission=answer, events=recorder.events,
        trajectory_path=output / "trajectory.jsonl", artifact_root=workspace,
        genesis_hash=task["determinism"]["trajectory_genesis_hash"],
    )
    envelope = ToolContractSet.load(ROOT / "configs" / "tool_contracts.json").validate_final_answer_artifact(envelope)
    write_json(output / "final_answer.json", envelope)
    evaluation = evaluate_files(
        final_answer_path=output / "final_answer.json",
        trajectory_path=output / "trajectory.jsonl",
        artifacts_path=workspace,
        config_path=config_path,
    )
    write_evaluation(output / "evaluation.json", evaluation)
    return {
        "case_id": case["case_id"], "mode": mode,
        "raw_score": evaluation.get("raw_score_before_scientific_cap"),
        "ungated_score": evaluation.get("ungated_score"),
        "official_score": evaluation.get("official_score"),
        "eligible": evaluation.get("eligibility", {}).get("eligible"),
        "output": output.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "baselines" / "v2_suite_results",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("weak_diagnosis_only", "adversarial_wrong_science", "deterministic_evidence"),
        default=("weak_diagnosis_only", "adversarial_wrong_science", "deterministic_evidence"),
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError("REFUSING_TO_OVERWRITE_BASELINE_SUITE")
    results: list[dict[str, Any]] = []
    for mode in args.modes:
        for case in load_suite():
            result = run_case(mode, case, output_root / mode / case["case_id"])
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    summary = {
        "schema_version": "2.0", "case_count": len(load_suite()), "results": results,
    }
    write_json(output_root / "summary.json", summary)


if __name__ == "__main__":
    main()

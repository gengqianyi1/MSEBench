#!/usr/bin/env python3
"""Fail-closed independent replay grader for Task 07 Harbor evidence."""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

WORK = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
EVIDENCE = Path(os.environ.get("CHEMBENCH_ORGANIZER_EVIDENCE", "/organizer_evidence"))
RUNTIME = Path("/task/private_data/runtime_package")
if not RUNTIME.is_dir():
    RUNTIME = Path(__file__).resolve().parent / "task_snapshot" / "private_data" / "runtime_package"

sys.path.insert(0, str(RUNTIME))
from evaluator.evaluate import evaluate_submission
from evaluator.evaluate_suite import aggregate_suite, load_suite
from tools.rdkit_tools import TOOL_REGISTRY

FROZEN_MODEL_IDS = {
    "deepseek-ai/DeepSeek-V4-Pro",
    "Pro/moonshotai/Kimi-K2.6",
    "zai-org/GLM-5.2",
    # Organizer account models on the antchat gateway (registered 2026-09-16):
    # this account cannot reach the siliconflow model IDs above.
    "ling-3.0-flash",
    "Ling-3.0-Flash",
    "GLM-5.3",
    "Kimi-K3",
    # 2026-09-28 fix: exact antchat gateway model IDs (membership test is
    # case-sensitive; ling_3flash reports "Ling-3.0-Flash", previously only the
    # mis-cased "ling-3.0-flash" was listed and never matched).
    "Qwen3.8-2.4T-A95B",
    "Qwen3.8-Flash-Next",
    "DeepSeek-V4.1-Flash",
    # 2026-09-29: ling_3plus SFT checkpoint (internal account, antchat).
    "plus_sft_ep3_256k_mtp_0811",
}
# Organizer-only lifecycle validation. The default model-evaluation allowlist is
# unchanged; this flag is supplied to the separate verifier, never by submissions.
if os.environ.get("CHEMBENCH_ORACLE_VALIDATION") == "1":
    FROZEN_MODEL_IDS = FROZEN_MODEL_IDS | {"organizer/scripted-oracle-v1"}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON_OBJECT_REQUIRED")
    return value


def _evidence_root(evidence_id: Any, expected_case_id: str) -> tuple[Path, str]:
    if not isinstance(evidence_id, str) or "\\" in evidence_id:
        raise ValueError("EVIDENCE_ID_INVALID")
    parts = PurePosixPath(evidence_id).parts
    if len(parts) != 2 or parts[1] != expected_case_id or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("EVIDENCE_ID_INVALID")
    root = (EVIDENCE / parts[0] / parts[1]).resolve()
    evidence_root = EVIDENCE.resolve()
    if evidence_root not in root.parents:
        raise ValueError("EVIDENCE_PATH_ESCAPE")
    return root, parts[0]


def _replay_response(config: dict[str, Any], event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise TypeError("TOOL_EVENT_INVALID")
    name = event.get("tool")
    arguments = event.get("arguments")
    if name not in TOOL_REGISTRY or not isinstance(arguments, dict):
        return {"ok": False, "error": "invalid_tool"}
    function = TOOL_REGISTRY[name]
    kwargs = dict(arguments)
    if "config" in inspect.signature(function).parameters:
        kwargs["config"] = config
    try:
        result = function(**kwargs)
    except (TypeError, ValueError):
        return {"ok": False, "tool": name, "error": "invalid_arguments"}
    return {"ok": True, "tool": name, "result": result}


def _process_events(events: list[dict[str, Any]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = [{"sequence": 1, "event": "run_start"}]
    for sequence, item in enumerate(events, 2):
        response = item["response"]
        success = response.get("ok") is True and isinstance(response.get("result"), dict)
        row = {
            "sequence": sequence,
            "event": "tool_call",
            "call_id": f"harbor-{sequence - 1:03d}",
            "tool": item.get("tool"),
            "arguments": item.get("arguments"),
            "status": "SUCCESS" if success else "FAILED",
        }
        if success:
            row["result"] = response["result"]
        else:
            row["error"] = {"type": "ToolError"}
        output.append(row)
    output.append(
        {
            "sequence": len(output) + 1,
            "event": "run_complete",
            "final_candidate": {"canonical_smiles": candidate.get("smiles")},
        }
    )
    return output


def _grade_case(index: int, config_path: Path, evidence_id: Any, runs: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    case_id = config["case"]["id"]
    root, token = _evidence_root(evidence_id, case_id)
    evidence = _object(root / "evidence.json")
    if (
        evidence.get("schema_version") != "task07-organizer-evidence/1.0"
        or evidence.get("session_id") != token
        or evidence.get("case_id") != case_id
        or evidence.get("case_index") != index
    ):
        raise ValueError("EVIDENCE_METADATA_MISMATCH")
    requested_model = evidence.get("requested_model_id")
    if (
        evidence.get("actual_model_id") != requested_model
        or evidence.get("exact_model_identity") is not True
    ):
        raise ValueError("EXACT_MODEL_IDENTITY_FAILED")
    candidate = evidence.get("candidate")
    tool_events = evidence.get("tool_events")
    if not isinstance(candidate, dict) or not isinstance(tool_events, list):
        raise TypeError("EVIDENCE_CONTENT_INVALID")
    if len(tool_events) > int(config["budget"]["max_tool_calls"]):
        raise ValueError("TOOL_BUDGET_EXCEEDED")

    replayed: list[dict[str, Any]] = []
    for raw in tool_events:
        expected = _replay_response(config, raw)
        if raw.get("response") != expected:
            raise ValueError("TOOL_REPLAY_MISMATCH")
        replayed.append(
            {"tool": raw.get("tool"), "arguments": raw.get("arguments"), "response": expected}
        )

    evaluated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in replayed:
        response = item["response"]
        result = response.get("result") if isinstance(response, dict) else None
        arguments = item.get("arguments")
        if (
            item.get("tool") == "evaluate_candidate"
            and isinstance(arguments, dict)
            and isinstance(result, dict)
            and result.get("ok") is True
            and result.get("valid") is True
            and isinstance(result.get("canonical_smiles"), str)
            and isinstance(arguments.get("smiles"), str)
            and result["canonical_smiles"] not in seen
        ):
            seen.add(result["canonical_smiles"])
            evaluated.append((arguments["smiles"], result["canonical_smiles"]))

    final_smiles = candidate.get("smiles")
    final_measurement = TOOL_REGISTRY["evaluate_candidate"](final_smiles, config)
    final_canonical = final_measurement.get("canonical_smiles") if isinstance(final_measurement, dict) else None
    answer = {
        "schema_version": config["output"]["schema_version"],
        "task_id": config["task"]["id"],
        "final_candidate": {key: candidate.get(key, "") for key in ("name", "smiles", "rationale")},
        "candidate_history": [
            {
                "name": f"evaluated_{number}",
                "smiles": raw,
                "rationale": "Harbor protected-tool evaluation",
            }
            for number, (raw, _) in enumerate(evaluated, 1)
        ],
        "run_metadata": {
            "tool_call_count": len(replayed),
            "budget_exceeded": False,
            "unique_evaluated_candidates": [canonical for _, canonical in evaluated],
            "final_candidate_evaluated": final_canonical in {canonical for _, canonical in evaluated},
        },
    }
    evaluation = evaluate_submission(
        answer,
        config,
        process_events=_process_events(replayed, candidate),
    )
    case_root = runs / case_id
    _write(case_root / "final_answer.json", answer)
    _write(case_root / "evaluation.json", evaluation)
    eligible = evaluation.get("eligible") is True or (
        isinstance(evaluation.get("score"), dict)
        and evaluation["score"].get("gate_passed") is True
    )
    _write(
        case_root / "status.json",
        {
            "run_status": "PASS" if eligible else "FAIL",
            "provider_status": evidence.get("provider_status", "HARBOR_PROTOCOL"),
        },
    )
    return evaluation, token


def main() -> int:
    suite_path = RUNTIME / "organizer_private" / "suite.yaml"
    _, paths = load_suite(suite_path)
    runs = WORK / "benchmark_run" / "protocol"
    failures: list[str] = []
    tokens: set[str] = set()
    for index, config_path in enumerate(paths, 1):
        try:
            artifact = _object(WORK / "cases" / f"case-{index}" / "case_result.json")
            submission = artifact.get("submission")
            if not isinstance(submission, dict) or submission.get("ok") is not True:
                raise ValueError("SUBMISSION_REFERENCE_INVALID")
            _, token = _grade_case(index, config_path, submission.get("evidence_id"), runs)
            tokens.add(token)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"CASE_{index}_{str(exc) or type(exc).__name__}")

    if len(tokens) > 1:
        failures.append("MIXED_SESSION_EVIDENCE")

    if failures:
        suite = {
            "schema_version": "qed_suite_evaluation/4.0",
            "case_count": len(paths),
            "case_results": [],
            "capability_score": 0.0,
            "blocked_count": 0,
            "missing_count": len(failures),
            "release_complete": False,
            "ranking_eligible": False,
        }
        score: float | None = 0.0
        status = "FAIL"
    else:
        suite = aggregate_suite(suite_path, runs)
        blocked = int(suite.get("blocked_count", 0))
        case_failed = any(row.get("run_status") == "FAIL" for row in suite.get("case_results", []))
        status = "BLOCKED" if blocked else ("FAIL" if case_failed else "PASS")
        score = None if blocked else float(suite.get("capability_score") or 0.0)

    rows = suite.get("case_results", [])
    campaign = {
        "schema_version": "task07-harbor-campaign/1.0",
        "protocol": "chembench-hidden-suite/1.0",
        "case_count": len(paths),
        "session_count": len(tokens),
        "rows": rows,
        "all_replays_match": not failures,
        "failure_codes": failures,
    }
    _write(runs / "suite_evaluation.json", suite)
    _write(WORK / "benchmark_run" / "suite_evaluation.json", suite)
    _write(WORK / "benchmark_run" / "campaign_summary.json", campaign)
    _write(
        WORK / "benchmark_result.json",
        {
            "task_id": "task_07_qed_molecular_design",
            "status": status,
            "score": score,
            "protocol": "chembench-hidden-suite/1.0",
        },
    )
    _write(
        WORK / "final_answer.json",
        {"task_id": "task_07_qed_molecular_design", "case_count": len(paths), "score": score},
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    reward_path = LOGS / "reward.txt"
    if score is not None and not failures:
        reward = max(0.0, min(1.0, score / 100.0))
        reward_path.write_text(str(reward), encoding="utf-8")
    else:
        # BLOCKED and protocol/evidence failures are not model scores.  Removing
        # a stale reward is deliberate: Harbor must surface missing reward and
        # use grader-result.json for the infrastructure/protocol attribution.
        reward_path.unlink(missing_ok=True)
    _write(
        LOGS / "grader-result.json",
        {"score": score, "status": status, "failure_codes": failures, "suite": suite},
    )
    print(json.dumps({"score": score, "status": status, "failure_codes": failures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

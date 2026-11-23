#!/usr/bin/env python3
"""Fail-closed independent replay of Task 09 organizer evidence."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

WORK = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))
EVIDENCE = Path(os.environ.get("CHEMBENCH_ORGANIZER_EVIDENCE", "/organizer_evidence"))
RUNTIME = Path("/task/private_data/runtime_package")
if not RUNTIME.is_dir():
    RUNTIME = Path(__file__).resolve().parent / "task_snapshot" / "private_data" / "runtime_package"

sys.path.insert(0, str(RUNTIME))
from evaluator.evaluate_multicase_run_v3 import evaluate_case
from evaluator.evaluate_suite import aggregate_suite


# 2026-09-27 fix: task09 previously trusted whatever model identity the
# evidence metadata claimed; tasks 07/08 already gate on a frozen allowlist.
# Ported from tasks/task07|08/tests/protocol_grader.py: the frozen siliconflow
# identities plus the three organizer antchat-gateway model ids
# (ling-3.0-flash / GLM-5.3 / Kimi-K3, registered 2026-09-16).
FROZEN_MODEL_IDS = {
    "deepseek-ai/DeepSeek-V4-Pro",
    "Pro/moonshotai/Kimi-K2.6",
    "zai-org/GLM-5.2",
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
# Organizer-only lifecycle validation; submission-facing identity gates stay.
if os.environ.get("CHEMBENCH_ORACLE_VALIDATION") == "1":
    FROZEN_MODEL_IDS = FROZEN_MODEL_IDS | {"organizer/scripted-oracle-v1"}


PROTOCOL_FAILURE_GATES = {
    "exact_model_identity",
    "trajectory_integrity",
    "artifact_provenance",
    "replay_consistency",
    "scientific_invariants",
}


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


def _root(evidence_id: Any, case_id: str) -> tuple[Path, str]:
    if not isinstance(evidence_id, str) or "\\" in evidence_id:
        raise ValueError("EVIDENCE_ID_INVALID")
    parts = PurePosixPath(evidence_id).parts
    if len(parts) != 2 or parts[1] != case_id or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("EVIDENCE_ID_INVALID")
    root = (EVIDENCE / parts[0] / parts[1]).resolve()
    if EVIDENCE.resolve() not in root.parents:
        raise ValueError("EVIDENCE_PATH_ESCAPE")
    return root, parts[0]


def main() -> int:
    suite = _object(RUNTIME / "organizer_private" / "suite.json")
    cases = suite["cases"]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    tokens: set[str] = set()

    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        try:
            artifact = _object(WORK / "cases" / f"case-{index}" / "case_result.json")
            submission = artifact.get("submission")
            if not isinstance(submission, dict) or submission.get("ok") is not True:
                raise ValueError("SUBMISSION_REFERENCE_INVALID")
            case_root, token = _root(submission.get("evidence_id"), case_id)
            tokens.add(token)
            metadata = _object(case_root / "metadata.json")
            if metadata.get("session_id") != token or metadata.get("case_id") != case_id:
                raise ValueError("EVIDENCE_METADATA_MISMATCH")
            # 2026-09-27 fix: ported EXACT_MODEL_IDENTITY gate from tasks 07/08.
            # The requested model must be allowlisted and the evidence must self
            # attest an exact requested==actual identity.
            requested_model = metadata.get("requested_model_id")
            if (
                metadata.get("actual_model_id") != requested_model
                or metadata.get("exact_model_identity") is not True
            ):
                raise ValueError("EXACT_MODEL_IDENTITY_FAILED")
            rows.append(evaluate_case(case_root=case_root, case=case, project_root=RUNTIME))
        except Exception as exc:
            code = f"CASE_{index}_{str(exc) or type(exc).__name__}"
            failures.append(code)
            rows.append(
                {
                    "schema_version": "3.0",
                    "case_id": case_id,
                    "status": "FAIL",
                    "provider_status": "EVIDENCE_ERROR",
                    "eligible": False,
                    "failed_gates": [code],
                    "scientific_score": 0.0,
                    "process_score": 0.0,
                    "official_score": 0.0,
                }
            )

    if len(tokens) > 1:
        failures.append("MIXED_SESSION_EVIDENCE")

    aggregation = aggregate_suite(rows, [case["case_id"] for case in cases])
    coverage = aggregation["coverage"]
    if failures:
        score: float | None = 0.0
        status = "FAIL"
    elif coverage["blocked"]:
        score = None
        status = "BLOCKED"
    else:
        score = round(float(aggregation["macro_official_score_excluding_blocked"] or 0.0), 4)
        status = "FAIL" if coverage["fail"] else "PASS"

    campaign = {
        "schema_version": "task09-harbor-campaign/1.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/3.0",
        "case_count": len(cases),
        "session_count": len(tokens),
        "rows": rows,
        "all_replays_match": not failures,
        "failure_codes": failures,
    }
    _write(WORK / "benchmark_run" / "protocol" / "suite_evaluation.json", aggregation)
    _write(WORK / "benchmark_run" / "suite_evaluation.json", aggregation)
    _write(WORK / "benchmark_run" / "campaign_summary.json", campaign)
    _write(
        WORK / "benchmark_result.json",
        {
            "task_id": "adaptive_xtb_torsion_profile_agent_v3",
            "status": status,
            "score": score,
            "protocol": "adaptive_torsion_sampling_tool_loop/3.0",
        },
    )
    _write(
        WORK / "final_answer.json",
        {"task_id": "adaptive_xtb_torsion_profile_agent_v3", "case_count": len(cases), "score": score},
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    protocol_gate_failure = any(
        PROTOCOL_FAILURE_GATES.intersection(row.get("failed_gates", []))
        for row in rows
    )
    reward_path = LOGS / "reward.txt"
    if score is not None and not failures and not protocol_gate_failure:
        reward = max(0.0, min(1.0, score / 100.0))
        reward_path.write_text(str(reward), encoding="utf-8")
    else:
        # BLOCKED and protocol/evidence failures are not model scores.  Removing
        # a stale reward is deliberate: Harbor must preserve their attribution.
        reward_path.unlink(missing_ok=True)
    _write(
        LOGS / "grader-result.json",
        {
            "score": score,
            "status": status,
            "failure_codes": failures,
            "case_results": rows,
            "aggregation": aggregation,
        },
    )
    print(json.dumps({"score": score, "status": status, "failure_codes": failures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

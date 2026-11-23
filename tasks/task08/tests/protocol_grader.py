#!/usr/bin/env python3
"""Fail-closed replay of Task 08 organizer evidence bundles."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
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
from evaluator.evaluate import evaluate_files

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
# Organizer-only lifecycle validation; normal model identity gates stay enabled.
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


def _classify_submission_failure(artifact: dict[str, Any], submission: Any) -> str:
    """Root-cause code for a submission reference that cannot be graded.

    # 2026-09-27 fix: SUBMISSION_REFERENCE_INVALID was a single blanket code that
    # could not distinguish "the model never submitted" from "the sidecar
    # transport broke" from "the evaluator examined and rejected the candidate".
    # Classify from case_result.json fields (failure / submission.feedback /
    # submission.error / submission.status); the score semantics are unchanged,
    # only failure_codes can now name the root cause.
    """
    if isinstance(submission, dict):
        feedback = submission.get("feedback")
        if isinstance(feedback, dict) and (
            feedback.get("accepted") is False
            or submission.get("status") in {"invalid", "rejected"}
        ):
            return "SUBMISSION_REJECTED_BY_EVALUATOR"
        if "feedback" not in submission and submission.get("error") is not None:
            return "SIDECAR_INFRASTRUCTURE"
    if str(artifact.get("failure") or "") == "sidecar_infrastructure_failure":
        return "SIDECAR_INFRASTRUCTURE"
    return "NO_SUBMISSION"


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
    cases = [suite["public_case"], *suite["hidden_cases"]]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    tokens: set[str] = set()

    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        try:
            artifact = _object(WORK / "cases" / f"case-{index}" / "case_result.json")
            submission = artifact.get("submission")
            if not isinstance(submission, dict) or submission.get("ok") is not True:
                # 2026-09-27 fix: emit the three-way root-cause code instead of the
                # blanket SUBMISSION_REFERENCE_INVALID; downstream score handling
                # is untouched (any of these still fails the case with score 0).
                raise ValueError(_classify_submission_failure(artifact, submission))
            root, token = _root(submission.get("evidence_id"), case_id)
            tokens.add(token)
            metadata_path = root / "metadata.json"
            metadata = _object(metadata_path)
            if (
                metadata.get("schema_version") != "task08-organizer-evidence/1.0"
                or metadata.get("session_id") != token
                or metadata.get("case_id") != case_id
            ):
                raise ValueError("EVIDENCE_METADATA_MISMATCH")
            requested_model = metadata.get("requested_model_id")
            if (
                metadata.get("actual_model_id") != requested_model
                or metadata.get("exact_model_identity") is not True
            ):
                raise ValueError("EXACT_MODEL_IDENTITY_FAILED")
            provider_status = str(metadata.get("provider_status", "HARBOR_PROTOCOL"))
            if provider_status.startswith("BLOCKED"):
                rows.append(
                    {
                        "case_id": case_id,
                        "status": "BLOCKED",
                        "provider_status": provider_status,
                        "official_score": None,
                        "eligible": None,
                        "included_in_numeric_aggregate": False,
                    }
                )
                continue
            result = evaluate_files(
                final_answer_path=root / "final_answer.json",
                trajectory_path=root / "trajectory.jsonl",
                artifacts_path=root / "workspace",
                config_path=RUNTIME / case["config"],
            )
            raw_score = result.get("official_score")
            if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
                raise ValueError("OFFICIAL_SCORE_INVALID")
            score = float(raw_score)
            eligible = result.get("eligibility", {}).get("eligible") is True
            rows.append(
                {
                    "case_id": case_id,
                    "status": "PASS" if eligible else "FAIL",
                    "provider_status": provider_status,
                    "official_score": score,
                    "eligible": eligible,
                    "included_in_numeric_aggregate": True,
                    "evaluation": result,
                }
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            code = f"CASE_{index}_{str(exc) or type(exc).__name__}"
            failures.append(code)
            rows.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "provider_status": "EVIDENCE_ERROR",
                    "official_score": 0.0,
                    "eligible": False,
                    "included_in_numeric_aggregate": True,
                    "failure_code": code,
                }
            )

    if len(tokens) > 1:
        failures.append("MIXED_SESSION_EVIDENCE")

    blocked = sum(row["status"] == "BLOCKED" for row in rows)
    passed = sum(row["status"] == "PASS" for row in rows)
    failed = sum(row["status"] == "FAIL" for row in rows)
    numeric = [Decimal(str(row["official_score"])) for row in rows if row["included_in_numeric_aggregate"]]
    mean = (
        (sum(numeric) / Decimal(len(numeric))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if numeric
        else None
    )
    score: float | None
    if failures:
        score = 0.0
    elif blocked:
        score = None
    else:
        score = float(mean) if mean is not None else 0.0
    status = "FAIL" if failures else ("BLOCKED" if blocked else ("FAIL" if failed else "PASS"))
    aggregation = {
        "schema_version": "2.0",
        "suite_case_count": len(rows),
        "coverage": {"pass": passed, "fail": failed, "blocked": blocked, "scored": len(numeric)},
        "macro_official_score_excluding_blocked": float(mean) if mean is not None else None,
        "ranking_eligible": bool(rows and blocked == 0 and not failures),
        "blocked_policy": "excluded_from_numeric_aggregate_and_disqualifies_ranking_coverage",
        "cases": rows,
    }
    campaign = {
        "schema_version": "task08-harbor-campaign/1.0",
        "protocol": "chembench-hidden-suite/1.0",
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
            "task_id": "task_08_xtb_failure_diagnosis_restart",
            "status": status,
            "score": score,
            "protocol": "chembench-hidden-suite/1.0",
        },
    )
    _write(
        WORK / "final_answer.json",
        {"task_id": "task_08_xtb_failure_diagnosis_restart", "case_count": len(cases), "score": score},
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    reward_path = LOGS / "reward.txt"
    if score is not None and not failures:
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

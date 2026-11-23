from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_ID = "task_09_adaptive_xtb_torsion_profile"
MODEL_MAP = {
    "deepseek-v4-pro": ("deepseek", "deepseek-ai/DeepSeek-V4-Pro"),
    "kimi-k2.6": ("kimi", "Pro/moonshotai/Kimi-K2.6"),
    "GLM-5.2": ("glm", "zai-org/GLM-5.2"),
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return (
        isinstance(left, (int, float)) and not isinstance(left, bool)
        and isinstance(right, (int, float)) and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)
    )


def _fail(code: str, detail: str) -> dict[str, Any]:
    return {"score": 0.0, "subscores": {"independent_replay": 0.0}, "failure_codes": [code], "evidence": {"detail": detail}}


def grade(context: Any) -> dict[str, Any]:
    submission = Path(context.submission_dir).resolve()
    runtime = Path(__file__).resolve().parents[1] / "private_data" / "runtime_package"
    try:
        result = _json(submission / "benchmark_result.json")
        final = _json(submission / "final_answer.json")
        if result.get("task_id") != TASK_ID or final.get("task_id") != TASK_ID:
            return _fail("TASK_ID_MISMATCH", "task identity is inconsistent")
        alias = result.get("model_name")
        if alias not in MODEL_MAP or final.get("model_name") != alias:
            return _fail("MODEL_ALIAS_INVALID", "model alias is missing or inconsistent")
        logical, expected_id = MODEL_MAP[alias]
        runs_root = (submission / "benchmark_run" / logical).resolve()
        if submission not in runs_root.parents or not runs_root.is_dir():
            return _fail("RUN_ROOT_INVALID", "model run directory is missing")

        for name in tuple(sys.modules):
            if name == "evaluator" or name.startswith("evaluator."):
                del sys.modules[name]
        sys.path.insert(0, str(runtime))
        try:
            from evaluator.evaluate_multicase_run_v3 import evaluate_case
            from evaluator.evaluate_suite import aggregate_suite
        finally:
            sys.path.pop(0)

        suite = _json(runtime / "organizer_private" / "suite.json")
        cases = suite.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("private suite has no cases")
        records: list[dict[str, Any]] = []
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("invalid private case")
            case_id = str(case["case_id"])
            case_root = runs_root / case_id
            replay = evaluate_case(case_root, case, project_root=runtime)
            if replay != _json(case_root / "evaluation.json"):
                return _fail("CASE_REPLAY_MISMATCH", case_id)
            records.append(replay)
        replay_suite = aggregate_suite(records, [str(case["case_id"]) for case in cases])
        if replay_suite != _json(runs_root / "suite_evaluation.json"):
            return _fail("SUITE_REPLAY_MISMATCH", "suite aggregate differs from private replay")

        summary = _json(submission / "benchmark_run" / "campaign_summary.json")
        all_rows = summary.get("rows")
        if not isinstance(all_rows, list):
            return _fail("CAMPAIGN_SUMMARY_INVALID", "campaign rows are missing")
        rows = [row for row in all_rows if isinstance(row, dict) and row.get("logical_model") == logical]
        if not rows:
            return _fail("CAMPAIGN_SUMMARY_INVALID", "declared model has no campaign rows")
        for row in rows:
            if row.get("requested_model_id") != expected_id:
                return _fail("REQUESTED_MODEL_MISMATCH", str(row.get("case_id", "unknown")))
            if row.get("run_status") != "BLOCKED" and (
                row.get("actual_model_id") != expected_id or row.get("exact_model_identity") is not True
            ):
                return _fail("EXACT_MODEL_IDENTITY_FAILED", str(row.get("case_id", "unknown")))
        coverage = replay_suite["coverage"]
        blocked = int(coverage.get("blocked", 0))
        expected_status = "BLOCKED" if blocked else ("FAIL" if int(coverage.get("fail", 0)) else "PASS")
        expected_score = None if blocked else replay_suite.get("macro_official_score_excluding_blocked")
        if result.get("status") != expected_status or not _same_number(result.get("score"), expected_score):
            return _fail("DECLARED_RESULT_MISMATCH", "declared status or score differs from replay")
        evidence = {"case_count": len(cases), "independently_replayed_cases": len(records), "blocked_cases": blocked, "exact_provider_model_id": expected_id, "suite_score": expected_score}
        if expected_status == "BLOCKED":
            return {"status": "BLOCKED", "score": None, "subscores": {}, "failure_codes": ["EXTERNAL_RUN_BLOCKED"], "evidence": evidence}
        numeric = float(expected_score or 0.0)
        return {"score": numeric, "subscores": {"private_suite_macro": numeric}, "failure_codes": [] if expected_status == "PASS" else ["SCIENTIFIC_CASE_FAILURE"], "evidence": evidence}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return _fail("PRIVATE_REPLAY_ERROR", type(exc).__name__)

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_ID = "task_07_qed_molecular_design"
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
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)
    )


def _fail(code: str, detail: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "subscores": {"independent_replay": 0.0},
        "failure_codes": [code],
        "evidence": {"detail": detail},
    }


def grade(context: Any) -> dict[str, Any]:
    submission = Path(context.submission_dir).resolve()
    task_dir = Path(__file__).resolve().parents[1]
    runtime = task_dir / "private_data" / "runtime_package"
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
            from evaluator.evaluate import evaluate_submission
            from evaluator.evaluate_suite import aggregate_suite, load_suite
        finally:
            sys.path.pop(0)

        suite_path = runtime / "organizer_private" / "suite.yaml"
        _, case_paths = load_suite(suite_path)
        replay_count = 0
        for case_path in case_paths:
            import yaml

            case = yaml.safe_load(case_path.read_text(encoding="utf-8"))
            case_id = str(case["case"]["id"])
            case_root = runs_root / case_id
            status = _json(case_root / "status.json")
            if status.get("run_status") == "BLOCKED":
                continue
            stored = _json(case_root / "evaluation.json")
            replay = evaluate_submission(
                case_root / "final_answer.json",
                case_path,
                trajectory_path=case_root / "trajectory.jsonl",
            )
            if replay != stored:
                return _fail("CASE_REPLAY_MISMATCH", case_id)
            replay_count += 1

        replay_suite = aggregate_suite(suite_path, runs_root)
        stored_suite = _json(runs_root / "suite_evaluation.json")
        if replay_suite != stored_suite:
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

        blocked = int(replay_suite.get("blocked_count", 0))
        expected_status = "BLOCKED" if blocked else (
            "FAIL" if any(row.get("run_status") == "FAIL" for row in rows) else "PASS"
        )
        expected_score = None if blocked else replay_suite.get("capability_score")
        if result.get("status") != expected_status or not _same_number(result.get("score"), expected_score):
            return _fail("DECLARED_RESULT_MISMATCH", "declared status or score differs from replay")
        evidence = {
            "case_count": len(case_paths),
            "independently_replayed_cases": replay_count,
            "blocked_cases": blocked,
            "exact_provider_model_id": expected_id,
            "suite_score": expected_score,
        }
        if expected_status == "BLOCKED":
            return {
                "status": "BLOCKED",
                "score": None,
                "subscores": {},
                "failure_codes": ["EXTERNAL_RUN_BLOCKED"],
                "evidence": evidence,
            }
        numeric = float(expected_score or 0.0)
        return {
            "score": numeric,
            "subscores": {"private_suite_capability": numeric},
            "failure_codes": [] if expected_status == "PASS" else ["SCIENTIFIC_CASE_FAILURE"],
            "evidence": evidence,
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return _fail("PRIVATE_REPLAY_ERROR", type(exc).__name__)

"""Aggregate deterministic per-case evaluations with explicit BLOCKED state."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate import evaluate_files
from evaluator.trusted_validation import canonical_json_bytes


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("SUITE_NOT_OBJECT")
    return value


def _cases(suite: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(suite.get("cases"), list):
        return [item for item in suite["cases"] if isinstance(item, Mapping)]
    values: list[Mapping[str, Any]] = []
    if isinstance(suite.get("public_case"), Mapping):
        values.append(suite["public_case"])
    if isinstance(suite.get("hidden_cases"), list):
        values.extend(item for item in suite["hidden_cases"] if isinstance(item, Mapping))
    return values


def evaluate_suite(suite_path: Path, runs_root: Path) -> dict[str, Any]:
    suite = _load(suite_path)
    records: list[dict[str, Any]] = []
    scores: list[Decimal] = []
    for case in _cases(suite):
        case_id = str(case.get("case_id", ""))
        run = runs_root / case_id
        status_path = run / "status.json"
        if status_path.is_file():
            try:
                status = _load(status_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                records.append({
                    "case_id": case_id,
                    "status": "BLOCKED",
                    "blocked_reason": "STATUS_INPUT_ERROR",
                    "included_in_numeric_aggregate": False,
                })
                continue
            if status.get("run_status") == "BLOCKED":
                records.append({
                    "case_id": case_id,
                    "status": "BLOCKED",
                    "blocked_reason": "PROVIDER_UNAVAILABLE_AFTER_RETRIES",
                    "included_in_numeric_aggregate": False,
                })
                continue
        required = {
            "final_answer": run / "final_answer.json",
            "trajectory": run / "trajectory.jsonl",
            "workspace": run / "workspace",
        }
        missing = sorted(name for name, path in required.items() if not path.exists())
        if missing:
            records.append({
                "case_id": case_id,
                "status": "BLOCKED",
                "blocked_reason": "MISSING_RUN_ARTIFACTS",
                "missing": missing,
                "included_in_numeric_aggregate": False,
            })
            continue
        try:
            evaluation = evaluate_files(
                final_answer_path=required["final_answer"],
                trajectory_path=required["trajectory"],
                artifacts_path=required["workspace"],
                config_path=ROOT / str(case["config"]),
            )
        except (OSError, ValueError, TypeError, KeyError):
            records.append({
                "case_id": case_id,
                "status": "BLOCKED",
                "blocked_reason": "EVALUATOR_INPUT_ERROR",
                "included_in_numeric_aggregate": False,
            })
            continue
        score = Decimal(str(evaluation.get("official_score", 0)))
        passed = evaluation.get("eligibility", {}).get("eligible") is True
        records.append({
            "case_id": case_id,
            "status": "PASS" if passed else "FAIL",
            "official_score": int(score) if score == score.to_integral_value() else float(score),
            "raw_score_before_scientific_cap": evaluation.get("raw_score_before_scientific_cap"),
            "included_in_numeric_aggregate": True,
            "failed_gates": evaluation.get("eligibility", {}).get("failed_gates", []),
        })
        scores.append(score)
    blocked = sum(item["status"] == "BLOCKED" for item in records)
    passed = sum(item["status"] == "PASS" for item in records)
    failed = sum(item["status"] == "FAIL" for item in records)
    mean = (
        (sum(scores) / Decimal(len(scores))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        if scores else None
    )
    return {
        "schema_version": "2.0",
        "suite_case_count": len(records),
        "coverage": {
            "pass": passed,
            "fail": failed,
            "blocked": blocked,
            "scored": len(scores),
        },
        "macro_official_score_excluding_blocked": (
            int(mean) if mean is not None and mean == mean.to_integral_value()
            else float(mean) if mean is not None else None
        ),
        "ranking_eligible": bool(records and blocked == 0),
        "blocked_policy": "excluded_from_numeric_aggregate_and_disqualifies_ranking_coverage",
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_suite(args.suite, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result, newline=True, indent=2))


if __name__ == "__main__":
    main()

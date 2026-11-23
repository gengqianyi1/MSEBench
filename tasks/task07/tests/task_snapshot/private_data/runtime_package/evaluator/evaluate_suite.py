"""Aggregate independently evaluated QED cases without treating BLOCKED as zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SuiteEvaluationError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SuiteEvaluationError(f"{path} must contain a YAML object")
    return value


def load_suite(path: Path) -> tuple[dict[str, Any], list[Path]]:
    suite = _load_yaml(path)
    raw_cases = suite.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteEvaluationError("organizer suite must list at least one case")
    cases: list[Path] = []
    for raw in raw_cases:
        if not isinstance(raw, str) or not raw:
            raise SuiteEvaluationError("suite case paths must be non-empty strings")
        case_path = (PROJECT_ROOT / raw).resolve()
        if PROJECT_ROOT.resolve() not in case_path.parents:
            raise SuiteEvaluationError("suite case path escapes the project root")
        if not case_path.is_file():
            raise SuiteEvaluationError(f"suite case is missing: {raw}")
        cases.append(case_path)
    return suite, cases


def aggregate_suite(suite_path: Path, runs_root: Path) -> dict[str, Any]:
    suite, case_paths = load_suite(suite_path)
    rows: list[dict[str, Any]] = []
    scored: list[float] = []
    blocked_count = 0
    missing_count = 0
    seen_ids: set[str] = set()

    for case_path in case_paths:
        case_cfg = _load_yaml(case_path)
        case_meta = case_cfg.get("case")
        if not isinstance(case_meta, dict) or not isinstance(case_meta.get("id"), str):
            raise SuiteEvaluationError(f"case metadata is invalid: {case_path}")
        case_id = case_meta["id"]
        if case_id in seen_ids:
            raise SuiteEvaluationError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        case_dir = runs_root / case_id
        status_path = case_dir / "status.json"
        evaluation_path = case_dir / "evaluation.json"
        provider_status = "NOT_RECORDED"
        declared_status = None
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise SuiteEvaluationError(f"invalid status record for {case_id}")
            declared_status = status.get("run_status")
            provider_status = str(status.get("provider_status", "NOT_RECORDED"))
        if declared_status == "BLOCKED":
            blocked_count += 1
            rows.append(
                {
                    "case_id": case_id,
                    "visibility": case_meta.get("visibility"),
                    "provider_status": provider_status,
                    "run_status": "BLOCKED",
                    "eligibility": None,
                    "scientific_score": None,
                    "process_score": None,
                    "official_case_score": None,
                    "included_in_capability_mean": False,
                }
            )
            continue
        if not evaluation_path.is_file():
            missing_count += 1
            rows.append(
                {
                    "case_id": case_id,
                    "visibility": case_meta.get("visibility"),
                    "provider_status": provider_status,
                    "run_status": "MISSING",
                    "eligibility": None,
                    "scientific_score": None,
                    "process_score": None,
                    "official_case_score": None,
                    "included_in_capability_mean": False,
                }
            )
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        score = evaluation.get("score")
        if isinstance(score, dict):
            raw_total = score.get("total_score")
            eligible = score.get("gate_passed") is True
            components = score.get("components", {})
        else:
            raw_total = evaluation.get("total_score", score)
            eligible = evaluation.get("eligible") is True
            components = evaluation.get("score_breakdown", {})
        if not isinstance(raw_total, (int, float)):
            raise SuiteEvaluationError(f"invalid evaluation score for {case_id}")
        total = float(raw_total)
        run_status = "PASS" if eligible else "FAIL"
        # v4 uses process checks only as an eligibility gate; process behavior
        # does not carry points.  Report no process score instead of a
        # misleading, apparently unattainable numeric zero.
        process_score = components.get("process") if isinstance(components, dict) else None
        scored.append(total)
        rows.append(
            {
                "case_id": case_id,
                "visibility": case_meta.get("visibility"),
                "provider_status": provider_status,
                "run_status": run_status,
                "eligibility": eligible,
                "scientific_score": total,
                "process_score": process_score,
                "official_case_score": total,
                "included_in_capability_mean": True,
            }
        )

    macro = round(mean(scored), 4) if scored else None
    stability = None
    capability = None
    if scored:
        stability = round(10.0 * min(scored) / max(mean(scored), 1.0), 4)
        capability = round(min(100.0, float(macro) + stability), 4)
    all_terminal = missing_count == 0
    ranking_eligible = all_terminal and blocked_count == 0
    return {
        "schema_version": "qed_suite_evaluation/4.0",
        "benchmark_id": suite.get("benchmark_id"),
        "case_count": len(case_paths),
        "case_results": rows,
        "capability_macro_average_excluding_blocked": macro,
        "cross_case_stability_score": stability,
        "capability_score": capability,
        "blocked_count": blocked_count,
        "missing_count": missing_count,
        "suite_execution_complete": all_terminal,
        "ranking_coverage_eligible": ranking_eligible,
        # Compatibility aliases retained for consumers of schema 4.0.  These
        # mean complete terminal coverage, not scientific success or approval
        # to publish this task.
        "release_complete": all_terminal,
        "ranking_eligible": ranking_eligible,
        "coverage_field_semantics": {
            "release_complete": "all configured cases reached a terminal record; this is not task release approval",
            "ranking_eligible": "all cases are terminal and none is BLOCKED; FAIL cases may still be present",
        },
        "blocked_policy": "BLOCKED is excluded from capability means and prevents ranking eligibility.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=PROJECT_ROOT / "organizer_private" / "suite.yaml"
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate_suite(args.suite, args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

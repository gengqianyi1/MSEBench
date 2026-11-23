"""Aggregate per-case v3 results with explicit PASS/FAIL/BLOCKED semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable


VALID_STATUSES = {"PASS", "FAIL", "BLOCKED"}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 100.0:
        raise ValueError(f"{label} out of range")
    return number


def aggregate_suite(cases: Iterable[dict[str, Any]], expected_case_ids: list[str]) -> dict[str, Any]:
    records = list(cases)
    identifiers = [str(record.get("case_id", "")) for record in records]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(expected_case_ids):
        raise ValueError("suite case coverage mismatch or duplicate case ID")
    normalized = []
    numeric = []
    for record in records:
        status = record.get("status")
        if status not in VALID_STATUSES:
            raise ValueError("invalid case status")
        provider_status = record.get("provider_status")
        if not isinstance(provider_status, str) or not provider_status:
            raise ValueError("provider status is required")
        included = status != "BLOCKED"
        if status == "BLOCKED":
            if any(record.get(name) is not None for name in ("official_score", "scientific_score", "process_score")):
                raise ValueError("BLOCKED case must have null numeric scores")
            official = scientific = process = None
        else:
            official = _finite_number(record.get("official_score"), "official_score")
            scientific = _finite_number(record.get("scientific_score"), "scientific_score")
            process = _finite_number(record.get("process_score"), "process_score")
            if scientific > 80.0 or process > 20.0:
                raise ValueError("scientific/process component exceeds its maximum")
            numeric.append((official, scientific, process))
        normalized.append(
            {
                "case_id": record["case_id"],
                "status": status,
                "provider_status": provider_status,
                "eligible": record.get("eligible") if status != "BLOCKED" else None,
                "failed_gates": record.get("failed_gates", []) if status != "BLOCKED" else [],
                "scientific_score": scientific,
                "scientific_max_score": 80,
                "process_score": process,
                "process_max_score": 20,
                "official_score": official,
                "included_in_numeric_aggregate": included,
            }
        )
    mean = lambda index: round(sum(row[index] for row in numeric) / len(numeric), 12) if numeric else None
    coverage = {name.lower(): sum(row["status"] == name for row in normalized) for name in VALID_STATUSES}
    return {
        "schema_version": "3.0",
        "blocked_policy": "excluded_from_numeric_aggregate_and_disqualifies_ranking_coverage",
        "suite_case_count": len(expected_case_ids),
        "coverage": {**coverage, "scored": len(numeric)},
        "macro_official_score_excluding_blocked": mean(0),
        "macro_scientific_score_excluding_blocked": mean(1),
        "macro_process_score_excluding_blocked": mean(2),
        "ranking_eligible": coverage["blocked"] == 0 and len(normalized) == len(expected_case_ids),
        "cases": normalized,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    expected = [str(item["case_id"]) for item in suite["cases"]]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.evaluations.glob("*.json"))]
    result = aggregate_suite(records, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())

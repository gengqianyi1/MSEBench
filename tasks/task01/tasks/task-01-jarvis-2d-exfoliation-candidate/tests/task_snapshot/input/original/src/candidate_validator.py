from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from jsonschema import Draft202012Validator


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: List[Dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
    return records


def _schema_errors(records: Iterable[Dict[str, Any]], schema_path: Path, limit: int = 25) -> List[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors: List[str] = []
    for index, record in enumerate(records):
        for err in validator.iter_errors(record):
            location = "/".join(str(x) for x in err.absolute_path) or "<record>"
            errors.append(f"record {index}: {location}: {err.message}")
            if len(errors) >= limit:
                return errors
    return errors


def validate_candidate_library(
    candidate_path: Path,
    record_schema_path: Path,
    required_count: int = 1000,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    records: List[Dict[str, Any]] = []

    if not candidate_path.exists():
        return {
            "passed": False,
            "errors": [f"candidate library not found: {candidate_path}"],
            "warnings": [],
            "record_count": 0,
            "schema_error_count": 0,
            "duplicate_candidate_id_count": 0,
        }

    try:
        records = read_jsonl(candidate_path)
    except Exception as exc:
        return {
            "passed": False,
            "errors": [f"candidate library is unreadable: {exc}"],
            "warnings": [],
            "record_count": 0,
            "schema_error_count": 0,
            "duplicate_candidate_id_count": 0,
        }

    if len(records) > required_count:
        errors.append(f"candidate library must contain at most {required_count} records; found {len(records)}")
    elif len(records) < required_count:
        warnings.append(
            f"candidate library is short ({len(records)}/{required_count}); it remains scoreable with the fixed {required_count}-record denominator"
        )

    schema_errors = _schema_errors(records, record_schema_path)
    if schema_errors:
        errors.append(f"candidate record schema validation failed for at least {len(schema_errors)} issue(s)")
        warnings.extend(schema_errors)

    ids = [str(record.get("candidate_id", "")) for record in records]
    counts = Counter(ids)
    duplicate_count = sum(count for candidate_id, count in counts.items() if candidate_id and count > 1)
    if duplicate_count:
        warnings.append(f"{duplicate_count} records use duplicated candidate_id values; they receive no validity credit")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "record_count": len(records),
        "schema_error_count": len(schema_errors),
        "duplicate_candidate_id_count": duplicate_count,
    }

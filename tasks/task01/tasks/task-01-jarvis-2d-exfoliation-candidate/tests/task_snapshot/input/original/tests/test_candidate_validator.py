import gzip
import json
from pathlib import Path

from src.candidate_validator import validate_candidate_library


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "tasks/jarvis_2d_exfoliation/candidate_record_schema.json"


def _record(index: int):
    return {
        "candidate_id": f"CAND-{index:06d}",
        "generation_index": index,
        "random_seed": 42,
        "predicted_exfoliation_energy_mev_atom": float(index),
        "prediction_uncertainty_mev_atom": 10.0,
        "applicability_domain_score": 0.5,
        "prediction_model_sha256": "0" * 64,
        "structure": {"placeholder": index},
        "provenance": {
            "strategy": "unit-test",
            "parent_ids": [],
            "operations": [{"name": "unit_test", "parameters": {"index": index}}],
        },
    }


def _write(path: Path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_candidate_validator_accepts_1000_schema_valid_records(tmp_path):
    path = tmp_path / "candidate_library.json.gz"
    _write(path, [_record(i) for i in range(1000)])
    report = validate_candidate_library(path, SCHEMA, required_count=1000)
    assert report["passed"], report
    assert report["record_count"] == 1000
    assert report["schema_error_count"] == 0


def test_candidate_validator_scores_short_library_with_warning(tmp_path):
    path = tmp_path / "candidate_library.json.gz"
    _write(path, [_record(i) for i in range(999)])
    report = validate_candidate_library(path, SCHEMA, required_count=1000)
    assert report["passed"]
    assert any("fixed 1000-record denominator" in warning for warning in report["warnings"])
    assert report["errors"] == []


def test_duplicate_candidate_ids_are_reported_without_making_file_unreadable(tmp_path):
    path = tmp_path / "candidate_library.json.gz"
    records = [_record(i) for i in range(1000)]
    records[-1]["candidate_id"] = records[0]["candidate_id"]
    _write(path, records)
    report = validate_candidate_library(path, SCHEMA, required_count=1000)
    assert report["passed"], report
    assert report["duplicate_candidate_id_count"] == 2

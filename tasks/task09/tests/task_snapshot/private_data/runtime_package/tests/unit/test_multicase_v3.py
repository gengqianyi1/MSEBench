from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluator.evaluate_suite import aggregate_suite


ROOT = Path(__file__).resolve().parents[2]


def _record(case_id: str, status: str = "PASS") -> dict:
    if status == "BLOCKED":
        return {
            "case_id": case_id,
            "status": status,
            "provider_status": "AUTHENTICATION_FAILED",
            "official_score": None,
            "scientific_score": None,
            "process_score": None,
        }
    return {
        "case_id": case_id,
        "status": status,
        "provider_status": "NOT_APPLICABLE_TEST",
        "eligible": status == "PASS",
        "failed_gates": [] if status == "PASS" else ["complete_protocol"],
        "official_score": 75.0 if status == "PASS" else 0.0,
        "scientific_score": 60.0,
        "process_score": 15.0,
    }


def test_blocked_is_excluded_and_disqualifies_ranking() -> None:
    result = aggregate_suite([_record("a"), _record("b", "BLOCKED")], ["a", "b"])
    assert result["macro_official_score_excluding_blocked"] == 75.0
    assert result["coverage"] == {"pass": 1, "fail": 0, "blocked": 1, "scored": 1}
    assert result["ranking_eligible"] is False


def test_blocked_cannot_carry_a_fake_numeric_score() -> None:
    record = _record("a", "BLOCKED")
    record["official_score"] = 0.0
    with pytest.raises(ValueError, match="BLOCKED case must have null"):
        aggregate_suite([record], ["a"])


def test_duplicate_or_missing_case_coverage_is_rejected() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        aggregate_suite([_record("a"), _record("a")], ["a", "b"])


def test_native_hidden_cases_are_complete_hash_bound_and_path_redacted() -> None:
    suite = json.loads((ROOT / "organizer_private" / "suite.json").read_text(encoding="utf-8"))
    hidden = [case for case in suite["cases"] if case["visibility"] == "hidden"]
    assert len(hidden) == 3
    for case in hidden:
        case_root = ROOT / case["case_root"]
        oracle = json.loads((case_root / "oracle_profile.json").read_text(encoding="utf-8"))
        assert oracle["grid_angles_deg"] == list(range(0, 360, 5))
        assert len(oracle["relative_energies_kcal_mol"]) == 72
        provenance = json.loads((case_root / "oracle_provenance.json").read_text(encoding="utf-8"))
        assert provenance["native_attempt_count"] == 72
        assert provenance["all_angles_eligible"] is True
        for angle in range(0, 360, 5):
            attempt = case_root / "native_attempts" / f"angle_{angle:03d}"
            result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
            assert result["profile_eligible"] is True
            assert result["stdout_projection"] == "native-program-call-path-redaction/1.0"
            assert b"C:\\Users\\" not in (attempt / "stdout.log").read_bytes()
            manifest = json.loads((attempt / "attempt_manifest.json").read_text(encoding="utf-8"))
            for item in manifest["files"]:
                member = attempt / item["path"]
                assert hashlib.sha256(member.read_bytes()).hexdigest().upper() == item["sha256"]


def test_morphology_suite_has_four_distinct_native_profiles() -> None:
    value = json.loads((ROOT / "organizer_private" / "morphology_validation.json").read_text(encoding="utf-8"))
    assert value["case_count"] == 4
    assert value["distinct_molecules"] == 4
    assert value["all_profiles_have_72_native_grid_points"] is True
    ranges = list(value["energy_ranges_kcal_mol"].values())
    assert len({round(float(item), 2) for item in ranges}) == 4
    extrema = {
        (len(item["local_minima_angles_deg"]), len(item["local_maxima_angles_deg"]))
        for item in value["cases"]
    }
    assert len(extrema) >= 3

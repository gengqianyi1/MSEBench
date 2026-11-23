from __future__ import annotations

import json
from pathlib import Path

import yaml

from evaluator.evaluate_suite import aggregate_suite
from evaluator.scoring import compute_score
from tools.rdkit_tools import evaluate_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "organizer_private" / "suite.yaml"


def _case_paths() -> list[Path]:
    suite = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    return [PROJECT_ROOT / item for item in suite["cases"]]


def test_all_official_cases_have_distinct_valid_feasible_starts() -> None:
    starts: set[str] = set()
    families: set[str] = set()
    for path in _case_paths():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        result = evaluate_candidate(config["task"]["start_smiles"], config)
        assert result["valid"] is True
        assert result["feasible"] is True
        assert result["canonical_smiles"] not in starts
        starts.add(result["canonical_smiles"])
        families.add(config["case"]["family"])
    assert len(starts) == 5
    assert len(families) == 5


def test_v4_score_has_no_duplicate_absolute_or_process_credit() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "task.yaml").read_text(encoding="utf-8"))
    result = compute_score(
        final_qed=0.82,
        start_qed=0.769002,
        unique_feasible_candidates=6,
        process_checks={
            "minimum_calls_met": True,
            "final_candidate_evaluated": True,
            "within_tool_budget": True,
        },
        hard_constraints_passed=True,
        process_eligible=True,
        config=config,
        objective_passed=True,
        feedback_revision_quality=True,
    )
    assert set(result["components"]) == {
        "final_chemical_quality",
        "feedback_revision",
    }
    assert "process" not in result["components"]
    assert result["total_score"] == 90.0


def test_blocked_is_excluded_and_prevents_ranking(tmp_path: Path) -> None:
    case_paths = _case_paths()
    for index, path in enumerate(case_paths):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))["case"]["id"]
        directory = tmp_path / case
        directory.mkdir()
        if index == 0:
            (directory / "status.json").write_text(
                json.dumps({"run_status": "BLOCKED", "provider_status": "NETWORK_FAILED"}),
                encoding="utf-8",
            )
        else:
            (directory / "evaluation.json").write_text(
                json.dumps({"score": {"total_score": 30.0, "gate_passed": True, "components": {"process": 0.0}}}),
                encoding="utf-8",
            )
    result = aggregate_suite(SUITE_PATH, tmp_path)
    assert result["blocked_count"] == 1
    assert result["capability_macro_average_excluding_blocked"] == 30.0
    assert result["ranking_eligible"] is False

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "task-06-terpyridine-coupled-torsion-landscape"


def _load(name: str):
    path = TASK / "tests" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"task06_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task06_public_surface_is_single_seed_and_route_neutral() -> None:
    public = TASK / "environment" / "public"
    assert [path.name for path in public.rglob("*.xyz")] == ["terpyridine_seed.xyz"]
    question = (public / "question.md").read_text(encoding="utf-8")
    assert "at most 12 xTB jobs and 72 requested scan points" in question
    assert "55 scientific turns" in question
    assert "8 finalization-only turns" in question
    assert "Choose the candidate coordinates" in question
    serialized = "\n".join(path.relative_to(public).as_posix().lower() for path in public.rglob("*"))
    assert "private" not in serialized
    assert "golden" not in serialized


def test_task06_sidecar_preserves_frozen_budget_and_seven_generic_tools() -> None:
    module = _load("private_suite_adapter")
    adapter = module.create_adapter(TASK / "tests" / "task_snapshot")
    case = adapter.public_case(0)
    assert adapter.case_count == 1
    assert case["budget"] == {
        "max_agent_turns": 63,
        "scientific_turns": 55,
        "finalization_turns": 8,
        "max_tool_calls": 240,
        "compute_credits": 12,
        "max_scan_points": 72,
        "operation_costs": {"sp": 1, "opt": 1, "hess": 1, "scan": 1},
    }
    assert {tool["name"] for tool in case["public_tools"]} == {
        "list_files",
        "read_file",
        "write_file",
        "run_python",
        "xtb_run",
        "budget_status",
        "submit",
    }
    serialized = repr(case).lower()
    assert "reference" not in serialized
    assert "dihedral" not in serialized
    assert "basin" not in serialized


def test_task06_three_frozen_golden_replays_score_one_hundred() -> None:
    module = _load("protocol_grader")
    snapshot = TASK / "tests" / "task_snapshot"
    oracle = snapshot / "organizer_private" / "oracle"
    for repetition in ("repetition_01", "repetition_02", "repetition_03"):
        result = module.grade_evidence(oracle / repetition, snapshot)
        assert result["score"] == 100.0
        assert result["score"] == result["task_local_score"]
        assert result["score"] == result["scientific_basis_score"]
        assert result["score"] == result["comparison_score"]
        assert result["canonical_replay_consistent"] is True


def test_task06_three_invalid_answer_contracts_score_zero(tmp_path: Path) -> None:
    module = _load("protocol_grader")
    snapshot = TASK / "tests" / "task_snapshot"
    source = snapshot / "organizer_private" / "oracle" / "repetition_01"
    for mutation in ("missing", "invalid_json", "schema_invalid"):
        root = tmp_path / mutation
        shutil.copytree(source, root)
        answer = root / "accepted_submission" / "final_answer.json"
        if mutation == "missing":
            answer.unlink()
        elif mutation == "invalid_json":
            answer.write_text("{", encoding="utf-8")
        else:
            answer.write_text(json.dumps({"schema_version": "invalid"}), encoding="utf-8")
        result = module.grade_evidence(root, snapshot)
        assert result["score"] == 0.0

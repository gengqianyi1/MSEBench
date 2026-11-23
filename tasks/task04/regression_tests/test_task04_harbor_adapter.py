from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "task-04-histidine-microstate-ensemble"


def _load_adapter():
    path = TASK / "tests" / "private_suite_adapter.py"
    spec = importlib.util.spec_from_file_location("task04_private_suite_adapter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_adapter(TASK / "tests" / "task_snapshot")


def _load_protocol_grader_module():
    path = TASK / "tests" / "protocol_grader.py"
    spec = importlib.util.spec_from_file_location("task04_protocol_grader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task04_public_surface_is_single_seed_and_contains_no_private_reference() -> None:
    public = TASK / "environment" / "public"
    xyz_files = list(public.rglob("*.xyz"))
    assert [path.name for path in xyz_files] == ["histidine_seed.xyz"]
    assert not any("reference" in path.name.lower() for path in public.rglob("*"))
    question = (public / "question.md").read_text(encoding="utf-8")
    assert "pH" in question
    assert "固定中性分子式" in question


def test_task04_sidecar_publishes_only_generic_tools_and_frozen_budget() -> None:
    adapter = _load_adapter()
    case = adapter.public_case(0)
    assert adapter.case_count == 1
    assert case["budget"] == {
        "max_agent_turns": 160,
        "max_tool_calls": 240,
        "compute_credits": 56,
        "operation_costs": {"sp": 1, "opt": 1, "hess": 4},
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
    assert "private_reference" not in serialized
    assert "identity" not in serialized
    assert "hessian_select" not in serialized


def test_task04_three_frozen_golden_replays_score_one_hundred() -> None:
    module = _load_protocol_grader_module()
    snapshot = TASK / "tests" / "task_snapshot"
    oracle = snapshot / "organizer_private" / "oracle_harbor"
    metrics = {
        "agent_turns_used": 1,
        "scientific_turns_used": 1,
        "finalization_turns_used": 0,
        "tool_calls_used": 1,
        "termination_reason": "submitted",
    }
    for repetition in ("repetition_01", "repetition_02", "repetition_03"):
        result = module.grade_evidence(oracle / repetition, snapshot, metrics, 160)
        assert result["score"] == 100.0
        assert result["score"] == result["task_local_score"]
        assert result["score"] == result["scientific_basis_score"]
        assert result["score"] == result["comparison_score"]


def test_task04_three_invalid_answer_contracts_score_zero(tmp_path: Path) -> None:
    module = _load_protocol_grader_module()
    snapshot = TASK / "tests" / "task_snapshot"
    source = snapshot / "organizer_private" / "oracle_harbor" / "repetition_01"
    metrics = {
        "agent_turns_used": 1,
        "scientific_turns_used": 1,
        "finalization_turns_used": 0,
        "tool_calls_used": 1,
        "termination_reason": "submitted",
    }
    mutations = ("missing", "invalid_json", "schema_invalid")
    for mutation in mutations:
        run = tmp_path / mutation
        shutil.copytree(source, run)
        answer = run / "accepted_submission" / "final_answer.json"
        if mutation == "missing":
            answer.unlink()
        elif mutation == "invalid_json":
            answer.write_text("{", encoding="utf-8")
        else:
            answer.write_text(json.dumps({"schema_version": "invalid"}), encoding="utf-8")
        result = module.grade_evidence(run, snapshot, metrics, 160)
        assert result["score"] == 0.0

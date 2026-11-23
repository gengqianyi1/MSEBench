from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "task-05-caffeine-water-binding-decomposition"


def _load(name: str):
    path = TASK / "tests" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"task05_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task05_public_surface_preserves_full_candidate_set_and_contract() -> None:
    public = TASK / "environment" / "public"
    inputs = public / "data" / "inputs"
    assert len(list(inputs.glob("complex_pose_*.xyz"))) == 14
    assert {path.name for path in inputs.glob("monomer_*.xyz")} == {
        "monomer_A.xyz",
        "monomer_B.xyz",
    }
    question = (public / "question.md").read_text(encoding="utf-8")
    assert "65 个规划/执行回合" in question
    assert "最多 4 个 finalization 回合" in question
    assert "最多 2 个 finalization 回合" not in question
    serialized = "\n".join(path.as_posix() for path in public.rglob("*"))
    assert "private_data" not in serialized
    assert "golden" not in serialized.lower()


def test_task05_sidecar_publishes_only_generic_tools_and_frozen_budget() -> None:
    module = _load("private_suite_adapter")
    adapter = module.create_adapter(TASK / "tests" / "task_snapshot")
    case = adapter.public_case(0)
    assert adapter.case_count == 1
    assert case["budget"] == {
        "max_agent_turns": 69,
        "scientific_turns": 65,
        "finalization_turns": 4,
        "max_tool_calls": 260,
        "max_xtb_jobs": 16,
        "wall_time_seconds": 9000,
        # Audit 2026-09-27: frozen budget pin synced with the finalization
        # wall clock change 600 -> 1800 seconds (see private_suite_adapter).
        "finalization_wall_time_seconds": 1800,
    }
    assert {
        tool.get("name") or tool.get("function", {}).get("name")
        for tool in case["public_tools"]
    } == {
        "read_file",
        "write_file",
        "run_python",
        "xtb_run",
        "xtb_parse",
        "budget_status",
        "submit",
    }
    serialized = repr(case).lower()
    assert "reference" not in serialized
    assert "ranking" not in serialized
    assert "cluster" not in serialized


def test_task05_three_frozen_golden_replays_score_one_hundred() -> None:
    module = _load("protocol_grader")
    snapshot = TASK / "tests" / "task_snapshot"
    oracle = snapshot / "organizer_private" / "oracle"
    for repetition in ("repetition_01", "repetition_02", "repetition_03"):
        root = oracle / repetition
        result = module.grade_evidence(
            root / "submission",
            snapshot,
            root / "harness_manifest_for_grader.json",
        )
        assert result["score"] == 100.0
        assert result["score"] == result["task_local_score"]
        assert result["score"] == result["scientific_basis_score"]
        assert result["score"] == result["comparison_score"]
        assert result["canonical_replay_consistent"] is True


def test_task05_three_invalid_answer_contracts_score_zero(tmp_path: Path) -> None:
    module = _load("protocol_grader")
    snapshot = TASK / "tests" / "task_snapshot"
    source = snapshot / "organizer_private" / "oracle" / "repetition_01"
    for mutation in ("missing", "invalid_json", "schema_invalid"):
        root = tmp_path / mutation
        shutil.copytree(source, root)
        answer = root / "submission" / "final_answer.json"
        if mutation == "missing":
            answer.unlink()
        elif mutation == "invalid_json":
            answer.write_text("{", encoding="utf-8")
        else:
            answer.write_text(json.dumps({"schema_version": "invalid"}), encoding="utf-8")
        result = module.grade_evidence(
            root / "submission",
            snapshot,
            root / "harness_manifest_for_grader.json",
        )
        assert result["score"] == 0.0


@pytest.mark.skipif(sys.platform != "linux", reason="Task 05 xTB sidecar sessions require Linux/WSL2")
def test_task05_partial_sidecar_submit_uses_schema_valid_release(tmp_path: Path) -> None:
    module = _load("private_suite_adapter")
    adapter = module.create_adapter(TASK / "tests" / "task_snapshot")
    release_contract = json.loads(
        (TASK / "tests" / "task_snapshot" / "source_package" / "release_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert release_contract["package_release"] == "1.19.0+eng.20260826"
    state = adapter.start_session(tmp_path / "evidence", "partial-session")
    try:
        result = adapter.submit(0, {"status": "partial"}, [], state)
        assert result["ok"] is True
        manifest_path = (
            tmp_path
            / "evidence"
            / result["evidence_id"]
            / "submission"
            / "run_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["package_release"] == "1.19.0+eng.20260826"
    finally:
        state["temp"].cleanup()


def test_task05_participant_ledger_keeps_finalization_separate():
    module = _load('private_suite_adapter')
    adapter = module.create_adapter(TASK / 'tests/task_snapshot')
    from harness.schema_validation import validate_budget_ledger
    state = {'calls': 19, 'organizer_accounting': {
        'scientific_turns_used': 65, 'finalization_turns_used': 4,
        'tool_calls_used': 20, 'provider_attempts': 69, 'elapsed_seconds': 300,
        'phase': 'finalization', 'exhaustion_trigger': 'scientific_turn_limit',
        'finalization_tool_calls': 3, 'rejected_requests': 1,
    }}
    ledger = adapter._budget_document(state, 1, 1)
    validate_budget_ledger(ledger, adapter.schemas)
    assert ledger['totals']['agent_turns'] == 65
    assert ledger['finalization_turns_used'] == 4
    assert ledger['totals'] == ledger['events'][-1]['cumulative']
    assert ledger['tool_calls_used'] == 20


def test_task05_verifier_preserves_sealed_ledger():
    module = _load('protocol_grader')
    submission = TASK / 'tests/task_snapshot/organizer_private/oracle/repetition_01/submission'
    inner = json.loads((submission / 'run_manifest.json').read_text(encoding='utf-8'))
    outer = module._outer_manifest(submission, module._metrics({'agent_turns_used':69}))
    assert outer['budget_ledger'] == inner['budget_ledger']

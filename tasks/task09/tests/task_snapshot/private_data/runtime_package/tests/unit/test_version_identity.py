from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_v11_task_and_case_identity_are_exact() -> None:
    task = yaml.safe_load(
        (ROOT / "configs" / "task.yaml").read_text(encoding="utf-8")
    )
    case = json.loads(
        (ROOT / "benchmark_case" / "case_spec.json").read_text(encoding="utf-8")
    )
    assert task["task"] == {
        "id": "adaptive_xtb_torsion_profile_agent_v1_1",
        "benchmark_version": "1.1.0",
        "case_id": "2_fluorobiphenyl_central_torsion_v1",
        "protocol": "adaptive_torsion_sampling_tool_loop/1.1",
        "formal_campaign_id": "formal_campaign_v1_1_rc1",
    }
    assert case["task_id"] == task["task"]["id"]
    assert case["benchmark_version"] == task["task"]["benchmark_version"]
    assert case["protocol"] == task["task"]["protocol"]
    assert case["case_id"] == task["task"]["case_id"]


def test_v11_scientific_case_is_unchanged() -> None:
    case = json.loads(
        (ROOT / "benchmark_case" / "case_spec.json").read_text(encoding="utf-8")
    )
    assert case["molecule"]["target_dihedral_atom_maps"] == [6, 7, 8, 9]
    assert case["scientific_invariants"] == {
        "charge": 0,
        "embed_algorithm": "ETKDGv3",
        "initial_geometry": "RDKit ETKDGv3",
        "method": "GFN2-xTB",
        "phase": "gas",
        "preoptimization": "constrained UFF",
        "reference_seed": 20260720,
        "requested_final_angle_tolerance_deg": 2.0,
        "threads": 1,
        "uhf": 0,
        "xtb_version": "6.7.1",
    }
    assert case["allowed_angles_deg"] == list(range(0, 360, 5))


def test_v11_prompt_exposes_phase_rules_but_not_oracle_answers() -> None:
    prompt = (ROOT / "prompts" / "agent_prompt.md").read_text(encoding="utf-8")
    required = (
        "at most six new unique angles",
        "each refinement phase may add at most three",
        "three feedback rounds",
        "Do not call `submit_final`",
        "largest circular gaps",
        "high curvature",
        "model disagreement",
    )
    assert all(fragment in prompt for fragment in required)
    forbidden = (
        "1.523942",
        "3.799162",
        "global minimum at 45",
        "minima at 45",
        "maxima at 0",
    )
    assert all(fragment not in prompt for fragment in forbidden)


def test_v11_archive_name_is_versioned() -> None:
    source = "\n".join(
        (ROOT / "package_tools" / name).read_text(encoding="utf-8")
        for name in ("archive.py", "policy.py")
    )
    assert (
        "LiuChengchun_Adaptive_xTB_Torsion_Profile_Agent_Benchmark_v1_1.zip"
        in source
    )


def test_v11_runtime_and_evaluator_identity_are_not_stale() -> None:
    runner = (ROOT / "framework" / "agent_runner.py").read_text(
        encoding="utf-8"
    )
    trusted = (ROOT / "framework" / "trusted_final.py").read_text(
        encoding="utf-8"
    )
    evaluator = (ROOT / "evaluator" / "evaluate.py").read_text(
        encoding="utf-8"
    )
    trajectory = (ROOT / "framework" / "trajectory.py").read_text(
        encoding="utf-8"
    )
    active = runner + trusted + evaluator + trajectory
    assert "adaptive_xtb_torsion_profile_agent_v1\"" not in active
    assert 'adaptive-torsion-evaluator/1.0' not in active
    assert 'formal-v2' not in active
    assert 'deterministic-baseline/1.0' not in active
    assert 'formal_campaign_v1_1_rc1' in active
    assert 'adaptive-torsion-evaluator/1.1' in active

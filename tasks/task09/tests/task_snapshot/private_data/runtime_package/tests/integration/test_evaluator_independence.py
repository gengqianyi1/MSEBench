from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from evaluator.evaluate_v2 import EvaluationError, _load_trajectory
from framework.trajectory import TrajectoryRecorder


ROOT = Path(__file__).resolve().parents[2]


def test_evaluator_does_not_import_runner_or_tools() -> None:
    for path in (ROOT / "evaluator").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == "framework" or name.startswith("framework.") for name in imports)
        assert not any(name == "tools" or name.startswith("tools.") for name in imports)


def _write_closed_test_trajectory(path: Path) -> None:
    recorder = TrajectoryRecorder(
        run_id="independent-parser-test",
        timestamp_fn=lambda: "2026-07-23T00:00:00.000000Z",
    )
    recorder.append(
        "run_start",
        {
            "requested_model_id": "deterministic-baseline/2.0",
            "task_id": "adaptive_xtb_torsion_profile_agent_v2",
            "benchmark_version": "2.0.0",
            "case_id": "2_fluorobiphenyl_central_torsion_v1",
            "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        },
    )
    recorder.append(
        "run_failure",
        {"status": "TEST_TERMINAL", "raw_provider_response_persisted": False},
    )
    recorder.write(path)


def test_independent_trajectory_parser_is_deterministic(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _write_closed_test_trajectory(trajectory)
    first = _load_trajectory(trajectory)
    second = _load_trajectory(trajectory)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_independent_trajectory_parser_detects_tamper(tmp_path: Path) -> None:
    destination = tmp_path / "trajectory.jsonl"
    _write_closed_test_trajectory(destination)
    lines = destination.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace(
        "adaptive_xtb_torsion_profile_agent_v2",
        "adaptive_xtb_torsion_profile_agent_v1_x",
        1,
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="hash"):
        _load_trajectory(destination)

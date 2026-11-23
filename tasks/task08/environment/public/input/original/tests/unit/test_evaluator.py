from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from evaluator.evaluate import evaluate_files, evaluation_bytes
from evaluator.trusted_validation import canonical_json_bytes, selected_line_range_sha256
from tests.unit.test_evaluator_helpers import (
    build_complete_fixture,
    refresh_final_answer_envelope,
    rehash_events,
)


@pytest.fixture()
def complete_fixture(tmp_path: Path) -> dict[str, Path]:
    return build_complete_fixture(tmp_path)


def _evaluate(paths: dict[str, Path]) -> dict:
    return evaluate_files(
        final_answer_path=paths["final"],
        trajectory_path=paths["trajectory"],
        artifacts_path=paths["case"],
        config_path=paths["config"],
    )


def test_complete_independent_run_scores_100(complete_fixture: dict[str, Path]) -> None:
    result = _evaluate(complete_fixture)

    assert result["evaluation_status"] == "COMPLETED"
    assert result["ungated_score"] == 100
    assert result["official_score"] == 100
    assert result["eligibility"]["eligible"] is True
    assert result["eligibility"]["failed_gates"] == []
    assert result["trusted_evidence"]["artifacts"]["optimization_converged"] is True
    assert result["trusted_evidence"]["artifacts"]["hessian_completed"] is True
    assert result["issues"] == []


def test_repeated_evaluation_is_byte_identical(complete_fixture: dict[str, Path]) -> None:
    first = evaluation_bytes(_evaluate(complete_fixture))
    second = evaluation_bytes(_evaluate(complete_fixture))

    assert first == second
    assert first.endswith(b"\n")
    assert b"\\r\\n" not in first


def test_hash_chain_mutation_is_a_hard_gate(complete_fixture: dict[str, Path]) -> None:
    lines = complete_fixture["trajectory"].read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[5])
    event["payload"]["arguments"]["start_line"] = 2
    lines[5] = json.dumps(event, sort_keys=True, separators=(",", ":"))
    complete_fixture["trajectory"].write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    result = _evaluate(complete_fixture)

    assert result["official_score"] == 0
    assert "trace_hash_chain_valid" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "TRACE_EVENT_HASH" for issue in result["issues"])


def test_missing_terminal_is_not_a_complete_lifecycle(complete_fixture: dict[str, Path]) -> None:
    def remove_terminal(events: list[dict]) -> None:
        events.pop()

    rehash_events(complete_fixture["trajectory"], remove_terminal)
    result = _evaluate(complete_fixture)

    assert result["official_score"] == 0
    assert "complete_lifecycle" in result["eligibility"]["failed_gates"]


def test_duplicate_tool_call_id_fails_pairing(complete_fixture: dict[str, Path]) -> None:
    def duplicate(events: list[dict]) -> None:
        calls = [event for event in events if event["event"] == "tool_call"]
        calls[1]["payload"]["call_id"] = calls[0]["payload"]["call_id"]

    rehash_events(complete_fixture["trajectory"], duplicate)
    result = _evaluate(complete_fixture)

    assert result["official_score"] == 0
    assert "complete_lifecycle" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "TOOL_CALL_DUPLICATE" for issue in result["issues"])


def test_actual_model_mismatch_is_a_hard_gate(complete_fixture: dict[str, Path]) -> None:
    def mismatch(events: list[dict]) -> None:
        for event in events:
            if event["event"] == "provider_response_metadata":
                event["payload"]["actual_model"] = "unexpected/fallback"
                break

    rehash_events(complete_fixture["trajectory"], mismatch)
    result = _evaluate(complete_fixture)

    assert result["ungated_score"] == 100
    assert result["official_score"] == 0
    assert "exact_model_identity_confirmed" in result["eligibility"]["failed_gates"]


def test_baseline_replay_only_exempts_model_identity(complete_fixture: dict[str, Path]) -> None:
    def baseline(events: list[dict]) -> None:
        start = events[0]["payload"]
        start["run_type"] = "BASELINE_REPLAY"
        start["execution_mode"] = "REPLAY"
        start["requested_model"] = None
        for event in events:
            if event["event"] == "provider_response_metadata":
                event["payload"]["actual_model"] = "scripted-baseline"

    rehash_events(complete_fixture["trajectory"], baseline)
    refresh_final_answer_envelope(complete_fixture)
    result = _evaluate(complete_fixture)

    identity = result["eligibility"]["gates"]["exact_model_identity_confirmed"]
    assert identity["status"] == "EXEMPT"
    assert result["eligibility"]["eligible"] is True
    assert result["official_score"] == 100


def test_final_answer_must_match_submit_event(complete_fixture: dict[str, Path]) -> None:
    final_answer = json.loads(complete_fixture["final"].read_text(encoding="utf-8"))
    final_answer["model_submission"]["rationale"] = "A different post-hoc answer."
    complete_fixture["final"].write_bytes(canonical_json_bytes(final_answer, newline=True, indent=2))

    result = _evaluate(complete_fixture)

    assert result["official_score"] == 0
    assert "complete_lifecycle" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "FINAL_TRACE_MISMATCH" for issue in result["issues"])


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("tool_counts", "tool_calls", 999),
        ("trace", "trace_sha256", "0" * 64),
        ("artifacts", "workspace_sha256", "1" * 64),
        ("model_identity", "exact_match", False),
        ("execution_mode", "native_mode", "NO_NATIVE_EXECUTION"),
        ("final_validation", "hessian_completed", False),
    ],
)
def test_forged_trusted_runtime_is_a_hard_gate(
    complete_fixture: dict[str, Path],
    section: str,
    field: str,
    replacement: object,
) -> None:
    value = json.loads(complete_fixture["final"].read_text(encoding="utf-8"))
    value["trusted_runtime"][section][field] = replacement
    complete_fixture["final"].write_bytes(
        canonical_json_bytes(value, newline=True, indent=2)
    )

    result = _evaluate(complete_fixture)

    assert result["official_score"] == 0
    assert "trusted_runtime_verified" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "TRUSTED_RUNTIME_MISMATCH" for issue in result["issues"])


def test_workspace_case_manifest_detects_frozen_file_drift(
    complete_fixture: dict[str, Path],
) -> None:
    job = complete_fixture["case"] / "job_spec.json"
    job.write_bytes(job.read_bytes() + b"\n")
    refresh_final_answer_envelope(complete_fixture)

    result = _evaluate(complete_fixture)

    assert result["official_score"] == 0
    assert "final_artifacts_current" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "WORKSPACE_CASE_INVENTORY" for issue in result["issues"])


def test_cli_writes_deterministic_output(complete_fixture: dict[str, Path], tmp_path: Path) -> None:
    output_a = tmp_path / "evaluation-a.json"
    output_b = tmp_path / "evaluation-b.json"
    command = [
        sys.executable,
        "evaluator/evaluate.py",
        "--final-answer",
        str(complete_fixture["final"]),
        "--trajectory",
        str(complete_fixture["trajectory"]),
        "--artifacts",
        str(complete_fixture["case"]),
        "--config",
        str(complete_fixture["config"]),
        "--output",
    ]
    first = subprocess.run(command + [str(output_a)], cwd=Path(__file__).resolve().parents[2], check=False)
    second = subprocess.run(command + [str(output_b)], cwd=Path(__file__).resolve().parents[2], check=False)

    assert first.returncode == 0
    assert second.returncode == 0
    assert output_a.read_bytes() == output_b.read_bytes()


@pytest.mark.parametrize(
    ("raw", "selected"),
    [
        (b"first\nsecond\nthird\n", b"second\n"),
        (b"first\r\nsecond\r\nthird", b"second\r\n"),
        (b"first\nsecond", b"second"),
    ],
)
def test_evidence_hash_preserves_original_line_bytes(
    tmp_path: Path, raw: bytes, selected: bytes
) -> None:
    path = tmp_path / "evidence.log"
    path.write_bytes(raw)

    assert selected_line_range_sha256(path, 2, 2) == hashlib.sha256(
        selected
    ).hexdigest().upper()

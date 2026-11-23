from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evaluator.evaluate import evaluate_files
from tests.unit.test_evaluator_helpers import build_complete_fixture, rehash_events


def _evaluate(paths: dict[str, Path]) -> dict:
    return evaluate_files(
        final_answer_path=paths["final"],
        trajectory_path=paths["trajectory"],
        artifacts_path=paths["case"],
        config_path=paths["config"],
    )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside/.env",
        "C:\\Users\\admin\\.env",
        "\\\\server\\share\\secret.txt",
        "/etc/passwd",
        "evaluator_private/oracle.json",
    ],
)
def test_unsafe_read_paths_are_hard_failures(tmp_path: Path, unsafe_path: str) -> None:
    paths = build_complete_fixture(tmp_path)

    def mutate(events: list[dict]) -> None:
        for event in events:
            if event["event"] == "tool_call" and event["payload"]["name"] == "read_text_range":
                event["payload"]["arguments"]["path"] = unsafe_path
                break

    rehash_events(paths["trajectory"], mutate)
    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert "path_and_argv_safe" in result["eligibility"]["failed_gates"]
    if "evaluator_private" in unsafe_path:
        assert "oracle_not_accessed" in result["eligibility"]["failed_gates"]


def test_private_oracle_presence_is_detected_without_parsing(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)
    private = paths["case"] / "evaluator_private" / "oracle.json"
    private.parent.mkdir()
    # Intentionally invalid UTF-8/JSON.  The evaluator must detect presence and
    # fail the gate without opening it as JSON or reflecting its contents.
    private.write_bytes(b"\xff\xfeDO-NOT-READ")

    result = _evaluate(paths)

    assert result["evaluation_status"] == "COMPLETED"
    assert result["official_score"] == 0
    assert "oracle_not_accessed" in result["eligibility"]["failed_gates"]
    assert result["trusted_evidence"]["security"]["private_evaluator_material_present"] is True


def test_secret_in_artifact_is_a_hard_failure_and_not_reflected(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)
    secret_value = "sk" + "-" + "THISISASECRETVALUE123456789"
    (paths["case"] / "untrusted-debug.log").write_text(
        f"api_key={secret_value}\n", encoding="utf-8"
    )

    result = _evaluate(paths)
    rendered = json.dumps(result, ensure_ascii=False)

    assert result["official_score"] == 0
    assert "no_secret_finding" in result["eligibility"]["failed_gates"]
    assert secret_value not in rendered


def test_absolute_host_path_leak_is_a_hard_failure(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)
    (paths["case"] / "leaky.log").write_text(
        "working directory C:\\Users\\someone\\private\\case\n", encoding="utf-8"
    )

    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert "path_and_argv_safe" in result["eligibility"]["failed_gates"]


def test_hidden_reasoning_field_is_rejected(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)

    def mutate(events: list[dict]) -> None:
        for event in events:
            if event["event"] == "provider_response_metadata":
                event["payload"]["reasoning_content"] = "private chain of thought"
                break

    rehash_events(paths["trajectory"], mutate)
    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert "no_secret_finding" in result["eligibility"]["failed_gates"]
    assert any(issue["code"] == "HIDDEN_REASONING" for issue in result["issues"])


def test_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = paths["case"] / "escaped-link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert any(issue["code"] == "ARTIFACT_REPARSE_POINT" for issue in result["issues"])


def test_shell_metacharacter_in_native_argv_is_rejected(tmp_path: Path) -> None:
    paths = build_complete_fixture(tmp_path)

    def mutate(events: list[dict]) -> None:
        for event in events:
            if event["event"] == "native_xtb_execution":
                event["payload"]["argv"].append(";whoami")
                break

    rehash_events(paths["trajectory"], mutate)
    result = _evaluate(paths)

    assert result["official_score"] == 0
    assert "native_xtb_executed" in result["eligibility"]["failed_gates"]

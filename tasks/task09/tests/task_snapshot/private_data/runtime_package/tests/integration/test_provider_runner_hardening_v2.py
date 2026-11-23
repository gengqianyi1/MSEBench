from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import pytest

import framework.agent_runner as runner_module
from framework.agent_runner import (
    FormalAgentRunner,
    REDACTED_MODEL_ID,
    parse_single_json_action,
)
from framework.canonical_state import (
    CANONICAL_STATE_MAX_CHARS,
    build_canonical_state,
    canonical_state_text,
)
from framework.redaction import contains_key_like_recursive
from framework.schemas import FORMAL_MODELS, MissingCredentialError
from framework.siliconflow_adapter import ProviderCallError
from scripts.provider_preflight import _error_projection, preflight_one


MODEL = FORMAL_MODELS["kimi"]
SECRET = "sk" + "-abcdefghijklmnopqrstuvwxyz123456"
TEST_EXECUTION_FREEZE_BINDING = {
    "schema_version": "execution-freeze-binding/2.0",
    "content_sha256": "A" * 64,
    "commitment_file_sha256": "B" * 64,
    "file_count": 1,
    "total_bytes": 1,
}


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def _valid_response(content: str = "{}") -> dict[str, Any]:
    return {
        "model": MODEL,
        "choices": [{"message": {"content": content}}],
        "usage": _usage(),
    }


class _SequenceTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.mark.parametrize(
    "malformed",
    [
        {"model": MODEL, "usage": _usage()},
        {"model": MODEL, "choices": [{}], "usage": _usage()},
        {
            "model": MODEL,
            "choices": [{"message": {}}],
            "usage": _usage(),
        },
    ],
    ids=["missing-choices", "missing-message", "missing-content"],
)
def test_missing_provider_output_structure_gets_one_same_state_rerequest(
    tmp_path: Path,
    malformed: dict[str, Any],
) -> None:
    transport = _SequenceTransport([malformed, malformed])
    destination = tmp_path / "run"

    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _seconds: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "FAIL"
    assert result["trusted_runtime"]["safe_status"] == "PROTOCOL_VIOLATION"
    assert len(transport.calls) == 2
    first = json.loads(transport.calls[0]["messages"][1]["content"])
    repeated = json.loads(transport.calls[1]["messages"][1]["content"])
    assert first["phase"] == repeated["phase"] == "phase0"
    assert first["sampling"] == repeated["sampling"]
    assert first["protocol_rerequest"] is False
    assert repeated["protocol_rerequest"] is False
    usage = json.loads(
        (destination / "usage_report.json").read_text(encoding="utf-8")
    )
    assert usage["provider_turns"] == 2
    assert usage["successful_provider_turns"] == 2
    assert [item["output_chars"] for item in usage["turns"]] == [0, 0]
    assert all(
        item["canonical_state_chars"] > 0
        and re.fullmatch(r"[0-9A-F]{64}", item["canonical_state_sha256"])
        for item in usage["turns"]
    )


def test_missing_model_is_identity_fail_with_only_safe_sentinel(
    tmp_path: Path,
) -> None:
    response = _valid_response(
        json.dumps(
            {
                "action": "run_scan_batch",
                "angles_deg": [0, 60, 120, 180, 240, 300],
                "finalize": False,
                "confidence": None,
                "rationale": SECRET,
            }
        )
    )
    response.pop("model")
    destination = tmp_path / "run"

    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=_SequenceTransport([response]),
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _seconds: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "FAIL"
    assert result["trusted_runtime"]["safe_status"] == "IDENTITY_MISMATCH"
    assert result["trusted_runtime"]["actual_model_id"] == REDACTED_MODEL_ID
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".txt", ".log"}
    )
    assert SECRET not in persisted


def test_failed_provider_logical_turn_keeps_state_accounting_skeleton(
    tmp_path: Path,
) -> None:
    safe_error = {
        "status": "AUTHENTICATION_FAILED",
        "error_category": "AUTHENTICATION_FAILED",
        "http_status": 401,
        "provider_error_code": None,
        "sanitized_provider_message": "Provider request failed.",
        "retryable": False,
    }
    destination = tmp_path / "run"

    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=_SequenceTransport([ProviderCallError(safe_error)]),
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _seconds: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "BLOCKED"
    usage = json.loads(
        (destination / "usage_report.json").read_text(encoding="utf-8")
    )
    assert usage["provider_turns"] == 1
    assert usage["successful_provider_turns"] == 0
    assert usage["provider_usage_unavailable"] is True
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["total_tokens"] is None
    assert len(usage["turns"]) == 1
    skeleton = usage["turns"][0]
    assert skeleton["output_chars"] is None
    assert skeleton["usage"] is None
    assert skeleton["canonical_state_chars"] > 0
    assert re.fullmatch(r"[0-9A-F]{64}", skeleton["canonical_state_sha256"])


def test_unexpected_transport_failure_keeps_unknown_usage_skeleton(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run"

    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=_SequenceTransport([RuntimeError(SECRET)]),
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _seconds: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "FAIL"
    assert result["trusted_runtime"]["safe_status"] == "RUNNER_POLICY_FAILURE"
    usage = json.loads(
        (destination / "usage_report.json").read_text(encoding="utf-8")
    )
    assert usage["provider_turns"] == 1
    assert usage["successful_provider_turns"] == 0
    assert usage["provider_usage_unavailable"] is True
    assert (
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
    ) == (None, None, None)
    assert len(usage["turns"]) == 1
    skeleton = usage["turns"][0]
    assert skeleton["output_chars"] is None
    assert skeleton["usage"] is None
    assert skeleton["canonical_state_chars"] > 0
    assert re.fullmatch(r"[0-9A-F]{64}", skeleton["canonical_state_sha256"])
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".txt", ".log"}
    )
    assert SECRET not in persisted


def test_transport_initialization_failure_reports_zero_turn_unknown_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing(_root: Path) -> str:
        raise MissingCredentialError("MISSING_CREDENTIAL")

    monkeypatch.setattr(runner_module, "load_runtime_credential", missing)
    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=None,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _seconds: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "BLOCKED"
    usage = json.loads(
        (destination / "usage_report.json").read_text(encoding="utf-8")
    )
    assert usage["provider_turns"] == 0
    assert usage["successful_provider_turns"] == 0
    assert usage["turns"] == []
    assert usage["provider_usage_unavailable"] is True
    assert (
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
    ) == (None, None, None)


def test_decoded_escaped_secret_in_rationale_is_rejected_recursively() -> None:
    action = {
        "action": "run_scan_batch",
        "angles_deg": [0, 60, 120, 180, 240, 300],
        "finalize": False,
        "confidence": None,
        "rationale": f'Nested JSON: {{"api_key":"{SECRET}"}}',
    }
    wire = json.dumps(action, separators=(",", ":"))
    assert '\\"api_key\\"' in wire
    assert contains_key_like_recursive(action)
    with pytest.raises(Exception):
        parse_single_json_action(wire)


def test_failed_attempt_inspection_is_bounded_in_next_canonical_state() -> None:
    retained_analysis = {
        "selected_model": {"family": "fourier", "order": 2},
        "recommended_refinement_regions": [[85, 95]],
    }
    inspection = {
        "ok": True,
        "status": "FAILED_ATTEMPT_INSPECTION",
        "requested_angle_deg": 90,
        "attempt_count": 2,
        "attempts": [
            {
                "attempt_id": f"attempt-{index}",
                "return_code": 1,
                "timed_out": False,
                "converged": False,
                "energy_parsed": False,
                "angle_error_deg": None,
                "structure_validation": {
                    "atom_count_ok": True,
                    "atom_order_ok": True,
                    "composition_ok": True,
                    "formal_charge_ok": True,
                    "connectivity_ok": False,
                },
            }
            for index in (1, 2)
        ],
        "replacement_sampling_allowed": True,
        "budget": {"native_attempts": 2},
    }
    state = build_canonical_state(
        runtime_state={
            "current_phase": "phase1",
            "unique_angles": [0, 60, 90, 120, 180, 240],
            "valid_angles": [0, 60, 120, 180, 240],
            "failed_angles": [90],
            "unique_angle_count": 6,
            "unique_angles_remaining": 6,
            "valid_unique_angle_count": 5,
            "feedback_rounds": 0,
            "profile_fit_rounds": 1,
            "eligibility_gaps": {"valid_angles_needed": 5},
            "recovery_allowed": False,
        },
        latest_tool_result=inspection,
        retained_profile_analysis=retained_analysis,
    )

    assert state["latest_profile_analysis"] == retained_analysis
    projected = state["latest_failed_attempt_inspection"]
    assert projected["requested_angle_deg"] == 90
    assert projected["attempt_count"] == 2
    assert projected["replacement_sampling_allowed"] is True
    assert all("attempt_id" not in item for item in projected["attempts"])
    assert len(canonical_state_text(state)) <= CANONICAL_STATE_MAX_CHARS


def test_preflight_reprojects_unknown_error_category_to_exact_allowlist() -> None:
    receipt = _error_projection(
        {
            "status": "MODEL_PERMISSION_DENIED",
            "error_category": "NOT_AN_ALLOWED_CATEGORY",
            "http_status": 403,
            "provider_error_code": SECRET,
            "sanitized_provider_message": SECRET,
        },
        requested_model=MODEL,
        timestamp_utc="2026-07-27T00:00:00Z",
    )
    assert receipt["error_category"] == "UNKNOWN_PROVIDER_ERROR"
    assert receipt["provider_error_code"] is None
    assert SECRET not in json.dumps(receipt)


def test_preflight_missing_model_is_identity_failure() -> None:
    response = _valid_response('{"ok":true}')
    response.pop("model")
    result = preflight_one(
        "kimi",
        api_key="placeholder",
        transport_factory=lambda _config, _key: _SequenceTransport([response]),
        now=lambda: "2026-07-27T00:00:00Z",
    )
    assert result["status"] == "FAIL"
    assert result["error_category"] == "IDENTITY_MISMATCH"
    assert result["actual_model"] is None


def test_preflight_key_like_model_identifier_is_redacted() -> None:
    response = _valid_response('{"ok":true}')
    response["model"] = SECRET
    result = preflight_one(
        "kimi",
        api_key="placeholder",
        transport_factory=lambda _config, _key: _SequenceTransport(
            [response]
        ),
        now=lambda: "2026-07-27T00:00:00Z",
    )
    serialized = json.dumps(result, sort_keys=True)
    assert result["status"] == "FAIL"
    assert result["error_category"] == "IDENTITY_MISMATCH"
    assert result["actual_model"] is None
    assert SECRET not in serialized


def test_native_invariant_exception_is_nonretryable_runner_fail_closed(
    tmp_path: Path,
) -> None:
    action = {
        "action": "run_scan_batch",
        "angles_deg": [0, 60, 120, 180, 240, 300],
        "finalize": False,
        "confidence": None,
        "rationale": "Initial coverage.",
    }
    transport = _SequenceTransport(
        [_valid_response(json.dumps(action, separators=(",", ":")))]
    )

    def broken_native(**_kwargs):
        raise RuntimeError(f"native invariant breach {SECRET}")

    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        native_runner=broken_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _seconds: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "FAIL"
    assert result["trusted_runtime"]["safe_status"] == "RUNNER_POLICY_FAILURE"
    assert len(transport.calls) == 1
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in destination.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    assert SECRET not in persisted

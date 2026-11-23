from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from evaluator.evaluate_v2 import EvaluationError, evaluate_run
from framework.agent_runner import FormalAgentRunner
from framework.siliconflow_adapter import ProviderCallError
from framework.trajectory import GENESIS_HASH, canonical_json, canonical_json_bytes
from tests.integration import test_canonical_runner_v2 as canonical_fixture
from tools.science import canonical_json_bytes as pretty_json_bytes


MODEL = canonical_fixture.MODEL


class _ErrorTransport:
    def __init__(self, error: ProviderCallError) -> None:
        self.error = error

    def complete(self, **_kwargs):
        raise self.error


def _valid_formal_run(tmp_path: Path) -> tuple[Path, Path]:
    project, binding = canonical_fixture._test_execution_project(tmp_path)
    transport = canonical_fixture._Transport(
        [
            canonical_fixture._decision(
                [0, 60, 120, 180, 240, 300], False
            ),
            canonical_fixture._decision([30, 150, 270], False),
            canonical_fixture._decision([90, 210, 330], True),
        ]
    )
    run = tmp_path / "run"
    envelope = FormalAgentRunner(
        label="kimi",
        run_root=run,
        transport=transport,
        native_runner=canonical_fixture._fake_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=binding,
        retry_sleep=lambda _seconds: None,
    ).run()
    assert envelope["trusted_runtime"]["status"] == "PASS"
    assert evaluate_run(run, project_root=project, write=False)[
        "eligible"
    ]
    return project, run


def _load_events(run: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run / "trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _rechain(run: Path, events: list[dict]) -> None:
    previous = GENESIS_HASH
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
        event["prev_event_hash"] = previous
        without_hash = {
            key: value for key, value in event.items() if key != "event_hash"
        }
        event["event_hash"] = hashlib.sha256(
            canonical_json_bytes(without_hash)
        ).hexdigest()
        previous = event["event_hash"]
    (run / "trajectory.jsonl").write_text(
        "\n".join(canonical_json(event) for event in events) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_copied_reference_output_is_rejected_after_rehashing_summary(
    tmp_path: Path,
) -> None:
    project, run = _valid_formal_run(tmp_path)
    attempt = (
        run
        / "artifacts"
        / "scan_summaries"
        / "batch_01_angle_000_attempt_01"
    )
    output = attempt / "xtbopt.xyz"
    output.write_bytes((project / "benchmark_case" / "reference.xyz").read_bytes())
    summary_path = attempt / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_entry = next(
        item
        for item in summary["safe_files"]
        if item["path"].endswith("/xtbopt.xyz")
    )
    output_entry["size_bytes"] = output.stat().st_size
    output_entry["sha256"] = hashlib.sha256(
        output.read_bytes()
    ).hexdigest().upper()
    summary_path.write_bytes(pretty_json_bytes(summary))

    with pytest.raises(
        EvaluationError, match="scientific facts|native evidence"
    ):
        evaluate_run(run, project_root=project, write=False)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runtime: runtime.update({"status": "FAIL"}),
        lambda runtime: runtime["model_identity"].update(
            {"model_override_used": True}
        ),
        lambda runtime: runtime["trajectory"].update({"event_count": 1}),
        lambda runtime: runtime["profile"].update(
            {"artifact_sha256": "A" * 64}
        ),
    ],
    ids=[
        "runtime-status",
        "model-override",
        "trajectory-count",
        "profile-artifact-hash",
    ],
)
def test_closed_runtime_tampering_is_rejected(
    tmp_path: Path, mutation
) -> None:
    project, run = _valid_formal_run(tmp_path)
    final_path = run / "final_answer.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    mutation(final["trusted_runtime"])
    final_path.write_bytes(pretty_json_bytes(final))

    with pytest.raises(EvaluationError):
        evaluate_run(run, project_root=project, write=False)


def test_inserted_rejected_identity_event_cannot_pass_success(
    tmp_path: Path,
) -> None:
    project, run = _valid_formal_run(tmp_path)
    events = _load_events(run)
    accepted_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "provider_response_accepted"
    )
    forged = deepcopy(events[accepted_index])
    forged["event"] = "provider_response_rejected_identity"
    forged["payload"]["actual_model_id"] = "fallback-model"
    forged["payload"]["exact_identity"] = False
    events.insert(accepted_index, forged)
    _rechain(run, events)

    with pytest.raises(EvaluationError, match="model identity"):
        evaluate_run(run, project_root=project, write=False)


def test_fake_inspection_action_cannot_bypass_action_budget(
    tmp_path: Path,
) -> None:
    project, run = _valid_formal_run(tmp_path)
    events = _load_events(run)
    first_result_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "public_action_result"
    )
    template = events[first_result_index]
    action = deepcopy(template)
    action["event"] = "public_action"
    action["payload"] = {
        "turn": 2,
        "action": {
            "action": "inspect_failed_attempt",
            "angle_deg": 0,
        },
    }
    result = deepcopy(template)
    result["payload"] = {
        "turn": 2,
        "action": "inspect_failed_attempt",
        "tool_result_chars": 100,
        "result": {
            "ok": True,
            "status": "FAILED_ATTEMPT_INSPECTION",
            "requested_angle_deg": 0,
            "attempt_count": 2,
            "attempts": [{}, {}],
            "replacement_sampling_allowed": False,
            "budget": template["payload"]["result"]["budget"],
        },
    }
    events[first_result_index + 1:first_result_index + 1] = [action, result]
    _rechain(run, events)

    with pytest.raises(
        EvaluationError, match="public action|inspection|action"
    ):
        evaluate_run(run, project_root=project, write=False)


def test_failure_status_mismatch_and_key_like_identity_are_illegal(
    tmp_path: Path,
) -> None:
    project, binding = canonical_fixture._test_execution_project(tmp_path)
    transport = canonical_fixture._Transport(
        [
            canonical_fixture._decision(
                [0, 60, 120, 180, 240, 300], False
            )
        ],
        model="not-the-requested-model",
    )
    run = tmp_path / "run"
    FormalAgentRunner(
        label="kimi",
        run_root=run,
        transport=transport,
        native_runner=canonical_fixture._fake_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=binding,
        retry_sleep=lambda _seconds: None,
    ).run()
    final_path = run / "final_answer.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["trusted_runtime"]["status"] = "BLOCKED"
    final["trusted_runtime"]["actual_model_id"] = (
        "sk" + "-maliciouscredential123456"
    )
    final_path.write_bytes(pretty_json_bytes(final))

    evaluation = evaluate_run(run, project_root=project, write=False)
    assert evaluation["run_status"] == "FAIL"
    assert evaluation["legal_failure_envelope"] is False


def test_legal_provider_failure_without_response_is_accepted(
    tmp_path: Path,
) -> None:
    project, binding = canonical_fixture._test_execution_project(tmp_path)
    safe_error = {
        "status": "AUTHENTICATION_FAILED",
        "error_category": "AUTHENTICATION_FAILED",
        "http_status": 401,
        "provider_error_code": None,
        "sanitized_provider_message": "Provider request failed.",
        "retryable": False,
    }
    run = tmp_path / "run"
    envelope = FormalAgentRunner(
        label="kimi",
        run_root=run,
        transport=_ErrorTransport(ProviderCallError(safe_error)),
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=binding,
        retry_sleep=lambda _seconds: None,
    ).run()
    assert envelope["trusted_runtime"]["status"] == "BLOCKED"

    evaluation = evaluate_run(run, project_root=project, write=False)
    assert evaluation["run_status"] == "BLOCKED"
    assert evaluation["legal_failure_envelope"] is True
    assert evaluation["usage_accounting"]["passed"] is True

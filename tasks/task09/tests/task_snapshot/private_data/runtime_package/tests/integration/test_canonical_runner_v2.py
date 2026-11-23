from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from framework.agent_runner import FormalAgentRunner, parse_single_json_action
from framework.trajectory import load_jsonl
from evaluator.evaluate_v2 import evaluate_run


MODEL = "Pro/moonshotai/Kimi-K2.6"
TEST_EXECUTION_FREEZE_BINDING = {
    "schema_version": "execution-freeze-binding/2.0",
    "content_sha256": "A" * 64,
    "commitment_file_sha256": "B" * 64,
    "file_count": 1,
    "total_bytes": 1,
}


def _canonical_pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _test_execution_project(tmp_path: Path) -> tuple[Path, dict]:
    source = Path(__file__).resolve().parents[2]
    project = tmp_path / "evaluation_project"
    for relative in (
        "benchmark_case/reference.xyz",
        "evaluator_private/oracle_profile.json",
        "evaluator_private/oracle_commitment.json",
    ):
        destination = project.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.joinpath(*relative.split("/")), destination)
    identity = {
        "benchmark_version": "2.0.0",
        "case_id": "2_fluorobiphenyl_central_torsion_v1",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "task_id": "adaptive_xtb_torsion_profile_agent_v2",
    }
    files = []
    for path in sorted(
        item for item in project.rglob("*") if item.is_file()
    ):
        files.append(
            {
                "path": path.relative_to(project).as_posix(),
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest().upper(),
                "size_bytes": path.stat().st_size,
            }
        )
    content = {
        "commitment_kind": "execution",
        "identity": identity,
        "files": files,
    }
    content_sha = hashlib.sha256(_canonical_pretty(content)).hexdigest().upper()
    payload = {
        "schema_version": "execution-commitment/2.0",
        "commitment_kind": "execution",
        "identity": identity,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "content_sha256": content_sha,
    }
    raw = _canonical_pretty(payload)
    (project / "execution_commitment.json").write_bytes(raw)
    return project, {
        "schema_version": "execution-freeze-binding/2.0",
        "content_sha256": content_sha,
        "commitment_file_sha256": hashlib.sha256(raw).hexdigest().upper(),
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
    }


def _fake_native(
    *,
    reference_path: Path,
    angle_deg: int,
    attempt_root: Path,
    timeout_seconds: int,
) -> dict:
    attempt_root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[2]
    baseline_root = (
        source_root
        / "trajectories"
        / "baseline"
        / "artifacts"
        / "scan_summaries"
    )
    fixture_root = None
    fixture_summary = None
    for candidate in sorted(baseline_root.iterdir()):
        if not candidate.is_dir():
            continue
        candidate_summary = json.loads(
            (candidate / "summary.json").read_text(encoding="utf-8")
        )
        if candidate_summary["requested_angle_deg"] == angle_deg:
            fixture_root = candidate
            fixture_summary = candidate_summary
            break
    assert fixture_root is not None
    assert fixture_summary is not None
    (attempt_root / "input.xyz").write_bytes(
        (fixture_root / "input.xyz").read_bytes()
    )
    signed = angle_deg if angle_deg <= 180 else angle_deg - 360
    (attempt_root / "xcontrol.inp").write_text(
        "$constrain\n"
        "  force constant=1.0\n"
        f"  dihedral: 6, 7, 8, 9, {signed:.1f}\n"
        "$end\n",
        encoding="ascii",
        newline="\n",
    )
    (attempt_root / "xtbopt.xyz").write_bytes(
        (fixture_root / "xtbopt.xyz").read_bytes()
    )
    energy = float(fixture_summary["total_energy_hartree"])
    (attempt_root / "stdout.log").write_text(
        "GEOMETRY OPTIMIZATION CONVERGED\n"
        f"| TOTAL ENERGY {energy:.12f} Eh |\n",
        encoding="ascii",
        newline="\n",
    )
    (attempt_root / "stderr.log").write_text(
        "normal termination of xtb\n",
        encoding="ascii",
        newline="\n",
    )
    (attempt_root / "attempt_manifest.json").write_text(
        '{"algorithm":"SHA-256","files":[],"schema_version":"1.0"}',
        encoding="ascii",
    )
    reference_sha = hashlib.sha256(
        reference_path.read_bytes()
    ).hexdigest().upper()
    input_sha = hashlib.sha256(
        (attempt_root / "input.xyz").read_bytes()
    ).hexdigest().upper()
    executable = "XTB_FIXTURE"
    return {
        "method": "GFN2-xTB",
        "phase": "gas",
        "charge": 0,
        "uhf": 0,
        "threads": 1,
        "requested_final_angle_tolerance_deg": 2.0,
        "argv": [
            executable,
            "input.xyz",
            "--gfn",
            "2",
            "--chrg",
            "0",
            "--uhf",
            "0",
            "--opt",
            "tight",
            "--input",
            "xcontrol.inp",
        ],
        "shell": False,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "xtb": {
            "executable": executable,
            "version": "6.7.1",
            "binary_sha256": (
                "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
            ),
            "version_exit_code": 0,
        },
        "return_code": 0,
        "timed_out": False,
        "converged": True,
        "total_energy_hartree": energy,
        "final_angle_deg": fixture_summary["final_angle_deg"],
        "angle_error_deg": fixture_summary["angle_error_deg"],
        "structure_validation": fixture_summary["structure_validation"],
        "profile_eligible": True,
        "reference_sha256": reference_sha,
        "prepared_input": {
            "reference_sha256": reference_sha,
            "input_sha256": input_sha,
        },
        "timeout_seconds": timeout_seconds,
    }


def _three_angle_failure_native(**kwargs) -> dict:
    record = _fake_native(**kwargs)
    if kwargs["angle_deg"] in {0, 60, 120}:
        attempt_root = kwargs["attempt_root"]
        (attempt_root / "xtbopt.xyz").unlink()
        (attempt_root / "stdout.log").write_text(
            "FAILED TO CONVERGE\n",
            encoding="ascii",
            newline="\n",
        )
        (attempt_root / "stderr.log").write_text(
            "abnormal native termination\n",
            encoding="ascii",
            newline="\n",
        )
        record.update(
            {
                "return_code": 1,
                "converged": False,
                "total_energy_hartree": None,
                "final_angle_deg": None,
                "angle_error_deg": None,
                "structure_validation": {
                    "atom_count_ok": False,
                    "atom_order_ok": False,
                    "composition_ok": False,
                    "formal_charge_ok": False,
                    "connectivity_ok": False,
                },
                "profile_eligible": False,
            }
        )
    return record


def _one_angle_failure_native(**kwargs) -> dict:
    record = _fake_native(**kwargs)
    if kwargs["angle_deg"] == 0:
        attempt_root = kwargs["attempt_root"]
        (attempt_root / "xtbopt.xyz").unlink()
        (attempt_root / "stdout.log").write_text(
            "FAILED TO CONVERGE\n",
            encoding="ascii",
            newline="\n",
        )
        (attempt_root / "stderr.log").write_text(
            "abnormal native termination\n",
            encoding="ascii",
            newline="\n",
        )
        record.update(
            {
                "return_code": 1,
                "converged": False,
                "total_energy_hartree": None,
                "final_angle_deg": None,
                "angle_error_deg": None,
                "structure_validation": {
                    "atom_count_ok": False,
                    "atom_order_ok": False,
                    "composition_ok": False,
                    "formal_charge_ok": False,
                    "connectivity_ok": False,
                },
                "profile_eligible": False,
            }
        )
    return record


def _decision(angles: list[int], finalize: bool) -> str:
    return json.dumps(
        {
            "action": "run_scan_batch",
            "angles_deg": angles,
            "finalize": finalize,
            "confidence": "high" if finalize else None,
            "rationale": "Use only the latest compact profile feedback.",
        },
        separators=(",", ":"),
    )


class _Transport:
    def __init__(self, contents: list[str], model: str = MODEL) -> None:
        self.contents = list(contents)
        self.model = model
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        content = self.contents.pop(0)
        return {
            "model": self.model,
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }


def test_formal_runner_uses_three_fresh_states_and_terminal_finalization(
    tmp_path: Path,
) -> None:
    evaluation_project, freeze_binding = _test_execution_project(tmp_path)
    transport = _Transport(
        [
            _decision([0, 60, 120, 180, 240, 300], False),
            _decision([30, 150, 270], False),
            _decision([90, 210, 330], True),
        ]
    )
    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        native_runner=_fake_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=freeze_binding,
        retry_sleep=lambda _: None,
    ).run()
    assert result["trusted_runtime"]["status"] == "PASS"
    assert result["trusted_runtime"]["unique_angles"] == 12
    assert result["trusted_runtime"]["feedback_rounds"] == 2
    assert result["trusted_runtime"]["profile_fit_rounds"] == 3
    assert len(transport.calls) == 3
    assert all(len(call["messages"]) == 2 for call in transport.calls)
    assert all(call["tools"] == [] for call in transport.calls)
    assert all(
        call["response_format"] == {"type": "json_object"}
        for call in transport.calls
    )
    second_context = transport.calls[1]["messages"][1]["content"]
    assert "trajectory" not in second_context.lower()
    assert "stdout" not in second_context.lower()
    assert len(second_context) <= 7000
    events = load_jsonl(destination / "trajectory.jsonl")
    assert events[-2]["event"] == "final_submitted"
    assert events[-1]["event"] == "run_complete"
    assert events[-2]["payload"]["extra_provider_turn"] is False
    usage = json.loads((destination / "usage_report.json").read_text())
    assert usage["provider_turns"] == 3
    assert usage["total_tokens"] == 360
    evaluation = evaluate_run(
        destination, project_root=evaluation_project, write=True
    )
    assert evaluation["run_status"] == "PASS"
    assert evaluation["eligible"] is True
    assert evaluation["eligibility"]["gates"][
        "gate8_execution_freeze_binding"
    ] is True
    assert evaluation["eligibility"]["gates"]["usage_accounting"] is True
    assert evaluation["eligibility"]["gates"]["feedback_event_order"] is True
    assert evaluation["reconstructed"]["usage_accounting"][
        "provider_turns"
    ] == 3
    assert evaluation["reconstructed"]["decision_flow"][
        "scan_decision_turns"
    ] == [1, 2, 3]
    replay = evaluate_run(
        destination, project_root=evaluation_project, write=False
    )
    assert replay == evaluation


def test_exact_identity_mismatch_fails_before_native(tmp_path: Path) -> None:
    evaluation_project, freeze_binding = _test_execution_project(tmp_path)
    transport = _Transport(
        [_decision([0, 60, 120, 180, 240, 300], False)],
        model="not-the-requested-model",
    )
    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        native_runner=_fake_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=freeze_binding,
        retry_sleep=lambda _: None,
    ).run()
    assert result["trusted_runtime"]["status"] == "FAIL"
    assert result["trusted_runtime"]["safe_status"] == "IDENTITY_MISMATCH"
    assert (
        result["trusted_runtime"]["actual_model_id"]
        == "not-the-requested-model"
    )
    assert not list((tmp_path / "raw").rglob("input.xyz"))
    evaluation = evaluate_run(
        destination, project_root=evaluation_project, write=True
    )
    assert evaluation["run_status"] == "FAIL"
    assert evaluation["legal_failure_envelope"] is True
    assert evaluation["actual_model_identity"] == "not-the-requested-model"
    assert evaluation["exact_model_identity"] is False
    assert evaluation["usage_accounting"]["passed"] is True


def test_native_metadata_invariant_breach_is_terminal_and_nonretryable(
    tmp_path: Path,
) -> None:
    _evaluation_project, freeze_binding = _test_execution_project(tmp_path)
    transport = _Transport(
        [_decision([0, 60, 120, 180, 240, 300], False)]
    )
    calls = 0

    def bad_native(**kwargs) -> dict:
        nonlocal calls
        calls += 1
        record = _fake_native(**kwargs)
        record["method"] = "FORBIDDEN-METHOD"
        return record

    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        native_runner=bad_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=freeze_binding,
        retry_sleep=lambda _: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "FAIL"
    assert (
        result["trusted_runtime"]["safe_status"]
        == "NATIVE_INVARIANT_BREACH"
    )
    assert calls == 1
    usage = json.loads(
        (destination / "usage_report.json").read_text(encoding="utf-8")
    )
    assert usage["native_xtb_attempts"] == 1


def test_irrecoverable_native_valid_point_shortfall_is_fail_not_blocked(
    tmp_path: Path,
) -> None:
    evaluation_project, freeze_binding = _test_execution_project(tmp_path)
    transport = _Transport(
        [_decision([0, 60, 120, 180, 240, 300], False)]
    )
    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        native_runner=_three_angle_failure_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=freeze_binding,
        retry_sleep=lambda _: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "FAIL"
    assert (
        result["trusted_runtime"]["safe_status"]
        == "NATIVE_ELIGIBILITY_IMPOSSIBLE"
    )
    assert len(list((tmp_path / "raw").rglob("attempt_manifest.json"))) == 9
    evaluation = evaluate_run(
        destination, project_root=evaluation_project, write=True
    )
    assert evaluation["run_status"] == "FAIL"
    assert evaluation["legal_failure_envelope"] is True
    assert evaluation["execution_freeze_binding"]["passed"] is True


def test_failed_attempt_inspection_retains_latest_profile_feedback(
    tmp_path: Path,
) -> None:
    evaluation_project, freeze_binding = _test_execution_project(tmp_path)
    transport = _Transport(
        [
            _decision([0, 60, 120, 180, 240, 300], False),
            json.dumps(
                {"action": "inspect_failed_attempt", "angle_deg": 0},
                separators=(",", ":"),
            ),
            _decision([30, 150, 270], False),
            _decision([90, 210], True),
        ]
    )
    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        native_runner=_one_angle_failure_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=freeze_binding,
        retry_sleep=lambda _: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "PASS"
    before_inspection = json.loads(
        transport.calls[1]["messages"][1]["content"]
    )
    after_inspection = json.loads(
        transport.calls[2]["messages"][1]["content"]
    )
    assert before_inspection["latest_profile_analysis"] is not None
    assert (
        after_inspection["latest_profile_analysis"]
        == before_inspection["latest_profile_analysis"]
    )
    assert (
        after_inspection["latest_result_status"]
        == "FAILED_ATTEMPT_INSPECTION"
    )
    evaluation = evaluate_run(
        destination, project_root=evaluation_project, write=True
    )
    assert evaluation["run_status"] == "PASS"
    assert evaluation["reconstructed"]["decision_flow"][
        "scan_decision_turns"
    ] == [1, 3, 4]


def test_one_protocol_rerequest_reuses_the_same_scientific_state(
    tmp_path: Path,
) -> None:
    transport = _Transport(
        [
            "not json",
            _decision([0, 60, 120, 180, 240, 300], False),
            _decision([30, 150, 270], False),
            _decision([90, 210, 330], True),
        ]
    )
    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=transport,
        native_runner=_fake_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _: None,
    ).run()

    assert result["trusted_runtime"]["status"] == "PASS"
    assert len(transport.calls) == 4
    first = json.loads(transport.calls[0]["messages"][1]["content"])
    repeated = json.loads(transport.calls[1]["messages"][1]["content"])
    assert first["protocol_rerequest"] is False
    assert repeated["protocol_rerequest"] is False
    assert first == repeated
    usage = json.loads((destination / "usage_report.json").read_text())
    assert usage["provider_turns"] == 4
    assert usage["turns"][0]["tool_result_chars"] is None
    assert all(
        isinstance(item["tool_result_chars"], int)
        and 0 < item["tool_result_chars"] <= 6000
        for item in usage["turns"][1:]
    )
    events = load_jsonl(destination / "trajectory.jsonl")
    violations = [
        event for event in events if event["event"] == "protocol_violation"
    ]
    assert len(violations) == 1
    assert violations[0]["payload"]["same_state_rerequest_allowed"] is True


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"action":"inspect_failed_attempt","angle_deg":90}', 90),
        (
            '```json\n{"action":"inspect_failed_attempt","angle_deg":95}\n```',
            95,
        ),
        (
            '```\n{"action":"inspect_failed_attempt","angle_deg":100}\n```',
            100,
        ),
    ],
)
def test_parser_allows_only_deterministic_normalization(
    content: str, expected: int
) -> None:
    assert parse_single_json_action(content)["angle_deg"] == expected


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "prefix {}",
        "{} suffix",
        "[]",
        '{"a":1,"a":2}',
        "```json\n{}\n```\nextra",
        '{"authorization":"Bearer abcdefghijklmnopqrstuvwxyz"}',
    ],
)
def test_parser_rejects_non_single_or_key_like_output(content: str) -> None:
    with pytest.raises(Exception):
        parse_single_json_action(content)

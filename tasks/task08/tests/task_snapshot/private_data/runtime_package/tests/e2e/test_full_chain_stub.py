from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pytest

from evaluator.evaluate import evaluate_files, evaluation_bytes, write_evaluation
from framework import AgentRunner, ScriptedTransport
from package_tools.archive import build_flat_archive
from package_tools.archive_validation import (
    extract_safely,
    preflight_archive,
    validate_extracted_static,
)
from package_tools.manifest import file_entries, write_manifest
from package_tools.policy import PACKAGE_GATE_CHECKS, is_allowed, normalize_relative


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baselines" / "live"
BASELINE_WORKSPACE = BASELINE / "workspace"
MODEL = "Pro/moonshotai/Kimi-K2.6"

_PREPARED_FILES = frozenset(
    {
        "attempt.json",
        "input.xyz",
        "input_manifest.json",
        "prepared.json",
        "prepared_command.json",
        "provenance.json",
        "xcontrol.inp",
    }
)
_VALIDATION_FILES = frozenset({"artifact_manifest.json", "validation.json"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _baseline_tool_pairs() -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in (BASELINE / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = {
        event["payload"]["call_id"]: event["payload"]
        for event in events
        if event.get("event") == "tool_result"
    }
    pairs = []
    for event in events:
        if event.get("event") != "tool_call":
            continue
        call = event["payload"]
        result = results[call["call_id"]]
        assert result["ok"] is True
        pairs.append(
            {
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
                "result": result["result"],
            }
        )
    opt_prepare = next(
        pair for pair in pairs
        if pair["name"] == "prepare_restart"
        and pair["arguments"].get("phase") == "geometry_optimization"
    )
    submit = next(pair for pair in pairs if pair["name"] == "submit_final")
    source = opt_prepare["result"]["source_geometry"]
    submit["arguments"]["diagnosis"]["conflict_resolution"] = (
        "SCHEDULER_STATUS_NOT_SCIENTIFIC_COMPLETION"
    )
    submit["arguments"]["diagnosis"]["legacy_disposition"] = (
        "NOT_CURRENT_INFORMATION_ONLY"
    )
    submit["arguments"]["restart_summary"]["starting_geometry_source"] = {
        "attempt_id": "attempt_000_initial_failure",
        "path": source["path"],
        "frame_index": source["frame_index"],
        "file_sha256": source["file_sha256"],
        "frame_sha256": source["frame_sha256"],
    }
    assert [pair["name"] for pair in pairs] == [
        "list_case_files",
        "read_text_range",
        "read_text_range",
        "parse_xtb_log",
        "inspect_scheduler_record",
        "validate_xyz",
        "inspect_artifact_provenance",
        "inspect_artifact_provenance",
        "prepare_restart",
        "run_xtb",
        "validate_attempt",
        "prepare_restart",
        "run_xtb",
        "validate_attempt",
        "submit_final",
    ]
    return pairs


def _provider_response(request_number: int, pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls = [
        {
            "id": pair["call_id"],
            "type": "function",
            "function": {
                "name": pair["name"],
                "arguments": json.dumps(
                    pair["arguments"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
        for pair in pairs
    ]
    return {
        "id": f"stub-request-{request_number:02d}",
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Deterministic test-fixture tool step.",
                    "tool_calls": calls,
                },
            }
        ],
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
    }


class _FrozenReceiptRegistry:
    """Stage frozen receipts while truthfully declaring that no native process ran."""

    live_native_execution = False

    def __init__(
        self,
        *,
        workspace: Path,
        expected: Sequence[Mapping[str, Any]],
        fail_run_xtb: bool = False,
    ) -> None:
        self.workspace = workspace
        self.expected = [deepcopy(dict(pair)) for pair in expected]
        self.fail_run_xtb = fail_run_xtb
        self.calls: list[str] = []
        self.native_process_invocations = 0

    def _stage(self, attempt_id: str, phase: str) -> None:
        source = BASELINE_WORKSPACE / "attempts" / attempt_id
        target = self.workspace / "attempts" / attempt_id
        target.mkdir(parents=True, exist_ok=True)
        names = {path.name for path in source.iterdir() if path.is_file()}
        if phase == "prepare":
            selected = names & _PREPARED_FILES
        elif phase == "run":
            selected = names - _PREPARED_FILES - _VALIDATION_FILES
        elif phase == "validate":
            selected = names & _VALIDATION_FILES
        else:  # pragma: no cover - test helper programming error
            raise AssertionError(phase)
        for name in sorted(selected):
            shutil.copy2(source / name, target / name)

    def dispatch(self, name: str, arguments: dict[str, Any], *, call_id: str) -> dict[str, Any]:
        index = len(self.calls)
        assert index < len(self.expected), f"unexpected extra tool call: {name}"
        pair = self.expected[index]
        assert (call_id, name, arguments) == (
            pair["call_id"],
            pair["name"],
            pair["arguments"],
        )
        self.calls.append(name)

        if name == "prepare_restart":
            self._stage(str(pair["result"]["attempt_id"]), "prepare")
        elif name == "run_xtb":
            if self.fail_run_xtb:
                return {
                    "ok": False,
                    "tool": name,
                    "call_id": call_id,
                    "error": {"code": "SIMULATED_NATIVE_FAILURE"},
                }
            # This copies frozen bytes; it never starts xTB or another process.
            self._stage(str(arguments["attempt_id"]), "run")
        elif name == "validate_attempt":
            self._stage(str(arguments["attempt_id"]), "validate")

        return {
            "ok": True,
            "tool": name,
            "call_id": call_id,
            "result": deepcopy(pair["result"]),
        }


def _run_stub(
    root: Path,
    *,
    false_success_after_native_failure: bool = False,
) -> tuple[Path, Path, _FrozenReceiptRegistry]:
    pairs = _baseline_tool_pairs()
    if false_success_after_native_failure:
        plan = [*pairs[:10], pairs[14]]
        groups = [plan[:8], plan[8:10], plan[10:]]
    else:
        plan = pairs
        groups = [pairs[:8], pairs[8:11], pairs[11:14], pairs[14:]]

    workspace = root / "workspace"
    shutil.copytree(ROOT / "benchmark_case", workspace)
    output = root / "runner_output"
    registry = _FrozenReceiptRegistry(
        workspace=workspace,
        expected=plan,
        fail_run_xtb=false_success_after_native_failure,
    )
    transport = ScriptedTransport(
        [_provider_response(index, group) for index, group in enumerate(groups, 1)]
    )
    runner = AgentRunner(
        task_config=ROOT / "configs" / "task.yaml",
        logical_model="kimi",
        output_dir=output,
        tool_registry=registry,
        transport=transport,
        execution_mode="TEST_STUB",
        native_mode="NO_NATIVE_EXECUTION",
        run_type="TEST",
        run_id=(
            "e2e-false-success-after-native-failure"
            if false_success_after_native_failure
            else "e2e-deterministic-full-chain"
        ),
        artifact_root=workspace,
        timestamp_fn=lambda: "2026-07-20T00:00:00Z",
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )
    result = runner.run()
    assert result.status == "COMPLETE"
    assert registry.calls == [pair["name"] for pair in plan]
    assert registry.native_process_invocations == 0
    return output, workspace, registry


@pytest.fixture(scope="module")
def successful_stub_chain(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    output, workspace, _registry = _run_stub(tmp_path_factory.mktemp("full_chain"))
    return output, workspace


def test_agent_runner_stub_chain_is_scientifically_replayable_but_not_formal(
    successful_stub_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    output, workspace = successful_stub_chain
    envelope = json.loads((output / "final_answer.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert set(envelope) == {"schema_version", "model_submission", "trusted_runtime"}
    trusted = envelope["trusted_runtime"]
    assert trusted["final_validation"] == {
        "artifacts_current": True,
        "final_attempt_id": "attempt_002",
        "final_attempt_validated": True,
        "hessian_completed": True,
        "invariants_preserved": True,
        "optimization_completed": True,
    }
    assert [item["attempt_id"] for item in trusted["attempt_history"]] == [
        "attempt_001",
        "attempt_002",
    ]
    assert metadata["execution_mode"] == "TEST_STUB"
    assert metadata["native_mode"] == "NO_NATIVE_EXECUTION"
    current_inventory = _inventory(workspace)
    v1_inventory = _inventory(BASELINE_WORKSPACE)
    assert current_inventory["case_manifest.json"] != v1_inventory["case_manifest.json"]
    current_inventory.pop("case_manifest.json")
    v1_inventory.pop("case_manifest.json")
    assert current_inventory == v1_inventory

    result = evaluate_files(
        final_answer_path=output / "final_answer.json",
        trajectory_path=output / "trajectory.jsonl",
        artifacts_path=workspace,
        config_path=ROOT / "configs" / "task.yaml",
    )
    second = evaluate_files(
        final_answer_path=output / "final_answer.json",
        trajectory_path=output / "trajectory.jsonl",
        artifacts_path=workspace,
        config_path=ROOT / "configs" / "task.yaml",
    )
    assert evaluation_bytes(result) == evaluation_bytes(second)
    assert result["ungated_score"] == 61
    assert result["official_score"] == 0
    assert result["eligibility"]["eligible"] is False
    gates = result["eligibility"]["gates"]
    for name in (
        "complete_lifecycle",
        "decisive_log_evidence_read",
        "trace_hash_chain_valid",
        "final_answer_schema_valid",
        "trusted_runtime_verified",
    ):
        assert gates[name]["passed"] is True, name
    assert gates["formal_mode_truthful"]["passed"] is False
    assert gates["native_xtb_executed"]["passed"] is False
    for name in (
        "optimization_converged",
        "hessian_completed",
        "final_artifacts_current",
        "scientific_invariants_preserved",
    ):
        assert gates[name]["passed"] is False, name

    # Exercise the documented independent CLI twice; it never imports the
    # runner, calls a provider, or launches xTB.
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    environment.pop("GJLD_API_KEY", None)
    cli_outputs = [tmp_path / "evaluation-1.json", tmp_path / "evaluation-2.json"]
    for target in cli_outputs:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluator" / "evaluate.py"),
                "--final-answer",
                str(output / "final_answer.json"),
                "--trajectory",
                str(output / "trajectory.jsonl"),
                "--artifacts",
                str(workspace),
                "--config",
                str(ROOT / "configs" / "task.yaml"),
                "--output",
                str(target),
            ],
            cwd=ROOT,
            env=environment,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert cli_outputs[0].read_bytes() == cli_outputs[1].read_bytes()
    assert cli_outputs[0].read_bytes() == evaluation_bytes(result)


def test_native_tool_failure_is_preserved_and_model_claim_cannot_fake_recovery(
    tmp_path: Path,
) -> None:
    output, workspace, registry = _run_stub(
        tmp_path,
        false_success_after_native_failure=True,
    )
    events = [
        json.loads(line)
        for line in (output / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed_runs = [
        event
        for event in events
        if event["event"] == "tool_result"
        and event["payload"].get("name") == "run_xtb"
        and event["payload"].get("ok") is False
    ]
    assert len(failed_runs) == 1
    assert not any(event["event"] == "native_xtb_execution" for event in events)
    assert not any(event["event"] == "attempt_validated" for event in events)
    assert registry.native_process_invocations == 0

    envelope = json.loads((output / "final_answer.json").read_text(encoding="utf-8"))
    assert envelope["model_submission"]["final_status"] == "RECOVERED"
    assert envelope["trusted_runtime"]["final_validation"] == {
        "artifacts_current": False,
        "final_attempt_id": "attempt_002",
        "final_attempt_validated": False,
        "hessian_completed": False,
        "invariants_preserved": False,
        "optimization_completed": False,
    }
    assert envelope["trusted_runtime"]["attempt_history"][0]["native_executed"] is False

    evaluation = evaluate_files(
        final_answer_path=output / "final_answer.json",
        trajectory_path=output / "trajectory.jsonl",
        artifacts_path=workspace,
        config_path=ROOT / "configs" / "task.yaml",
    )
    assert evaluation["official_score"] == 0
    assert evaluation["eligibility"]["eligible"] is False
    gates = evaluation["eligibility"]["gates"]
    for name in (
        "native_xtb_executed",
        "optimization_converged",
        "hessian_completed",
        "final_artifacts_current",
    ):
        assert gates[name]["passed"] is False, name
    assert gates["trusted_runtime_verified"]["passed"] is True


def test_v1_frozen_live_baseline_is_explicitly_invalidated_by_v2_contract() -> None:
    result = evaluate_files(
        final_answer_path=BASELINE / "final_answer.json",
        trajectory_path=BASELINE / "trajectory.jsonl",
        artifacts_path=BASELINE_WORKSPACE,
        config_path=ROOT / "configs" / "task.yaml",
    )
    assert result["eligibility"]["eligible"] is False
    assert result["official_score"] == 0
    assert result["issues"]
    assert evaluation_bytes(result) != (BASELINE / "evaluation.json").read_bytes()


def test_stub_outputs_fit_flat_archive_and_manifest_static_chain(
    successful_stub_chain: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    output, workspace = successful_stub_chain
    payload = tmp_path / "payload"
    trajectory = payload / "trajectories" / "kimi"
    trajectory.mkdir(parents=True)
    for name in ("trajectory.jsonl", "dialogue.md", "metadata.json", "final_answer.json"):
        shutil.copy2(output / name, trajectory / name)
    evaluation = evaluate_files(
        final_answer_path=output / "final_answer.json",
        trajectory_path=output / "trajectory.jsonl",
        artifacts_path=workspace,
        config_path=ROOT / "configs" / "task.yaml",
    )
    write_evaluation(trajectory / "evaluation.json", evaluation)
    artifact_files = [path for path in workspace.rglob("*") if path.is_file()]
    (trajectory / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "file_count": len(artifact_files),
                "files": file_entries(workspace, artifact_files),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    for root_name in ("README.md", "TASK.md", "run_all.ps1"):
        shutil.copy2(ROOT / root_name, payload / root_name)
    (payload / "configs").mkdir()
    release = {
        "schema_version": "1.0",
        "status": "PASS",
        "models": {},
        "gates": {
            "package": {
                "status": "PASS",
                "passed": True,
                "checks": {name: True for name in PACKAGE_GATE_CHECKS},
            }
        },
    }
    (payload / "configs" / "release.json").write_text(
        json.dumps(release, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staged_files = sorted(path for path in payload.rglob("*") if path.is_file())
    assert all(
        is_allowed(normalize_relative(path.relative_to(payload)))
        for path in staged_files
    )
    write_manifest(
        payload / "MANIFEST.json",
        {
            "schema_version": "1.0",
            "release_status": "PASS",
            "models": {},
            "file_count_excluding_manifest": len(staged_files),
            "files": file_entries(payload, staged_files),
        },
    )

    archive = tmp_path / "candidate.zip"
    built = build_flat_archive(payload, archive)
    preflight = preflight_archive(archive)
    assert built["archive_sha256"] == preflight["archive_sha256"]
    assert preflight["passed"] is True, preflight["errors"]
    extracted = extract_safely(archive, tmp_path / "clean")
    static = validate_extracted_static(extracted)
    assert static["passed"] is True, static["errors"]
    assert static["manifest"]["passed"] is True
    assert static["release_commit"]["package_gate_checks_complete"] is True

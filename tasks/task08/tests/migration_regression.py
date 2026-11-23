#!/usr/bin/env python3
"""Task 08 native sidecar Oracle and organizer-evidence security regression."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def grade(work: Path, evidence: Path, logs: Path, name: str):
    old = {key: os.environ.get(key) for key in (
        "CHEMBENCH_WORKSPACE", "CHEMBENCH_ORGANIZER_EVIDENCE", "CHEMBENCH_LOGS"
    )}
    os.environ.update({
        "CHEMBENCH_WORKSPACE": str(work),
        "CHEMBENCH_ORGANIZER_EVIDENCE": str(evidence),
        "CHEMBENCH_LOGS": str(logs),
    })
    try:
        module = load(name, TASK / "tests" / "protocol_grader.py")
        module.main()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return json.loads((logs / "grader-result.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    work = output / "workspace"
    evidence = output / "organizer_evidence"
    work.mkdir()

    fixture_model = "zai-org/GLM-5.2"
    os.environ["CHEMBENCH_MODEL_ID"] = fixture_model
    os.environ["CHEMBENCH_ACTUAL_MODEL_ID"] = fixture_model

    module = load("task08_adapter_regression", TASK / "tests" / "private_suite_adapter.py")
    adapter = module.create_adapter(TASK / "tests" / "task_snapshot")
    state = adapter.start_session(evidence, "oracle-replay")
    for index in range(adapter.case_count):
        dialogue = json.loads(
            (TASK / "steps" / f"case-{index + 1}" / "solution" / "oracle_dialogue.json").read_text(encoding="utf-8")
        )
        tool_events = []
        candidate = None
        for turn in dialogue:
            for call in turn.get("tool_calls", []):
                response = adapter.call_tool(index, call.get("tool"), call.get("arguments"), state)
                tool_events.append({"tool": call.get("tool"), "arguments": call.get("arguments"), "response": response})
                if response.get("ok") is not True:
                    manager = state["registries"][index]["registry"].attempt_manager
                    conda = shutil.which("conda")
                    probe = subprocess.run(
                        [conda, "env", "list", "--json"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                        env=manager._safe_environment(1),
                    )
                    document = json.loads(probe.stdout.decode("utf-8"))
                    expected = manager._xtb_config.get("conda_environment")
                    matches = [
                        value for value in document.get("envs", [])
                        if Path(value).name.casefold() == str(expected).casefold()
                    ]
                    summary = {
                        "status": "BLOCKED",
                        "task": 8,
                        "case_index": index + 1,
                        "tool": call.get("tool"),
                        "public_response": response,
                        "organizer_response": state["registries"][index]["records"][-1]["response"],
                        "runtime_probe": {"conda": conda, "expected": expected, "matches": matches},
                        "live_model_executed": False,
                        "reason": "FROZEN_NATIVE_RUNTIME_UNAVAILABLE",
                    }
                    write(output / "summary.json", summary)
                    print(json.dumps(summary, indent=2))
                    return 2
            if isinstance(turn.get("candidate"), dict):
                candidate = turn["candidate"]
        assert candidate is not None
        submission = adapter.submit(index, candidate, tool_events, state)
        assert submission.get("ok") is True, submission
        write(work / "cases" / f"case-{index + 1}" / "case_result.json", {
            "case_id": submission["case_id"],
            "candidate": {"final_status": "FORGED_WORKSPACE_VALUE"},
            "tool_events": [{"forged": True}],
            "submission": submission,
        })

    honest = grade(work, evidence, output / "honest_logs", "task08_grader_honest")
    assert honest["score"] == 100.0 and not honest["failure_codes"], honest

    injected = grade(work, evidence, output / "self_report_logs", "task08_grader_self_report")
    assert injected["score"] == honest["score"] and not injected["failure_codes"], injected

    identity_evidence = output / "missing_identity_evidence"
    shutil.copytree(evidence, identity_evidence)
    first_identity = identity_evidence / "oracle-replay" / "aspirin_gfn2_opt_hess_restart_v1" / "metadata.json"
    identity_payload = json.loads(first_identity.read_text(encoding="utf-8"))
    identity_payload["actual_model_id"] = None
    identity_payload["exact_model_identity"] = False
    write(first_identity, identity_payload)
    identity_rejected = grade(
        work, identity_evidence, output / "missing_identity_logs", "task08_grader_missing_identity"
    )
    assert identity_rejected["score"] == 0.0 and any(
        "EXACT_MODEL_IDENTITY_FAILED" in code for code in identity_rejected["failure_codes"]
    ), identity_rejected
    assert not (output / "missing_identity_logs" / "reward.txt").exists()

    unsafe_work = output / "unsafe_workspace"
    shutil.copytree(work, unsafe_work)
    unsafe = json.loads((unsafe_work / "cases" / "case-1" / "case_result.json").read_text())
    unsafe["submission"]["evidence_id"] = "../oracle-replay/aspirin_gfn2_opt_hess_restart_v1"
    write(unsafe_work / "cases" / "case-1" / "case_result.json", unsafe)
    rejected = grade(unsafe_work, evidence, output / "unsafe_logs", "task08_grader_unsafe")
    assert rejected["score"] == 0.0 and rejected["failure_codes"], rejected
    assert not (output / "unsafe_logs" / "reward.txt").exists()

    summary = {
        "status": "PASS",
        "task": 8,
        "cases": adapter.case_count,
        "legacy_oracle_score": 100.0,
        "harbor_grader_fixture_score": honest["score"],
        "fixture_model_identity": fixture_model,
        "live_model_executed": False,
        "missing_model_identity_rejected": True,
        "workspace_self_report_ignored": True,
        "unsafe_evidence_id_rejected": True,
        "required_artifacts_present": all((work / path).exists() for path in (
            "benchmark_result.json", "final_answer.json", "benchmark_run/campaign_summary.json",
            "benchmark_run/protocol/suite_evaluation.json",
        )),
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

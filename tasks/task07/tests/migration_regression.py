#!/usr/bin/env python3
"""Task 07 organizer-evidence, security, and legacy-parity regression."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

TASK = Path(__file__).resolve().parents[1]
RUNTIME = TASK / "tests" / "task_snapshot" / "private_data" / "runtime_package"


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
    legacy_runs = output / "legacy_replay"
    work.mkdir()

    fixture_model = "zai-org/GLM-5.2"
    os.environ["CHEMBENCH_MODEL_ID"] = fixture_model
    os.environ["CHEMBENCH_ACTUAL_MODEL_ID"] = fixture_model

    adapter_module = load("task07_adapter_regression", TASK / "tests" / "private_suite_adapter.py")
    adapter = adapter_module.create_adapter(TASK / "tests" / "task_snapshot")
    state = adapter.start_session(evidence, "oracle-replay")
    sys.path.insert(0, str(RUNTIME))
    from baselines.baseline_agent import run_baseline
    from evaluator.evaluate import evaluate_submission
    from evaluator.evaluate_suite import aggregate_suite, load_suite

    suite_path = RUNTIME / "organizer_private" / "suite.yaml"
    _, case_paths = load_suite(suite_path)
    for index, config_path in enumerate(case_paths, 1):
        events = []
        answer = run_baseline(config_path, process_events=events)
        public_events = []
        for event in events:
            if event.get("event") != "tool_call":
                continue
            response = adapter.call_tool(index - 1, event.get("tool"), event.get("arguments"), state)
            public_events.append({
                "tool": event.get("tool"),
                "arguments": event.get("arguments"),
                "response": response,
            })
        candidate = answer["final_candidate"]
        submission = adapter.submit(index - 1, candidate, public_events, state)
        write(work / "cases" / f"case-{index}" / "case_result.json", {
            "case_id": submission["case_id"],
            "candidate": {"name": "workspace-self-report", "smiles": "C", "rationale": "ignored"},
            "tool_events": [{"forged": True}],
            "submission": submission,
        })
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        case_id = config["case"]["id"]
        evaluation = evaluate_submission(answer, config, process_events=events)
        write(legacy_runs / case_id / "evaluation.json", evaluation)
        write(legacy_runs / case_id / "status.json", {
            "run_status": "PASS" if evaluation.get("eligible") else "FAIL",
            "provider_status": "FROZEN_REPLAY",
        })

    legacy_suite = aggregate_suite(suite_path, legacy_runs)
    honest = grade(work, evidence, output / "honest_logs", "task07_grader_honest")
    assert not honest["failure_codes"], honest
    assert honest["score"] == legacy_suite["capability_score"], (honest, legacy_suite)
    honest_score = honest["score"]

    self_report = json.loads((work / "cases" / "case-1" / "case_result.json").read_text())
    self_report["candidate"] = {"name": "forged", "smiles": "invalid", "rationale": "forged"}
    self_report["tool_events"] = [{"response": {"ok": True, "result": {"qed": 1.0}}}]
    write(work / "cases" / "case-1" / "case_result.json", self_report)
    injected = grade(work, evidence, output / "self_report_logs", "task07_grader_self_report")
    assert injected["score"] == honest_score and not injected["failure_codes"], injected

    identity_evidence = output / "missing_identity_evidence"
    shutil.copytree(evidence, identity_evidence)
    first_identity = identity_evidence / "oracle-replay" / "qed_public_01" / "evidence.json"
    identity_payload = json.loads(first_identity.read_text(encoding="utf-8"))
    identity_payload["actual_model_id"] = None
    identity_payload["exact_model_identity"] = False
    write(first_identity, identity_payload)
    identity_rejected = grade(
        work, identity_evidence, output / "missing_identity_logs", "task07_grader_missing_identity"
    )
    assert identity_rejected["score"] == 0.0 and any(
        "EXACT_MODEL_IDENTITY_FAILED" in code for code in identity_rejected["failure_codes"]
    ), identity_rejected
    assert not (output / "missing_identity_logs" / "reward.txt").exists()

    unsafe_work = output / "unsafe_workspace"
    shutil.copytree(work, unsafe_work)
    unsafe = json.loads((unsafe_work / "cases" / "case-1" / "case_result.json").read_text())
    unsafe["submission"]["evidence_id"] = "../oracle-replay/qed_public_01"
    write(unsafe_work / "cases" / "case-1" / "case_result.json", unsafe)
    rejected = grade(unsafe_work, evidence, output / "unsafe_logs", "task07_grader_unsafe")
    assert rejected["score"] == 0.0 and rejected["failure_codes"], rejected
    assert not (output / "unsafe_logs" / "reward.txt").exists()

    summary = {
        "status": "PASS",
        "task": 7,
        "cases": len(case_paths),
        "legacy_score": legacy_suite["capability_score"],
        "harbor_grader_fixture_score": honest_score,
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

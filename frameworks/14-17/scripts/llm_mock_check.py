#!/usr/bin/env python3
"""Judge a Harbor LLM run that was driven by scripts/mock_llm_server.py.

What a correct pipeline must show, per scenario:
  every scenario   one trial; 5 rounds ran; no trial, agent or setup exception in any round;
                   the model was actually called in all 5 rounds (mock server log)
  normal, large    reward and completeness identical to the Oracle's, because the mock replays the
                   shipped solution candidates
  large            the model requests really exceeded 128 KB (otherwise the scenario tested nothing)
  invalid-round    round 2 has no candidate, the run is still scored: completeness not COMPLETE,
                   reward 0.0 -- a model failure, not an infrastructure failure

  python scripts/llm_mock_check.py --job jobs/<job> --task <task-dir> --scenario normal --mock-log LOG
"""
from __future__ import annotations

import os as _os, sys as _sys  # noqa: E401
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

ORACLE_REWARD = {
    "task-14-multiobjective-mol-design": 0.47,
    "task-15-lipophilic-sulfonamide-edit-budget": 0.33,
    "task-16-tablet-microenvironment-cqa": 1.0,
    "task-17-spherical-agglomeration": 1.0,
}
ARG_LIMIT = 131072


def _reward(block) -> float | None:
    return ((block or {}).get("rewards") or {}).get("reward")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--mock-log", type=Path, required=True)
    args = ap.parse_args()

    problems: list[str] = []
    trials = sorted(args.job.glob("*/result.json"))
    if len(trials) != 1:
        print(f"  ! expected one trial under {args.job}, found {len(trials)}")
        print("\nstatus = FAIL")
        return 1
    trial = trials[0].parent
    result = json.loads(trials[0].read_text(encoding="utf-8"))
    steps = result.get("step_results") or []

    if result.get("exception_info"):
        problems.append(f"trial exception: {json.dumps(result['exception_info'])[:400]}")
    for step in steps:
        info = step.get("exception_info")
        if info:
            name = step.get("step_name") or step.get("name")
            problems.append(f"{name}: {info.get('exception_type')}: {str(info.get('exception_message'))[:400]}")
    if len(steps) != 5:
        problems.append(f"{len(steps)} of 5 rounds ran")
    rounds = [_reward(step.get("verifier_result")) for step in steps]
    final = _reward(result.get("verifier_result"))

    details_path = trial / "steps" / "round-5" / "verifier" / "details.json"
    details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.is_file() else {}
    gate = (details.get("reward_gate") or {}).get("protocol_completeness_verdict")
    legacy = details.get("legacy_result") or {}

    calls = []
    for line in args.mock_log.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if "round" in record:
            calls.append(record)
        if "error" in record:
            problems.append(f"mock server error: {record['error']}")
    called_rounds = sorted({c["round"] for c in calls})
    if called_rounds != [1, 2, 3, 4, 5]:
        problems.append(f"model called in rounds {called_rounds}, expected 1-5")
    biggest = max((c["request_bytes"] for c in calls), default=0)

    if args.scenario in ("normal", "large"):
        want = ORACLE_REWARD[args.task]
        if final is None or abs(final - want) > 1e-9:
            problems.append(f"final reward {final}, Oracle reward is {want}")
        if gate != "COMPLETE":
            problems.append(f"completeness {gate}, expected COMPLETE")
    if args.scenario == "large" and biggest <= ARG_LIMIT:
        problems.append(f"largest model request {biggest} bytes: the >128 KB path was not exercised")
    if args.scenario == "invalid-round":
        if gate == "COMPLETE":
            problems.append("completeness COMPLETE although round 2 had no candidate")
        if final != 0.0:
            problems.append(f"final reward {final}, expected 0.0 for an incomplete run")

    print(f"  task        : {args.task}   scenario: {args.scenario}")
    print(f"  rounds      : {len(steps)} ran; rewards {rounds}; final {final}")
    print(f"  grader      : score {legacy.get('score')}  codes {legacy.get('failure_codes')}  completeness {gate}")
    print(f"  model calls : {len(calls)} in rounds {called_rounds}; largest request {biggest} bytes")
    for problem in problems:
        print(f"  ! {problem}")
    print(f"\nstatus = {'PASS' if not problems else 'FAIL'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

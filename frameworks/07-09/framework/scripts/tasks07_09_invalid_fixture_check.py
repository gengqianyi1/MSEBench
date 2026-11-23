#!/usr/bin/env python3
"""Replay the published malformed case-result fixtures through each private grader."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    7: "task-07-qed-molecular-design",
    8: "task-08-xtb-failure-diagnosis-restart",
    9: "task-09-adaptive-xtb-torsion-profile",
}


def load_grader(number: int, path: Path):
    for module_name in list(sys.modules):
        if module_name == "evaluator" or module_name.startswith("evaluator."):
            del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(f"task{number:02d}_invalid_grader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load grader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_grader(number: int, work: Path, evidence: Path, logs: Path) -> dict:
    task = ROOT / "tasks" / TASKS[number]
    names = (
        "CHEMBENCH_WORKSPACE",
        "CHEMBENCH_LOGS",
        "CHEMBENCH_ORGANIZER_EVIDENCE",
        "CHEMBENCH_ORACLE_VALIDATION",
    )
    old = {name: os.environ.get(name) for name in names}
    os.environ.update(
        {
            "CHEMBENCH_WORKSPACE": str(work),
            "CHEMBENCH_LOGS": str(logs),
            "CHEMBENCH_ORGANIZER_EVIDENCE": str(evidence),
            "CHEMBENCH_ORACLE_VALIDATION": "1",
        }
    )
    try:
        # Grader path constants are intentionally bound at import time.
        grader = load_grader(number, task / "tests" / "protocol_grader.py")
        grader.main()
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    result = json.loads((logs / "grader-result.json").read_text(encoding="utf-8"))
    if result.get("status") != "FAIL" or not result.get("failure_codes"):
        raise AssertionError((number, result))
    if (logs / "reward.txt").exists():
        raise AssertionError(f"task {number}: invalid fixture emitted numeric reward")
    return result


def main() -> int:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="chembench-invalid-fixtures-") as raw:
        temporary = Path(raw)
        for number, name in TASKS.items():
            task = ROOT / "tasks" / name
            fixture_root = task / "tests" / "invalid_fixtures"
            manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
            source_work = ROOT / "release_evidence" / f"task{number:02d}" / "oracle_submission"
            evidence = ROOT / "release_evidence" / f"task{number:02d}" / "organizer_evidence"
            for item in manifest["fixtures"]:
                scenario = temporary / f"task{number:02d}-{Path(item['file']).stem}"
                work = scenario / "workspace"
                logs = scenario / "logs"
                shutil.copytree(source_work, work)
                target = work / "cases" / "case-1" / "case_result.json"
                shutil.copyfile(fixture_root / item["file"], target)
                result = run_grader(number, work, evidence, logs)
                if not any(item["expected_failure"] in code for code in result["failure_codes"]):
                    raise AssertionError((number, item, result["failure_codes"]))
                rows.append(
                    {
                        "task": number,
                        "fixture": item["file"],
                        "expected_failure": item["expected_failure"],
                        "status": "PASS",
                        "numeric_reward_emitted": False,
                    }
                )
    print(json.dumps({"status": "PASS", "fixture_count": len(rows), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

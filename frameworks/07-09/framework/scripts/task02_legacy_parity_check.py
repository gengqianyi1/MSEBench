#!/usr/bin/env python3
"""Same-model Task 02 parity evidence by replaying the legacy private scorer.

The recorded Harbor job used DeepSeek-V4-Flash and produced evidence ID
``AAICWp6IeRUg1W0YFKejIOZcFhw2m_-F``. This script replays the legacy scorer on
that workspace and compares it to the Harbor job raw score.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "task-02-oligothiophene-smd-sapt"
SOURCE = TASK / "tests" / "task_snapshot" / "input" / "original" / "src"
WORKSPACE = (
    ROOT
    / "parity"
    / "formal_task02_evidence"
    / "AAICWp6IeRUg1W0YFKejIOZcFhw2m_-F"
    / "task_02_oligothiophene_smd_sapt"
    / "workspace"
)
PRIVATE_REFERENCE = TASK / "tests" / "task_snapshot" / "private_data" / "scoring_reference.json"
TASK_CONFIG = TASK / "tests" / "task_snapshot" / "input" / "original" / "tasks" / "oligothiophene_smd_sapt_v3" / "task_config.yaml"
JOB_RESULT = ROOT / "jobs_fresh_task02f" / "fresh-task02f" / "result.json"
OUTPUT = ROOT / "parity" / "task02_same_model_parity.json"


def main() -> int:
    sys.path.insert(0, str(SOURCE))
    import yaml
    from instance_generator import prepare_runtime_instance
    from scoring import score_rules

    if not WORKSPACE.is_dir() or not PRIVATE_REFERENCE.is_file() or not JOB_RESULT.is_file():
        print(json.dumps({"status": "FAIL", "error": "parity evidence missing"}))
        return 1

    job = json.loads(JOB_RESULT.read_text(encoding="utf-8"))
    metrics = []
    for agent_name, stats in (job.get("stats", {}).get("evals") or {}).items():
        for metric in stats.get("metrics") or []:
            metrics.append((agent_name, metric))
    if not metrics:
        print(json.dumps({"status": "FAIL", "error": "no job metrics"}))
        return 1
    harbor_score = float(metrics[0][1]["raw_score"])
    result = json.loads((WORKSPACE / "result.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    reference = json.loads(PRIVATE_REFERENCE.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="task02-parity-") as directory:
        native_root = Path(directory) / "native"
        native_root.mkdir()
        native_workspace = native_root / "workspace"
        shutil.copytree(WORKSPACE, native_workspace)
        _, truth = prepare_runtime_instance(cfg, native_workspace)
        truth["scoring_reference"] = reference
        rule = score_rules(native_root, native_workspace, result, instance_truth=truth, task_cfg=cfg)

    legacy_score = float(rule["score"])
    ok = math.isclose(harbor_score, legacy_score, abs_tol=1.0e-9)
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "PASS" if ok else "FAIL",
                "agent": metrics[0][0],
                "harbor_job_raw_score": harbor_score,
                "legacy_replay_score": legacy_score,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS" if ok else "FAIL", "agent": metrics[0][0], "harbor_job_raw_score": harbor_score, "legacy_replay_score": legacy_score}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

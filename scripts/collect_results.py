#!/usr/bin/env python3
"""Aggregate run evidence from all three families into experiments/index.csv.

Sources:
  chembench: $RUNS/taskNN/<mode>/<attempt>/deployment_record.json
  harbor:    $RUNS/**/jobs/*/result.json                 (Harbor job results)
  organizer: $ORG_RUNTIME/taskNN/*/output/run_audit.json (ext4 via _organizer/runtime symlink)

Usage: python3 scripts/collect_results.py [--runs-root PATH] [-o experiments/index.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _reward_from_job(result_path: Path):
    try:
        for reward in result_path.parent.rglob("reward.txt"):
            return float(reward.read_text().strip())
    except (ValueError, OSError):
        return None
    return None


def chembench_rows(runs: Path):
    for rec in sorted(runs.glob("task*/**/deployment_record.json")):
        try:
            data = json.loads(rec.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        yield {"family": "chembench",
               "task": rec.relative_to(runs).parts[0].removeprefix("task"),
               "mode": data.get("mode", ""),
               "score": data.get("reward"),
               "status": "PASS" if data.get("regression_passed") else "FAIL",
               "evidence": str(rec.parent), "at": ""}


def harbor_rows(runs: Path):
    for result in sorted(runs.glob("**/jobs/*/result.json")):
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats = data.get("stats") or {}
        yield {"family": "harbor", "task": "", "mode": "",
               "score": _reward_from_job(result),
               "status": "PASS" if stats.get("n_errored_trials") == 0 else "FAIL",
               "evidence": str(result.parent), "at": data.get("finished_at", "")}


def organizer_rows(runtime: Path):
    for audit in sorted(runtime.glob("task*/**/output/run_audit.json")):
        try:
            data = json.loads(audit.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        yield {"family": "organizer", "task": str(data.get("task", "")),
               "mode": data.get("mode", ""), "score": data.get("score"),
               "status": data.get("status", ""), "evidence": str(audit.parent), "at": ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path,
                        default=Path(os.getenv("MSEBENCH_RUNS_ROOT", str(ROOT / "runs"))))
    parser.add_argument("--org-runtime", type=Path,
                        default=ROOT / "frameworks/07-09/runtime")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path(os.getenv("MSEBENCH_RUNS_ROOT", str(ROOT / "runs"))) / "experiments" / "index.csv")
    args = parser.parse_args()
    rows = list(chembench_rows(args.runs_root)) + list(harbor_rows(args.runs_root)) \
        + list(organizer_rows(args.org_runtime))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["family", "task", "mode", "score", "status", "evidence", "at"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

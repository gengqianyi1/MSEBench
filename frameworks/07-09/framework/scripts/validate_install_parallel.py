#!/usr/bin/env python3
"""Run Harbor install-only validation for every task with bounded concurrency.

This Windows-side helper avoids asking Harbor to recursively hash the complete
dataset over DrvFS before it can start any trial. Each task is submitted as an
independent local path, so failures and timings remain attributable per task.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path


def run_task(task: Path, jobs_dir: Path, distro: str) -> dict[str, object]:
    started = time.time()
    job_name = f"{task.name}_install"
    log_dir = jobs_dir / "install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.name}.log"
    linux_task = "/mnt/d/" + task.as_posix().split("D:/", 1)[-1]
    linux_jobs = "/mnt/d/" + jobs_dir.as_posix().split("D:/", 1)[-1]
    command = [
        "wsl.exe", "-d", distro, "-u", "root", "--", "env",
        "PATH=/opt/chembench-harbor/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH=/mnt/d/ChemBench/workspace/MatBench_Harbor_Combined_20260902_independent",
        "harbor", "run", "--path", linux_task, "--agent", "nop",
        "--install-only", "--n-concurrent", "1", "--jobs-dir", linux_jobs,
        "--job-name", job_name, "--yes",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, errors="replace")
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "task": task.name,
        "exit_code": completed.returncode,
        "elapsed_sec": round(time.time() - started, 1),
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_dir", type=Path)
    parser.add_argument("jobs_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--distro", default="Ubuntu-24.04")
    args = parser.parse_args()
    tasks = sorted(p for p in args.tasks_dir.glob("task-*") if p.is_dir())
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, task, args.jobs_dir, args.distro): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: str(item["task"]))
    summary = args.jobs_dir / "install_validation_summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(item["exit_code"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

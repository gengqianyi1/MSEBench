#!/usr/bin/env python3
"""Consolidate every family's COMPLETE job content + verdict files into
results/taskNN/ + results/index.csv.

Runtime stays on ext4 on WSL2 (frozen-grader os.replace semantics); after each task
completes this copies the FULL job directory into results/taskNN/run/ and the small
verdict files (deployment/run_audit/grader-result/reward) to results/taskNN/ root.

Idempotent: safe to run repeatedly while the evaluation is in progress.
Usage: python3 scripts/consolidate_results.py
"""
from __future__ import annotations

import csv
import glob as globmod
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("MSEBENCH_RUNS_ROOT", str(ROOT / "runs")))
DEFAULT_RESULTS = ROOT / "results"
OUT = Path(os.environ.get("MSEBENCH_RESULTS_ROOT", str(DEFAULT_RESULTS)))
IS_MAIN = OUT.resolve() == DEFAULT_RESULTS.resolve()

FIELDS = ["task", "family", "score", "reward", "status", "grader", "evidence",
          "scientific_reward", "completeness", "llm_request_gate", "score_delivery"]


def _newest(glob_pattern: str) -> Path | None:
    matches = [Path(p) for p in globmod.glob(glob_pattern)]
    try:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    return matches[0] if matches else None


def _newest_where(glob_pattern: str, predicate) -> Path | None:
    matches = [Path(p) for p in globmod.glob(glob_pattern)]
    try:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    for p in matches:
        try:
            if predicate(p):
                return p
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return None


def _copy(src: Path | None, dst: Path) -> None:
    if not src:
        return
    try:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except OSError:
        pass


def _chown_tree(path: Path) -> None:
    try:
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{path}:{path}",
             "matbench/harbor-runner:20260906", "chown", "-R",
             f"{os.getuid()}:{os.getgid()}", str(path)],
            capture_output=True, timeout=300,
        )
    except Exception:
        pass


def _copy_full(src, task_dst: Path) -> None:
    if not src:
        return
    src = Path(src)
    run_dir = task_dst / "run"
    marker = task_dst / "run.source"
    try:
        if run_dir.exists() and marker.exists() and marker.read_text(encoding="utf-8").strip() == str(src):
            return
        if src.is_dir():
            _chown_tree(src)
            task_dst.mkdir(parents=True, exist_ok=True)
            if run_dir.exists() and marker.exists():
                import datetime as _dt
                stamp = _dt.datetime.fromtimestamp(marker.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
                hist = task_dst / "_history" / stamp
                hist.mkdir(parents=True, exist_ok=True)
                for rec in sorted(task_dst.glob("*.json")):
                    if not (hist / rec.name).exists():
                        shutil.copy2(rec, hist / rec.name)
                if not (hist / "run.source").exists():
                    shutil.copy2(marker, hist / "run.source")
            if run_dir.exists():
                shutil.rmtree(run_dir)

            def _ignore(dirpath, names):
                if Path(dirpath).resolve() == src.resolve():
                    return [n for n in names if n.startswith("task-")]
                return []

            shutil.copytree(src, run_dir, ignore=_ignore)
            marker.write_text(str(src), encoding="utf-8")
    except OSError:
        pass


def harbor(n: int) -> dict:
    row = {"task": f"{n:02d}", "family": "harbor", "score": None, "reward": None,
           "status": "?", "grader": None, "evidence": None, "full_src": None}
    src_dir = _newest(str(RUNS / f"task{n}-*")) or _newest(str(ROOT / f"results/task{n}-*"))
    if not src_dir:
        return row
    # 2026-09-21 (口径修正): the LATEST run directory is authoritative. Its
    # newest trial is reported as-is: scored -> score, exception/verifier_error
    # -> INVALID, no verdict -> blank. A failed rerun is a visible failure,
    # never silently replaced by an older run's score.
    trial = _newest(str(src_dir / "model_run" / "jobs" / "*" / f"task-0{n}-*" / "result.json"))
    if trial:
        model_run = trial.parent.parent.parent.parent
        try:
            data = json.loads(trial.read_text())
        except (OSError, json.JSONDecodeError):
            return row
        vr = (data.get("verifier_result") or {}).get("rewards") or {}
        row["reward"] = vr.get("reward")
        row["score"] = vr.get("raw_score") if vr.get("raw_score") is not None else vr.get("final_score")
        if data.get("exception_info") is not None:
            # The verifier still scores a trial whose agent crashed (the
            # 2026-09-16 lesson): that 0 is an infrastructure artifact, not
            # a model score. Keep the run out of the scored index.
            row["status"] = "INVALID"
            row["reward"] = None
            row["score"] = None
            row["score_delivery"] = "not-a-model-score"
        else:
            row["status"] = "PASS" if vr.get("reward") is not None else "FAIL"
            details = trial.parent / "verifier" / "details.json"
            dv = {}
            if details.is_file():
                try:
                    dv = json.loads(details.read_text())
                except (OSError, json.JSONDecodeError):
                    dv = {}
            if dv:
                row["llm_request_gate"] = dv.get("llm_request_gate")
                row["scientific_reward"] = dv.get("scientific_score")
            if row["status"] == "PASS" and dv.get("status") == "verifier_error":
                # Grader itself failed (task01 2026-09-19: model never
                # submitted, so the sidecar never exported the control-plane
                # accounting record and the grader failed closed). The 0.0
                # written afterwards is a fallback, not a scientific score.
                row["status"] = "INVALID"
                row["reward"] = None
                row["score"] = None
            if row["status"] == "INVALID":
                row["score_delivery"] = "not-a-model-score"
        row["evidence"] = str(model_run)
        row["full_src"] = str(model_run)
        dst = OUT / f"task{n:02d}"
        _copy(trial, dst / "trial_result.json")
        _copy(_newest(str(model_run / f"task0{n}_llm_calls.jsonl")), dst / "llm_calls.jsonl")
    return row


def chembench(n: int, runs: Path = RUNS) -> dict:
    attempt = _newest_where(
        str(runs / f"task{n:02d}" / "model" / "*"),
        lambda p: json.loads((p / "deployment_record.json").read_text()).get("mode") == "model"
        and json.loads((p / "deployment_record.json").read_text()).get("reward") is not None,
    )
    row = {"task": f"{n:02d}", "family": "chembench", "score": None, "reward": None,
           "status": "?", "grader": None, "evidence": None, "full_src": None}
    if not attempt:
        return row
    rec = attempt / "deployment_record.json"
    try:
        if not rec.is_file():
            return row
        d = json.loads(rec.read_text())
    except (OSError, json.JSONDecodeError):
        return row
    row["reward"] = d.get("reward")
    row["score"] = d.get("reward")
    exit_code = d.get("harbor_exit_code")
    if exit_code is not None and exit_code != 0:
        # harbor exited non-zero (agent/sidecar crash): the reward the verifier
        # wrote afterwards is not a model measurement.
        row["status"] = "INVALID"
        row["reward"] = None
        row["score"] = None
    else:
        row["status"] = "PASS" if d.get("regression_passed") else "FAIL"
    row["evidence"] = str(attempt)
    row["full_src"] = str(attempt)
    dst = OUT / f"task{n:02d}"
    _copy(rec, dst / "deployment_record.json")
    grader = _newest(str(attempt / "jobs" / "*" / "task-*" / "verifier" / "grader-result.json"))
    if grader:
        row["grader"] = str(grader)
        _copy(grader, dst / "grader-result.json")
    return row


def organizer(n: int) -> dict:
    env_runtime = os.environ.get("CHEMBENCH_RUNTIME_ROOT")
    runtime = Path(env_runtime) if env_runtime else (ROOT / "frameworks/07-09/runtime")
    if runtime.is_symlink():
        runtime = Path(os.path.realpath(runtime))
    audit = _newest_where(
        str(runtime / f"task0{n}" / "*" / "output" / "run_audit.json"),
        lambda p: json.loads(p.read_text()).get("mode") == "live_llm",
    )
    row = {"task": f"{n:02d}", "family": "organizer", "score": None, "reward": None,
           "status": "?", "grader": None, "evidence": None, "full_src": None}
    if not audit:
        return row
    d = json.loads(audit.read_text())
    row["score"] = d.get("score")
    row["reward"] = d.get("reward")
    row["status"] = d.get("status")
    if d.get("errors"):
        # provider/sidecar errors recorded by the run audit: the score is not
        # a clean model measurement either.
        row["status"] = "INVALID"
        row["reward"] = None
        row["score"] = None
    row["evidence"] = str(audit.parent)
    row["full_src"] = str(audit.parent)
    dst = OUT / f"task{n:02d}"
    _copy(audit, dst / "run_audit.json")
    grader = _newest(str(audit.parent / "jobs" / "*" / "*" / "steps" / "*" / "verifier" / "grader-result.json"))
    if grader:
        row["grader"] = str(grader)
        _copy(grader, dst / "grader-result.json")
    return row


def textqa() -> dict:
    src = _newest(str(OUT / "textqa" / "*"))
    row = {"task": "textqa", "family": "textqa", "score": None, "reward": None,
           "status": "?", "grader": None, "evidence": None, "full_src": None}
    if not src:
        return row
    summary = src / "summary.json"
    if summary.is_file():
        d = json.loads(summary.read_text())
        row["score"] = d.get("accuracy")
        row["reward"] = d.get("accuracy")
        errors = d.get("errors") or 0
        total = d.get("total")
        if total and errors >= total:
            row["status"] = "INVALID"  # every call failed: no model measurement
        else:
            row["status"] = "PASS" if errors == 0 else "FAIL"
        row["evidence"] = str(src)
    return row


def task1920(n: int) -> dict:
    # 2026-09-28 fix: scan every campaign under the task1920 root and take the newest cell.
    # The old hard-coded glm_lowthink default silently ignored newer kimi campaigns.
    root = Path(os.environ.get("MSEBENCH_TASK1920_ROOT", str(RUNS / "task1920")))
    cells = sorted(root.glob(f"*/*-task{n}"), key=lambda p: p.stat().st_mtime, reverse=True)
    cell = cells[0] if cells else None
    row = {"task": f"{n:02d}", "family": "chembench", "score": None, "reward": None,
           "status": "?", "grader": None, "evidence": None, "full_src": None}
    if not cell:
        return row
    result = cell / "result.json"
    if result.is_file():
        d = json.loads(result.read_text())
        score = d.get("score")
        row["score"] = score
        row["reward"] = (score / 100.0) if isinstance(score, (int, float)) else None
        if d.get("status") == "COMPLETED":
            row["status"] = "PASS"
        elif "INFRASTRUCTURE" in str(d.get("status") or ""):
            # INFRASTRUCTURE_FAILURE / EVIDENCE_OR_INFRASTRUCTURE_FAILURE:
            # not a model score (INVALID_SUBMISSION stays a real model 0).
            row["status"] = "INVALID"
            row["reward"] = None
            row["score"] = None
        else:
            row["status"] = "FAIL"
        row["evidence"] = str(cell)
        row["full_src"] = str(cell)
        dst = OUT / f"task{n:02d}"
        _copy(result, dst / "result.json")
    return row


_T1417 = {"14": "task-14-multiobjective-mol-design", "15": "task-15-lipophilic-sulfonamide-edit-budget",
          "16": "task-16-tablet-microenvironment-cqa", "17": "task-17-spherical-agglomeration"}


def task1417(n: int) -> dict:
    job = _newest(str(ROOT / "frameworks/14-17" / "jobs" / f"llm-{_T1417[str(n)]}-*"))
    row = {"task": f"{n:02d}", "family": "chembench", "score": None, "reward": None,
           "status": "?", "grader": None, "evidence": None, "full_src": None}
    if not job:
        return row
    trial = _newest(str(job / "*/result.json"))
    if not trial:
        return row
    try:
        r = json.loads(trial.read_text())
    except (OSError, json.JSONDecodeError):
        return row
    reward = (r.get("verifier_result") or {}).get("rewards", {}).get("reward")
    score = None
    details = trial.parent / "steps" / "round-5" / "verifier" / "details.json"
    if not details.is_file():
        details = trial.parent / "verifier" / "details.json"
    gate = {}
    if details.is_file():
        try:
            d = json.loads(details.read_text())
            score = (d.get("legacy_result") or {}).get("score")
            gate = d.get("reward_gate") or {}
        except (OSError, json.JSONDecodeError):
            pass
    exc = (r.get("exception_info") or {}).get("exception_type")
    verdict = gate.get("protocol_completeness_verdict")
    if exc is not None:
        # agent/harness exception: the verifier still writes a reward, but the
        # run never measured the model (14-17 2026-09-16 lesson).
        row["status"] = "INVALID"
    elif verdict is not None and verdict != "COMPLETE":
        # grader rule (public task_config.yaml): an incomplete trajectory is
        # NOT rankable and its reward is forced to 0.0 by design. Record the
        # run as INVALID instead of leaking that structural 0 into the index;
        # the diagnostic scientific score stays in details.json.
        row["status"] = "INVALID"
    else:
        row["status"] = "PASS" if reward is not None else "FAIL"
    if row["status"] == "INVALID":
        row["score"] = None
        row["reward"] = None
    else:
        row["score"] = score
        row["reward"] = reward
    row["evidence"] = str(job)
    row["full_src"] = str(job)
    row["scientific_reward"] = gate.get("scientific_reward")
    row["completeness"] = verdict
    if row["status"] == "INVALID":
        row["score_delivery"] = "not-a-model-score"
    _copy(details, OUT / f"task{n:02d}" / "details.json")
    return row


def main() -> int:
    rows: list[dict] = []
    for n in (1, 2, 3):
        rows.append(harbor(n))
    for n in (4, 5, 6, 18):
        rows.append(chembench(n))
    for n in (7, 8, 9):
        rows.append(organizer(n))
    for n in (10, 11, 12, 13):
        rows.append(chembench(n))
    if IS_MAIN:
        for n in (14, 15, 16, 17):
            rows.append(task1417(n))
        for n in (19, 20):
            rows.append(task1920(n))
        rows.append(textqa())

    for r in rows:
        fs = r.pop("full_src", None)
        for key in ("scientific_reward", "completeness", "llm_request_gate", "score_delivery"):
            r.setdefault(key, None)
        _copy_full(fs, OUT / f"task{r['task']}")

    idx = OUT / "index.csv"
    with idx.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    done = sum(1 for r in rows if r["reward"] is not None or r["status"] not in ("?",))
    invalid = [r for r in rows if r.get("status") == "INVALID"]
    for r in invalid:
        print(f"INVALID (not a model score; zero suppressed): task {r['task']} "
              f"family {r['family']} evidence {r.get('evidence')}")
    print(f"consolidated {done}/{len(rows)} tasks -> {idx} "
          f"({len(invalid)} INVALID run(s) with score/reward blanked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

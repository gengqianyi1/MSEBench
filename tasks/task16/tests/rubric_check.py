#!/usr/bin/env python3
"""Execute the task's own public rubric.json against the rebuilt submission surface.

`environment/public/rubric.json` declares the STRUCTURAL contract: which files a finished
submission must contain, and which fields inside them. Every check carries 0 points -- the
scientific score is entirely the private grader's job. Its purpose is to separate an
INVALID_SUBMISSION from a submission that was graded and scored zero.

Until 2026-09-10 nothing in the chain executed it. `tests/test.sh` ran final_grader.py and
run_legacy_grader.py, neither of which reads rubric.json, so the public contract and the real
pipeline were free to drift -- and had: all four tasks declared `final_answer_declares_task`
CRITICAL while tasks 16/17 wrote `final_answer.json` without `task_id`, and tasks 14/15 never
wrote the file at all. A round-5 reviewer found the task-17 half of that by reading; this script
exists so the next drift is caught by running.

It runs BEFORE the legacy grader and is advisory by design: it prints its verdict and always
exits 0, because turning a structural mismatch into a non-zero exit would convert a submission
problem into a Harbor verifier_error -- an infrastructure failure -- which is precisely the
misattribution Evaluation §16.5 forbids. The failing check names are written to
/logs/verifier/rubric_check.json for the record, and `--strict` (used by the author-side
regression, never by the container) makes it exit non-zero instead.

    python /tests/rubric_check.py            # in the verifier, advisory
    python tests/rubric_check.py --strict    # author-side CI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORK = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
TASK = Path(os.environ.get("CHEMBENCH_TASK_ROOT", "/task"))
LOGS = Path(os.environ.get("CHEMBENCH_LOGS", "/logs/verifier"))


def _find_rubric(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for candidate in (TASK / "rubric.json", TASK / "input" / "rubric.json"):
        if candidate.is_file():
            return candidate
    # author-side layout: the public copy next to the task directory
    here = Path(__file__).resolve()
    for parent in here.parents:
        p = parent / "environment" / "public" / "rubric.json"
        if p.is_file():
            return p
    return None


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError):
        return None


def _run_check(check: dict) -> tuple[bool, str]:
    kind = check.get("type")
    target = WORK / str(check.get("path", ""))
    if kind == "file_exists":
        return target.is_file(), f"{check.get('path')} present"
    if kind == "field_present":
        doc = _load(target)
        if not isinstance(doc, dict):
            return False, f"{check.get('path')} unreadable"
        missing = [k for k in (check.get("keys") or []) if k not in doc]
        return not missing, f"missing field(s) {missing}" if missing else "fields present"
    if kind == "count_in_range":
        doc = _load(target)
        if not isinstance(doc, dict):
            return False, f"{check.get('path')} unreadable"
        for key in (check.get("keys") or []):
            value = doc.get(key)
            if not isinstance(value, (list, tuple)):
                return False, f"{key} is not a list"
            low = check.get("min_count", 0)
            high = check.get("max_count", len(value))
            if not (low <= len(value) <= high):
                return False, f"{key} has {len(value)}, expected {low}..{high}"
        return True, "counts in range"
    # An unknown check type must not silently pass -- that is how a contract rots.
    return False, f"unsupported check type {kind!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    rubric_path = _find_rubric(args.rubric)
    if rubric_path is None:
        print(json.dumps({"status": "SKIP", "reason": "rubric.json not found"}))
        return 1 if args.strict else 0
    rubric = _load(rubric_path)
    if not isinstance(rubric, dict):
        print(json.dumps({"status": "SKIP", "reason": f"unreadable rubric {rubric_path}"}))
        return 1 if args.strict else 0

    results, failed_critical = [], []
    for dimension, body in (rubric.get("dimensions") or {}).items():
        for check in (body.get("checks") or []):
            ok, detail = _run_check(check)
            row = {"dimension": dimension, "name": check.get("name"),
                   "critical": bool(check.get("critical")), "ok": ok, "detail": detail}
            results.append(row)
            if not ok and row["critical"]:
                failed_critical.append(check.get("name"))

    report = {
        "status": "PASS" if not failed_critical else "STRUCTURAL_CONTRACT_VIOLATION",
        "rubric": str(rubric_path),
        "rubric_version": rubric.get("version"),
        "failed_critical": failed_critical,
        "checks": results,
        "note": ("Advisory: a structural mismatch is a submission problem, not an infrastructure "
                 "failure, so this never fails the verifier by itself. The private grader still "
                 "scores the run and run_legacy_grader.py still writes the reward."),
    }
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "rubric_check.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    print(json.dumps({k: report[k] for k in ("status", "rubric_version", "failed_critical")},
                     ensure_ascii=False))
    return (1 if failed_critical else 0) if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tasks 14-17: run the Oracle and every invalid submission through the real verifier chain.

`harness_contract.yaml :: ownership.author_required` asks each of these four tasks for "one valid
Oracle and at least three invalid submissions". The Oracle existed; the invalid submissions did not,
for any task in the repository. This script supplies them and checks them the only way that means
anything -- through the same chain the Harbor verifier runs.

WHAT CHANGED ON 2026-09-04, AND WHY IT MATTERS
----------------------------------------------
The first version of this script hardcoded the pipeline:

    final_grader.py  ->  private_grader/grader.py

That is what tasks 14 and 15 do. It is NOT what tasks 16 and 17 did: their `tests/test.sh` ran only
`run_legacy_grader.py` and never called `final_grader.py`, so `/workspace/trajectory.jsonl` was
never built and every Harbor round scored 0.0 with `MISSING_TRAJECTORY`. This script reported
24/24 PASS through all of it, because it ran a pipeline the verifier does not run. A checker that
assumes the wiring cannot detect broken wiring.

So the pipeline is now READ OUT OF `tests/test.sh` (`_pipeline_from_test_sh`) and executed in that
order. `run_legacy_grader.py` is the one step emulated in-process rather than executed, because it
addresses container-absolute paths (`/task`, `/workspace`, `/logs/verifier`, `/run`); the script
asserts it is the last step so that everything before it is genuinely reproduced.

`--self-test` re-runs each Oracle with the `final_grader.py` step deleted from the pipeline and
requires the score to change. That is the 2026-09-03 defect, replayed: a check that cannot fail is
not evidence.

    python scripts/task14_17_oracle_and_invalid_check.py
    python scripts/task14_17_oracle_and_invalid_check.py --task 16
    python scripts/task14_17_oracle_and_invalid_check.py --self-test
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    "task-14-multiobjective-mol-design",
    "task-15-lipophilic-sulfonamide-edit-budget",
    "task-16-tablet-microenvironment-cqa",
    "task-17-spherical-agglomeration",
]
# `python /tests/foo.py` / `python3 /tests/foo.py`, one per line, which is the whole grammar these
# four test.sh files use. Anything richer than that must not be silently ignored -- see _pipeline.
_STEP = re.compile(r"^\s*python3?\s+/tests/([A-Za-z0-9_./-]+\.py)\s*$")
_IGNORABLE = re.compile(r"^\s*(#.*)?$|^\s*(set|echo|cd|export|shopt)\b|^#!")
GRADER_STEP = "run_legacy_grader.py"


def _pipeline_from_test_sh(task_dir: Path) -> list[str]:
    """The scripts `tests/test.sh` actually runs, in order.

    Raises on any line this parser does not understand, rather than skipping it: an unrecognised
    line could be another grading step, and silently dropping it would reintroduce exactly the
    assume-the-wiring bug this function exists to prevent.
    """
    path = task_dir / "tests" / "test.sh"
    steps: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _STEP.match(line)
        if match:
            steps.append(match.group(1))
        elif not _IGNORABLE.match(line):
            raise SystemExit(f"{path}:{lineno}: unparsed command in test.sh: {line!r}\n"
                             f"  Extend _STEP/_IGNORABLE -- do not let this check skip a "
                             f"grading step it does not understand.")
    if not steps or steps[-1] != GRADER_STEP:
        raise SystemExit(f"{path}: expected {GRADER_STEP} to be the last step, got {steps!r}")
    return steps


def _purge_task_modules(snapshot: Path) -> None:
    """Drop modules imported from a previous task's snapshot.

    All four tasks ship same-named modules (`evaluator`, `scoring`, `utils`, `harness`,
    `response_model`). Grading them in one process means the first task to import wins `sys.modules`
    and every later grader silently reads the wrong task's code.
    """
    root = str(snapshot.resolve())
    for name, module in list(sys.modules.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin and origin != "built-in" and str(Path(origin).resolve()).startswith(root):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not str(Path(p).resolve()).startswith(root)]
    importlib.invalidate_caches()


def grade_workspace(task: str, rounds: dict[int, str], pipeline: list[str] | None = None) -> dict:
    """Run one submission through the pipeline `tests/test.sh` declares."""
    task_dir = ROOT / "tasks" / task
    snapshot = task_dir / "tests" / "task_snapshot"
    if pipeline is None:
        pipeline = _pipeline_from_test_sh(task_dir)
    with tempfile.TemporaryDirectory(prefix="chembench-fixture-") as tmp:
        work = Path(tmp) / "workspace"
        (work / "rounds").mkdir(parents=True)
        for n, text in rounds.items():
            d = work / "rounds" / f"round-{n}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "candidate.json").write_text(text, encoding="utf-8")

        for step in pipeline[:-1]:
            proc = subprocess.run(
                [sys.executable, str(task_dir / "tests" / step)],
                # Task 15's final_grader reads the task root from the environment; the verifier
                # image sets it to /task. Reproduce that rather than relying on the default.
                env=dict(os.environ, CHEMBENCH_WORKSPACE=str(work),
                         CHEMBENCH_TASK_ROOT=str(snapshot)),
                capture_output=True, text=True)
            if proc.returncode != 0:
                return {"raised": True, "error_type": f"{step}_failed",
                        "error": proc.stderr.strip()[-400:], "pipeline": pipeline}

        # pipeline[-1] is run_legacy_grader.py, emulated here: it resolves /task, /workspace,
        # /logs/verifier and /run, which only exist inside the verifier image.
        meta = yaml.safe_load((snapshot / "task.yaml").read_text(encoding="utf-8"))
        _purge_task_modules(snapshot)
        spec = importlib.util.spec_from_file_location(
            f"legacy_grader_{task.replace('-', '_')}", snapshot / meta["grader_path"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        run = Path(tmp) / "run"
        (run / "submission").mkdir(parents=True)
        (run / "logs").mkdir(parents=True)
        try:
            spec.loader.exec_module(module)
            result = getattr(module, meta.get("grader_entrypoint", "grade"))(SimpleNamespace(
                task_dir=snapshot, submission_dir=work, run_dir=run,
                private_data_dir=snapshot / meta.get("private_data_path", "private_data")))
        except Exception as exc:            # noqa: BLE001 -- the point is to report, not to crash
            return {"raised": True, "error_type": type(exc).__name__, "error": str(exc)[:400],
                    "pipeline": pipeline}
        finally:
            _purge_task_modules(snapshot)
        completeness = (result.get("evidence") or {}).get("protocol_completeness") or {}
        return {
            "raised": False,
            "score": result.get("score"),
            "failure_codes": result.get("failure_codes"),
            "protocol_completeness": completeness.get("verdict"),
            "rounds_with_a_candidate": completeness.get("rounds_with_a_candidate"),
            "pipeline": pipeline,
        }


def _oracle_rounds(task_dir: Path) -> dict[int, str]:
    return {n: (task_dir / "steps" / f"round-{n}" / "solution" / "candidate.json")
            .read_text(encoding="utf-8") for n in range(1, 6)}


def _expected_score(task_dir: Path):
    path = task_dir / "tests" / "invalid_submissions" / "ORACLE_EXPECTED.json"
    return json.loads(path.read_text(encoding="utf-8"))["score"] if path.is_file() else None


def check_task(task: str) -> dict:
    task_dir = ROOT / "tasks" / task
    errors: list[str] = []
    cases: list[dict] = []
    pipeline = _pipeline_from_test_sh(task_dir)

    # ---------------------------------------------------------------- the valid Oracle
    got = grade_workspace(task, _oracle_rounds(task_dir), pipeline)
    expected_score = _expected_score(task_dir)
    if got["raised"]:
        errors.append(f"oracle raised {got['error_type']}: {got['error']}")
    elif expected_score is not None and got["score"] != expected_score:
        errors.append(f"oracle scored {got['score']}, expected {expected_score}")
    elif got["protocol_completeness"] != "COMPLETE":
        errors.append(f"oracle completeness is {got['protocol_completeness']}, expected COMPLETE")
    cases.append({"case": "ORACLE", "expected_score": expected_score, **got})

    # ---------------------------------------------------------------- the invalid submissions
    fixtures = sorted((task_dir / "tests" / "invalid_submissions").glob("*/expected.json"))
    for exp_path in fixtures:
        name = exp_path.parent.name
        expect = json.loads(exp_path.read_text(encoding="utf-8"))
        rounds = {int(p.stem.split("-")[1]): p.read_text(encoding="utf-8")
                  for p in sorted(exp_path.parent.glob("round-*.json"))}
        got = grade_workspace(task, rounds, pipeline)
        problems = []
        if expect.get("must_not_raise") and got["raised"]:
            problems.append(f"grader raised {got['error_type']}: {got['error']}")
        if not got["raised"]:
            if "score" in expect and got["score"] != expect["score"]:
                problems.append(f"score {got['score']} != expected {expect['score']}")
            if "protocol_completeness" in expect and \
                    got["protocol_completeness"] != expect["protocol_completeness"]:
                problems.append(f"completeness {got['protocol_completeness']} "
                                f"!= expected {expect['protocol_completeness']}")
            if "rounds_with_a_candidate" in expect and \
                    got["rounds_with_a_candidate"] != expect["rounds_with_a_candidate"]:
                problems.append(f"rounds_with_a_candidate {got['rounds_with_a_candidate']} "
                                f"!= expected {expect['rounds_with_a_candidate']}")
        errors.extend(f"{name}: {p}" for p in problems)
        cases.append({"case": name, "expect": expect, **got, "problems": problems})

    return {"task": task, "pipeline": pipeline, "n_invalid_fixtures": len(fixtures),
            "cases": cases, "errors": errors, "status": "PASS" if not errors else "FAIL"}


def self_test() -> int:
    """Replay the 2026-09-03 defect: drop final_grader.py and require the Oracle score to move.

    If the score survives the removal, this checker is not actually exercising `final_grader.py`
    and its PASS means nothing for the artifacts that step produces.
    """
    ok = True
    for task in TASKS:
        task_dir = ROOT / "tasks" / task
        full = _pipeline_from_test_sh(task_dir)
        crippled = [s for s in full if s != "final_grader.py"]
        if crippled == full:
            print(f"  FAIL {task}: test.sh has no final_grader.py step to remove")
            ok = False
            continue
        expected = _expected_score(task_dir)
        got = grade_workspace(task, _oracle_rounds(task_dir), crippled)
        moved = got["raised"] or got["score"] != expected
        mark = "ok  " if moved else "FAIL"
        detail = (f"RAISED {got['error_type']}" if got["raised"]
                  else f"score={got['score']} codes={got['failure_codes']}")
        print(f"  {mark} {task:<46} without final_grader -> {detail}  (expected {expected})")
        ok &= moved
    print("\nself-test " + ("PASS: the check fails when the pipeline is broken" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="check only this task number, e.g. 16")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    selected = [t for t in TASKS if not args.task or f"-{args.task}-" in t]
    if not selected:
        raise SystemExit(f"no task matches {args.task!r}")

    reports = []
    for task in selected:
        report = check_task(task)
        reports.append(report)
        print(f"=== {task}  [{report['status']}]  "
              f"{report['n_invalid_fixtures']} invalid fixture(s)")
        print(f"    test.sh pipeline: {' -> '.join(report['pipeline'])}")
        for case in report["cases"]:
            if case["raised"]:
                verdict = f"RAISED {case['error_type']}"
            else:
                verdict = (f"score={case['score']} "
                           f"completeness={case['protocol_completeness']}"
                           f"({case['rounds_with_a_candidate']}/5) "
                           f"codes={case['failure_codes']}")
            oracle_errors = [e for e in report["errors"] if e.startswith("oracle ")]
            failed = bool(case.get("problems")) or (case["case"] == "ORACLE" and oracle_errors)
            mark = "FAIL" if failed else "ok  "
            print(f"  {mark} {case['case']:<22} {verdict}")
        for e in report["errors"]:
            print(f"       ! {e}")

    overall = "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL"
    print(f"\nstatus = {overall}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"status": overall, "reports": reports}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

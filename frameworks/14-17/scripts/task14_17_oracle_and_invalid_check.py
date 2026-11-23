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

THE REFERENCE-OPTIMUM CASE (added 2026-09-06)
--------------------------------------------
The shipped Oracle is a LIFECYCLE oracle: for tasks 14 and 15 it replays a calibrated legacy model
trajectory and scores 47 and 33, which pins the runtime chain and gives the legacy-vs-Harbor parity
fixture its expected number. It deliberately does not claim scientific optimality
(`harness_contract.yaml :: author_confirmations.oracle_is_optimal_solution` records `rejected` for
task 14). Reading 33 as "the ladder tops out at 33" is the obvious misreading, and the 2026-09-04
framework handoff made exactly it.

So a second valid case is checked: the enumerated reference optimum, submitted through the same
chain, must score the top of the ladder. The candidate is BUILT AT CHECK TIME from each task's
private reference file rather than stored as a fixture -- storing it would put one more copy of the
answer in the tree and could drift from `private_data`. For tasks 16 and 17 the shipped Oracle is
already the reference design, so this case doubles as a drift check between
`steps/*/solution/candidate.json` and `private_data/`.

THE REWARD IS CHECKED, NOT ONLY THE SCORE (added 2026-09-10)
-----------------------------------------------------------
This checker used to compare the private grader's `score` and stop there. That is exactly how the
hole six of the eight round-5 reviewers found survived four rounds of review: an incomplete
trajectory produced the right score (a best-so-far lower bound) and the right failure code
(`INCOMPLETE_MULTI_ROUND`), and the checker passed -- while `run_legacy_grader.py` turned that same
lower bound into a full Harbor reward, so a 3/5 run was worth as much on a leaderboard as a 5/5 one.
The score was never the number that mattered.

So `grade_workspace` now runs `run_legacy_grader.py` for real, in a temporary container-shaped
layout, and returns the `reward.json` it writes. Fixtures declare `harbor_reward`: a number, or the
string "scientific" meaning "must equal score/100". A fixture whose completeness is not COMPLETE and
whose reward is non-zero now FAILS.

`--self-test` re-runs each Oracle with the `final_grader.py` step deleted from the pipeline and
requires the score to change; it requires the reference optimum to outscore the lifecycle Oracle
wherever the two differ; and it removes the completeness gate from `run_legacy_grader.py` and
requires an incomplete fixture's reward to become non-zero. A check that cannot fail is not
evidence.

    python scripts/task14_17_oracle_and_invalid_check.py
    python scripts/task14_17_oracle_and_invalid_check.py --task 16
    python scripts/task14_17_oracle_and_invalid_check.py --self-test
"""
from __future__ import annotations
import os as _os, sys as _sys  # noqa: E401
# Never write __pycache__ into tasks/: this script loads task code, and bytecode dropped there
# changes the task's Harbor checksum, so datasets/*/dataset.toml would stop matching the tree.
_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # inherited by every subprocess
_sys.dont_write_bytecode = True

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

# Where each task keeps its enumerated optimum, and the candidate field it goes into. These are the
# same files `task14_17_boundary_check.py` treats as the declared secret inventory.
REFERENCE_OPTIMUM = {
    "task-14-multiobjective-mol-design": (
        "private_data/grader/reference/reference_target.yaml", ("reference", "best_smiles_in_window"), "smiles"),
    "task-15-lipophilic-sulfonamide-edit-budget": (
        "private_data/grader/reference/reference_target.yaml", ("reference", "best_smiles_in_window"), "smiles"),
    "task-16-tablet-microenvironment-cqa": (
        "private_data/answers/reference_enumeration.json", ("reference", "design"), "formulation"),
    "task-17-spherical-agglomeration": (
        "private_data/answers/reference_enumeration.json", ("reference_design",), "process"),
}


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

        # pipeline[-1] is run_legacy_grader.py. It used to be emulated here; it is now EXECUTED,
        # because the reward it writes is the number that reaches a leaderboard and emulating it
        # meant the completeness gate was never exercised by this check. It addresses /task,
        # /workspace, /logs/verifier and /run, which it now takes from the environment.
        logs = Path(tmp) / "logs"
        run = Path(tmp) / "run"
        logs.mkdir(parents=True, exist_ok=True)
        (run / "submission").mkdir(parents=True, exist_ok=True)
        (run / "logs").mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, str(task_dir / "tests" / pipeline[-1])],
            env=dict(os.environ, CHEMBENCH_WORKSPACE=str(work), CHEMBENCH_TASK_ROOT=str(snapshot),
                     CHEMBENCH_LOGS=str(logs), CHEMBENCH_RUN_DIR=str(run)),
            capture_output=True, text=True)
        details_path, reward_path = logs / "details.json", logs / "reward.json"
        if not reward_path.is_file():
            return {"raised": True, "error_type": f"{pipeline[-1]}_wrote_no_reward",
                    "error": proc.stderr.strip()[-400:], "pipeline": pipeline}
        reward = json.loads(reward_path.read_text(encoding="utf-8")).get("reward")
        details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.is_file() else {}
        result = details.get("legacy_result") or {}
        if proc.returncode != 0 and not result:
            return {"raised": True, "error_type": f"{pipeline[-1]}_failed",
                    "error": proc.stderr.strip()[-400:], "pipeline": pipeline}
        completeness = (result.get("evidence") or {}).get("protocol_completeness") or {}
        rubric = logs / "rubric_check.json"
        return {
            "raised": False,
            "score": result.get("score"),
            "reward": reward,
            "failure_codes": result.get("failure_codes"),
            "protocol_completeness": completeness.get("verdict"),
            "rounds_with_a_candidate": completeness.get("rounds_with_a_candidate"),
            "reward_gate": details.get("reward_gate"),
            "rubric_status": (json.loads(rubric.read_text(encoding="utf-8")).get("status")
                              if rubric.is_file() else None),
            "pipeline": pipeline,
        }


def _reward_problems(got: dict, expect: dict) -> list[str]:
    """Compare the reward Harbor would actually receive against what the fixture declares.

    `harbor_reward` is either a number, or "scientific" meaning "must equal score/100". The second
    form exists so a fixture need not restate the scientific score, which is the private grader's
    business; what the fixture pins is the RELATIONSHIP between completeness and reward.
    """
    if "harbor_reward" not in expect:
        return []
    want, reward, score = expect["harbor_reward"], got.get("reward"), got.get("score")
    if reward is None:
        return ["no reward written"]
    if want == "scientific":
        if score is None:
            return ["expected a scientific reward but no score was produced"]
        return ([] if abs(reward - score / 100.0) < 1e-9
                else [f"reward {reward} != score/100 ({score / 100.0})"])
    if abs(reward - float(want)) > 1e-9:
        return [f"reward {reward} != expected {want}"]
    if float(want) == 0.0 and got.get("protocol_completeness") == "COMPLETE":
        return ["fixture expects reward 0 but the run is COMPLETE -- check the fixture"]
    return []


def _reference_optimum_rounds(task: str, task_dir: Path) -> dict[int, str]:
    """The enumerated optimum, as a five-round submission, read out of the task's private data."""
    rel, path, unit = REFERENCE_OPTIMUM[task]
    source = task_dir / "tests" / "task_snapshot" / rel
    raw = source.read_text(encoding="utf-8")
    doc = json.loads(raw) if source.suffix == ".json" else yaml.safe_load(raw)
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            raise SystemExit(f"{source}: reference path {'.'.join(path)} missing; schema changed?")
        doc = doc[key]
    payload = json.dumps({"candidate": {
        "name": "reference_optimum",
        "rationale": "enumerated optimum inside the specification window",
        unit: doc}}, ensure_ascii=False)
    return {n: payload for n in range(1, 6)}


def _oracle_rounds(task_dir: Path) -> dict[int, str]:
    return {n: (task_dir / "steps" / f"round-{n}" / "solution" / "candidate.json")
            .read_text(encoding="utf-8") for n in range(1, 6)}


def _expectations(task_dir: Path) -> dict:
    path = task_dir / "tests" / "invalid_submissions" / "ORACLE_EXPECTED.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _expected_score(task_dir: Path):
    return _expectations(task_dir).get("score")


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
    errors += [f"oracle: {q}" for q in _reward_problems(got, {"harbor_reward": "scientific"})]
    cases.append({"case": "ORACLE", "expected_score": expected_score, **got})

    # ------------------------------------------------- the reference optimum (top of the ladder)
    expected_top = _expectations(task_dir).get("reference_optimum_score")
    got = grade_workspace(task, _reference_optimum_rounds(task, task_dir), pipeline)
    problems = []
    if got["raised"]:
        problems.append(f"raised {got['error_type']}: {got['error']}")
    else:
        if expected_top is not None and got["score"] != expected_top:
            problems.append(f"score {got['score']} != expected {expected_top}")
        if got["failure_codes"]:
            problems.append(f"reference optimum reported failure codes {got['failure_codes']}")
        if got["protocol_completeness"] != "COMPLETE":
            problems.append(f"completeness {got['protocol_completeness']}, expected COMPLETE")
        problems.extend(_reward_problems(got, {"harbor_reward": "scientific"}))
    errors.extend(f"REFERENCE_OPTIMUM: {p}" for p in problems)
    cases.append({"case": "REFERENCE_OPTIMUM", "expected_score": expected_top,
                  **got, "problems": problems})

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
            problems.extend(_reward_problems(got, expect))
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

    print()
    for task in TASKS:
        task_dir = ROOT / "tasks" / task
        full = _pipeline_from_test_sh(task_dir)
        lifecycle = _expected_score(task_dir)
        top = grade_workspace(task, _reference_optimum_rounds(task, task_dir), full)
        # Where the lifecycle oracle is below the ladder top, the reference optimum must beat it --
        # otherwise this case is not actually submitting a different, better design and proves
        # nothing about whether the top of the ladder is reachable.
        if top["raised"]:
            print(f"  FAIL {task:<46} reference optimum raised {top['error_type']}")
            ok = False
        elif lifecycle is not None and lifecycle < 100.0 and not (top["score"] > lifecycle):
            print(f"  FAIL {task:<46} reference optimum {top['score']} does not beat "
                  f"lifecycle oracle {lifecycle}")
            ok = False
        else:
            relation = "==" if top["score"] == lifecycle else ">"
            print(f"  ok   {task:<46} reference optimum {top['score']} {relation} "
                  f"lifecycle oracle {lifecycle}")

    # Third: remove the completeness gate from run_legacy_grader.py and require an incomplete
    # fixture's reward to come back non-zero. Without this, "reward == 0 for a 3/5 run" could be
    # true for the wrong reason -- a grader that returns 0 anyway -- and the gate would be
    # decorative. This edits a COPY; the delivered file is never touched.
    print()
    import shutil, tempfile
    for task in TASKS:
        task_dir = ROOT / "tasks" / task
        fixture = task_dir / "tests" / "invalid_submissions" / "missing_rounds"
        rounds = {int(q.stem.split("-")[1]): q.read_text(encoding="utf-8")
                  for q in sorted(fixture.glob("round-*.json"))}
        gated = grade_workspace(task, rounds)
        grader = task_dir / "tests" / "run_legacy_grader.py"
        original = grader.read_text(encoding="utf-8")
        backup = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
        backup.write(original); backup.close()
        try:
            grader.write_text(
                original.replace("reward = scientific_reward if complete else 0.0",
                                 "reward = scientific_reward  # gate removed by --self-test"),
                encoding="utf-8")
            ungated = grade_workspace(task, rounds)
        finally:
            shutil.copyfile(backup.name, grader)
            Path(backup.name).unlink(missing_ok=True)
        assert grader.read_text(encoding="utf-8") == original, "self-test failed to restore the grader"
        moved = (not ungated["raised"]) and ungated.get("reward") not in (None, 0.0) \
                and gated.get("reward") == 0.0
        mark = "ok  " if moved else "FAIL"
        print(f"  {mark} {task:<46} 3/5 rounds: gated reward={gated.get('reward')} "
              f"-> ungated reward={ungated.get('reward')}")
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
                verdict = (f"score={case['score']} reward={case.get('reward')} "
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

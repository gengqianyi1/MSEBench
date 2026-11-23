#!/usr/bin/env python3
"""Grader fixtures for task 14.

Two kinds of check, and they answer different questions.

REPLAY (the important one). Every archived first-round trajectory is graded again through the
migrated adapter and compared with the score the master repository's own scorer recorded at the
time. This is the only check that can catch the adapter silently becoming a *different* ruler from
the one the release calibration verifies -- a hand-written fixture cannot, because a fixture only
proves the grader agrees with whatever the fixture's author believed.

FIXTURES (the boundary cases the review asked for, 14-P2). Constructed submissions that exercise the
paths a real trajectory rarely reaches: a missing file, an empty trajectory, malformed JSON, a
molecule that is feasible but does not improve the objective, a molecule that beats the reference
(achievement must clip at 1.0 rather than exceed it), and a tie between two rounds at the same best
value. None of these proves a model can score; `AUTHOR_GUIDE.md` §6 is explicit about that, and the
replay block above is what proves the ruler is the right one.

    python private_grader/test_grader.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_ENV = "TASK14_ARCHIVE"          # optional override for the replay corpus
# Optional superset: the full first-round archive, available only when the package sits
# inside the development repository. Absent on a clean server -- which is exactly why the
# corpus in private_grader/replay_corpus/ ships with the package.
EXTERNAL_ARCHIVE = TASK_DIR.parents[4] / "A_multiobjective_mol_design_benchmark" / "runs_round1_archive"


def load_grader():
    spec = importlib.util.spec_from_file_location("t14_grader", TASK_DIR / "private_grader" / "grader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Ctx:
    instance_id: str
    task_version: str
    task_dir: Path
    submission_dir: Path
    run_dir: Path
    private_data_dir: Path
    run_manifest: dict


def make_ctx(tmp: Path) -> Ctx:
    submission = tmp / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    return Ctx("task_14_multiobjective_mol_design", "2.0.0", TASK_DIR,
               submission, tmp, TASK_DIR / "private_data", {})


def write_trajectory(submission: Path, rounds: list[dict]) -> None:
    lines = []
    for index, entry in enumerate(rounds, 1):
        lines.append(json.dumps({
            "round": index,
            "submitted_candidate": ({"name": f"c{index}", "smiles": entry["smiles"],
                                     "rationale": "fixture"} if entry.get("smiles") else None),
            "evaluation": {"smiles": entry.get("smiles", "")},
            "parse_error": entry.get("parse_error"),
        }, ensure_ascii=False))
    (submission / "trajectory.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (submission / "run_summary.json").write_text(json.dumps({
        "prompt_version": "A4-2026-08-01", "checks": {},
    }), encoding="utf-8")


def start_smiles() -> str:
    import yaml
    cfg = yaml.safe_load((TASK_DIR / "input" / "tasks" / "q1_multiobjective" /
                          "task_config.yaml").read_text(encoding="utf-8"))
    return cfg["start_smiles"]


def reference_smiles() -> str:
    import yaml
    ref = yaml.safe_load((TASK_DIR / "private_data" / "grader" / "reference" /
                          "reference_target.yaml").read_text(encoding="utf-8"))
    return ref["reference"]["best_smiles_in_window"]


def main() -> int:
    grader = load_grader()
    failures: list[str] = []

    def check(condition: bool, what: str, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'} {what}" + (f"\n         {detail}" if detail and not condition else ""))
        if not condition:
            failures.append(what)

    # ---------------------------------------------------------------------------------- replay
    print("\nA. replay: archived trajectories must reproduce the recorded score exactly")
    # The corpus ships INSIDE the package (private_grader/replay_corpus/), so this half runs on a
    # clean server too. Before it did, the replay silently skipped anywhere but the author's machine
    # -- and the replay is the half that proves the adapter is the ruler the calibration verifies.
    # The external archive is still honoured when present, as a superset.
    shipped = TASK_DIR / "private_grader" / "replay_corpus"
    cases = []
    if shipped.is_dir():
        for case in sorted(p for p in shipped.iterdir() if p.is_dir()):
            expected = json.loads((case / "expected.json").read_text(encoding="utf-8"))["total_score"]
            cases.append((case, expected, case.name))
    external = Path(os.environ[ARCHIVE_ENV]) if os.environ.get(ARCHIVE_ENV) else EXTERNAL_ARCHIVE
    if external.is_dir():
        shipped_names = {name for _, _, name in cases}
        for run in sorted(p for p in external.iterdir() if p.is_dir()):
            if run.name in shipped_names or not (run / "final_report.json").is_file():
                continue
            stored = json.loads((run / "final_report.json").read_text(encoding="utf-8"))
            cases.append((run, (stored.get("score") or {}).get("total_score"), run.name))
    check(len(cases) >= 3, f"at least 3 replay cases available ({len(cases)})")
    for case, expected, name in cases:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            for f in case.iterdir():
                if f.is_file() and f.name != "expected.json":
                    shutil.copy2(f, ctx.submission_dir / f.name)
            got = grader.grade(ctx)
        check(expected is not None and abs(float(got["score"]) - float(expected)) < 1e-9,
              f"{name[:44]}: {expected} reproduced", f"adapter returned {got['score']}")

    print("\nB. boundary fixtures")
    start, ref = start_smiles(), reference_smiles()

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(Path(tmp))          # no trajectory at all
        result = grader.grade(ctx)
        check(result["score"] == 0.0 and "MISSING_TRAJECTORY" in result["failure_codes"],
              "missing trajectory scores 0 with MISSING_TRAJECTORY", str(result["failure_codes"]))

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(Path(tmp))
        (ctx.submission_dir / "trajectory.jsonl").write_text("", encoding="utf-8")
        (ctx.submission_dir / "run_summary.json").write_text("{}", encoding="utf-8")
        result = grader.grade(ctx)
        check(result["score"] == 0.0 and "NO_PARSEABLE_SUBMISSION" in result["failure_codes"],
              "empty trajectory scores 0 without terminating the harness",
              f"{result['score']} {result['failure_codes']}")

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(Path(tmp))
        write_trajectory(ctx.submission_dir, [{"smiles": start}] * 5)
        result = grader.grade(ctx)
        check(result["score"] == 0.0 and "NO_USABLE_DESIGN" in result["failure_codes"],
              "resubmitting the start molecule scores 0 (feasible but not improving)",
              f"{result['score']} {result['failure_codes']}")

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(Path(tmp))
        write_trajectory(ctx.submission_dir, [{"smiles": "not a molecule"},
                                              {"smiles": "C1CC"},          # unparseable valence
                                              {"smiles": "[Na+].[Cl-]"},   # salt: inadmissible
                                              {"smiles": ""},
                                              {"smiles": "CC(=O)O.CC(=O)O"}])
        result = grader.grade(ctx)
        check(result["score"] == 0.0, "only illegal submissions scores 0", str(result["score"]))

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(Path(tmp))
        write_trajectory(ctx.submission_dir, [{"smiles": ref}])
        result = grader.grade(ctx)
        achievement = result["evidence"]["achievement_fraction"]
        check(result["score"] == 100.0, "the reference design itself scores 100", str(result["score"]))
        check(achievement is not None and abs(float(achievement) - 1.0) < 1e-9,
              "achievement of the reference design is exactly 1.0", str(achievement))

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(Path(tmp))
        write_trajectory(ctx.submission_dir, [{"smiles": ref}, {"smiles": start}, {"smiles": ref}])
        result = grader.grade(ctx)
        check(result["score"] == 100.0,
              "a tie at the best value, with a worse round between, still scores the best",
              str(result["score"]))
        check(result["evidence"]["best_round"] in (1, 3),
              "the tie resolves to one of the tied rounds", str(result["evidence"]["best_round"]))

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(Path(tmp))
        (ctx.submission_dir / "trajectory.jsonl").write_text(
            '{"round": 1, "submitted_candidate": {"smiles": "' + start + '"}}\n'
            'this line is not json\n', encoding="utf-8")
        (ctx.submission_dir / "run_summary.json").write_text("{}", encoding="utf-8")
        try:
            result = grader.grade(ctx)
            ok = isinstance(result.get("score"), (int, float))
        except Exception as exc:                                          # noqa: BLE001
            ok, result = False, {"error": str(exc)}
        check(ok, "a malformed trajectory line does not crash the grader", str(result))

    print(f"\n{'FAILED: ' + str(len(failures)) if failures else 'all grader fixtures passed'}")
    for item in failures:
        print(f"  - {item}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

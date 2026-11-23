"""Deterministic scientific grader for task 15, exposed as the harness's `grade(context)`.

This is an ADAPTER, not a reimplementation. The scoring rule -- the seven hard constraints including the
Tanimoto similarity floor that acts as the edit budget, the usability definition (feasible AND QED
strictly above the start), the achievement fraction against the offline enumeration's best, and the
band ladder -- lives in the task's own scorer, which the
master repository's `check_consistency.py` asserts against the enumeration on every release. Copying
that logic here would create a second ruler that could drift from the one the calibration verifies,
which is the exact failure this suite has hit before: two artifacts on disk carrying different
numbers for the same quantity. So this file resolves paths, calls the real scorer, and translates
its report into the common score schema.

What it reads (and nothing else):
  * `context.submission_dir/trajectory.jsonl`  -- the submitted molecules, one record per round
  * `context.submission_dir/run_summary.json`  -- prompt identity and per-round infrastructure checks
  * `context.private_data_dir/grader/...`      -- the reference optimum and the band ladder
  * `context.task_dir/input/tasks/.../task_config.yaml` -- the public contract that was in force

The score is a pure function of the submitted SMILES: every property is recomputed with RDKit from
the structure, so the grader never trusts a number the run reported about itself, and replaying a
fixed submission always produces the same score.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

# §16.6 protocol completeness. One implementation for all four tasks; the rule it applies is the one
# each task declares in its own PUBLIC task_config, so the code is shared and the rule is per-task.
def _find_harness_root() -> Path:
    """Locate the directory holding `harness/round_completeness.py` by searching upward.

    This used to be `parents[3]`, which is correct for the round-3 delivery layout
    (`<package>/tasks/<task>/private_grader/grader.py`) and WRONG for the Harbor verifier image,
    where the task snapshot is mounted at `/task` and the grader is `/task/private_grader/grader.py`
    -- there `parents[3]` is `/`, so the import raised `ModuleNotFoundError: No module named
    'harness'` and every Harbor verifier run failed before scoring. `run_legacy_grader.py` does not
    put `/task` on `sys.path`, so nothing downstream rescued it.

    A fixed depth is the wrong shape for something that has to work in two layouts. Searching for
    the marker works in both and fails loudly if the module is genuinely absent.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "harness" / "round_completeness.py").is_file():
            return candidate
    raise ModuleNotFoundError(
        "harness/round_completeness.py not found above " + str(here)
        + "; the task snapshot must ship it next to private_grader/")


_HARNESS = _find_harness_root()
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))
from harness import round_completeness as _rc  # noqa: E402


def _load_public_iteration(task_dir: Path) -> dict:
    """Read `iteration:` from the model-visible task config -- the declared, public contract."""
    import yaml
    path = task_dir / "input" / "tasks" / TASK_SUBDIR / "task_config.yaml"
    if not path.is_file():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("iteration") or {}


TASK_SUBDIR = "q1_lipophilic"



@contextlib.contextmanager
def _isolated_task_imports(task_dir: Path, search_paths):
    """Import this task's modules without inheriting -- or leaving behind -- another task's.

    Every task in this suite ships modules with the SAME names: `evaluator`, `scoring`, `utils`,
    `rescore_run`, `response_model`. `run_benchmark.py --all` grades them all in one process, so
    whichever task is graded first wins `sys.modules` and every later grader silently gets the wrong
    package. Measured before this guard existed: grading 17 then 16 in one process raised
    `AttributeError: module 'rescore_run' has no attribute 'submitted_formulations'`, because
    `rescore_run` was still task 17's. An AttributeError is the lucky outcome -- two tasks whose
    modules happen to share a function signature would have mis-scored in silence.

    So the context purges every module already imported from ANY task tree, prepends this task's
    paths, and purges again on the way out. The purge is derived from the tasks root rather than a
    hard-coded name list, so a task added later is covered without editing this.
    """
    tasks_root = task_dir.parent.resolve()

    def purge() -> None:
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            if not origin:
                continue
            try:
                resolved = Path(origin).resolve()
                resolved.relative_to(tasks_root)
            except (ValueError, OSError):
                continue
            if "private_grader" in resolved.parts:
                continue          # the grader modules themselves are loaded by explicit path
            del sys.modules[name]

    saved_path = list(sys.path)
    purge()
    sys.path[:0] = [str(p) for p in search_paths if str(p) not in sys.path]
    try:
        yield
    finally:
        sys.path[:] = saved_path
        purge()

def _prepare_imports(task_dir: Path, private_dir: Path) -> Any:
    """Put the public evaluator and the private scorer on the path, then hand back the scorer.

    The scorer resolves its own inputs from module-level constants that assume the master
    repository's `public/` + `grader/` layout. In the migrated package the same two halves live at
    `input/` and `private_data/grader/`, so the constants are re-pointed explicitly here rather than
    by recreating the old directory shape. Re-pointing is visible and fails loudly; a symlink farm
    or a chdir would not.
    """
    public_src = task_dir / "input" / "src"
    grader_src = private_dir / "grader" / "src"
    grader_scripts = private_dir / "grader" / "scripts"
    for path in (public_src, grader_src, grader_scripts):
        if not path.is_dir():
            raise FileNotFoundError(f"grader cannot run: missing {path}")
    # sys.path is managed by _isolated_task_imports, which grade() wraps this call in. Doing it here
    # as well would leave the entries behind for the next task graded in the same process.

    from rdkit import RDLogger                                            # noqa: E402
    RDLogger.DisableLog("rdApp.*")

    import score_run                                                      # noqa: E402
    # ROOT as well as the two file constants. The scorer records its own provenance as
    # `PUBLIC_TASK_CFG.relative_to(ROOT)`, and its ROOT is inferred from where score_run.py sits --
    # which in this layout is inside private_data/, a directory the public config is deliberately
    # NOT under. Re-pointing ROOT at the task directory puts both halves back under one root, which
    # is what the provenance field is trying to express in the first place.
    score_run.ROOT = task_dir
    score_run.PUBLIC_TASK_CFG = task_dir / "input" / "tasks" / TASK_SUBDIR / "task_config.yaml"
    score_run.REFERENCE_CFG = private_dir / "grader" / "reference" / "reference_target.yaml"
    for name, path in (("PUBLIC_TASK_CFG", score_run.PUBLIC_TASK_CFG),
                       ("REFERENCE_CFG", score_run.REFERENCE_CFG)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"grader cannot run: {name} not found at {path}")
    return score_run, (public_src, grader_src, grader_scripts)


def grade(context) -> dict:
    task_dir = Path(context.task_dir).resolve()
    private_dir = Path(context.private_data_dir).resolve() if context.private_data_dir else None
    submission = Path(context.submission_dir).resolve()
    if private_dir is None:
        raise RuntimeError("private_data_path is not configured; the reference is unreachable")

    trajectory = submission / "trajectory.jsonl"
    if not trajectory.is_file():
        return {
            "score": 0.0,
            "subscores": {},
            "failure_codes": ["MISSING_TRAJECTORY"],
            "evidence": {"reason": "submission/trajectory.jsonl absent; nothing to grade"},
        }

    search = (task_dir / "input" / "src", private_dir / "grader" / "src",
              private_dir / "grader" / "scripts")
    with _isolated_task_imports(task_dir, search):
        return _grade_isolated(task_dir, private_dir, submission)


def _grade_isolated(task_dir: Path, private_dir: Path, submission: Path) -> dict:
    score_run, _ = _prepare_imports(task_dir, private_dir)

    # write=False: a grader must not write into the submission it is grading. quiet=True: its
    # progress table belongs in a terminal, not in a harness log.
    #
    # SystemExit is caught deliberately. The scorer is also a command-line tool, and as a CLI it
    # exits with a message when a run directory holds no submitted molecule. Inside the harness that
    # same condition is an ordinary, scoreable outcome -- a model that never produced a parseable
    # candidate scores zero -- and a grader that terminates the process instead of returning would
    # take the whole evaluation down with it and lose the other tasks' results. `AUTHOR_GUIDE.md` §6
    # requires a determinate result for missing, malformed and out-of-range submissions; this is the
    # missing case.
    try:
        report = score_run.grade(submission, write=False, quiet=True)
    except SystemExit as exc:
        return {
            "score": 0.0,
            "subscores": {"optimality": 0.0, "constraint_satisfaction": 0.0},
            "failure_codes": ["NO_PARSEABLE_SUBMISSION"],
            "evidence": {
                "reason": str(exc) or "the scorer found no submitted molecule in the trajectory",
                "scoring_note": ("Zero rounds produced a molecule the evaluator could read. That is "
                                 "a scored outcome, not a grader error: the task awards points only "
                                 "for a usable design, and there was none."),
            },
        }
    score_block = report.get("score") or {}
    diagnostics = score_block.get("continuous_diagnostics") or {}
    details = score_block.get("optimality_score_details") or {}

    total = float(score_block.get("total_score") or 0.0)

    # Failure attribution, kept separate on purpose (§16.5 / §10): a round lost to the plumbing is
    # not the same finding as a round the model spent on an infeasible molecule, and a reviewer must
    # be able to tell them apart without opening the trajectory.
    checks = report.get("checks") or {}
    failure_codes: list[str] = []
    if diagnostics.get("n_rounds_usable", 0) == 0:
        failure_codes.append("NO_USABLE_DESIGN")
    for key, code in (("n_llm_call_failures", "LLM_CALL_FAILURE"),
                      ("n_empty_llm_responses", "EMPTY_LLM_RESPONSE"),
                      ("n_truncated_llm_responses", "TRUNCATED_LLM_RESPONSE"),
                      ("n_parse_failures", "UNPARSEABLE_SUBMISSION")):
        count = int(checks.get(key) or 0)
        if count:
            failure_codes.append(f"{code}:{count}")

    # §16.6: did every declared round carry a molecule the scorer could read? A verdict kept
    # separate from the score, applying the rule the public task_config declares in advance. The
    # per-round records come from the task's own scorer, so this cannot disagree with the scoring.
    _per_round = score_block.get("per_round") or report.get("per_round") or []
    completeness = _rc.assess(
        rounds_with_candidate=[r.get("round") for r in _per_round if r.get("smiles")],
        observed_round_numbers=[r.get("round") for r in _per_round],
        iteration_cfg=_load_public_iteration(task_dir),
        run_summary=report.get("run_summary") or {},
    )
    failure_codes.extend(_rc.failure_code(completeness))

    return {
        "score": total,
        "subscores": {
            "optimality": total,
            "constraint_satisfaction": 0.0,   # a gate, never a scoring category
        },
        "failure_codes": failure_codes,
        "evidence": {
            "protocol_completeness": completeness,
            "achievement_fraction": details.get("achievement_fraction"),
            "achievement_raw_unclipped": diagnostics.get("achievement_raw_unclipped"),
            "best_usable_objective": details.get("best_successful_objective"),
            "objective_gain_over_start": diagnostics.get("objective_gain_over_start"),
            "n_rounds": diagnostics.get("n_rounds"),
            "n_rounds_with_a_parsed_submission": diagnostics.get("n_rounds_with_a_parsed_submission"),
            "n_rounds_valid": diagnostics.get("n_rounds_valid"),
            "n_rounds_feasible": diagnostics.get("n_rounds_feasible"),
            "n_rounds_usable": diagnostics.get("n_rounds_usable"),
            "best_round": diagnostics.get("best_round"),
            "per_round_objective": diagnostics.get("per_round_objective"),
            "violation_histogram": diagnostics.get("violation_histogram"),
            "prompt_version": report.get("prompt_version"),
            "question_sha256": report.get("question_sha256"),
            "graded_by": "task-native scorer via private_grader/grader.py",
            "scoring_note": (
                "Constraint satisfaction is a gate worth zero points; all 100 points come from how "
                "much of the reachable QED headroom the best usable submission captured under the edit budget. The "
                "reference is the best design an offline enumeration found, not a proof of global "
                "optimality, and achievement is clipped at 1.0."),
        },
    }

"""Deterministic scientific grader for task 17, exposed as the harness's `grade(context)`.

An ADAPTER over the task's own scorer, with the same seam as task 16: `scripts/rescore_run.py` is a
command-line tool whose `main()` mixes the scoring with console tables and file writes, so the three
scoring calls are made here directly and in the same order and with the same arguments it uses:

    rounds      = rescore_run.submitted_processes(submission_dir)
    start_eval  = evaluator.evaluate_reference_start(cfg["objective"])
    evaluations = [evaluator.evaluate_design(proc, cfg["objective"]) for _, proc in rounds]
    score       = scoring.final_score(start_eval, evaluations, cfg["constraints"],
                                      cfg["scoring"], cfg["reference"])

`final_score` in `src/scoring.py` IS the ruler; the master repository's `check_consistency.py`
asserts it against the enumeration and against `design_space.POINT_BANDS` on every release.

Two things this task needs that its siblings do not. The response mechanism (`src/agglomeration.py`,
the population balance and the frozen surrogate) is grader-side, so the grader imports it from
`private_data/src` while the model-visible `input/src` holds only a raising stub. And the grading
config is the full `task_config.yaml`, which was withheld from `input/` entirely -- 17 ships a
separately generated `task_config.public.yaml` with no reference block and no band ladder.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path


TASK_SUBDIR = "q1_agglomeration"

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


def _improving_percentile(private_dir: Path, achievement) -> float | None:
    """Where this achievement sits among the enumerated designs that improve on the start, in percent.

    Round-3 audit context: 9 and 12 points look three apart, and are in fact the 29th and the 60th
    percentile of every in-budget design that beats the registered process. The band ladder is
    anchored at a reference only 2 designs in 19,779 approach, so the raw number understates the
    spread. This puts the population next to it.
    """
    if achievement is None:
        return None
    path = private_dir / "answers" / "reference_enumeration.json"
    if not path.is_file():
        return None
    try:
        import json as _json
        share = (_json.loads(path.read_text(encoding="utf-8"))["gates"]["g3"]
                 ["achievement_distribution_over_improving_designs"]["share_at_or_above"])
    except (ValueError, KeyError):
        return None
    pts = sorted((float(k), float(v)) for k, v in share.items())
    a = float(achievement)
    if a <= pts[0][0]:
        return round(100.0 * (1.0 - pts[0][1]), 1)
    if a >= pts[-1][0]:
        return round(100.0 * (1.0 - pts[-1][1]), 1)
    for (x0, s0), (x1, s1) in zip(pts, pts[1:]):
        if x0 <= a <= x1:
            t = 0.0 if x1 == x0 else (a - x0) / (x1 - x0)
            return round(100.0 * (1.0 - (s0 + t * (s1 - s0))), 1)
    return None


def _load_public_iteration(task_dir: Path) -> dict:
    """Read `iteration:` from the model-visible task config -- the declared, public contract."""
    import yaml
    for rel in (Path("input") / "tasks" / TASK_SUBDIR / "task_config.yaml",
                Path("input") / "tasks" / TASK_SUBDIR / "task_config.public.yaml"):
        path = task_dir / rel
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data.get("iteration") or {}
    return {}


def _run_summary(submission_dir: Path) -> dict:
    """The runner's own per-round accounting, when it wrote any."""
    import json as _json
    path = submission_dir / "run_summary.json"
    if not path.is_file():
        return {}
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}



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

def _prepare(task_dir: Path, private_dir: Path):
    public_src = task_dir / "input" / "src"
    private_src = private_dir / "src"
    private_scripts = private_dir / "scripts"
    for path in (private_src, public_src, private_scripts):
        if not path.is_dir():
            raise FileNotFoundError(f"grader cannot run: missing {path}")
    # Order matters and must be written down explicitly, not produced by a loop of insert(0, ...) --
    # that reverses the sequence, which put input/src (the raising stub of response_model) ahead of
    # private_data/src (the real one) and made every grade fail with ResponseModelUnavailable.
    # sys.path itself is managed by _isolated_task_imports, which grade() wraps this call in, and
    # it prepends these three in exactly this order. Setting it here too would leave the entries
    # behind for whichever task is graded next in the same process.
    # private_src first: it holds the real response_model (and agglomeration.py behind it), while
    # input/src holds the stub of the same module name.

    import evaluator                                                      # noqa: E402
    import scoring                                                        # noqa: E402
    import rescore_run                                                    # noqa: E402
    from utils import load_yaml                                           # noqa: E402

    config_path = private_dir / "tasks" / TASK_SUBDIR / "task_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"grader cannot run: the grader-side task config is missing at {config_path}. "
            f"input/ carries only task_config.public.yaml, which has no reference block.")
    cfg = load_yaml(config_path)
    reference = cfg.get("reference") or {}
    if reference.get("best_objective_in_budget") is None:
        raise RuntimeError(
            "the grader-side task config carries no reference optimum -- refusing to report a score "
            "that cannot be computed")
    return evaluator, scoring, rescore_run, cfg, reference


def grade(context) -> dict:
    task_dir = Path(context.task_dir).resolve()
    private_dir = Path(context.private_data_dir).resolve() if context.private_data_dir else None
    submission = Path(context.submission_dir).resolve()
    if private_dir is None:
        raise RuntimeError("private_data_path is not configured; the reference is unreachable")

    if not (submission / "trajectory.jsonl").is_file():
        return {
            "score": 0.0,
            "subscores": {},
            "failure_codes": ["MISSING_TRAJECTORY"],
            "evidence": {"reason": "submission/trajectory.jsonl absent; nothing to grade"},
        }

    search = (private_dir / "src", task_dir / "input" / "src",
              private_dir / "scripts")
    with _isolated_task_imports(task_dir, search):
        return _grade_isolated(task_dir, private_dir, submission)


def _grade_isolated(task_dir: Path, private_dir: Path, submission: Path) -> dict:
    evaluator, scoring, rescore_run, cfg, reference = _prepare(task_dir, private_dir)

    rounds = rescore_run.submitted_processes(submission)
    if not any(process for _, process in rounds):
        return {
            "score": 0.0,
            "subscores": {"optimality": 0.0, "feasibility": 0.0},
            "failure_codes": ["NO_PARSEABLE_SUBMISSION"],
            "evidence": {"reason": "no round recorded a process the evaluator could read",
                         "scoring_note": ("A scored outcome, not a grader error: points are awarded "
                                          "only for a usable process and there was none.")},
        }

    objective_cfg, constraints_cfg = cfg["objective"], cfg["constraints"]
    start_eval = evaluator.evaluate_reference_start(objective_cfg)
    evaluations = [evaluator.evaluate_design(process, objective_cfg) for _, process in rounds]
    score = scoring.final_score(
        start_eval=start_eval,
        round_evaluations=evaluations,
        constraints_cfg=constraints_cfg,
        scoring_cfg=cfg["scoring"],
        reference_cfg=reference,
    )

    total = float(score.get("total_score") or 0.0)
    details = score.get("optimality_score_details") or {}

    # Failure attribution split by the thing that actually blocked the round, because C2 (the change
    # budget) and C3-C7 (safety and capacity) fail for unrelated reasons: a batch over budget was
    # measured and scored zero, a batch that breaks a safety limit was never measured at all.
    n_valid = sum(1 for ev in evaluations if getattr(ev, "valid", False))
    n_over_budget = 0
    n_unrunnable = 0
    for ev in evaluations:
        if not getattr(ev, "valid", False):
            continue
        broken = [name for name, ok in
                  evaluator.check_constraints(ev, constraints_cfg).items() if not ok]
        if any(name.upper().startswith("C2") or "budget" in name.lower() for name in broken):
            n_over_budget += 1
        if any(not (name.upper().startswith("C2") or "budget" in name.lower()) for name in broken):
            n_unrunnable += 1

    failure_codes: list[str] = []
    if int(score.get("num_successful_designs") or 0) == 0:
        failure_codes.append("NO_USABLE_PROCESS")
    if n_valid < len(evaluations):
        failure_codes.append(f"NOT_IN_DESIGN_SPACE:{len(evaluations) - n_valid}")
    if n_over_budget:
        failure_codes.append(f"OVER_CHANGE_BUDGET:{n_over_budget}")
    if n_unrunnable:
        failure_codes.append(f"NOT_RUNNABLE:{n_unrunnable}")

    # §16.6: did every declared round carry a candidate the scorer could read? A verdict kept
    # separate from the score, applying the rule the public task_config declares in advance.
    _pub = _load_public_iteration(task_dir)
    completeness = _rc.assess(
        rounds_with_candidate=[r for r, design in rounds if design],
        observed_round_numbers=[r for r, _ in rounds],
        iteration_cfg=_pub,
        run_summary=_run_summary(submission),
    )
    failure_codes.extend(_rc.failure_code(completeness))

    return {
        "score": total,
        "subscores": {"optimality": total, "feasibility": 0.0},
        "failure_codes": failure_codes,
        "evidence": {
            "protocol_completeness": completeness,
            "achievement_percentile_among_improving_designs":
                _improving_percentile(private_dir, details.get("achievement_fraction")),
            "achievement_fraction": details.get("achievement_fraction"),
            "best_usable_objective": details.get("best_successful_objective"),
            "baseline_objective": getattr(start_eval, "desirability", None),
            "num_successful_designs": score.get("num_successful_designs"),
            "n_rounds_graded": len(evaluations),
            "n_rounds_in_design_space": n_valid,
            "n_rounds_over_change_budget": n_over_budget,
            "n_rounds_not_runnable": n_unrunnable,
            "graded_by": "task-native final_score via private_grader/grader.py",
            "scoring_note": (
                "Feasibility is a gate worth zero. A batch outside the design space or breaking a "
                "safety or capacity limit is not measured at all; a batch over the weighted "
                "change budget IS measured and still scores nothing. All 100 points come from how "
                "much of the reachable desirability headroom the best usable process captured."),
            "data_nature": (
                "SYNTHETIC. The four attributes are simulator outputs from a mechanistic population "
                "balance plus a surrogate fitted to it; no real drug substance, registered product "
                "or filed process exists behind any number here."),
        },
    }

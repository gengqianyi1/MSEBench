"""Deterministic scientific grader for task 16, exposed as the harness's `grade(context)`.

Like its siblings this is an ADAPTER, but the seam is a different shape. Tasks 14 and 15 ship a
scorer with a reusable `grade(run_dir)` function; this package's `scripts/grade_run.py` is a
command-line tool whose `main()` interleaves the scoring with argument parsing, a ruler-version
guard, console formatting and writing `graded_report.json` into the run. None of that belongs inside
a harness grader -- writing into the submission under evaluation especially -- so the four scoring
calls are made here directly:

    rounds      = rescore_run.submitted_formulations(submission_dir)
    start_eval  = evaluator.evaluate_reference_start(cfg["objective"])
    evaluations = [evaluator.evaluate_design(form, cfg["objective"]) for _, form in rounds]
    score       = scoring.final_score(...)

Those are the same four calls in the same order with the same arguments as `grade_run.main()` makes.
`final_score` in `src/scoring.py` IS the ruler -- the release gate, the desirability, the achievement
fraction and the band ladder all live there, and the master repository's `check_consistency.py`
asserts it against the enumeration on every release. Re-deriving any of it here would create a second
ruler that could drift from the verified one, which is the failure this suite has hit before.

The config used is the GRADER-side `task_config.yaml`, which carries the `reference:` block. The
model-visible copy under `input/` has that block replaced by a stub, so the run itself could not
compute a score even if it tried -- and `build_chembench_v2.py` asserts at build time that no
grader-side value survives anywhere in `input/`.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path


TASK_SUBDIR = "q1_microenvironment"

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


def _qualifying_percentile(private_dir: Path, achievement) -> float | None:
    """Where this achievement sits among the enumerated qualifying designs, in percent.

    Round-3 audit context: 6-17 points reads as near-total failure until you know that 17 is better
    than 84 % of every design in the neighbourhood that passes the release gate. The score maps
    achievement onto a ladder anchored at the single best design, so the number alone understates
    what it took to get there. This puts the population back next to it.

    Interpolated between the sparse thresholds the enumeration recorded -- no more precision is
    claimed than that table carries.
    """
    if achievement is None:
        return None
    path = private_dir / "answers" / "attribution_round3.json"
    if not path.is_file():
        return None
    try:
        import json as _json
        share = (_json.loads(path.read_text(encoding="utf-8"))
                 ["offline"]["achievement_of_qualifying"]["share_at_or_above"])
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
    """Import the task's own evaluator, scorer and trajectory reader, and load the grading config."""
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
    # private_src goes first on purpose: it holds the REAL response_model, and input/src holds the
    # raising stub of the same module name. Getting this order wrong would not silently mis-score --
    # the stub raises -- but the failure would be reported as a grader error rather than a
    # configuration mistake, so it is worth being explicit about.

    import evaluator                                                      # noqa: E402
    import scoring                                                        # noqa: E402
    import rescore_run                                                    # noqa: E402
    from utils import load_yaml                                           # noqa: E402

    config_path = private_dir / "tasks" / TASK_SUBDIR / "task_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"grader cannot run: the grader-side task config is missing at {config_path}. "
            f"The model-visible copy under input/ carries no reference block by design.")
    cfg = load_yaml(config_path)
    reference = cfg.get("reference") or {}
    if reference.get("best_objective_in_budget") is None:
        raise RuntimeError(
            "the grader-side task config carries no reference optimum; this is the public copy, "
            "not the grading one -- refusing to report a score that cannot be computed")
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

    rounds = rescore_run.submitted_formulations(submission)
    if not rounds:
        return {
            "score": 0.0,
            "subscores": {"optimality": 0.0, "feasibility": 0.0, "release_gate": 0.0},
            "failure_codes": ["NO_PARSEABLE_SUBMISSION"],
            "evidence": {"reason": "no round recorded a formulation the evaluator could read",
                         "scoring_note": ("A scored outcome, not a grader error: points are awarded "
                                          "only for a qualifying formulation and there was none.")},
        }

    start_eval = evaluator.evaluate_reference_start(cfg["objective"])
    evaluations = [evaluator.evaluate_design(form, cfg["objective"]) for _, form in rounds]
    score = scoring.final_score(
        start_eval=start_eval,
        round_evaluations=evaluations,
        constraints=cfg["constraints"],
        scoring_cfg=cfg["scoring"],
        reference_cfg=reference,
        release_gate_cfg=cfg.get("release_gate"),
    )

    total = float(score.get("total_score") or 0.0)
    details = score.get("optimality_score_details") or {}
    metrics = score.get("continuous_metrics") or {}
    gate = score.get("release_gate") or {}

    # The four gates are reported separately because they fail for completely different reasons and
    # a single "score 0" hides which one bit. The release gate in particular is the one the task is
    # built around: a formulation can improve D and still be worth nothing.
    failure_codes: list[str] = []
    if metrics.get("n_rounds_qualifying", 0) == 0:
        failure_codes.append("NO_QUALIFYING_FORMULATION")
    blocked = int(gate.get("n_blocked_by_cqa_compliance_only") or 0)
    if blocked:
        failure_codes.append(f"BLOCKED_BY_RELEASE_GATE:{blocked}")
    if metrics.get("n_rounds_measured", 0) < len(rounds):
        failure_codes.append(
            f"NOT_MEASURED:{len(rounds) - int(metrics.get('n_rounds_measured') or 0)}")
    if metrics.get("n_rounds_budget_feasible", 0) == 0:
        failure_codes.append("NO_ROUND_INSIDE_CHANGE_BUDGET")

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
        "subscores": {
            "optimality": total,
            "feasibility": 0.0,      # a gate, never a scoring category
            "release_gate": 0.0,     # likewise
        },
        "failure_codes": failure_codes,
        "evidence": {
            "protocol_completeness": completeness,
            "achievement_percentile_among_qualifying_designs":
                _qualifying_percentile(private_dir, details.get("achievement_fraction")),
            "achievement_fraction": details.get("achievement_fraction"),
            "achievement_raw": metrics.get("best_qualifying_raw_achievement"),
            "best_qualifying_desirability": metrics.get("best_qualifying_desirability"),
            "best_qualifying_round": metrics.get("best_qualifying_round"),
            "best_by_desirability": metrics.get("best_by_desirability"),
            "best_by_desirability_was_compliant": metrics.get("best_by_desirability_was_compliant"),
            "n_rounds_graded": len(rounds),
            "n_rounds_measured": metrics.get("n_rounds_measured"),
            "n_rounds_budget_feasible": metrics.get("n_rounds_budget_feasible"),
            "n_rounds_cqa_compliant": metrics.get("n_rounds_cqa_compliant"),
            "n_rounds_qualifying": metrics.get("n_rounds_qualifying"),
            "n_blocked_by_release_gate_only": blocked,
            "ruler_version": cfg.get("ruler_version"),
            "package_prompt_version": cfg.get("prompt_version"),
            "graded_by": "task-native final_score via private_grader/grader.py",
            "scoring_note": (
                "Four gates in order: design-space membership, change budget, the release gate (all "
                "five CQAs strictly inside specification), and D above the baseline. Missing any one "
                "scores 0 for that submission. A formulation that improves D but breaches one "
                "specification is not a partial success -- it cannot be released."),
            "data_nature": (
                "SYNTHETIC. Every CQA is the output of a fitted surrogate over a closed-form "
                "simulator; no measurement, product or dossier exists behind any number here."),
        },
    }

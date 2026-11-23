"""Grader-side entry point: score a run directory produced by the PUBLIC package.

Common review point A: the reference optimum, the reference design and the band anchors must not sit
on the model-visible side. So the public package ships `reference:` as an empty stub, a public run
records every submission but no score, and the arithmetic happens here, inside the grader package.

The score is a pure function of the submitted formulations, so this needs no API call and is exactly
reproducible:

    $PY scripts/grade_run.py runs/<run_dir>                    # print + write graded_report.json
    $PY scripts/grade_run.py runs/<run_dir> --update-report    # also rewrite final_report.json
    $PY scripts/grade_run.py runs/<run_dir> --json             # machine-readable only

It refuses to grade a run whose recorded ruler_version disagrees with this package's, because a score
is only comparable to another score at the same ruler version. Pass --force-version to override, and
expect to justify it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluator import evaluate_design, evaluate_reference_start   # noqa: E402
from scoring import final_score                                  # noqa: E402
from utils import load_yaml                                      # noqa: E402
from rescore_run import submitted_formulations                   # noqa: E402

TASK_CFG = ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default=str(TASK_CFG))
    ap.add_argument("--update-report", action="store_true")
    ap.add_argument("--force-version", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    cfg = load_yaml(args.config)
    ref = cfg.get("reference") or {}
    if ref.get("best_objective_in_budget") is None:
        raise SystemExit(
            "this package carries no reference optimum, so it cannot grade. You are in the PUBLIC "
            "package; run scripts/grade_run.py inside the grader package instead.")

    ruler = str(cfg.get("ruler_version", "unversioned"))
    report_path = run_dir / "final_report.json"
    stored = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    run_ruler = str(stored.get("ruler_version")
                    or (stored.get("score") or {}).get("ruler_version") or "unversioned")
    if run_ruler != ruler and not args.force_version:
        raise SystemExit(
            f"the run was produced under ruler_version {run_ruler!r} and this package is {ruler!r}. "
            f"Scores are only comparable at the same ruler version -- re-run the model, or pass "
            f"--force-version if you know why you want a cross-version number.")

    rounds = submitted_formulations(run_dir)
    if not rounds:
        raise SystemExit(f"no submitted formulations recorded in {run_dir}")

    start_eval = evaluate_reference_start(cfg["objective"])
    evaluations = [evaluate_design(form, cfg["objective"]) for _, form in rounds]
    score = final_score(start_eval=start_eval, round_evaluations=evaluations,
                        constraints=cfg["constraints"], scoring_cfg=cfg["scoring"],
                        reference_cfg=ref, release_gate_cfg=cfg.get("release_gate"))
    score["graded_by"] = "scripts/grade_run.py"
    score["graded_against_ruler_version"] = ruler
    score["run_prompt_version"] = stored.get("prompt_version")
    score["package_prompt_version"] = cfg.get("prompt_version")
    if (stored.get("prompt_version") is not None
            and int(stored["prompt_version"]) != int(cfg.get("prompt_version", 1))):
        score["prompt_version_warning"] = (
            "this run was produced under prompt_version %s and the package is at %s. The score "
            "recomputes, because scoring is a pure function of the submitted formulations, but the "
            "run is evidence about a task the package no longer poses -- do not quote it as current "
            "calibration." % (stored["prompt_version"], cfg.get("prompt_version")))
        print("WARNING: " + score["prompt_version_warning"])
    score["run_dir"] = run_dir.name
    score["n_rounds_graded"] = len(rounds)

    out = run_dir / "graded_report.json"
    out.write_text(json.dumps(score, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps({k: v for k, v in score.items()
                          if k not in ("continuous_metrics", "release_gate",
                                       "best_by_objective_design", "best_successful_design",
                                       "constraint_diagnostics_for_best_by_objective")},
                         indent=2, ensure_ascii=False, default=str))
    else:
        det = score["optimality_score_details"]
        cm = score["continuous_metrics"]
        print(f"run                        : {run_dir.name}")
        print(f"ruler version              : {ruler}")
        print(f"rounds graded              : {len(rounds)}")
        print(f"rounds measured            : {cm['n_rounds_measured']}")
        print(f"rounds inside the budget   : {cm['n_rounds_budget_feasible']}")
        print(f"rounds CQA-compliant       : {cm['n_rounds_cqa_compliant']}")
        print(f"rounds qualifying          : {cm['n_rounds_qualifying']}")
        print(f"blocked by the release gate: "
              f"{score['release_gate']['n_blocked_by_cqa_compliance_only']}")
        print(f"best qualifying D          : {cm['best_qualifying_desirability']}"
              f"  (round {cm['best_qualifying_round']})")
        print(f"best D of any submission   : {cm['best_by_desirability']}"
              f"  (compliant: {cm['best_by_desirability_was_compliant']})")
        print(f"achievement (clipped)      : {det['achievement_fraction']}")
        print(f"achievement (raw)          : {cm['best_qualifying_raw_achievement']}")
        print(f"success                    : {score['success']}")
        print(f"TOTAL SCORE                : {score['total_score']} / "
              f"{det['optimality_max_points']}")
        print(f"written -> {out}")

    if args.update_report and report_path.exists():
        stored["score"] = score
        stored["success"] = bool(score.get("success", False))
        stored["ruler_version"] = ruler
        stored["graded_by"] = "scripts/grade_run.py"
        report_path.write_text(json.dumps(stored, indent=2, ensure_ascii=False, default=str),
                               encoding="utf-8")
        print(f"updated -> {report_path}")


if __name__ == "__main__":
    main()

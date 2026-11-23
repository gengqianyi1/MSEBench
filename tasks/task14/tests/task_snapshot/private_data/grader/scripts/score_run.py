"""Grade a finished run. GRADER-SIDE ONLY -- this is the isolated scoring end.

Second-round review, 姚涵予 #2 / 田双燕 §2.5 (evaluation isolation): the reference answer used to sit
in `tasks/q1_multiobjective/task_config.yaml`, the file the run harness loads, and the harness itself
computed the score and wrote the reference value into every `final_report.json`. Scoring now happens
here instead. This script is the only place that opens `grader/reference/reference_target.yaml`, and
it needs nothing from the run except the molecules that were submitted -- which are recorded in the
trajectory. Scoring is therefore a pure function of the trajectory: no API calls, and the same run
always produces the same score.

`rescore_run.py` is a one-line alias for this file, so the package contains exactly one scorer.

Run:
    python grader/scripts/score_run.py runs/<run_dir>
    python grader/scripts/score_run.py runs/<run_dir> --no-write     # print only
    python grader/scripts/score_run.py --all                         # every run under runs/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
# Run directories used to live at the repository root while the harness config said `../runs`,
# which is correct only while `public/` sits inside this repository. In a standalone participant
# package there is no parent to write into, so the second-round packages set `run_dir: runs` and the
# directory lives inside the package. Both layouts are accepted here so that first-round run
# directories stay gradeable.
RUNS = ROOT / "public" / "runs" if (ROOT / "public" / "runs").is_dir() else ROOT / "runs"
sys.path.insert(0, str(ROOT / "public" / "src"))
sys.path.insert(0, str(ROOT / "grader" / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import MoleculeEvaluation, check_constraints, constraint_margins, evaluate_smiles  # noqa: E402
from scoring import final_score  # noqa: E402
from utils import load_yaml  # noqa: E402

PUBLIC_TASK_CFG = ROOT / "public" / "tasks" / "q1_multiobjective" / "task_config.yaml"
REFERENCE_CFG = ROOT / "grader" / "reference" / "reference_target.yaml"


def load_grading_config() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Public task definition + the grader's private reference and bands."""
    task_cfg = load_yaml(PUBLIC_TASK_CFG)
    ref_cfg = load_yaml(REFERENCE_CFG)
    if ref_cfg.get("task_id") != task_cfg.get("task_id"):
        raise SystemExit(
            f"task_id mismatch: public={task_cfg.get('task_id')!r} "
            f"grader={ref_cfg.get('task_id')!r}. The reference file belongs to a different task."
        )
    return task_cfg, ref_cfg["reference"], ref_cfg["scoring"]


def submitted_smiles(run_dir: Path) -> List[Tuple[Optional[int], str]]:
    """Pull the submitted SMILES per round out of whichever log the run wrote.

    A round in which the model never submitted (LLM failure or an unparseable message) yields an
    empty string, so it still occupies a round and still counts as a lost round rather than silently
    disappearing.
    """
    traj = run_dir / "trajectory.jsonl"
    out: List[Tuple[Optional[int], str]] = []
    if traj.exists():
        for line in traj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            # A trajectory line that is not JSON is a damaged record, not a program error.
            # AUTHOR_GUIDE.md §6 requires a grader to give a determinate result for malformed input;
            # skipping the line keeps the rest of the run gradeable, and the round it belonged to
            # still shows up as a round with no submission rather than silently disappearing.
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "submitted_candidate" not in rec and "evaluation" not in rec:
                continue  # a pre-split trajectory that only logged the prompt
            cand = rec.get("submitted_candidate") or {}
            smi = cand.get("smiles")
            if smi is None:
                ev = rec.get("evaluation") or {}
                smi = ev.get("smiles")
            out.append((rec.get("round"), smi or ""))
    if out:
        return out
    # Fallback for a run whose trajectory predates the split: recover from the raw API log.
    debug = run_dir / "llm_api_debug.jsonl"
    if debug.exists():
        for line in debug.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            # A trajectory line that is not JSON is a damaged record, not a program error.
            # AUTHOR_GUIDE.md §6 requires a grader to give a determinate result for malformed input;
            # skipping the line keeps the rest of the run gradeable, and the round it belonged to
            # still shows up as a round with no submission rather than silently disappearing.
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = ((rec.get("llm_call") or {}).get("content")) or ""
            try:
                obj = json.loads(content[content.index("{"):content.rindex("}") + 1])
                smi = (obj.get("candidate") or {}).get("smiles")
                if smi:
                    out.append((rec.get("round"), smi))
            except (ValueError, AttributeError, json.JSONDecodeError):
                continue
    return out


def protocol_completeness(per_round: List[Dict[str, Any]], iteration_cfg: Dict[str, Any],
                          summary: Dict[str, Any]) -> Dict[str, Any]:
    """Did every declared round carry a parseable candidate? A verdict, not an inference.

    Round-3 audit (刘丞军 P1): the second round decided after the fact that "losing fewer than two
    rounds still counts as a valid run". A degradation rule chosen once the scores are visible can
    always be placed where it flatters the result, so the rule now lives in the public
    `task_config.yaml` (`iteration.round_completeness`) and is applied here from that config.

    Deliberately kept separate from the score. Best-of-N over the trajectory is non-decreasing in N,
    so a run that lost a round can only have scored the same or less than it would have with all of
    them: its score is a valid LOWER BOUND, which is a real measurement and not a discard reason
    (§10 forbids letting an infrastructure loss into the science column). What it is not is
    two-sided evidence, so the verdict travels with the score wherever the score goes.
    """
    cfg = (iteration_cfg or {}).get("round_completeness") or {}
    declared = int((iteration_cfg or {}).get("max_rounds") or len(per_round))
    with_candidate = [r for r in per_round if r["smiles"]]
    missing = []
    seen = {r["round"] for r in per_round if r["round"] is not None}
    for r in per_round:
        if not r["smiles"]:
            # The trajectory kept the round but nothing parseable came back for it. Which of the two
            # causes it was is a run-log fact, not a grader fact, so read it rather than guess.
            missing.append({"round": r["round"], "cause": "LLM_CALL_FAILURE_OR_UNPARSEABLE"})
    for n in range(1, declared + 1):
        if n not in seen:
            missing.append({"round": n, "cause": "NO_ROUND_RECORD"})
    # Prefer the runner's own per-round failure codes when it recorded them: it knows whether the
    # provider never answered or answered with something unparseable, and the grader does not.
    codes = (summary or {}).get("round_failures") or (summary or {}).get("failure_codes") or []
    if isinstance(codes, list) and codes:
        for m in missing:
            for c in codes:
                if isinstance(c, dict) and c.get("round") == m["round"] and c.get("code"):
                    m["cause"] = c["code"]

    complete = len(missing) == 0 and len(with_candidate) == declared
    trigger = cfg.get("rerun_required_missing_rounds")
    return {
        "declared_rounds": declared,
        "rounds_with_a_candidate": len(with_candidate),
        "n_missing_rounds": len(missing),
        "missing_rounds": sorted(missing, key=lambda m: (m["round"] is None, m["round"])),
        "verdict": "COMPLETE" if complete else "INCOMPLETE_MULTI_ROUND",
        "score_interpretation": ("two-sided" if complete else
                                 "lower bound only -- fewer attempts can only lower best-of-N"),
        "rerun_required": (False if trigger is None else len(missing) >= int(trigger)),
        "rule_source": "public task_config.yaml :: iteration.round_completeness",
        "may_not_support": cfg.get("incomplete_run_may_not_support", []) if not complete else [],
    }


def _violation_histogram(per_round: List[Dict[str, Any]]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in per_round:
        for name in r["constraints_violated"]:
            hist[name] = hist.get(name, 0) + 1
        for code in r["rejection_reasons"] or []:
            key = f"input:{code}"
            hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def grade(run_dir: Path, write: bool = True, quiet: bool = False) -> Dict[str, Any]:
    task_cfg, reference, scoring_cfg = load_grading_config()
    start_smiles = task_cfg["start_smiles"]
    constraints = task_cfg["constraints"]
    sim_cfg = constraints["similarity"]
    target_cfg = task_cfg["optimization_target"]

    def ev(s: str) -> MoleculeEvaluation:
        return evaluate_smiles(s, start_smiles=start_smiles, similarity_cfg=sim_cfg)

    start_eval = ev(start_smiles)
    rounds = submitted_smiles(run_dir)
    if not rounds:
        raise SystemExit(f"no submitted molecules found in {run_dir}")

    if not quiet:
        print(f"grading {run_dir.name}")
        print(f"  reference QED = {reference['best_objective_in_window']}   "
              f"start QED = {start_eval.qed:.6f}\n")
        print(f"{'R':>2} {'QED':>8} {'MW':>7} {'LogP':>6} {'SA':>5} {'sim':>6} {'alerts':>6}  verdict")

    evaluations: List[MoleculeEvaluation] = []
    per_round: List[Dict[str, Any]] = []
    for rnd, smi in rounds:
        e = ev(smi)
        evaluations.append(e)
        broken = [k for k, v in check_constraints(e, constraints).items() if not v]
        improved = (e.qed is not None and start_eval.qed is not None and e.qed > start_eval.qed)
        usable = (not broken) and improved
        per_round.append({
            "round": rnd,
            "smiles": smi or None,
            "canonical_smiles": e.canonical_smiles,
            "valid": e.valid,
            "rejection_reasons": e.rejection_reasons,
            "objective_value": e.qed,
            "constraints_violated": broken,
            # How far each violated constraint was from its boundary. A reviewer asking "how close
            # was this?" should not have to recompute it.
            "constraint_margins": constraint_margins(e, constraints)["per_constraint"],
            "improves_on_start": improved,
            "usable": usable,
        })
        if not quiet:
            if not e.valid:
                reason = ",".join(e.rejection_reasons) if e.rejection_reasons else "invalid"
                print(f"{rnd:>2} {'--':>8}  REJECTED ({reason})  {smi[:48]!r}")
                continue
            verdict = "usable" if usable else "REJECTED: " + ", ".join(
                broken or ["QED not improved over the start"])
            print(f"{rnd:>2} {e.qed:>8.4f} {e.molecular_weight:>7.1f} {e.logp:>6.2f} "
                  f"{e.sa_score:>5.2f} {e.similarity_to_start:>6.3f} {e.n_alerts:>6}  {verdict}")
            print(f"   {e.canonical_smiles}")

    score = final_score(
        start_eval=start_eval,
        round_evaluations=evaluations,
        constraints=constraints,
        scoring_cfg=scoring_cfg,
        reference_cfg=reference,
        target_cfg=target_cfg,
    )

    # Continuous diagnostics. Two runs can land in the same band and still be very different; these
    # fields let a reviewer see the difference without re-deriving anything.
    usable_rounds = [r for r in per_round if r["usable"]]
    best_round = max(usable_rounds, key=lambda r: r["objective_value"], default=None)
    det = score["optimality_score_details"]
    score["continuous_diagnostics"] = {
        "achievement_raw_unclipped": (
            None if start_eval.qed is None or det.get("best_successful_objective") is None
            else round((det["best_successful_objective"] - start_eval.qed)
                       / (float(reference["best_objective_in_window"]) - start_eval.qed), 6)
        ),
        "best_usable_objective": det.get("best_successful_objective"),
        "objective_gain_over_start": (
            None if det.get("best_successful_objective") is None
            else round(det["best_successful_objective"] - start_eval.qed, 6)
        ),
        "n_rounds": len(per_round),
        "n_rounds_with_a_parsed_submission": sum(1 for r in per_round if r["smiles"]),
        "n_rounds_valid": sum(1 for r in per_round if r["valid"]),
        "n_rounds_feasible": sum(1 for r in per_round if r["valid"] and not r["constraints_violated"]),
        "n_rounds_usable": len(usable_rounds),
        "best_round": best_round["round"] if best_round else None,
        "per_round_objective": [r["objective_value"] for r in per_round],
        "per_round_usable": [r["usable"] for r in per_round],
        "violation_histogram": _violation_histogram(per_round),
    }
    score["per_round"] = per_round

    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    # Sits next to the score rather than inside the diagnostics, because §16.6 requires it to travel
    # with the score: a reader who sees the number must see the completeness verdict in the same
    # glance, not two levels down.
    completeness = protocol_completeness(per_round, task_cfg.get("iteration") or {}, summary)
    score["protocol_completeness"] = completeness

    report = {
        "success": bool(score.get("success", False)),
        "protocol_completeness": completeness,
        "task_name": task_cfg["task_name"],
        "task_id": task_cfg["task_id"],
        "prompt_version": summary.get("prompt_version") or task_cfg.get("prompt_version"),
        "question_sha256": summary.get("question_sha256"),
        "graded_by": "grader/scripts/score_run.py",
        "graded_against": {
            "public_task_config": str(PUBLIC_TASK_CFG.relative_to(ROOT)),
            "grader_reference": str(REFERENCE_CFG.relative_to(ROOT)),
        },
        "fixed_iteration_budget": summary.get("fixed_iteration_budget", len(per_round)),
        "score": score,
        "target": {
            "optimization_target": target_cfg,
            "start_smiles": start_smiles,
            "constraints": constraints,
        },
        "checks": summary.get("checks", {}),
        "files": summary.get("files", {}),
        "start_molecule": start_eval.to_dict(),
        "candidates": [e.to_dict() for e in evaluations],
    }

    if not quiet:
        cd = score["continuous_diagnostics"]
        print(f"\nusable analogs           : {score['num_successful_optimized_analogs']}")
        print(f"best usable QED          : {det.get('best_successful_objective')}")
        print(f"achievement              : {det['achievement_fraction']}")
        print(f"feasible rounds          : {cd['n_rounds_feasible']} / {cd['n_rounds']}")
        print(f"best round               : {cd['best_round']}")
        pc = score["protocol_completeness"]
        print(f"protocol completeness    : {pc['verdict']} "
              f"({pc['rounds_with_a_candidate']}/{pc['declared_rounds']} rounds carried a candidate)")
        if pc["verdict"] != "COMPLETE":
            print(f"                           score is a {pc['score_interpretation']}; "
                  f"rerun_required={pc['rerun_required']}")
        print(f"TOTAL SCORE              : {score['total_score']} / 100")

    if write:
        out = run_dir / "final_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
        if not quiet:
            shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
            print(f"written -> {shown}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="a directory under runs/")
    ap.add_argument("--all", action="store_true", help="grade every run under runs/")
    ap.add_argument("--no-write", action="store_true", help="print without writing final_report.json")
    args = ap.parse_args()

    if args.all:
        runs = sorted(p for p in RUNS.iterdir() if p.is_dir())
        if not runs:
            raise SystemExit("no run directories found under runs/")
        rows = []
        for run in runs:
            report = grade(run, write=not args.no_write, quiet=True)
            model = (json.loads((run / "api_config_used.json").read_text(encoding="utf-8")).get("model")
                     if (run / "api_config_used.json").exists() else None)
            rows.append((model or "?", report["score"]["total_score"],
                         report["score"]["optimality_score_details"]["achievement_fraction"],
                         report["score"]["continuous_diagnostics"]["n_rounds_usable"], run.name))
        print(f"{'model':20s} {'score':>5} {'achievement':>12} {'usable':>7}  run")
        for model, total, ach, usable, name in rows:
            print(f"{model:20s} {total:>5} {ach:>12} {usable:>7}  {name}")
        return

    if not args.run_dir:
        raise SystemExit("usage: python grader/scripts/score_run.py runs/<run_dir> [--no-write] | --all")
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")
    grade(run_dir, write=not args.no_write)


if __name__ == "__main__":
    main()

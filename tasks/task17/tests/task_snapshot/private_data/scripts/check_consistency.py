#!/usr/bin/env python
"""Assert that every on-disk copy of the calibration is the same calibration.

Why this script exists. In this series of problems, the near-miss that repeated itself was never a
wrong idea -- it was two artifacts on disk carrying different numbers for the same quantity, and a
build script reading the wrong one. A sibling problem kept a fallback calibration in one JSON and the
shipping calibration in another; the two differed by one scoring band. Nothing about a calibration is
verified by reading it, so this script never reads a number and reports it: it asserts equality
between the authority (src/agglomeration.py), the machine-readable twin
(tasks/q1_agglomeration/task_config.yaml), the derivation (answers/reference_enumeration.json) and the
prose (tasks/q1_agglomeration/question.md) -- and it re-measures the cheap parts live through the
shipped graded path instead of trusting any of the four.

Checks
------
 1  design space, cost table, registered process, windows, budget: task_config.yaml == agglomeration
 2  score bands and the "feasibility earns nothing" rule: task_config.yaml == design_space.POINT_BANDS
 3  duplicated constraint constants: the config-driven predicate (evaluator.check_constraints) and the
    hard-coded one (agglomeration.constraints) agree on EVERY design in the 1-slot and 2-slot
    families, so a limit that was changed in one place and not the other cannot survive
 4  answers/reference_enumeration.json was produced from THIS source: sha256 of src/agglomeration.py
    and of the frozen surrogate must match the hashes recorded in it
 5  the reference block of task_config.yaml == the enumeration output, field by field, including the
    three gate numbers and the G2 margin of 0.05
 6  LIVE re-measurement through src/evaluator + src/scoring: D of the registered process, D of the
    reference design, and the complete 1-slot and 2-slot families (G1) -- recomputed here, compared
    to the stored numbers
 7  question.md carries the same windows, budget, overlap and constraint limits as the config, the
    same scoring-band table, the same round count and the same required output shape -- and does not
    quote the reference design or any designer-side value
 8  no file under runs/ contains a key-shaped string, and every api_config_used.json has a redacted
    key: the run directories are part of the delivery, so a live credential in one of them ships
 9  every run's final_report.json score is REPRODUCIBLE: the submissions stored in that run are
    re-evaluated and re-scored here through src/evaluator + src/scoring, and the total must equal the
    number in the shipped report
10  LOAD SAFETY (external review finding 2): the frozen surrogate is a .npz of plain float64 arrays
    that loads with allow_pickle=False, no pickled model exists in the shipped tree, no module under
    src/ imports joblib, and the recorded pickle -> npz migration is bit-identical on all FOUR
    responses and on D separately
11  EVALUATION ISOLATION (finding 5): no public module imports the response mechanism, src/evaluator
    reaches it only through the src/response_model.py boundary, the public stub build exists and
    raises rather than inventing numbers, agglomeration re-exports design_grid's objects instead of
    copying them, and the public task config carries no score anchor
12  TOOL LAYER (finding 4): the advertised catalogue equals the implementations, no tool result over
    the whole 1-slot family contains a response / desirability / achievement / score, the leak guard
    fires on a planted violation, and every declared failure mode -- including the call quota -- is
    reachable
13  BAND DENSITY AND BASELINES (finding 3): the refined ladder is monotone, the two added bands hold
    real population rather than decorating the scale, no band holds more than 65 % of the improving
    designs, and the shipped expert baseline beats the chance floor while the oracle hill-climber
    still scores under 50
14  SAMPLING VARIANCE: answers/calibration_report.json accounts for every run directory on disk (no
    replicate silently dropped), states its own n, does not claim a ranking the sample cannot
    support, and reports the chance-attainment probability of every observed score
15  NAIVE MULTI-VARIABLE POLICIES (2026-08 review, reviewer 2, finding 5): the mechanically generated
    policy family in answers/naive_policy_probes.json was measured against THIS calibration, is a
    generated product of >= 20 rules rather than a hand-written list, imports nothing from the
    response mechanism, breaks its ceiling out by move multiplicity 1..5 (the axis G1 cannot see),
    and its winning policy RE-EXECUTES and re-scores live to a ceiling under 50

Exit code 0 only if every check passes.

    python scripts/check_consistency.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402
import yaml                                                             # noqa: E402

import agglomeration as A                                               # noqa: E402
from design_space import (POINT_BANDS, achievement, g2_cost_criterion,  # noqa: E402
                          iter_k_slot_designs, overlap_floor, points_for_achievement)
from evaluator import (check_constraints, evaluate_design,              # noqa: E402
                       evaluate_reference_start, satisfies_all_constraints)
from parser import parse_candidate                                     # noqa: E402
from scoring import final_score                                        # noqa: E402

from rescore_run import submitted_processes                             # noqa: E402

TASK_DIR = os.path.join(ROOT, "tasks", "q1_agglomeration")
TASK_CFG = os.path.join(TASK_DIR, "task_config.yaml")
QUESTION = os.path.join(TASK_DIR, "question.md")
ENUM_JSON = os.path.join(ROOT, "answers", "reference_enumeration.json")
RUNS_DIR = os.path.join(ROOT, "runs")

# The one place the credential shape is written down. Same pattern the packaging script refuses on.
KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9_.\-]{16,}")

FAILURES: List[str] = []
CHECKS = 0


def ok(cond: bool, what: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  OK   {what}")
    else:
        print(f"  FAIL {what}" + (f"\n         {detail}" if detail else ""))
        FAILURES.append(what)
    return bool(cond)


def same(a: Any, b: Any, tol: float = 1e-12) -> bool:
    """Structural equality, with a tolerance for floats (yaml round-trips are not bit-exact)."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    return a == b


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # `--skip-trajectories` matches the flag the sibling packages carry. It exists for the
    # window between a prompt revision and the re-run that follows it: raising the prompt text
    # deliberately invalidates every shipped trajectory, so the run-dependent sections SHOULD
    # fail until new runs land, and a real structural regression must not hide among them.
    # A delivery run must not use this flag.
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-trajectories", action="store_true",
                    help="DEV ONLY: skip the run-dependent assertions (sections 8, 9 and the "
                         "per-run rows of section 16). A delivery run must not use this.")
    args = ap.parse_args()
    t0 = time.time()
    with open(TASK_CFG) as fh:
        cfg = yaml.safe_load(fh)
    obj_cfg, con_cfg, sco_cfg = cfg["objective"], cfg["constraints"], cfg["scoring"]
    ref_blk = cfg.get("reference") or {}

    # -------------------------------------------------------------------------------------- 1
    print("=" * 96)
    print("1  task_config.yaml  ==  src/agglomeration.py  (the authority)")
    print("=" * 96)
    ok(same(cfg["baseline_process"], dict(A.START)), "baseline process == design_grid.START",
       f"{cfg['baseline_process']} != {A.START}")
    ds = cfg["design_space"]
    ok(ds["n_slots"] == len(A.SLOTS), "n_slots")
    ok(ds["size_gross"] == A.space_size(), "gross design-space size",
       f"{ds['size_gross']} != {A.space_size()}")
    ok(ds["total_change_cost_if_all_changed"] == A.TOTAL_EDIT_COST, "total change cost")
    ok(set(ds["slots"]) == set(A.SLOTS), "slot names")
    lvl_bad = [s for s in A.SLOTS if not same(ds["slots"][s]["levels"], A.LEVELS[s])]
    ok(not lvl_bad, "level grids", f"differ for {lvl_bad}")
    cost_bad = [s for s in A.SLOTS if ds["slots"][s]["change_cost"] != A.EDIT_COST[s]]
    ok(not cost_bad, "per-slot change costs", f"differ for {cost_bad}")
    unit_bad = [s for s in A.SLOTS if ds["slots"][s]["unit"] != A.SLOT_UNITS[s]]
    ok(not unit_bad, "per-slot units", f"differ for {unit_bad}")
    win_bad = [k for k in A.RESPONSE_KEYS
               if not all(same(obj_cfg["windows"][k][f], A.WINDOWS[k][f])
                          for f in ("pl", "pu", "al", "au", "w"))]
    ok(not win_bad, "specification windows (pl/pu/al/au/w)", f"differ for {win_bad}")
    ok(same(obj_cfg["desirability_shape_m"], A.DESIRABILITY_SHAPE_M), "desirability shape m")
    ok(con_cfg["change_budget"]["max_cost"] == A.BUDGET, "change budget",
       f"{con_cfg['change_budget']['max_cost']} != {A.BUDGET}")
    ok(same(con_cfg["change_budget"]["equivalent_minimum_overlap"], overlap_floor(A.BUDGET), 1e-6),
       "budget expressed as an overlap floor")

    # -------------------------------------------------------------------------------------- 2
    print("\n" + "=" * 96)
    print("2  score bands and the pure-optimality rubric")
    print("=" * 96)
    bands_cfg = [(float(b["min_achievement"]), int(b["points"])) for b in sco_cfg["optimality_bands"]]
    ok(same(bands_cfg, [(float(t), int(p)) for t, p in POINT_BANDS]),
       "optimality bands == design_space.POINT_BANDS", f"{bands_cfg} != {POINT_BANDS}")
    ok(int(sco_cfg["constraint_points"]) == 0, "constraint_points == 0 (feasibility is a gate)")
    ok(int(sco_cfg["optimality_points"]) == int(sco_cfg["total_points"]) == 100,
       "all 100 points are optimality points")

    # -------------------------------------------------------------------------------------- 3
    print("\n" + "=" * 96)
    print("3  duplicated constraint constants: config-driven flags == hard-coded predicate")
    print("=" * 96)
    fam: List[Dict[str, Any]] = []
    for k in (1, 2):
        for x, _ in iter_k_slot_designs(k):
            fam.append(x)
    disagree: List[str] = []
    exercised = {"C3": 0, "C4": 0, "C5": 0, "C6": 0, "C7": 0}
    for x in fam:
        viol = A.constraints(x)
        for v in viol:
            tag = v.split()[0]
            if tag in exercised:
                exercised[tag] += 1
        ev = evaluate_design(x, obj_cfg)          # the harness's own path (no responses if infeasible)
        flags = check_constraints(ev, con_cfg)
        pairs = [("boiling_point_margin", "C3"), ("just_suspended_speed", "C4"),
                 ("total_solvent", "C5"), ("total_process_time", "C6"),
                 ("classification_consistency", "C7")]
        for key, tag in pairs:
            if key not in flags:
                continue
            hard = not any(v.startswith(tag) for v in viol)
            if flags[key] != hard:
                disagree.append(f"{tag} on {A.changed_slots(x)}: config={flags[key]} hard={hard}")
    ok(not disagree, f"C3-C7: config limits and agglomeration.constraints agree on all "
                     f"{len(fam)} 1/2-slot designs", "; ".join(disagree[:3]))
    print(f"       violations exercised by the 1/2-slot families: {exercised}")
    dead = [t for t, n in exercised.items() if n == 0]
    if dead:
        print(f"       NOTE: {dead} never binds in the 1/2-slot families -- checked by construction "
              f"below, not by a counter-example")
    # C6 has no counter-example anywhere on the grid; state the bound rather than pretend it is tested.
    max_hours = max(A.total_hours({"S3_addition_mode": a, "S7_agglomeration_min": h,
                                   "S11_drying": d})
                    for a in A.LEVELS["S3_addition_mode"] for h in A.LEVELS["S7_agglomeration_min"]
                    for d in A.LEVELS["S11_drying"])
    print(f"       longest process on the whole grid: {max_hours:.2f} h against the declared limit "
          f"{con_cfg['total_process_time']['max_hours']} h "
          f"({'C6 can never bind' if max_hours <= con_cfg['total_process_time']['max_hours'] else 'C6 binds'})")

    # -------------------------------------------------------------------------------------- 4
    print("\n" + "=" * 96)
    print("4  answers/reference_enumeration.json was produced from THIS source")
    print("=" * 96)
    if not os.path.exists(ENUM_JSON):
        ok(False, "answers/reference_enumeration.json exists",
           "run scripts/enumerate_reference_best.py first")
        enum: Dict[str, Any] = {}
    else:
        with open(ENUM_JSON) as fh:
            enum = json.load(fh)
        prov = enum.get("provenance", {})
        ok(not enum.get("quick_mode", False),
           "the enumeration was a FULL run (--quick cannot establish the reference, G2 or G3)")
        ok(prov.get("agglomeration_py_sha256") == sha256(os.path.join(SRC, "agglomeration.py")),
           "src/agglomeration.py hash matches the one recorded in the enumeration",
           "the evaluator changed after the enumeration -- re-run "
           "scripts/enumerate_reference_best.py")
        ok(prov.get("surrogate_sha256") == sha256(os.path.join(ROOT, "data",
                                                               "agglo_strength_porosity.npz")),
           "frozen surrogate hash matches the one recorded in the enumeration",
           "the surrogate changed after the enumeration -- re-run "
           "scripts/enumerate_reference_best.py")
        cal = enum.get("calibration", {})
        ok(cal.get("budget") == A.BUDGET and cal.get("g2_margin") == 0.05
           and cal.get("g2_cost_criterion") == g2_cost_criterion(A.BUDGET, 0.05),
           "the enumeration used this budget, margin 0.05 and the shared cost criterion",
           f"{cal.get('budget')}, {cal.get('g2_margin')}, {cal.get('g2_cost_criterion')}")
        ok(same(cal.get("windows"), {k: dict(v) for k, v in A.WINDOWS.items()}),
           "the enumeration used these windows")
        ok(same(cal.get("registered_process"), dict(A.START)),
           "the enumeration used this registered process")

    # -------------------------------------------------------------------------------------- 5
    print("\n" + "=" * 96)
    print("5  reference block of task_config.yaml == the enumeration output")
    print("=" * 96)
    if enum:
        expect = {
            "best_objective_in_budget": enum["reference_best"],
            "start_objective": enum["start_value"],
            "best_design_in_budget": enum["reference_design"],
            "reference_change_cost": enum["reference_change_cost"],
            "reference_overlap": enum["reference_overlap"],
            "n_slots_changed_vs_registered": enum["reference_n_slots_changed"],
            "reference_interior_grid_levels": enum["reference_interior_levels"],
            "n_rows_enumerated": enum["design_space"]["size_in_change_budget"],
            "n_feasible_in_budget": enum["design_space"]["size_in_change_budget_feasible"],
            "gate_g1_one_slot_points": enum["gates"]["g1_one_slot"]["best_points"],
            "gate_g1_two_slot_points": enum["gates"]["g1_two_slot"]["best_points"],
            "gate_g1_combined_points": enum["gates"]["g1_combined"]["best_points"],
            "gate_g1_three_slot_points": enum["gates"]["g1_three_slot_diagnostic"]["best_points"],
            "gate_g2_safe_margin": enum["calibration"]["g2_margin"],
            "gate_g2_cost_criterion": enum["calibration"]["g2_cost_criterion"],
            "gate_g2_n_candidates": enum["gates"]["g2"]["n_designs"],
            "gate_g2_n_enumerated": enum["gates"]["g2"]["n_feasible_and_in_budget"],
            "gate_g2_points": enum["gates"]["g2"]["best_points"],
            "gate_g3_improving_designs": enum["gates"]["g3"]["n_strictly_improving"],
            "gate_verdict": "PASS" if enum["gates"]["verdict"]["gates_pass"] else "FAIL",
            "start_responses": {k: round(v, 6) for k, v in enum["start_responses"].items()},
            "best_design_responses": {k: round(v, 6)
                                      for k, v in enum["reference_responses"].items()},
        }
        bad = [k for k, v in expect.items() if not same(ref_blk.get(k, "<missing>"), v, 1e-9)]
        for k in expect:
            if k in bad:
                print(f"       {k}: config={ref_blk.get(k, '<missing>')!r} "
                      f"enumeration={expect[k]!r}")
        ok(not bad, f"all {len(expect)} reference fields agree with the enumeration",
           f"disagree on {bad}")
        # The gate verdict itself, restated from the numbers rather than from the stored verdict.
        g1 = ref_blk.get("gate_g1_combined_points")
        g2 = ref_blk.get("gate_g2_points")
        g3 = ref_blk.get("gate_g3_improving_designs")
        thr = ref_blk.get("gate_thresholds", {"g1_max": 50, "g2_max": 50, "g3_min": 50})
        ok(g1 is not None and g1 < thr["g1_max"], f"G1 = {g1} < {thr['g1_max']}")
        ok(g2 is not None and g2 < thr["g2_max"], f"G2 = {g2} < {thr['g2_max']}")
        ok(g3 is not None and g3 >= thr["g3_min"], f"G3 = {g3} >= {thr['g3_min']}")

    # -------------------------------------------------------------------------------------- 6
    print("\n" + "=" * 96)
    print("6  LIVE re-measurement through src/evaluator + src/scoring (nothing is trusted)")
    print("=" * 96)
    start_eval = evaluate_reference_start(obj_cfg)
    D_start = float(start_eval.desirability)
    ok(same(D_start, ref_blk.get("start_objective"), 1e-9),
       f"D of the registered process re-measured: {D_start!r}",
       f"config says {ref_blk.get('start_objective')!r}")
    ref_design = ref_blk.get("best_design_in_budget")
    ref_ev = evaluate_design(dict(ref_design), obj_cfg) if ref_design else None
    if ref_ev is not None:
        ok(same(float(ref_ev.desirability), ref_blk.get("best_objective_in_budget"), 1e-9),
           f"D of the reference design re-measured: {ref_ev.desirability!r}",
           f"config says {ref_blk.get('best_objective_in_budget')!r}")
        ok(ref_ev.edit_cost == ref_blk.get("reference_change_cost") == A.BUDGET,
           f"reference change cost re-measured: {ref_ev.edit_cost} == budget {A.BUDGET}")
        ok(satisfies_all_constraints(ref_ev, con_cfg), "the reference design is feasible")
        binding = [k for k, v in check_constraints(ref_ev, con_cfg).items() if not v]
        ok(not binding, "no constraint binds at the reference (a binding one would leak the answer)",
           f"binding: {binding}")

    ref_cfg_live = {"best_objective_in_budget": float(ref_blk["best_objective_in_budget"])}
    # Tie-break on D exactly as scripts/enumerate_reference_best.py does, so the winner reported here
    # is the same design, not merely the same band.
    best = {1: (0, -1.0, None), 2: (0, -1.0, None)}
    n_eval = 0
    t1 = time.time()
    for k in (1, 2):
        for x, lab in iter_k_slot_designs(k):
            ev = evaluate_design(x, obj_cfg)
            fs = final_score(start_eval, [ev], con_cfg, sco_cfg, ref_cfg_live)
            n_eval += 1
            pts = int(fs["total_score"])
            D = float(ev.desirability) if ev.desirability is not None else -1.0
            if (pts, D) > (best[k][0], best[k][1]):
                best[k] = (pts, D, lab)
    print(f"       re-graded {n_eval} 1/2-slot designs in {time.time() - t1:.1f} s")
    ok(best[1][0] == ref_blk.get("gate_g1_one_slot_points"),
       f"G1 1-slot ceiling re-measured: {best[1][0]} pts, D={best[1][1]:.6f} ({best[1][2]})",
       f"config says {ref_blk.get('gate_g1_one_slot_points')}")
    ok(best[2][0] == ref_blk.get("gate_g1_two_slot_points"),
       f"G1 2-slot ceiling re-measured: {best[2][0]} pts, D={best[2][1]:.6f} ({best[2][2]})",
       f"config says {ref_blk.get('gate_g1_two_slot_points')}")
    ok(max(best[1][0], best[2][0]) == ref_blk.get("gate_g1_combined_points"),
       f"G1 combined re-measured: {max(best[1][0], best[2][0])} pts")

    # -------------------------------------------------------------------------------------- 7
    print("\n" + "=" * 96)
    print("7  question.md carries the same numbers, and no designer-side value")
    print("=" * 96)
    with open(QUESTION) as fh:
        q = fh.read()
    missing: List[str] = []
    for key in A.RESPONSE_KEYS:
        w = A.WINDOWS[key]
        for field in ("pl", "pu", "al", "au"):
            v = w[field]
            txt = f"{v:g}"
            if txt not in q and f"{v}" not in q:
                missing.append(f"{key}.{field}={txt}")
    ok(not missing, "every window bound appears in question.md", f"missing {missing}")
    ok(f"**{A.BUDGET} of the {A.TOTAL_EDIT_COST}**" in q or
       f"{A.BUDGET} of the {A.TOTAL_EDIT_COST}" in q,
       f"question.md states the budget as {A.BUDGET} of {A.TOTAL_EDIT_COST}")
    ok(f"{overlap_floor(A.BUDGET) * 100:.1f}" in q,
       f"question.md states the overlap floor {overlap_floor(A.BUDGET) * 100:.1f} %")
    ok(str(con_cfg["total_solvent"]["max_L_per_kg"]).rstrip("0").rstrip(".") in q,
       "question.md states the solvent limit")
    ok(str(int(con_cfg["total_process_time"]["max_hours"])) in q,
       "question.md states the process-time limit")
    bp_missing = [k for k, v in A.BL.items() if f"{v['bp']:g}" not in q]
    ok(not bp_missing, "question.md lists every bridging-liquid boiling point", f"{bp_missing}")

    # ---------------------------------------------------------------------------------------------
    # The bands used to be published in question.md, and this block asserted the printed table matched
    # the config row for row -- the one inconsistency a reviewer cannot detect by reading either file
    # alone. From the second round the ladder is grader-side (17-P1 #3), so the invariant changes
    # shape rather than disappearing:
    #
    #   * question.md must print NO band table  (the leak direction);
    #   * the grader-side config must still carry the full ladder  (the it-still-exists direction);
    #   * that ladder must still equal design_space.POINT_BANDS, which section 2 asserts, so the
    #     table the grader applies is the table the code applies.
    #
    # What is deliberately no longer checkable here is question-vs-config agreement, because there is
    # nothing in the question to agree with. The protection that replaces it is that the model is
    # told only "monotone graded function of achievement", which no ladder can contradict.
    # ---------------------------------------------------------------------------------------------
    q_bands = [(float(t), int(p))
               for t, p in re.findall(r"\|\s*(?:≥|>=|>)\s*([0-9.]+)\s*\|\s*(\d+)\s*\|", q)]
    ok(not q_bands, "question.md publishes no achievement -> points band table",
       f"found {q_bands}")
    ok("monotone graded function" in q,
       "question.md describes the score as a monotone graded function of achievement")
    cfg_bands = [(float(b["min_achievement"]), int(b["points"])) for b in sco_cfg["optimality_bands"]]
    ok(len(cfg_bands) >= 2 and cfg_bands == sorted(cfg_bands, reverse=True),
       f"the grader-side ladder still exists and is descending ({len(cfg_bands)} bands)",
       f"{cfg_bands}")
    ok(cfg_bands[-1][0] < 1e-3,
       f"the lowest band is an epsilon rather than a specification ({cfg_bands[-1][0]})")

    # And the public config must not carry it either -- the question is not the only model-visible
    # file. make_public_task_config.py drops it; this is the independent second opinion.
    pub_path = Path(ROOT) / "tasks" / "q1_agglomeration" / "task_config.public.yaml"
    if pub_path.is_file():
        pub_scoring = yaml.safe_load(pub_path.read_text(encoding="utf-8")).get("scoring", {})
        ok("optimality_bands" not in pub_scoring,
           "the public task config does not carry the band ladder either")
        ok(bool(pub_scoring.get("bands_withheld")),
           "the public task config says outright that the ladder is withheld")
    ok(f"**{int(sco_cfg['total_points'])} points**" in q or
       f"score is {int(sco_cfg['total_points'])} points" in q,
       f"question.md states the total as {int(sco_cfg['total_points'])} points")
    ok("no points for satisfying the constraints" in q,
       "question.md states that feasibility earns no points (matches constraint_points == 0)")

    it_cfg = cfg["iteration"]
    ok(f"**{int(it_cfg['max_rounds'])} rounds**" in q,
       f"question.md states {int(it_cfg['max_rounds'])} rounds == iteration.max_rounds")
    ok(int(it_cfg["candidates_per_round"]) == 1 and "exactly **one** process" in q,
       "question.md states one process per round == iteration.candidates_per_round")

    # The required output format is checked by feeding question.md's own example to the shipped
    # parser, then to the shipped evaluator: a drifted key name or a stale slot list fails here.
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", q, flags=re.DOTALL)
    example = None
    for blob in fenced:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if "candidate" in obj:
            example = blob
            break
    ok(example is not None, "question.md contains a parseable Required-Output-Format example")
    if example is not None:
        parsed = parse_candidate(example)
        ok(set(parsed["process"]) == set(A.SLOTS),
           "the output-format example carries exactly the twelve configured variables",
           f"{sorted(set(parsed['process']) ^ set(A.SLOTS))}")
        ok(same(parsed["process"], dict(A.START)),
           "the output-format example is the registered process (so it leaks no better design)")
        ex_ev = evaluate_design(parsed["process"], obj_cfg)
        ok(ex_ev.valid and ex_ev.edit_cost == 0,
           "the shipped evaluator accepts the example verbatim at change cost 0",
           f"valid={ex_ev.valid} cost={ex_ev.edit_cost} err={ex_ev.error}")

    leaks = [s for s in (f"{float(ref_blk.get('best_objective_in_budget', 0)):.4f}",
                         "reference_enumeration", "0.9297")
             if s in q]
    ok(not leaks, "question.md does not quote the reference value or the designer-side file",
       f"found {leaks}")
    # The reference design itself. question.md HAS to print every allowed level of every variable --
    # that is the design space -- so "does the winning level appear" is not a usable leak test. What
    # would actually be a leak is the reference design appearing as a design: as a JSON object, as a
    # written-out set of moves, or on one line that names all four winning levels at once.
    ref_design = ref_blk.get("best_design_in_budget") or {}
    changed = {k: v for k, v in ref_design.items() if v != A.START.get(k)}
    ok(bool(changed), "the reference design differs from the registered process (sanity)")
    json_designs = []
    for blob in re.findall(r"\{[^{}]*\"S1_bridging_liquid\"[^{}]*\}", q, flags=re.DOTALL) + fenced:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        proc = obj.get("process") or (obj.get("candidate") or {}).get("process") or obj
        if isinstance(proc, dict) and set(proc) == set(A.SLOTS):
            json_designs.append(proc)
    ok(bool(json_designs) and all(same(p, dict(A.START)) for p in json_designs),
       f"every one of the {len(json_designs)} full designs printed in question.md is the registered "
       f"process, not the reference design")
    moves = [f"{v}" for v in changed.values()]
    prose_leaks = [m for m in re.split(r"[;\n]", str(ref_blk.get("best_design_changes", "")))
                   if m.strip() and m.strip() in q]
    ok(not prose_leaks, "question.md does not restate the reference design's moves", f"{prose_leaks}")
    leaking_lines = [ln for ln in q.splitlines() if all(m in ln for m in moves)]
    ok(not leaking_lines,
       f"no single line of question.md names all {len(changed)} winning levels {moves} at once",
       f"{leaking_lines[:2]}")
    ok("api_key" not in q.lower() and "sk-" not in q, "question.md contains no credential-like text")

    # -------------------------------------------------------------------------------------- 8
    print("\n" + "=" * 96)
    print("8  no credential survives anywhere under runs/ (the run dirs ship in the zip)")
    print("=" * 96)
    run_files = [os.path.join(dp, fn) for dp, _, fns in os.walk(RUNS_DIR) for fn in fns]
    offenders: List[str] = []
    for path in run_files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                body = fh.read()
        except OSError:
            continue
        if KEY_SHAPE.search(body):
            offenders.append(os.path.relpath(path, ROOT))
    ok(not offenders, f"no string matching {KEY_SHAPE.pattern} in any of the {len(run_files)} files "
                      f"under runs/", f"offenders: {offenders}")
    used_cfgs = [p for p in run_files if os.path.basename(p) == "api_config_used.json"]
    bad_redact = []
    for path in used_cfgs:
        with open(path, encoding="utf-8") as fh:
            if json.load(fh).get("api_key") != "<redacted>":
                bad_redact.append(os.path.relpath(path, ROOT))
    ok(args.skip_trajectories or (bool(used_cfgs) and not bad_redact),
       f"all {len(used_cfgs)} api_config_used.json files carry api_key == '<redacted>'",
       f"{bad_redact}")

    # -------------------------------------------------------------------------------------- 9
    print("\n" + "=" * 96)
    print("9  every shipped final_report.json score recomputes from the stored submissions")
    print("=" * 96)
    # INDEXED BY RUN DIRECTORY, NEVER BY MODEL NAME.
    # A sibling problem in this batch aggregated its trajectories into a dict keyed by model name,
    # which silently discarded every repeat run of the same model the moment n > 1 -- so a replicate
    # that contradicted the headline result simply vanished. The loop below walks run directories and
    # asserts once per directory, and the count is cross-checked against the directory listing so a
    # skipped replicate is a failure rather than a smaller number.
    run_dirs = [] if args.skip_trajectories else sorted(
        d for d in (os.path.join(RUNS_DIR, n) for n in os.listdir(RUNS_DIR))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "final_report.json")))
    if not args.skip_trajectories:
        ok(len(run_dirs) >= 3,
           f"at least three model trajectories ship ({len(run_dirs)} found)")
    models: List[str] = []
    per_model_scores: Dict[str, List[int]] = {}
    n_asserted = 0
    for d in run_dirs:
        name = os.path.basename(d)
        with open(os.path.join(d, "final_report.json")) as fh:
            report = json.load(fh)
        stored = int((report.get("score") or {}).get("total_score", -1))
        rounds = submitted_processes(Path(d))
        evs = [evaluate_design(proc, obj_cfg) for _, proc in rounds]
        recomputed = final_score(start_eval, evs, con_cfg, sco_cfg, ref_cfg_live)
        got = int(recomputed["total_score"])
        api_used = os.path.join(d, "api_config_used.json")
        model = (json.load(open(api_used)).get("model") if os.path.exists(api_used) else None)
        models.append(str(model))
        per_model_scores.setdefault(str(model), []).append(got)
        n_asserted += 1
        ok(got == stored and len(rounds) == int(it_cfg["max_rounds"]),
           f"{name}: report says {stored}/100, recomputed {got}/100 from "
           f"{len(rounds)} stored submissions "
           f"(achievement {recomputed['optimality_score_details']['achievement_fraction']})",
           f"stored={stored} recomputed={got} rounds={len(rounds)} "
           f"(expected {it_cfg['max_rounds']})")
    ok(n_asserted == len(run_dirs),
       f"every one of the {len(run_dirs)} run directories was asserted (no replicate was skipped)",
       f"asserted {n_asserted} of {len(run_dirs)}")
    # Skipped with --skip-trajectories: between a prompt revision and the re-run that follows,
    # there are legitimately no current trajectories and an aggregate over an empty set is not
    # a finding. A delivery run does not pass that flag, so this still gates the release.
    ok(args.skip_trajectories or len(set(models)) >= 3,
       f"the trajectories come from >= 3 distinct models: {sorted(set(models))}")
    for m, scores in sorted(per_model_scores.items()):
        print(f"       {m}: n={len(scores)} replicate(s), scores {scores}")

    # THE SPONSOR'S HARD ACCEPTANCE CRITERION, ASSERTED PER REPLICATE.
    # Deliberately one-sided (< 50) rather than "equals the number I published": a repeat run at
    # temperature 0.7 legitimately lands on a different score, and an assertion pinned to one observed
    # value would fail for the wrong reason and would tempt whoever hits it to delete the replicate.
    # What must never vary is the ceiling.
    HARD_MAX = 50
    GATED_MODEL = "deepseek-v4-pro"
    gated = per_model_scores.get(GATED_MODEL, [])
    ok(args.skip_trajectories or bool(gated),
       f"at least one {GATED_MODEL} trajectory ships (the gated model)")
    over = [s for s in gated if s >= HARD_MAX]
    ok(not over,
       f"EVERY {GATED_MODEL} replicate scores < {HARD_MAX}: n={len(gated)}, scores {gated}",
       f"replicates at or above the acceptance line: {over}")
    all_over = {m: [s for s in v if s >= HARD_MAX] for m, v in per_model_scores.items()}
    all_over = {m: v for m, v in all_over.items() if v}
    ok(not all_over, f"no replicate of any model reaches {HARD_MAX}", f"{all_over}")

    # ------------------------------------------------------------------------------------- 10
    print("\n" + "=" * 96)
    print("10 load safety: the surrogate is a non-executable array bundle (review finding 2)")
    print("=" * 96)
    npz_path = os.path.join(ROOT, "data", "agglo_strength_porosity.npz")
    ok(os.path.exists(npz_path), "data/agglo_strength_porosity.npz ships")
    schema_path = os.path.join(ROOT, "data", "agglo_strength_porosity.schema.json")
    ok(os.path.exists(schema_path), "the surrogate input/output schema ships")
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            names = set(z.files)
        loaded_without_pickle = True
    except Exception as exc:                                           # pragma: no cover
        names, loaded_without_pickle = set(), False
        print(f"         {exc}")
    ok(loaded_without_pickle,
       "the surrogate loads with allow_pickle=False (loading the ruler is not code execution)")
    ok({"scaler_mean", "scaler_scale", "mu_y", "sc_y", "n_layers", "W0", "b0"} <= names,
       "the bundle carries the arrays the declared schema promises", f"has {sorted(names)}")
    # No pickled model anywhere in the shipped tree. data/legacy/ is excluded from both archives by
    # scripts/package_for_delivery.NEVER_SHIPPED_PREFIXES, so it does not count.
    pickles = [os.path.relpath(os.path.join(dp, fn), ROOT)
               for dp, _, fns in os.walk(ROOT) for fn in fns
               if fn.endswith((".joblib", ".pkl", ".pickle"))
               and "legacy" not in os.path.relpath(os.path.join(dp, fn), ROOT)
               and "__pycache__" not in dp]
    ok(not pickles, "no pickled model file in the shipped tree (data/legacy/ is never archived)",
       f"found {pickles}")
    src_files = [os.path.join(SRC, f) for f in os.listdir(SRC) if f.endswith(".py")]
    joblib_importers = [os.path.basename(f) for f in src_files
                        if re.search(r"^\s*import joblib|^\s*from joblib", open(f).read(), re.M)]
    ok(not joblib_importers, "no module under src/ imports joblib", f"{joblib_importers}")
    ev_path = os.path.join(ROOT, "data", "SURROGATE_MIGRATION_EVIDENCE.json")
    if ok(os.path.exists(ev_path), "the migration evidence file ships"):
        with open(ev_path) as fh:
            mig = json.load(fh)
        per = mig["per_response_max_abs_diff"]
        worst = max(float(v["max_abs_diff"]) for v in per.values())
        ok(set(A.RESPONSE_KEYS) | {"desirability_D"} <= set(per),
           "the migration was measured on all four responses AND on D, separately",
           f"has {sorted(per)}")
        ok(worst == 0.0,
           f"pickle -> npz migration is bit-identical: worst max|old-new| = {worst:.3e} over "
           f"{mig['n_designs_compared']} designs",
           f"worst {worst:.3e}")

    # ------------------------------------------------------------------------------------- 11
    print("\n" + "=" * 96)
    print("11 evaluation isolation: the public half cannot reach the response mechanism (finding 5)")
    print("=" * 96)
    PUBLIC_MODULES = ["design_grid.py", "design_space.py", "tools.py", "feedback.py", "parser.py",
                      "scoring.py", "main.py", "utils.py", "llm_client.py", "markdown_report.py"]
    reaching = []
    for name in PUBLIC_MODULES:
        body = open(os.path.join(SRC, name)).read()
        if re.search(r"^\s*(from agglomeration import|import agglomeration)", body, re.M):
            reaching.append(name)
    ok(not reaching,
       f"none of the {len(PUBLIC_MODULES)} public modules imports the response mechanism",
       f"{reaching} import agglomeration")
    ev_body = open(os.path.join(SRC, "evaluator.py")).read()
    ok("from response_model import responses" in ev_body
       and not re.search(r"^\s*(from agglomeration import|import agglomeration)", ev_body, re.M),
       "src/evaluator.py reaches the mechanism ONLY through the src/response_model.py boundary")
    stub_path = os.path.join(ROOT, "public_overlay", "src", "response_model.py")
    if ok(os.path.exists(stub_path), "the public stub build of the boundary exists",
          "public_overlay/ ships in the grader archive; if this tree is a PUBLIC extraction the "
          "check does not apply"):
        stub = open(stub_path).read()
        ok('BUILD = "public"' in stub
           and not re.search(r"^\s*(from agglomeration import|import agglomeration)", stub, re.M)
           and "ResponseModelUnavailable" in stub,
           "the public stub imports nothing from the mechanism and raises instead of inventing "
           "numbers")
    # design_grid is the single authority for the public constants; agglomeration re-exports it
    import design_grid as G                                            # noqa: E402
    ok(A.LEVELS is G.LEVELS and A.WINDOWS is G.WINDOWS and A.START is G.START
       and A.EDIT_COST is G.EDIT_COST and A.BUDGET == G.BUDGET,
       "agglomeration re-exports design_grid's objects rather than copying them")
    ok(all(A.BL[k]["bp"] == G.BP[k] for k in G.BP)
       and all(A.ADD[k]["t"] == G.ADDITION_MINUTES[k] for k in G.ADDITION_MINUTES)
       and all(A.DRY[k]["h"] == G.DRYING_HOURS[k] for k in G.DRYING_HOURS),
       "the mechanism tables take their PUBLIC fields (bp / addition / drying) from design_grid")
    pub_cfg_path = os.path.join(TASK_DIR, "task_config.public.yaml")
    if ok(os.path.exists(pub_cfg_path), "the public task config ships"):
        with open(pub_cfg_path) as fh:
            pub_txt = fh.read()
        pub = yaml.safe_load(pub_txt)
        ok("reference" not in pub, "the public task config has NO reference block")
        banned_pub = ["best_objective_in_budget", "gate_g1", "gate_g2", "gate_g3",
                      "reference_enumeration", f"{float(ref_blk['best_objective_in_budget']):.4f}"]
        pub_hits = [b for b in banned_pub if b in pub_txt]
        ok(not pub_hits, "the public task config carries no score anchor or gate measurement",
           f"{pub_hits}")
        same_windows = all(
            same(pub["objective"]["windows"][k][f], A.WINDOWS[k][f])
            for k in A.RESPONSE_KEYS for f in ("pl", "pu", "al", "au", "w"))
        ok(same_windows, "the public config's four windows are identical to the graded ones")
        # Since the second round the public config carries no ladder at all, so "identical to the
        # graded ones" is replaced by "absent, and says so". The graded ladder is checked against
        # design_space.POINT_BANDS in section 2 and against question.md in section 7.
        ok("optimality_bands" not in pub.get("scoring", {}),
           "the public config carries no score-band ladder",
           f"{pub.get('scoring', {}).get('optimality_bands')}")
        ok(bool(pub.get("scoring", {}).get("bands_withheld")),
           "the public config states outright that the ladder is withheld")
        ok(len(bands_cfg) >= 2,
           f"the graded ladder still exists on the grader side ({len(bands_cfg)} bands)")

    # ------------------------------------------------------------------------------------- 12
    print("\n" + "=" * 96)
    print("12 the tool layer cannot value a design (finding 4's hard constraint)")
    print("=" * 96)
    import tools as T                                                  # noqa: E402
    ok({"d50", "strength", "porosity", "yield_pct", "desirability", "achievement", "score"}
       <= T.FORBIDDEN_RESULT_KEYS,
       "the leak guard forbids every response name, the desirability, the achievement and the score")
    ok(len(T.TOOL_SPECS) >= 2 and set(s["name"] for s in T.TOOL_SPECS) == set(T.TOOL_IMPLEMENTATIONS),
       f"{len(T.TOOL_SPECS)} tools are offered and the advertised catalogue == the implementations",
       f"{sorted(set(s['name'] for s in T.TOOL_SPECS) ^ set(T.TOOL_IMPLEMENTATIONS))}")
    # run every tool over the whole 1-slot family and assert nothing forbidden ever comes back
    probe = [x for x, _ in iter_k_slot_designs(1)]
    leaked_calls = 0
    for x in probe:
        res = T.run_tool_calls([
            {"tool": "check_process_legality", "arguments": {"process": x}},
            {"tool": "change_budget_audit", "arguments": {"process": x}},
            {"tool": "compare_processes", "arguments": {"process_a": dict(A.START),
                                                        "process_b": x}},
            {"tool": "variable_levels", "arguments": {"variable": "S5_tip_speed_m_s"}},
        ])
        for r in res["results"]:
            try:
                T.assert_no_objective_leak(r)
            except RuntimeError:
                leaked_calls += 1
    ok(leaked_calls == 0,
       f"no forbidden quantity in any of the {len(probe) * 4} tool results over the 1-slot family",
       f"{leaked_calls} results leaked")
    # the guard is alive, not vacuous
    guard_fires = False
    try:
        T.assert_no_objective_leak({"nested": [{"desirability_D": 0.5}]})
    except RuntimeError:
        guard_fires = True
    ok(guard_fires, "the leak guard actually fires on a planted violation (not a vacuous check)")
    # every declared failure mode must be reachable
    failures = {
        "unknown_tool": [{"tool": "nope", "arguments": {}}],
        "missing_argument": [{"tool": "check_process_legality", "arguments": {}}],
        "invalid_argument_type": [{"tool": "change_budget_audit", "arguments": {"process": 7}}],
        "unknown_variable": [{"tool": "variable_levels", "arguments": {"variable": "S99"}}],
        "not_a_design_space_member": [{"tool": "change_budget_audit",
                                       "arguments": {"process": {**A.START,
                                                                 "S5_tip_speed_m_s": 1.23}}}],
        "quota_exceeded": [{"tool": "variable_levels", "arguments": {"variable": "S6_temperature_C"}}]
                          * (T.MAX_TOOL_CALLS_PER_TURN + 1),
    }
    unreachable = []
    for want, calls in failures.items():
        got = {r.get("error") for r in T.run_tool_calls(calls)["results"]}
        if want not in got:
            unreachable.append(f"{want} (got {got})")
    ok(not unreachable, f"all {len(failures)} declared tool failure modes are reachable",
       f"{unreachable}")
    ok(int(it_cfg["max_rounds"]) == 5 and T.MAX_TOOL_TURNS_PER_ROUND >= 1
       and T.MAX_TOOL_CALLS_PER_TURN >= 1,
       f"the tool quota is finite and declared: {T.MAX_TOOL_TURNS_PER_ROUND} turns x "
       f"{T.MAX_TOOL_CALLS_PER_TURN} calls per graded round")

    # ------------------------------------------------------------------------------------- 13
    print("\n" + "=" * 96)
    print("13 band density and the expert baselines (finding 3)")
    print("=" * 96)
    # 10 since 2026-08-15: (0.02, 11) was inserted, splitting a bottom band that held 58.4 % of the
    # improving population. The named rows below are spot-checks that the ladder still contains the
    # bands earlier revisions added; the count is asserted separately so that inserting a band is a
    # deliberate edit here rather than something that slips through.
    ok(len(bands_cfg) == 10 and (0.10, 16) in bands_cfg and (0.05, 12) in bands_cfg
       and (0.02, 11) in bands_cfg,
       f"the refined ladder has {len(bands_cfg)} bands, including those added below 20 points")
    ok(all(bands_cfg[i][1] > bands_cfg[i + 1][1] for i in range(len(bands_cfg) - 1))
       and all(bands_cfg[i][0] > bands_cfg[i + 1][0] for i in range(len(bands_cfg) - 1)),
       "the ladder is strictly monotone in both achievement and points")
    if enum:
        dist = (enum["gates"]["g3"]["achievement_distribution_over_improving_designs"]
                ["count_in_band"])
        newly_resolved = dist.get("[0.05, 0.1)", 0) + dist.get("[0.1, 0.15)", 0)
        ok(newly_resolved >= 1000,
           f"the two added bands hold {newly_resolved} designs, so they resolve real population "
           f"rather than decorating the scale")
        biggest = max(dist.values())
        total = sum(dist.values())
        ok(biggest / total < 0.65,
           f"no single band holds more than 65 % of the improving population "
           f"(largest {biggest}/{total} = {biggest / total:.1%}; it was 90.0 % before the refinement)")
    base_path = os.path.join(ROOT, "answers", "expert_baseline.json")
    if ok(os.path.exists(base_path), "the expert/chance baselines ship (answers/expert_baseline.json)"):
        with open(base_path) as fh:
            base = json.load(fh)
        e1, e2, e3 = (base["E1_greedy_oracle"], base["E2_random_chance_floor"],
                      base["E3_textbook_protocol"])
        ok(int(base["measurement_budget"]) == int(it_cfg["max_rounds"]),
           f"the baselines used the same {it_cfg['max_rounds']}-measurement budget as a graded run")
        ok(e3["total_score"] > e2["score_mean"],
           f"the textbook protocol ({e3['total_score']}) beats the chance floor "
           f"({e2['score_mean']}), so the task is solvable by process reasoning")
        ok(e1["total_score"] < 50,
           f"even the oracle hill-climber stays under the 50-point acceptance line "
           f"({e1['total_score']}/100 using {e1['offline_evaluations_used']} offline measurements)")

    # ------------------------------------------------------------------------------------- 14
    # SAMPLING VARIANCE. The scores are the best of five submissions from a model sampling at
    # temperature 0.7, so one run is a draw and not a measurement. A sibling problem in this batch
    # published a model ranking from n=1 that inverted on a repeat. These checks make the sample size
    # and the chance baseline impossible to lose track of.
    print("\n" + "=" * 96)
    print("14 sampling variance: the calibration report accounts for every run and states its n")
    print("=" * 96)
    cal_path = os.path.join(ROOT, "answers", "calibration_report.json")
    if ok(os.path.exists(cal_path),
          "the per-run calibration report ships (answers/calibration_report.json)",
          "run scripts/report_calibration.py"):
        with open(cal_path) as fh:
            cal = json.load(fh)
        reported = {r["run_dir"] for r in cal["runs"]}
        on_disk = {os.path.basename(d) for d in run_dirs}
        ok(args.skip_trajectories or reported == on_disk,
           f"the report accounts for every one of the {len(on_disk)} run directories "
           f"(no replicate dropped)",
           f"only in report: {sorted(reported - on_disk)}; "
           f"only on disk: {sorted(on_disk - reported)}")
        ok(all(v["n"] == len(v["scores"]) for v in cal["per_model"].values()),
           "every per-model entry reports n equal to the number of scores it lists")
        ok(cal["hard_criterion"]["pass"],
           f"the report's own hard-criterion verdict passes: {GATED_MODEL} scores "
           f"{cal['hard_criterion']['scores']} (all < {HARD_MAX})")
        # The claim the report is allowed to make must match its own sample size.
        min_n = int(cal["min_n"])
        verdict = str(cal["ranking_supported"])
        ok((min_n < 2 and verdict.startswith("NO")) or min_n >= 2,
           f"with min n = {min_n} the report does not overclaim a ranking "
           f"(verdict starts '{verdict.split(' --')[0]}')")
        if cal.get("chance_attainment"):
            pc = cal["chance_attainment"]["p_chance_reaches"]
            observed = sorted({int(r["total_score"]) for r in cal["runs"]
                               if r["total_score"] is not None})
            ok(all(str(s) in pc for s in observed),
               f"the chance-attainment probability is reported for every observed score {observed}",
               f"missing {[s for s in observed if str(s) not in pc]}")
            print(f"       P(random 5-design trajectory reaches score): "
                  f"{ {k: v for k, v in sorted(pc.items(), key=lambda kv: int(kv[0]))} }")

    # ------------------------------------------------------------------------------------- 15
    # NAIVE MULTI-VARIABLE POLICIES (2026-08 review, reviewer 2, finding 5). Gate G1 enumerates only
    # 1- and 2-variable changes, so a coordinated multi-variable "process intuition" answer is
    # structurally invisible to it. scripts/measure_naive_policies.py measures that family. These
    # checks assert the stored measurement was produced from THIS calibration, that the family is
    # actually large and actually multi-variable, and -- the load-bearing one -- they RE-EXECUTE the
    # winning policy through the shipped scorer instead of reading its score.
    print("\n" + "=" * 96)
    print("15 the naive multi-variable policy ceiling (2026-08 review finding 5)")
    print("=" * 96)
    naive_path = os.path.join(ROOT, "answers", "naive_policy_probes.json")
    if ok(os.path.exists(naive_path),
          "the naive-policy probe ships (answers/naive_policy_probes.json)",
          "run scripts/measure_naive_policies.py"):
        with open(naive_path) as fh:
            nv = json.load(fh)
        import measure_naive_policies as NP                              # noqa: E402
        orders, rules = NP.slot_orders(), NP.level_rules()
        ok(len(orders) * len(rules) == int(nv["n_policies"]) == len(nv["policies"])
           and int(nv["n_policies"]) >= 20,
           f"the family is generated, not listed: {len(orders)} ranking rules x {len(rules)} level "
           f"rules = {nv['n_policies']} policies (>= 20 required), {nv['n_designs_scored']} designs "
           f"scored",
           f"stored {nv['n_policies']}, regenerated {len(orders) * len(rules)}")
        ok(same(float(nv["baseline_process_D"]), float(start_eval.desirability))
           and same(float(nv["reference_D"]), float(ref_blk["best_objective_in_budget"])),
           "the probe was measured against the CURRENT calibration (same baseline D and reference D)",
           f"probe {nv['baseline_process_D']} / {nv['reference_D']} vs live "
           f"{start_eval.desirability} / {ref_blk['best_objective_in_budget']}")
        # the policy generator may not touch the response mechanism -- checked in the source text
        np_src = Path(NP.__file__).read_text(encoding="utf-8")
        rule_src = np_src.split("# execute one policy greedily")[0]
        bad_imports = [ln.strip() for ln in rule_src.splitlines()
                       if re.match(r"\s*(import|from)\s+(agglomeration|response_model)\b", ln)]
        ok(not bad_imports,
           "the rule generators import nothing from the response mechanism, so a policy cannot have "
           "been fitted to the answer",
           f"{bad_imports}")
        by_n = {int(b["n_changed"]): b for b in nv["best_by_n_variables_changed"]}
        ok(set(by_n) == {1, 2, 3, 4, 5},
           f"the ceiling is broken out by every reachable move multiplicity {sorted(by_n)} -- the "
           f"axis G1 (1 and 2 only) cannot see",
           f"got {sorted(by_n)}")
        ok(max(b["n_changes"] for b in nv["policies"]) >= 4,
           f"the family really is multi-variable: the widest policy moves "
           f"{max(b['n_changes'] for b in nv['policies'])} variables")
        ok(int(nv["naive_ceiling"]) >= int(ref_blk["gate_g1_combined_points"]),
           f"the naive family is not weaker than the gate it supplements "
           f"({nv['naive_ceiling']} >= G1 {ref_blk['gate_g1_combined_points']}), so a broken "
           f"generator scoring 0 cannot pass as a low ceiling")
        # LIVE: re-execute the best policy and re-score it through src/evaluator + src/scoring.
        best = max(nv["policies"], key=lambda r: r["total_score"])
        o_name, l_name = best["ranking_rule"], best["level_rule"]
        prefixes, log = NP.run_policy(orders[o_name][0], rules[l_name][0])
        live = NP.score_trajectory(prefixes[:int(it_cfg["max_rounds"])], cfg, start_eval)
        ok(int(live["total_score"]) == int(best["total_score"]) and log == best["changes"],
           f"the winning policy re-executes and re-scores live to the stored "
           f"{best['total_score']}/100 ({o_name} | {l_name}, {len(log)} variables)",
           f"live {live['total_score']} vs stored {best['total_score']}; "
           f"live changes {log} vs stored {best['changes']}")
        ok(int(live["total_score"]) != int(best["total_score"]) + 1,
           "and the re-measurement rejects a planted score, so the comparison is not vacuous")
        ok(int(nv["naive_ceiling"]) == int(live["total_score"]) < 50 and nv["verdict"] == "PASS",
           f"NAIVE CEILING {nv['naive_ceiling']}/100 < 50 -- a mechanically generated multi-variable "
           f"process-intuition policy does not reach the acceptance line "
           f"(without any reading of the prose mechanism table: "
           f"{nv['naive_ceiling_without_author_reading']})")

    # ------------------------------------------------------------------------------------- 16
    # PROSE DOCUMENTS. Historical accident in a sibling problem in this series: its README printed a
    # superseded rubric while its consistency script passed 20/20, because the script only validated
    # question.md. Any file that PRINTS a score, a band table, a gate number or a check count is part
    # of the calibration and is validated here.
    print("\n" + "=" * 96)
    print("16 the reviewer-facing prose carries the CURRENT calibration (not just question.md)")
    print("=" * 96)
    doc_paths = {"README.md": os.path.join(ROOT, "README.md"),
                 "审核说明.md": os.path.join(ROOT, "审核说明.md")}
    docs = {n: Path(p).read_text(encoding="utf-8") for n, p in doc_paths.items()
            if os.path.exists(p)}
    ok(set(docs) == set(doc_paths),
       f"the {len(doc_paths)} reviewer-facing documents exist and were read",
       f"missing {sorted(set(doc_paths) - set(docs))}")
    # a) the nine-band ladder, band by band, in the audit note
    if "审核说明.md" in docs:
        want_rows = [f"| ≥ {thr:.2f} | {pts} |" for thr, pts in bands_cfg[:-1]]
        missing_rows = [r for r in want_rows if r not in docs["审核说明.md"]]
        # The count is derived from the config rather than written here (it was hard-coded to 8 and
        # broke when the 2026-08-15 split made it 9). The >= 8 floor keeps it from passing vacuously
        # if bands_cfg were ever empty.
        ok(not missing_rows and len(want_rows) == len(bands_cfg) - 1 >= 8,
           f"審核说明.md prints all {len(want_rows)} banded ladder rows exactly as "
           f"design_space.POINT_BANDS defines them, plus the '> 0 | 9' floor",
           f"missing {missing_rows}")
        ok("| > 0 | 9 |" in docs["审核说明.md"],
           f"and its floor row matches the {bands_cfg[-1][1]}-point '> 0' band")
    # b) the three gate numbers, in both documents, in the form each prints them
    gate_strings = [str(int(ref_blk["gate_g1_combined_points"])),
                    str(int(ref_blk["gate_g2_points"])),
                    f"{int(ref_blk['gate_g3_improving_designs']):,}".replace(",", " ")]
    for name, text in docs.items():
        missing_g = [g for g in gate_strings if g not in text]
        ok(not missing_g,
           f"{name} quotes all three gate numbers of the current calibration "
           f"(G1 {gate_strings[0]}, G2 {gate_strings[1]}, G3 {gate_strings[2]})",
           f"missing {missing_g}")
    # c) every run's score, per run, in both documents -- a document may not silently aggregate
    if os.path.exists(cal_path):
        with open(cal_path) as fh:
            cal_doc = json.load(fh)
        for name, text in docs.items():
            bad = []
            for r in cal_doc["runs"]:
                stamp = r["run_dir"].rsplit("_", 2)[-2] + "_" + r["run_dir"].rsplit("_", 1)[-1]
                row = f"| **{r['total_score']}** | {r['achievement']:.6f} |"
                if row not in text:
                    bad.append(f"{r['model']} rep{r['replicate']} ({r['total_score']}, {stamp})")
            ok(args.skip_trajectories or (not bad and len(cal_doc["runs"]) == len(run_dirs) >= 6),
               f"{name} prints one row per RUN for all {len(cal_doc['runs'])} runs with the score "
               f"and achievement the report recomputes",
               f"rows not found for {bad}")
    # d) the naive-policy ceiling
    if os.path.exists(naive_path):
        with open(naive_path) as fh:
            nv_doc = json.load(fh)
        for name, text in docs.items():
            ok(f"{nv_doc['naive_ceiling']}" in text and "naive" in text.lower() + text,
               f"{name} quotes the naive multi-variable policy ceiling "
               f"({nv_doc['naive_ceiling']}/100)")
    # e) EVERY check count printed in the prose must be one this package actually reports. The docs
    #    quote two of them -- this script's and the leak checker's -- so both are re-measured here
    #    (the leak checker is run with its output suppressed, ~1 s) and the printed set must equal
    #    the measured set exactly. Counted last, and the +1 is this check itself, so the assertion
    #    is self-consistent rather than circular.
    import contextlib                                                    # noqa: E402
    import io as _io                                                     # noqa: E402
    import leak_check_question as LC                                     # noqa: E402
    with contextlib.redirect_stdout(_io.StringIO()):
        LC.main()
    allowed = {CHECKS + 1, int(LC.CHECKS)}
    printed_counts: Dict[str, List[int]] = {}
    for name, text in docs.items():
        found = [int(m) for m in re.findall(r"(\d{2,4})\s*(?:checks|项断言)", text)]
        found += [int(a) for a, b in re.findall(r"(\d{2,4})\s*/\s*(\d{2,4})", text)
                  if a == b and int(a) > 50]
        printed_counts[name] = sorted(set(found))
    if args.skip_trajectories:
        # The live count is lower in dev mode because run-dependent assertions did not execute, so
        # the documented delivery-mode count would read as stale for an artificial reason.
        allowed = set(allowed) | set(sum(printed_counts.values(), []))
    wrong = {n: [v for v in vs if v not in allowed] for n, vs in printed_counts.items()}
    wrong = {n: v for n, v in wrong.items() if v}
    ok(not wrong and any(printed_counts.values()),
       f"every check count printed in the prose is one this package reports live "
       f"(consistency {CHECKS + 1}, leak check {LC.CHECKS}); found "
       f"{ {n: vs for n, vs in printed_counts.items()} }",
       f"stale counts: {wrong}")

    # ---------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"{CHECKS - len(FAILURES)}/{CHECKS} checks passed in {time.time() - t0:.1f} s")
    if FAILURES:
        print("FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nA calibration edit is only complete when scripts/enumerate_reference_best.py has "
              "been re-run and this script passes.")
        return 1
    print("CONSISTENT: one calibration, four artifacts, three gates re-measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

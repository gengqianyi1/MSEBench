"""What score can be reached WITHOUT solving the problem? Measured with the shipped ruler.

The question text says out loud which side of its window each CQA currently sits on. That statement
is unavoidable -- without it the task is unanswerable -- so the honest thing to do is measure what it
buys. Every probe below is a MECHANICAL enumeration of a family of answers, never a hand-written list
of moves: a hand-written list contains only the moves the designer thought of, which confirms a
design instead of testing it.

Probes:
  (a) every 1-slot change
  (b) every <= 2-slot change
  (c) the microenvironment-pH subspace   {T3_acid_id, T4_acid_pct, T5_meg_pct, T8_mgst_pct, T12_man_pct}
  (d) the hygroscopicity subspace        {T1_mcc_grade, T2_mcc_pct, T6_dis_id, T7_dis_pct, T10_sil_pct}
  (e) the mechanics / flow subspace      {T1_mcc_grade, T8_mgst_pct, T9_blend_min, T10_sil_pct, T11_press_MPa}
  (f) the dissolution subspace           {T7_dis_pct, T8_mgst_pct, T9_blend_min, T11_press_MPa, T12_man_pct}
  (h) process parameters only            {T9_blend_min, T11_press_MPa}
  plus family ceilings by move magnitude: every design within k notches of the baseline set.

  (g) composition-only (all ten formulation slots, no process change) is enumerated for information
      and EXCLUDED from the naive maximum. It is an exhaustive sweep of 10 of the 12 slots -- roughly
      a quarter of the whole feasible set -- so it contains the global optimum and returns 100 points
      on ANY instance, no matter how hard. It measures brute force, not naivety. Reporting it inside
      the naive maximum printed "100" directly above a "threshold < 50" line and made a probe that had
      passed look like a failure.

Run:
    python scripts/measure_naive_answers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_space import (VectorOracle, Desirability, achievement, calibrate_windows,  # noqa: E402
                          describe_row, enumerate_rows, filler_pct_vec, n_changed_vec,
                          n_identity_vec, overlap_vec, points_for, START_IDX, to_design)
from evaluator import scorer_from_config  # noqa: E402
from formulation import CQA_KEYS, LEVELS, MIN_FILLER_PCT, SLOTS, START  # noqa: E402
from utils import load_yaml  # noqa: E402

TASK_CFG = ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml"
NAIVE_THRESHOLD = 50

# Probes are named by real slot name so a reviewer can check the membership without decoding a legend.
SUBSPACES = {
    "(c) microenvironment-pH subspace": ("T3_acid_id", "T4_acid_pct", "T5_meg_pct",
                                         "T8_mgst_pct", "T12_man_pct"),
    "(d) hygroscopicity subspace": ("T1_mcc_grade", "T2_mcc_pct", "T6_dis_id",
                                    "T7_dis_pct", "T10_sil_pct"),
    "(e) mechanics/flow subspace": ("T1_mcc_grade", "T8_mgst_pct", "T9_blend_min",
                                    "T10_sil_pct", "T11_press_MPa"),
    "(f) dissolution subspace": ("T7_dis_pct", "T8_mgst_pct", "T9_blend_min",
                                 "T11_press_MPa", "T12_man_pct"),
    "(h) process parameters only": ("T9_blend_min", "T11_press_MPa"),
}
NON_NAIVE = {
    "(g) composition-only, no process change": tuple(s for s in SLOTS
                                                    if s not in ("T9_blend_min", "T11_press_MPa")),
}


NLEV = [len(LEVELS[s]) for s in SLOTS]
SI = {s: i for i, s in enumerate(SLOTS)}
PROC = ("T9_blend_min", "T11_press_MPa")
IDENT = ("T3_acid_id", "T6_dis_id", "T12_man_pct")


def baseline_mass(slot: str) -> float:
    """Mass % the baseline formulation puts behind a condition. Printed in question.md."""
    if slot in PROC:
        return 0.0
    if slot == "T1_mcc_grade":
        return float(START["T2_mcc_pct"])
    if slot == "T3_acid_id":
        return float(START["T4_acid_pct"])
    if slot == "T6_dis_id":
        return float(START["T7_dis_pct"])
    return float(START[slot])


def slot_orders() -> dict:
    """Thirteen priority rules over the twelve conditions, from printed data only.

    `most_headroom` was added after an independent audit: without it, and without the budget loop in
    build_policies, the policy family topped out at 9/100 and the reported ceiling understated what a
    reviewer writing their own generator would find (20/100). A ceiling is only worth quoting if
    widening the family in the obvious directions does not move it, so both obvious directions are in
    the shipped generator now.
    """
    mid = {s: (len(LEVELS[s]) - 1) / 2 for s in SLOTS}
    pos = {s: LEVELS[s].index(START[s]) for s in SLOTS}
    span = {s: (0.0 if isinstance(LEVELS[s][0], str)
                else (max(LEVELS[s]) - min(LEVELS[s])) / (abs(float(START[s])) + 1e-9))
            for s in SLOTS}
    # How many grid notches the baseline set point has on its wider side: "the condition with the
    # most room left to move" is an obvious priority rule and uses nothing but the printed grids.
    headroom = {s: max(pos[s], len(LEVELS[s]) - 1 - pos[s]) for s in SLOTS}
    return {
        "most_headroom": sorted(SLOTS, key=lambda s: (-headroom[s], SLOTS.index(s))),
        "most_levels": sorted(SLOTS, key=lambda s: -len(LEVELS[s])),
        "fewest_levels": sorted(SLOTS, key=lambda s: len(LEVELS[s])),
        "far_from_grid_mid": sorted(SLOTS, key=lambda s: -abs(pos[s] - mid[s])),
        "near_grid_mid": sorted(SLOTS, key=lambda s: abs(pos[s] - mid[s])),
        "largest_mass": sorted(SLOTS, key=lambda s: -baseline_mass(s)),
        "smallest_mass": sorted(SLOTS, key=lambda s: baseline_mass(s)),
        "printed_order": list(SLOTS),
        "reverse_printed": list(SLOTS)[::-1],
        "process_first": list(PROC) + [s for s in SLOTS if s not in PROC],
        "process_last": [s for s in SLOTS if s not in PROC] + list(PROC),
        "identity_first": list(IDENT) + [s for s in SLOTS if s not in IDENT],
        "widest_relative_span": sorted(SLOTS, key=lambda s: -span[s]),
    }


def level_rules() -> dict:
    """Twelve level-choice rules. Each takes the baseline index and the grid length."""
    def clamp(k, n):
        return int(min(max(k, 0), n - 1))
    r = {}
    for st in (1, 2, 3):
        r[f"up{st}"] = lambda k, n, st=st: clamp(k + st, n)
        r[f"down{st}"] = lambda k, n, st=st: clamp(k - st, n)
    r["to_mid"] = lambda k, n: clamp(int(round((n - 1) / 2)), n)
    r["to_max"] = lambda k, n: n - 1
    r["to_min"] = lambda k, n: 0
    r["to_far_end"] = lambda k, n: (n - 1) if k < (n - 1) / 2 else 0
    r["one_toward_mid"] = lambda k, n: clamp(k + (1 if k < (n - 1) / 2 else -1), n)
    r["two_toward_mid"] = lambda k, n: clamp(k + (2 if k < (n - 1) / 2 else -2), n)
    return r


def policy_feasible(Rm: np.ndarray, cons: dict, floor: float) -> np.ndarray:
    from design_space import acid_ok_vec
    Rm = np.atleast_2d(Rm)
    return (acid_ok_vec(Rm) & (filler_pct_vec(Rm) >= MIN_FILLER_PCT)
            & (n_identity_vec(Rm) <= int(cons["max_identity_changes"]["max"]))
            & (overlap_vec(Rm) >= floor - 1e-12)
            & (n_changed_vec(Rm) <= int(cons["max_slot_changes"]["max"])))


def build_policies(budgets: tuple = (2, 3, 4)) -> tuple:
    """One design per (ranking rule, level rule, budget) triple, repaired to feasibility.

    The budget loop matters and is not decoration. Spending all four permitted changes is not always
    the best mechanical policy -- the fourth change costs composition overlap and can undo the first
    three -- so a generator that always spends four understates the family's ceiling. Measured: with
    budget 4 only the family tops out at 9/100; allowing 2 and 3 as well it reaches 20/100. The
    honest number to defend is the wider one.
    """
    cfg = load_yaml(TASK_CFG)
    cons, floor = cfg["constraints"], float(cfg["reference"]["calibration"]["overlap_floor"])
    rows, names = [], []
    for budget in budgets:
        for oname, order in slot_orders().items():
            for lname, rule in level_rules().items():
                row = START_IDX.copy()
                for s in order[:budget]:
                    i = SI[s]
                    row[i] = rule(int(START_IDX[i]), NLEV[i])
                    # keep the acid identity/amount pair coherent, or the design is not evaluable
                    if s == "T3_acid_id" and LEVELS[s][row[i]] == "none":
                        row[SI["T4_acid_pct"]] = LEVELS["T4_acid_pct"].index(0.0)
                    elif s == "T4_acid_pct" and float(LEVELS[s][row[i]]) == 0.0:
                        row[SI["T3_acid_id"]] = LEVELS["T3_acid_id"].index("none")
                while not bool(policy_feasible(row[None, :], cons, floor)[0]):
                    changed = [s for s in order if row[SI[s]] != START_IDX[SI[s]]]
                    if not changed:
                        break
                    row[SI[changed[-1]]] = START_IDX[SI[changed[-1]]]
                rows.append(row)
                names.append(f"{oname}|{lname}|budget{budget}")
    return rows, names


def main() -> None:
    cfg = load_yaml(TASK_CFG)
    cal = cfg["reference"]["calibration"]
    cons = cfg["constraints"]
    floor = float(cal["overlap_floor"])

    O = VectorOracle()
    R = enumerate_rows(int(cons["max_slot_changes"]["max"]))
    Y = O.predict_rows(R)
    y0 = O.predict_rows(START_IDX[None, :])[0]
    win = calibrate_windows(y0, Y, float(cal["pull"]), float(cal["marg"]))
    # THE shipped ruler: built by the same function src/main.py grades with, then asserted equal to a
    # fresh recomputation of the windows. A re-derived look-alike is exactly the failure this whole
    # script exists to avoid.
    sc = scorer_from_config(cfg["objective"])
    for i, key in enumerate(CQA_KEYS):
        w = cfg["objective"]["windows"][key]
        assert (w["lower"], w["target"], w["upper"]) == (win["L"][i], win["T"][i], win["U"][i]), \
            "recomputed windows differ from task_config.yaml -- run scripts/check_consistency.py"
    assert np.abs(sc.L - win["L"]).max() == 0 and np.abs(sc.U - win["U"]).max() == 0, \
        "the config scorer is not on the recomputed windows -- run scripts/check_consistency.py"

    D = sc.D_of(Y)
    D0 = sc.D_one(y0)
    Dref = float(cfg["reference"]["best_objective_in_budget"])
    # Ruler 2.0.0: the scoring-eligible set is budget-feasible AND CQA-compliant AND improving.
    # Every family ceiling below is therefore measured over the set that can actually score
    # (audit finding #3) -- a shortcut that improves D by breaching a specification buys nothing.
    # TWO masks, and the difference matters (audit finding #3 changed the ruler under this script).
    #   BUDGET  designs a submission could legally BE -- this is what sizes a family, because an
    #           agent brute-forcing a family has to measure every budget-feasible member of it,
    #           compliant or not;
    #   FEAS    designs that can actually SCORE under ruler 2.0.0: budget-feasible AND all five CQAs
    #           inside their windows AND better than the start. This is what sets a ceiling.
    # Using FEAS for both would silently recalibrate the brute-force cutoff, because the compliant
    # subset is ~13x smaller than the budget-feasible one.
    BUDGET = ((overlap_vec(R) >= floor - 1e-12)
              & (n_changed_vec(R) <= int(cons["max_slot_changes"]["max"]))
              & (n_identity_vec(R) <= int(cons["max_identity_changes"]["max"]))
              & (filler_pct_vec(R) >= MIN_FILLER_PCT))
    FEAS = BUDGET & sc.in_spec(Y) & (sc.D_of(Y) > sc.D_one(y0))
    NC = n_changed_vec(R)
    changed = [tuple(SLOTS[i] for i in range(len(SLOTS)) if R[j][i] != START_IDX[i])
               for j in range(len(R))]

    print("baseline D = %.6f   reference D = %.6f   feasible designs = %d"
          % (D0, Dref, int(FEAS.sum())))
    print("\nnaive answers to 'the background states which side of its window each CQA sits on':")

    out = {"naive_probes": {}, "non_naive_probes": {}, "notch_families": {}}
    naive_max = 0

    # ONE exclusion rule, applied to every family by the same arithmetic. An independent audit found
    # the previous version folded the 8,643-design (f) subspace into the naive maximum but silently
    # left the 35,397-design "<= 2 notch" family out, so the headline was not comparable across
    # probes. The line is now a printed number: a family whose exhaustive size exceeds
    # BRUTE_FORCE_SIZE is a search over a sizeable fraction of the whole feasible set, so its ceiling
    # measures how big the sweep was and not what the naive reading of the Background buys. Reaching
    # it requires thousands of measurements; the agent has five. Both the included maximum and every
    # excluded ceiling are printed and stored, so a reviewer can move the line and see what happens.
    BRUTE_FORCE_SIZE = 10000
    excluded_ceilings = {}

    def report(label, mask, naive=True, force_exclude=False):
        """`mask` selects the family out of the BUDGET-feasible set. Its size decides whether the
        family is brute force; its scoring-eligible subset decides its ceiling."""
        nonlocal naive_max
        n_budget = int((mask & BUDGET).sum())
        scoring = mask & FEAS
        n = int(scoring.sum())
        j = int(np.argmax(np.where(scoring, D, -1.0)))
        pts = points_for(D[j], D0, Dref) if n else 0
        excluded = force_exclude or n_budget > BRUTE_FORCE_SIZE
        tag = ""
        if naive and not excluded:
            naive_max = max(naive_max, pts)
        else:
            tag = ("   [EXCLUDED from the naive maximum: exhaustive %d-design sweep > %d, i.e. brute "
                   "force rather than a naive reading]" % (n_budget, BRUTE_FORCE_SIZE))
            excluded_ceilings[label] = {"n_budget_feasible": n_budget, "n_scoring": n,
                                        "points": pts}
        print("   %-40s n=%7d (scoring %6d)  best D=%.6f ach=%.4f -> %3d pts%s"
              % (label, n_budget, n, D[j], achievement(D[j], D0, Dref), pts, tag))
        print("        %s" % describe_row(R[j]))
        rec = {"n_budget_feasible": n_budget, "n_scoring": n, "best_value": float(D[j]),
               "points": pts,
               "best_changes": describe_row(R[j]), "counted_in_naive_maximum": naive and not excluded}
        (out["naive_probes"] if naive else out["non_naive_probes"])[label] = rec

    report("(a) every 1-slot change", BUDGET & (NC == 1))
    report("(b) every <= 2-slot change", BUDGET & (NC <= 2))
    for label, slots in SUBSPACES.items():
        mask = BUDGET & np.array([set(c) <= set(slots) for c in changed])
        report(label, mask)
    for label, slots in NON_NAIVE.items():
        mask = BUDGET & np.array([set(c) <= set(slots) for c in changed])
        report(label, mask, naive=False, force_exclude=True)

    print("\nfamily ceilings by move magnitude (every design within k notches on every slot):")
    notch = np.zeros(len(R), dtype=int)
    for i, s in enumerate(SLOTS):
        notch = np.maximum(notch, np.abs(R[:, i].astype(int) - int(START_IDX[i])))
    for k in (1, 2, 3, 4):
        n_budget = int((BUDGET & (notch <= k)).sum())
        scoring = FEAS & (notch <= k)
        n = int(scoring.sum())
        j = int(np.argmax(np.where(scoring, D, -1.0)))
        pts = points_for(D[j], D0, Dref) if n else 0
        # Same rule, same number, applied to the BUDGET-feasible size of the family.
        excluded = n_budget > BRUTE_FORCE_SIZE
        if not excluded:
            naive_max = max(naive_max, pts)
        else:
            excluded_ceilings[f"<= {k} notch"] = {"n_budget_feasible": n_budget, "n_scoring": n,
                                                  "points": pts}
        print("   <= %d notch on every slot                 n=%7d (scoring %6d)  best D=%.6f"
              " -> %3d pts%s"
              % (k, n_budget, n, D[j], pts,
                 "   [EXCLUDED: > %d designs]" % BRUTE_FORCE_SIZE if excluded else ""))
        out["notch_families"][f"<= {k} notch"] = {
            "n_budget_feasible": n_budget, "n_scoring": n, "points": pts,
            "counted_in_naive_maximum": not excluded}

    # ---------------------------------------------------------------------------------------------
    # Hand-computable policies. The families above are exhaustive subspace searches, which a model
    # with five measurements cannot run. This probe is the opposite extreme and the more realistic
    # threat: every ranking rule and every level rule below is computable from what question.md
    # PRINTS -- the level grids, the baseline set point of each condition, the mass each condition
    # carries, the constraint numbers -- and nothing else. No CQA value is consulted. Each of the
    # 12 x 12 combinations spends the whole 4-condition budget in one shot, is repaired to
    # feasibility by dropping its lowest-priority change until it fits the budget (so a policy is
    # never scored 0 merely for over-spending), and is then scored with the shipped ruler.
    #
    # This is the ONE family in this file that a five-round agent could actually execute, so it is the
    # number to defend. It is therefore generated deliberately wide: 13 ranking rules x 12 level rules
    # x 3 budgets = 468 policies. See build_policies for why the budget loop is not optional.
    # ---------------------------------------------------------------------------------------------
    print("\nhand-computable policies (ranking rule x level rule x budget, printed data only):")
    pol_rows, pol_names = build_policies()
    Rp = np.array(pol_rows, dtype=np.int16)
    okp = policy_feasible(Rp, cons, floor)
    Dp = np.where(okp, sc.D_of(O.predict_rows(Rp)), -1.0)
    Pp = np.array([points_for(v, D0, Dref) if v >= 0 else 0 for v in Dp])
    order = np.argsort(-(Pp + Dp * 1e-9))
    for q in order[:6]:
        print("   %-34s %3d pts  D=%.6f  %s"
              % (pol_names[q], Pp[q], Dp[q], describe_row(Rp[q])))
    print("   %d policies, best = %d pts (achievement %.4f)"
          % (len(Rp), int(Pp.max()), achievement(float(Dp[order[0]]), D0, Dref)))
    out["hand_policies"] = {
        "n_policies": int(len(Rp)), "best_points": int(Pp.max()),
        "best_achievement": achievement(float(Dp[order[0]]), D0, Dref),
        "best_policy": pol_names[int(order[0])],
        "best_design": describe_row(Rp[int(order[0])]),
        "top": [{"policy": pol_names[int(q)], "points": int(Pp[q]), "value": float(Dp[q]),
                 "changes": describe_row(Rp[int(q)])} for q in order[:6]]}
    naive_max = max(naive_max, int(Pp.max()))

    out["naive_answer_points"] = naive_max
    out["threshold"] = NAIVE_THRESHOLD
    out["brute_force_size_cutoff"] = BRUTE_FORCE_SIZE
    out["excluded_ceilings"] = excluded_ceilings
    out["verdict"] = "PASS" if naive_max < NAIVE_THRESHOLD else "FAIL"
    print("\nnaive_answer_points = %d   [threshold < %d]   -> %s"
          % (naive_max, NAIVE_THRESHOLD, out["verdict"]))
    print("exclusion rule: any family larger than %d designs is brute force, not naivety. Excluded, "
          "with their ceilings, so the line can be moved and re-read:" % BRUTE_FORCE_SIZE)
    for label in sorted(excluded_ceilings):
        rec = excluded_ceilings[label]
        print("   %-42s n=%7d (scoring %6d) -> %3d pts"
              % (label, rec["n_budget_feasible"], rec["n_scoring"], rec["points"]))

    path = ROOT / "answers" / "naive_answer_probes.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()

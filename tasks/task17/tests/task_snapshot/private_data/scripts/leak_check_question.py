#!/usr/bin/env python
"""Audit tasks/q1_agglomeration/question.md: complete, leak-free, and neutral about all 12 variables.

Three things are checked, and they are three different failure modes.

A  COMPLETE -- the question text must state everything the agent is graded on, re-derived from
   src/agglomeration.py (the one ruler) rather than eyeballed: every slot, every level, every change
   cost, every window bound and weight, the desirability shape, the change budget and its overlap
   floor, all five hard process limits with the constants they need (boiling points, the
   just-suspended coefficient and exponent, the solvent formula, the addition and drying durations),
   the round count, the achievement formula and the score bands. A question that omits one of these
   grades the model on information it was never given.

B  LEAK-FREE -- the question text must not carry any designer-side quantity from
   answers/reference_enumeration.json: the reference objective value in any rounding, the reference
   batch's four responses or per-response desirabilities, the enumeration's own change annotations, or
   the path of the designer-only file. It must also not disclose the reference DESIGN implicitly: no
   single line may name three or more of the reference's target levels, which is what a "worked
   example" or an over-helpful hint would look like. Publicly stated structure (the 9 / 20 / 33
   ceilings, the neighbourhood sizes, the four measured responses of the registered process) is NOT a
   leak -- it is the fairness disclosure, and it is asserted to be present in A.

C  NEUTRAL -- the twelve variables must be presented even-handedly. The optimum touches four of them
   and leaves eight alone, several of which are exactly the ones a textbook reading of the background
   would reach for first. That asymmetry must not be visible in the prose: every variable gets a
   mechanism line and a level row, no variable is described with valence or directive language
   ("avoid", "recommend", "optimal", "useless", ...), and the mechanism lines of the four variables the
   reference moves must not be systematically longer or shorter than those of the eight it does not.

Exit code 0 and the line 'LEAK CHECK CLEAN' only if every check passes.

    python scripts/leak_check_question.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import yaml                                                              # noqa: E402

import agglomeration as A                                                # noqa: E402
from design_space import POINT_BANDS, g2_cost_criterion, overlap_floor    # noqa: E402

QUESTION = os.path.join(ROOT, "tasks", "q1_agglomeration", "question.md")
TASK_CFG = os.path.join(ROOT, "tasks", "q1_agglomeration", "task_config.yaml")
ENUM_JSON = os.path.join(ROOT, "answers", "reference_enumeration.json")

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


def norm(text: str) -> str:
    """Normalise for substring search: collapse whitespace, unify dashes and quotes."""
    t = text.replace("–", "-").replace("—", "-").replace("−", "-")
    t = t.replace("‘", "'").replace("’", "'").replace("µ", "u")
    return re.sub(r"[ \t]+", " ", t)


def num_forms(v: float) -> List[str]:
    """Every plausible way a level or a limit could legitimately be written."""
    out = {f"{v:g}", str(v)}
    if float(v) == int(float(v)):
        out.add(str(int(float(v))))
        out.add(f"{int(float(v))}.0")
    for nd in (1, 2, 3):
        out.add(f"{float(v):.{nd}f}")
    return sorted(out)


def present(text: str, v: Any) -> bool:
    if isinstance(v, str):
        return v in text
    return any(f in text for f in num_forms(float(v)))


def main() -> int:
    with open(QUESTION) as fh:
        raw = fh.read()
    q = norm(raw)
    lines = [norm(l) for l in raw.splitlines()]
    with open(TASK_CFG) as fh:
        cfg = yaml.safe_load(fh)
    with open(ENUM_JSON) as fh:
        enum = json.load(fh)

    print("=" * 96)
    print("A  the question states everything the agent is graded on (re-derived from src/)")
    print("=" * 96)

    missing = [s for s in A.SLOTS if s not in q]
    ok(not missing, "all 12 variable names appear", f"missing {missing}")

    # Each variable's levels and change cost must appear on the SAME table row as the variable, so the
    # agent can read its grid and price a change without guessing -- and so that a level deleted from
    # one row is not masked by the same number appearing on another variable's row.
    cost_rows: Dict[str, str] = {}
    for s in A.SLOTS:
        cand = [l for l in lines if l.lstrip().startswith(f"| `{s}`")]
        cost_rows[s] = cand[-1] if cand else ""
    ok(all(cost_rows.values()), "every variable has a level row",
       f"missing {[s for s in A.SLOTS if not cost_rows[s]]}")
    bad_cost = [s for s in A.SLOTS
                if not re.search(rf"\|\s*{A.EDIT_COST[s]}\s*\|?\s*$", cost_rows[s] or "")]
    ok(not bad_cost, "every variable's change cost is stated on its own level row",
       f"cost not found for {bad_cost}")

    missing_levels: List[str] = []
    for s in A.SLOTS:
        row = cost_rows[s]
        for lv in A.LEVELS[s]:
            if not present(row, lv):
                missing_levels.append(f"{s}={lv!r}")
    ok(not missing_levels,
       f"all {sum(len(v) for v in A.LEVELS.values())} declared levels appear on their own row",
       f"missing {missing_levels}")

    ok(f"{A.BUDGET} of the {A.TOTAL_EDIT_COST}" in q,
       f"the change budget is stated as {A.BUDGET} of the {A.TOTAL_EDIT_COST}")
    ok(f"{overlap_floor(A.BUDGET) * 100:.1f}" in q,
       f"the equivalent overlap floor {overlap_floor(A.BUDGET) * 100:.1f} % is stated")
    ok(str(A.TOTAL_EDIT_COST) in q, "the all-twelve change cost is stated")

    win_missing: List[str] = []
    for key, w in A.WINDOWS.items():
        for field in ("pl", "pu", "al", "au"):
            if not present(q, w[field]):
                win_missing.append(f"{key}.{field}={w[field]:g}")
    ok(not win_missing, "all 16 window bounds appear", f"missing {win_missing}")
    ok(all(k in q for k in A.RESPONSE_KEYS), "all four response names appear")
    ok(f"({int(A.WINDOWS['d50']['w'])}, {int(A.WINDOWS['strength']['w'])}, "
       f"{int(A.WINDOWS['porosity']['w'])}, {int(A.WINDOWS['yield_pct']['w'])})" in q,
       "the desirability weight vector is stated")
    ok(f"m = {A.DESIRABILITY_SHAPE_M:g}" in q, "the desirability exponent m is stated")
    ok("3^(1/m)" in q, "the desirability constant C = 3^(1/m) is stated")
    ok("0.25" in q, "the desirability value at the acceptable limit is stated")

    bp_missing = [k for k, v in A.BL.items() if not present(q, v["bp"])]
    ok(not bp_missing, "C3: every bridging-liquid boiling point is listed", f"{bp_missing}")
    ok("10" in q and "boiling point" in q, "C3: the 10 degC margin is stated")
    ok("0.42" in q and "0.30" in q and "40" in q,
       "C4: the just-suspended coefficient, exponent and reference loading are stated")
    ok(present(q, cfg["constraints"]["total_solvent"]["max_L_per_kg"]) and "1000/" in q,
       "C5: the solvent formula and its limit are stated")
    ok(present(q, cfg["constraints"]["total_process_time"]["max_hours"]),
       "C6: the processing-time limit is stated")
    add_missing = [k for k, v in A.ADD.items() if not present(q, v["t"])]
    dry_missing = [k for k, v in A.DRY.items() if not present(q, v["h"])]
    ok(not add_missing and not dry_missing,
       "C6: every addition and drying duration is stated so the limit can be checked by hand",
       f"addition {add_missing} drying {dry_missing}")
    ok("recycle" in q and "fines" in q, "C7: the recycle self-consistency rule is stated")

    start_missing = [f"{k}={v!r}" for k, v in A.START.items() if not present(q, v)]
    ok(not start_missing, "the registered process is given in full", f"missing {start_missing}")
    ok(all(present(q, round(float(v), 4)) or present(q, round(float(v), 2))
           for v in enum["start_responses"].values()),
       "the four measured responses of the registered process are given")

    rounds = int(cfg["iteration"]["max_rounds"])
    ok(f"**{rounds} rounds**" in q or f"{rounds} rounds" in q, f"the {rounds}-round budget is stated")
    ok("best usable process anywhere in the trajectory" in q,
       "the best-in-trajectory rule is stated")
    ok("achievement" in q and "clipped at 1.0" in q, "the achievement definition is stated")

    # ---------------------------------------------------------------------------------------------
    # THE SCORE BANDS ARE NO LONGER PUBLISHED, AND THIS CHECK ENFORCES THAT (second round, 17-P1 #3).
    #
    # This assertion used to run the other way: every band had to appear, as a fairness disclosure.
    # Two things were wrong with that. It contradicted README.md and the public boundary stub, which
    # both said the score anchors were withheld -- so whichever half a reviewer read first was wrong.
    # And a published ladder tells the model where the effort stops paying: seeing "0.30 -> 33,
    # 0.45 -> 47" invites stopping at a threshold instead of optimising, which is the opposite of
    # what five rounds of coordinated redesign are meant to measure.
    #
    # The question now states only that the score is a monotone graded function of achievement. A
    # band is a (threshold, points) pair, so leaking either half of a pair on one line is enough to
    # give the ladder away; that pairing is what is denied here. Bare point values are not denied on
    # their own -- small integers like 9, 12 and 20 occur innocently -- but a line carrying both the
    # threshold and its points is a table row by any other name.
    # ---------------------------------------------------------------------------------------------
    leaked_bands = []
    for thr, pts in POINT_BANDS:
        t, p = f"{thr:g}", str(pts)
        for line in q.splitlines():
            if t in line and re.search(rf"(?<![\d.]){p}(?![\d.%])", line):
                leaked_bands.append(f"{t}->{p}: {line.strip()[:70]}")
                break
    ok(not leaked_bands, "no score band (threshold paired with its points) appears in the question",
       f"leaked {leaked_bands}")
    ok("| achievement | Points |" not in q, "the band table itself has been removed")
    ok("monotone graded function" in q,
       "the question says the score is a monotone graded function of achievement")
    for phrase in ("band thresholds", "not given to you"):
        ok(phrase in q, f"the question states that the ladder is withheld ('{phrase}')")
    ok("no points for satisfying the constraints" in q,
       "the pure-optimality rubric (feasibility earns nothing) is stated")
    for phrase in ("simulated attributes", "change-budget bookkeeping",
                   "names of any hard constraints you violated",
                   "per-attribute window report", "history table"):
        ok(phrase in q, f"the feedback contract states what IS returned: '{phrase}'")
    for phrase in ("does **not** return", "the reference optimum", "your achievement fraction",
                   "nothing about the simulator's mechanism"):
        ok(phrase in q, f"the feedback contract states what is NOT returned: '{phrase}'")
    # The optional tool layer is part of the interaction contract, so the question must declare it AND
    # declare its hard limit -- a model must know that no tool will value a design for it.
    for phrase in ("Optional diagnostic tools", "No tool returns any quality"):
        ok(phrase in q, f"the question declares the tool contract: '{phrase}'")
    # Honest labelling (external review item 1): the question must say what the data is.
    for phrase in ("synthetic simulator", "None of it is experimental data",
                   "no registered product", "withheld from"):
        ok(phrase in q, f"the question states the data nature: '{phrase}'")
    ok("registered process" not in q.lower(),
       "the question never calls the baseline a 'registered process'")
    ok("measured attributes" not in q and "its measured values" not in q,
       "the question never calls a simulator output a 'measured' attribute")

    # The two JSON blocks in the question are executable claims: the registered-process block must BE
    # the registered process, and the output-format block must survive the shipped parser and the
    # shipped design-space check. A question whose own example is off-grid teaches the wrong format.
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    ok(len(blocks) == 2, f"the question carries exactly two JSON blocks (found {len(blocks)})")
    parsed_start = None
    parsed_example = None
    for b in blocks:
        try:
            obj = json.loads(b)
        except json.JSONDecodeError as exc:
            ok(False, "every JSON block in the question is valid JSON", str(exc))
            continue
        if "candidate" in obj:
            parsed_example = obj
        else:
            parsed_start = obj
    ok(parsed_start == A.START, "the registered-process block IS src/agglomeration.START",
       f"differs: {parsed_start}")
    if parsed_example is not None:
        from parser import parse_candidate                              # noqa: E402
        try:
            cand = parse_candidate(json.dumps(parsed_example))
            same_as_start = A.normalise_design(cand["process"]) == A.START
            detail = ""
        except Exception as exc:                                        # ParseError / DesignError
            same_as_start = False
            detail = f"{type(exc).__name__}: {exc}"
        ok(same_as_start,
           "the output-format example parses with src/parser and is an on-grid design", detail)
    else:
        ok(False, "the output-format example block was not found")

    # THE GATE MEASUREMENTS ARE NO LONGER PUBLISHED IN THE QUESTION, AND THIS CHECK ENFORCES THAT.
    #
    # The first version of this problem printed the measured ceilings of the obvious strategy families
    # (change one variable -> N points, leave budget unspent -> M points) plus the exact population of
    # every scoring band, as a fairness disclosure. External review found that this, together with the
    # response mechanism, put more in the model-visible material than completing the task requires: a
    # model told "one variable is worth 9, two are worth 20, holding budget back caps you at 33" has
    # been handed the shape of the answer -- spend everything, move about four variables -- without
    # doing any process reasoning.
    #
    # Those numbers now live on the grading side only (answers/reference_enumeration.json,
    # task_config.yaml's reference: block, 审核说明.md), and the assertion is inverted: the question
    # text must NOT contain them. Only counts with >= 4 digits are tested, because small integers such
    # as the point values themselves legitimately appear in the band table.
    g = enum["gates"]
    designer_counts = {
        "1-slot family size": g["g1_one_slot"]["n_feasible_and_in_budget"],
        "1+2-slot family size": g["g1_combined"]["n_feasible_and_in_budget"],
        "1+2-slot family gross": g["g1_combined"]["n_designs"],
        "G2 family size": g["g2"]["n_feasible_and_in_budget"],
        "G2 candidate count": g["g2"]["n_designs"],
        "feasible in-budget count": g["g3"]["n_feasible_in_budget"],
        "improving-design count": g["g3"]["n_strictly_improving"],
        "in-budget count before the constraints":
            enum["design_space"]["size_in_change_budget"],
    }
    for k, v in g["g3"]["points_ladder"].items():
        designer_counts[f"G3 ladder {k}"] = v
    for k, v in (g["g3"]["achievement_distribution_over_improving_designs"]["count_in_band"]
                 .items()):
        designer_counts[f"achievement band {k}"] = v
    thin = " ".join(q.split())
    present_counts = [what for what, v in designer_counts.items()
                      if int(v) >= 1000
                      and (f"{int(v):,}".replace(",", " ") in thin or str(int(v)) in thin)]
    ok(not present_counts,
       f"none of the {sum(1 for v in designer_counts.values() if int(v) >= 1000)} designer-side "
       f"enumeration counts appears in the question text",
       f"leaked: {present_counts}")
    for phrase in ("best score reachable", "feasible designs graded", "ceilings measured",
                   "exhaustively enumerated", "strategy family"):
        ok(phrase not in q,
           f"the question does not publish the offline gate table ('{phrase}' absent)")
    # The G2 cost criterion is a designer-side threshold too: it tells the model exactly how much of
    # the budget it must spend to escape the safe region.
    ok(f"cost <= {g2_cost_criterion(A.BUDGET)}" not in q
       and f"cost ≤ {g2_cost_criterion(A.BUDGET)}" not in q,
       "the question does not publish the G2 cost criterion")

    print("\n" + "=" * 96)
    print("B  no designer-side quantity from answers/reference_enumeration.json is disclosed")
    print("=" * 96)

    secrets: Dict[str, float] = {"reference_best": float(enum["reference_best"])}
    for k, v in enum["reference_responses"].items():
        secrets[f"reference_response.{k}"] = float(v)
    for k, v in enum["reference_desirability_per_response"].items():
        secrets[f"reference_desirability.{k}"] = float(v)

    # Rounded to >= 3 decimals: fewer decimals collide with legitimate public constants (the C4
    # coefficient 0.42 contains "0.4", which is also porosity rounded to 1 dp).
    leaked: List[str] = []
    for name, v in secrets.items():
        for nd in range(3, 10):
            s = f"{v:.{nd}f}"
            if s.rstrip("0").rstrip(".") and s in q:
                leaked.append(f"{name} as {s}")
    for pat in (r"(?<![\d.])0\.9[23](?![\d])", r"(?<![\d.])0\.93\d*(?![\d])"):
        if re.search(pat, q):
            leaked.append(f"reference objective matched by {pat}")
    ok(not leaked, "no reference objective, response or per-response desirability value appears",
       f"leaked: {leaked}")

    forbidden_text = ["reference_enumeration", "answers/", "best_objective_in_budget",
                      enum["reference_changes"], "designer_only", "top_25_in_budget"]
    hits = [t for t in forbidden_text if t and t in q]
    ok(not hits, "no designer-side file, field name or enumeration annotation appears", f"{hits}")

    ok("api_key" not in q.lower() and "sk-" not in q and "PUT_YOUR" not in q,
       "no credential-like text")

    # Implicit disclosure of the reference DESIGN: no single line may name 3+ of its target levels.
    ref = enum["reference_design"]
    tokens: List[str] = []
    for s in A.SLOTS:
        if ref[s] != A.START[s]:
            tokens.append(str(ref[s]) if not isinstance(ref[s], str) else ref[s])
    dense = []
    for i, l in enumerate(lines, 1):
        n = sum(1 for t in tokens if t in l)
        if n >= 3:
            dense.append(f"line {i} carries {n} of {len(tokens)} reference levels: {l.strip()[:80]}")
    ok(not dense, f"no line names 3+ of the {len(tokens)} levels the reference changes to",
       "; ".join(dense))

    print("\n" + "=" * 96)
    print("C  all twelve variables are presented neutrally")
    print("=" * 96)

    mech: Dict[str, str] = {}
    for s in A.SLOTS:
        rows = [l for l in lines if l.lstrip().startswith(f"| `{s}`")]
        # the mechanism table row is the one WITHOUT a trailing change-cost cell
        body = [r for r in rows if not re.search(rf"\|\s*{A.EDIT_COST[s]}\s*\|?\s*$", r)]
        mech[s] = (body[0] if body else "")
    ok(all(mech.values()), "every variable has a mechanism row",
       f"missing {[s for s in A.SLOTS if not mech[s]]}")
    short = [s for s in A.SLOTS if len(mech[s]) < 60]
    ok(not short, "no variable is dismissed with a stub description", f"too short: {short}")

    valence = ["avoid", "recommend", "prefer", "useless", "ineffective", "pointless", "trap",
               "optimal", "no benefit", "do not use", "best choice", "textbook answer",
               "obvious move", "red herring", "decoy", "distractor", "irrelevant"]
    bad_valence = [(s, w) for s in A.SLOTS for w in valence if w in mech[s].lower()]
    ok(not bad_valence, "no mechanism row uses valence or directive language", f"{bad_valence}")
    bad_valence_rows = [(s, w) for s in A.SLOTS for w in valence
                        if w in (cost_rows[s] or "").lower()]
    ok(not bad_valence_rows, "no level row uses valence or directive language",
       f"{bad_valence_rows}")

    moved = [s for s in A.SLOTS if ref[s] != A.START[s]]
    kept = [s for s in A.SLOTS if ref[s] == A.START[s]]
    lm = sum(len(mech[s]) for s in moved) / len(moved)
    lk = sum(len(mech[s]) for s in kept) / len(kept)
    ratio = lm / lk if lk else 0.0
    ok(0.62 <= ratio <= 1.60,
       f"mechanism rows are not weighted toward the variables the optimum moves "
       f"(mean {lm:.0f} chars for the {len(moved)} it moves vs {lk:.0f} for the {len(kept)} it does "
       f"not, ratio {ratio:.2f})",
       "the four variables the reference changes are described at a different length from the rest")

    print("\n" + "=" * 96)
    print(f"{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("LEAK CHECK CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

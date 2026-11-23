#!/usr/bin/env python
"""Derive tasks/q1_agglomeration/task_config.public.yaml from the full (grader) task config.

Generated rather than hand-maintained, because the two would drift and the drift would be invisible:
a public config that quietly kept a stale window is worse than no public config. This script strips
the designer-side material and asserts that nothing designer-side survives.

REMOVED for the public build:
  * `reference:`   the reference optimum, its design, its responses, its per-response desirabilities,
                   the change annotation, the three difficulty-gate measurements, the achievement
                   percentiles and the score-band populations -- i.e. every score anchor;
  * every prose field that quotes an offline ceiling (`task_summary` is rewritten, not trimmed);
  * the pointer to answers/ and to the surrogate file.

KEPT, because the agent is graded on them and must be given them:
  the twelve level grids and change costs, the baseline process, the objective and its four windows,
  the constraints, the iteration budget, the feedback switches, and the score bands.

Run:
    python scripts/make_public_task_config.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import yaml                                                              # noqa: E402

FULL = os.path.join(ROOT, "tasks", "q1_agglomeration", "task_config.yaml")
PUBLIC = os.path.join(ROOT, "tasks", "q1_agglomeration", "task_config.public.yaml")

DROP_TOP_LEVEL = ["reference"]

# Second round, 2026-08-09. The achievement -> points ladder is now grader-side for every problem in
# this suite. Two reasons, one of them raised by review (17-P1 #3) and one of them a consistency
# defect this package had on its own:
#
#   * Publishing the ladder tells the model how much headroom is worth chasing before the effort
#     stops paying. That suppresses precisely the search the five rounds measure -- a model that can
#     read "0.30 -> 33 points, 0.45 -> 47" can stop at a threshold instead of optimising.
#   * The package contradicted itself: question.md printed the whole table while README.md and the
#     public boundary stub both stated that the score anchors were withheld. Whichever half a
#     reviewer read first was wrong.
#
# The grader-side task_config.yaml keeps `scoring.optimality_bands`; the ladder still has to match
# design_space.POINT_BANDS, and check_consistency.py section 2 still asserts that. What changes is
# that the public half no longer carries it, and leak_check_question.py now denies it in the
# question text.
DROP_SCORING_KEYS = ["optimality_bands"]

PUBLIC_SUMMARY = """
  A needle-habit drug substance is spherically agglomerated (quasi-emulsion solvent-diffusion /
  bridging-liquid route) in a SYNTHETIC SIMULATOR to make it flowable and directly compressible. No
  experimental data is involved: the four responses are simulator outputs, produced by a mechanistic
  model plus a surrogate network fitted to that model with synthetic noise (seed 20260726), and the
  mechanism is held on the grading side. All four simulated attributes of the baseline process sit
  outside two-sided specification windows at once, each on a different side, so no attribute can be
  repaired in isolation. Under a controlled-change budget of 6 weighted change points out of 22
  (equivalently: at least 72.7 % of the baseline process must survive), the agent redesigns the twelve
  declared process variables over five rounds to maximise the weighted Derringer-Suich desirability D
  over the four windows. Every one of the twelve variables enters all four response models, so the
  objective cannot be evaluated by hand or solved as a system of equations: five rounds of simulated
  measurement, plus an optional diagnostic tool layer that reports legality, runnability and change
  cost but never a quality attribute, are the only information channels.
"""

DATA_NATURE = """
  SYNTHETIC. Every response in this benchmark is the output of a synthetic process simulator. There is
  no registered product, no filed process and no experimental measurement anywhere in it. The baseline
  process is a designed starting point, not a real process. Evaluation is deterministic: no noise is
  added at evaluation time. A score measures coordinated multi-variable decision-making under a change
  budget with narrow-band numerical feedback, and must NOT be read as evidence about real spherical
  agglomeration performance or as a property of any real material. The full model card -- mechanism,
  coefficients, the synthetic noise distribution, fit statistics, applicability domain and limitations
  -- is a grader-side document (data/MODEL_CARD.md), because a model that can read the response
  mechanism can optimise against the response surface instead of solving the process problem.
"""


def main() -> int:
    with open(FULL) as fh:
        cfg = yaml.safe_load(fh)

    for key in DROP_TOP_LEVEL:
        cfg.pop(key, None)

    dropped_scoring = {}
    for key in DROP_SCORING_KEYS:
        value = cfg.get("scoring", {}).pop(key, None)
        if value is not None:
            dropped_scoring[key] = value
    cfg.setdefault("scoring", {})["bands_withheld"] = (
        "The achievement -> points ladder is grader-side only. The score is a monotone graded "
        "function of achievement; the thresholds and the points per band are not disclosed to the "
        "model, in this config, in question.md or in any round's feedback.")

    cfg["task_summary"] = PUBLIC_SUMMARY.strip()
    cfg["data_nature"] = DATA_NATURE.strip()
    cfg["package_side"] = ("public -- model-visible. The reference optimum, the score anchors, the "
                           "difficulty-gate measurements and the response mechanism are NOT in this "
                           "file or in this package.")

    # Strip designer-side prose from the surviving blocks. These notes explain gate measurements and
    # point at answers/, neither of which belongs in a model-visible file.
    for path in (("design_space", "change_cost_rationale"), ("objective", "note"),
                 ("iteration", "note")):
        node = cfg
        for k in path[:-1]:
            node = node.get(k, {})
        if isinstance(node, dict) and path[-1] in node:
            node[path[-1]] = re.sub(r"\s+", " ", str(node[path[-1]])).strip()

    # The neighbourhood sizes are enumeration OUTPUT, not part of the contract: telling the model how
    # many in-budget designs are feasible narrows its search and is not derivable by hand. The public
    # archive's own post-condition check caught these, which is why they are dropped here.
    if isinstance(cfg.get("design_space"), dict):
        for k in ("size_in_change_budget", "size_in_change_budget_feasible"):
            cfg["design_space"].pop(k, None)
    if isinstance(cfg.get("objective"), dict):
        cfg["objective"].pop("note", None)          # names the population balance and the surrogate
    if isinstance(cfg.get("constraints"), dict):
        tpt = cfg["constraints"].get("total_process_time")
        if isinstance(tpt, dict):
            tpt.pop("note", None)

    body = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=100)
    header = (
        "# =================================================================================================\n"
        "# PUBLIC (model-visible) task configuration. GENERATED by scripts/make_public_task_config.py\n"
        "# from tasks/q1_agglomeration/task_config.yaml -- do not hand-edit; regenerate.\n"
        "#\n"
        "# The grader-side config additionally carries a `reference:` block with the reference optimum,\n"
        "# its design and responses, the three difficulty-gate measurements and the score-band\n"
        "# populations, and a `scoring.optimality_bands` ladder. None of that is here, and\n"
        "# scripts/package_for_delivery.py asserts it.\n"
        "#\n"
        "# DATA NATURE: synthetic simulator, not experimental data. See `data_nature` below.\n"
        "# =================================================================================================\n"
    )
    with open(PUBLIC, "w", encoding="utf-8") as fh:
        fh.write(header + body)

    # -------- assert nothing designer-side survived
    text = open(PUBLIC, encoding="utf-8").read()
    # Scan the YAML body only: the generated header legitimately NAMES the block it removed.
    banned = ["best_objective_in_budget", "best_design_in_budget", "gate_g1", "gate_g2",
              "gate_g3", "reference_enumeration", "answers/", "0.929797", "0.9297",
              "agglo_strength_porosity", "population balance", "MLP", "surrogate in data",
              # neighbourhood sizes are enumeration OUTPUT, not part of the graded contract, and are
              # not derivable by hand: telling the model how large the feasible region is narrows its
              # search. Caught by the public archive's own post-condition check, then removed here.
              "size_in_change_budget", "96124", "96 124", "114708", "114 708",
              # the achievement -> points ladder is grader-side from the second round on
              "optimality_bands", "min_achievement"]
    hits = [b for b in banned if b in body]
    if re.search(r"(?m)^reference:", body):
        hits.append("reference: block at top level")
    if hits:
        print(f"FAILED: designer-side strings survived in the public config: {hits}")
        return 1

    pub = yaml.safe_load(text)
    need = ["task_id", "task_name", "baseline_process", "design_space", "objective", "constraints",
            "iteration", "scoring", "feedback"]
    missing = [k for k in need if k not in pub]
    if missing:
        print(f"FAILED: the public config lost something the agent is graded on: {missing}")
        return 1
    if "reference" in pub:
        print("FAILED: the reference block is still present")
        return 1

    # the graded contract must be byte-equal on every field that decides a score
    import agglomeration as A                                            # noqa: E402
    for k in A.RESPONSE_KEYS:
        for f in ("pl", "pu", "al", "au", "w"):
            assert abs(float(pub["objective"]["windows"][k][f]) - float(A.WINDOWS[k][f])) < 1e-12, k
    assert pub["constraints"]["change_budget"]["max_cost"] == A.BUDGET
    # The public half must NOT carry the ladder any more; the grader-side config still does, and
    # check_consistency.py section 2 asserts that copy against design_space.POINT_BANDS.
    assert "optimality_bands" not in pub["scoring"], "the band ladder leaked into the public config"
    assert pub["scoring"].get("bands_withheld"), "the public config must say the ladder is withheld"
    assert int(pub["scoring"]["constraint_points"]) == 0

    print(f"wrote {os.path.relpath(PUBLIC, ROOT)}  ({os.path.getsize(PUBLIC)} bytes)")
    print(f"  dropped: {DROP_TOP_LEVEL} + scoring{DROP_SCORING_KEYS} + designer prose")
    print(f"  kept and verified identical: 4 windows x 5 fields, change budget {A.BUDGET}, "
          f"constraint_points 0")
    print(f"  withheld: the {len(dropped_scoring.get('optimality_bands', []))}-band achievement "
          f"ladder (grader-side only)")
    print("  no designer-side string survives the ban list")
    return 0


if __name__ == "__main__":
    sys.exit(main())

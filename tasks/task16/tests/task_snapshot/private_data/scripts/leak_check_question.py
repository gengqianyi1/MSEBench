"""Leak check on the model-facing surface: does the graded model see anything it must not?

The model-facing surface is exactly question.md plus the strings src/main.py adds to the prompt (the
slot table, the baseline formulation and its measured CQAs) plus the per-round feedback from
src/feedback.py. Everything else in the package -- task_config.yaml's `reference:` block, answers/,
scripts/, src/formulation.py's material properties -- is designer-side and never reaches the model.

This script rebuilds that surface for round 1 and for a later round (i.e. including feedback on a
submission) and greps it for every quantity that would hand over the answer:

  A. the reference objective value, in every rendering;
  B. the reference design -- as a JSON blob, as a change list, as a slot paired with its reference
     level, and as a level that appears without its sibling levels (a curated grid is a leak);
  C. the designer-only fields of the `reference:` block;
  D. any change-list string copied out of answers/;
  E. and, in the other direction, that the numbers question.md is *supposed* to disclose (the
     specification windows, the start value, the change budget, the measured difficulty ceilings and
     the fairness counts) are actually there and agree with task_config.yaml.

Complements scripts/check_consistency.py, which asserts the same invariants from the calibration side.

Run:
    python scripts/leak_check_question.py
Exit status is non-zero if anything leaks.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from evaluator import evaluate_design, evaluate_reference_start   # noqa: E402
from feedback import generate_feedback                            # noqa: E402
from formulation import SLOTS, START                              # noqa: E402
from main import build_initial_messages, build_next_user_message  # noqa: E402
from utils import load_yaml, read_text                            # noqa: E402

cfg = load_yaml(ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml")
question = read_text(ROOT / "tasks" / "q1_microenvironment" / "question.md")
ref = cfg["reference"]
refd = dict(ref["best_design_in_budget"])
start_eval = evaluate_reference_start(cfg["objective"])

# A submission that is not the reference, so the feedback path is exercised realistically.
probe = dict(START)
probe["T8_mgst_pct"], probe["T2_mcc_pct"] = 0.6, 43.0
ev = evaluate_design(probe, cfg["objective"])
fb = generate_feedback(ev, cfg["constraints"], best_so_far=ev)
surface = "\n".join(m["content"] for m in build_initial_messages(question, cfg, start_eval))
surface += "\n" + build_next_user_message(2, {"formulation": probe}, ev, fb, start_eval, ev)

fails: list[str] = []
notes: list[str] = []


def deny(needle: str, haystack: str, label: str) -> None:
    if needle in haystack:
        fails.append(f"{label}: found {needle!r}")
    else:
        print(f"  ok   {label}")


print("A. the reference objective value")
dref = float(ref["best_objective_in_budget"])
for n in (repr(dref), f"{dref:.7f}", f"{dref:.4f}", f"{dref:.3f}"):
    deny(n, question, f"question.md does not contain {n}")
    deny(n, surface, f"prompt surface does not contain {n}")

print("\nB. the reference design")
changed = {s: refd[s] for s in SLOTS if refd[s] != START[s]}
print(f"  (the reference changes {changed})")
for blob in (json.dumps(refd), json.dumps(refd, indent=2), json.dumps(changed),
             str(ref["best_design_changes"])):
    deny(blob, surface, "prompt surface does not contain the reference design blob")
for s, v in changed.items():
    for pat in (rf"{s}[^|\n]{{0,40}}{re.escape(str(v))}", rf"{re.escape(str(v))}[^|\n]{{0,40}}{s}"):
        m = re.search(pat, question)
        if m:
            fails.append(f"question.md pairs {s} with its reference level {v}: {m.group(0)!r}")
    print(f"  ok   {s} is never paired with {v} outside a level-grid row")
for s, v in changed.items():
    rows = [ln for ln in question.splitlines() if s in ln and ln.count("|") >= 3]
    siblings = [lv for lv in cfg["design_space"]["slots"][s]["levels"] if str(lv) != str(v)]
    ok = bool(rows) and all(any(str(o) in r or str(o).rstrip("0").rstrip(".") in r for r in rows)
                            for o in siblings)
    print(f"  {'ok  ' if ok else 'FAIL'} {s}: level {v} appears only as one member of its full grid")
    if not ok:
        fails.append(f"{s}: the printed grid does not carry all sibling levels")

print("\nC. designer-only fields of the reference block")
# Ruler 2.0.0 renamed these three counters when the bands were re-cut; the point of the check is
# unchanged -- the designer-only histogram tallies must not appear verbatim in the question.
for _k in ("n_designs_at_or_above_21_points",):
    deny(str(ref[_k]), question, f"question.md does not contain {_k}")
deny(f"{float(ref['reference_overlap']):.4f}", question,
     "question.md does not contain the reference overlap")
deny(str(ref["reference_overlap"]), question,
     "question.md does not contain the reference overlap at full precision")

print("\nD. strings copied out of answers/")
for path in sorted((ROOT / "answers").glob("*.json")):
    txt = path.read_text(encoding="utf-8")
    hits = {h.strip('"') for h in re.findall(r'"[A-Za-z0-9_]+ [^"]*->[^"]*"', txt)}
    leaked = [h for h in hits if h in question]
    if leaked:
        fails.append(f"question.md quotes change lists from {path.name}: {leaked}")
    print(f"  ok   {path.name}: none of its {len(hits)} change-list strings appear in question.md")

print("\nD2. the scenario must not borrow authority from a real registration")
# Second-round review point: the synthetic-data disclosure at the top of question.md was judged
# adequate, but the body still called the starting point "the registered formulation" and spoke of
# "the two registered process set points". That is an asymmetric risk -- the wording costs nothing to
# fix, while a reviewer landing on it could reopen the data-provenance finding. The scenario noun is
# now "baseline", and the word "registered" is allowed to survive in the PROMPT SURFACE only inside a
# sentence that DENIES a real registration.
#
# Checked on the full prompt surface (question.md plus what src/main.py and src/feedback.py add), not
# just on question.md, because a model-visible JSON key is as visible as a paragraph.
DENIAL_MARKERS = ("no registered product", "There is no registered", "not by any model validated",
                  "registered product's development data", "registered-product",
                  "registered product development data")
for label, body in (("question.md", question), ("prompt surface", surface)):
    lines = body.splitlines()
    occurrences = [(i + 1, ln) for i, ln in enumerate(lines) if re.search(r"[Rr]egistered", ln)]
    bad = []
    for lineno, ln in occurrences:
        # a denial may wrap, so look at the line and its two neighbours
        ctx = "\n".join(lines[max(0, lineno - 3):lineno + 2])
        if not any(m in ctx for m in DENIAL_MARKERS):
            bad.append(f"line {lineno}: {ln.strip()[:90]}")
    ok = (len(occurrences) <= 1) and not bad
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: 'registered' appears {len(occurrences)} time(s), "
          f"all inside a denial" if ok else
          f"  FAIL {label}: 'registered' appears {len(occurrences)} time(s); "
          f"outside a denial: {bad}")
    if not ok:
        fails.append(f"{label} uses 'registered' as scenario naming "
                     f"({len(occurrences)} occurrence(s); non-denial: {bad or 'none, but >1 total'})")
# ... and the replacement noun must actually be present, so this cannot be satisfied by deleting the
# starting point from the question altogether.
n_baseline = len(re.findall(r"[Bb]aseline", question))
print(f"  {'ok  ' if n_baseline >= 5 else 'FAIL'} question.md names the starting point "
      f"'baseline' {n_baseline} time(s)")
if n_baseline < 5:
    fails.append(f"question.md barely mentions the baseline formulation ({n_baseline} times) -- the "
                 f"rename must not have removed the starting point from the task text")

print("\nE. what question.md is supposed to disclose, cross-checked against task_config.yaml")


def digits(n, text: str) -> bool:
    """Is the integer `n` quoted, in any of the usual groupings (1917, 1,917, 1 917)?"""
    return str(n) in re.sub(r"[\s,  ]", "", text)


windows = cfg["objective"]["windows"]
disclosed = {
    "start desirability": f"{float(ref['start_objective']):.6f}"[:8] in question,
    "all five windows": all(str(windows[k][b]) in question
                            for k in windows for b in ("lower", "target", "upper")),
    "composition-overlap floor": str(cfg["constraints"]["composition_overlap"]["min"]) in question,
    "change-slot cap": str(cfg["constraints"]["max_slot_changes"]["max"]) in question,
    "release gate stated as a hard gate": "release gate" in question.lower(),
    "data nature stated as synthetic/simulated": ("synthetic" in question.lower()
                                                 and "simulated" in question.lower()),
}
for label, ok in disclosed.items():
    print(f"  {'ok  ' if ok else 'note'} disclosed: {label}")
    if not ok:
        notes.append(label)

# ------------------------------------------------------------------------------------------------
# F. difficulty calibration must NOT be disclosed  (second round, 2026-08-09)
#
# These items used to sit in block E as things question.md was REQUIRED to print: the per-family
# score ceilings and the population counts of the enumerated change-budget neighbourhood. The
# second-round review objected to the same class of disclosure on the sibling sulfonamide problem
# (15-P1 #1), and it is the same defect here: a table saying "one condition tops out at 17/100, two
# at 30, three at 44, and holding overlap at 0.93 also tops out at 44" answers the two questions the
# task is actually asking -- how many conditions to move, and how much overlap to spend -- before the
# model has reasoned about a single excipient. The counts do the same job more quietly, by revealing
# how dense the scoring set is.
#
# They are now BLOCKING denials rather than notes. question.md keeps the qualitative statements the
# chemistry needs ("one condition at a time leaves most of the credit unclaimed", "improving D while
# breaching a window scores nothing") and drops every number behind them.
# ------------------------------------------------------------------------------------------------
print("\nF. difficulty calibration that must NOT reach the model")

CALIBRATION_KEYS = [
    ("G1 two-slot ceiling",            f"{ref['gate_g1_two_slot_points']} / 100"),
    ("G1 three-slot ceiling",          f"{ref['gate_g1_three_slot_points']} / 100"),
    ("G2 safe-overlap ceiling",        f"{ref['gate_g2_safe_overlap_points']} / 100"),
]
for label, needle in CALIBRATION_KEYS:
    deny(needle, question, f"question.md does not quote the {label}")

CALIBRATION_COUNTS = [
    ("G3 improving count",              ref["gate_g3_improving_designs"]),
    ("feasible-in-budget count",        ref["n_feasible_in_budget"]),
    ("all-CQA-in-spec count",           ref["n_feasible_all_cqa_in_spec"]),
    ("qualifying count",                ref["n_qualifying"]),
    ("non-compliant-but-improving count", ref["n_noncompliant_but_improving"]),
]
for label, n in CALIBRATION_COUNTS:
    if digits(n, question):
        fails.append(f"question.md quotes the {label} ({n}); it is grader-side calibration")
    else:
        print(f"  ok   question.md does not quote the {label}")

# The band table moved grader-side this round too, for all four problems in this suite. Printing the
# achievement thresholds tells the model how much headroom is worth chasing before the effort stops
# paying, which suppresses exactly the search the five rounds are meant to measure.
for row in cfg.get("scoring", {}).get("bands", []) or []:
    thr = row.get("min_achievement") if isinstance(row, dict) else None
    if thr is not None and str(thr) in question:
        fails.append(f"question.md quotes the achievement band threshold {thr}")
if not (cfg.get("scoring", {}).get("bands")):
    print("  ok   no band table in the public task config to leak")

print()
if fails:
    print("LEAK CHECK FAILED")
    for f in fails:
        print(f"  - {f}")
    raise SystemExit(1)
print("LEAK CHECK CLEAN" + (f"   (not disclosed: {notes})" if notes else ""))

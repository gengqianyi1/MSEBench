"""Does touching a given condition family cap the score? Measured, per family, with the shipped ruler.

Section 2.3 of 审核说明.md makes a chemistry claim -- "the textbook first reaction to a high
microenvironment pH is to add or strengthen the acidifier, and on this product that move cannot pay
for itself" -- and a claim like that is only worth printing if it is a number. This script turns each
claim into a ceiling: among all budget-feasible designs that TOUCH a family of slots, what is the
highest score reachable? A family whose ceiling is 100 is not a trap, whatever the chemistry story
says, and the sentence has to go.

Measured against the recalibrated (smooth) ruler this changed two of the three claims that the first,
hard-zero calibration supported, which is exactly why it is a script and not a paragraph:

    family                      hard-zero ruler     smooth ruler (shipped)
    acidifier (T3/T4)                33/100                47/100     still a real cap
    mannitol (T12)                   63/100               100/100     NOT a trap any more
    no trap slot touched            100/100               100/100     unchanged

Run:
    python scripts/measure_trap_families.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from design_space import (VectorOracle, achievement, calibrate_windows,  # noqa: E402
                          describe_row, enumerate_rows, filler_pct_vec, n_changed_vec,
                          n_identity_vec, overlap_vec, points_for, START_IDX)
from evaluator import scorer_from_config  # noqa: E402
from formulation import CQA_KEYS, LEVELS, MIN_FILLER_PCT, SLOTS  # noqa: E402
from utils import load_yaml  # noqa: E402

TASK_CFG = ROOT / "tasks" / "q1_microenvironment" / "task_config.yaml"
OUT = ROOT / "answers" / "trap_family_probes.json"

# The three families 审核说明.md section 2.3 argues about. "Trap slots" are the four things a
# formulator reaches for first on this product: the acidifier (identity or amount), meglumine, the
# disintegrant identity, and mannitol.
TRAP_SLOTS = ("T3_acid_id", "T4_acid_pct", "T5_meg_pct", "T6_dis_id", "T12_man_pct")
FAMILIES = {
    "touches the acidifier (T3/T4)": ("T3_acid_id", "T4_acid_pct"),
    "introduces mannitol (T12)": ("T12_man_pct",),
    "touches meglumine (T5)": ("T5_meg_pct",),
    "changes the disintegrant identity (T6)": ("T6_dis_id",),
}


def main() -> None:
    cfg = load_yaml(TASK_CFG)
    cal = cfg["reference"]["calibration"]
    cons = cfg["constraints"]
    floor = float(cal["overlap_floor"])

    O = VectorOracle()
    R = enumerate_rows(int(cons["max_slot_changes"]["max"]))
    Y = O.predict_rows(R)
    y0 = O.predict_rows(START_IDX[None, :])[0]

    # The shipped ruler, taken from the config the grader reads, then asserted to be on the windows a
    # fresh calibration re-derives. A re-derived look-alike is the failure mode this series has hit
    # four times: the offline probe and the real grader must be the same object.
    win = calibrate_windows(y0, Y, float(cal["pull"]), float(cal["marg"]))
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
    # Ruler 2.0.0 compliance gate: only designs that can actually score are counted.
    FEAS = (sc.in_spec(Y) & (D > D0) & (overlap_vec(R) >= floor - 1e-12)
            & (n_changed_vec(R) <= int(cons["max_slot_changes"]["max"]))
            & (n_identity_vec(R) <= int(cons["max_identity_changes"]["max"]))
            & (filler_pct_vec(R) >= MIN_FILLER_PCT))

    def touched(slots) -> np.ndarray:
        idx = [SLOTS.index(s) for s in slots]
        return (R[:, idx] != START_IDX[idx]).any(axis=1)

    def ceiling(mask: np.ndarray) -> dict:
        m = mask & FEAS
        n = int(m.sum())
        if n == 0:
            return {"n": 0, "points": 0, "best_value": None, "best_achievement": 0.0,
                    "best_changes": ""}
        j = int(np.flatnonzero(m)[np.argmax(D[m])])
        ach = achievement(float(D[j]), D0, Dref)
        return {"n": n, "points": int(points_for(float(D[j]), D0, Dref)), "best_value": float(D[j]),
                "best_achievement": float(ach), "best_changes": describe_row(R[j])}

    print("baseline D = %.7f   reference D = %.7f   scoring-eligible designs "
          "(budget-feasible AND all five CQAs in spec AND improving) = %d"
          % (D0, Dref, int(FEAS.sum())))
    print("\nceiling among SCORING-ELIGIBLE designs that TOUCH each family:")
    out = {"families": {}, "trap_slots": list(TRAP_SLOTS)}
    for name, slots in FAMILIES.items():
        rec = ceiling(touched(slots))
        rec["slots"] = list(slots)
        out["families"][name] = rec
        print("  %-40s n = %6d   ceiling %3d / 100   (achievement %.4f)   %s"
              % (name, rec["n"], rec["points"], rec["best_achievement"], rec["best_changes"]))

    clean = ceiling(~touched(TRAP_SLOTS))
    clean["slots"] = [s for s in SLOTS if s not in TRAP_SLOTS]
    out["families"]["no trap slot touched"] = clean
    print("  %-40s n = %6d   ceiling %3d / 100   (achievement %.4f)   %s"
          % ("no trap slot touched", clean["n"], clean["points"], clean["best_achievement"],
             clean["best_changes"]))

    # Is the reference design itself inside the clean family? If not, "avoid the trap slots" is not
    # the lesson of this problem and section 2.3 has no claim left to make.
    ref = np.array([LEVELS[s].index(cfg["reference"]["best_design_in_budget"][s]) for s in SLOTS],
                   dtype=np.int16)
    ti = [SLOTS.index(s) for s in TRAP_SLOTS]
    ref_clean = not bool((ref[ti] != START_IDX[ti]).any())
    out["reference_touches_no_trap_slot"] = ref_clean
    print("\nreference design touches no trap slot: %s" % ref_clean)

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s" % OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()

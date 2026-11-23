"""Designer/reviewer check: everything stated in more than one place must agree.

This exists because of a specific, repeated failure. During the offline design of this problem, one
on-disk artifact held a FALLBACK calibration (pull 0.28 / marg 0.50) while the instance file
specified another (0.22 / 0.30) -- a build script reading the wrong one would have shipped the wrong
specification windows, with a difficulty gate one band away from failing. Nobody noticed by reading.
So the invariant is asserted, not inspected: the calibration in task_config.yaml must equal what
scripts/enumerate_reference_best.py recomputes from scratch, right now, with the shipped oracle.

Checks:
  1. the oracle loads, and the vectorised path equals the scalar path exactly;
  2. the baseline formulation is on-grid, closes mass, and satisfies every hard constraint;
  3. the CALIBRATION stored in task_config.yaml (windows, start value, reference value, gates) equals
     a fresh recomputation -- no cached JSON is trusted for this;
  4. answers/reference_enumeration.json agrees with task_config.yaml (no drifted artifact);
  5. the difficulty and fairness gates still pass (G1 < 50, G2 < 50, G3 >= 50);
  6. the scoring block is internally consistent and constraint_points is 0;
  7. question.md quotes the real windows and the real constraints; the only complete formulation it
     prints is the baseline one; and every difficulty ceiling and count it discloses equals the
     enumeration (the ceilings are disclosed on purpose -- the reference design and value are not);
  8. question.md presents all twelve variables neutrally -- complete level grid for every slot, every
     level code decoded into a substance -- and states no excipient mechanism direction;
  9. the reference optimum sits on no grid endpoint and no constraint boundary;
 10. no run output contains anything key-shaped;
 11. src/scoring.py BEHAVES the way the scoring block of task_config.yaml describes -- every band
     boundary is exercised from both sides with a synthetic submission, the feasibility gate really
     awards zero, achievement really clips at 1.0, and a non-zero constraint_points really raises;
 12. question.md's band table, hard-constraint numbers, round budget and output-format template agree
     with task_config.yaml -- and the template parses through the real src/parser.py;
 13. every run's stored final_report.json score is reproducible: re-scoring the submissions saved in
     that run's trajectory with the shipped scorer returns the same total, achievement and counts;
 14. 审核说明.md -- the cross-review document -- is synchronised with the shipped ruler: the window
     table, the gate ceilings, the naive-probe ceiling and BOTH score columns (this calibration and
     the withdrawn hard-zero one) are re-derived and asserted against the runs on disk, so the
     fairness argument cannot quote a number the code no longer produces.

Run:
    python scripts/check_consistency.py
Exit status is non-zero if any assertion fails, so this is usable as a pre-delivery hook.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from design_space import (POINT_BANDS, VectorOracle, Desirability, achievement,  # noqa: E402
                          budget_mask, compliance_mask, qualifying_mask,
                          acid_ok_vec, calibrate_windows, comp_matrix, describe_row,
                          enumerate_exact_k, enumerate_rows, feat_matrix, filler_pct_vec,
                          n_changed_vec, n_identity_vec, overlap_vec, points_for, START_IDX,
                          to_design, to_row)
from evaluator import (FormulationEvaluation, check_constraints, evaluate_design,  # noqa: E402
                       evaluate_reference_start, oracle_vs_generator_drift, scorer_from_config)
from formulation import (CQA_KEYS, LEVELS, MIN_FILLER_PCT, SLOTS, START,  # noqa: E402
                         composition, ruzicka_overlap)
from response_model import Oracle, feature_vector  # noqa: E402
from parser import parse_candidate  # noqa: E402
from scoring import final_score  # noqa: E402
from tools import (DEFAULT_MAX_TOOL_TURNS_PER_ROUND, DEFAULT_TOOL_BUDGET_PER_ROUND,  # noqa: E402
                   TOOL_NAMES, run_tool_call, run_tool_calls, tool_catalogue)
from utils import load_yaml  # noqa: E402

from rescore_run import submitted_formulations  # noqa: E402

TASK_DIR = ROOT / "tasks" / "q1_microenvironment"
TASK_CFG = TASK_DIR / "task_config.yaml"
QUESTION = TASK_DIR / "question.md"
ENUM = ROOT / "answers" / "reference_enumeration.json"
REVIEW = ROOT / "审核说明.md"
NAIVE = ROOT / "answers" / "naive_answer_probes.json"
OLD_RUNS = ROOT / "runs_old_hardzero"

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def approx(a, b, tol=1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main() -> None:
    # `--skip-trajectories` matches the flag the sibling packages in this suite already carry. It
    # exists for the window between a prompt revision and the re-run that follows it: raising
    # prompt_version deliberately invalidates every shipped trajectory, so section 13 SHOULD fail
    # until new runs land. Without a way to separate the two, a real structural regression would hide
    # among a dozen expected "no trajectories yet" failures. A delivery run must not use this flag.
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-trajectories", action="store_true",
                    help="DEV ONLY: skip section 13 (per-run score reproduction). A delivery run "
                         "must not use this.")
    args = ap.parse_args()

    cfg = load_yaml(TASK_CFG)
    constraints = cfg["constraints"]
    objective_cfg = cfg["objective"]
    ref_cfg = cfg["reference"]
    cal = ref_cfg["calibration"]

    print("1. oracle, and the vectorised path against the scalar path")
    O = VectorOracle()
    Osc = Oracle()
    R = enumerate_rows(int(cal.get("max_slot_changes", 4)) if "max_slot_changes" in cal else 4)
    rng = np.random.default_rng(11)
    sel = rng.choice(len(R), 60, replace=False)
    mx_feat = mx_or = mx_comp = mx_ov = 0.0
    for j in sel:
        d = to_design(R[j])
        mx_feat = max(mx_feat, float(np.abs(feature_vector(d) - feat_matrix(R[j][None, :])[0]).max()))
        mx_or = max(mx_or, float(np.abs(Osc.predict([d])[0] - O.predict_rows(R[j][None, :])[0]).max()))
        w = np.array([composition(d)[s] for s in comp_order()])
        mx_comp = max(mx_comp, float(np.abs(w - comp_matrix(R[j][None, :])[0]).max()))
        mx_ov = max(mx_ov, abs(ruzicka_overlap(d) - float(overlap_vec(R[j][None, :])[0])))
    check(mx_feat == 0.0, f"feat_matrix == feature_vector exactly (max diff {mx_feat:.3e})")
    check(mx_or == 0.0, f"vectorised oracle == scalar oracle exactly (max diff {mx_or:.3e})")
    check(mx_comp < 1e-12, f"comp_matrix == composition() (max diff {mx_comp:.3e})")
    check(mx_ov < 1e-12, f"overlap_vec == ruzicka_overlap (max diff {mx_ov:.3e})")
    drift = oracle_vs_generator_drift(START)
    win_width = np.array([objective_cfg["windows"][k]["upper"] - objective_cfg["windows"][k]["lower"]
                          for k in CQA_KEYS])
    rel = drift / win_width
    check(rel.max() < 0.02,
          f"oracle-vs-generator drift at the baseline formulation < 2 % of window width "
          f"(max {rel.max() * 100:.2f} %)")

    print("\n2. the baseline formulation")
    start_eval = evaluate_design(dict(START), objective_cfg)
    check(start_eval.design is not None, "baseline formulation is on-grid")
    check(start_eval.valid, f"baseline formulation is manufacturable ({start_eval.error})")
    for name, ok in check_constraints(start_eval, constraints).items():
        check(ok, f"baseline formulation satisfies {name}")
    check(approx(start_eval.composition_overlap, 1.0), "baseline formulation has overlap 1.0")
    if start_eval.desirability_per_cqa:
        ds = np.array(list(start_eval.desirability_per_cqa.values()))
        # With the smooth desirability d_i > 0 always, so "d > 0" is no longer evidence of anything;
        # the in-spec flag is computed against the printed limits instead.
        check(bool(start_eval.all_cqa_in_spec),
              "baseline formulation is INSIDE every window (no CQA already out of specification)")
        check(ds.max() < 0.05,
              f"baseline formulation sits on the dangerous edge of every window "
              f"(max per-CQA desirability {ds.max():.6g})")
    check(dict(cfg["baseline_formulation"]) == {k: START[k] for k in SLOTS},
          "task_config.baseline_formulation == src/formulation.START")

    # The declared design space must be the one the oracle and the enumeration actually use. This is
    # the same class of invariant as the calibration check: a level grid that exists in two places is
    # a level grid that can drift, and a drifted grid changes the difficulty silently.
    ds = cfg["design_space"]
    check(list(ds["slots"].keys()) == SLOTS, "task_config.design_space slot order == formulation.SLOTS")
    bad = [s for s in SLOTS
           if [norm_level(v) for v in ds["slots"][s]["levels"]] != [norm_level(v) for v in LEVELS[s]]]
    check(not bad, f"task_config.design_space levels == formulation.LEVELS ({bad})")
    from formulation import (API_PCT, CORE_TABLET_MASS_MG, IDENTITY_SLOTS,  # noqa: E402
                             PROCESS_SLOTS, SUBSTANCE_LABELS, design_space_size)
    check(list(ds["identity_change_slots"]) == IDENTITY_SLOTS
          and list(ds["process_slots"]) == PROCESS_SLOTS,
          "task_config.design_space identity/process slots == formulation")
    check(approx(ds["core_tablet_mass_mg"], CORE_TABLET_MASS_MG) and approx(ds["api_pct_of_core"], API_PCT),
          "task_config.design_space core mass and API load == formulation")
    sz = design_space_size()
    check(int(ds["size_gross"]) == sz["gross"] and int(ds["size_acid_consistent"]) == sz["acid_consistent"],
          f"task_config.design_space sizes == formulation.design_space_size() ({sz})")
    check(dict(ds["substances"]) == SUBSTANCE_LABELS,
          "task_config.design_space.substances == formulation.SUBSTANCE_LABELS")

    print("\n3. the calibration in task_config.yaml equals a FRESH recomputation")
    Y = O.predict_rows(R)
    y0 = O.predict_rows(START_IDX[None, :])[0]
    win = calibrate_windows(y0, Y, float(cal["pull"]), float(cal["marg"]))
    L, T, U = win["L"], win["T"], win["U"]
    shape = objective_cfg["desirability"]
    sc = Desirability(L, T, U, hill_exponent=float(shape["hill_exponent"]),
                      d_at_spec_limit=float(shape["d_at_spec_limit"]),
                      tail_decay_multiplier=float(shape["tail_decay_multiplier"]),
                      floor=float(shape["floor"]),
                      plateau_fraction=float(shape["plateau_fraction"]),
                      weights=[float(x) for x in shape["weights"]])
    for i, key in enumerate(CQA_KEYS):
        w = objective_cfg["windows"][key]
        check(approx(w["lower"], L[i]) and approx(w["target"], T[i]) and approx(w["upper"], U[i]),
              f"window {key} == recomputed ({w['lower']}/{w['target']}/{w['upper']} vs "
              f"{L[i]}/{T[i]}/{U[i]})")
    # The desirability SHAPE is part of the ruler, so it is subject to the same
    # stated-in-two-places-must-agree rule as the windows.
    cal_shape = cal["desirability"]
    check(set(cal_shape) == set(shape) - {"form"},
          f"reference.calibration.desirability has the same keys as objective.desirability "
          f"({sorted(set(cal_shape) ^ (set(shape) - {'form'}))})")
    for k in sorted(set(cal_shape) & set(shape)):
        a, b = cal_shape[k], shape[k]
        same = (all(approx(x, y) for x, y in zip(a, b)) and len(a) == len(b)
                if isinstance(a, list) else approx(a, b))
        check(same, f"objective.desirability.{k} == reference.calibration.desirability.{k} "
                    f"({b} vs {a})")
    check(approx(constraints["composition_overlap"]["min"], cal["overlap_floor"]),
          "constraints.composition_overlap.min == reference.calibration.overlap_floor")

    # ---- the desirability really is smooth, and really has no hard zero -------------------------
    # This is the property the recalibration exists to create, so it is asserted rather than trusted.
    us = np.linspace(0.0, 4.0, 4001)
    dus = np.array([sc.d_at_u(u) for u in us])
    check(bool((dus > 0).all()), "d_i > 0 everywhere: the desirability has no hard zero")
    check(bool((np.diff(dus) <= 1e-15).all()),
          "d_i is non-increasing in the distance from the target (every window is two-sided)")
    check(approx(sc.d_at_u(0.0), 1.0, 1e-12), "d_i = 1 at the target")
    d_limit = float(shape["floor"]) + (1.0 - float(shape["floor"])) * float(shape["d_at_spec_limit"])
    check(approx(sc.d_at_u(1.0), d_limit, 1e-12),
          f"d_i at the printed specification limit == floor + (1-floor) * d_at_spec_limit "
          f"({sc.d_at_u(1.0):.6g} vs {d_limit:.6g})")
    jump = abs(sc.d_at_u(1.0 + 1e-6) - sc.d_at_u(1.0 - 1e-6))
    check(jump < 1e-8, f"d_i is continuous across the specification limit (jump {jump:.2e}) -- the "
                       f"cliff that scored every calibration model 0 is gone")
    check(sc.d_at_u(50.0) >= float(shape["floor"]) * (1 - 1e-9),
          "d_i decays to the configured floor rather than to 0")
    # ... and the floor is far below the baseline formulation's own per-CQA desirability, or every
    # design would beat the start and the problem would collapse in the other direction.
    d_start_min = float(sc.d_of(y0[None, :])[0].min())
    check(float(shape["floor"]) < d_start_min / 100.0,
          f"the desirability floor {float(shape['floor']):.1e} is at least 100x below the "
          f"baseline formulation's per-CQA desirability {d_start_min:.3e}")

    floor = float(cal["overlap_floor"])
    D = sc.D_of(Y)
    D0 = sc.D_one(y0)
    OV, NC, NI, FILL = overlap_vec(R), n_changed_vec(R), n_identity_vec(R), filler_pct_vec(R)
    max_slots = int(constraints["max_slot_changes"]["max"])
    max_ident = int(constraints["max_identity_changes"]["max"])
    # Ruler 2.0.0: THREE populations, and the distinction is the whole point of audit finding #3.
    #   BUDGET  the PACMP change budget only        -- a design-space fact
    #   IN      all five CQAs inside [L, U]         -- the release / compliance gate
    #   QUAL    BUDGET & IN & D > D_start           -- the ONLY set that can score
    # Every gate below is measured over QUAL, and the reference optimum is the compliant argmax.
    IN = sc.in_spec(Y)
    BUDGET = ((OV >= floor - 1e-12) & (NC <= max_slots) & (NI <= max_ident)
              & (FILL >= float(constraints["mass_closure"]["min_filler_pct"])))
    check((BUDGET == budget_mask(R, floor, max_slots, max_ident,
                                 float(constraints["mass_closure"]["min_filler_pct"]))).all(),
          "the locally spelled-out budget predicate == design_space.budget_mask")
    check((IN == compliance_mask(Y, sc)).all(),
          "sc.in_spec == design_space.compliance_mask (the release gate has one definition)")
    jref = int(np.argmax(np.where(BUDGET & IN, D, -1.0)))
    Dref = float(D[jref])
    QUAL = BUDGET & IN & (D > D0)
    check((QUAL == qualifying_mask(R, Y, sc, D0, floor, max_slots, max_ident,
                                   float(constraints["mass_closure"]["min_filler_pct"]))).all(),
          "the locally spelled-out qualifying predicate == design_space.qualifying_mask")
    FEAS = QUAL

    check(approx(ref_cfg["start_objective"], D0),
          f"reference.start_objective == recomputed ({ref_cfg['start_objective']} vs {D0})")
    check(approx(ref_cfg["best_objective_in_budget"], Dref),
          f"reference.best_objective_in_budget == recomputed "
          f"({ref_cfg['best_objective_in_budget']} vs {Dref})")
    check(dict(ref_cfg["best_design_in_budget"]) == to_design(R[jref]),
          "reference.best_design_in_budget == recomputed argmax")
    check(approx(ref_cfg["reference_overlap"], float(OV[jref])),
          "reference.reference_overlap == recomputed")
    check(int(ref_cfg["n_feasible_in_budget"]) == int(BUDGET.sum()),
          f"reference.n_feasible_in_budget == recomputed ({BUDGET.sum()})")
    check(int(cfg["design_space"]["size_in_change_budget"]) == int(BUDGET.sum()),
          "design_space.size_in_change_budget == reference.n_feasible_in_budget")
    check(int(ref_cfg["n_rows_enumerated"]) == int(len(R)),
          f"reference.n_rows_enumerated == recomputed ({len(R)})")
    check(int(ref_cfg["n_feasible_all_cqa_in_spec"]) == int((IN & BUDGET).sum()),
          f"reference.n_feasible_all_cqa_in_spec == recomputed ({int((IN & BUDGET).sum())})")
    check(approx(ref_cfg["five_window_closure_rate"], (IN & BUDGET).sum() / BUDGET.sum(), 1e-12),
          "reference.five_window_closure_rate == recomputed")
    check(int(ref_cfg["n_qualifying"]) == int(QUAL.sum()),
          f"reference.n_qualifying == recomputed ({int(QUAL.sum())})")
    check(approx(ref_cfg["qualifying_rate"], QUAL.sum() / BUDGET.sum(), 1e-12),
          "reference.qualifying_rate == recomputed")
    check(int(ref_cfg["n_noncompliant_but_improving"]) == int((BUDGET & ~IN & (D > D0)).sum()),
          f"reference.n_noncompliant_but_improving == recomputed "
          f"({int((BUDGET & ~IN & (D > D0)).sum())}) -- these scored under ruler 1.x and score 0 now")
    # Fairness floors. The 0.x ruler let only 2.0 % of the budget-feasible set close all five windows
    # and only 0.98 % score at all; 1.x fixed the second number by letting non-compliant designs
    # score, which is what audit finding #3 rejected. 2.0.0 has to clear both floors WITH the release
    # gate on, so these two assertions are the ones that keep the problem answerable.
    check((IN & BUDGET).sum() / BUDGET.sum() > 0.04,
          f"more than 4 % of the budget-feasible set closes all five windows "
          f"({100.0 * (IN & BUDGET).sum() / BUDGET.sum():.2f} %)")
    check(QUAL.sum() / BUDGET.sum() > 0.05,
          f"more than 5 % of the budget-feasible set can score UNDER THE RELEASE GATE "
          f"({100.0 * QUAL.sum() / BUDGET.sum():.2f} %)")
    check(bool(IN[jref]),
          "the reference optimum has all five CQAs inside their windows (a non-releasable reference "
          "would reintroduce audit finding #3 into the ruler itself)")
    check(float(np.max(np.where(BUDGET & ~IN, D, -1.0))) < Dref,
          "no non-compliant design in the budget beats the compliant reference optimum, so the "
          "out-of-window tail is steep enough that 'sacrifice one CQA' is never the argmax "
          f"(best non-compliant D = {float(np.max(np.where(BUDGET & ~IN, D, -1.0))):.6f} "
          f"< {Dref:.6f})")

    # the scorer the harness actually uses must be the same ruler, shape parameters included
    sc_cfg = scorer_from_config(objective_cfg)
    check(np.abs(sc_cfg.L - L).max() == 0 and np.abs(sc_cfg.T - T).max() == 0
          and np.abs(sc_cfg.U - U).max() == 0 and sc_cfg.params() == sc.params(),
          f"the scorer built from task_config.yaml IS the recomputed scorer "
          f"({sc_cfg.params()} vs {sc.params()})")
    check(float(np.abs(sc_cfg.D_of(Y[:5000]) - sc.D_of(Y[:5000])).max()) == 0.0,
          "the config scorer and the recomputed scorer agree bit-for-bit on 5,000 designs")
    # ... and the two PATHS agree: the vectorised D used by every enumeration and gate above against
    # the scalar D that src/main.py actually grades a submission with. Every earlier problem in this
    # series that shipped a wrong difficulty did it by measuring gates with a stand-in ruler.
    sel2 = rng.choice(len(R), 40, replace=False)
    worst = 0.0
    for j in sel2:
        ev_j = evaluate_design(to_design(R[j]), objective_cfg)
        worst = max(worst, abs(float(ev_j.desirability) - float(D[j])) / max(float(D[j]), 1e-300))
    check(worst < 1e-8,
          f"scalar evaluate_design D == vectorised D on 40 random designs (max relative "
          f"difference {worst:.2e}; the gap is only the 9-significant-digit rounding in the report)")

    print("\n4. gates in task_config.yaml equal a fresh recomputation, and still pass")
    g = {}
    for k in (1, 2, 3):
        Rk = enumerate_exact_k(k)
        Yk = O.predict_rows(Rk)
        ok = (acid_ok_vec(Rk) & (filler_pct_vec(Rk) >= MIN_FILLER_PCT)
              & (n_identity_vec(Rk) <= max_ident) & (overlap_vec(Rk) >= floor - 1e-12)
              & compliance_mask(Yk, sc) & (sc.D_of(Yk) > D0))    # ruler 2.0.0 release gate
        Dk = np.where(ok, sc.D_of(Yk), -1.0)
        g[k] = points_for(float(Dk.max()), D0, Dref)
    check(ref_cfg["gate_g1_one_slot_points"] == g[1], f"G1 one-slot == recomputed ({g[1]})")
    check(ref_cfg["gate_g1_two_slot_points"] == max(g[1], g[2]),
          f"G1 two-slot == recomputed ({max(g[1], g[2])})")
    check(ref_cfg["gate_g1_three_slot_points"] == max(g.values()),
          f"G1 three-slot == recomputed ({max(g.values())})")

    margin = float(ref_cfg["gate_g2_safe_margin"])
    FEAS2 = ((OV >= floor + margin - 1e-12) & (NC <= max_slots) & (NI <= max_ident)
             & (FILL >= MIN_FILLER_PCT) & IN & (D > D0))          # ruler 2.0.0 release gate
    g2 = points_for(float(np.where(FEAS2, D, -1.0).max()), D0, Dref)
    check(ref_cfg["gate_g2_safe_overlap_points"] == g2, f"G2 == recomputed ({g2})")
    g3 = int(QUAL.sum())
    check(ref_cfg["gate_g3_improving_designs"] == g3, f"G3 == recomputed ({g3})")

    check(max(g[1], g[2]) < 50, "G1 < 50 (difficulty gate)")
    check(g2 < 50, "G2 < 50 (difficulty gate)")
    check(g3 >= 50, "G3 >= 50 (fairness gate: the objective is genuinely improvable)")
    # The histogram spans the BUDGET-feasible set; a non-compliant design scores 0, which is exactly
    # what makes the 0 bucket the size it is.
    pts_all = np.array([points_for(D[q], D0, Dref) if QUAL[q] else 0
                        for q in np.where(BUDGET)[0]])
    n21, n30, n44 = (int((pts_all >= 21).sum()), int((pts_all >= 30).sum()),
                     int((pts_all >= 44).sum()))
    check(ref_cfg["n_designs_at_or_above_21_points"] == n21,
          f"n_designs_at_or_above_21_points == recomputed ({n21})")
    check(ref_cfg["n_designs_at_or_above_30_points"] == n30,
          f"n_designs_at_or_above_30_points == recomputed ({n30})")
    check(ref_cfg["n_designs_at_or_above_44_points"] == n44,
          f"n_designs_at_or_above_44_points == recomputed ({n44})")
    check(n21 >= 100,
          f"at least 100 designs reach 21 points UNDER THE RELEASE GATE, i.e. the scoring surface is "
          f"a slope and not a wall ({n21})")
    band_pts = sorted({0} | {int(p) for _, p in POINT_BANDS})
    hist = {str(p): int((pts_all == p).sum()) for p in band_pts}
    check({str(k): int(v) for k, v in ref_cfg["score_histogram"].items()} == hist,
          f"reference.score_histogram == recomputed ({hist})")
    n_populated = sum(1 for p in band_pts if p > 0 and hist[str(p)] > 0)
    check(n_populated == len(band_pts) - 1,
          f"every one of the {len(band_pts) - 1} scoring bands is populated inside the change budget "
          f"({n_populated} populated; {hist})")

    print("\n5. answers/reference_enumeration.json has not drifted")
    if not ENUM.exists():
        check(False, f"{ENUM.relative_to(ROOT)} exists (run scripts/enumerate_reference_best.py)")
    else:
        e = json.loads(ENUM.read_text(encoding="utf-8"))
        ec = e["calibration"]
        check(approx(ec["pull"], cal["pull"]) and approx(ec["marg"], cal["marg"])
              and approx(ec["overlap_floor"], cal["overlap_floor"]),
              "enumeration calibration == task_config calibration")
        check(all(approx(ec["desirability"][k], v) if not isinstance(v, list)
                  else all(approx(x, y) for x, y in zip(ec["desirability"][k], v))
                  for k, v in cal["desirability"].items()),
              f"enumeration desirability shape == task_config calibration.desirability "
              f"({ec['desirability']})")
        check(approx(e["start"]["value"], D0), "enumeration start value == recomputed")
        check(approx(e["reference"]["value"], Dref), "enumeration reference value == recomputed")
        check(e["windows"]["L"] == L.tolist() and e["windows"]["T"] == T.tolist()
              and e["windows"]["U"] == U.tolist(), "enumeration windows == recomputed")
        check(e["gates"]["G1_two_slot_points"] == max(g[1], g[2])
              and e["gates"]["G2_points"] == g2 and e["gates"]["G3_count"] == g3,
              "enumeration gates == recomputed")
        check(e["gate_verdict"] == "PASS", "enumeration gate verdict is PASS")

    print("\n6. scoring block")
    scoring = cfg["scoring"]
    check(int(scoring["constraint_points"]) == 0,
          "constraint_points == 0 (feasibility is a gate, not credit)")
    check(scoring["optimality_points"] == scoring["total_points"],
          "optimality_points == total_points")
    bands = scoring["optimality_bands"]
    mins = [float(b["min_achievement"]) for b in bands]
    ptss = [int(b["points"]) for b in bands]
    check(mins == sorted(mins, reverse=True), "band thresholds are descending")
    check(ptss == sorted(ptss, reverse=True), "band points are descending")
    check(max(ptss) == scoring["total_points"], "top band awards the full total")
    check([(m, p) for m, p in zip(mins, ptss)] == [(float(a), int(p)) for a, p in POINT_BANDS],
          "task_config bands == design_space.POINT_BANDS (the enumeration's own bands)")

    print("\n7. question.md")
    text = QUESTION.read_text(encoding="utf-8") if QUESTION.exists() else ""
    check(bool(text), "question.md exists and is non-empty")
    if text:
        body = strip_html_comments(text)
        for i, key in enumerate(CQA_KEYS):
            w = objective_cfg["windows"][key]
            quoted = all(fmt(v) in body for v in (w["lower"], w["target"], w["upper"]))
            check(quoted, f"question.md quotes the {key} window "
                          f"{w['lower']}/{w['target']}/{w['upper']}")
        check(fmt(constraints["composition_overlap"]["min"]) in body,
              "question.md quotes the composition-overlap floor")
        check(str(max_slots) in body, "question.md quotes the change-slot cap")

        # ------------------------------------------------------------------------------------------
        # THE RULER, AS PRINTED. The model is scored by objective.desirability; what it can read is
        # the prose in question.md. If the two disagree the problem is unfair in the most direct way
        # available, so every number of the shape is asserted present, and the printed d(u) table is
        # asserted equal to what the SHIPPED scorer returns at each printed u -- not to a formula
        # re-implemented here. This block is the reason the recalibration could be shipped at all:
        # the first version of this problem printed a piecewise-cubic desirability that the code no
        # longer implements, and nothing failed.
        # ------------------------------------------------------------------------------------------
        des = objective_cfg["desirability"]
        for label, value, rendered in (
                ("hill_exponent M", des["hill_exponent"], [fmt(des["hill_exponent"])]),
                ("d_at_spec_limit e", des["d_at_spec_limit"], ["2.0e-4", "2e-4"]),
                ("tail_decay_multiplier kappa", des["tail_decay_multiplier"],
                 [fmt(des["tail_decay_multiplier"])]),
                ("floor", des["floor"], ["1.0e-6", "1e-6"])):
            check(any(r in body for r in rendered),
                  f"question.md prints the shipped desirability {label} = {value}")
        check(float(des["plateau_fraction"]) == 0.0,
              "the shipped desirability has no plateau (plateau_fraction 0), so question.md's "
              "'d_i is 1 at the target' is literally true rather than approximately true")
        check(all(w == des["weights"][0] for w in des["weights"]),
              "the shipped weights are equal, as question.md's '(d_1...d_5)^(1/5)' says")
        # The d(u) ruler table, row by row, against the shipped scorer.
        #
        # Parsed LINE BY LINE on purpose. The first version of this block used one DOTALL regex
        # (r"^\| u_i .*?\n\|[-:| ]+\|\n\| d_i \|(.*)\|\s*$") whose trailing `(.*)\|\s*$` swallowed
        # every later table in the file, so it reported "11 u vs 196 d" and killed the run before
        # sections 8-13 could speak. A markdown row is a line; parse it as one.
        sc_cfg = scorer_from_config(objective_cfg)

        def _cells(line: str) -> List[str]:
            """Data cells of a markdown row, with the leading label cell dropped."""
            parts = [c.strip() for c in line.strip().strip("|").split("|")]
            return parts[1:]

        lines = body.splitlines()
        u_row = d_row = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith("| u_i ") and i + 2 < len(lines) \
                    and re.fullmatch(r"\|[-:| ]+\|", lines[i + 1].strip()) \
                    and lines[i + 2].lstrip().startswith("| d_i "):
                u_row, d_row = lines[i], lines[i + 2]
                break
        check(u_row is not None and d_row is not None,
              "question.md prints a d_i-versus-u ruler table")
        if u_row is not None and d_row is not None:
            us: List[float | None] = []
            for cell in _cells(u_row):
                m = re.search(r"[\d.]+", cell)
                us.append(float(m.group(0)) if m else None)
            ds = _cells(d_row)
            check(len(us) == len(ds),
                  f"the ruler table is rectangular ({len(us)} u vs {len(ds)} d)")
            check(len(us) >= 6, f"the ruler table has enough rungs to pin the shape ({len(us)})")
            for u, shown in zip(us, ds):
                if u is None or not shown:
                    continue
                # d as a function of u is the same on both sides of every window, so probing the
                # first CQA at T + u*(U-T) exercises the whole ruler -- and it goes through d_of(),
                # the array path the grader itself calls, not through a re-implementation here.
                y = np.array([T[0] + u * (U[0] - T[0])] + list(T[1:]))
                real = float(sc_cfg.d_of(y[None, :])[0, 0])
                check(abs(real - sc_cfg.d_at_u(u)) <= 1e-12,
                      f"d_of() and d_at_u() agree at u = {u} ({real:.6g})")
                approxish = shown.startswith("~") or shown.startswith("≈")
                want = float(shown.lstrip("~≈").replace("−", "-"))
                tol = max(abs(real) * (0.35 if approxish else 0.006), 5e-8)
                check(abs(real - want) <= tol,
                      f"question.md's ruler row d({u}) = {shown} == the shipped scorer "
                      f"({real:.6g})")
        # No hard zero: the single sentence the recalibration exists to make true.
        check("strictly positive everywhere" in body,
              "question.md states that d_i is strictly positive everywhere (no hard zero)")
        check("is not annihilated" in body,
              "question.md states that an out-of-window CQA is penalised rather than annihilated")
        start_ev_q = evaluate_reference_start(objective_cfg)
        check(f"{float(start_ev_q.desirability):.7f}"[:9] in body.replace(" ", ""),
              f"question.md prints the baseline formulation's D = {float(start_ev_q.desirability):.7f}")
        for key in CQA_KEYS:
            d_i = float(start_ev_q.desirability_per_cqa[key])
            check(f"{d_i:.6f}" in body,
                  f"question.md prints the baseline formulation's d_{key} = {d_i:.6f}")

        leak = f"{Dref:.4f}"[:5]
        check(leak not in body, "question.md does NOT leak the reference objective value")
        # ... nor the reference's CQA vector, nor its change list, which would name the four moves.
        ref_ev = evaluate_design(to_design(R[jref]), objective_cfg)
        for key in CQA_KEYS:
            v = float(ref_ev.cqa[key])
            check(f"{v:.2f}" not in body,
                  f"question.md does NOT leak the reference's {key} = {v:.4f}")
        check(describe_row(R[jref]) not in body,
              "question.md does NOT print the reference's change list")
        # Anti-leak, done properly.
        #
        # The first version of this check asserted that no level of the reference design appeared
        # anywhere in the text. That is the wrong invariant in both directions: numeric levels such
        # as 4 or 0.4 occur all over any honest question text, and the level grids MUST print every
        # declared level, including the ones the reference happens to use -- suppressing them would
        # itself be the leak, because a curated grid tells the model which levels the designer cares
        # about (see section 8).
        #
        # What actually gives the answer away is a COMPLETE formulation. So: every complete 12-slot
        # formulation spelled out in question.md must be the baseline one, which is the only
        # formulation the model is entitled to see.
        ref_design = to_design(R[jref])
        printed = complete_formulations(text)
        check(bool(printed),
              "question.md spells out a complete 12-slot formulation (the output-format example)")
        start_norm = {s: norm_level(START[s]) for s in SLOTS}
        ref_norm = {s: norm_level(ref_design[s]) for s in SLOTS}
        for obj in printed:
            check(obj == start_norm,
                  "every complete formulation printed in question.md is the baseline one")
            check(obj != ref_norm, "question.md does NOT print the reference formulation")
        # The G1/G2 ceilings ARE disclosed on purpose in 'Why This Is Not Straightforward': telling
        # the model that one-, two- and three-condition answers top out, and that keeping the overlap
        # 0.05 above the floor tops out, is honest about the difficulty structure and names no
        # direction and no slot. What must never appear is the reference design or its value, which
        # is what the two checks above assert. Consistency of the disclosed ceilings with the
        # measured gates is asserted here so the prose cannot drift away from the enumeration.
        proc = [SLOTS.index("T9_blend_min"), SLOTS.index("T11_press_MPa")]
        nonproc = [i for i in range(len(SLOTS)) if i not in proc]
        only_process = (R[:, nonproc] == START_IDX[None, nonproc]).all(1)
        g_proc = points_for(float(np.where(FEAS & only_process, D, -1.0).max()), D0, Dref)
        disclosed = {"gate_g1_one_slot_points": g[1],
                     "gate_g1_two_slot_points": max(g[1], g[2]),
                     "gate_g1_three_slot_points": max(g.values()),
                     "gate_g2_safe_overlap_points": g2,
                     "process-set-points-only ceiling": g_proc}
        for key, pts in disclosed.items():
            if f"{pts} / 100" not in body:
                print(f"  note question.md does not quote the {key} ceiling ({pts} / 100); "
                      f"disclosure is optional, but a quoted ceiling that disagrees with the "
                      f"enumeration is not")
        quoted = {int(m) for m in re.findall(r"(?<![\d.])(\d{1,3}) / 100", body)}
        wrong = sorted(quoted - (set(disclosed.values()) | {100}))
        # The fairness counts are disclosed too ('the objective is widely improvable'), so they must
        # equal the enumeration as well. Any of the usual renderings of a five-figure number counts.
        digits = re.sub(r"[ ,  ]", "", body)
        for label, n in (("n_feasible_in_budget", int(FEAS.sum())),
                         ("n_feasible_all_cqa_in_spec", int(((D > 0) & FEAS).sum())),
                         ("gate_g3_improving_designs", g3)):
            if str(n) not in digits:
                print(f"  note question.md does not quote {label} = {n}; disclosure is optional, "
                      f"but a quoted count that disagrees with the enumeration is not")
        check(not wrong,
              f"every 'x / 100' ceiling quoted in question.md is a measured gate value ({wrong})")

    print("\n8. question.md is neutral about the twelve variables, and states no mechanism direction")
    # Five slots -- the acidifier identity and level, the meglumine level, the superdisintegrant
    # identity and the mannitol level -- are used by NO high-scoring design, because the alkaline
    # filler is strongly buffered and the textbook "acidify it" move cannot move the microenvironment
    # pH far enough to pay for what it costs elsewhere. That is deliberate chemistry, and it only
    # works as a test if the question text neither recommends nor disparages those slots. So the
    # invariant is: either the text discusses ALL twelve variables or it discusses none of them by
    # name outside the machine-generated level table and the output-format template. A text that
    # names a subset is editorialising, whichever way it leans.
    if text:
        prose = strip_html_comments(text)
        prose = re.sub(r"```.*?```", "", prose, flags=re.DOTALL)      # output-format template
        prose = re.sub(r"^\s*T\d+_\w+ \(.*$", "", prose, flags=re.MULTILINE)   # level-table lines
        named = [s for s in SLOTS if s in prose]
        check(len(named) in (0, len(SLOTS)),
              f"question.md names either all twelve variables or none in its prose "
              f"(names {len(named)}: {named})")
        # Requirement: the model must supply the excipient mechanism itself. These are the specific
        # attributions that would hand it over; the window rationales (a U-shaped pH-degradation
        # profile, static-driven segregation, the dissolution envelope) are NOT mechanism directions
        # and must stay.
        banned = ["is alkaline", "alkalinity", "buffered", "buffering", "hygroscop",
                  "over-lubricat", "overlubricat", "moisture sorber", "largest sorber",
                  "deliquesc", "raises the microenvironment", "lowers the microenvironment",
                  "retards dissolution", "coarser grade flows"]
        hits = [b for b in banned if b in prose.lower()]
        check(not hits, f"question.md states no excipient mechanism direction (hits: {hits})")

        # Equal-weight presentation, asserted rather than eyeballed. Five slots -- the acidifier
        # identity and level, the meglumine level, the superdisintegrant identity and the mannitol
        # level -- are used by no high-scoring design. A curated or truncated level grid for any slot
        # would leak that, whichever way it leaned, so every slot must have its COMPLETE declared
        # grid printed on its own row of the variable table.
        for s in SLOTS:
            row = grid_row(prose, s)
            if not row:
                check(False, f"question.md has a variable-table row for {s}")
                continue
            missing = [lv for lv in LEVELS[s] if not level_in_row(lv, row, s)]
            check(not missing,
                  f"question.md prints every declared level of {s} on its row (missing {missing})")

        # Fairness the other way round: every level code the model must choose between has to be
        # decodable into a substance, or the task tests vocabulary instead of formulation chemistry.
        # Names only -- section 8's banned list still forbids attaching properties to them.
        from formulation import SUBSTANCE_LABELS                     # noqa: E402
        codes = [c for c in list(LEVELS["T3_acid_id"]) + list(LEVELS["T6_dis_id"])
                 if c != "none"] + ["MAN", "MEG"]
        undecoded = [c for c in codes
                     if SUBSTANCE_LABELS[c].split()[0].lower() not in prose.lower()]
        check(not undecoded,
              f"question.md spells out the substance behind every level code (undecoded {undecoded})")
        grades = grid_row(prose, "T1_mcc_grade")
        check(all(gr in prose for gr in LEVELS["T1_mcc_grade"]) and bool(grades),
              "question.md names all four microcrystalline cellulose grades")

    print("\n9. the reference optimum is not sitting on a boundary (calibration rule)")
    ref_design = to_design(R[jref])
    on_endpoint = [s for s in SLOTS if ref_design[s] in (LEVELS[s][0], LEVELS[s][-1])
                   and ref_design[s] != START[s]]
    check(not on_endpoint, f"no CHANGED slot of the reference takes a grid endpoint ({on_endpoint})")
    check(float(ref_design["T8_mgst_pct"]) > float(constraints["lubricant_range"]["min"]) + 1e-9,
          "reference lubricant level is an interior point, not the anti-sticking floor")
    check(float(ref_design["T4_acid_pct"]) < float(constraints["acidifier_total"]["max"]) - 1e-9,
          "reference acidifier level is not at the palatability ceiling")
    check(float(OV[jref]) > floor,
          f"reference overlap {OV[jref]:.4f} is strictly above the floor {floor}")
    check(float(OV[jref]) < floor + 0.015,
          f"reference overlap {OV[jref]:.4f} sits tight against the floor (mass budget is binding)")

    # ---------------------------------------------------------------------------------------------
    # 10. Credentials.
    #
    # Scoped to runs* -- every run tree, not just runs/ -- because runs_old_hardzero/ ships too and a
    # key leaks the same way out of a superseded run as out of a current one. Placeholders are allowed
    # so that the '<redacted>' the framework writes, and the PUT_YOUR... in the templates, do not
    # abort the check; anything else of that shape does.
    # ---------------------------------------------------------------------------------------------
    print("\n10. no leaked credentials in run outputs")
    key_like = re.compile(r"sk-[A-Za-z0-9_.\-]{16,}")
    placeholders = ("PUT_YOUR", "<redacted>", "your_api_key", "YOUR_KEY")
    run_trees = sorted(p for p in ROOT.iterdir()
                       if p.is_dir() and (p.name == "runs" or p.name.startswith("runs_")))
    check(len(run_trees) >= 1, f"at least one run tree exists ({[p.name for p in run_trees]})")
    offenders, n_scanned = [], 0
    for tree in run_trees:
        for path in tree.rglob("*"):
            if not path.is_file():
                continue
            try:
                body = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            n_scanned += 1
            for hit in key_like.findall(body):
                if any(m in hit for m in placeholders):
                    continue
                offenders.append(f"{path.relative_to(ROOT)}: {hit[:12]}...")
    check(not offenders,
          f"no key-shaped strings under any runs*/ tree "
          f"({[p.name for p in run_trees]}, {n_scanned} files) (offenders: {offenders})")
    # The scanner must be able to see a planted key, or "clean" means only "found nothing".
    # ASSEMBLED at runtime, never written out literally: a literal key-shaped constant in this file
    # would be found by scripts/package_for_delivery.py's own scan and abort the delivery build. A
    # self-test for a leak detector must not itself look like a leak.
    probe = "sk-" + "0123456789abcdef" * 2
    check(bool(key_like.search(probe)) and not any(m in probe for m in placeholders),
          "the key-shape pattern actually matches a key-shaped string (scanner is not vacuous)")
    # runs_old_hardzero/ is archived but is NOT one of the >= 3 trajectories. That distinction has to
    # be legible to a reviewer who only reads the zip, so the README is required to exist.
    for tree in run_trees:
        if tree.name != "runs":
            check((tree / "README.md").is_file(),
                  f"{tree.name}/README.md exists, saying these are superseded runs and not this "
                  f"problem's trajectories")

    # ---------------------------------------------------------------------------------------------
    # 11. The scoring BLOCK vs the scoring CODE.
    #
    # Sections 3 and 6 compare numbers with numbers. That is not enough: the failure mode this whole
    # file exists to catch is a simplified stand-in that disagrees with the real ruler. A band table
    # can be internally tidy and still be scored by code that reads it differently -- awarding the
    # lower band at the boundary, forgetting the feasibility gate, or not clipping. So here the bands
    # are exercised through src/scoring.final_score itself, from BOTH sides of every threshold, with
    # synthetic submissions built by overriding D on a real evaluated design (so every feasibility
    # field is genuinely populated by the evaluator, not faked).
    # ---------------------------------------------------------------------------------------------
    print("\n11. src/scoring.py behaves the way the task_config scoring block says")
    scoring_cfg = dict(cfg["scoring"])
    ref_cfg_for_score = dict(cfg["reference"])
    start_ev = evaluate_reference_start(objective_cfg)
    D_start = float(start_ev.desirability)
    D_ref = float(ref_cfg_for_score["best_objective_in_budget"])

    def submission_with_D(value: float, feasible: bool = True) -> FormulationEvaluation:
        """A real evaluated design with its objective overridden to `value`."""
        base = start_ev
        if not feasible:
            # A genuinely infeasible submission: 5 changed conditions, overlap below the floor.
            infeasible_design = dict(START)
            infeasible_design.update({"T2_mcc_pct": 28.0, "T7_dis_pct": 1.0, "T8_mgst_pct": 0.3,
                                      "T10_sil_pct": 0.6, "T12_man_pct": 10.0})
            base = evaluate_design(infeasible_design, objective_cfg)
        return dataclasses.replace(base, desirability=float(value))

    def score_of(evaluations) -> dict:
        """final_score, but a refusal is reported as a failed check rather than a traceback.

        final_score deliberately raises on a config with constraint_points != 0. That is the right
        behaviour for the harness and the wrong behaviour for a checker, which has to keep going and
        print the whole list, so the refusal is turned into an unscoreable result here."""
        try:
            return final_score(start_eval=start_ev, round_evaluations=list(evaluations),
                               constraints=constraints, scoring_cfg=scoring_cfg,
                               reference_cfg=ref_cfg_for_score,
                               release_gate_cfg=cfg.get("release_gate"))
        except ValueError as exc:
            return {"total_score": f"REFUSED: {exc}", "optimality_score": None,
                    "constraint_points": None, "num_successful_designs": None,
                    "optimality_score_details": {"achievement_fraction": None,
                                                 "optimality_max_points": None}}

    def D_for_achievement(a: float) -> float:
        return D_start + a * (D_ref - D_start)

    bands_cfg = [(float(b["min_achievement"]), int(b["points"])) for b in scoring_cfg["optimality_bands"]]
    for i, (thr, pts) in enumerate(bands_cfg):
        # just inside the band -> exactly this band's points
        got = score_of([submission_with_D(D_for_achievement(thr + 1e-6))])
        check(got["total_score"] == pts,
              f"scoring.py awards {pts} at achievement {thr}+eps (got {got['total_score']})")
        # just below the band -> the NEXT lower band's points, or 0 below the last band
        below = bands_cfg[i + 1][1] if i + 1 < len(bands_cfg) else 0
        got = score_of([submission_with_D(D_for_achievement(max(thr - 1e-6, 0.0)))])
        check(got["total_score"] == below,
              f"scoring.py awards {below} at achievement {thr}-eps (got {got['total_score']})")

    got = score_of([submission_with_D(D_ref * 10.0)])
    check(got["total_score"] == int(scoring_cfg["total_points"])
          and approx(got["optimality_score_details"]["achievement_fraction"], 1.0, 1e-6),
          f"scoring.py clips achievement at 1.0 and pays the full {scoring_cfg['total_points']} "
          f"for beating the reference (got {got['total_score']})")
    check(score_of([submission_with_D(D_start)])["total_score"] == 0,
          "scoring.py awards 0 for a submission that does not improve on the baseline formulation")
    got = score_of([submission_with_D(D_ref, feasible=False)])
    check(got["total_score"] == 0 and got["num_successful_designs"] == 0,
          "scoring.py awards 0 to an INFEASIBLE submission even at the reference objective "
          "(feasibility is a gate)")
    got = score_of([submission_with_D(D_for_achievement(0.5))])
    check(got["total_score"] == got["optimality_score"] and int(got["constraint_points"]) == 0,
          "scoring.py total == optimality points and reports constraint_points 0")
    check(got["optimality_score_details"]["optimality_max_points"] == int(scoring_cfg["optimality_points"]),
          "scoring.py uses task_config.scoring.optimality_points as its maximum")
    raised = False
    try:
        final_score(start_eval=start_ev, round_evaluations=[submission_with_D(D_ref)],
                    constraints=constraints, scoring_cfg={**scoring_cfg, "constraint_points": 25},
                    reference_cfg=ref_cfg_for_score)
    except ValueError:
        raised = True
    check(raised, "scoring.py REFUSES a config with constraint_points != 0")
    # And the best-over-the-trajectory promise in iteration.note: an early good round must not be
    # lost by a later bad one.
    want_062 = next(p for a, p in bands_cfg if 0.62 >= a)
    got = score_of([submission_with_D(D_for_achievement(0.62)), submission_with_D(D_start)])
    check(got["total_score"] == want_062,
          f"scoring.py scores the BEST usable round, not the last one (iteration.note); expected "
          f"{want_062}, got {got['total_score']}")

    # ---- the RELEASE GATE (audit finding #3). These four assertions are the whole finding. --------
    # A real non-compliant design from the neighbourhood, with its objective left alone.
    j_nc = int(np.argmax(np.where(BUDGET & ~IN, D, -1.0)))
    ev_nc = evaluate_design(to_design(R[j_nc]), objective_cfg)
    check(ev_nc.all_cqa_in_spec is False and ev_nc.n_cqa_in_spec < 5,
          f"the probe design really is non-compliant (missed {ev_nc.cqa_out_of_spec})")
    check(all(check_constraints(ev_nc, constraints).values()),
          "the probe design satisfies every hard constraint, so ONLY the release gate can block it")
    got = score_of([ev_nc])
    check(got["total_score"] == 0 and got["success"] is False
          and got["num_successful_designs"] == 0,
          f"scoring.py awards 0 and success=False to a budget-feasible, D-improving submission with "
          f"one CQA out of specification (D={ev_nc.desirability:.6f}, which would have been "
          f"{points_for(float(ev_nc.desirability), D0, Dref)} points without the release gate)")
    check(got["release_gate"]["n_blocked_by_cqa_compliance_only"] == 1,
          "the report names it under release_gate.blocked_by_cqa_compliance")
    # ... and a compliant design with a LOWER D beats it over the same run.
    j_c = int(np.argmax(np.where(QUAL & (D < ev_nc.desirability), D, -1.0)))
    ev_c = evaluate_design(to_design(R[j_c]), objective_cfg)
    got = score_of([ev_nc, ev_c])
    check(got["total_score"] == points_for(float(ev_c.desirability), D0, Dref)
          and got["total_score"] > 0,
          f"in a run holding both, the score is the COMPLIANT design's "
          f"({ev_c.desirability:.6f} -> {got['total_score']}), not the higher-D non-compliant one's")
    check(got["continuous_metrics"]["best_by_desirability_was_compliant"] is False,
          "continuous_metrics still records that the highest-D submission was non-compliant")
    # The declared switch must be LOAD-BEARING, not decorative. A mutation test found that flipping
    # release_gate.all_cqa_in_spec_required to false changed nothing, because the gate was hardcoded.
    rg = cfg.get("release_gate") or {}
    check(rg.get("all_cqa_in_spec_required") is True,
          f"task_config.release_gate.all_cqa_in_spec_required is true "
          f"(got {rg.get('all_cqa_in_spec_required')!r})")
    refused = False
    try:
        final_score(start_eval=start_ev, round_evaluations=[submission_with_D(D_ref)],
                    constraints=constraints, scoring_cfg=scoring_cfg,
                    reference_cfg=ref_cfg_for_score,
                    release_gate_cfg={**rg, "all_cqa_in_spec_required": False})
    except ValueError:
        refused = True
    check(refused,
          "scoring.py REFUSES a config that declares the CQA release gate optional (the declared "
          "switch is load-bearing, not documentation)")

    # ---------------------------------------------------------------------------------------------
    # 12. question.md vs task_config.yaml on the rubric, the constraints, the round budget and the
    #     output format. Section 7 covers the windows and the leak checks; this covers everything the
    #     model needs in order to know how it will be graded and what to emit.
    # ---------------------------------------------------------------------------------------------
    print("\n12. question.md band table, constraints, rounds and output format vs task_config")
    if text:
        body = strip_html_comments(text)

        # -- band table -----------------------------------------------------------------------------
        # Withdrawn from the model-visible side in prompt_version 3 (2026-08-09), together with the
        # per-family score ceilings. This block used to assert the printed ladder matched the config
        # row for row; it now asserts the ladder is absent and that the question says so, which is the
        # same invariant pointed the other way. The ladder itself is unchanged and is still asserted
        # against src/scoring.py earlier in this script -- what moved is who can see it.
        #
        # Rationale, and it is the same one that took the per-family ceilings out: a model that can
        # read "achievement 0.29 -> 30 points, 0.36 -> 36" learns where the effort stops paying and
        # can stop at a threshold. Five rounds of coordinated reformulation are supposed to measure
        # how far it can push D, not how cheaply it can reach a printed step.
        rows = re.findall(r"^\|\s*(≥|>=|>)\s*([0-9.]+)\s*\|\s*(\d+)\s*\|", body, flags=re.MULTILINE)
        check(not rows,
              f"question.md publishes no achievement -> points band table (found {len(rows)} row(s): "
              f"{rows[:3]})")
        check("monotone graded function" in body,
              "question.md describes the score as a monotone graded function of achievement instead")
        check("not given to you" in body,
              "question.md states outright that the thresholds and points per band are withheld")
        check(len(bands_cfg) >= 2,
              f"the ladder still exists on the grader side ({len(bands_cfg)} bands)")
        check(re.search(r"no submission cleared all four gates the score is\s+\*\*0\*\*", body)
              is not None
              or "the score is **0**" in body,
              "question.md still states that a run with no qualifying formulation scores 0")
        check(f"{int(scoring_cfg['total_points'])} points" in body,
              f"question.md states the total is {scoring_cfg['total_points']} points")
        check("no points for satisfying the constraints" in body.lower()
              or "**no points for satisfying the constraints**" in body.lower(),
              "question.md states that constraint satisfaction earns no points (constraint_points 0)")
        # The achievement formula the model is told must be the one the scorer uses.
        check(re.search(r"achievement\s*=\s*\(\s*your_best_usable_D\s*-\s*D_baseline\s*\)\s*/\s*"
                        r"\(\s*D_reference\s*-\s*D_baseline\s*\)", body) is not None,
              "question.md prints the achievement formula the scorer implements")
        check("clipped at 1.0" in body, "question.md discloses that achievement is clipped at 1.0")

        # -- hard constraints ---------------------------------------------------------------------
        def line_with(*needles: str) -> str:
            for line in body.splitlines():
                if all(n in line for n in needles):
                    return line
            return ""

        check(bool(line_with("T8_mgst_pct",
                             fmt(constraints["lubricant_range"]["min"]),
                             fmt(constraints["lubricant_range"]["max"]))),
              f"question.md states the lubricant range "
              f"{fmt(constraints['lubricant_range']['min'])}-"
              f"{fmt(constraints['lubricant_range']['max'])} % on one line with T8_mgst_pct")
        check(bool(line_with("T4_acid_pct", fmt(constraints["acidifier_total"]["max"]))),
              f"question.md states the acidifier ceiling {fmt(constraints['acidifier_total']['max'])} %")
        check(bool(line_with("filler", fmt(constraints["mass_closure"]["min_filler_pct"]))),
              f"question.md states the q.s. filler floor "
              f"{fmt(constraints['mass_closure']['min_filler_pct'])} %")
        check(bool(re.search(rf"[Aa]t most\s+\*?\*?{max_ident}\*?\*?\s+excipient identity change", body)),
              f"question.md states the identity-change cap ({max_ident})")
        check(bool(re.search(rf"[Aa]t most\s+\*?\*?{max_slots}\*?\*?\s+(?:of the 12 |changed )", body)),
              f"question.md states the changed-condition cap ({max_slots})")
        n_declared = len([k for k in ("acid_consistency", "mass_closure", "composition_overlap",
                                      "max_slot_changes", "max_identity_changes", "lubricant_range",
                                      "acidifier_total") if constraints.get(k) is not None])
        numbered = re.findall(r"^\s*(\d+)\.\s+\*\*", body, flags=re.MULTILINE)
        # on-grid + validated process ranges are stated in question.md on top of the seven declared
        # ones, and check_constraints adds on_grid_design itself, so the list must be at least as
        # long as the declared set and must not be shorter.
        check(len(numbered) >= n_declared,
              f"question.md enumerates at least the {n_declared} declared hard constraints "
              f"(found {len(numbered)} numbered items)")

        # -- round budget -------------------------------------------------------------------------
        it = cfg["iteration"]
        check(f"**{int(it['max_rounds'])} rounds**" in body,
              f"question.md states the round budget ({it['max_rounds']} rounds)")
        if int(it["candidates_per_round"]) == 1:
            check("exactly **one** formulation" in body,
                  "question.md states one formulation per round (candidates_per_round 1)")
        # feedback contract: every switch that is on must be promised, every switch that is off must
        # be named as withheld.
        fb = cfg["feedback"]
        promised = {"return_measured_cqa_values": "measured CQA values",
                    "return_desirability": "desirability D",
                    "return_budget_bookkeeping": "budget bookkeeping",
                    "return_feasibility_violations": "feasibility constraints violated",
                    "return_best_so_far": "best D you have reached so far"}
        for key, phrase in promised.items():
            if bool(fb[key]):
                check(phrase in body, f"question.md promises what feedback.{key}=true returns "
                                      f"('{phrase}')")
        withheld_marker = body.lower().split("does **not** return")
        check(len(withheld_marker) > 1, "question.md has an explicit 'does not return' clause")
        if len(withheld_marker) > 1:
            clause = withheld_marker[1][:600].lower()
            for key, phrase in (("return_per_cqa_window_verdicts", "inside or outside"),
                                ("return_per_cqa_desirability", "per-cqa"),
                                ("return_reference_value", "reference target"),
                                ("return_achievement_or_score", "score"),
                                ("return_success_verdict", "verdict")):
                if not bool(fb[key]):
                    check(phrase in clause,
                          f"question.md names what feedback.{key}=false withholds ('{phrase}')")

        # -- output format ------------------------------------------------------------------------
        fences = re.findall(r"```json\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
        check(len(fences) == 1, f"question.md has exactly one json output-format template "
                                f"({len(fences)})")
        if fences:
            try:
                parsed = parse_candidate(fences[0])
                ok_parse, perr = True, ""
            except Exception as exc:                                    # noqa: BLE001
                parsed, ok_parse, perr = None, False, str(exc)
            check(ok_parse, f"the output-format template in question.md parses through "
                            f"src/parser.parse_candidate ({perr})")
            if parsed:
                check(list(parsed["formulation"].keys()) == SLOTS,
                      "the template's formulation lists all 12 slots in task_config slot order")
                ev_tpl = evaluate_design(parsed["formulation"], objective_cfg)
                check(ev_tpl.valid and approx(ev_tpl.composition_overlap, 1.0),
                      "the template's formulation is the baseline formulation and evaluates")
            for fieldname in ("name", "rationale", "formulation"):
                check(f"`candidate.{fieldname}`" in body,
                      f"question.md documents the required field candidate.{fieldname}")

    # ---------------------------------------------------------------------------------------------
    # 13. Every shipped run's score must be reproducible from what the run stored. A trajectory whose
    #     final_report cannot be re-derived from its own submissions is not evidence of anything.
    # ---------------------------------------------------------------------------------------------
    print("\n13. every run's final_report.json score is reproducible from its stored submissions")
    #
    # Each run directory carries the task_config.yaml it was graded with, and THAT is the config a
    # stored report must be reproducible from -- not the current one. Re-scoring an old trajectory
    # with a newer ruler answers a different question ("what would it have scored"), which is what
    # scripts/rescore_run.py --config is for. Grading each run against its own stored config keeps
    # this assertion biting across a recalibration instead of failing for the wrong reason, and the
    # mismatch is reported loudly so a stale trajectory can never pass as current evidence.
    run_dirs = [] if args.skip_trajectories else sorted(
        p for p in (ROOT / "runs").iterdir() if p.is_dir())
    if not args.skip_trajectories:
        check(len(run_dirs) >= 3, f"at least 3 shipped runs ({len(run_dirs)})")
    models = set()
    stale_runs = []
    for run in run_dirs:
        rel = run.relative_to(ROOT)
        report_path = run / "final_report.json"
        if not report_path.exists():
            check(False, f"{rel}/final_report.json exists")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stored = report.get("score") or {}
        rounds = submitted_formulations(run)
        check(len(rounds) == int(cfg["iteration"]["max_rounds"]),
              f"{rel} stored all {cfg['iteration']['max_rounds']} rounds ({len(rounds)})")
        run_cfg_path = run / "task_config.yaml"
        run_cfg = load_yaml(run_cfg_path) if run_cfg_path.exists() else cfg
        run_obj, run_ref = run_cfg["objective"], dict(run_cfg["reference"])
        run_cal = json.dumps(run_cfg["reference"]["calibration"], sort_keys=True, default=str)
        cur_cal = json.dumps(cfg["reference"]["calibration"], sort_keys=True, default=str)
        if run_cal != cur_cal:
            stale_runs.append(str(rel))
        api_used = run / "api_config_used.json"
        if api_used.exists():
            used = json.loads(api_used.read_text(encoding="utf-8"))
            models.add(used.get("model"))
            check(str(used.get("api_key")) == "<redacted>",
                  f"{rel}/api_config_used.json has api_key '<redacted>' "
                  f"(got {used.get('api_key')!r})")
        else:
            check(False, f"{rel}/api_config_used.json exists")
        if "desirability" not in run_obj:
            check(False, f"{rel} was graded with the superseded piecewise-cubic desirability, which "
                         f"this code no longer implements, so its stored score cannot be "
                         f"re-derived. Re-run the model against the current task_config.yaml; the "
                         f"offline estimate is scripts/rescore_run.py {rel} --out <path>")
            continue
        evs = [evaluate_design(form, run_obj) for _, form in rounds]
        run_start = evaluate_reference_start(run_obj)
        try:
            recomputed = final_score(start_eval=run_start, round_evaluations=evs,
                                     constraints=run_cfg["constraints"],
                                     scoring_cfg=run_cfg["scoring"], reference_cfg=run_ref)
        except ValueError as exc:
            recomputed = {"total_score": f"REFUSED: {exc}"}
        if not isinstance(recomputed["total_score"], int):
            check(False, f"{rel} is re-scoreable at all (scorer said {recomputed['total_score']})")
            continue
        check(int(stored.get("total_score", -1)) == int(recomputed["total_score"]),
              f"{rel} total_score recomputes ({stored.get('total_score')} vs "
              f"{recomputed['total_score']})")
        check(approx((stored.get("optimality_score_details") or {}).get("achievement_fraction"),
                     recomputed["optimality_score_details"]["achievement_fraction"], 1e-6),
              f"{rel} achievement_fraction recomputes")
        check(int(stored.get("num_successful_designs", -1)) == int(recomputed["num_successful_designs"]),
              f"{rel} num_successful_designs recomputes "
              f"({stored.get('num_successful_designs')} vs {recomputed['num_successful_designs']})")
        check(int(stored.get("constraint_points", -1)) == 0,
              f"{rel} final_report constraint_points == 0")
    # Guarded by --skip-trajectories for the same reason as section 13: between a prompt
    # revision and the re-run that follows it there are legitimately no current trajectories,
    # and an aggregate over an empty set is not a finding.
    if not args.skip_trajectories:
        check(len(models) >= 3,
              f"the shipped runs cover >= 3 distinct model(s) among the run dirs ({sorted(models)})")
    if stale_runs:
        print(f"  NOTE {len(stale_runs)} run(s) were graded under a DIFFERENT calibration than the "
              f"current task_config.yaml and were re-scored against their own stored config: "
              f"{stale_runs}. They are not evidence for the current calibration until the models "
              f"are re-run; use scripts/rescore_run.py --config for the offline estimate.")

    # ---------------------------------------------------------------------------------------------
    # 14. 审核说明.md -- the reviewer-facing document -- against a fresh recomputation.
    #
    # This section exists because of a real failure. The recalibration changed two chemistry claims in
    # section 2.3 of that document ("touching the acidifier caps you at 33/100" became 47, and
    # "introducing mannitol caps you at 63/100" became 100, i.e. mannitol stopped being a trap), and
    # nothing in this file noticed. A prose claim about a measured quantity has to be pinned to the
    # measurement or it will go stale at the next recalibration, exactly like a cached calibration JSON.
    # So: recompute the family ceilings here, assert answers/trap_family_probes.json agrees, and assert
    # the document quotes those same numbers.
    # ---------------------------------------------------------------------------------------------
    print("\n14. 审核说明.md quotes measured numbers, and those numbers == a fresh recomputation")
    check(REVIEW.exists(), f"{REVIEW.name} exists")
    review = REVIEW.read_text(encoding="utf-8") if REVIEW.exists() else ""

    def touches(slots) -> np.ndarray:
        idx = [SLOTS.index(s) for s in slots]
        return (R[:, idx] != START_IDX[idx]).any(axis=1)

    def ceiling_of(mask: np.ndarray) -> tuple[int, int]:
        m = mask & FEAS
        n = int(m.sum())
        if n == 0:
            return 0, 0
        return n, int(points_for(float(D[m].max()), D0, Dref))

    TRAP_SLOTS = ("T3_acid_id", "T4_acid_pct", "T5_meg_pct", "T6_dis_id", "T12_man_pct")
    fam_recomputed = {
        "touches the acidifier (T3/T4)": ceiling_of(touches(("T3_acid_id", "T4_acid_pct"))),
        "introduces mannitol (T12)": ceiling_of(touches(("T12_man_pct",))),
        "touches meglumine (T5)": ceiling_of(touches(("T5_meg_pct",))),
        "changes the disintegrant identity (T6)": ceiling_of(touches(("T6_dis_id",))),
        "no trap slot touched": ceiling_of(~touches(TRAP_SLOTS)),
    }
    TRAPS = ROOT / "answers" / "trap_family_probes.json"
    if not TRAPS.exists():
        check(False, f"answers/{TRAPS.name} exists (run scripts/measure_trap_families.py)")
    else:
        stored_fam = json.loads(TRAPS.read_text(encoding="utf-8"))["families"]
        for name, (n, pts) in fam_recomputed.items():
            rec = stored_fam.get(name) or {}
            check(int(rec.get("n", -1)) == n and int(rec.get("points", -1)) == pts,
                  f"trap_family_probes '{name}' == recomputed (n {rec.get('n')} vs {n}, "
                  f"points {rec.get('points')} vs {pts})")
    # Every trap-family row of the document must carry BOTH the family size and the ceiling, on one
    # line, so a stale ceiling cannot hide next to a fresh count.
    for name, (n, pts) in fam_recomputed.items():
        rows = [ln for ln in review.splitlines() if f"{n:,}" in ln]
        check(any(f"{pts} / 100" in ln for ln in rows),
              f"审核说明 quotes the '{name}' family as {n:,} designs -> {pts} / 100")
    check("甘露醇" in review and "陷阱" in review,
          "审核说明 still discusses the mannitol family (the claim was rewritten, not deleted "
          "silently)")
    if fam_recomputed["introduces mannitol (T12)"][1] >= 50:
        check("不成立" in review or "不再是陷阱" in review,
              f"the mannitol ceiling is {fam_recomputed['introduces mannitol (T12)'][1]} / 100, so "
              f"审核说明 must say out loud that the 'mannitol is a trap' claim no longer holds")

    # The gate values, the counts and the three measured model scores, as quoted.
    for label, value in (("G1 one-slot", g[1]), ("G1 two-slot", max(g[1], g[2])),
                         ("G1 three-slot", max(g.values())), ("G2", g2)):
        check(f"{value} / 100" in review or f"**{value}**" in review,
              f"审核说明 quotes the measured {label} ceiling ({value} / 100)")
    for label, value in (("n_feasible_in_budget", int(BUDGET.sum())),
                         ("all-CQA-in-spec", int(ref_cfg["n_feasible_all_cqa_in_spec"])),
                         ("qualifying (scoring-eligible)", int(QUAL.sum())),
                         ("non-compliant but improving",
                          int(ref_cfg["n_noncompliant_but_improving"])),
                         ("G3 improving", g3),
                         (">= 21 points", n21), (">= 30 points", n30), (">= 44 points", n44)):
        check(f"{value:,}" in review, f"审核说明 quotes {label} = {value:,}")
    check(f"{D0:.7f}" in review.replace(" ", ""),
          f"审核说明 quotes the baseline formulation's D = {D0:.7f}")
    check(f"{Dref:.7f}" in review.replace(" ", ""),
          f"审核说明 quotes the reference D = {Dref:.7f}")

    if NAIVE.exists():
        nv = json.loads(NAIVE.read_text(encoding="utf-8"))
        naive_max = max(int(p["points"]) for p in nv["naive_probes"].values())
        hand = int(nv["hand_policies"]["best_points"])
        n_pol = int(nv["hand_policies"]["n_policies"])
        check(naive_max < 50, f"the naive-answer maximum is below 50 ({naive_max})")
        check(int(ref_cfg["gate_naive_answer_points"]) == naive_max,
              f"task_config.gate_naive_answer_points == the measured naive ceiling "
              f"({ref_cfg['gate_naive_answer_points']} vs {naive_max})")
        check(int(ref_cfg["gate_hand_policy_points"]) == hand,
              f"task_config.gate_hand_policy_points == the measured hand-policy ceiling "
              f"({ref_cfg['gate_hand_policy_points']} vs {hand})")
        check(int(ref_cfg["n_hand_policies"]) == n_pol,
              f"task_config.n_hand_policies == the shipped generator's family size "
              f"({ref_cfg['n_hand_policies']} vs {n_pol})")
        check(f"{naive_max} / 100" in review or f"**{naive_max}**" in review,
              f"审核说明 quotes the naive-answer ceiling ({naive_max} / 100)")
        check(f"{hand} / 100" in review, f"审核说明 quotes the hand-policy ceiling ({hand} / 100)")
        check(str(n_pol) in review,
              f"审核说明 quotes the size of the mechanical policy family ({n_pol})")
        # The audit that forced this widening found 20 where the shipped generator reported 9. Assert
        # the generator is still the wider one, so the documented ceiling cannot silently narrow.
        check(n_pol >= 400,
              f"the shipped policy generator still spans the widened family ({n_pol} policies; a "
              f"budget-4-only generator produces 144 and understates the ceiling)")
    else:
        check(False, "answers/naive_answer_probes.json exists (run scripts/measure_naive_answers.py)")

    # The measured model scores, and the fact that they are the CURRENT runs.
    #
    # There are REPLICATES: sampling temperature is 0.7, so one run per model does not establish a
    # ranking. Every replicate must therefore be quoted, not just the last one per model -- the 1.x
    # version of this block keyed a dict by model name and silently kept whichever run sorted last,
    # which would have let a replicate go undocumented.
    per_model: dict = {}
    for run in sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir()):
        rep = run / "final_report.json"
        api = run / "api_config_used.json"
        if rep.exists() and api.exists():
            model = json.loads(api.read_text(encoding="utf-8")).get("model")
            pts = int((json.loads(rep.read_text(encoding="utf-8")).get("score") or {})
                      .get("total_score", -1))
            per_model.setdefault(model, []).append((run.name, pts))
    # Guarded by --skip-trajectories for the same reason as section 13: between a prompt
    # revision and the re-run that follows it there are legitimately no current trajectories,
    # and an aggregate over an empty set is not a finding.
    if not args.skip_trajectories:
        check(len(per_model) >= 3,
              f"the shipped runs cover >= 3 distinct models ({sorted(per_model)})")
    for model, runs_of in sorted(per_model.items()):
        for run_name, sc_pts in runs_of:
            rows = [ln for ln in review.splitlines() if model in ln]
            check(any(f"{sc_pts} / 100" in ln or f"**{sc_pts}**" in ln for ln in rows),
                  f"审核说明 quotes {model} replicate {run_name[-15:]} scoring {sc_pts} / 100")
        check(len(runs_of) >= 1, f"{model} has at least one shipped replicate")
    ds_scores = [pts for _, pts in per_model.get("deepseek-v4-pro", [])]
    if not args.skip_trajectories:
        check(bool(ds_scores) and max(ds_scores) < 50,
              f"the hard acceptance criterion holds on EVERY deepseek-v4-pro replicate: "
              f"{ds_scores}, max {max(ds_scores) if ds_scores else None} < 50")
    # And the spread must be documented rather than papered over: with replicates present, the
    # document has to say out loud that a single run does not rank the models.
    if any(len(v) > 1 for v in per_model.values()):
        check(("重复" in review or "replicate" in review.lower()) and "方差" in review,
              "审核说明 discusses replicate spread / variance, since replicates are shipped")

    # Superseded numbers may be quoted -- the recalibration argument is a BEFORE/AFTER argument and
    # needs them -- but never bare. Every occurrence must either sit in the rework section or be
    # labelled as the first calibration within its own line or the few lines above it (a table's
    # header column counts). A bare 0.017478 in a live paragraph is precisely the defect that made
    # this rework necessary.
    review_lines = review.splitlines()
    mark_line = next((i for i, ln in enumerate(review_lines) if ln.startswith("## 十、")), -1)
    check(mark_line > 0, "审核说明 has the rework section (## 十、) that records the recalibration")
    STALE_MARKERS = ("第一版", "硬零", "旧", "before", "上一版")
    for stale in ("0.017478", "1,917", "3,849", "0.248108"):
        bare = []
        for i, ln in enumerate(review_lines):
            if stale not in ln:
                continue
            if mark_line >= 0 and i > mark_line:
                continue
            context = "\n".join(review_lines[max(0, i - 6):i + 1])
            if not any(m in context for m in STALE_MARKERS):
                bare.append(i + 1)
        check(not bare,
              f"the superseded value {stale} is never quoted unlabelled outside the rework section "
              f"(unlabelled at lines {bare})")

    # The delivery claim about readable trajectories, checked rather than asserted in prose.
    for run in sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir()):
        for fname in ("trajectory.readable.txt", "trajectory.readable.md"):
            check((run / fname).is_file(),
                  f"{run.relative_to(ROOT)}/{fname} exists (human-readable trajectory)")


    # ---------------------------------------------------------------------------------------------
    # 15. The four remaining audit findings, checked rather than asserted in prose.
    #
    #   #1 data provenance    every model-visible and reviewer-visible file says "synthetic", and the
    #                         retracted claims appear nowhere;
    #   #2 safe format        no executable deserialisation, and the npz is the ruler;
    #   #4 version freeze     one ruler version, named identically everywhere, including in the runs;
    #   #5 tool layer         the declared tools ARE the implemented tools, they return none of the
    #                         forbidden quantities, and the shipped runs actually exercise them.
    #
    # Framework rule: EVERY file that prints a score, a band table or a scoring rule is checked --
    # question.md, README.md and 审核说明.md. A previous problem in this series passed 20/20 while its
    # README printed a superseded rubric, because only question.md was checked.
    # ---------------------------------------------------------------------------------------------
    print("\n15. audit findings #1 / #2 / #4 / #5")
    readme_p, card_p, chlog_p = ROOT / "README.md", ROOT / "data" / "DATA_CARD.md", ROOT / "CHANGELOG.md"
    readme = readme_p.read_text(encoding="utf-8") if readme_p.exists() else ""
    card = card_p.read_text(encoding="utf-8") if card_p.exists() else ""
    chlog = chlog_p.read_text(encoding="utf-8") if chlog_p.exists() else ""

    # ---- #1 data provenance ----
    check(card_p.exists(), "data/DATA_CARD.md exists")
    for label, body in (("question.md", text or ""), ("README.md", readme),
                        ("审核说明.md", review), ("DATA_CARD.md", card)):
        low = body.lower()
        check("synthetic" in low or "合成" in body or "模拟" in body,
              f"{label} states the data is synthetic / simulated")
    RETRACTED = ["validated multivariate prediction model",
                 "development data set for this product",
                 "registered product development data",
                 "fitted to fixed-seed in-house data",
                 "in-house data"]
    for phrase in RETRACTED:
        hits = [lab for lab, body in (("question.md", text or ""), ("README.md", readme),
                                      ("task_config.yaml", TASK_CFG.read_text(encoding="utf-8")),
                                      ("审核说明.md", review))
                if phrase.lower() in body.lower()]
        check(not hits, f"the retracted provenance claim {phrase!r} appears nowhere (found in {hits})")
    prov = cfg.get("data_provenance") or {}
    check(prov.get("nature") in ("synthetic / simulated", "synthetic", "simulated"),
          f"task_config.data_provenance.nature says synthetic (got {prov.get('nature')!r})")
    check(prov.get("is_real_product_data") is False and prov.get("is_real_measurement_data") is False,
          "task_config.data_provenance denies real product and real measurement data")
    check(int(prov.get("seed", -1)) == 20260727,
          f"task_config.data_provenance.seed == the generator seed (got {prov.get('seed')})")
    check("noise_model" in prov and "0.01" in str(prov["noise_model"]),
          "task_config.data_provenance states the noise model, including its magnitude")

    # ---- #2 safe format ----
    check(not (ROOT / "data" / "oracle.pkl").exists(), "data/oracle.pkl is absent from the package")
    check((ROOT / "data" / "oracle.npz").is_file() and (ROOT / "data" / "oracle_schema.json").is_file()
          and (ROOT / "data" / "oracle_migration_check.json").is_file(),
          "data/oracle.npz + oracle_schema.json + oracle_migration_check.json all present")
    mig = json.loads((ROOT / "data" / "oracle_migration_check.json").read_text(encoding="utf-8"))
    check(mig["max_abs_diff_overall"] == 0.0 and mig["max_abs_diff_scalar_path"] == 0.0,
          f"the recorded pickle->npz difference is exactly 0 on both paths over "
          f"{mig['n_designs_compared']:,} designs")
    import hashlib as _hl
    sch = json.loads((ROOT / "data" / "oracle_schema.json").read_text(encoding="utf-8"))
    check(_hl.sha256((ROOT / "data" / "oracle.npz").read_bytes()).hexdigest() == sch["sha256_npz"],
          "sha256(data/oracle.npz) == data/oracle_schema.json.sha256_npz")
    # ---- #2b the response-surface BOUNDARY (second-round audit of the public/grader split) ----
    # src/response_model.py is the ONLY module allowed to hold the property table, the CQA simulator
    # and the surrogate weights, because the public archive replaces it with a raising stub. Two
    # things have to stay true or the boundary is decorative:
    #   * the dependency runs ONE WAY: formulation.py (public) must not import response_model;
    #   * nothing reaches around the boundary to touch the weights file or the mechanism directly.
    bmod = ROOT / "src" / "response_model.py"
    check(bmod.is_file(), "src/response_model.py exists (the response-surface boundary module)")
    btext = bmod.read_text(encoding="utf-8") if bmod.is_file() else ""
    for name in ("SUB", "MCC_GRADE", "cqa_true", "class Oracle", "feature_vector", "train_oracle"):
        check(name in btext, f"the boundary module holds {name}")
    ftext = (ROOT / "src" / "formulation.py").read_text(encoding="utf-8")
    IMPORT_RX = r"^\s*(?:from\s+response_model\s+import|import\s+response_model)"
    check(re.search(IMPORT_RX, ftext, flags=re.MULTILINE) is None,
          "src/formulation.py does NOT import response_model (the dependency is one-way, so the "
          "public half can ship formulation.py unchanged; its docstring may still point at it)")
    for name in ("SUB:", "MCC_GRADE:", "def cqa_true", "class Oracle", "def feature_vector",
                 "def get_oracle", "def train_oracle"):
        check(name not in ftext,
              f"src/formulation.py no longer defines {name.strip(':')} (it moved behind the boundary)")
    # Who is allowed to import the mechanism at all. Everything else must go through evaluator /
    # design_space, which are the two public modules that do it.
    # IMPORTING the boundary is what matters, not mentioning it in a docstring.
    ALLOWED_MECHANISM_IMPORTERS = {
        "src/design_space.py", "src/evaluator.py",                      # public, via the boundary
        "scripts/enumerate_reference_best.py", "scripts/check_consistency.py",
        "scripts/audit_recalibration.py", "scripts/verify_oracle_npz.py",
        "scripts/migrate_oracle_to_npz.py",
        "scripts/package_for_delivery.py",        # generates the public stub; imports nothing
    }
    reachers = []
    for pyf in sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))):
        rel = str(pyf.relative_to(ROOT))
        if rel == "src/response_model.py":
            continue
        body = pyf.read_text(encoding="utf-8")
        if (re.search(IMPORT_RX, body, flags=re.MULTILINE)
                or re.search(r"""np\.load\(|["']data/oracle\.npz["']""", body)):
            if rel not in ALLOWED_MECHANISM_IMPORTERS:
                reachers.append(rel)
    check(not reachers,
          f"only the declared modules import the response surface; nothing else does ({reachers})")
    # And the harness itself must not: main.py / tools.py / feedback.py / parser.py / scoring.py are
    # what a public run executes, and none of them may reach the surface.
    for rel in ("src/main.py", "src/tools.py", "src/feedback.py", "src/parser.py", "src/scoring.py"):
        body = (ROOT / rel).read_text(encoding="utf-8")
        check(re.search(IMPORT_RX, body, flags=re.MULTILINE) is None and "oracle.npz" not in body,
              f"{rel} does not reach the response surface directly")

    bad_deser = []
    for pyf in sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))):
        if pyf.name in ("migrate_oracle_to_npz.py", "verify_oracle_npz.py"):
            continue
        body = pyf.read_text(encoding="utf-8")
        for rx in (r"\bimport\s+pickle\b", r"\bpickle\.loads?\s*\(", r"\bjoblib\.load\s*\(",
                   r"\btorch\.load\s*\(", r"allow_pickle\s*=\s*True"):
            if re.search(rx, body):
                bad_deser.append(f"{pyf.relative_to(ROOT)}:{rx}")
    check(not bad_deser, f"no executable deserialisation in src/ or scripts/ ({bad_deser})")

    # ---- #4 version freeze ----
    ver = str(cfg.get("ruler_version", ""))
    check(bool(ver), f"task_config declares a ruler_version ({ver!r})")
    check(str(cfg["scoring"].get("ruler_version", "")) == ver,
          "task_config.scoring.ruler_version == task_config.ruler_version")
    check(f"## {ver}" in chlog, f"CHANGELOG.md has a section for ruler {ver}")
    for label, body in (("README.md", readme), ("审核说明.md", review)):
        check(ver in body, f"{label} names the frozen ruler version {ver}")
    # prompt_version is the OTHER version, and it is the one that decides whether a TRAJECTORY is
    # still evidence. Renaming the scenario noun on 2026-07-30 changed a model-visible feedback key
    # and several headings without touching the scoring rule, so ruler_version stayed at 2.0.0 while
    # prompt_version went 1 -> 2 and every shipped run was re-run rather than re-scored.
    pver = cfg.get("prompt_version")
    check(isinstance(pver, int) and pver >= 1,
          f"task_config declares an integer prompt_version ({pver!r})")
    check(int(cfg["scoring"].get("prompt_version", -1)) == int(pver),
          "task_config.scoring.prompt_version == task_config.prompt_version")
    check("prompt_version" in chlog,
          "CHANGELOG.md explains prompt_version (what obliges a re-run vs a re-score)")
    for label, body in (("README.md", readme), ("审核说明.md", review)):
        check("prompt_version" in body,
              f"{label} distinguishes prompt_version from ruler_version")
    check(not (ROOT / "runs_old_hardzero").exists(),
          "the superseded runs_old_hardzero/ tree is gone (one scoring version's results ship)")
    check(not list((ROOT / "answers").glob("rescore_new_calibration_*.json")),
          "no superseded rescore artefacts are left in answers/")
    # every file that prints the band table must print THIS band table
    for label, body in (("README.md", readme), ("审核说明.md", review)):
        for a, pts in POINT_BANDS[:1] + POINT_BANDS[-1:]:
            pass
    for label, body in (("README.md", readme), ("审核说明.md", review)):
        stale_bands = [s for s in ("0.75 | 80", "0.60 | 63", "0.45 | 47", "0.30 | 33", "0.15 | 20")
                       if s in body]
        check(not stale_bands, f"{label} does not print a superseded band row ({stale_bands})")

    # ---- #5 tool layer ----
    tcfg = cfg.get("tools") or {}
    check(bool(tcfg.get("enabled")), "task_config.tools.enabled is true")
    declared = [d["name"] for d in tcfg.get("available", [])]
    check(sorted(declared) == sorted(TOOL_NAMES),
          f"task_config.tools.available == src/tools.TOOL_NAMES ({declared} vs {list(TOOL_NAMES)})")
    check(sorted(d["tool"] for d in tool_catalogue()) == sorted(TOOL_NAMES),
          "src/tools.tool_catalogue() covers exactly the implemented tools")
    check(len(declared) >= 2, f"at least two tools are offered ({len(declared)})")
    for name in TOOL_NAMES:
        check(name in (text or ""), f"question.md names the {name} tool")
    check(int(tcfg["call_budget_per_round"]) == DEFAULT_TOOL_BUDGET_PER_ROUND
          and int(tcfg["max_tool_turns_per_round"]) == DEFAULT_MAX_TOOL_TURNS_PER_ROUND,
          "the declared tool budget == the src/tools defaults the harness uses")
    # A tool must not return the objective, any CQA, or a compliance verdict. Probed, not trusted.
    FORBIDDEN = ("desirability", "cqa", "achievement", "score", "in_spec", "microenvironment_ph",
                 "q30", "reference")
    probes = [
        {"tool": "validate_formulation", "arguments": {"formulation": dict(START)}},
        {"tool": "composition_overlap", "arguments": {"formulation": dict(START)}},
        {"tool": "excipient_note", "arguments": {"code": "MGST"}},
        {"tool": "excipient_note", "arguments": {"code": "CCS"}},
    ]
    leaked = []
    for call in probes:
        res = run_tool_call(call, constraints)
        for key in res:
            if any(f in key.lower() for f in FORBIDDEN):
                leaked.append(f"{call['tool']}.{key}")
    check(not leaked, f"no tool returns a key naming the objective, a CQA or a verdict ({leaked})")
    # ... and every advertised failure mode is really reachable.
    fails_expected = {
        "unknown_tool": {"tool": "no_such_tool"},
        "bad_arguments": {"tool": "validate_formulation", "arguments": {}},
        "missing_slots": {"tool": "validate_formulation",
                          "arguments": {"formulation": {"T1_mcc_grade": "PH102"}}},
        "off_grid": {"tool": "validate_formulation",
                     "arguments": {"formulation": {**START, "T2_mcc_pct": 33.0}}},
        "unknown_excipient": {"tool": "excipient_note", "arguments": {"code": "NOPE"}},
    }
    for kind, call in fails_expected.items():
        got = run_tool_call(call, constraints)
        check(got.get("error") == kind,
              f"tool failure mode {kind!r} is reachable (got {got.get('error')!r})")
    quota = run_tool_calls([{"tool": "excipient_note", "arguments": {"code": "CCS"}}] * 3,
                          constraints, 2)
    check(quota["n_executed"] == 2
          and quota["results"][-1].get("error") == "quota_exceeded",
          "tool failure mode 'quota_exceeded' is reachable and stops execution at the budget")
    for kind in ("unknown_tool", "bad_arguments", "off_grid", "missing_slots",
                 "unknown_excipient", "quota_exceeded"):
        check(any(kind in str(x) for x in tcfg.get("failure_modes_returned", [])),
              f"task_config.tools.failure_modes_returned documents {kind}")
    # the shipped runs must actually exercise the layer, or finding #5 is not evidenced
    tool_calls_total, runs_with_tools, versions, pversions = 0, 0, set(), set()
    for run in sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir()):
        rep = run / "final_report.json"
        if not rep.exists():
            continue
        rj = json.loads(rep.read_text(encoding="utf-8"))
        versions.add(str((rj.get("score") or {}).get("ruler_version")))
        pversions.add(rj.get("prompt_version"))
        tl = ((rj.get("checks") or {}).get("tool_layer") or {})
        n = int(tl.get("n_tool_calls", 0))
        tool_calls_total += n
        runs_with_tools += 1 if n else 0
        check(isinstance((rj.get("score") or {}).get("continuous_metrics"), dict),
              f"{run.name}/final_report.json carries score.continuous_metrics (finding #4)")
        check(isinstance((rj.get("score") or {}).get("release_gate"), dict),
              f"{run.name}/final_report.json carries score.release_gate (finding #3)")
    if not args.skip_trajectories:
        check(versions == {ver},
              f"every shipped run records ruler_version {ver} ({sorted(versions)})")
        check(pversions == {int(pver)},
              f"every shipped run records prompt_version {pver} -- a trajectory produced under "
              f"older prompt text is not evidence about the current task ({sorted(pversions)})")
        check(runs_with_tools >= 1 and tool_calls_total >= 1,
              f"the shipped trajectories actually exercise the tool layer "
              f"({tool_calls_total} tool calls across {runs_with_tools} run(s))")

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all consistency checks passed")


def comp_order():
    from formulation import SUBSTANCES
    return SUBSTANCES


def strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def norm_level(v) -> str:
    """Compare a level the way the evaluator does: 40 == 40.0 == '40.0', 'CIT' == 'cit'."""
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v).strip().lower()


def complete_formulations(text: str) -> list[dict]:
    """Every complete 12-slot formulation object spelled out in the text.

    Innermost-brace match, so the `formulation` object inside the output-format example is found
    without tripping over the enclosing `candidate` object.
    """
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        blob = m.group(0)
        got = {}
        for s in SLOTS:
            mm = re.search(rf'"{s}"\s*:\s*"?([A-Za-z0-9_.+\-]+)"?', blob)
            if mm:
                got[s] = norm_level(mm.group(1))
        if len(got) == len(SLOTS):
            out.append(got)
    return out


def grid_row(prose: str, slot: str) -> str:
    """The first markdown table row that declares `slot` -- i.e. its level grid."""
    for line in prose.splitlines():
        if slot in line and line.count("|") >= 3:
            return line
    return ""


def level_in_row(level, row: str, slot: str = "") -> bool:
    """Is `level` one of the values printed on this row? Numbers are compared numerically, so a row
    that prints 1.0 satisfies the level 1, and the cell holding the slot name is excluded so that
    the '4' in 'T4_acid_pct' cannot stand in for a missing level."""
    cells = "|".join(c for c in row.split("|") if not slot or slot not in c)
    if isinstance(level, str):
        return level.lower() in cells.lower()
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", cells)]
    return any(abs(n - float(level)) <= 1e-9 for n in nums)


def fmt(v) -> str:
    """Render a number the way question.md would: 78.9 not 78.900000000000006."""
    f = float(v)
    s = f"{f:g}"
    return s


if __name__ == "__main__":
    main()

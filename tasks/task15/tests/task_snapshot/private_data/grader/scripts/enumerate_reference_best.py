"""Designer-only: enumerate the local chemical space, derive the reference target, and prove the
difficulty mechanism with numbers.

The optimality block is scored against a reference QED. This script derives that value by
systematic enumeration so the value stored in task_config.yaml is reproducible rather than
hand-picked, and it prints the three gate measurements that justify the instance:

  G1  thick tinker ceiling     -- the best score reachable by ALL one-slot and ALL two-slot edits.
  G2  safe-similarity ceiling  -- the best score reachable while keeping similarity >= floor + 0.05.
  G3  fairness                 -- how many distinct feasible improving designs exist, and how the
                                  score bands are populated across them.
  G4  checklist-only ceiling   -- what a policy that only ensures FEASIBILITY is worth. Added in the
                                  2026-07 revision, when the tool layer gave the model a boolean
                                  constraint checklist: feasibility became cheap to verify, so the
                                  honest question is whether feasibility alone now buys a pass. It
                                  does not, because the objective is still not computable.

It also reports where the ORDINARY medicinal-chemistry moves land. Those moves are listed as a
chemist would list them AFTER reading the constraints, which matters: on an earlier problem in this
series the equivalent check only tested the moves the constraints had already been written to block,
so it passed while the task was in fact easy.

Framing: this is the best QED found by *this* enumeration, not a proof of global optimality.
Molecular space is unbounded and a model can legitimately do better -- that happened earlier in this
series, which is why every slot below carries the alternatives a chemist would really consider.
Achievement is clipped at 1.0, so meeting or exceeding the target earns full marks.

Run:
    python scripts/enumerate_reference_best.py
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "public" / "src"))
sys.path.insert(0, str(ROOT / "grader" / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import evaluate_smiles, check_constraints, satisfies_all_constraints  # noqa: E402
from utils import load_yaml  # noqa: E402

# Constraints and the starting molecule come from the PUBLIC task config; the score bands come from
# the grader-only reference file. Splitting the inputs like this is what keeps the enumeration honest
# after the public/grader split: if someone edits a window on the public side, the gates recomputed
# here move with it.
TASK_CFG = ROOT / "public" / "tasks" / "q1_lipophilic" / "task_config.yaml"
REFERENCE_CFG = ROOT / "grader" / "reference" / "reference_target.yaml"
OUT = ROOT / "grader" / "answers" / "reference_enumeration.json"

# ---------------------------------------------------------------------------------------------
# The chemotype, as a product of five ORDERED slots:
#
#     aryl(-SUB)(-CAP) - S(=O)(=O) - N(NSUB) - LINK - TAIL
#
# Ordered slots give a principled definition of an "edit": changing one slot is one edit. That is
# what makes the tinker ceiling in G1 a measurement rather than a judgement call.
# ---------------------------------------------------------------------------------------------
# Donor-bearing options (hydroxyl, hydroxymethyl, amine, acetamido, primary amide/sulfonamide) are
# present in every slot. This is not decoration: QED's eight desirability components include a
# hydrogen-bond-donor term, so a library with no donors cannot estimate the reachable optimum. An
# earlier version of this file omitted them, reported a one-edit ceiling of 9, and the shipped task
# then scored 63 because the model reached for a phenol in round one. See 审核说明.md section 4.
TAIL = [
    "N(C)C", "N(C)CC", "N(CC)CC", "N(C)C(C)C",
    "N1CCCC1", "N1CCCCC1", "N1CCCCCC1", "N1CCOCC1", "N1CCN(C)CC1",
    "N1CCCC1C", "N1CCCCC1C", "N1CCC(C)CC1", "N1CCC(F)(F)CC1", "N1CCOC(C)C1",
    "N1CCC(O)CC1", "N1CCCCC1(C)C", "N1CCC(OC)CC1", "N1CCCC(C)C1",
    "C1CCCCC1", "C1CCCC1", "C1CC1", "C(C)(C)C",
    "N1CCC(CO)CC1", "N1CCNCC1",
]
LINK = ["", "C", "CC", "CCC"]
NSUB = ["", "C", "CC"]
SUB = ["", "F", "Cl", "C", "OC", "C#N", "O", "CO", "NC", "NC(C)=O", "C(N)=O"]
CAP = ["", "F", "Cl", "Br", "C", "CC", "C(C)C", "C(C)(C)C", "OC", "C#N", "SC",
       "C(F)(F)F", "OC(F)(F)F", "C1CC1", "O", "CO", "NC", "NC(C)=O", "C(N)=O"]

SLOTS = [TAIL, LINK, NSUB, SUB, CAP]
SLOT_NAMES = ["TAIL", "LINK", "NSUB", "SUB", "CAP"]

# VOCABULARY VERSION. Bump it whenever any slot list above, the chemotype template in smiles_for(),
# or ORDINARY_MOVES below changes -- the reference target, the four gates and the published
# fourteen-move table are all derived from this vocabulary, so a silent edit here invalidates every
# one of them. The version and the coverage boundary are mirrored into
# grader/reference/reference_target.yaml (`enumeration_vocabulary`) and asserted equal by
# check_consistency.py, and the boundary is written out in prose in 审核说明.md section 6.
#
# WHAT THE VOCABULARY COVERS
#   - the five substitution points of the shipped chemotype, independently and in every combination:
#     the amine tail (24 options: acyclic amines, 5/6/7-membered azacycles, morpholine/piperazine,
#     methyl/hydroxy/methoxy/gem-difluoro/hydroxymethyl-decorated rings, and four carbocyclic tails),
#     the linker length (direct bond / C1 / C2 / C3), the sulfonamide N-substituent (H / Me / Et),
#     and the two aryl substituents (11 and 19 options: H, halogens, small alkyl, CF3, OMe, OCF3,
#     SMe, CN, OH, CH2OH, NHMe, acetamido, primary amide, cyclopropyl);
#   - hydrogen-bond DONORS in every slot (OH, CH2OH, NH, NHMe, acetamido, primary amide). This is not
#     decoration: QED's eight desirability components include a donor term, so a library with no
#     donors cannot estimate the reachable optimum. An earlier version of this file omitted them,
#     reported a one-edit ceiling of 9, and the shipped task then scored 63 because the model reached
#     for a phenol in round one;
#   - 15 hand-listed moves OUTSIDE the chemotype (ORDINARY_MOVES), which is where the published
#     fourteen-move table comes from.
#
# WHAT IT DOES NOT COVER (so the reference is a target found by THIS enumeration, not a global
# optimum -- achievement is clipped at 1.0 precisely so that a model may legitimately beat it):
#   - any change to the benzene core itself: heteroaryl replacement, ring fusion, ring contraction;
#   - any change to the sulfonyl group itself beyond the hand-listed sulfone move;
#   - a third aryl substituent, or a different substitution PATTERN -- the two ring positions are
#     fixed by the template in smiles_for();
#   - stereochemistry, E/Z isomers, tautomers, macrocycles, spiro systems;
#   - substituents outside the listed lexicon (e.g. larger heteroaryl caps, sulfonyl/phosphonyl
#     substituents, longer or branched linkers beyond C3).
VOCABULARY_VERSION = "1.0.0"

# Moves a chemist would reach for that the five-slot chemotype cannot express, so they are not
# already covered by the mechanically generated one-slot table. They are written down AFTER reading
# the actual constraints of this instance (maximize QED, MW 345-385, LogP 1.0-2.3, TPSA 62-88,
# SA <= 3.5, zero alerts, similarity >= 0.40), and every one of them is a sensible first attempt.
# Aniline is included because it is an unconditional Brenk alert and a reviewer should see that
# confirmed rather than asserted.
START = "CC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1"
ORDINARY_MOVES = {
    "acetamido -> aniline NH2 (alert check)": "Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "acetamido -> hydroxyl": "Oc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "acetamido -> methoxy": "COc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "acetamido -> nitrile": "N#Cc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "acetamido -> methanesulfonamido": "CS(=O)(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "drop the acetamido entirely": "Fc1ccc(S(=O)(=O)N(C)CCCN2CCCC2)cc1",
    "acetamido -> propionamido": "CCC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "ring fluorine -> chlorine": "CC(=O)Nc1ccc(Cl)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "drop the ring fluorine": "CC(=O)Nc1cccc(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "N-methyl -> N-H on the sulfonamide": "CC(=O)Nc1ccc(F)c(S(=O)(=O)NCCCN2CCCC2)c1",
    "shorten the linker C3 -> C2": "CC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCN2CCCC2)c1",
    "pyrrolidine -> piperidine": "CC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCCC2)c1",
    "pyrrolidine -> morpholine": "CC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCOCC2)c1",
    "pyrrolidine -> dimethylamino": "CC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN(C)C)c1",
    "sulfonamide -> sulfone (drop the N)": "CC(=O)Nc1ccc(F)c(S(=O)(=O)CCCN2CCCC2)c1",
}


def smiles_for(idx):
    """aryl(-SUB)(-CAP) - S(=O)(=O) - N(NSUB) - LINK - TAIL, or None for a nonsensical combination.

    A nitrogen tail with an empty linker would place nitrogen directly on the sulfonamide nitrogen,
    which is not a motif anyone would make, so those combinations are skipped rather than scored.
    """
    tail, link, nsub, sub, cap = (SLOTS[i][idx[i]] for i in range(5))
    if link == "" and tail.startswith("N"):
        return None
    amide_n = f"N({nsub})" if nsub else "N"
    if cap and sub:
        ring = f"c1cc({sub})ccc1{cap}"
    elif cap:
        ring = f"c1ccc({cap})cc1"
    elif sub:
        ring = f"c1cccc({sub})c1"
    else:
        ring = "c1ccccc1"
    return f"O=S(=O)({amide_n}{link}{tail}){ring}"


def k_slot_neighbours(idx, k):
    out = []
    for combo in itertools.combinations(range(5), k):
        for vals in itertools.product(*(range(len(SLOTS[c])) for c in combo)):
            if any(vals[j] == idx[combo[j]] for j in range(k)):
                continue
            nxt = list(idx)
            for j, c in enumerate(combo):
                nxt[c] = vals[j]
            out.append(tuple(nxt))
    return out


def main() -> None:
    cfg = load_yaml(TASK_CFG)
    ref_cfg = load_yaml(REFERENCE_CFG)
    start_smiles = cfg["start_smiles"]
    constraints = cfg["constraints"]
    sim_cfg = constraints["similarity"]
    sim_floor = float(sim_cfg["min"])
    bands = [(float(b["min_achievement"]), int(b["points"]))
             for b in ref_cfg["scoring"]["optimality_bands"]]

    def ev(smiles: str):
        return evaluate_smiles(smiles, start_smiles=start_smiles, similarity_cfg=sim_cfg)

    start = ev(start_smiles)
    print(f"START  {start_smiles}")
    print(f"       QED={start.qed:.4f} MW={start.molecular_weight:.1f} LogP={start.logp:.3f} "
          f"TPSA={start.tpsa:.2f} SA={start.sa_score:.2f} alerts={start.n_alerts}")
    start_broken = [k for k, v in check_constraints(start, constraints).items() if not v]
    print(f"       start feasible: "
          f"{'YES' if not start_broken else 'NO -> ' + ','.join(start_broken)}")
    print(f"WINDOWS  MW [{constraints['molecular_weight']['min']}, "
          f"{constraints['molecular_weight']['max']}]  "
          f"LogP [{constraints['logp']['min']}, {constraints['logp']['max']}]  "
          f"SA <= {constraints['sa_score']['max']}  alerts = 0  similarity >= {sim_floor}")

    all_idx = list(itertools.product(*(range(len(s)) for s in SLOTS)))
    print(f"\nenumerating {len(all_idx)} designs (slot sizes {[len(s) for s in SLOTS]}) ...")
    evals, index_of = {}, {}
    n_skipped_nonsensical, n_invalid = 0, 0
    for i in all_idx:
        smi = smiles_for(i)
        if smi is None:
            n_skipped_nonsensical += 1
            continue
        e = ev(smi)
        if e.valid:
            evals[i] = e
            index_of.setdefault(e.canonical_smiles, i)
        else:
            n_invalid += 1
    print(f"  {len(evals)} valid, {n_skipped_nonsensical} skipped as nonsensical combinations, "
          f"{n_invalid} rejected by the admissibility gate")

    start_idx = index_of.get(start.canonical_smiles)
    if start_idx is None:
        raise SystemExit("ERROR: the starting molecule is not a member of the enumerated library; "
                         "the slot vocabulary and start_smiles have drifted apart.")

    feasible = {i: e for i, e in evals.items()
                if e.qed > start.qed and satisfies_all_constraints(e, constraints)}
    if not feasible:
        raise SystemExit("ERROR: no feasible improving design; the task as configured is unsolvable.")

    ref_idx, ref_eval = max(feasible.items(), key=lambda kv: kv[1].qed)
    reference = ref_eval.qed
    denom = reference - start.qed

    def score(e) -> int:
        if not satisfies_all_constraints(e, constraints):
            return 0
        achievement = max(0.0, min(1.0, (e.qed - start.qed) / denom))
        for threshold, pts in bands:
            if achievement >= threshold:
                return pts
        return 0

    print(f"\nfeasible improving designs: {len(feasible)}")
    print(f"REFERENCE  QED={reference:.4f}  similarity={ref_eval.similarity_to_start:.3f}  "
          f"{ref_eval.canonical_smiles}")
    print(f"           MW={ref_eval.molecular_weight:.1f} LogP={ref_eval.logp:.2f} "
          f"SA={ref_eval.sa_score:.2f} alerts={ref_eval.n_alerts}")
    changed = [SLOT_NAMES[s] for s in range(5) if start_idx[s] != ref_idx[s]]
    print(f"           slots changed vs the start: {len(changed)} ({', '.join(changed)})")

    print("\nG1  THICK TINKER CEILING -- best score reachable by k-slot edits")
    g1 = 0
    k_slot_ceilings = {}
    for k in (1, 2, 3):
        group = [evals[n] for n in k_slot_neighbours(start_idx, k) if n in evals]
        feas_n = sum(1 for e in group if satisfies_all_constraints(e, constraints))
        ceiling = max((score(e) for e in group), default=0)
        k_slot_ceilings[str(k)] = {"tested": len(group), "feasible": feas_n, "ceiling": ceiling}
        if k <= 2:
            g1 = max(g1, ceiling)
        print(f"      {k}-slot: {len(group):5d} designs, {feas_n:4d} feasible, ceiling = {ceiling}")
    print(f"      G1 (one- and two-slot edits) = {g1}  -> "
          f"{'PASS (< 50)' if g1 < 50 else 'REJECT (>= 50)'}")

    # 0.05, not 0.10: on the first calibration of this problem deepseek-v4-pro submitted a design at
    # similarity 0.492 against a 0.40 floor and scored 63. A model demonstrably operates that close
    # to the floor, so the "safe" region has to be defined as floor + 0.05.
    safe_margin = 0.05
    safe = [e for e in feasible.values() if e.similarity_to_start >= sim_floor + safe_margin]
    g2 = max((score(e) for e in safe), default=0)
    print(f"\nG2  SAFE-SIMILARITY CEILING -- best score with similarity >= "
          f"{sim_floor + safe_margin:.2f}")
    print(f"      {len(safe)} such designs, ceiling = {g2}  -> "
          f"{'PASS (< 50)' if g2 < 50 else 'REJECT (>= 50)'}")
    print("      similarity band        n feasible   max score")
    similarity_bands = []
    for lo, hi in [(0.40, 0.45), (0.45, 0.50), (0.50, 0.60), (0.60, 0.75), (0.75, 1.01)]:
        members = [e for e in feasible.values() if lo <= e.similarity_to_start < hi]
        band_max = max((score(e) for e in members), default=0)
        similarity_bands.append({"lo": lo, "hi": hi, "n": len(members), "max_score": band_max})
        print(f"      [{lo:.2f}, {hi:.2f}){'':<11} {len(members):>6} {band_max:>11}")

    counts = Counter(score(e) for e in feasible.values())
    print(f"\nG3  FAIRNESS -- score distribution over the {len(feasible)} feasible designs")
    for pts in sorted(counts, reverse=True):
        print(f"      score {pts:3d}: {counts[pts]:4d} designs "
              f"({100 * counts[pts] / len(feasible):5.1f}%)")
    print(f"      -> {'OK' if len(feasible) >= 50 else 'possibly too tight'}")

    # ------------------------------------------------------------------------------------------
    # G4  CHECKLIST-ONLY CEILING
    #
    # The tool layer added in the 2026-07 revision hands the model a boolean feasibility checklist,
    # so "submit something feasible" is now cheap to execute reliably. That raises a fair question
    # from a reviewer: did the tools collapse the task? G4 answers it with a number. It is the score
    # distribution of a policy that can verify feasibility perfectly but cannot see the objective --
    # i.e. one that picks uniformly among feasible improving designs. If that policy's expected score
    # approached the pass mark, the tools would have given the answer away.
    # ------------------------------------------------------------------------------------------
    all_feasible_scores = [score(e) for e in feasible.values()]
    g4_expected = sum(all_feasible_scores) / len(all_feasible_scores)
    g4_p50 = sum(1 for s in all_feasible_scores if s >= 50) / len(all_feasible_scores)
    g4_max = max(all_feasible_scores)
    # A checklist can also confirm feasibility for designs that do NOT improve the objective; those
    # score 0. Counting them is the more honest denominator for "submit anything the checklist likes".
    checklist_ok = [e for e in evals.values() if satisfies_all_constraints(e, constraints)]
    g4_expected_incl_nonimproving = (
        sum(score(e) for e in checklist_ok) / len(checklist_ok) if checklist_ok else 0.0
    )
    print("\nG4  CHECKLIST-ONLY CEILING -- what feasibility-verification alone is worth")
    print(f"      uniform over the {len(feasible)} feasible IMPROVING designs:")
    print(f"        expected score = {g4_expected:.1f}   P(score >= 50) = {g4_p50:.3f}   "
          f"max = {g4_max}")
    print(f"      uniform over all {len(checklist_ok)} designs the checklist accepts "
          f"(improving or not):")
    print(f"        expected score = {g4_expected_incl_nonimproving:.1f}")
    print(f"      -> {'PASS (expected < 50)' if g4_expected < 50 else 'REJECT (>= 50)'}: "
          f"the objective is still not computable, so feasibility alone does not buy a pass")

    # The single-slot moves ranked by score, LABELLED. This list is generated mechanically from the
    # vocabulary rather than hand-written, which removes the bias that broke this check on two
    # earlier problems: a hand-written list can only contain moves the designer already thought of,
    # so it silently confirms the design instead of testing it. Everything reachable in one edit is
    # here by construction.
    def label(from_idx, to_idx) -> str:
        parts = []
        for s_i in range(5):
            if from_idx[s_i] != to_idx[s_i]:
                a = SLOTS[s_i][from_idx[s_i]] or "H"
                b = SLOTS[s_i][to_idx[s_i]] or "H"
                parts.append(f"{SLOT_NAMES[s_i]} {a} -> {b}")
        return "; ".join(parts)

    print("\nEVERY ONE-SLOT MOVE, ranked by score (mechanically generated, not hand-picked):")
    one = [(n, evals[n]) for n in k_slot_neighbours(start_idx, 1) if n in evals]
    one.sort(key=lambda t: -score(t[1]))
    one_slot_rows = []
    for n, e in one:
        broken = [k for k, v in check_constraints(e, constraints).items() if not v]
        one_slot_rows.append({"label": label(start_idx, n), "qed": e.qed,
                              "mw": e.molecular_weight, "logp": e.logp,
                              "similarity": e.similarity_to_start, "score": score(e),
                              "violated": broken, "smiles": e.canonical_smiles})
    for row in one_slot_rows[:12]:
        print(f"      score={row['score']:3d} QED={row['qed']:.4f} MW={row['mw']:6.1f} "
              f"LogP={row['logp']:5.2f} sim={row['similarity']:.3f}  {row['label']:44s} "
              f"{'FEASIBLE' if not row['violated'] else 'blocked: ' + ','.join(row['violated'])}")

    print("\nMOVES OUTSIDE THE ENUMERATED CHEMOTYPE (hand-listed, to show they are covered too):")
    worst = 0
    outside_rows = []
    for move_label, smiles in ORDINARY_MOVES.items():
        e = ev(smiles)
        if not e.valid:
            print(f"      {move_label:38s} INVALID SMILES")
            continue
        broken = [k for k, v in check_constraints(e, constraints).items() if not v]
        pts = score(e)
        worst = max(worst, pts)
        outside_rows.append({"label": move_label, "qed": e.qed, "mw": e.molecular_weight,
                             "logp": e.logp, "similarity": e.similarity_to_start,
                             "score": pts, "violated": broken, "smiles": e.canonical_smiles})
        verdict = "FEASIBLE" if not broken else "blocked: " + ",".join(broken)
        print(f"      {move_label:38s} QED={e.qed:.4f} MW={e.molecular_weight:6.1f} "
              f"sim={e.similarity_to_start:.3f} score={pts:3d}  {verdict}")
    print(f"      best ordinary move = {worst}")

    ranked = sorted(feasible.values(), key=lambda e: e.qed, reverse=True)
    print("\nTOP FEASIBLE DESIGNS")
    for e in ranked[:10]:
        print(f"      score={score(e):3d} QED={e.qed:.4f} MW={e.molecular_weight:6.1f} "
              f"LogP={e.logp:5.2f} SA={e.sa_score:.2f} sim={e.similarity_to_start:.3f}  "
              f"{e.canonical_smiles}")

    stored = ref_cfg.get("reference", {}).get("best_objective_in_window")
    match = stored is not None and abs(float(stored) - reference) < 1e-4
    print(f"\ntask_config reference.best_objective_in_window = {stored} -> "
          f"{'MATCHES enumeration' if match else f'UPDATE NEEDED ({reference:.6f})'}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        # The vocabulary this whole file is a function of. Mirrored into reference_target.yaml and
        # asserted equal there, so a silent edit to a slot list cannot leave stale gates behind.
        "vocabulary": {
            "version": VOCABULARY_VERSION,
            "slot_names": SLOT_NAMES,
            "slot_sizes": {name: len(s) for name, s in zip(SLOT_NAMES, SLOTS)},
            "cartesian_product": len(all_idx),
            "n_skipped_nonsensical": n_skipped_nonsensical,
            "n_rejected_by_admissibility": n_invalid,
            "n_valid_designs": len(evals),
            "n_out_of_chemotype_moves": len(ORDINARY_MOVES),
        },
        "start_smiles": start.canonical_smiles,
        "start_qed": start.qed,
        "best_objective_in_window": reference,
        "best_smiles_in_window": ref_eval.canonical_smiles,
        "n_feasible_solutions_found": len(feasible),
        "n_slots_changed_vs_start": len(changed),
        "slots_changed_vs_start": changed,
        "gate_g1_one_slot_ceiling": k_slot_ceilings["1"]["ceiling"],
        "gate_g1_thick_tinker_ceiling": g1,
        "gate_g2_safe_similarity_ceiling": g2,
        "gate_g2_safe_margin": safe_margin,
        "gate_g3_n_feasible": len(feasible),
        "gate_g4_checklist_only_expected_score": round(g4_expected, 1),
        "gate_g4_checklist_only_p_at_least_50": round(g4_p50, 3),
        "gate_g4_checklist_only_max_score": g4_max,
        "gate_g4_checklist_only_expected_incl_nonimproving": round(g4_expected_incl_nonimproving, 1),
        "gate_g4_n_checklist_accepted_designs": len(checklist_ok),
        "best_ordinary_move_score": worst,
        "score_distribution": {str(k): v for k, v in sorted(counts.items(), reverse=True)},
        "similarity_bands": similarity_bands,
        "one_slot_moves": one_slot_rows,
        "out_of_chemotype_moves": outside_rows,
        "k_slot_ceilings": k_slot_ceilings,
        "windows": {"mw": [constraints["molecular_weight"]["min"], constraints["molecular_weight"]["max"]],
                    "logp": [constraints["logp"]["min"], constraints["logp"]["max"]],
                    "tpsa": ([constraints["tpsa"]["min"], constraints["tpsa"]["max"]]
                             if constraints.get("tpsa") else None),
                    "sa_max": constraints["sa_score"]["max"],
                    "similarity_min": sim_floor},
        "start_properties": {"qed": start.qed, "mw": start.molecular_weight, "logp": start.logp,
                             "tpsa": start.tpsa, "sa": start.sa_score, "hbd": None,
                             "n_alerts": start.n_alerts},
        "reference_properties": {"qed": ref_eval.qed, "mw": ref_eval.molecular_weight,
                                 "logp": ref_eval.logp, "tpsa": ref_eval.tpsa,
                                 "sa": ref_eval.sa_score,
                                 "similarity": ref_eval.similarity_to_start},
        "top": [{"qed": e.qed, "mw": e.molecular_weight, "logp": e.logp, "sa": e.sa_score,
                 "similarity": e.similarity_to_start, "score": score(e),
                 "smiles": e.canonical_smiles} for e in ranked[:50]],
    }, indent=2), encoding="utf-8")
    print(f"written -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

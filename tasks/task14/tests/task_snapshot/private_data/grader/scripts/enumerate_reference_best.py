"""Designer-only: enumerate the local chemical space, derive the reference target, and prove the
difficulty mechanism with numbers.

The optimality block is scored against a reference QED. This script derives that value by systematic
enumeration of an ORDERED SLOT PRODUCT, so the value stored in `grader/reference/reference_target.yaml`
is reproducible rather than hand-picked, and it prints the three gate measurements that justify the
instance:

  G1  thick tinker ceiling     -- the best score reachable by ALL one-slot and ALL two-slot edits.
  G2  safe-similarity ceiling  -- the best score reachable while keeping similarity >= floor + 0.05.
  G3  fairness                 -- how many distinct feasible improving designs exist, and how the
                                  score bands are populated across them (the band distribution has to
                                  be a ramp, not a wall).

Why ordered slots: changing one slot is one edit, which turns "could a model just tinker its way to a
pass?" from a judgement call into a measurement. The one-slot table printed at the end is generated
mechanically from the vocabulary rather than hand-written -- a hand-written list of "ordinary moves"
can only contain the moves the designer already thought of, so it confirms the design instead of
testing it.

Framing: this is the best QED found by *this* enumeration, not a proof of global optimality.
Molecular space is unbounded and a model can legitimately do better; achievement is clipped at 1.0,
so meeting or exceeding the target earns full marks.

Run:
    python grader/scripts/enumerate_reference_best.py
    python grader/scripts/enumerate_reference_best.py --jobs 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "public" / "src"))
sys.path.insert(0, str(ROOT / "grader" / "src"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import evaluate_smiles, check_constraints, satisfies_all_constraints  # noqa: E402
from utils import load_yaml  # noqa: E402

# Constraints and the starting molecule come from the PUBLIC task config; the score bands and the
# similarity margin come from the grader-only reference file. Splitting the inputs like this keeps
# the enumeration honest after the public/grader split: edit a window on the public side and the
# gates recomputed here move with it.
TASK_CFG = ROOT / "public" / "tasks" / "q1_multiobjective" / "task_config.yaml"
REFERENCE_CFG = ROOT / "grader" / "reference" / "reference_target.yaml"
OUT = ROOT / "grader" / "answers" / "reference_enumeration.json"

# ---------------------------------------------------------------------------------------------
# The chemotype, as a product of six ORDERED slots:
#
#     ARYL(-SUB)(-CAP) - N(ANSUB) - C(=O) - N(NSUB) - LINK - TAIL     (a disubstituted aryl urea)
#
# The aryl urea replaces the benzoate-ester/acetanilide chemotype the first release shared with
# another problem in this suite (second-round review, 田双燕 §2.4). The starting molecule in
# task_config.yaml was not chosen by hand: it is a survivor of a mechanical screen of every feasible
# member of this library as a candidate start, against the three gates measured here.
#
# Donor-bearing options (hydroxyl, hydroxymethyl, amide, acetamido) appear in every slot on purpose:
# QED's eight desirability components include a hydrogen-bond-donor term, so a library with no donors
# cannot estimate the reachable optimum. Aniline (a free aryl NH2) is deliberately absent -- it is an
# unconditional Brenk alert, so it can only produce infeasible designs.
# ---------------------------------------------------------------------------------------------
TAIL = [
    "C", "CC", "CCC", "C(C)C", "CCCC", "C(C)(C)C", "CC(C)C", "CCCCC",
    "C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1", "CC1CC1", "CC1CCCC1", "CC1CCCCC1",
    "N1CCCC1", "N1CCCCC1", "N1CCOCC1", "N1CCN(C)CC1", "N1CCC(O)CC1", "N1CCCCCC1",
    "N1CCC(F)(F)CC1", "N1CCS(=O)(=O)CC1",
    "CO", "CCO", "CC(C)O", "CCOC", "N(C)C", "CC#N", "CC(N)=O",
    "c1ccccc1", "c1ccncc1", "Cc1ccccc1", "C1CCOCC1", "C1COCC1",
]
LINK = ["", "C", "CC", "CCC", "CC(C)"]
NSUB = ["", "C", "CC", "C(C)C"]
SUB = ["", "F", "Cl", "C", "CC", "OC", "OCC", "C#N", "O", "CO", "NC(C)=O", "C(N)=O",
       "N(C)C", "S(C)(=O)=O"]
CAP = ["", "F", "Cl", "Br", "C", "CC", "C(C)C", "C(C)(C)C", "OC", "OCC", "C#N", "SC",
       "C(F)(F)F", "OC(F)(F)F", "C1CC1", "O", "CO", "NC(C)=O", "C(N)=O", "S(C)(=O)=O",
       "N(C)C", "CS(C)(=O)=O"]
# Substituent on the ANILIDE nitrogen. Added after the first calibration run: deepseek-v4-pro
# submitted an N-methylated anilide in round 3, i.e. a move the five-slot vocabulary could not
# express, so the gates measured over that vocabulary did not bound it. N-methylation of an anilide
# NH is a standard medicinal-chemistry move (it removes a donor and a large slice of polar surface),
# and a gate measured over a vocabulary that cannot express it is not a bound. See 审核说明.md §3.
ANSUB = ["", "C"]

SLOTS = [TAIL, LINK, NSUB, SUB, CAP, ANSUB]
SLOT_NAMES = ["TAIL", "LINK", "NSUB", "SUB", "CAP", "ANSUB"]
N_SLOTS = len(SLOTS)


def ring_of(sub: str, cap: str) -> str:
    """Benzene attached at position 1, SUB meta (3), CAP para (4).

    Ring-closure label 2 is used for the aromatic ring so that a SUB or CAP carrying its own ring
    (cyclopropyl) can safely use label 1 inside it.
    """
    if cap and sub:
        return f"c2cc({sub})c({cap})cc2"
    if cap:
        return f"c2ccc({cap})cc2"
    if sub:
        return f"c2cc({sub})ccc2"
    return "c2ccccc2"


def smiles_for(idx):
    """ARYL(SUB)(CAP)-N(ANSUB)-C(=O)-N(NSUB)-LINK-TAIL, or None for a nonsensical combination.

    A nitrogen, oxygen or aryl tail with an empty linker would place a heteroatom directly on the
    urea nitrogen (a semicarbazide) or make a triaryl urea, neither of which is the chemotype under
    study, so those combinations are skipped rather than scored.
    """
    tail, link, nsub, sub, cap, ansub = (SLOTS[i][idx[i]] for i in range(N_SLOTS))
    if link == "" and (tail.startswith("N") or tail.startswith("O") or tail.startswith("c")):
        return None
    n1 = f"N({ansub})" if ansub else "N"
    n2 = f"N({nsub})" if nsub else "N"
    return f"O=C({n1}{ring_of(sub, cap)}){n2}{link}{tail}"


def k_slot_neighbours(idx, k):
    out = []
    for combo in itertools.combinations(range(N_SLOTS), k):
        for vals in itertools.product(*(range(len(SLOTS[c])) for c in combo)):
            if any(vals[j] == idx[combo[j]] for j in range(k)):
                continue
            nxt = list(idx)
            for j, c in enumerate(combo):
                nxt[c] = vals[j]
            out.append(tuple(nxt))
    return out


_CTX = {}


def _init(start_smiles, sim_cfg):
    sys.path.insert(0, str(ROOT / "public" / "src"))
    from rdkit import RDLogger as RL
    RL.DisableLog("rdApp.*")
    _CTX["start"] = start_smiles
    _CTX["sim"] = sim_cfg


def _work(chunk):
    out = []
    for idx in chunk:
        smi = smiles_for(idx)
        if smi is None:
            continue
        e = evaluate_smiles(smi, start_smiles=_CTX["start"], similarity_cfg=_CTX["sim"])
        if not e.valid:
            continue
        out.append((idx, e.canonical_smiles, e.qed, e.molecular_weight, e.logp, e.tpsa,
                    e.sa_score, e.n_alerts, e.similarity_to_start))
    return out


class Row:
    """Duck-typed stand-in so check_constraints/satisfies_all_constraints can be reused verbatim."""

    __slots__ = ("valid", "molecular_weight", "logp", "tpsa", "qed", "sa_score",
                 "similarity_to_start", "n_alerts", "canonical_smiles")

    def __init__(self, canon, qed, mw, logp, tpsa, sa, n_alerts, sim):
        self.valid = True
        self.canonical_smiles = canon
        self.qed = qed
        self.molecular_weight = mw
        self.logp = logp
        self.tpsa = tpsa
        self.sa_score = sa
        self.n_alerts = n_alerts
        self.similarity_to_start = sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    cfg = load_yaml(TASK_CFG)
    ref_cfg = load_yaml(REFERENCE_CFG)
    start_smiles = cfg["start_smiles"]
    constraints = cfg["constraints"]
    sim_cfg = constraints["similarity"]
    sim_floor = float(sim_cfg["min"])
    bands = [(float(b["min_achievement"]), int(b["points"]))
             for b in ref_cfg["scoring"]["optimality_bands"]]
    safe_margin = float(ref_cfg["gates"]["gate_g2_safe_margin"])

    start = evaluate_smiles(start_smiles, start_smiles=start_smiles, similarity_cfg=sim_cfg)
    print(f"START  {start_smiles}")
    print(f"       QED={start.qed:.4f} MW={start.molecular_weight:.1f} LogP={start.logp:.3f} "
          f"TPSA={start.tpsa:.2f} SA={start.sa_score:.2f} alerts={start.n_alerts}")
    start_broken = [k for k, v in check_constraints(start, constraints).items() if not v]
    print(f"       start feasible: "
          f"{'YES' if not start_broken else 'NO -> ' + ','.join(start_broken)}")
    print(f"WINDOWS  MW [{constraints['molecular_weight']['min']}, "
          f"{constraints['molecular_weight']['max']}]  "
          f"LogP [{constraints['logp']['min']}, {constraints['logp']['max']}]  "
          f"TPSA [{constraints['tpsa']['min']}, {constraints['tpsa']['max']}]  "
          f"SA <= {constraints['sa_score']['max']}  alerts = 0  similarity >= {sim_floor}")

    all_idx = [i for i in itertools.product(*(range(len(s)) for s in SLOTS))
               if smiles_for(i) is not None]
    print(f"\nenumerating {len(all_idx)} designs (slot sizes {[len(s) for s in SLOTS]}) "
          f"on {args.jobs} process(es) ...", flush=True)

    rows, index_of = {}, {}
    chunks = [all_idx[i::args.jobs * 8] for i in range(args.jobs * 8)]
    with Pool(args.jobs, initializer=_init, initargs=(start_smiles, sim_cfg)) as pool:
        for part in pool.imap_unordered(_work, chunks):
            for idx, canon, qed, mwv, logpv, tpsav, sav, nal, simv in part:
                rows[idx] = Row(canon, qed, mwv, logpv, tpsav, sav, nal, simv)
                index_of.setdefault(canon, idx)
    print(f"  {len(rows)} parsed, {len(index_of)} distinct molecules")

    start_idx = index_of.get(start.canonical_smiles)
    if start_idx is None:
        raise SystemExit("ERROR: the starting molecule is not a member of the enumerated library; "
                         "the slot vocabulary and start_smiles have drifted apart.")

    # Deduplicate by canonical SMILES so a molecule reachable by two index paths is counted once.
    # Sorted, because the pool returns chunks in completion order: without this the tie between two
    # designs with identical QED would be broken by process scheduling, and the reference SMILES
    # stored in reference_target.yaml would not be reproducible.
    unique = {}
    for idx in sorted(rows):
        r = rows[idx]
        unique.setdefault(r.canonical_smiles, r)

    feasible = {s: r for s, r in unique.items()
                if r.qed > start.qed and satisfies_all_constraints(r, constraints)}
    if not feasible:
        raise SystemExit("ERROR: no feasible improving design; the task as configured is unsolvable.")

    # Deterministic tie-break on the canonical SMILES: two designs really can share a QED to six
    # decimals here (a fluorine and a nitrile swapped between the two ring positions).
    ref_eval = max(feasible.values(), key=lambda r: (r.qed, r.canonical_smiles))
    reference = ref_eval.qed
    denom = reference - start.qed

    def score(r) -> int:
        if not satisfies_all_constraints(r, constraints) or r.qed <= start.qed:
            return 0
        achievement = max(0.0, min(1.0, (r.qed - start.qed) / denom))
        for threshold, pts in bands:
            if achievement >= threshold:
                return pts
        return 0

    print(f"\nfeasible improving designs: {len(feasible)}")
    print(f"REFERENCE  QED={reference:.6f}  similarity={ref_eval.similarity_to_start:.3f}  "
          f"{ref_eval.canonical_smiles}")
    print(f"           MW={ref_eval.molecular_weight:.1f} LogP={ref_eval.logp:.2f} "
          f"TPSA={ref_eval.tpsa:.1f} SA={ref_eval.sa_score:.2f} alerts={ref_eval.n_alerts}")
    ref_idx = index_of[ref_eval.canonical_smiles]
    changed = [SLOT_NAMES[s] for s in range(N_SLOTS) if start_idx[s] != ref_idx[s]]
    print(f"           slots changed vs the start: {len(changed)} ({', '.join(changed)})")
    print(f"           objective span start -> reference: {denom:.4f} QED units")

    print("\nG1  THICK TINKER CEILING -- best score reachable by k-slot edits")
    g1 = 0
    k_slot_ceilings = {}
    for k in (1, 2, 3):
        group = [rows[n] for n in k_slot_neighbours(start_idx, k) if n in rows]
        feas_n = sum(1 for r in group if satisfies_all_constraints(r, constraints))
        ceiling = max((score(r) for r in group), default=0)
        k_slot_ceilings[str(k)] = {"tested": len(group), "feasible": feas_n, "ceiling": ceiling}
        if k <= 2:
            g1 = max(g1, ceiling)
        print(f"      {k}-slot: {len(group):6d} designs, {feas_n:5d} feasible, ceiling = {ceiling}")
    print(f"      G1 (one- and two-slot edits) = {g1}  -> "
          f"{'PASS (< 50)' if g1 < 50 else 'REJECT (>= 50)'}")

    # 0.05, not 0.10: on an earlier problem in this series deepseek-v4-pro submitted a design at
    # similarity 0.492 against a 0.40 floor and scored 63. A model demonstrably operates that close
    # to the floor, so the "safe" region has to be defined as floor + 0.05 -- and the screen that
    # produced this starting molecule used exactly the same margin, so this gate is not a re-test
    # under looser rules than the screen.
    safe = [r for r in feasible.values() if r.similarity_to_start >= sim_floor + safe_margin]
    g2 = max((score(r) for r in safe), default=0)
    print(f"\nG2  SAFE-SIMILARITY CEILING -- best score with similarity >= "
          f"{sim_floor + safe_margin:.2f}")
    print(f"      {len(safe)} such designs, ceiling = {g2}  -> "
          f"{'PASS (< 50)' if g2 < 50 else 'REJECT (>= 50)'}")
    print("      similarity band        n feasible   max score")
    similarity_bands = []
    edges = [round(sim_floor + 0.05 * i, 2) for i in range(6)] + [1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        members = [r for r in feasible.values() if lo <= r.similarity_to_start < hi]
        band_max = max((score(r) for r in members), default=0)
        similarity_bands.append({"lo": lo, "hi": hi, "n": len(members), "max_score": band_max})
        print(f"      [{lo:.2f}, {hi:.2f}){'':<11} {len(members):>6} {band_max:>11}")

    # ------------------------------------------------------------------------------------------
    # P-graded  --  is a COMPETENT design able to land, or is the feasible region a knife edge?
    #
    # Added after three consecutive instances passed G1/G2/G3 and still produced six all-zero
    # trajectories. The models were not incompetent: they repeatedly reached QED within 0.02 of the
    # reference and were rejected by margins of 0.004-0.048. G1/G2 bound the CEILING; nothing
    # bounded whether a good design can actually satisfy everything at once. These two numbers do.
    #
    #   P1  of every design in the library at QED >= start + 0.6*span (what a competent model aims
    #       for), what fraction is feasible?                              REQUIRED >= 5%
    #   P2  how many feasible improving designs have MARGIN on every constraint at once --
    #       similarity >= floor+0.05, LogP >= 0.05 inside both bounds, MW >= 3 Da inside both
    #       bounds? A model cannot land on a knife edge.                  REQUIRED >= 10
    # ------------------------------------------------------------------------------------------
    hi_cut = start.qed + 0.6 * denom
    high_qed = [r for r in unique.values() if r.qed >= hi_cut]
    def in_windows(r) -> bool:
        return (constraints["molecular_weight"]["min"] <= r.molecular_weight <= constraints["molecular_weight"]["max"]
                and constraints["logp"]["min"] <= r.logp <= constraints["logp"]["max"]
                and constraints["tpsa"]["min"] <= r.tpsa <= constraints["tpsa"]["max"]
                and r.sa_score <= constraints["sa_score"]["max"] and r.n_alerts == 0)
    n_hi = len(high_qed)
    n_hi_windows = sum(1 for r in high_qed if in_windows(r))
    n_hi_all = sum(1 for r in high_qed if satisfies_all_constraints(r, constraints))
    p1_windows = (n_hi_windows / n_hi) if n_hi else 0.0
    p1_all = (n_hi_all / n_hi) if n_hi else 0.0
    mw_c, lp_c = constraints["molecular_weight"], constraints["logp"]
    comfortable = [r for r in feasible.values()
                   if r.similarity_to_start >= sim_floor + safe_margin
                   and mw_c["min"] + 3.0 <= r.molecular_weight <= mw_c["max"] - 3.0
                   and lp_c["min"] + 0.05 <= r.logp <= lp_c["max"] - 0.05]
    print(f"\nP-graded  CAN A COMPETENT DESIGN LAND?")
    print(f"      P1: {n_hi} designs in the library reach QED >= {hi_cut:.4f} "
          f"(start + 0.6 x span)")
    print(f"          {n_hi_windows} of them fit the property windows   ({100 * p1_windows:.1f}%)")
    print(f"          {n_hi_all} of them satisfy EVERY constraint       ({100 * p1_all:.1f}%)"
          f"  -> {'PASS (>= 5%)' if p1_all >= 0.05 else 'REJECT (< 5%)'}")
    print(f"      P2: {len(comfortable)} feasible improving designs have margin on every "
          f"constraint at once  -> {'PASS (>= 10)' if len(comfortable) >= 10 else 'REJECT (< 10)'}")

    counts = Counter(score(r) for r in feasible.values())
    print(f"\nG3  FAIRNESS -- score distribution over the {len(feasible)} feasible designs")
    for pts in sorted(counts, reverse=True):
        print(f"      score {pts:3d}: {counts[pts]:5d} designs "
              f"({100 * counts[pts] / len(feasible):5.1f}%)")
    print(f"      -> {'OK (>= 50 feasible improving designs)' if len(feasible) >= 50 else 'TOO TIGHT'}")

    def label(from_idx, to_idx) -> str:
        parts = []
        for s_i in range(N_SLOTS):
            if from_idx[s_i] != to_idx[s_i]:
                a = SLOTS[s_i][from_idx[s_i]] or "H"
                b = SLOTS[s_i][to_idx[s_i]] or "H"
                parts.append(f"{SLOT_NAMES[s_i]} {a} -> {b}")
        return "; ".join(parts)

    print("\nEVERY ONE-SLOT MOVE, ranked by score (mechanically generated, not hand-picked):")
    one = [(n, rows[n]) for n in k_slot_neighbours(start_idx, 1) if n in rows]
    one.sort(key=lambda t: (-score(t[1]), -t[1].qed))
    one_slot_rows = []
    for n, r in one:
        broken = [k for k, v in check_constraints(r, constraints).items() if not v]
        one_slot_rows.append({"label": label(start_idx, n), "qed": r.qed,
                              "mw": r.molecular_weight, "logp": r.logp,
                              "similarity": r.similarity_to_start, "score": score(r),
                              "violated": broken, "smiles": r.canonical_smiles})
    for row in one_slot_rows[:14]:
        print(f"      score={row['score']:3d} QED={row['qed']:.4f} MW={row['mw']:6.1f} "
              f"LogP={row['logp']:5.2f} sim={row['similarity']:.3f}  {row['label']:42s} "
              f"{'FEASIBLE' if not row['violated'] else 'blocked: ' + ','.join(row['violated'])}")

    ranked = sorted(feasible.values(), key=lambda r: (-r.qed, r.canonical_smiles))
    print("\nTOP FEASIBLE DESIGNS")
    for r in ranked[:10]:
        print(f"      score={score(r):3d} QED={r.qed:.4f} MW={r.molecular_weight:6.1f} "
              f"LogP={r.logp:5.2f} SA={r.sa_score:.2f} sim={r.similarity_to_start:.3f}  "
              f"{r.canonical_smiles}")

    stored = ref_cfg.get("reference", {}).get("best_objective_in_window")
    match = stored is not None and abs(float(stored) - reference) < 1e-4
    print(f"\nreference_target.yaml reference.best_objective_in_window = {stored} -> "
          f"{'MATCHES enumeration' if match else f'UPDATE NEEDED ({reference:.6f})'}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "start_smiles": start.canonical_smiles,
        "start_qed": start.qed,
        "best_objective_in_window": reference,
        "best_smiles_in_window": ref_eval.canonical_smiles,
        "objective_span": round(denom, 6),
        "n_feasible_solutions_found": len(feasible),
        "n_slots_changed_vs_start": len(changed),
        "slots_changed_vs_start": changed,
        "n_designs_enumerated": len(rows),
        "n_distinct_molecules": len(unique),
        "gate_g1_one_slot_ceiling": k_slot_ceilings["1"]["ceiling"],
        "gate_g1_thick_tinker_ceiling": g1,
        "gate_g2_safe_similarity_ceiling": g2,
        "gate_g2_safe_margin": safe_margin,
        "gate_g2_n_safe_designs": len(safe),
        "gate_g3_n_feasible": len(feasible),
        "p_graded_high_qed_cut": round(hi_cut, 6),
        "p_graded_n_high_qed": n_hi,
        "p_graded_high_qed_in_windows": n_hi_windows,
        "p_graded_high_qed_feasible": n_hi_all,
        "p_graded_p1_windows_fraction": round(p1_windows, 4),
        "p_graded_p1_all_fraction": round(p1_all, 4),
        "p_graded_p2_n_comfortable": len(comfortable),
        "n_feasible_improving_one_slot_edits": sum(
            1 for n_ in k_slot_neighbours(start_idx, 1)
            if n_ in rows and score(rows[n_]) > 0),
        "score_distribution": {str(k): v for k, v in sorted(counts.items(), reverse=True)},
        "similarity_bands": similarity_bands,
        "one_slot_moves": one_slot_rows,
        "k_slot_ceilings": k_slot_ceilings,
        "windows": {"mw": [constraints["molecular_weight"]["min"], constraints["molecular_weight"]["max"]],
                    "logp": [constraints["logp"]["min"], constraints["logp"]["max"]],
                    "tpsa": [constraints["tpsa"]["min"], constraints["tpsa"]["max"]],
                    "sa_max": constraints["sa_score"]["max"],
                    "similarity_min": sim_floor},
        "start_properties": {"qed": start.qed, "mw": start.molecular_weight, "logp": start.logp,
                             "tpsa": start.tpsa, "sa": start.sa_score, "n_alerts": start.n_alerts},
        "reference_properties": {"qed": ref_eval.qed, "mw": ref_eval.molecular_weight,
                                 "logp": ref_eval.logp, "tpsa": ref_eval.tpsa,
                                 "sa": ref_eval.sa_score,
                                 "similarity": ref_eval.similarity_to_start},
        "top": [{"qed": r.qed, "mw": r.molecular_weight, "logp": r.logp, "sa": r.sa_score,
                 "similarity": r.similarity_to_start, "score": score(r),
                 "smiles": r.canonical_smiles} for r in ranked[:50]],
    }, indent=2), encoding="utf-8")
    print(f"written -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

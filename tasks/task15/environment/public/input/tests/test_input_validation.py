"""Regression tests for candidate admissibility (review finding 2).

The review found that the evaluator would parse and measure multi-component inputs and inorganic
ions. These tests pin the fix so it cannot silently regress:

  * salts, mixtures, counter-ions and bare inorganic ions are REJECTED, with a stable reason code;
  * abnormal valences, isotope labels, atom maps, wildcards and radicals are REJECTED;
  * an admissible molecule scores identically no matter how its SMILES was written (standardisation
    is idempotent and order-independent);
  * the constraint windows are inclusive at both ends and exclusive just outside;
  * a rejected input fails every constraint flag, so it cannot pick up partial credit anywhere.

Run standalone (no pytest needed):
    python tests/test_input_validation.py
It also works under pytest if that is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from evaluator import MoleculeEvaluation, check_constraints, evaluate_smiles  # noqa: E402
from utils import load_yaml  # noqa: E402
from validation import (  # noqa: E402
    R_ATOM_MAP,
    R_ELEMENT,
    R_EMPTY,
    R_ISOTOPE,
    R_MULTI_FRAGMENT,
    R_NET_CHARGE,
    R_NO_CARBON,
    R_NON_STRING,
    R_RADICAL,
    R_UNPARSEABLE,
    R_WILDCARD,
    validate_smiles_input,
)

TASK_CFG = load_yaml(ROOT / "tasks" / "q1_lipophilic" / "task_config.yaml")
START = TASK_CFG["start_smiles"]
CONSTRAINTS = TASK_CFG["constraints"]
SIM_CFG = CONSTRAINTS["similarity"]

_failures: list[str] = []
_checks = 0


def check(condition: bool, message: str) -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def expect_rejected(smiles, expected_code: str, label: str) -> None:
    """The input must be rejected, and `expected_code` must be among the reasons."""
    res = validate_smiles_input(smiles)
    ok = (not res.ok) and expected_code in res.reasons
    check(ok, f"rejected [{expected_code}]: {label} "
              f"(got ok={res.ok}, reasons={res.reasons})")
    # A rejected input must also be unusable end-to-end, not merely flagged.
    ev = evaluate_smiles(smiles, start_smiles=START, similarity_cfg=SIM_CFG)
    check(not ev.valid and ev.qed is None,
          f"  -> evaluate_smiles refuses to measure it: {label}")


def expect_accepted(smiles: str, label: str) -> None:
    res = validate_smiles_input(smiles)
    check(res.ok, f"accepted: {label} (reasons={res.reasons})")


# =================================================================================================
def test_salts_and_multicomponent() -> None:
    print("\n1. salts, mixtures and counter-ions")
    expect_rejected(f"{START}.Cl", R_MULTI_FRAGMENT, "start molecule as its HCl salt")
    expect_rejected(f"{START}.O", R_MULTI_FRAGMENT, "start molecule as a hydrate")
    expect_rejected("CC(=O)[O-].[Na+]", R_MULTI_FRAGMENT, "sodium acetate")
    expect_rejected("CN(C)CCCN1CCCC1.Cl.Cl", R_MULTI_FRAGMENT, "bis-HCl salt")
    expect_rejected("CCO.CCO", R_MULTI_FRAGMENT, "two copies of the same molecule")
    expect_rejected(f"{START}.{START}", R_MULTI_FRAGMENT, "the start molecule submitted twice")
    expect_rejected("c1ccccc1.[Na+].[Cl-]", R_MULTI_FRAGMENT, "three components")


def test_inorganic_ions() -> None:
    print("\n2. bare ions and inorganic species")
    expect_rejected("[Na+].[Cl-]", R_MULTI_FRAGMENT, "sodium chloride")
    expect_rejected("[Cl-]", R_NET_CHARGE, "chloride ion")
    expect_rejected("[Na+]", R_NET_CHARGE, "sodium ion")
    expect_rejected("O=S(=O)([O-])[O-]", R_NET_CHARGE, "sulfate dianion")
    expect_rejected("OS(=O)(=O)O", R_NO_CARBON, "sulfuric acid (neutral but not organic)")
    expect_rejected("O", R_NO_CARBON, "water")
    expect_rejected("N", R_NO_CARBON, "ammonia")


def test_charge() -> None:
    print("\n3. net formal charge")
    expect_rejected("CC(=O)[O-]", R_NET_CHARGE, "acetate anion")
    expect_rejected("C[N+](C)(C)C", R_NET_CHARGE, "tetramethylammonium cation")
    expect_rejected("CC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCC[NH+]2CCCC2)c1", R_NET_CHARGE,
                    "protonated form of the start molecule")
    # Net-zero species with formal charges on individual atoms are legitimate single molecules.
    expect_accepted("O=[N+]([O-])c1ccccc1", "nitrobenzene (net-zero, charged atoms)")
    expect_accepted("C[N+](C)(C)CC(=O)[O-]", "betaine zwitterion (net zero)")


def test_abnormal_valence() -> None:
    print("\n4. abnormal valences")
    expect_rejected("CC(C)(C)(C)C", R_UNPARSEABLE, "pentavalent carbon")
    expect_rejected("C[N](C)(C)C", R_UNPARSEABLE, "neutral tetravalent nitrogen")
    expect_rejected("CF(C)C", R_UNPARSEABLE, "trivalent fluorine")
    expect_rejected("c1ccccc1(C)C", R_UNPARSEABLE, "over-substituted aromatic carbon")
    expect_rejected("C(=O)(=O)(=O)C", R_UNPARSEABLE, "impossible carbonyl count")


def test_isotopes_maps_wildcards_radicals() -> None:
    print("\n5. isotopes, atom maps, wildcards, radicals")
    expect_rejected("[13CH3]C(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1", R_ISOTOPE,
                    "13C-labelled start molecule")
    expect_rejected("[2H]c1ccccc1", R_ISOTOPE, "deuterated benzene")
    expect_rejected("[CH3:1]C(=O)Nc1ccccc1", R_ATOM_MAP, "atom-mapped acetanilide")
    expect_rejected("*c1ccccc1", R_WILDCARD, "wildcard attachment point")
    expect_rejected("[CH3]", R_RADICAL, "methyl radical")
    expect_rejected("[O]c1ccccc1", R_RADICAL, "phenoxy radical")


def test_disallowed_elements() -> None:
    print("\n6. elements outside the organic set")
    expect_rejected("C[Si](C)(C)C", R_ELEMENT, "tetramethylsilane")
    expect_rejected("OB(O)c1ccccc1", R_ELEMENT, "phenylboronic acid")
    expect_rejected("C[Se]C", R_ELEMENT, "dimethyl selenide")
    expect_rejected("CC(=O)O[Sn](CCCC)(CCCC)CCCC", R_ELEMENT, "an organotin reagent")
    # Allowed heteroatoms must still pass.
    # Deliberately unrelated to the starting chemotype. An earlier version used a bromo-acetanilide
    # sulfonamide, which is the motif the grader's reference happens to use -- a test fixture is not
    # the place to hint at the answer, even accidentally.
    expect_accepted("Brc1ccc(CCN2CCOCC2)cc1", "an aryl bromide (Br is allowed)")
    expect_accepted("CP(=O)(OC)OC", "a phosphonate (P is allowed)")


def test_empty_and_nonstring() -> None:
    print("\n7. empty and non-string input")
    expect_rejected("", R_EMPTY, "empty string")
    expect_rejected("   ", R_EMPTY, "whitespace only")
    expect_rejected(None, R_NON_STRING, "None")
    expect_rejected(12345, R_NON_STRING, "an integer")
    expect_rejected(["CCO"], R_NON_STRING, "a list")
    expect_rejected("not_a_smiles_at_all", R_UNPARSEABLE, "arbitrary text")


def test_standardization_is_score_preserving() -> None:
    """An admissible molecule must score identically however its SMILES was written."""
    print("\n8. standardisation does not change the score of a legal molecule")
    variants = {
        "start, as configured": START,
        "start, kekulised": Chem.MolToSmiles(Chem.MolFromSmiles(START), kekuleSmiles=True,
                                             canonical=False) if Chem.MolFromSmiles(START) else START,
        "start, canonical round-trip": Chem.MolToSmiles(Chem.MolFromSmiles(START)),
        "start, written from the other end": "O=S(=O)(N(C)CCCN1CCCC1)c1cc(NC(C)=O)ccc1F",
        "start, with surrounding whitespace": f"  {START}  ",
    }
    baseline = evaluate_smiles(START, start_smiles=START, similarity_cfg=SIM_CFG)
    check(baseline.valid, "baseline start molecule is admissible")
    for label, smi in variants.items():
        ev = evaluate_smiles(smi, start_smiles=START, similarity_cfg=SIM_CFG)
        same = (
            ev.valid
            and ev.canonical_smiles == baseline.canonical_smiles
            and ev.qed == baseline.qed
            and ev.molecular_weight == baseline.molecular_weight
            and ev.logp == baseline.logp
            and ev.tpsa == baseline.tpsa
            and ev.sa_score == baseline.sa_score
            and ev.similarity_to_start == baseline.similarity_to_start
            and ev.n_alerts == baseline.n_alerts
        )
        check(same, f"identical score before/after standardisation: {label}")

    # Idempotence: standardising an already-standardised molecule changes nothing.
    once = validate_smiles_input(START)
    twice = validate_smiles_input(once.canonical_smiles)
    check(once.ok and twice.ok and once.canonical_smiles == twice.canonical_smiles,
          "standardisation is idempotent")

    # A different legal analogue, same invariance.
    other = "CC(=O)Nc1ccc(C)c(S(=O)(=O)N(C)CCCN2CCCC2)c1"
    a = evaluate_smiles(other, start_smiles=START, similarity_cfg=SIM_CFG)
    b = evaluate_smiles(Chem.MolToSmiles(Chem.MolFromSmiles(other)), start_smiles=START,
                        similarity_cfg=SIM_CFG)
    check(a.valid and b.valid and a.qed == b.qed and a.similarity_to_start == b.similarity_to_start,
          "a second legal analogue is also standardisation-invariant")


def test_boundary_structures() -> None:
    """Constraint windows are inclusive at both ends, and just-outside values are rejected."""
    print("\n9. constraint-window boundaries")
    mw = CONSTRAINTS["molecular_weight"]
    logp = CONSTRAINTS["logp"]
    tpsa = CONSTRAINTS["tpsa"]
    sa_max = CONSTRAINTS["sa_score"]["max"]
    sim_min = SIM_CFG["min"]

    def probe(**over):
        base = dict(
            smiles="synthetic", valid=True, canonical_smiles="synthetic",
            molecular_weight=(mw["min"] + mw["max"]) / 2,
            logp=(logp["min"] + logp["max"]) / 2,
            tpsa=(tpsa["min"] + tpsa["max"]) / 2,
            qed=0.5, sa_score=sa_max - 0.5,
            similarity_to_start=sim_min + 0.1, n_alerts=0, alerts=[],
        )
        base.update(over)
        return check_constraints(MoleculeEvaluation(**base), CONSTRAINTS)

    eps = 1e-9
    check(probe()["molecular_weight"], "mid-window molecule passes MW")
    check(probe(molecular_weight=mw["min"])["molecular_weight"], f"MW floor {mw['min']} is inclusive")
    check(probe(molecular_weight=mw["max"])["molecular_weight"], f"MW ceiling {mw['max']} is inclusive")
    check(not probe(molecular_weight=mw["min"] - eps)["molecular_weight"], "just below the MW floor fails")
    check(not probe(molecular_weight=mw["max"] + eps)["molecular_weight"], "just above the MW ceiling fails")
    check(probe(logp=logp["min"])["logp"], f"LogP floor {logp['min']} is inclusive")
    check(probe(logp=logp["max"])["logp"], f"LogP ceiling {logp['max']} is inclusive")
    check(not probe(logp=logp["max"] + eps)["logp"], "just above the LogP ceiling fails")
    check(probe(tpsa=tpsa["min"])["tpsa"], f"TPSA floor {tpsa['min']} is inclusive")
    check(probe(tpsa=tpsa["max"])["tpsa"], f"TPSA ceiling {tpsa['max']} is inclusive")
    check(not probe(tpsa=tpsa["min"] - eps)["tpsa"], "just below the TPSA floor fails")
    check(probe(sa_score=sa_max)["sa_score"], f"SA cap {sa_max} is inclusive")
    check(not probe(sa_score=sa_max + eps)["sa_score"], "just above the SA cap fails")
    check(probe(similarity_to_start=sim_min)["similarity"], f"similarity floor {sim_min} is inclusive")
    check(not probe(similarity_to_start=sim_min - eps)["similarity"], "just below the similarity floor fails")
    check(not probe(n_alerts=1)["no_structural_alerts"], "one structural alert fails")


def test_rejected_input_fails_every_flag() -> None:
    """A rejected input must not collect a single satisfied constraint anywhere."""
    print("\n10. a rejected input satisfies no constraint")
    for smi in ["CC(=O)[O-].[Na+]", "[Na+]", "CC(C)(C)(C)C", "", f"{START}.Cl"]:
        ev = evaluate_smiles(smi, start_smiles=START, similarity_cfg=SIM_CFG)
        flags = check_constraints(ev, CONSTRAINTS)
        check(not any(flags.values()),
              f"no constraint flag is True for {smi!r} (flags={sorted(k for k, v in flags.items() if v)})")
        check(bool(ev.rejection_reasons) or ev.error is not None,
              f"a machine-readable reason is recorded for {smi!r}")


def test_start_molecule_still_admissible() -> None:
    print("\n11. the configured start molecule is still admissible and feasible")
    ev = evaluate_smiles(START, start_smiles=START, similarity_cfg=SIM_CFG)
    check(ev.valid, "start molecule is admissible")
    flags = check_constraints(ev, CONSTRAINTS)
    for name, satisfied in flags.items():
        check(satisfied, f"start satisfies {name}")
    check(ev.similarity_to_start == 1.0, "start molecule is at similarity 1.0 to itself")


TESTS = [
    test_salts_and_multicomponent,
    test_inorganic_ions,
    test_charge,
    test_abnormal_valence,
    test_isotopes_maps_wildcards_radicals,
    test_disallowed_elements,
    test_empty_and_nonstring,
    test_standardization_is_score_preserving,
    test_boundary_structures,
    test_rejected_input_fails_every_flag,
    test_start_molecule_still_admissible,
]


def main() -> None:
    for fn in TESTS:
        fn()
    print(f"\n{_checks - len(_failures)}/{_checks} assertions passed "
          f"across {len(TESTS)} test groups")
    # Guard against a vacuous pass: if the suite silently stopped generating assertions, fail.
    if _checks < 80:
        print(f"FAILED: only {_checks} assertions ran; the suite is expected to run at least 80")
        raise SystemExit(1)
    if _failures:
        print("FAILED:")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all input-validation regression tests passed")


if __name__ == "__main__":
    main()

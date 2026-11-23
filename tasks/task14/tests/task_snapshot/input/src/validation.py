"""Input admissibility for a submitted candidate: exactly one neutral organic molecule.

Second-round review finding (田双燕 §2.3): the benchmark had no regression coverage for malformed
input, and the evaluator handed any RDKit-parseable string straight to the descriptor calls. That
accepted salts (`...[O-].[Na+]`), multi-component mixtures, bare inorganic ions, isotopically
labelled atoms and atom-mapped reaction fragments. All of those change molecular weight, LogP, TPSA,
QED and the fingerprint, so a candidate that is not a single synthesisable analogue could still be
measured as if it were one -- while `question.md` had been telling the model since the first release
that a salt, a mixture or a metal-containing compound is not an acceptable submission.

The task semantics are "one neutral organic analogue of the starting molecule", so that is now
enforced explicitly and the input is *rejected*, never silently repaired. In particular the largest
fragment is NOT extracted from a multi-component input: stripping a counter-ion would turn an
inadmissible submission into a scoring one. Standardisation is applied only to inputs that are
already admissible, so that two different SMILES strings for the same molecule are guaranteed to
score identically.

Every rejection carries a machine-readable reason code, and a rejection ends the round, never the
run: `tests/test_input_validation.py` pins the codes for salts, mixtures, ions, abnormal valences,
isotopes, atom maps and the boundary structures, and `tests/test_harness_robustness.py` pins that
none of them can abort the five-round loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Elements a small-molecule medicinal-chemistry analogue of the starting series may contain.
# Deliberately narrow: no metals, no metalloids, no selenium. Hydrogen is implicit but listed so an
# explicit-H input reports cleanly.
ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

# Reason codes. Stable strings -- the regression tests and the tool layer both assert on them.
R_NON_STRING = "non_string_input"
R_EMPTY = "empty_input"
R_UNPARSEABLE = "unparseable_smiles"
R_MULTI_FRAGMENT = "multiple_fragments"
R_NET_CHARGE = "net_charge_nonzero"
R_ELEMENT = "disallowed_element"
R_ISOTOPE = "isotope_label"
R_ATOM_MAP = "atom_map_number"
R_WILDCARD = "wildcard_atom"
R_NO_CARBON = "no_carbon"
R_RADICAL = "radical_electron"
R_STD_FAILED = "standardization_failed"
R_STD_FRAGMENTED = "fragments_after_standardization"
R_STD_CHARGED = "charge_after_standardization"

REASON_TEXT = {
    R_NON_STRING: "the submitted SMILES field is not a string",
    R_EMPTY: "the submitted SMILES field is empty",
    R_UNPARSEABLE: "RDKit could not parse the SMILES (syntax error or an impossible valence)",
    R_MULTI_FRAGMENT: "the input contains more than one disconnected component (salt, mixture or counter-ion); submit exactly one molecule",
    R_NET_CHARGE: "the net formal charge is not zero; submit a neutral molecule",
    R_ELEMENT: "the input contains an element outside the allowed organic set",
    R_ISOTOPE: "the input carries an isotope label",
    R_ATOM_MAP: "the input carries atom-map numbers",
    R_WILDCARD: "the input contains a wildcard/query atom",
    R_NO_CARBON: "the input contains no carbon atom, so it is not an organic analogue",
    R_RADICAL: "the input contains an unpaired electron (radical)",
    R_STD_FAILED: "standardization of the input failed",
    R_STD_FRAGMENTED: "standardization split the input into several components",
    R_STD_CHARGED: "standardization left a non-zero net formal charge",
}


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    detail: Optional[str] = None
    mol: Optional[Chem.Mol] = None
    canonical_smiles: Optional[str] = None
    standardized_smiles: Optional[str] = None
    n_fragments: Optional[int] = None
    net_charge: Optional[int] = None
    elements: List[str] = field(default_factory=list)

    def message(self) -> Optional[str]:
        if self.ok:
            return None
        parts = [REASON_TEXT.get(r, r) for r in self.reasons]
        if self.detail:
            parts.append(self.detail)
        return "Rejected input: " + "; ".join(parts)


def _standardize(mol: Chem.Mol) -> Chem.Mol:
    """Normalise functional-group drawing so equivalent SMILES score identically.

    `rdMolStandardize.Cleanup` runs sanitisation, functional-group normalisation (e.g. the several
    accepted ways of drawing a nitro or sulfonyl group), metal disconnection and re-ionisation. It is
    applied only after the admissibility checks, so it cannot rescue an inadmissible input -- and the
    post-conditions below re-assert single-fragment and neutrality in case normalisation moved
    something.
    """
    return rdMolStandardize.Cleanup(mol)


def validate_smiles_input(smiles) -> ValidationResult:
    """Admissibility gate: exactly one neutral organic molecule, then standardised."""
    if not isinstance(smiles, str):
        return ValidationResult(ok=False, reasons=[R_NON_STRING])
    text = smiles.strip()
    if not text:
        return ValidationResult(ok=False, reasons=[R_EMPTY])

    # sanitize=True is the default and is what rejects abnormal valences (pentavalent carbon,
    # tetravalent neutral nitrogen, hypervalent halogen ...): MolFromSmiles returns None.
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ValidationResult(ok=False, reasons=[R_UNPARSEABLE])

    reasons: List[str] = []
    details: List[str] = []

    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    n_fragments = len(frags)
    if n_fragments != 1:
        reasons.append(R_MULTI_FRAGMENT)
        details.append(f"{n_fragments} components")

    net_charge = Chem.GetFormalCharge(mol)
    if net_charge != 0:
        reasons.append(R_NET_CHARGE)
        details.append(f"net charge {net_charge:+d}")

    elements = sorted({a.GetSymbol() for a in mol.GetAtoms()})
    bad_elements = sorted(e for e in elements if e not in ALLOWED_ELEMENTS and e != "*")
    if bad_elements:
        reasons.append(R_ELEMENT)
        details.append("disallowed: " + ",".join(bad_elements))

    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        reasons.append(R_WILDCARD)
    if any(a.GetIsotope() != 0 for a in mol.GetAtoms()):
        reasons.append(R_ISOTOPE)
    if any(a.GetAtomMapNum() != 0 for a in mol.GetAtoms()):
        reasons.append(R_ATOM_MAP)
    if any(a.GetNumRadicalElectrons() != 0 for a in mol.GetAtoms()):
        reasons.append(R_RADICAL)
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        reasons.append(R_NO_CARBON)

    if reasons:
        return ValidationResult(
            ok=False,
            reasons=reasons,
            detail="; ".join(details) or None,
            n_fragments=n_fragments,
            net_charge=net_charge,
            elements=elements,
        )

    try:
        std = _standardize(mol)
        if std is None:
            raise ValueError("Cleanup returned None")
        Chem.SanitizeMol(std)
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult(ok=False, reasons=[R_STD_FAILED], detail=str(exc),
                                n_fragments=n_fragments, net_charge=net_charge, elements=elements)

    if len(Chem.GetMolFrags(std, asMols=False, sanitizeFrags=False)) != 1:
        return ValidationResult(ok=False, reasons=[R_STD_FRAGMENTED],
                                n_fragments=n_fragments, net_charge=net_charge, elements=elements)
    if Chem.GetFormalCharge(std) != 0:
        return ValidationResult(ok=False, reasons=[R_STD_CHARGED],
                                n_fragments=n_fragments, net_charge=net_charge, elements=elements)

    return ValidationResult(
        ok=True,
        mol=std,
        canonical_smiles=Chem.MolToSmiles(std, canonical=True),
        standardized_smiles=Chem.MolToSmiles(std, canonical=True),
        n_fragments=1,
        net_charge=0,
        elements=sorted({a.GetSymbol() for a in std.GetAtoms()}),
    )

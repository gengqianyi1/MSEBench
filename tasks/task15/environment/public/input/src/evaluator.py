from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem import RDConfig

from validation import validate_smiles_input

# Synthetic Accessibility scorer ships in RDKit Contrib, not on the default path.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402


# Build the combined structural-alert catalog once. PAINS + Brenk + NIH together
# cover promiscuous binders, reactive/toxic groups, and problematic functionality.
_ALERT_CATALOG_NAMES = ["PAINS", "BRENK", "NIH"]


def _build_alert_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    for name in _ALERT_CATALOG_NAMES:
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(params)


_ALERT_CATALOG = _build_alert_catalog()


@dataclass
class MoleculeEvaluation:
    smiles: str
    valid: bool
    canonical_smiles: Optional[str] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    qed: Optional[float] = None
    sa_score: Optional[float] = None
    similarity_to_start: Optional[float] = None
    n_alerts: Optional[int] = None
    alerts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Machine-readable admissibility codes from validation.py. Empty for an accepted molecule.
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _morgan_tanimoto_similarity(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp_a = generator.GetFingerprint(mol_a)
    fp_b = generator.GetFingerprint(mol_b)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def compute_similarity_to_start(
    candidate_mol: Chem.Mol,
    start_smiles: str,
    similarity_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    if not start_smiles:
        return None
    # The reference point goes through the same standardisation as the candidate, so the similarity
    # is computed between two canonical forms and cannot depend on how either was written.
    start_check = validate_smiles_input(start_smiles)
    if not start_check.ok:
        return None
    start_mol = start_check.mol
    similarity_cfg = similarity_cfg or {}
    radius = int(similarity_cfg.get("radius", 2))
    n_bits = int(similarity_cfg.get("n_bits", 2048))
    return round(
        _morgan_tanimoto_similarity(candidate_mol, start_mol, radius=radius, n_bits=n_bits),
        6,
    )


def _list_alerts(mol: Chem.Mol) -> List[str]:
    """Return the descriptions of all matched structural alerts."""
    matches = _ALERT_CATALOG.GetMatches(mol)
    descriptions = []
    for entry in matches:
        desc = entry.GetDescription() or "unnamed_alert"
        descriptions.append(desc)
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for d in descriptions:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def evaluate_smiles(
    smiles: str,
    start_smiles: Optional[str] = None,
    similarity_cfg: Optional[Dict[str, Any]] = None,
) -> MoleculeEvaluation:
    """Evaluate a molecule against the full multi-objective descriptor set.

    Admissibility comes first: anything that is not a single neutral organic molecule is rejected
    outright (see validation.py) and never reaches a descriptor call, so a salt or a mixture cannot
    be measured as though it were a candidate analogue. Descriptors are then computed on the
    STANDARDISED molecule, which makes the result independent of how an admissible molecule was
    written.
    """
    check = validate_smiles_input(smiles)
    if not check.ok:
        return MoleculeEvaluation(
            smiles=smiles if isinstance(smiles, str) else str(smiles),
            valid=False,
            error=check.message(),
            rejection_reasons=list(check.reasons),
        )

    smiles = smiles.strip()

    try:
        mol = check.mol
        canonical = check.canonical_smiles
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        qed_value = QED.qed(mol)
        sa_value = sascorer.calculateScore(mol)
        alerts = _list_alerts(mol)

        similarity = None
        if start_smiles:
            similarity = compute_similarity_to_start(
                candidate_mol=mol,
                start_smiles=start_smiles,
                similarity_cfg=similarity_cfg,
            )

        return MoleculeEvaluation(
            smiles=smiles,
            valid=True,
            canonical_smiles=canonical,
            molecular_weight=round(float(mw), 4),
            logp=round(float(logp), 4),
            tpsa=round(float(tpsa), 4),
            qed=round(float(qed_value), 6),
            sa_score=round(float(sa_value), 4),
            similarity_to_start=similarity,
            n_alerts=len(alerts),
            alerts=alerts,
        )
    except Exception as exc:
        return MoleculeEvaluation(smiles=smiles, valid=False, error=str(exc))


def check_constraints(evaluation: MoleculeEvaluation, constraints: Dict[str, Any]) -> Dict[str, bool]:
    """Return one boolean flag per hard constraint declared in the task config.

    A constraint key that is absent from the config is simply not checked, so a task can declare
    only the windows its mechanism needs instead of carrying a placeholder window that constrains
    nothing. `tpsa` is optional for exactly that reason.
    """
    flags = {
        "valid_smiles": bool(evaluation.valid),
        "molecular_weight": False,
        "logp": False,
        "sa_score": False,
        "no_structural_alerts": False,
        "similarity": False,
    }
    tpsa_cfg = constraints.get("tpsa")
    if tpsa_cfg is not None:
        flags["tpsa"] = False

    if not evaluation.valid:
        return flags

    mw_cfg = constraints["molecular_weight"]
    logp_cfg = constraints["logp"]
    sa_cfg = constraints["sa_score"]
    sim_cfg = constraints.get("similarity")

    flags["molecular_weight"] = mw_cfg["min"] <= evaluation.molecular_weight <= mw_cfg["max"]
    flags["logp"] = logp_cfg["min"] <= evaluation.logp <= logp_cfg["max"]
    if tpsa_cfg is not None:
        flags["tpsa"] = tpsa_cfg["min"] <= evaluation.tpsa <= tpsa_cfg["max"]
    flags["sa_score"] = evaluation.sa_score is not None and evaluation.sa_score <= sa_cfg["max"]
    flags["no_structural_alerts"] = (evaluation.n_alerts is not None and evaluation.n_alerts == 0)

    if sim_cfg is None:
        flags["similarity"] = True
    else:
        flags["similarity"] = (
            evaluation.similarity_to_start is not None
            and evaluation.similarity_to_start >= sim_cfg["min"]
        )

    return flags


def satisfies_all_constraints(evaluation: MoleculeEvaluation, constraints: Dict[str, Any]) -> bool:
    flags = check_constraints(evaluation, constraints)
    return all(flags.values())

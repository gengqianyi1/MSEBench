#!/usr/bin/env python
"""Evaluator for lithium electrolyte solvent design."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

TASK_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TASK_DIR.parents[1]
sys.path.insert(0, str(TASK_DIR.parents[0]))


ALLOWED_ELEMENTS = {"C", "H", "O", "F"}
BANNED_CANONICAL = set()

FORBIDDEN_SMARTS = {
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "peroxide": "[OX2][OX2]",
    "acid_halide": "[CX3](=O)[F,Cl,Br,I]",
    "anhydride": "[CX3](=O)O[CX3](=O)",
}


def load_smiles(argv: List[str]) -> str:
    if len(argv) < 2:
        return json.load(sys.stdin)["smiles"]
    arg = argv[1]
    path = Path(arg)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["smiles"]
    return arg


def count_atoms(mol: Chem.Mol, symbol: str) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)


def has_smarts(mol: Chem.Mol, smarts: str) -> bool:
    patt = Chem.MolFromSmarts(smarts)
    return bool(patt is not None and mol.HasSubstructMatch(patt))


def sa_score(mol: Chem.Mol) -> float:
    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    hetero = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"})
    chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    fluorines = count_atoms(mol, "F")
    value = 1.2 + 0.06 * heavy + 0.35 * rings + 0.18 * max(0, hetero - 5)
    value += 0.25 * chiral + 0.45 * (spiro + bridge) + 0.10 * max(0, fluorines - 3)
    return round(value, 3)


def descriptor_surrogate(smiles: str) -> Dict[str, Any]:
    """Fixed task-internal descriptor; it is not an orbital-energy calculation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"value": None, "ok": False, "method": "rdkit_descriptor_surrogate_v2"}
    logp = Crippen.MolLogP(mol)
    o_count = count_atoms(mol, "O")
    f_count = count_atoms(mol, "F")
    rotors = Descriptors.NumRotatableBonds(mol)
    value = -6.92 + 0.055 * (o_count - 4) - 0.035 * f_count + 0.025 * (rotors - 4) + 0.035 * logp
    return {"value": round(float(value), 4), "ok": True, "method": "rdkit_descriptor_surrogate_v2"}


def has_three_oxygen_substituents(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        oxygen_neighbors = sum(neighbor.GetSymbol() == "O" for neighbor in atom.GetNeighbors())
        if oxygen_neighbors >= 3:
            return True
    return False


def item(value: Any, passed: bool, target: str, tool: str, points: float, tool_available: bool = True) -> Dict[str, Any]:
    return {
        "value": value,
        "passed": bool(passed),
        "target": target,
        "tool": tool,
        "tool_available": bool(tool_available),
        "points_awarded": points if passed else 0.0,
        "points_possible": points,
    }


def hard_fail(smiles: str) -> Dict[str, Any]:
    feedback = {
        "candidate": smiles,
        "valid": False,
        "feedback_mode": "rule_evaluation",
        "constraints": {
            "C0_valid_smiles": {
                "rule": "RDKit must parse and sanitize the submitted SMILES.",
                "value": False,
                "passed": False,
            }
        },
        "score": 0.0,
        "all_constraints_met": False,
    }
    return {
        "candidate": smiles,
        "valid": False,
        "score": 0.0,
        "all_constraints_met": False,
        "constraint_results": feedback["constraints"],
        "properties": {},
        "agent_feedback": feedback,
        "failure_analysis": {"failure_modes": []},
    }


def evaluate(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return hard_fail(smiles)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return hard_fail(smiles)
    if mol.GetNumAtoms() == 0:
        return hard_fail(smiles)

    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    heavy = mol.GetNumHeavyAtoms()
    illegal = [el for el in elements if el not in ALLOWED_ELEMENTS]
    canonical = Chem.MolToSmiles(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    fragment_count = len(Chem.GetMolFrags(mol))

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    o_count = count_atoms(mol, "O")
    f_count = count_atoms(mol, "F")
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    forbidden = [name for name, smarts in FORBIDDEN_SMARTS.items() if has_smarts(mol, smarts)]
    if has_three_oxygen_substituents(mol):
        forbidden.append("carbon_with_three_or_more_oxygen_neighbors")
    sa = sa_score(mol)
    homo = descriptor_surrogate(smiles)

    checks: Dict[str, Dict[str, Any]] = {}
    c0_value = {
        "elements": elements,
        "illegal_elements": illegal,
        "formal_charge": formal_charge,
        "heavy_atoms": heavy,
        "MW": round(mw, 3),
        "fragment_count": fragment_count,
        "radical_electrons": radical_electrons,
    }
    c0 = (
        not illegal
        and formal_charge == 0
        and 7 <= heavy <= 16
        and mw < 250.0
        and fragment_count == 1
        and radical_electrons == 0
    )
    checks["C0_search_space"] = item(
        c0_value,
        c0,
        "allowed elements in {C,H,O,F}; formal_charge == 0; 7 <= heavy_atoms <= 16; MW < 250; fragment_count == 1; radical_electrons == 0",
        "RDKit",
        0.0,
    )

    c1 = homo["value"] is not None and -7.20 < homo["value"] < -6.35
    checks["C1"] = item(homo["value"], c1, "-7.20 < descriptor_surrogate < -6.35", homo["method"], 20.0, homo["ok"])

    c2_value = {"MW": round(mw, 3), "logP": round(logp, 3)}
    c2 = 155.0 <= mw <= 210.0 and -0.2 <= logp <= 0.9
    checks["C2"] = item(c2_value, c2, "155 <= MW <= 210 and -0.2 <= logP <= 0.9", "RDKit", 20.0)

    c3 = 4 <= rotors <= 5
    checks["C3"] = item(rotors, c3, "4 <= rotatable_bonds <= 5", "RDKit", 20.0)

    c4_value = {"O_count": o_count, "HBD_count": hbd}
    c4 = 4 <= o_count <= 5 and hbd == 0
    checks["C4"] = item(c4_value, c4, "4 <= O_count <= 5 and HBD_count == 0", "RDKit", 20.0)

    c5_value = {"SA_score": sa, "F_count": f_count, "ring_count": rings, "is_aromatic": aromatic, "radical_electrons": radical_electrons, "forbidden_groups": forbidden}
    c5 = sa < 2.8 and f_count <= 2 and rings == 0 and not aromatic and radical_electrons == 0 and not forbidden
    checks["C5"] = item(c5_value, c5, "SA_score < 2.8; F_count <= 2; ring_count == 0; is_aromatic == false; radical_electrons == 0; forbidden_SMARTS_count == 0, including no orthoester/acetal center", "RDKit + SMARTS", 20.0)

    checks["C6_leakage_control"] = item(
        {"canonical_smiles": canonical, "banned": canonical in BANNED_CANONICAL},
        canonical not in BANNED_CANONICAL,
        "canonical_smiles must not match reserved leakage-control examples",
        "RDKit canonical SMILES",
        0.0,
    )

    hard_gates_passed = bool(c0 and checks["C6_leakage_control"]["passed"])
    score = round(sum(check["points_awarded"] for check in checks.values()), 2) if hard_gates_passed else 0.0
    failed = [cid for cid, check in checks.items() if not check["passed"]]
    all_constraints_met = not failed
    properties = {
        "HOMO_eV": homo["value"],
        "homo_method": homo.get("method"),
        "MW": round(mw, 3),
        "logP": round(logp, 3),
        "rotatable_bonds": rotors,
        "O_count": o_count,
        "F_count": f_count,
        "HBD_count": hbd,
        "SA_score": sa,
        "ring_count": rings,
        "is_aromatic": aromatic,
        "radical_electrons": radical_electrons,
        "forbidden_groups": forbidden,
        "heavy_atoms": heavy,
        "fragment_count": fragment_count,
        "elements": elements,
    }
    feedback_constraints = {
        cid: {
            "rule": check["target"],
            "value": check["value"],
            "passed": check["passed"],
        }
        for cid, check in checks.items()
    }
    return {
        "candidate": smiles,
        "valid": True,
        "score": score,
        "all_constraints_met": all_constraints_met,
        "constraint_results": checks,
        "properties": properties,
        "agent_feedback": {
            "candidate": smiles,
            "valid": True,
            "feedback_mode": "rule_evaluation",
            "constraints": feedback_constraints,
            "score": score,
            "all_constraints_met": all_constraints_met,
        },
        "failure_analysis": {"failure_modes": []},
    }


def main() -> None:
    print(json.dumps(evaluate(load_smiles(sys.argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

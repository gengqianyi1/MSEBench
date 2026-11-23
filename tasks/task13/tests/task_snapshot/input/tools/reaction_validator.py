#!/usr/bin/env python
"""Public, reference-free reaction/route validator for Task 13.

Usage: python input/tools/reaction_validator.py route.json
The JSON file may be a route array or an object with top-level ``route``.
This module is also the chemistry authority imported by the private grader.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


REACTION_LIBRARY: dict[str, str] = {
    "suzuki_coupling": "[c:1][Br,I].[c:2][B]([OH])[OH]>>[c:1][c:2]",
    "ester_hydrolysis": "[C:1](=[O:2])[OX2][CH3]>>[C:1](=[O:2])[OX2H]",
    "amide_coupling": "[C:1](=[O:2])[OX2H1].[NX3;H2,H1;!$(NC=O):3]>>[C:1](=[O:2])[N:3]",
    "n_acylation": "[NX3;H2,H1;!$(NC=O):1].[C:2](=[O:3])[Cl]>>[N:1][C:2]=[O:3]",
    "sn2_n_alkylation": "[NX3;H2,H1;!$(NC=O):1].[C;X4:2][Cl,Br,I]>>[N:1][C:2]",
    "fischer_esterification": "[C:1](=[O:2])[OX2H].[OX2H1:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
    "sulfonamide_formation": "[S:1](=[O:2])(=[O:3])[Cl].[NX3;H2,H1:4]>>[S:1](=[O:2])(=[O:3])[N:4]",
    "reductive_amination": "[C:1]=[O:2].[NX3;H2,H1;!$(NC=O):3]>>[C:1][N:3]",
}

INVENTORY = [
    "COC(=O)c1cc(Br)cc(Br)c1", "Cc1ccc(B(O)O)cc1", "OB(O)c1ccccc1", "OCCBr",
    "NCc1ccccc1", "CC(=O)Cl", "COC(=O)c1ccc(Br)cc1", "COC(=O)c1cccc(Br)c1",
    "COC(=O)c1ccccc1", "COC(=O)c1ccc(Br)c(Br)c1", "COC(=O)c1c(Br)cccc1Br",
    "Cc1cccc(B(O)O)c1", "Cc1ccccc1B(O)O", "COc1ccc(B(O)O)cc1", "Fc1ccc(B(O)O)cc1",
    "Clc1ccc(B(O)O)cc1", "Nc1ccccc1", "NCCc1ccccc1", "CNc1ccccc1", "ClCc1ccccc1",
    "BrCCBr", "CI", "CCO", "CC(=O)Br", "O=C(O)c1ccccc1",
]


def normalize_reaction_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s-]+", "_", value.strip().casefold())


def canonical(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else None


def mol_weight(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return round(float(Descriptors.MolWt(mol)), 4) if mol is not None else None


def run_named_reaction(reaction_name: str, reactant_smiles: list[str]) -> tuple[set[str], str | None]:
    normalized = normalize_reaction_name(reaction_name)
    if normalized not in REACTION_LIBRARY:
        return set(), f"unknown_reaction:{reaction_name}"
    if not isinstance(reactant_smiles, list) or not reactant_smiles:
        return set(), "no_reactants"
    mols = [Chem.MolFromSmiles(value) if isinstance(value, str) else None for value in reactant_smiles]
    if any(mol is None for mol in mols):
        return set(), "invalid_reactant_smiles"
    reaction = AllChem.ReactionFromSmarts(REACTION_LIBRARY[normalized])
    if len(mols) != reaction.GetNumReactantTemplates():
        return set(), f"reactant_count_mismatch:need_{reaction.GetNumReactantTemplates()}_got_{len(mols)}"
    products: set[str] = set()
    for ordered in permutations(mols):
        try:
            product_sets = reaction.RunReactants(ordered)
        except Exception:
            continue
        for product_set in product_sets:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    products.add(Chem.MolToSmiles(product, canonical=True, isomericSmiles=True))
                except Exception:
                    continue
    return products, None


def step_produces(reaction_name: str, reactant_smiles: list[str], declared_product: str) -> dict[str, Any]:
    products, error = run_named_reaction(reaction_name, reactant_smiles)
    declared = canonical(declared_product)
    return {
        "reaction": normalize_reaction_name(reaction_name),
        "declared_product_canonical": declared,
        "fired": bool(products),
        "matched": declared is not None and declared in products,
        "n_products": len(products),
        "products": sorted(products),
        "error": error,
    }


def _heavy_atom_counts(smiles: str) -> Counter | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def check_atom_conservation(reactant_smiles: list[str], product_smiles: str) -> dict[str, Any]:
    product_counts = _heavy_atom_counts(product_smiles)
    if product_counts is None:
        return {"conserved": False, "phantom_atoms": {}, "error": "invalid_product_smiles"}
    reactant_counts: Counter = Counter()
    for reactant in reactant_smiles:
        counts = _heavy_atom_counts(reactant)
        if counts is None:
            return {"conserved": False, "phantom_atoms": {}, "error": "invalid_reactant_smiles"}
        reactant_counts += counts
    phantom = {element: count - reactant_counts.get(element, 0) for element, count in product_counts.items() if count > reactant_counts.get(element, 0)}
    return {
        "conserved": not phantom,
        "phantom_atoms": phantom,
        "reactant_heavy": dict(reactant_counts),
        "product_heavy": dict(product_counts),
        "error": None,
    }


def atom_economy(reactant_smiles: list[str], product_smiles: str) -> float | None:
    product_mw = mol_weight(product_smiles)
    reactant_mw = [mol_weight(value) for value in reactant_smiles]
    if product_mw is None or any(value is None for value in reactant_mw) or not sum(reactant_mw):
        return None
    return round(product_mw / sum(reactant_mw), 6)


def is_purchasable(smiles: str, inventory_canonical: set[str]) -> bool:
    value = canonical(smiles)
    return value is not None and value in inventory_canonical


def build_inventory_set(inventory_smiles: list[str]) -> set[str]:
    return {value for item in inventory_smiles if (value := canonical(item)) is not None}


def validate_route(route: Any) -> dict[str, Any]:
    if not isinstance(route, list):
        return {"parsed": False, "valid_process": False, "error": "route must be an array", "steps": []}
    available = build_inventory_set(INVENTORY)
    results = []
    for index, step in enumerate(route, start=1):
        reaction = step.get("reaction") if isinstance(step, dict) else None
        reactants = step.get("reactants") if isinstance(step, dict) else []
        product = step.get("product") if isinstance(step, dict) else None
        reactants = reactants if isinstance(reactants, list) else []
        reaction_result = step_produces(reaction, reactants, product) if product else {"reaction": normalize_reaction_name(reaction), "fired": False, "matched": False, "products": [], "error": "missing_product"}
        unavailable = [value for value in reactants if canonical(value) not in available]
        conservation = check_atom_conservation(reactants, product) if product and reactants else {"conserved": False, "phantom_atoms": {}}
        economy = atom_economy(reactants, product) if product and reactants else None
        result = {
            "index": index,
            "reaction": reaction_result["reaction"],
            "reaction_in_library": reaction_result["reaction"] in REACTION_LIBRARY,
            "reactants_available": not unavailable and bool(reactants),
            "unavailable_reactants": unavailable,
            "fired": reaction_result["fired"],
            "product_matches": reaction_result["matched"],
            "produced_products": reaction_result.get("products", []),
            "atom_conservation_ok": conservation["conserved"],
            "phantom_atoms": conservation.get("phantom_atoms", {}),
            "atom_economy": economy,
            "atom_economy_ok": economy is not None and economy >= 0.50,
            "error": reaction_result.get("error"),
        }
        declared = canonical(product) if product else None
        if result["product_matches"] and declared is not None:
            available.add(declared)
        results.append(result)
    keys = ("reaction_in_library", "reactants_available", "fired", "product_matches", "atom_conservation_ok", "atom_economy_ok")
    return {
        "parsed": True,
        "n_steps": len(route),
        "n_steps_within_limit": 2 <= len(route) <= 6,
        "valid_process": 2 <= len(route) <= 6 and all(all(step[key] for key in keys) for step in results),
        "steps": results,
        "note": "This public validator does not reveal or test the private reference route. The private grader additionally checks final-target equality and assigns points.",
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python input/tools/reaction_validator.py ROUTE_JSON")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    route = payload.get("route") if isinstance(payload, dict) else payload
    print(json.dumps(validate_route(route), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

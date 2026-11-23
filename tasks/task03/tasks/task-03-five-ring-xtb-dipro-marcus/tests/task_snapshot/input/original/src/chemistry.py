from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except Exception as exc:  # pragma: no cover
    Chem = None
    AllChem = None
    Descriptors = None

from geometry import XYZ, write_xyz


@dataclass
class RDKitValidation:
    smiles: str
    valid_smiles: bool
    canonical_smiles: Optional[str] = None
    single_component: bool = False
    neutral: bool = False
    allowed_elements: bool = False
    no_wildcards: bool = False
    ring_count_total: Optional[int] = None
    fused_ring_cluster_sizes: Optional[List[int]] = None
    largest_fused_ring_cluster_size: Optional[int] = None
    exactly_one_fused_ring_cluster: bool = False
    no_extra_rings: bool = False
    five_ring_fused_core: bool = False
    heavy_atom_count: Optional[int] = None
    molecular_weight: Optional[float] = None
    size_range: bool = False
    constraint_flags: Optional[Dict[str, bool]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require_rdkit():
    if Chem is None:
        raise RuntimeError("RDKit is not installed or could not be imported.")


def _has_query_or_wildcard(mol) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return True
        if atom.HasQuery():
            return True
    return False


def _ring_clusters(atom_rings: List[tuple], shared_atom_threshold: int = 2) -> List[Set[int]]:
    n = len(atom_rings)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ring_sets = [set(r) for r in atom_rings]
    for i in range(n):
        for j in range(i + 1, n):
            if len(ring_sets[i].intersection(ring_sets[j])) >= shared_atom_threshold:
                union(i, j)

    comps: Dict[int, Set[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), set()).add(i)
    return list(comps.values())


def validate_smiles_rdkit(
    smiles: str,
    constraints: Dict[str, Any],
) -> RDKitValidation:
    _require_rdkit()

    allowed = set(constraints["allowed_elements"])
    shared_threshold = int(constraints.get("fused_ring_shared_atom_threshold", 2))
    mw_cfg = constraints["molecular_weight"]
    ha_cfg = constraints["heavy_atom_count"]
    mw_min = mw_cfg.get("min")
    mw_max = mw_cfg.get("max")
    mw_max_exclusive = mw_cfg.get("max_exclusive")
    ha_min = ha_cfg.get("min")
    ha_max = ha_cfg.get("max")
    ha_max_exclusive = ha_cfg.get("max_exclusive")

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return RDKitValidation(smiles=smiles, valid_smiles=False, error=str(exc))

    if mol is None:
        return RDKitValidation(smiles=smiles, valid_smiles=False, error="RDKit could not parse SMILES.")

    canonical = Chem.MolToSmiles(mol, canonical=True)
    fragments = Chem.GetMolFrags(mol, asMols=False)
    single_component = len(fragments) == 1
    formal_charge = Chem.GetFormalCharge(mol)
    neutral = formal_charge == int(constraints.get("formal_charge", 0))
    no_wildcards = not _has_query_or_wildcard(mol)

    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    allowed_elements = all(sym in allowed for sym in atom_symbols)

    ring_info = mol.GetRingInfo()
    atom_rings = list(ring_info.AtomRings())
    ring_count_total = len(atom_rings)
    clusters = _ring_clusters(atom_rings, shared_atom_threshold=shared_threshold) if atom_rings else []
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
    largest = cluster_sizes[0] if cluster_sizes else 0

    exactly_one = len(clusters) == 1
    no_extra = ring_count_total == largest
    five_core = largest == int(constraints["largest_fused_ring_cluster_size"]) and exactly_one and no_extra

    heavy_atom_count = mol.GetNumHeavyAtoms()
    molecular_weight = float(Descriptors.MolWt(mol))
    size_range = True
    if ha_min is not None:
        size_range = size_range and heavy_atom_count >= int(ha_min)
    if ha_max is not None:
        size_range = size_range and heavy_atom_count <= int(ha_max)
    if ha_max_exclusive is not None:
        size_range = size_range and heavy_atom_count < int(ha_max_exclusive)
    if mw_min is not None:
        size_range = size_range and molecular_weight >= float(mw_min)
    if mw_max is not None:
        size_range = size_range and molecular_weight <= float(mw_max)
    if mw_max_exclusive is not None:
        size_range = size_range and molecular_weight < float(mw_max_exclusive)

    flags = {
        "valid_smiles": True,
        "single_component": single_component,
        "neutral": neutral,
        "allowed_elements": allowed_elements,
        "no_wildcards": no_wildcards,
        "five_ring_fused_core": five_core,
        "no_extra_rings": no_extra,
        "size_range": size_range,
    }

    return RDKitValidation(
        smiles=smiles,
        valid_smiles=True,
        canonical_smiles=canonical,
        single_component=single_component,
        neutral=neutral,
        allowed_elements=allowed_elements,
        no_wildcards=no_wildcards,
        ring_count_total=ring_count_total,
        fused_ring_cluster_sizes=cluster_sizes,
        largest_fused_ring_cluster_size=largest,
        exactly_one_fused_ring_cluster=exactly_one,
        no_extra_rings=no_extra,
        five_ring_fused_core=five_core,
        heavy_atom_count=heavy_atom_count,
        molecular_weight=round(molecular_weight, 6),
        size_range=size_range,
        constraint_flags=flags,
    )


def rdkit_embed_to_xyz(
    smiles: str,
    xyz_path: str | Path,
    rdkit_cfg: Dict[str, Any],
) -> None:
    _require_rdkit()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError("Cannot embed invalid SMILES.")

    mol = Chem.AddHs(mol)
    seed = int(rdkit_cfg.get("embedding_random_seed", 61453))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed

    num_conformers = max(1, int(rdkit_cfg.get("num_conformers", 1)))
    if num_conformers == 1:
        cids = [AllChem.EmbedMolecule(mol, params)]
        if cids[0] < 0:
            raise RuntimeError(f"RDKit EmbedMolecule failed with code {cids[0]}.")
    else:
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))
        if not cids:
            raise RuntimeError("RDKit EmbedMultipleConfs failed to generate conformers.")

    max_iters = int(rdkit_cfg.get("max_optimization_iters", 500))
    ff = rdkit_cfg.get("forcefield", "MMFF").upper()
    fallback = rdkit_cfg.get("fallback_forcefield", "UFF").upper()
    selected_cid = cids[0]

    try:
        if ff == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
            selected_cid = cids[min(range(len(results)), key=lambda i: results[i][1])]
        elif ff == "UFF":
            results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
            selected_cid = cids[min(range(len(results)), key=lambda i: results[i][1])]
        else:
            raise RuntimeError("Requested MMFF but parameters unavailable.")
    except Exception:
        if fallback == "UFF":
            results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
            selected_cid = cids[min(range(len(results)), key=lambda i: results[i][1])]
        else:
            raise

    conf = mol.GetConformer(int(selected_cid))
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append([p.x, p.y, p.z])

    import numpy as np
    xyz = XYZ(
        elements=elements,
        coords=np.array(coords, dtype=float),
        comment=f"RDKit embedded {smiles} | conformer {selected_cid} of {num_conformers}",
    )
    write_xyz(xyz_path, xyz)

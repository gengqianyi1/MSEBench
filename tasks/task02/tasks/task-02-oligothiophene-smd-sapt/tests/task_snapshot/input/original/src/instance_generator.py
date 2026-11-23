from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

from utils import write_json


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _construct_capped_oligomer(repeat_smiles: str, length: int):
    from rdkit import Chem

    if length != 2:
        raise ValueError("The bundled V2 generator currently supports oligomer_length=2.")
    units = []
    for _ in range(length):
        mol = Chem.MolFromSmiles(repeat_smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"Could not parse repeat unit: {repeat_smiles}")
        units.append(mol)

    combined = Chem.CombineMols(units[0], units[1])
    rw = Chem.RWMol(combined)
    offset = units[0].GetNumAtoms()

    def dummy_and_neighbor(mol, isotope: int, atom_offset: int = 0) -> Tuple[int, int]:
        matches = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0 and a.GetIsotope() == isotope]
        if len(matches) != 1:
            raise ValueError(f"Expected one isotope-{isotope} dummy per repeat; found {len(matches)}")
        dummy = matches[0]
        nbrs = list(dummy.GetNeighbors())
        if len(nbrs) != 1:
            raise ValueError("Each connection dummy must have exactly one neighbor")
        return dummy.GetIdx() + atom_offset, nbrs[0].GetIdx() + atom_offset

    # Required rule: next unit's 2* neighbor bonds to previous unit's 1* neighbor.
    _, previous_1_neighbor = dummy_and_neighbor(units[0], 1, 0)
    _, next_2_neighbor = dummy_and_neighbor(units[1], 2, offset)
    rw.AddBond(previous_1_neighbor, next_2_neighbor, Chem.BondType.SINGLE)

    dummy_indices = sorted(
        [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True
    )
    for idx in dummy_indices:
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    return mol


def _write_structure_files(mol, evidence_dir: Path, seed: int) -> Dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed & 0x7FFFFFFF)
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("RDKit embedding failed while preparing seed evidence")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        pass

    sdf_path = evidence_dir / "provided_structure.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    xyz_path = evidence_dir / "provided_structure.xyz"
    Chem.MolToXYZFile(mol, str(xyz_path))

    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return {
        "sdf": sdf_path,
        "xyz": xyz_path,
        "canonical_smiles": canonical,
        "formula": formula,
        "explicit_atom_count": mol.GetNumAtoms(),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "molecular_weight": Descriptors.MolWt(mol),
    }


def prepare_runtime_instance(task_cfg: Dict[str, Any], workspace: Path, seed_override: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed_cfg = int(task_cfg.get("instance_generation", {}).get("seed", 20260723))
    seed = int(seed_override if seed_override is not None else seed_cfg)
    rng = random.Random(seed)

    evidence_dir = workspace / "seed_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    molecule_cfg = task_cfg["molecule"]
    mol = _construct_capped_oligomer(molecule_cfg["repeat_unit_smiles"], int(molecule_cfg["oligomer_length"]))
    struct = _write_structure_files(mol, evidence_dir, seed)

    structure_metadata = {
        "artifact": "provided_structure.sdf",
        "sha256": sha256_file(struct["sdf"]),
        "xyz_sha256": sha256_file(struct["xyz"]),
        "formula": struct["formula"],
        "canonical_smiles": struct["canonical_smiles"],
        "explicit_atom_count": struct["explicit_atom_count"],
        "heavy_atom_count": struct["heavy_atom_count"],
        "charge": int(molecule_cfg["charge"]),
        "multiplicity": int(molecule_cfg["multiplicity"]),
        "note": "Supplied structure is a convenience artifact and must be independently audited.",
    }
    solvent_probe = {
        "artifact": "solvent_probe.json",
        "solvent_name": "water",
        "smiles": task_cfg["solvents"]["water"]["smiles"],
        "smd_name": task_cfg["solvents"]["water"]["smd_name"],
        "xtb_alpb_name": task_cfg["solvents"]["water"]["xtb_alpb_name"],
    }
    environment_note = {
        "artifact": "environment_note.json",
        "required_executables": ["python", "xtb", "psi4"],
        "required_python_modules": ["rdkit", "pyscf"],
        "claim": "Availability must be confirmed by executable micro-tests, not by this note.",
    }

    generation = task_cfg.get("instance_generation", {})
    structure_kind = rng.choice(list(generation.get("structure_anomaly_types") or ["stale_hash", "atom_count_mismatch"]))
    solvent_kind = rng.choice(list(generation.get("solvent_anomaly_types") or ["solvent_label_mismatch", "solvent_alias_mismatch"]))
    anomalies: list[dict[str, Any]] = []

    if structure_kind == "stale_hash":
        structure_metadata["sha256"] = "0" * 64
        structure_reason = "The recorded SDF SHA-256 does not match the supplied SDF bytes."
    elif structure_kind == "atom_count_mismatch":
        structure_metadata["explicit_atom_count"] = int(structure_metadata["explicit_atom_count"]) + 2
        structure_reason = "The recorded explicit atom count does not match the supplied SDF."
    else:
        raise ValueError(f"Unsupported structure anomaly type: {structure_kind}")
    anomalies.append({"artifact":"seed_evidence/structure_metadata.json","anomaly_type":structure_kind,"anomaly_reason":structure_reason})

    if solvent_kind == "solvent_label_mismatch":
        solvent_probe["solvent_name"] = "n-hexane"
        solvent_reason = "The solvent label conflicts with its water SMILES and program aliases."
    elif solvent_kind == "solvent_alias_mismatch":
        solvent_probe["xtb_alpb_name"] = "hexane"
        solvent_reason = "The ALPB alias conflicts with the water SMILES, label, and SMD name."
    else:
        raise ValueError(f"Unsupported solvent anomaly type: {solvent_kind}")
    anomalies.append({"artifact":"seed_evidence/solvent_probe.json","anomaly_type":solvent_kind,"anomaly_reason":solvent_reason})

    write_json(evidence_dir / "structure_metadata.json", structure_metadata)
    write_json(evidence_dir / "solvent_probe.json", solvent_probe)
    write_json(evidence_dir / "environment_note.json", environment_note)
    (evidence_dir / "README.txt").write_text(
        "Audit every artifact in this directory. One or more artifacts may contain independently detectable inconsistencies; the number is not disclosed.\n",
        encoding="utf-8",
    )

    conformer_seeds = rng.sample(range(1000, 9999), 6)
    torsion_starts = sorted(rng.sample([-165, -135, -95, -55, -20, 20, 55, 95, 135, 165], 6))
    public_instance: Dict[str, Any] = {
        "instance_id": f"oligothiophene-v3-0-{seed}",
        "instance_seed": seed,
        "benchmark_version": str(task_cfg.get("benchmark_version", "1.0")),
        "scoring_policy_id": task_cfg.get("scoring_policy_id"),
        "molecule": molecule_cfg,
        "solvents": task_cfg["solvents"],
        "methods": task_cfg["methods"],
        "constants": task_cfg["constants"],
        "compute_budget": {
            "enabled": bool(task_cfg.get("tool_call_budget_enabled", False)),
            "preferred_tool_calls": task_cfg.get("preferred_tool_call_budget"),
            "max_tool_calls": task_cfg.get("fixed_tool_call_budget"),
            "tool_call_penalties": {},
            **task_cfg.get("resource_guidance", {}),
        },
        "gas_conformer_seeds": conformer_seeds,
        "inter_repeat_torsion_starts_deg": torsion_starts,
        "sensitivity_options": task_cfg.get("sensitivity_options", []),
        "coordinate_hash_spec": {
            "record": "atomic_number,x,y,z", "coordinate_unit": "angstrom",
            "round_digits": int(task_cfg["constants"].get("coordinate_round_digits", 10)),
            "separator": "LF", "encoding": "UTF-8", "algorithm": "SHA-256",
        },
        "seed_evidence_files": sorted(str(p.relative_to(workspace)).replace("\\", "/") for p in evidence_dir.iterdir() if p.is_file()),
        "protocol": "protocol.yaml", "external_tool_module": "benchmark_tools.py",
        "stage_A_schema": "stage_A_schema.json", "stage_B_schema": "stage_B_schema.json",
        "stage_C_schema": "stage_C_schema.json", "result_schema": "result_schema.json",
    }
    write_json(workspace / "instance.json", public_instance)

    truth = {
        "instance_seed": seed,
        "anomalies": anomalies,
        "expected_formula": struct["formula"],
        "expected_canonical_smiles": struct["canonical_smiles"],
        "expected_explicit_atom_count": struct["explicit_atom_count"],
        "expected_heavy_atom_count": struct["heavy_atom_count"],
        "valid_structure_sha256": sha256_file(struct["sdf"]),
    }
    return public_instance, truth

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Set

from chemistry import validate_smiles_rdkit, rdkit_embed_to_xyz
from xtb_runner import compute_reorganization_energy, run_dipro, marcus_hole_rate


@dataclass
class CandidateEvaluation:
    candidate_id: Optional[str]
    smiles: str
    valid: bool
    status: str
    canonical_smiles: Optional[str] = None
    constraint_flags: Optional[Dict[str, bool]] = None
    ring_count_total: Optional[int] = None
    fused_ring_cluster_sizes: Optional[list] = None
    largest_fused_ring_cluster_size: Optional[int] = None
    heavy_atom_count: Optional[int] = None
    molecular_weight: Optional[float] = None
    lambda_h_eV: Optional[float] = None
    Vh_DIPRO_eV: Optional[float] = None
    Vh_DIPRO_meV: Optional[float] = None
    marcus_hole_rate_s_inv: Optional[float] = None
    target_hit: bool = False
    xtb_success: bool = False
    dipro_success: bool = False
    workdir: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _constraint_valid(flags: Dict[str, bool]) -> bool:
    return all(bool(v) for v in flags.values())


def _status_from_rdkit(rd) -> str:
    if not rd.valid_smiles:
        return "invalid_smiles"
    flags = rd.constraint_flags or {}
    if not flags.get("single_component", False):
        return "multiple_components"
    if not flags.get("neutral", False):
        return "non_neutral"
    if not flags.get("allowed_elements", False):
        return "disallowed_elements"
    if not flags.get("no_wildcards", False):
        return "wildcard_or_query_atom"
    if not flags.get("five_ring_fused_core", False):
        return "invalid_fused_ring_core"
    if not flags.get("no_extra_rings", False):
        return "invalid_fused_ring_core"
    if not flags.get("size_range", False):
        return "size_out_of_range"
    return "success"


def evaluate_candidate(
    candidate: Dict[str, Any],
    task_cfg: Dict[str, Any],
    workdir: str | Path,
    seen_canonical_smiles: Set[str],
    run_xtb: bool = True,
) -> CandidateEvaluation:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    candidate_id = candidate.get("candidate_id")
    smiles = candidate.get("smiles", "")

    rd = validate_smiles_rdkit(smiles, task_cfg["constraints"])
    flags = rd.constraint_flags or {
        "valid_smiles": False,
        "single_component": False,
        "neutral": False,
        "allowed_elements": False,
        "no_wildcards": False,
        "five_ring_fused_core": False,
        "no_extra_rings": False,
        "size_range": False,
    }

    unique = rd.canonical_smiles not in seen_canonical_smiles if rd.canonical_smiles else False
    flags["unique_smiles"] = unique

    status = _status_from_rdkit(rd)
    if status == "success" and not unique:
        status = "duplicate_smiles"

    valid_constraints = status == "success" and _constraint_valid(flags)

    base = CandidateEvaluation(
        candidate_id=candidate_id,
        smiles=smiles,
        valid=valid_constraints,
        status=status,
        canonical_smiles=rd.canonical_smiles,
        constraint_flags=flags,
        ring_count_total=rd.ring_count_total,
        fused_ring_cluster_sizes=rd.fused_ring_cluster_sizes,
        largest_fused_ring_cluster_size=rd.largest_fused_ring_cluster_size,
        heavy_atom_count=rd.heavy_atom_count,
        molecular_weight=rd.molecular_weight,
        workdir=str(workdir),
        error=rd.error,
        details={"rdkit": rd.to_dict()},
    )

    if not valid_constraints:
        return base

    if not run_xtb or not task_cfg.get("xtb", {}).get("enabled", True):
        base.status = "success_rdkit_only"
        return base

    initial_xyz = workdir / "rdkit_initial.xyz"
    try:
        rdkit_embed_to_xyz(smiles, initial_xyz, task_cfg["rdkit"])
    except Exception as exc:
        base.valid = False
        base.status = "rdkit_embedding_failed"
        base.error = str(exc)
        return base

    try:
        reorg = compute_reorganization_energy(initial_xyz, workdir / "xtb_reorganization", task_cfg["xtb"])
    except Exception as exc:
        base.valid = False
        base.status = "reorganization_energy_failed"
        base.error = str(exc)
        return base

    base.details["reorganization"] = reorg

    if not reorg.get("ok"):
        base.valid = False
        stage = reorg.get("stage")
        if stage == "neutral_opt":
            base.status = "xtb_neutral_opt_failed"
        elif stage == "cation_opt":
            base.status = "xtb_cation_opt_failed"
        else:
            base.status = "reorganization_energy_failed"
        base.error = reorg.get("error")
        return base

    base.lambda_h_eV = reorg.get("lambda_h_eV")
    base.xtb_success = True

    try:
        dipro = run_dipro(
            Path(reorg["neutral_opt_xyz"]),
            workdir / "dipro",
            task_cfg["packing"],
            task_cfg["dipro"],
            task_cfg["xtb"],
        )
    except Exception as exc:
        base.valid = False
        base.status = "dipro_failed"
        base.error = str(exc)
        return base

    base.details["dipro"] = dipro

    if not dipro.get("ok"):
        base.valid = False
        packing_search = dipro.get("packing_search") or {}
        if packing_search and not packing_search.get("ok", False):
            base.status = "dimer_construction_failed"
        else:
            base.status = "dipro_failed"
        base.error = dipro.get("error")
        return base

    base.Vh_DIPRO_eV = dipro.get("Vh_DIPRO_eV")
    base.Vh_DIPRO_meV = dipro.get("Vh_DIPRO_meV")
    base.dipro_success = True

    try:
        rate = marcus_hole_rate(base.lambda_h_eV, base.Vh_DIPRO_eV, task_cfg["marcus"])
    except Exception as exc:
        base.valid = False
        base.status = "marcus_rate_failed"
        base.error = str(exc)
        return base

    base.marcus_hole_rate_s_inv = rate
    coupling_ratio = float(base.Vh_DIPRO_eV) / float(base.lambda_h_eV)
    base.details["marcus_applicability"] = {
        "Vh_over_lambda": coupling_ratio,
        "nonadiabatic_regime_warning": coupling_ratio >= 1.0,
        "interpretation": (
            "Vh >= lambda_h; the nonadiabatic Marcus expression is retained only as a standardized benchmark proxy."
            if coupling_ratio >= 1.0 else
            "Vh < lambda_h; the value remains a standardized benchmark proxy, not an experimental mobility prediction."
        ),
    }
    base.status = "success"
    base.valid = True
    base.target_hit = rate > float(task_cfg["marcus"]["rate_threshold_s_inv"])
    return base

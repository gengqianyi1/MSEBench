from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_feedback(
    evaluation,
    task_cfg: Dict[str, Any],
    best_so_far,
) -> Dict[str, Any]:
    ev = evaluation.to_dict() if hasattr(evaluation, "to_dict") else evaluation
    threshold = float(task_cfg["marcus"]["rate_threshold_s_inv"])

    issues: List[str] = []
    for k, v in (ev.get("constraint_flags") or {}).items():
        if not v:
            issues.append(k)

    if ev.get("status") not in {"success", "success_rdkit_only"}:
        issues.append(ev.get("status"))

    rate = ev.get("marcus_hole_rate_s_inv")
    target_hit = bool(ev.get("target_hit"))
    dipro_details = (ev.get("details") or {}).get("dipro") or {}
    packing_search = dipro_details.get("packing_search") or {}

    summary_parts = []
    if ev.get("valid"):
        summary_parts.append("All hard constraints and required computations are satisfied.")
    else:
        summary_parts.append(f"Candidate status: {ev.get('status')}.")

    if rate is not None:
        summary_parts.append(f"Marcus hole rate = {rate:.6e} s^-1.")
        if rate > threshold:
            summary_parts.append(f"Target hit: rate exceeds {threshold:.1e} s^-1.")
        else:
            summary_parts.append(f"Target not reached: rate does not exceed {threshold:.1e} s^-1.")
    else:
        summary_parts.append("Marcus rate is unavailable because one or more evaluation steps failed.")

    if best_so_far is not None and best_so_far.marcus_hole_rate_s_inv is not None:
        summary_parts.append(
            f"Best rate so far is {best_so_far.marcus_hole_rate_s_inv:.6e} s^-1 from {best_so_far.candidate_id}."
        )

    return {
        "summary": " ".join(summary_parts),
        "issues": issues,
        "constraint_flags": ev.get("constraint_flags"),
        "computed_properties": {
            "ring_count_total": ev.get("ring_count_total"),
            "fused_ring_cluster_sizes": ev.get("fused_ring_cluster_sizes"),
            "largest_fused_ring_cluster_size": ev.get("largest_fused_ring_cluster_size"),
            "heavy_atom_count": ev.get("heavy_atom_count"),
            "molecular_weight": ev.get("molecular_weight"),
            "lambda_h_eV": ev.get("lambda_h_eV"),
            "Vh_DIPRO_eV": ev.get("Vh_DIPRO_eV"),
            "Vh_DIPRO_meV": ev.get("Vh_DIPRO_meV"),
            "marcus_hole_rate_s_inv": ev.get("marcus_hole_rate_s_inv"),
            "selected_packing_dx_A": packing_search.get("selected_dx_A"),
            "selected_packing_dy_A": packing_search.get("selected_dy_A"),
            "selected_packing_dz_A": packing_search.get("selected_dz_A"),
            "selected_packing_energy_hartree": packing_search.get("selected_energy_hartree"),
            "packing_search_n_trials": packing_search.get("n_trials"),
            "packing_search_n_successful_trials": packing_search.get("n_successful_trials"),
        },
        "target_hit": target_hit,
        "status": ev.get("status"),
        "best_rate_so_far_s_inv": None if best_so_far is None else best_so_far.marcus_hole_rate_s_inv,
        "best_candidate_id_so_far": None if best_so_far is None else best_so_far.candidate_id,
    }

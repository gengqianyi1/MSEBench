from __future__ import annotations

from typing import Any, Dict, List, Optional


def _ev_dict(ev):
    return ev.to_dict() if hasattr(ev, "to_dict") else ev


def select_best_rate(evaluations: List[Any]):
    valid = [ev for ev in evaluations if getattr(ev, "marcus_hole_rate_s_inv", None) is not None]
    if valid:
        return max(valid, key=lambda x: x.marcus_hole_rate_s_inv)
    if evaluations:
        # fallback: most satisfied constraints
        return max(evaluations, key=lambda x: sum(bool(v) for v in (x.constraint_flags or {}).values()))
    return None


def constraint_score(ev, scoring_cfg: Dict[str, Any]) -> int:
    """Compatibility field: constraints are eligibility gates, never points."""
    return 0


def rate_performance_score(rate: Optional[float], scoring_cfg: Dict[str, Any]) -> float:
    """Score only the explicitly recommended candidate on a linear rate scale."""
    if rate is None:
        return 0.0
    floor = float(scoring_cfg.get("rate_score_floor_s_inv", 1.0e14))
    ceiling = float(scoring_cfg.get("rate_score_ceiling_s_inv", 1.0e15))
    if rate < floor:
        return 0.0
    return min(100.0, 100.0 * float(rate) / ceiling)


def final_score(evaluations: List[Any], task_cfg: Dict[str, Any], recommended_candidate_id: Optional[str]) -> Dict[str, Any]:
    scoring_cfg = task_cfg["scoring"]
    matches = [ev for ev in evaluations if ev.candidate_id == recommended_candidate_id]
    selected = matches[0] if len(matches) == 1 else None
    flags = {} if selected is None else (selected.constraint_flags or {})
    eligible = bool(selected is not None and selected.status == "success" and all(flags.values()))
    selected_rate = selected.marcus_hole_rate_s_inv if eligible else None
    total = rate_performance_score(selected_rate, scoring_cfg)

    target_hits = [ev for ev in evaluations if getattr(ev, "target_hit", False)]
    successful = [ev for ev in evaluations if getattr(ev, "status", None) == "success"]

    ranked = sorted(
        [ev for ev in evaluations if getattr(ev, "marcus_hole_rate_s_inv", None) is not None],
        key=lambda x: x.marcus_hole_rate_s_inv,
        reverse=True,
    )

    return {
        "recommended_candidate_id": recommended_candidate_id,
        "recommendation_valid": len(matches) == 1,
        "eligible": eligible,
        "constraint_score": 0,
        "rate_performance_score": total,
        "total_score": total,
        "recommended_candidate": None if selected is None else selected.to_dict(),
        "recommended_rate_s_inv": selected_rate,
        "n_target_hits": len(target_hits),
        "n_successful_computed_candidates": len(successful),
        "ranked_candidate_ids_by_rate": [ev.candidate_id for ev in ranked],
    }

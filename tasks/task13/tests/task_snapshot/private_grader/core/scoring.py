"""
Trajectory-level scoring for the retrosynthesis-route benchmark.

Total: 100 points. Difficulty is calibrated so a strong model typically lands
near 60 when it builds a chemically coherent route but chooses the near-miss
amine partner. Full credit still requires a complete, fully-valid,
step-efficient route reaching the exact target under feedback.

Scoring is computed over the WHOLE trajectory (best route across all rounds),
mirroring the demo benchmark's "best-over-trajectory" philosophy.

Score breakdown (all thresholds come from scoring_cfg):

1. Best-step-progress: from the single best round, how many steps
   fired-and-matched AND used only available reactants AND conserved atoms AND
   passed the atom-economy floor, counted as a fraction of the reference/expected
   route length. This gives graded partial credit for partial chemistry.

2. Route validity: the best round's route satisfies every hard
   constraint (all flags True) -> full 20; otherwise a per-flag partial credit.
   Hard constraints include heavy-atom conservation (no atom created).

3. Target reached: a fully-valid route whose final product equals the
   target. This separates high-quality near-miss routes from solved routes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evaluator import RouteEvaluation, route_constraint_flags, route_is_fully_valid


def _valid_matched_steps(route_eval: RouteEvaluation) -> int:
    return sum(
        1 for se in route_eval.step_evals
        if se.fired and se.product_matches and se.reactants_available
        and se.atom_conservation_ok and se.atom_economy_ok
    )


def score_single_route(route_eval: RouteEvaluation, scoring_cfg: Dict[str, Any]) -> Dict[str, Any]:
    progress_max = float(scoring_cfg["step_progress_points"])
    validity_max = float(scoring_cfg["route_validity_points"])
    target_max = float(scoring_cfg["target_reached_points"])
    expected_len = int(scoring_cfg["expected_route_length"])

    # 1. step progress (graded)
    good_steps = _valid_matched_steps(route_eval)
    progress_ratio = min(good_steps / expected_len, 1.0) if expected_len > 0 else 0.0
    progress_points = round(progress_ratio * progress_max, 4)

    # 2. route validity (per-flag partial credit)
    flags = route_constraint_flags(route_eval)
    n_flags = len(flags)
    n_true = sum(1 for v in flags.values() if v)
    validity_points = round((n_true / n_flags) * validity_max, 4) if n_flags else 0.0

    # 3. target reached (all-or-nothing, requires a fully valid route)
    reached = route_is_fully_valid(route_eval) and route_eval.final_equals_target
    target_points = target_max if reached else 0.0

    total = round(progress_points + validity_points + target_points, 4)

    return {
        "total_route_score": total,
        "step_progress_points": progress_points,
        "step_progress_detail": {
            "valid_matched_steps": good_steps,
            "expected_route_length": expected_len,
            "progress_ratio": round(progress_ratio, 4),
        },
        "route_validity_points": validity_points,
        "route_validity_detail": {"flags": flags, "n_true": n_true, "n_flags": n_flags},
        "target_reached_points": target_points,
        "target_reached": reached,
    }


def _route_key(scored: Dict[str, Any]) -> tuple:
    # prefer reaching target, then higher total, then fewer-but-valid steps handled via total
    return (1 if scored["target_reached"] else 0, scored["total_route_score"])


def final_score(
    round_route_evals: List[RouteEvaluation],
    scoring_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Score the whole trajectory by selecting the best single round."""
    if not round_route_evals:
        return {
            "total_score": 0,
            "best_round_index": None,
            "success": False,
            "detail": None,
        }

    scored_rounds: List[Dict[str, Any]] = [
        score_single_route(re, scoring_cfg) for re in round_route_evals
    ]

    best_idx = max(range(len(scored_rounds)), key=lambda i: _route_key(scored_rounds[i]))
    best = scored_rounds[best_idx]

    return {
        "total_score": best["total_route_score"],
        "total_points": float(scoring_cfg.get("total_points", 100)),
        "best_round_index": best_idx + 1,
        "success": bool(best["target_reached"]),
        "detail": best,
        "per_round_scores": [
            {"round": i + 1, "total": s["total_route_score"], "target_reached": s["target_reached"]}
            for i, s in enumerate(scored_rounds)
        ],
    }

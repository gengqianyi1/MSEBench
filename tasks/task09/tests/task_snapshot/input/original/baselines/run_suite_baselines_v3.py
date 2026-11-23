"""Run truthful v3 suite calibration policies against frozen native points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluator.evaluate_suite import aggregate_suite
from evaluator.profile_recompute_v2 import recompute_automatic_profile
from evaluator.scoring_v2 import score_profile_v2


FIXED_12 = [0, 60, 120, 180, 240, 300, 30, 150, 270, 90, 210, 330]
WEAK_6 = [0, 60, 120, 180, 240, 300]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def oracle_aware_angles(values: list[float]) -> list[int]:
    selected = WEAK_6.copy()
    for _ in range(2):
        sample_values = [values[angle // 5] for angle in selected]
        prediction = recompute_automatic_profile(selected, sample_values)["predicted_energies"]
        candidates = [angle for angle in range(0, 360, 5) if angle not in selected]
        chosen = sorted(candidates, key=lambda angle: (-abs(prediction[angle // 5] - values[angle // 5]), angle))[:3]
        selected.extend(chosen)
    return selected


def case_evaluation(case: dict[str, Any], policy: str) -> dict[str, Any]:
    profile = json.loads((ROOT / case["oracle_path"]).read_text(encoding="utf-8"))
    oracle = [float(value) for value in profile["relative_energies_kcal_mol"]]
    if policy == "weak_6":
        angles = WEAK_6
        feedback = refinement = 0.0
        has_feedback_rounds = False
    elif policy == "fixed_12":
        angles = FIXED_12
        # The schedule is fixed before any observation.  Coverage is real, but
        # it is neither feedback-dependent nor a refinement policy.
        feedback = refinement = 0.0
        has_feedback_rounds = False
    elif policy == "oracle_guided_greedy":
        angles = oracle_aware_angles(oracle)
        feedback = refinement = 1.0
        has_feedback_rounds = True
    else:
        raise ValueError(policy)
    sampled = [oracle[angle // 5] for angle in angles]
    fit = recompute_automatic_profile(angles, sampled)
    complete = len(angles) >= 10
    gates = {
        "complete_protocol": complete,
        "minimum_valid_angles": complete,
        "minimum_feedback_rounds": has_feedback_rounds,
        "native_points_hash_bound": True,
        "scientific_invariants_preserved": True,
        "profile_recomputed_independently": True,
    }
    facts = {
        "adaptive_sampling": {
            "coverage_fraction": min(1.0, len(set(angles)) / 12.0),
            "feedback_fraction": feedback,
            "refinement_fraction": refinement,
            "efficiency_fraction": 1.0 if len(angles) <= 12 else 0.0,
        },
        "process_and_reproducibility": {
            "trajectory_integrity": True,
            "artifact_provenance": True,
            "exact_identity": True,
            "scientific_invariants": True,
            "replay_consistency": True,
        },
    }
    scored = score_profile_v2(
        predicted_energies_kcal_mol=fit["predicted_energies"],
        oracle_energies_kcal_mol=oracle,
        facts=facts,
        eligibility_gates=gates,
    )
    scientific = scored["categories"]["curve_accuracy"]["score"] + scored["categories"]["extrema_and_barriers"]["score"]
    process = scored["categories"]["adaptive_sampling"]["score"] + scored["categories"]["process_and_reproducibility"]["score"]
    return {
        "schema_version": "3.0",
        "case_id": case["case_id"],
        "policy": policy,
        "status": "PASS" if scored["eligible"] else "FAIL",
        "provider_status": "NOT_APPLICABLE_DETERMINISTIC_REFERENCE",
        "eligible": scored["eligible"],
        "failed_gates": scored["eligibility"]["failed_gates"],
        "selected_angles_deg": angles,
        "scientific_score": round(scientific, 12),
        "process_score": round(process, 12),
        "ungated_score": scored["ungated_total_score"],
        "official_score": scored["official_score"],
        "metrics": scored["metrics"],
        "reference_role": "evaluator calibration; not a language-model result",
        "process_evidence_mode": (
            "oracle-informed non-blind reference"
            if policy == "oracle_guided_greedy"
            else "precommitted non-adaptive schedule"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads((ROOT / "organizer_private" / "suite.json").read_text(encoding="utf-8"))
    expected = [item["case_id"] for item in suite["cases"]]
    summaries = {}
    for policy in ("weak_6", "fixed_12", "oracle_guided_greedy"):
        case_results = [case_evaluation(case, policy) for case in suite["cases"]]
        policy_root = args.output_root / policy
        for result in case_results:
            write_json(policy_root / "cases" / f"{result['case_id']}.json", result)
        aggregate = aggregate_suite(case_results, expected)
        aggregate["policy"] = policy
        aggregate["reference_role"] = "evaluator calibration; not a language-model result"
        write_json(policy_root / "suite_evaluation.json", aggregate)
        summaries[policy] = aggregate
    print(json.dumps({key: value["macro_official_score_excluding_blocked"] for key, value in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

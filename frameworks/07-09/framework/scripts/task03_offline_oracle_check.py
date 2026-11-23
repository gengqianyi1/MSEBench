#!/usr/bin/env python3
"""Offline Task 03 Oracle from the frozen private successful-reference lifecycle.

The runtime xTB/DIPRO Oracle remains pending until a sufficiently fast locked
environment is available. This script instead proves that the unified Harbor
scoring logic reconstructs a positive, deterministic final score from the
private calibrated reference molecules that already carry successful xTB/DIPRO
results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "task-03-five-ring-xtb-dipro-marcus"
SRC = TASK / "tests" / "task_snapshot" / "input" / "original" / "src"
REFERENCE = (
    TASK
    / "tests"
    / "task_snapshot"
    / "private_data"
    / "reference_examples"
    / "satisfied_molecules.json"
)
OUTPUT = ROOT / "parity" / "task03_offline_oracle.json"


def main() -> int:
    sys.path.insert(0, str(SRC))
    import yaml
    from evaluator import CandidateEvaluation
    from scoring import final_score

    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    molecules = sorted(
        payload["molecules"],
        key=lambda item: float(item["marcus_hole_rate_s_inv"]),
        reverse=True,
    )[:5]
    cfg = yaml.safe_load(
        (TASK / "tests" / "task_snapshot" / "input" / "original" / "tasks" / "five_ring_xtb_dipro_marcus" / "task_config.yaml").read_text(encoding="utf-8")
    )

    evaluations = []
    for index, molecule in enumerate(molecules, start=1):
        evaluation = CandidateEvaluation(
            candidate_id=f"R{index}",
            smiles=molecule["smiles"],
            valid=True,
            status="success",
            canonical_smiles=molecule["canonical_smiles"],
            constraint_flags={
                "valid_smiles": True,
                "single_component": True,
                "neutral": True,
                "allowed_elements": True,
                "no_wildcards": True,
                "five_ring_fused_core": True,
                "no_extra_rings": True,
                "size_range": True,
                "unique_smiles": True,
            },
            lambda_h_eV=float(molecule["lambda_h_eV"]),
            Vh_DIPRO_eV=float(molecule["Vh_DIPRO_eV"]),
            marcus_hole_rate_s_inv=float(molecule["marcus_hole_rate_s_inv"]),
            target_hit=bool(molecule["target_hit"]),
            xtb_success=bool(molecule["xtb_success"]),
            dipro_success=bool(molecule["dipro_success"]),
        )
        evaluations.append(evaluation)

    native = final_score(evaluations, cfg, "R1")
    result = {
        "score": native["total_score"],
        "final_recommendation": {"recommended_candidate_id": "R1"},
        "checks": {
            "independently_recomputed": False,
            "offline_reference_lifecycle": True,
            "n_evaluated": len(evaluations),
        },
        "evaluations": [item.to_dict() for item in evaluations],
    }
    failure_codes = []
    if not native["recommendation_valid"]:
        failure_codes.append("INVALID_FINAL_RECOMMENDATION")
    if not native["eligible"]:
        failure_codes.append("RECOMMENDED_CANDIDATE_INELIGIBLE")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "PASS" if native["total_score"] > 0 and not failure_codes else "FAIL",
                "oracle_score": native["total_score"],
                "recommended_candidate_id": "R1",
                "failure_codes": failure_codes,
                "result": result,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS" if native["total_score"] > 0 and not failure_codes else "FAIL", "oracle_score": native["total_score"], "failure_codes": failure_codes, "evidence": str(OUTPUT)}, indent=2))
    return 0 if native["total_score"] > 0 and not failure_codes else 1


if __name__ == "__main__":
    raise SystemExit(main())

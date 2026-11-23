from __future__ import annotations

from typing import Any, Dict


COMPLIANCE_PENALTY_WEIGHTS = {
    "schema_and_consistency": 2.0,
    "data_audit_accuracy": 1.0,
    "validation_evidence": 1.0,
    "reproducibility_artifacts": 1.0,
    "scientific_interpretation": 1.0,
}


def score_workflow(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Return transparent deductions without allocating extra workflow points."""
    breakdown = {
        key: 0.0 if bool(evidence.get(key)) else weight
        for key, weight in COMPLIANCE_PENALTY_WEIGHTS.items()
    }
    return {
        "score": 0.0,
        "max_score": 0.0,
        "breakdown": breakdown,
        "compliance_penalty": float(sum(breakdown.values())),
        "max_compliance_penalty": float(sum(COMPLIANCE_PENALTY_WEIGHTS.values())),
    }

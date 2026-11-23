"""Deterministic validation of structured scientific-interpretation evidence."""

from __future__ import annotations

import math
from typing import Any, Iterable


def interpretation_claim(payload: dict[str, Any], claim_id: str) -> dict[str, Any]:
    claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(claims, list):
        return {}
    for claim in claims:
        if isinstance(claim, dict) and str(claim.get("claim_id") or "") == claim_id:
            return claim
    return {}


def grounded_claim(
    payload: dict[str, Any],
    claim_id: str,
    trusted_job_ids: Iterable[str],
    *,
    required_terms: tuple[tuple[str, ...], ...] = (),
    minimum_explanation_length: int = 60,
) -> tuple[bool, dict[str, Any]]:
    claim = interpretation_claim(payload, claim_id)
    explanation = str(claim.get("explanation") or "").strip()
    evidence_jobs = claim.get("evidence_job_ids")
    evidence_jobs = [str(value) for value in evidence_jobs] if isinstance(evidence_jobs, list) else []
    trusted = set(map(str, trusted_job_ids))
    concepts = {str(value).strip().lower() for value in claim.get("concepts", []) if str(value).strip()} if isinstance(claim.get("concepts"), list) else set()
    terms_ok = all(any(term.lower() in concepts for term in alternatives) for alternatives in required_terms)
    jobs_ok = bool(evidence_jobs) and all(job_id in trusted for job_id in evidence_jobs)
    ok = len(explanation) >= minimum_explanation_length and jobs_ok and terms_ok
    return ok, {
        "claim_id": claim_id,
        "explanation_length": len(explanation),
        "evidence_job_ids": evidence_jobs,
        "all_jobs_trusted": jobs_ok,
        "concepts": sorted(concepts),
        "required_terms_satisfied": terms_ok,
    }


def quantitatively_grounded_claim(
    payload: dict[str, Any],
    claim_id: str,
    trusted_job_ids: Iterable[str],
    *,
    expected_quantities: dict[str, dict[str, Any]],
    minimum_explanation_length: int = 60,
) -> tuple[bool, dict[str, Any]]:
    """Validate a prose claim against evaluator-recomputed numeric facts.

    Private metric identifiers are never model-visible vocabulary.  A submitted
    quantity may therefore match either by the evaluator identifier or by its
    recomputed value, physical unit and required trusted jobs.  At least one
    matching quantity is required; extra scientific observations are audited
    but do not erase an otherwise grounded claim.
    """

    claim = interpretation_claim(payload, claim_id)
    explanation = str(claim.get("explanation") or "").strip()
    trusted = set(map(str, trusted_job_ids))
    claim_jobs_raw = claim.get("evidence_job_ids")
    claim_jobs = [str(value) for value in claim_jobs_raw] if isinstance(claim_jobs_raw, list) else []
    claim_jobs_ok = bool(claim_jobs) and all(job_id in trusted for job_id in claim_jobs)
    quantities = claim.get("quantities") if isinstance(claim.get("quantities"), list) else []
    checks: list[dict[str, Any]] = []
    for quantity in quantities:
        if not isinstance(quantity, dict):
            checks.append({"recognized": False, "reason": "quantity is not an object"})
            continue
        metric_id = str(quantity.get("metric_id") or "")
        registered_spec = expected_quantities.get(metric_id)
        value = quantity.get("value")
        unit = str(quantity.get("unit") or "")
        evidence_raw = quantity.get("evidence_job_ids")
        evidence_jobs = [str(item) for item in evidence_raw] if isinstance(evidence_raw, list) else []
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        trusted_jobs_ok = bool(evidence_jobs) and all(job_id in trusted for job_id in evidence_jobs)
        claim_link_ok = bool(set(evidence_jobs) & set(claim_jobs))
        candidates = [(metric_id, registered_spec)] if isinstance(registered_spec, dict) else list(expected_quantities.items())
        matches = []
        for expected_metric_id, candidate in candidates:
            if not isinstance(candidate, dict) or not numeric:
                continue
            expected_value = float(candidate["value"])
            tolerance = float(candidate.get("absolute_tolerance", 0.0))
            expected_unit = str(candidate.get("unit") or "")
            canonical_unit = {"dimensionless": "fraction", "1": "fraction"}.get(unit, unit)
            canonical_expected_unit = {"dimensionless": "fraction", "1": "fraction"}.get(expected_unit, expected_unit)
            expected_jobs = {str(item) for item in candidate.get("evidence_job_ids", [])}
            if (
                abs(float(value) - expected_value) <= tolerance
                and canonical_unit == canonical_expected_unit
                and (not expected_jobs or expected_jobs.issubset(set(evidence_jobs)))
            ):
                matches.append((str(expected_metric_id), candidate))
        matched_metric_id = matches[0][0] if len(matches) == 1 else None
        spec = matches[0][1] if len(matches) == 1 else registered_spec
        recognized = isinstance(registered_spec, dict)
        expected_value = float(spec["value"]) if isinstance(spec, dict) else math.nan
        tolerance = float(spec.get("absolute_tolerance", 0.0)) if isinstance(spec, dict) else math.nan
        value_ok = bool(numeric and isinstance(spec, dict) and abs(float(value) - expected_value) <= tolerance)
        expected_unit = str(spec.get("unit") or "") if isinstance(spec, dict) else ""
        unit_ok = bool(
            isinstance(spec, dict)
            and {"dimensionless": "fraction", "1": "fraction"}.get(unit, unit)
            == {"dimensionless": "fraction", "1": "fraction"}.get(expected_unit, expected_unit)
        )
        expected_jobs = {str(item) for item in (spec.get("evidence_job_ids", []) if isinstance(spec, dict) else [])}
        expected_jobs_ok = not expected_jobs or expected_jobs.issubset(set(evidence_jobs))
        valid = len(matches) == 1 and trusted_jobs_ok and claim_link_ok
        checks.append(
            {
                "metric_id": metric_id,
                "recognized": recognized,
                "matched_expected_metric_id": matched_metric_id,
                "submitted_value": value,
                "expected_value": expected_value if recognized else None,
                "absolute_tolerance": tolerance if recognized else None,
                "value_ok": value_ok,
                "submitted_unit": unit,
                "unit_ok": unit_ok,
                "evidence_job_ids": evidence_jobs,
                "trusted_jobs_ok": trusted_jobs_ok,
                "expected_jobs_ok": expected_jobs_ok,
                "claim_link_ok": claim_link_ok,
                "valid": valid,
            }
        )
    all_quantities_valid = bool(checks) and all(check.get("valid") is True for check in checks)
    has_valid_quantity = any(check.get("valid") is True for check in checks)
    explanation_ok = len(explanation) >= minimum_explanation_length
    ok = explanation_ok and claim_jobs_ok and has_valid_quantity
    return ok, {
        "claim_id": claim_id,
        "explanation_length": len(explanation),
        "explanation_ok": explanation_ok,
        "evidence_job_ids": claim_jobs,
        "all_jobs_trusted": claim_jobs_ok,
        "quantity_checks": checks,
        "all_quantities_valid": all_quantities_valid,
        "has_valid_quantity": has_valid_quantity,
    }

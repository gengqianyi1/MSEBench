"""Canonicalize immutable scientific scoring facts for deterministic replay."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCIENTIFIC_METADATA_KEYS = {
    "task_07": (
        "candidate_evidence",
        "trusted_values",
        "ranking",
        "uncertainty_classification",
    ),
    "task_07_redox_reorganization": (
        "candidate_evidence",
        "trusted_values",
        "ranking",
        "uncertainty_classification",
    ),
    "task_04": (
        "trusted_job_count",
        "derived_clusters",
        "candidate_tautomer_identity",
        "raw_job_gibbs",
        "microstate_population_total_variation",
        "tautomer_population_total_variation",
        "dominant_tautomer",
        "dominant_microstate",
    ),
    "task_04_tautomer_ensemble": (
        "trusted_job_count",
        "derived_clusters",
        "candidate_tautomer_identity",
        "raw_job_gibbs",
        "microstate_population_total_variation",
        "tautomer_population_total_variation",
        "dominant_tautomer",
        "dominant_microstate",
    ),
    "task_05": (
        "trusted_job_count",
        "derived_pose_clusters",
        "raw_interaction_energies",
        "raw_deformation_energies",
        "raw_binding_energies",
        "best_pose",
        "closure_errors_kcal_mol",
    ),
    "task_05_binding_decomposition": (
        "trusted_job_count",
        "derived_pose_clusters",
        "raw_interaction_energies",
        "raw_deformation_energies",
        "raw_binding_energies",
        "best_pose",
        "closure_errors_kcal_mol",
    ),
    "task_06": (
        "trusted_job_count",
        "forward_reverse_rmse_kcal_mol",
        "merged_profile_rmse_kcal_mol",
        "raw_independent_minima",
        "raw_major_barriers",
        "raw_lowest_rotation_barrier_kcal_mol",
        "direction_job_sets_disjoint",
        "shared_initial_geometry_sha256",
        "direction_provenance_gate_triggered",
    ),
    "task_06_torsion_profile": (
        "trusted_job_count",
        "forward_reverse_rmse_kcal_mol",
        "merged_profile_rmse_kcal_mol",
        "raw_independent_minima",
        "raw_major_barriers",
        "raw_lowest_rotation_barrier_kcal_mol",
        "direction_job_sets_disjoint",
        "shared_initial_geometry_sha256",
        "direction_provenance_gate_triggered",
    ),
}


def _canonical_sections(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections = []
    for section in report.get("sections", []) or []:
        items = []
        for item in section.get("items", []) or []:
            items.append(
                {
                    "item_id": item.get("item_id"),
                    "earned_points": item.get("earned_points"),
                    "max_points": item.get("max_points"),
                    "fraction": item.get("fraction"),
                }
            )
        sections.append(
            {
                "section_id": section.get("section_id"),
                "earned_points": section.get("earned_points"),
                "max_points": section.get("max_points"),
                "items": sorted(items, key=lambda value: str(value.get("item_id"))),
            }
        )
    return sorted(sections, key=lambda value: str(value.get("section_id")))


def canonicalize_report(report: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(report.get("task_id", ""))
    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
    keys = SCIENTIFIC_METADATA_KEYS.get(task_id, ())
    gates = [
        {
            "gate_id": gate.get("gate_id"),
            "total_zero": bool(gate.get("total_zero", False)),
            "reason": gate.get("reason"),
        }
        for gate in (report.get("hard_gates", []) or [])
        if isinstance(gate, dict)
    ]
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "total_score": report.get("total_score", report.get("score")),
        "max_score": report.get("max_score"),
        "sections": _canonical_sections(report),
        "hard_gates": sorted(gates, key=lambda value: str(value.get("gate_id"))),
        "scientific_metadata": {key: metadata[key] for key in keys if key in metadata},
    }


def canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    return json.dumps(
        canonicalize_report(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_report_hash(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_report_bytes(report)).hexdigest()


def compare_canonical_reports(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    expected_hash = canonical_report_hash(expected)
    observed_hash = canonical_report_hash(observed)
    return {
        "schema_version": "1.0.0",
        "match": expected_hash == observed_hash,
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "expected": canonicalize_report(expected),
        "observed": canonicalize_report(observed),
    }


def score_invariant_violations(report: Mapping[str, Any]) -> list[str]:
    """Explicit four-field equality invariant (spec 4.3): the four official
    score fields must be numerically identical."""
    fields = ("score", "task_local_score", "scientific_basis_score", "comparison_score")
    values: list[float] = []
    violations: list[str] = []
    for key in fields:
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            violations.append(f"{key}: not a number ({value!r})")
            continue
        values.append(float(value))
    if values and any(abs(value - values[0]) > 1e-9 for value in values[1:]):
        violations.append("four official score fields are not numerically equal: " + ", ".join(f"{key}={report.get(key)!r}" for key in fields))
    return violations
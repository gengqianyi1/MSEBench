"""Build v3 suite descriptors, commitments, configs, and morphology evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_multicase_suite_v3 import CASE_DEFINITIONS, canonical_bytes, sha256


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "organizer_private"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    values = [float(value) for value in profile["relative_energies_kcal_mol"]]
    minima = []
    maxima = []
    for index, value in enumerate(values):
        left = values[(index - 1) % len(values)]
        right = values[(index + 1) % len(values)]
        if value < left and value <= right:
            minima.append(index * 5)
        if value > left and value >= right:
            maxima.append(index * 5)
    return {
        "point_count": len(values),
        "energy_range_kcal_mol": round(max(values) - min(values), 12),
        "local_minima_angles_deg": minima,
        "local_maxima_angles_deg": maxima,
        "global_minimum_angles_deg": [index * 5 for index, value in enumerate(values) if abs(value - min(values)) <= 1.0e-8],
    }


def main() -> int:
    public_oracle = ROOT / "evaluator_private" / "oracle_profile.json"
    public_profile = json.loads(public_oracle.read_text(encoding="utf-8"))
    cases = [
        {
            "case_id": "2_fluorobiphenyl_central_torsion_v1",
            "visibility": "public",
            "molecule": "2-fluorobiphenyl",
            "case_root": "benchmark_case",
            "oracle_path": "evaluator_private/oracle_profile.json",
            "oracle_sha256": sha256(public_oracle),
            "morphology": profile_summary(public_profile),
        }
    ]
    generation_cases = []
    for case_id, definition in CASE_DEFINITIONS.items():
        root = PRIVATE / "cases" / case_id
        oracle_path = root / "oracle_profile.json"
        provenance_path = root / "oracle_provenance.json"
        profile = json.loads(oracle_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("native_attempt_count") != 72 or provenance.get("all_angles_eligible") is not True:
            raise RuntimeError(f"native coverage invalid for {case_id}")
        summary = profile_summary(profile)
        cases.append(
            {
                "case_id": case_id,
                "visibility": "hidden",
                "molecule": definition["molecule"],
                "case_root": f"organizer_private/cases/{case_id}",
                "config": f"organizer_configs/{case_id}.json",
                "oracle_path": f"organizer_private/cases/{case_id}/oracle_profile.json",
                "oracle_sha256": sha256(oracle_path),
                "provenance_sha256": sha256(provenance_path),
                "expected_morphology": definition["expected_morphology"],
                "morphology": summary,
            }
        )
        config = {
            "schema_version": "3.0",
            "task_id": "adaptive_xtb_torsion_profile_agent_v3",
            "case_id": case_id,
            "protocol": "adaptive_torsion_sampling_tool_loop/3.0",
            "case_spec": f"organizer_private/cases/{case_id}/case_spec.json",
            "case_spec_sha256": sha256(root / "case_spec.json"),
            "reference_xyz": f"organizer_private/cases/{case_id}/reference.xyz",
            "reference_xyz_sha256": sha256(root / "reference.xyz"),
            "budget": {
                "max_unique_scan_angles": 12,
                "min_unique_valid_scan_angles": 10,
                "max_scan_batches": 4,
                "min_feedback_rounds": 2,
            },
            "scoring": {
                "scientific_score_max": 80,
                "process_score_max": 20,
                "blocked_excluded_from_numeric_aggregate": True,
            },
        }
        write_json(ROOT / "organizer_configs" / f"{case_id}.json", config)
        generation_cases.append(
            {
                "case_id": case_id,
                "native_attempt_count": 72,
                "all_angles_eligible": True,
                "oracle_sha256": sha256(oracle_path),
                "provenance_sha256": sha256(provenance_path),
                "raw_log_scope": "outer validation/native_raw_logs; excluded from release",
                "release_log_projection": "native-program-call-path-redaction/1.0",
            }
        )
    suite = {
        "schema_version": "3.0",
        "task_id": "adaptive_xtb_torsion_profile_agent_v3",
        "protocol": "adaptive_torsion_sampling_tool_loop/3.0",
        "aggregation": {
            "case_weight": "equal",
            "blocked_cases_excluded": True,
            "ranking_requires_complete_coverage": True,
            "scientific_score_max_per_case": 80,
            "process_score_max_per_case": 20,
        },
        "cases": cases,
    }
    suite_path = PRIVATE / "suite.json"
    write_json(suite_path, suite)
    public_suite = {
        "schema_version": "3.0",
        "task_id": suite["task_id"],
        "protocol": suite["protocol"],
        "aggregation": suite["aggregation"],
        "cases": [
            {
                "case_id": cases[0]["case_id"],
                "visibility": "public",
                "case_root": "benchmark_case",
            }
        ],
        "official_organizer_case_count": len(cases),
    }
    write_json(ROOT / "configs" / "suite_public.json", public_suite)
    private_commitment_payload = {
        "schema_version": "3.0",
        "suite_sha256": sha256(suite_path),
        "case_oracle_sha256": {item["case_id"]: item["oracle_sha256"] for item in cases},
    }
    private_commitment_path = PRIVATE / "suite_commitment.json"
    write_json(private_commitment_path, private_commitment_payload)
    public_commitment = {
        "schema_version": "3.0",
        "task_id": suite["task_id"],
        "organizer_case_count": len(cases),
        "private_suite_commitment_sha256": sha256(private_commitment_path),
        "distribution_policy": "hashes are integrity commitments and do not disclose oracle values",
        "public_evaluator_oracle_dependency": False,
    }
    write_json(ROOT / "configs" / "oracle_commitment.json", public_commitment)
    write_json(
        PRIVATE / "native_generation_summary.json",
        {"schema_version": "3.0", "cases": generation_cases},
    )
    write_json(
        PRIVATE / "morphology_validation.json",
        {
            "schema_version": "3.0",
            "case_count": len(cases),
            "distinct_molecules": len({item["molecule"] for item in cases}),
            "all_profiles_have_72_native_grid_points": all(item["morphology"]["point_count"] == 72 for item in cases),
            "energy_ranges_kcal_mol": {item["case_id"]: item["morphology"]["energy_range_kcal_mol"] for item in cases},
            "cases": [{"case_id": item["case_id"], **item["morphology"]} for item in cases],
        },
    )
    print(json.dumps({"suite_sha256": sha256(suite_path), "case_count": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

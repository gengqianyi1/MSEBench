"""Independently validate the frozen organizer oracle and raw attempts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluator.evaluate import _read_xyz, _validate_attempt
from evaluator.scoring import directional_barriers_from_curve, periodic_local_extrema
from tools.science import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    private = ROOT / "evaluator_private"
    oracle_path = private / "oracle_profile.json"
    provenance_path = private / "oracle_provenance.json"
    commitment_path = private / "oracle_commitment.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    symbols, _ = _read_xyz(ROOT / "benchmark_case" / "reference.xyz")
    structure = json.loads((ROOT / "benchmark_case" / "reference_structure.json").read_text(encoding="utf-8"))
    bonds = {tuple(pair) for pair in structure["bonds_zero_based"]}
    attempts = [
        _validate_attempt(path, reference_symbols=symbols, expected_bonds=bonds)
        for path in sorted((private / "oracle_attempts").iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]
    angles = np.asarray(oracle["grid_angles_deg"], dtype=float)
    energies = np.asarray(oracle["relative_energies_kcal_mol"], dtype=float)
    symmetry = np.max(np.abs(energies[:36] - energies[36:]))
    extrema = periodic_local_extrema(angles, energies)
    barriers = directional_barriers_from_curve(angles, energies)
    checks = {
        "point_count_72": len(angles) == 72 and list(angles.astype(int)) == list(range(0, 360, 5)),
        "all_finite": bool(np.all(np.isfinite(energies))),
        "all_72_attempts_independently_valid": len(attempts) == 72 and all(item["valid"] for item in attempts),
        "oracle_profile_commitment": commitment["oracle_profile_sha256"] == sha256_file(oracle_path),
        "oracle_provenance_commitment": commitment["oracle_provenance_sha256"] == sha256_file(provenance_path),
        "reference_commitment": commitment["reference_xyz_sha256"] == sha256_file(ROOT / "benchmark_case" / "reference.xyz"),
        "minimum_zero": abs(float(np.min(energies))) <= 1.0e-12,
    }
    report = {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "successful_attempts": sum(item["valid"] for item in attempts),
        "failed_attempts": sum(not item["valid"] for item in attempts),
        "maximum_angle_error_deg": max(item.get("angle_error_deg", 999.0) for item in attempts),
        "half_turn_symmetry_max_abs_kcal_mol": float(symmetry),
        "half_turn_symmetry_diagnostic_only": True,
        "oracle_extrema": extrema,
        "oracle_barriers": barriers,
        "oracle_profile_sha256": sha256_file(oracle_path),
    }
    output = ROOT / "reports" / "oracle_independent_validation.json"
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

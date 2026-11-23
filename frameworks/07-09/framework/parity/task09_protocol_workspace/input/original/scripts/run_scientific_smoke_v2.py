"""Run the four frozen native smoke angles and emit package-safe receipts."""

from __future__ import annotations

import json
from pathlib import Path

from package_tools.provenance import write_package_safe_projection
from tools.science import (
    run_native_scan,
    sha256_file,
    verify_attempt_manifest,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
ANGLES = (0, 90, 180, 270)


def main() -> int:
    raw_root = (
        ROOT
        / "audit_tmp"
        / "local_host_provenance"
        / "gate1_scientific_smoke_v2"
    )
    projection_root = ROOT / "benchmark_case" / "smoke_projections"
    receipt_path = ROOT / "reports" / "gate1_scientific_smoke.json"
    if raw_root.exists() or projection_root.exists() or receipt_path.exists():
        raise RuntimeError("smoke destinations already exist; immutable run refused")
    raw_root.mkdir(parents=True, exist_ok=False)
    projection_root.mkdir(parents=True, exist_ok=False)
    records = []
    for angle in ANGLES:
        attempt = raw_root / f"angle_{angle:03d}" / "attempt_001"
        record = run_native_scan(
            reference_path=ROOT / "benchmark_case" / "reference.xyz",
            angle_deg=angle,
            attempt_root=attempt,
            timeout_seconds=180,
        )
        projection_path = projection_root / f"angle_{angle:03d}.json"
        projection = write_package_safe_projection(
            attempt, projection_path, exclusive=True
        )
        records.append(
            {
                "angle_deg": angle,
                "return_code": record["return_code"],
                "timed_out": record["timed_out"],
                "converged": record["converged"],
                "total_energy_hartree": record["total_energy_hartree"],
                "final_angle_deg": record["final_angle_deg"],
                "angle_error_deg": record["angle_error_deg"],
                "structure_validation": record["structure_validation"],
                "profile_eligible": record["profile_eligible"],
                "raw_attempt_manifest_valid": verify_attempt_manifest(attempt),
                "raw_attempt_manifest_sha256": record[
                    "attempt_manifest_sha256"
                ],
                "package_safe_projection": (
                    f"benchmark_case/smoke_projections/angle_{angle:03d}.json"
                ),
                "package_safe_projection_sha256": sha256_file(
                    projection_path
                ),
                "projection_self_hash": projection["projection_sha256"],
            }
        )
    passed = all(
        item["return_code"] == 0
        and item["timed_out"] is False
        and item["converged"] is True
        and item["profile_eligible"] is True
        and item["raw_attempt_manifest_valid"] is True
        for item in records
    )
    receipt = {
        "schema_version": "2.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "native_program": "xTB 6.7.1",
        "raw_host_provenance_packaged": False,
        "package_safe_projection_count": len(records),
        "smoke_angles": records,
        "passed": passed,
    }
    write_json(receipt_path, receipt, exclusive=True)
    print(
        json.dumps(
            {
                "passed": passed,
                "angles": list(ANGLES),
                "receipt_sha256": sha256_file(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

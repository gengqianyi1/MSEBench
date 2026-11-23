"""Legacy v1 oracle producer retained for provenance, disabled in v1.1.

The v1.1 campaign consumes the byte-identical, independently validated v1
oracle.  Regeneration from this sibling project is intentionally forbidden so
model outcomes cannot cause post-hoc oracle changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from tools.science import (
    HARTREE_TO_KCAL_MOL,
    canonical_json_bytes,
    run_native_scan,
    sha256_file,
    verify_attempt_manifest,
    xtb_version_record,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "evaluator_private"
ATTEMPTS = PRIVATE / "oracle_attempts"
REFERENCE = ROOT / "benchmark_case" / "reference.xyz"


def _load_result(path: Path) -> dict:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def _artifact_hashes(path: Path) -> dict[str, str]:
    return {
        member.name: sha256_file(member)
        for member in sorted(path.iterdir(), key=lambda item: item.name)
        if member.is_file()
    }


def main() -> int:
    if ROOT.name == "adaptive_xtb_torsion_profile_agent_v1_1":
        raise RuntimeError(
            "V1_1_FROZEN_ORACLE_REGENERATION_DISABLED:"
            "use reports/ORACLE_V1_1_ASSET_VALIDATION.json"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    points = []
    all_attempts = []
    retry_count = 0
    for angle in range(0, 360, 5):
        selected = None
        for attempt_number in (1, 2):
            attempt_id = f"angle_{angle:03d}_attempt_{attempt_number:02d}"
            attempt = ATTEMPTS / attempt_id
            if attempt.is_dir():
                if not args.resume:
                    raise FileExistsError(f"oracle attempt already exists: {attempt_id}")
                if not verify_attempt_manifest(attempt):
                    raise RuntimeError(f"oracle attempt manifest invalid: {attempt_id}")
                record = _load_result(attempt)
            else:
                if attempt_number == 2:
                    retry_count += 1
                record = run_native_scan(
                    reference_path=REFERENCE,
                    angle_deg=angle,
                    attempt_root=attempt,
                    timeout_seconds=180,
                )
            attempt_entry = {
                "attempt_id": attempt_id,
                "angle_deg": angle,
                "attempt_number": attempt_number,
                "profile_eligible": bool(record.get("profile_eligible")),
                "return_code": record.get("return_code"),
                "timed_out": record.get("timed_out"),
                "converged": record.get("converged"),
                "total_energy_hartree": record.get("total_energy_hartree"),
                "final_angle_deg": record.get("final_angle_deg"),
                "angle_error_deg": record.get("angle_error_deg"),
                "reference_sha256": record.get("reference_sha256"),
                "input_sha256": record.get("prepared_input", {}).get("input_sha256"),
                "attempt_manifest_sha256": sha256_file(attempt / "attempt_manifest.json"),
                "artifact_sha256": _artifact_hashes(attempt),
            }
            all_attempts.append(attempt_entry)
            if record.get("profile_eligible") is True:
                selected = attempt_entry
                break
        if selected is None:
            continue
        points.append(selected)
        print(
            f"angle={angle:03d} attempt={selected['attempt_number']} "
            f"energy={selected['total_energy_hartree']:.12f} "
            f"error={selected['angle_error_deg']:.6f}"
        )
    success = len(points)
    energies = [float(point["total_energy_hartree"]) for point in points]
    if success != 72 or not all(math.isfinite(value) for value in energies):
        quality = {
            "schema_version": "1.0",
            "status": "FAIL",
            "successful_angles": success,
            "failed_angles": 72 - success,
            "retry_count": retry_count,
            "duration_seconds": round(time.monotonic() - started, 6),
        }
        target = ROOT / "reports" / "oracle_quality.json"
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(quality))
        return 2
    minimum = min(energies)
    relative = [(value - minimum) * HARTREE_TO_KCAL_MOL for value in energies]
    oracle = {
        "schema_version": "1.0",
        "task_id": "adaptive_xtb_torsion_profile_agent_v1",
        "case_id": "2_fluorobiphenyl_central_torsion_v1",
        "protocol": "adaptive_torsion_sampling_tool_loop/1.0",
        "method": "GFN2-xTB",
        "grid_angles_deg": [point["angle_deg"] for point in points],
        "total_energies_hartree": energies,
        "relative_energies_kcal_mol": relative,
        "selected_attempt_ids": [point["attempt_id"] for point in points],
    }
    oracle_path = PRIVATE / "oracle_profile.json"
    if oracle_path.exists():
        if not args.resume or oracle_path.read_bytes() != canonical_json_bytes(oracle):
            raise FileExistsError("refusing to replace a different frozen oracle")
    else:
        with oracle_path.open("xb") as handle:
            handle.write(canonical_json_bytes(oracle))
    provenance = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "xtb": xtb_version_record(),
        "reference_xyz_sha256": sha256_file(REFERENCE),
        "scientific_invariants": {
            "method": "GFN2-xTB",
            "phase": "gas",
            "charge": 0,
            "uhf": 0,
            "threads": 1,
            "angle_grid_deg": 5,
            "requested_final_angle_tolerance_deg": 2.0,
            "independent_reference_initialization": True,
        },
        "attempts": all_attempts,
    }
    provenance_path = PRIVATE / "oracle_provenance.json"
    if provenance_path.exists():
        if not args.resume:
            raise FileExistsError("oracle provenance already exists")
    else:
        with provenance_path.open("xb") as handle:
            handle.write(canonical_json_bytes(provenance))
    commitment = {
        "schema_version": "1.0",
        "algorithm": "SHA-256",
        "oracle_profile_sha256": sha256_file(oracle_path),
        "oracle_provenance_sha256": sha256_file(provenance_path),
        "reference_xyz_sha256": sha256_file(REFERENCE),
        "point_count": 72,
    }
    commitment_path = PRIVATE / "oracle_commitment.json"
    if commitment_path.exists():
        if not args.resume or commitment_path.read_bytes() != canonical_json_bytes(commitment):
            raise FileExistsError("refusing to replace a different oracle commitment")
    else:
        with commitment_path.open("xb") as handle:
            handle.write(canonical_json_bytes(commitment))
    quality = {
        "schema_version": "1.0",
        "status": "PASS",
        "successful_angles": 72,
        "failed_angles": 0,
        "retry_count": retry_count,
        "native_attempt_count": len(all_attempts),
        "all_finite": all(math.isfinite(value) for value in relative),
        "maximum_final_angle_error_deg": max(float(point["angle_error_deg"]) for point in points),
        "all_reference_hashes_identical": {
            point["reference_sha256"] for point in points
        } == {sha256_file(REFERENCE)},
        "all_attempt_manifests_valid": all(
            verify_attempt_manifest(ATTEMPTS / attempt["attempt_id"]) for attempt in all_attempts
        ),
        "oracle_profile_sha256": sha256_file(oracle_path),
        "oracle_commitment_sha256": sha256_file(commitment_path),
        "duration_seconds": round(time.monotonic() - started, 6),
    }
    quality_path = ROOT / "reports" / "oracle_quality.json"
    if quality_path.exists():
        quality_path.unlink()
    with quality_path.open("xb") as handle:
        handle.write(canonical_json_bytes(quality))
    print(json.dumps(quality, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

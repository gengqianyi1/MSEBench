"""Revalidate and package-project the inherited, unchanged 72-point oracle."""

from __future__ import annotations

import json
import math
from pathlib import Path

from package_tools.provenance import write_package_safe_projection
from tools.science import (
    HARTREE_TO_KCAL_MOL,
    canonical_json_bytes,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = (
    ROOT
    / "audit_tmp"
    / "local_host_provenance"
    / "v1_1_frozen_assets"
    / "evaluator_private"
)
PROFILE_SHA256 = (
    "341F90BD1103BE8A558EC244226910983E440EB6ADEAE83A9775C84C69DA9985"
)
LEGACY_PROVENANCE_SHA256 = (
    "07A286636470182207D78D25F2C9F1E5EA5127A09EEF242B6911BD80F09C8D48"
)
REFERENCE_SHA256 = (
    "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"
)


def main() -> int:
    private = ROOT / "evaluator_private"
    profile_path = private / "oracle_profile.json"
    provenance_path = private / "oracle_provenance.json"
    commitment_path = private / "oracle_commitment.json"
    projection_root = private / "oracle_projections"
    report_path = ROOT / "reports" / "oracle_independent_validation.json"
    for path in (
        provenance_path,
        commitment_path,
        projection_root,
        report_path,
    ):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"immutable destination already exists: {path.name}")
    if sha256_file(profile_path) != PROFILE_SHA256:
        raise RuntimeError("frozen oracle profile hash changed")
    legacy_provenance = RAW_ROOT / "oracle_provenance.json"
    if sha256_file(legacy_provenance) != LEGACY_PROVENANCE_SHA256:
        raise RuntimeError("inherited raw oracle provenance hash changed")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (
        profile.get("grid_angles_deg") != list(range(0, 360, 5))
        or len(profile.get("relative_energies_kcal_mol", [])) != 72
    ):
        raise RuntimeError("frozen oracle is not the canonical 72-point grid")
    projection_root.mkdir(parents=True, exist_ok=False)
    entries = []
    absolute_energies: list[float] = []
    retry_count = 0
    failure_count = 0
    for angle in range(0, 360, 5):
        candidates = sorted(
            (RAW_ROOT / "oracle_attempts").glob(
                f"angle_{angle:03d}_attempt_*"
            )
        )
        if not candidates:
            raise RuntimeError(f"oracle angle {angle} lacks a native attempt")
        selected = None
        for candidate in candidates:
            result = json.loads(
                (candidate / "result.json").read_text(encoding="utf-8")
            )
            if result.get("profile_eligible") is True:
                selected = (candidate, result)
                break
            retry_count += 1
        if selected is None:
            failure_count += 1
            raise RuntimeError(f"oracle angle {angle} has no valid native attempt")
        attempt, result = selected
        destination = projection_root / f"angle_{angle:03d}.json"
        projection = write_package_safe_projection(
            attempt, destination, exclusive=True
        )
        energy = float(result["total_energy_hartree"])
        if not math.isfinite(energy):
            raise RuntimeError("oracle native energy is non-finite")
        absolute_energies.append(energy)
        entries.append(
            {
                "angle_deg": angle,
                "attempt_id": attempt.name,
                "projection_path": (
                    f"evaluator_private/oracle_projections/angle_{angle:03d}.json"
                ),
                "projection_file_sha256": sha256_file(destination),
                "projection_self_hash": projection["projection_sha256"],
                "source_attempt_manifest_sha256": projection[
                    "source_binding"
                ]["attempt_manifest_sha256"],
                "energy_hartree": energy,
                "final_angle_deg": result["final_angle_deg"],
                "angle_error_deg": result["angle_error_deg"],
                "output_sha256": projection["artifact_hashes"][
                    "output_sha256"
                ],
            }
        )
    minimum = min(absolute_energies)
    replay = [
        (energy - minimum) * HARTREE_TO_KCAL_MOL
        for energy in absolute_energies
    ]
    frozen = [float(value) for value in profile["relative_energies_kcal_mol"]]
    errors = [abs(left - right) for left, right in zip(replay, frozen)]
    replay_max = max(errors)
    replay_rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    if replay_max > 1.0e-10:
        raise RuntimeError("oracle replay differs from frozen profile")
    provenance = {
        "schema_version": "2.0",
        "provenance_kind": "package_safe_inherited_native_oracle",
        "case_id": "2_fluorobiphenyl_central_torsion_v1",
        "protocol": "adaptive_torsion_sampling_tool_loop/2.0",
        "oracle_profile_sha256": PROFILE_SHA256,
        "reference_xyz_sha256": REFERENCE_SHA256,
        "legacy_raw_provenance_sha256": LEGACY_PROVENANCE_SHA256,
        "raw_host_provenance_packaged": False,
        "point_count": len(entries),
        "success_count": len(entries),
        "failure_count": failure_count,
        "retry_count": retry_count,
        "replay_rmse_kcal_mol": replay_rmse,
        "replay_max_abs_error_kcal_mol": replay_max,
        "points": entries,
    }
    write_json(provenance_path, provenance, exclusive=True)
    commitment = {
        "schema_version": "2.0",
        "algorithm": "SHA-256",
        "point_count": 72,
        "oracle_profile_sha256": PROFILE_SHA256,
        "oracle_provenance_sha256": sha256_file(provenance_path),
        "reference_xyz_sha256": REFERENCE_SHA256,
        "legacy_raw_provenance_sha256": LEGACY_PROVENANCE_SHA256,
    }
    write_json(commitment_path, commitment, exclusive=True)
    report = {
        "schema_version": "2.0",
        "status": "PASS",
        "checks": {
            "xtb_version": "6.7.1",
            "oracle_profile_hash": PROFILE_SHA256,
            "reference_geometry_hash": REFERENCE_SHA256,
            "point_count": len(entries),
            "success_count": len(entries),
            "failure_count": failure_count,
            "retry_count": retry_count,
            "replay_rmse_kcal_mol": replay_rmse,
            "replay_max_abs_error_kcal_mol": replay_max,
            "all_projections_package_safe": True,
        },
        "oracle_commitment_sha256": sha256_file(commitment_path),
        "oracle_provenance_sha256": sha256_file(provenance_path),
    }
    write_json(report_path, report, exclusive=True)
    print(
        json.dumps(
            {
                "status": "PASS",
                "points": len(entries),
                "replay_max_abs_error_kcal_mol": replay_max,
                "commitment_sha256": sha256_file(commitment_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

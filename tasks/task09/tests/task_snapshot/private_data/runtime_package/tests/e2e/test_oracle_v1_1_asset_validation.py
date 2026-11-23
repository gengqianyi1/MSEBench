from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluator.evaluate import _read_xyz, _validate_attempt
from tools.science import HARTREE_TO_KCAL_MOL, sha256_file, verify_attempt_manifest


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "ORACLE_V1_1_ASSET_VALIDATION.json"
REFERENCE_SHA256 = "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"
PROFILE_SHA256 = "341F90BD1103BE8A558EC244226910983E440EB6ADEAE83A9775C84C69DA9985"
PROVENANCE_SHA256 = "07A286636470182207D78D25F2C9F1E5EA5127A09EEF242B6911BD80F09C8D48"
XTB_SHA256 = "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"


def _inventory_digest(root: Path) -> tuple[int, str]:
    entries = []
    for path in sorted(
        (member for member in root.rglob("*") if member.is_file()),
        key=lambda member: member.relative_to(root).as_posix(),
    ):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return len(entries), hashlib.sha256(payload).hexdigest().upper()


def test_frozen_v1_oracle_assets_are_complete_committed_and_replayable() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["scope"]["oracle_regenerated"] is False
    assert report["byte_identity"]["all_key_assets_byte_identical"] is True
    assert report["byte_identity"]["all_trees_byte_identical"] is True

    for asset in report["byte_identity"]["key_assets"]:
        path = ROOT / asset["path"]
        assert path.stat().st_size == asset["size_bytes"]
        assert sha256_file(path) == asset["sha256"] == asset["v1_sha256"]
        assert asset["byte_identical"] is True

    expected_trees = {
        item["relative_root"]: (item["new_file_count"], item["new_inventory_sha256"])
        for item in report["byte_identity"]["trees"]
    }
    assert _inventory_digest(ROOT / "evaluator_private" / "oracle_attempts") == expected_trees[
        "evaluator_private/oracle_attempts"
    ]
    assert _inventory_digest(ROOT / "benchmark_case" / "smoke_attempts") == expected_trees[
        "benchmark_case/smoke_attempts"
    ]

    oracle_path = ROOT / "evaluator_private" / "oracle_profile.json"
    provenance_path = ROOT / "evaluator_private" / "oracle_provenance.json"
    commitment_path = ROOT / "evaluator_private" / "oracle_commitment.json"
    reference_path = ROOT / "benchmark_case" / "reference.xyz"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    assert sha256_file(reference_path) == REFERENCE_SHA256
    assert sha256_file(oracle_path) == PROFILE_SHA256
    assert sha256_file(provenance_path) == PROVENANCE_SHA256
    assert commitment == {
        "algorithm": "SHA-256",
        "oracle_profile_sha256": PROFILE_SHA256,
        "oracle_provenance_sha256": PROVENANCE_SHA256,
        "point_count": 72,
        "reference_xyz_sha256": REFERENCE_SHA256,
        "schema_version": "1.0",
    }
    assert oracle["grid_angles_deg"] == list(range(0, 360, 5))

    symbols, _ = _read_xyz(reference_path)
    structure = json.loads(
        (ROOT / "benchmark_case" / "reference_structure.json").read_text(encoding="utf-8")
    )
    expected_bonds = {tuple(pair) for pair in structure["bonds_zero_based"]}
    expected_ids = [f"angle_{angle:03d}_attempt_01" for angle in range(0, 360, 5)]
    attempt_roots = sorted(
        (ROOT / "evaluator_private" / "oracle_attempts").iterdir(),
        key=lambda path: path.name,
    )
    assert [path.name for path in attempt_roots] == expected_ids
    attempts = [
        _validate_attempt(
            path, reference_symbols=symbols, expected_bonds=expected_bonds
        )
        for path in attempt_roots
    ]
    assert len(attempts) == 72
    assert all(attempt["valid"] for attempt in attempts)
    assert all(attempt["manifest_valid"] for attempt in attempts)
    assert max(attempt["angle_error_deg"] for attempt in attempts) == (
        report["oracle_validation"]["maximum_final_angle_error_deg"]
    )

    provenance_by_id = {
        attempt["attempt_id"]: attempt for attempt in provenance["attempts"]
    }
    assert set(provenance_by_id) == set(expected_ids)
    for root, attempt in zip(attempt_roots, attempts, strict=True):
        frozen = provenance_by_id[root.name]
        assert frozen["attempt_manifest_sha256"] == sha256_file(
            root / "attempt_manifest.json"
        )
        assert frozen["input_sha256"] == sha256_file(root / "input.xyz")
        assert frozen["reference_sha256"] == REFERENCE_SHA256
        assert frozen["total_energy_hartree"] == attempt["energy_hartree"]
        assert frozen["profile_eligible"] is True
        actual_artifacts = {
            member.name: sha256_file(member)
            for member in root.iterdir()
            if member.is_file()
        }
        assert frozen["artifact_sha256"] == dict(sorted(actual_artifacts.items()))

    absolute = np.asarray([attempt["energy_hartree"] for attempt in attempts])
    relative = (absolute - float(np.min(absolute))) * HARTREE_TO_KCAL_MOL
    assert np.array_equal(
        absolute, np.asarray(oracle["total_energies_hartree"], dtype=float)
    )
    assert np.array_equal(
        relative, np.asarray(oracle["relative_energies_kcal_mol"], dtype=float)
    )
    assert oracle["selected_attempt_ids"] == expected_ids

    smoke_report = json.loads(
        (ROOT / "reports" / "gate1_scientific_smoke.json").read_text(encoding="utf-8")
    )
    assert smoke_report["passed"] is True
    assert [row["angle_deg"] for row in smoke_report["smoke_angles"]] == [
        0,
        90,
        180,
        270,
    ]
    for row in smoke_report["smoke_angles"]:
        angle = row["angle_deg"]
        root = (
            ROOT
            / "benchmark_case"
            / "smoke_attempts"
            / f"angle_{angle:03d}"
            / "attempt_001"
        )
        assert verify_attempt_manifest(root)
        assert row["attempt_manifest_sha256"] == sha256_file(
            root / "attempt_manifest.json"
        )
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        reported_result = dict(row)
        reported_result.pop("attempt_manifest_sha256")
        assert reported_result == result
        assert result["return_code"] == 0
        assert result["converged"] is True
        assert result["profile_eligible"] is True
        assert result["structure_validation"]["composition_ok"] is True
        assert result["structure_validation"]["atom_order_ok"] is True
        assert result["structure_validation"]["connectivity_ok"] is True
        assert result["xtb"]["version"] == "6.7.1"
        assert result["xtb"]["binary_sha256"] == XTB_SHA256

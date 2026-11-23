from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from rdkit.Chem import rdMolTransforms

from tools.science import (
    AngleValidationError,
    MAPPED_SMILES,
    TARGET_DIHEDRAL_MAPS,
    circular_distance_deg,
    constraint_text,
    map_to_index,
    molecule_from_reference,
    molecule_signature,
    prepare_angle_input,
    sha256_file,
    target_indices,
    validate_angle,
    xtb_version_record,
)
from package_tools.provenance import validate_package_safe_projection


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_mapping_reference_and_composition() -> None:
    reference = ROOT / "benchmark_case" / "reference.xyz"
    molecule = molecule_from_reference(reference)
    signature = molecule_signature(molecule)
    assert MAPPED_SMILES.startswith("[F:1]")
    assert list(TARGET_DIHEDRAL_MAPS) == [6, 7, 8, 9]
    assert map_to_index(molecule) == {value: value - 1 for value in range(1, 14)}
    assert target_indices(molecule) == (5, 6, 7, 8)
    assert target_indices(molecule, one_based=True) == (6, 7, 8, 9)
    assert signature["composition"] == {"C": 12, "F": 1, "H": 9}
    assert signature["formal_charge"] == 0
    assert sha256_file(reference) == "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"


@pytest.mark.parametrize("angle", list(range(0, 360, 5)))
def test_every_grid_angle_is_valid(angle: int) -> None:
    assert validate_angle(angle) == angle


@pytest.mark.parametrize("angle", [-5, 360, 1, 2.5, float("nan"), True, "90"])
def test_illegal_angles_are_rejected(angle: object) -> None:
    with pytest.raises(AngleValidationError):
        validate_angle(angle)


def test_circular_distance_and_native_signed_constraint() -> None:
    assert circular_distance_deg(359, 1) == pytest.approx(2)
    assert circular_distance_deg(0, 180) == pytest.approx(180)
    assert "dihedral: 6, 7, 8, 9, -90.0" in constraint_text(270)


def test_constrained_uff_is_independent_and_hits_target(tmp_path: Path) -> None:
    reference = ROOT / "benchmark_case" / "reference.xyz"
    output = tmp_path / "input.xyz"
    prepared = prepare_angle_input(reference, 35, output)
    assert prepared["reference_sha256"] == sha256_file(reference)
    assert prepared["uff_angle_error_deg"] <= 0.05
    molecule = molecule_from_reference(output)
    measured = rdMolTransforms.GetDihedralDeg(molecule.GetConformer(), *target_indices(molecule)) % 360
    assert circular_distance_deg(measured, 35) <= 0.05
    assert sha256_file(reference) == "9225047D010D3D5640C108C1CD15CA99CB3E8A6BCEB19DD606B4071FA9FE86E4"


def test_gate1_native_smoke_receipt_and_package_safe_provenance() -> None:
    receipt = json.loads((ROOT / "reports" / "gate1_scientific_smoke.json").read_text(encoding="utf-8"))
    assert receipt["passed"] is True
    assert [record["angle_deg"] for record in receipt["smoke_angles"]] == [0, 90, 180, 270]
    for record in receipt["smoke_angles"]:
        assert record["return_code"] == 0
        assert record["converged"] is True
        assert record["profile_eligible"] is True
        assert math.isfinite(record["total_energy_hartree"])
        assert record["angle_error_deg"] <= 2.0
        assert all(
            record["structure_validation"][key] is True
            for key in ("atom_count_ok", "atom_order_ok", "composition_ok", "formal_charge_ok", "connectivity_ok")
        )
        projection_path = ROOT / record["package_safe_projection"]
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        assert validate_package_safe_projection(projection)
        assert sha256_file(projection_path) == record[
            "package_safe_projection_sha256"
        ]
        assert record["raw_attempt_manifest_valid"] is True
    assert receipt["raw_host_provenance_packaged"] is False


def test_exact_native_executable_identity() -> None:
    record = xtb_version_record()
    assert record["version"] == "6.7.1"
    assert record["binary_sha256"] == "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"

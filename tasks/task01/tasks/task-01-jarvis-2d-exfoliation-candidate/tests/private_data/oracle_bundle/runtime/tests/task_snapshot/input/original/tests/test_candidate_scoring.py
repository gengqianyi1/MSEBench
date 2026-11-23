import json
import gzip
from pathlib import Path

import pytest

pymatgen = pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure

from src.candidate_scoring import (
    CANDIDATE_WEIGHTS,
    _geometry_score,
    _normalize_2d_vacuum,
    _pathology_reason,
    _vacuum_axis_info,
)
from src.chgnet_stability import score_selected_candidates


ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = json.loads((ROOT / "data/candidate_generation_constraints.json").read_text())


def test_vacuum_normalization_preserves_composition_and_site_count():
    structure = Structure(
        Lattice.from_parameters(3.2, 3.2, 28.0, 90, 90, 120),
        ["Mo", "S", "S"],
        [[0, 0, 0.5], [1 / 3, 2 / 3, 0.47], [2 / 3, 1 / 3, 0.53]],
    )
    normalized = _normalize_2d_vacuum(structure)
    assert normalized.composition == structure.composition
    assert len(normalized) == len(structure)
    assert _vacuum_axis_info(normalized)["vacuum_gap"] >= 19.0


def test_reasonable_layer_has_nonzero_geometry_score():
    structure = Structure(
        Lattice.from_parameters(3.2, 3.2, 25.0, 90, 90, 120),
        ["Mo", "S", "S"],
        [[0, 0, 0.5], [1 / 3, 2 / 3, 0.46], [2 / 3, 1 / 3, 0.54]],
    )
    score, details = _geometry_score(structure, CONSTRAINTS)
    assert 0.0 < score <= 1.0
    assert details["vacuum_gap_angstrom"] > 10.0


def test_candidate_weights_sum_to_fifty():
    assert sum(CANDIDATE_WEIGHTS.values()) == 50.0


def test_overlapping_periodic_sites_are_pathological():
    structure = Structure(
        Lattice.from_parameters(3.2, 3.2, 25.0, 90, 90, 120),
        ["Mo", "S"],
        [[0, 0, 0.5], [0.001, 0.001, 0.5]],
    )
    assert _pathology_reason(structure, CONSTRAINTS) in {
        "absolute_atomic_overlap",
        "covalent_radius_overlap",
    }


def test_reasonable_layer_is_not_pathological():
    structure = Structure(
        Lattice.from_parameters(3.2, 3.2, 25.0, 90, 90, 120),
        ["Mo", "S", "S"],
        [[0, 0, 0.5], [1 / 3, 2 / 3, 0.46], [2 / 3, 1 / 3, 0.54]],
    )
    assert _pathology_reason(structure, CONSTRAINTS) is None


def test_constant_candidate_predictions_disable_chgnet_selection(tmp_path):
    candidate_path = tmp_path / "candidate_library.json.gz"
    with gzip.open(candidate_path, "wt", encoding="utf-8") as handle:
        for index in range(1000):
            handle.write(
                json.dumps(
                    {
                        "candidate_id": f"CAND-{index:06d}",
                        "predicted_exfoliation_energy_mev_atom": 0.0,
                    }
                )
                + "\n"
            )
    result = score_selected_candidates(
        selected_candidates=[
            {"candidate_id": f"CAND-{index:06d}"} for index in range(5)
        ],
        candidate_path=candidate_path,
        constraints_path=ROOT / "data" / "candidate_generation_constraints.json",
        output_dir=tmp_path / "chgnet",
        tool_attempts=1,
    )
    assert result["score"] == 0.0
    assert result["unique_prediction_values"] == 1
    assert "constant" in result["reason"]

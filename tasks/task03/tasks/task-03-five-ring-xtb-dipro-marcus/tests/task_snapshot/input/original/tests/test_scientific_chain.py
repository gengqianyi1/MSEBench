import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemistry import validate_smiles_rdkit
from geometry import XYZ, align_by_principal_axes
from scoring import rate_performance_score
from xtb_runner import marcus_hole_rate


def _task_config():
    return yaml.safe_load((ROOT / "tasks" / "five_ring_xtb_dipro_marcus" / "task_config.yaml").read_text(encoding="utf-8"))


def test_marcus_numeric_fixture_and_continuous_score_boundaries():
    cfg = _task_config()
    rate = marcus_hole_rate(0.250, 0.356, cfg["marcus"])
    assert np.isclose(rate, 3.78e14, rtol=0.02)
    scoring = cfg["scoring"]
    assert rate_performance_score(9.9e13, scoring) == 0.0
    assert rate_performance_score(1.0e14, scoring) == 10.0
    assert rate_performance_score(2.5e14, scoring) == 25.0
    assert rate_performance_score(1.0e15, scoring) == 100.0


def test_five_fused_ring_gate_rejects_nonfused_and_spiro_controls():
    constraints = _task_config()["constraints"]
    pentacene = validate_smiles_rdkit("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1", constraints)
    biphenyl = validate_smiles_rdkit("c1ccccc1-c2ccccc2", constraints)
    spiro = validate_smiles_rdkit("C1CCC2(CC1)CCCC2", constraints)
    assert pentacene.five_ring_fused_core
    assert not biphenyl.five_ring_fused_core
    assert not spiro.five_ring_fused_core


def test_principal_axis_alignment_is_invariant_to_eigenvector_signs():
    xyz = XYZ(
        elements=["C", "C", "N", "S"],
        coords=np.array([[0.0, 0.0, 0.0], [2.0, 0.1, 0.0], [0.3, 1.2, 0.2], [-0.2, 0.4, 1.5]]),
    )
    reference = align_by_principal_axes(xyz).coords
    original = np.linalg.eigh

    def sign_flipped(matrix):
        values, vectors = original(matrix)
        return values, vectors * np.array([-1.0, 1.0, -1.0])

    with patch("geometry.np.linalg.eigh", side_effect=sign_flipped):
        alternate = align_by_principal_axes(xyz).coords
    assert np.allclose(reference, alternate)


def test_packing_grid_brackets_original_lower_boundary():
    dz = _task_config()["packing"]["search"]["dz_values_A"]
    assert min(dz) == 3.0
    assert max(dz) == 4.0
    assert len(dz) == 6

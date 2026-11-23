import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path

from src.private_scoring import METRIC_WEIGHTS, piecewise_anchor_fraction, score_private, validate_private_labels


def _metadata():
    lower = {"direction": "lower", "floor": 100.0, "strong": 30.0, "ceiling": 0.1}
    higher = {"direction": "higher", "floor": 0.0, "strong": 0.6, "ceiling": 0.999}
    metrics = {
        name: dict(lower if name.endswith("mae") or name.endswith("rmse") else higher)
        for name in METRIC_WEIGHTS
    }
    return {
        "scoring_reference": {
            "status": "frozen_measured",
            "version": "test-v1",
            "strong_anchor_fraction": 0.65,
            "metrics": metrics,
        }
    }


def _labels_and_predictions():
    ids = [f"s{i:02d}" for i in range(20)]
    y = np.linspace(10, 100, 20)
    split = ["iid"] * 10 + ["chemical_system_ood"] * 10
    labels = pd.DataFrame({
        "sample_id": ids,
        "true_exfoliation_energy_mev_atom": y,
        "split_type": split,
        "score_regression": True,
        "invariance_group": ["g0"] + [None] * 19,
        "variant_type": ["original"] * 20,
    })
    variants = pd.DataFrame({
        "sample_id": ["s00_v1", "s00_v2"],
        "true_exfoliation_energy_mev_atom": [y[0], y[0]],
        "split_type": ["invariance", "invariance"],
        "score_regression": [False, False],
        "invariance_group": ["g0", "g0"],
        "variant_type": ["site_permutation", "origin_translation"],
    })
    labels = pd.concat([labels, variants], ignore_index=True)
    good = pd.DataFrame({
        "sample_id": labels.sample_id,
        "predicted_exfoliation_energy_mev_atom": labels.true_exfoliation_energy_mev_atom,
    })
    bad_values = list(reversed(y.tolist())) + [80.0, 90.0]
    bad = pd.DataFrame({
        "sample_id": labels.sample_id,
        "predicted_exfoliation_energy_mev_atom": bad_values,
    })
    return labels, good, bad


def test_piecewise_anchor_fraction_hits_registered_points():
    lower = {"direction": "lower", "floor": 10.0, "strong": 5.0, "ceiling": 1.0}
    assert piecewise_anchor_fraction(10.0, lower) == 0.0
    assert np.isclose(piecewise_anchor_fraction(5.0, lower), 0.65)
    assert piecewise_anchor_fraction(1.0, lower) == 1.0


def test_perfect_predictions_score_better_than_bad():
    labels, good, bad = _labels_and_predictions()
    good_score = score_private(good, labels, _metadata())["numeric_score"]
    bad_score = score_private(bad, labels, _metadata())["numeric_score"]
    assert good_score > bad_score
    assert good_score == sum(METRIC_WEIGHTS.values())


def test_constant_predictions_do_not_get_perfect_invariance_credit():
    labels, good, _ = _labels_and_predictions()
    constant = good.copy()
    constant["predicted_exfoliation_energy_mev_atom"] = 50.0
    raw = score_private(constant, labels, _metadata())["raw_metrics"]
    assert raw["invariance_joint_mae"] > 0


def test_provisional_anchors_are_rejected():
    labels, good, _ = _labels_and_predictions()
    meta = {"scoring_reference": {"status": "provisional_unmeasured", "metrics": {}}}
    try:
        score_private(good, labels, meta)
    except ValueError as exc:
        assert "frozen_measured" in str(exc)
    else:
        raise AssertionError("provisional anchors must not be scoreable")


def test_private_label_contract_rejects_missing_group_metadata():
    labels, _, _ = _labels_and_predictions()
    labels = labels.drop(columns=["variant_type"])
    try:
        validate_private_labels(labels)
    except ValueError as exc:
        assert "variant_type" in str(exc)
    else:
        raise AssertionError("incomplete private-label metadata must be rejected")


def test_property_weights_are_synchronized_across_code_config_and_metadata():
    root = Path(__file__).resolve().parents[1]
    task = yaml.safe_load((root / "tasks/jarvis_2d_exfoliation/task_config.yaml").read_text())
    metadata = json.loads((root / "data/dataset_metadata.json").read_text())
    assert task["scoring"]["property_private_metrics"] == METRIC_WEIGHTS
    assert metadata["property_metric_weights"] == METRIC_WEIGHTS

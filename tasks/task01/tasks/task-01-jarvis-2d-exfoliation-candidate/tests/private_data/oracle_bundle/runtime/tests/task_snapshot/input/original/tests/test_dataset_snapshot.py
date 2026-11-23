import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read_public(name: str) -> pd.DataFrame:
    return pd.read_json(ROOT / "data" / name, orient="records", lines=True, compression="infer")


def test_public_snapshot_metadata_matches_files():
    metadata = json.loads((ROOT / "data" / "dataset_metadata.json").read_text(encoding="utf-8"))
    train = _read_public("train.json.gz")
    validation = _read_public("validation.json.gz")
    test_public = _read_public("test_public.json.gz")

    assert metadata["snapshot_id"] == "jarvis2d-exfoliation-candidate-v0.4.0"
    assert metadata["counts"]["train"] == len(train) == 538
    assert metadata["counts"]["validation"] == len(validation) == 60
    assert metadata["counts"]["test_public_total"] == len(test_public) == 226
    assert "JSON Lines" in metadata["file_format"]["train.json.gz"]
    assert metadata["scoring_reference"]["status"] == "frozen_measured"
    assert metadata["scoring_reference"]["strong_anchor_fraction"] == 0.65
    assert "invariance_joint_mae" in metadata["scoring_reference"]["metrics"]
    assert metadata["candidate_generation"]["required_count"] == 1000
    assert metadata["candidate_generation"]["property_used_for_candidate_scoring"] is False
    assert (ROOT / "data/candidate_generation_constraints.json").is_file()


def test_public_test_ids_are_opaque():
    test_public = _read_public("test_public.json.gz")
    ids = test_public["sample_id"].astype(str)

    assert ids.is_unique
    assert ids.str.fullmatch(r"J2D-T-\d{6}").all()
    assert not ids.str.contains("__|JVASP|translated|supercell|variant", regex=True).any()


def test_private_group_split_invariants_when_labels_are_available():
    label_path = ROOT / "private" / "test_labels_private.json.gz"
    if not label_path.exists():
        label_path = ROOT.parent / "_private_task2_calibration" / "test_labels_private.json.gz"
    if not label_path.exists():
        pytest.skip("private labels are evaluator-only and not present in public releases")

    train = _read_public("train.json.gz")
    validation = _read_public("validation.json.gz")
    test_public = _read_public("test_public.json.gz")
    labels = pd.read_json(label_path, orient="records", lines=True, compression="infer")
    public_with_labels = test_public.merge(labels, on="sample_id", how="inner")
    base = public_with_labels[public_with_labels["score_regression"]]

    train_systems = set(train["chemical_system"])
    validation_systems = set(validation["chemical_system"])
    iid_systems = set(base.loc[base["split_type"] == "iid", "chemical_system"])
    ood_systems = set(base.loc[base["split_type"] == "chemical_system_ood", "chemical_system"])

    assert iid_systems <= train_systems
    assert validation_systems <= train_systems
    assert not (ood_systems & train_systems)
    assert not (ood_systems & validation_systems)

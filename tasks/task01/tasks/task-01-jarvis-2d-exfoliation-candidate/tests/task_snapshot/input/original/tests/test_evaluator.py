import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluator import evaluate_result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evaluator_recomputes_cross_file_evidence(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    train = pd.DataFrame({
        "sample_id": ["tr1", "tr2", "tr3"],
        "structure": [{}, {}, {}],
        "chemical_system": ["C-S", "C-S", "C-Se"],
        "exfoliation_energy_mev_atom": [10.0, 20.0, 30.0],
    })
    val = pd.DataFrame({
        "sample_id": ["v1", "v2"],
        "structure": [{}, {}],
        "chemical_system": ["C-S", "C-Se"],
        "exfoliation_energy_mev_atom": [40.0, 50.0],
    })
    test = pd.DataFrame({
        "sample_id": [f"t{i:02d}" for i in range(10)],
        "structure": [{} for _ in range(10)],
        "chemical_system": ["C-S"] * 10,
    })
    for name, frame in (("train.json.gz", train), ("validation.json.gz", val), ("test_public.json.gz", test)):
        frame.to_json(data / name, orient="records", lines=True, compression="gzip")
    hashes = {name: _sha256(data / name) for name in ("train.json.gz", "validation.json.gz", "test_public.json.gz")}
    (data / "dataset_metadata.json").write_text(json.dumps({
        "snapshot_id": "snap-test",
        "public_file_sha256": hashes,
    }))

    predictions = pd.DataFrame({
        "sample_id": test.sample_id,
        "predicted_exfoliation_energy_mev_atom": np.arange(10, dtype=float),
    })
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    (tmp_path / "model.joblib").write_bytes(pickle.dumps({"model": "placeholder"}))
    (tmp_path / "feature_names.json").write_text(json.dumps(["f1", "f2"]))
    validation_metrics = {
        "constant_baseline_mae_mev_atom": 10.0,
        "constant_baseline_rmse_mev_atom": 12.0,
        "mae_mev_atom": 8.0,
        "rmse_mev_atom": 9.0,
        "spearman": 0.7,
        "invariance_mae_mev_atom": 1.0,
    }
    (tmp_path / "training_log.json").write_text(json.dumps({
        "package_versions": {},
        "random_seeds": [42],
        "data_hashes": hashes,
        "validation_metrics": validation_metrics,
        "selected_hyperparameters": {},
        "elapsed_seconds": 1.0,
        "refit_policy": "none",
    }))

    labeled = pd.concat([train.exfoliation_energy_mev_atom, val.exfoliation_energy_mev_atom], ignore_index=True)
    ordered = predictions.sort_values(["predicted_exfoliation_energy_mev_atom", "sample_id"]).reset_index(drop=True)
    generator_dir = tmp_path / "generator"
    generator_dir.mkdir()
    (generator_dir / "generate.py").write_text("print('placeholder')\n")
    (generator_dir / "config.json").write_text(json.dumps({"seed": 42}))
    manifest = {
        "random_seed": 42,
        "input_files": [
            {"path": "data/train.json.gz", "sha256": hashes["train.json.gz"]},
            {"path": "data/candidate_generation_constraints.json", "sha256": "0" * 64},
        ],
        "library_versions": {"python": "test"},
        "generation_parameters": {},
        "submitted_count": 1000,
        "interface_version": "candidate-generator-v1",
    }
    (generator_dir / "generation_manifest.json").write_text(json.dumps(manifest))
    candidate_records = [
        {
            "candidate_id": f"CAND-{i:06d}",
            "generation_index": i,
            "random_seed": 42,
            "structure": {"dummy": i},
            "provenance": {
                "strategy": "unit-test placeholder",
                "parent_ids": [],
                "operations": [{"name": "unit_test", "parameters": {"index": i}}],
            },
        }
        for i in range(1000)
    ]
    import gzip
    with gzip.open(tmp_path / "candidate_library.json.gz", "wt", encoding="utf-8") as handle:
        for record in candidate_records:
            handle.write(json.dumps(record) + "\n")
    candidate_summary = {
        "submitted_count": 1000,
        "claimed_valid_count": 1000,
        "claimed_unique_count": 1000,
        "claimed_novel_count": 1000,
        "random_seed": 42,
        "generation_strategy_summary": "Deterministic placeholder records for evaluator unit testing.",
        "known_limitations": ["Structures are intentionally placeholders."],
    }
    (tmp_path / "candidate_summary.json").write_text(json.dumps(candidate_summary))

    result = {
        "success": True,
        "task_name": "jarvis_2d_exfoliation_prediction_and_candidate_generation",
        "dataset": {
            "name": "jarvis_dft_2d_exfoliation_frozen",
            "snapshot_id": "snap-test",
            "target": "exfoliation_energy_mev_atom",
            "target_unit": "meV/atom",
            "n_train": len(train),
            "n_validation": len(val),
            "n_test": len(test),
            "checksums_verified": True,
        },
        "data_audit": {
            "invalid_structure_count": 15,
            "duplicate_id_count": 0,
            "missing_train_target_count": 0,
            "missing_validation_target_count": 0,
            "target_min_mev_atom": float(labeled.min()),
            "target_q10_mev_atom": float(labeled.quantile(0.1)),
            "target_median_mev_atom": float(labeled.median()),
            "target_mean_mev_atom": float(labeled.mean()),
            "target_q90_mev_atom": float(labeled.quantile(0.9)),
            "target_max_mev_atom": float(labeled.max()),
            "target_std_mev_atom": float(labeled.std(ddof=1)),
            "elements": ["C", "S", "Se"],
            "element_count": 3,
            "chemical_system_count": 2,
        },
        "model": {
            "name": "TestModel",
            "feature_source": "pymatgen_custom",
            "n_features": 2,
            "n_features_before_selection": 2,
            "hyperparameters": {},
            "random_seed": 42,
            "uses_external_pretraining": False,
            "pretraining_disclosure": "none",
            "refit_on_train_validation": False,
        },
        "validation": {"method": "fixed validation", "model_selection_rule": "lowest MAE", **validation_metrics},
        "artifacts": {
            "predictions_file": "predictions.csv",
            "model_file": "model.joblib",
            "feature_names_file": "feature_names.json",
            "training_log": "training_log.json",
            "candidate_library_file": "candidate_library.json.gz",
            "candidate_summary_file": "candidate_summary.json",
            "generator_entrypoint": "generator/generate.py",
            "generator_config": "generator/config.json",
            "generation_manifest": "generator/generation_manifest.json",
        },
        "candidate_generation": candidate_summary,
        "recommended_candidates": [
            {
                "rank": i + 1,
                "sample_id": ordered.loc[i, "sample_id"],
                "predicted_exfoliation_energy_mev_atom": float(ordered.loc[i, "predicted_exfoliation_energy_mev_atom"]),
            }
            for i in range(10)
        ],
        "scientific_interpretation": {
            "important_features": ["f1"],
            "vacuum_sensitivity_assessment": "Raw volume and out-of-plane length were excluded from the feature set.",
            "applicability_domain": "This synthetic test applies only to the structures represented by the test fixture.",
            "limitations": "This is a unit-test artifact and does not support scientific interpretation or deployment.",
        },
        "checks": {
            "used_pymatgen_structures": True,
            "trained_constant_baseline": True,
            "trained_learned_model": True,
            "performed_invariance_check": True,
            "test_labels_not_accessed": True,
            "all_test_samples_predicted": True,
            "units_are_mev_per_atom": True,
            "no_external_network_or_label_lookup": True,
            "workspace_only_file_access": True,
            "generated_exactly_1000_candidates": True,
            "candidate_generation_is_reproducible": True,
            "candidate_library_uses_periodic_structures": True,
        },
    }
    report = evaluate_result(result, workspace=tmp_path)
    assert report["hard_gate_passed"], report
    assert report["score_eligible"], report
    assert report["score"] == 0, report
    assert report["max_score"] == 0, report


def test_public_errors_create_deductions_without_global_ineligibility():
    report = evaluate_result({}, workspace=None)
    assert not report["passed"]
    assert not report["score_eligible"]
    assert report["compliance_penalty"] == report["max_compliance_penalty"] == 6.0

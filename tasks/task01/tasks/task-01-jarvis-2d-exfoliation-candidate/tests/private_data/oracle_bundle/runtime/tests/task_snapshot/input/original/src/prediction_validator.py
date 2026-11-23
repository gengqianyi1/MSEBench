from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["sample_id", "predicted_exfoliation_energy_mev_atom"]


def load_test_ids(test_file: Path) -> List[str]:
    df = pd.read_json(test_file, orient="records", lines=True, compression="infer")
    if "sample_id" not in df.columns:
        raise ValueError("test file has no sample_id column")
    return df["sample_id"].astype(str).tolist()


def validate_predictions(prediction_file: Path, test_file: Path) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not prediction_file.exists():
        return {"passed": False, "errors": [f"missing predictions file: {prediction_file}"], "warnings": []}
    try:
        pred = pd.read_csv(prediction_file)
    except Exception as exc:
        return {"passed": False, "errors": [f"could not read predictions CSV: {exc}"], "warnings": []}
    missing = [c for c in REQUIRED_COLUMNS if c not in pred.columns]
    if missing:
        errors.append(f"missing prediction columns: {missing}")
        return {"passed": False, "errors": errors, "warnings": warnings}
    pred["sample_id"] = pred["sample_id"].astype(str)
    if pred["sample_id"].duplicated().any():
        errors.append("duplicate sample_id values in predictions")
    values = pd.to_numeric(pred["predicted_exfoliation_energy_mev_atom"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        errors.append("predictions contain non-finite or non-numeric values")
    try:
        expected = load_test_ids(test_file)
    except Exception as exc:
        errors.append(f"could not read public test IDs: {exc}")
        expected = []
    expected_set = set(expected)
    got_set = set(pred["sample_id"])
    missing_ids = sorted(expected_set - got_set)
    extra_ids = sorted(got_set - expected_set)
    if missing_ids:
        errors.append(f"missing {len(missing_ids)} test predictions")
    if extra_ids:
        errors.append(f"found {len(extra_ids)} unknown sample IDs")
    if len(pred) != len(expected):
        errors.append(f"prediction row count {len(pred)} does not equal test count {len(expected)}")
    if "predicted_uncertainty_mev_atom" in pred.columns:
        unc = pd.to_numeric(pred["predicted_uncertainty_mev_atom"], errors="coerce")
        if (~np.isfinite(unc.to_numpy(dtype=float))).any() or (unc < 0).any():
            warnings.append("uncertainty column contains invalid or negative values")
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_predictions": int(len(pred)),
        "n_expected": int(len(expected)),
    }

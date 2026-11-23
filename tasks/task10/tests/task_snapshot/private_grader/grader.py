from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


def _read_predictions(path: Path) -> tuple[dict[str, float], list[str]]:
    failures: list[str] = []
    predictions: dict[str, float] = {}
    if not path.is_file():
        return {}, ["MISSING_PREDICTIONS_FILE"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "prediction_eV"]:
            return {}, ["INVALID_COLUMNS"]
        for row in reader:
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                failures.append("EMPTY_ID")
                continue
            if row_id in predictions:
                failures.append("DUPLICATE_ID")
                continue
            try:
                value = float(row.get("prediction_eV", ""))
            except (TypeError, ValueError):
                failures.append("NON_NUMERIC_PREDICTION")
                continue
            if not math.isfinite(value):
                failures.append("NON_FINITE_PREDICTION")
                continue
            predictions[row_id] = value
    return predictions, sorted(set(failures))


def _read_labels(path: Path) -> dict[str, tuple[float, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["id"]: (float(row["gap_expt_eV"]), row["split"])
            for row in csv.DictReader(handle)
        }


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def _r2(y_true: list[float], y_pred: list[float]) -> float:
    mean = sum(y_true) / len(y_true)
    denom = sum((value - mean) ** 2 for value in y_true)
    if denom == 0.0:
        return 1.0 if all(a == b for a, b in zip(y_true, y_pred)) else 0.0
    return 1.0 - sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / denom


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _linear_lower_is_better(value: float, full_at: float, zero_at: float, points: float) -> float:
    if value <= full_at:
        return points
    if value >= zero_at:
        return 0.0
    return points * (zero_at - value) / (zero_at - full_at)


def _linear_higher_is_better(value: float, zero_at: float, full_at: float, points: float) -> float:
    if value <= zero_at:
        return 0.0
    if value >= full_at:
        return points
    return points * (value - zero_at) / (full_at - zero_at)


def grade(context) -> dict:
    prediction_path = context.submission_dir / "predictions.csv"
    pipeline_path = context.submission_dir / "pipeline.json"
    labels = _read_labels(context.private_data_dir / "labels.csv")
    predictions, failures = _read_predictions(prediction_path)
    expected = set(labels)
    submitted = set(predictions)
    if expected - submitted:
        failures.append("MISSING_IDS")
    if submitted - expected:
        failures.append("UNKNOWN_IDS")
    try:
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if not isinstance(pipeline, dict):
            failures.append("INVALID_PIPELINE_METADATA")
    except Exception:
        pipeline = {}
        failures.append("INVALID_PIPELINE_METADATA")

    failures = sorted(set(failures))
    protocol_score = 10.0 if not failures else 0.0
    if expected != submitted or any(code not in {"INVALID_PIPELINE_METADATA"} for code in failures):
        return {
            "score": protocol_score,
            "subscores": {"prediction_protocol": protocol_score, "metrics": 0.0},
            "failure_codes": failures,
            "evidence": {"expected_rows": len(expected), "submitted_rows": len(submitted)},
        }

    metrics: dict[str, float] = {}
    for split in ("valid", "test"):
        ids = sorted(row_id for row_id, (_, row_split) in labels.items() if row_split == split)
        truth = [labels[row_id][0] for row_id in ids]
        pred = [predictions[row_id] for row_id in ids]
        metrics[f"{split}_MAE_eV"] = _mae(truth, pred)
        metrics[f"{split}_R2"] = _r2(truth, pred)
    for group, predicate in (("zero_gap", lambda x: x == 0.0), ("positive_gap", lambda x: x > 0.0)):
        ids = sorted(row_id for row_id, (value, split) in labels.items() if split == "test" and predicate(value))
        if ids:
            metrics[f"test_{group}_MAE_eV"] = _mae(
                [labels[row_id][0] for row_id in ids], [predictions[row_id] for row_id in ids]
            )

    subscores = {
        "prediction_protocol": protocol_score,
        "validation_MAE": _linear_lower_is_better(metrics["valid_MAE_eV"], 0.30, 0.65, 30.0),
        "test_MAE": _linear_lower_is_better(metrics["test_MAE_eV"], 0.30, 1.20, 35.0),
        "validation_R2": _linear_higher_is_better(metrics["valid_R2"], 0.55, 0.75, 25.0),
    }
    subscores = {key: round(value, 6) for key, value in subscores.items()}
    if metrics["valid_MAE_eV"] > 0.65:
        failures.append("VALID_MAE_THRESHOLD")
    if metrics["test_MAE_eV"] > 1.20:
        failures.append("TEST_MAE_THRESHOLD")
    if metrics["valid_R2"] < 0.55:
        failures.append("VALID_R2_THRESHOLD")
    return {
        "score": sum(subscores.values()),
        "subscores": subscores,
        "failure_codes": failures,
        "evidence": {
            "metrics_recomputed_by_private_grader": {k: round(v, 8) for k, v in metrics.items()},
            "prediction_sha256": _hash(prediction_path),
            "pipeline_sha256": _hash(pipeline_path),
            "n_predictions": len(predictions),
        },
    }

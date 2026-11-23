from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score

PREDICTION_COLUMN = "predicted_exfoliation_energy_mev_atom"
TARGET_COLUMN = "true_exfoliation_energy_mev_atom"
REQUIRED_PRIVATE_LABEL_COLUMNS = {
    "sample_id",
    TARGET_COLUMN,
    "split_type",
    "score_regression",
    "invariance_group",
    "variant_type",
}
ALLOWED_PRIVATE_SPLITS = {"iid", "chemical_system_ood", "invariance"}

METRIC_WEIGHTS: Dict[str, float] = {
    "overall_mae": 14.0,
    "overall_rmse": 4.0,
    "overall_spearman": 4.0,
    "ood_mae": 8.0,
    "ood_spearman": 4.0,
    "hit_rate_10pct": 4.0,
    "ndcg_10pct": 4.0,
    "low_tail_mae": 2.0,
    "high_tail_mae": 2.0,
    "invariance_joint_mae": 4.0,
}


def _safe_spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    y_rank = pd.Series(y).rank(method="average").to_numpy(float)
    p_rank = pd.Series(p).rank(method="average").to_numpy(float)
    if np.allclose(y_rank, y_rank[0]) or np.allclose(p_rank, p_rank[0]):
        return 0.0
    y_centered = y_rank - float(np.mean(y_rank))
    p_centered = p_rank - float(np.mean(p_rank))
    denom = float(np.sqrt(np.sum(y_centered ** 2) * np.sum(p_centered ** 2)))
    if denom == 0.0:
        return 0.0
    value = float(np.sum(y_centered * p_centered) / denom)
    return float(value) if np.isfinite(value) else 0.0


def hit_rate_fraction(y: np.ndarray, p: np.ndarray, fraction: float = 0.1) -> float:
    if len(y) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(y) * fraction)))
    true_best = set(np.argsort(y, kind="mergesort")[:k].tolist())
    pred_best = set(np.argsort(p, kind="mergesort")[:k].tolist())
    return float(len(true_best & pred_best) / k)


def low_energy_ndcg(y: np.ndarray, p: np.ndarray, fraction: float = 0.1) -> float:
    if len(y) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(y) * fraction)))
    cutoff = float(np.quantile(y, fraction))
    relevance = np.maximum(0.0, cutoff - y)
    if np.allclose(relevance, 0.0):
        relevance = (y <= cutoff).astype(float)
    return float(ndcg_score(relevance.reshape(1, -1), (-p).reshape(1, -1), k=k))


def _mean_pairwise_abs(values: Iterable[float]) -> float | None:
    arr = [float(x) for x in values]
    if len(arr) < 2:
        return None
    return float(np.mean([abs(a - b) for a, b in combinations(arr, 2)]))


def _base_mask(df: pd.DataFrame) -> pd.Series:
    if "score_regression" in df.columns:
        return df["score_regression"].fillna(False).astype(bool)
    if "split_type" in df.columns:
        return ~df["split_type"].astype(str).str.lower().eq("invariance")
    return pd.Series(True, index=df.index)


def validate_private_labels(labels: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PRIVATE_LABEL_COLUMNS - set(labels.columns))
    if missing:
        raise ValueError(f"private labels missing required columns: {missing}")
    if labels["sample_id"].isna().any() or labels["sample_id"].duplicated().any():
        raise ValueError("private label sample_id values must be non-null and unique")
    splits = set(labels["split_type"].astype(str).str.lower())
    invalid_splits = sorted(splits - ALLOWED_PRIVATE_SPLITS)
    if invalid_splits:
        raise ValueError(f"private labels contain unsupported split_type values: {invalid_splits}")
    if not labels["score_regression"].fillna(False).astype(bool).any():
        raise ValueError("private labels contain no base regression rows")
    invariance = labels["split_type"].astype(str).str.lower().eq("invariance")
    if not invariance.any() or labels.loc[invariance, "invariance_group"].isna().any():
        raise ValueError("private invariance rows and group identifiers are required")


def compute_raw_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, float | None]:
    validate_private_labels(labels)
    merged = labels.merge(predictions, on="sample_id", how="left", validate="one_to_one")
    if merged[PREDICTION_COLUMN].isna().any():
        raise ValueError("missing predictions after label merge")

    merged[PREDICTION_COLUMN] = pd.to_numeric(merged[PREDICTION_COLUMN], errors="raise")
    merged[TARGET_COLUMN] = pd.to_numeric(merged[TARGET_COLUMN], errors="raise")
    if not np.isfinite(merged[PREDICTION_COLUMN].to_numpy(float)).all():
        raise ValueError("predictions contain non-finite values")

    base = merged[_base_mask(merged)].copy()
    if base.empty:
        raise ValueError("no base regression rows in private labels")
    y = base[TARGET_COLUMN].to_numpy(float)
    p = base[PREDICTION_COLUMN].to_numpy(float)

    split = base.get("split_type", pd.Series("iid", index=base.index)).astype(str).str.lower()
    ood = base[split.isin({"ood", "chemical_system_ood"})]

    q10 = float(np.quantile(y, 0.1))
    q90 = float(np.quantile(y, 0.9))
    low = base[base[TARGET_COLUMN] <= q10]
    high = base[base[TARGET_COLUMN] >= q90]

    inv = None
    if "invariance_group" in merged.columns:
        joint_errors = []
        for _, group in merged.dropna(subset=["invariance_group"]).groupby("invariance_group"):
            predictions = group[PREDICTION_COLUMN].to_numpy(float)
            truths = group[TARGET_COLUMN].to_numpy(float)
            spread = _mean_pairwise_abs(predictions)
            if spread is not None:
                center_error = abs(float(np.mean(predictions)) - float(np.mean(truths)))
                joint_errors.append(spread + center_error)
        if joint_errors:
            inv = float(np.mean(joint_errors))

    return {
        "overall_mae": float(mean_absolute_error(y, p)),
        "overall_rmse": float(mean_squared_error(y, p) ** 0.5),
        "overall_spearman": _safe_spearman(y, p),
        "ood_mae": float(mean_absolute_error(ood[TARGET_COLUMN], ood[PREDICTION_COLUMN])) if len(ood) else None,
        "ood_spearman": _safe_spearman(ood[TARGET_COLUMN].to_numpy(float), ood[PREDICTION_COLUMN].to_numpy(float)) if len(ood) else None,
        "hit_rate_10pct": hit_rate_fraction(y, p, 0.1),
        "ndcg_10pct": low_energy_ndcg(y, p, 0.1),
        "low_tail_mae": float(mean_absolute_error(low[TARGET_COLUMN], low[PREDICTION_COLUMN])) if len(low) else None,
        "high_tail_mae": float(mean_absolute_error(high[TARGET_COLUMN], high[PREDICTION_COLUMN])) if len(high) else None,
        "invariance_joint_mae": inv,
    }


def piecewise_anchor_fraction(value: float, anchor: Dict[str, Any], strong_fraction: float = 0.65) -> float:
    direction = str(anchor["direction"]).lower()
    floor = float(anchor["floor"])
    strong = float(anchor["strong"])
    ceiling = float(anchor["ceiling"])
    value = float(value)

    if direction == "lower":
        if not (floor > strong > ceiling):
            raise ValueError(f"lower-is-better anchors must satisfy floor > strong > ceiling: {anchor}")
        if value >= floor:
            return 0.0
        if value <= ceiling:
            return 1.0
        if value >= strong:
            return strong_fraction * (floor - value) / (floor - strong)
        return strong_fraction + (1.0 - strong_fraction) * (strong - value) / (strong - ceiling)

    if direction == "higher":
        if not (floor < strong < ceiling):
            raise ValueError(f"higher-is-better anchors must satisfy floor < strong < ceiling: {anchor}")
        if value <= floor:
            return 0.0
        if value >= ceiling:
            return 1.0
        if value <= strong:
            return strong_fraction * (value - floor) / (strong - floor)
        return strong_fraction + (1.0 - strong_fraction) * (value - strong) / (ceiling - strong)

    raise ValueError(f"unknown metric direction: {direction}")


def score_private(predictions: pd.DataFrame, labels: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw = compute_raw_metrics(predictions, labels)
    scoring_reference = metadata.get("scoring_reference") or {}
    if scoring_reference.get("status") != "frozen_measured":
        raise ValueError("scoring_reference.status must be 'frozen_measured'; provisional or guessed anchors are not scoreable")
    anchors = scoring_reference.get("metrics") or {}
    strong_fraction = float(scoring_reference.get("strong_anchor_fraction", 0.65))
    if not 0.0 < strong_fraction < 1.0:
        raise ValueError("strong_anchor_fraction must be in (0, 1)")

    missing_anchors = sorted(set(METRIC_WEIGHTS) - set(anchors))
    if missing_anchors:
        raise ValueError(f"missing scoring anchors: {missing_anchors}")

    breakdown: Dict[str, float] = {}
    for metric, weight in METRIC_WEIGHTS.items():
        value = raw.get(metric)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"required private metric unavailable: {metric}")
        fraction = piecewise_anchor_fraction(float(value), anchors[metric], strong_fraction)
        breakdown[metric] = float(weight * np.clip(fraction, 0.0, 1.0))

    numeric_score = float(sum(breakdown.values()))
    return {
        "numeric_score": round(numeric_score, 6),
        "numeric_max_score": float(sum(METRIC_WEIGHTS.values())),
        "raw_metrics": raw,
        "breakdown": breakdown,
        "scoring_reference_version": scoring_reference.get("version"),
        "strong_anchor_fraction": strong_fraction,
    }

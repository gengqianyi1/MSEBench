"""Public-data-only lifecycle Oracle for the Matbench pipeline task."""
import json
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(os.environ.get("CHEMBENCH_WORKSPACE", "/workspace"))
DATA = ROOT / "input/data"
TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def counts(formula: str, elements: list[str]) -> list[float]:
    values = {element: 0.0 for element in elements}
    for element, amount in TOKEN.findall(formula):
        values[element] = values.get(element, 0.0) + (float(amount) if amount else 1.0)
    total = sum(values.values()) or 1.0
    raw = [values.get(element, 0.0) for element in elements]
    return raw + [value / total for value in raw] + [total, len([value for value in raw if value])]


train = pd.read_csv(DATA / "train.csv")
targets = [pd.read_csv(DATA / name) for name in ("validation_features.csv", "test_features.csv")]
all_formulas = list(train["composition"].astype(str))
for frame in targets:
    all_formulas.extend(frame["composition"].astype(str))
elements = sorted({element for formula in all_formulas for element, _ in TOKEN.findall(formula)})
x_train = [counts(value, elements) for value in train["composition"].astype(str)]
model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=1729, n_jobs=-1)
model.fit(x_train, train["gap_expt_eV"].astype(float))

rows = []
for frame in targets:
    prediction = model.predict([counts(value, elements) for value in frame["composition"].astype(str)])
    rows.extend(zip(frame["id"].astype(str), prediction))
pd.DataFrame(rows, columns=["id", "prediction_eV"]).to_csv(ROOT / "predictions.csv", index=False)
(ROOT / "pipeline.json").write_text(json.dumps({
    "feature_representation": "element counts and normalized element fractions",
    "estimator": "ExtraTreesRegressor",
    "hyperparameters": {"n_estimators": 500, "min_samples_leaf": 2, "max_features": 0.8},
    "random_seed": 1729,
    "training_data": ["input/data/train.csv"],
}, indent=2), encoding="utf-8")
(ROOT / "final_answer.json").write_text(json.dumps({
    "summary": "Public-data-only deterministic element-composition ensemble.",
    "predictions": "predictions.csv",
    "pipeline": "pipeline.json",
}, indent=2), encoding="utf-8")

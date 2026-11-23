from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from private_scoring import METRIC_WEIGHTS, compute_raw_metrics

LOWER_IS_BETTER = {"overall_mae", "overall_rmse", "ood_mae", "low_tail_mae", "high_tail_mae", "invariance_joint_mae"}


def load_metrics(path: str, labels: pd.DataFrame) -> dict:
    pred = pd.read_csv(path)
    return compute_raw_metrics(pred, labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build measured private scoring anchors from three frozen reference submissions.")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--floor-predictions", required=True, help="Weak/constant baseline predictions on the final private test.")
    ap.add_argument("--strong-predictions", required=True, help="Official reproducible non-pretrained baseline predictions.")
    ap.add_argument("--ceiling-predictions", required=True, help="Pre-registered stronger reference predictions.")
    ap.add_argument("--version", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    labels = pd.read_json(args.labels, orient="records", lines=True, compression="infer")
    floor = load_metrics(args.floor_predictions, labels)
    strong = load_metrics(args.strong_predictions, labels)
    ceiling = load_metrics(args.ceiling_predictions, labels)

    anchors = {}
    for metric in METRIC_WEIGHTS:
        values = (floor.get(metric), strong.get(metric), ceiling.get(metric))
        if any(v is None for v in values):
            raise SystemExit(f"metric {metric} is unavailable in one of the reference submissions")
        direction = "lower" if metric in LOWER_IS_BETTER else "higher"
        f, s, c = map(float, values)
        valid = f > s > c if direction == "lower" else f < s < c
        if not valid:
            raise SystemExit(f"reference ordering invalid for {metric}: floor={f}, strong={s}, ceiling={c}, direction={direction}")
        anchors[metric] = {"direction": direction, "floor": f, "strong": s, "ceiling": c}

    output = {
        "status": "frozen_measured",
        "version": args.version,
        "strong_anchor_fraction": 0.65,
        "metrics": anchors,
        "reference_files": {
            "floor": str(Path(args.floor_predictions).name),
            "strong": str(Path(args.strong_predictions).name),
            "ceiling": str(Path(args.ceiling_predictions).name),
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

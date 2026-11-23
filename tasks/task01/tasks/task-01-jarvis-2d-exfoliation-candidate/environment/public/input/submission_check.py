"""Pre-submit artifact check for Task 01 (run from /workspace)."""
from pathlib import Path
import csv
import json
import sys

REQUIRED = [
    "final_answer.json", "result.json", "native_final_report.json", "data",
    "predictions.csv", "candidate_library.json.gz", "model.joblib",
    "feature_names.json", "training_log.json", "generator",
]

def main() -> int:
    root = Path.cwd()
    missing = [p for p in REQUIRED if not (root / p).exists()]
    errors = []
    for name in ("final_answer.json", "result.json", "native_final_report.json", "feature_names.json", "training_log.json"):
        path = root / name
        if path.is_file():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"{name}: invalid JSON ({exc})")
    predictions = root / "predictions.csv"
    if predictions.is_file():
        try:
            with predictions.open(newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
            if header != ["sample_id", "predicted_exfoliation_energy_mev_atom"]:
                errors.append("predictions.csv: incorrect header")
        except Exception as exc:
            errors.append(f"predictions.csv: unreadable ({exc})")
    print(json.dumps({"ok": not missing and not errors, "missing": missing, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not missing and not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())

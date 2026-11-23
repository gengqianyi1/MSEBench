"""Minimal public pre-submit check for the task snapshot."""
from pathlib import Path
REQUIRED = ["final_answer.json", "result.json", "native_final_report.json", "data", "predictions.csv", "candidate_library.json.gz", "model.joblib", "feature_names.json", "training_log.json", "generator"]
missing = [p for p in REQUIRED if not (Path.cwd()/p).exists()]
print({"ok": not missing, "missing": missing})


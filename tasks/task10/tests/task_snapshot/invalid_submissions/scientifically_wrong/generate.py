from __future__ import annotations
import csv, json, sys
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: generate.py labels.csv output_dir")
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    with Path(sys.argv[1]).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction_eV"]); writer.writeheader()
        for row in rows: writer.writerow({"id": row["id"], "prediction_eV": "0.0"})
    (out / "pipeline.json").write_text(json.dumps({"dataset_name":"matbench_expt_gap","model_type":"constant_baseline"}, indent=2)+"\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

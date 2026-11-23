from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

TARGET = "exfoliation_energy_mev_atom"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    root = Path(args.data_dir)

    frames = {}
    invalid_total = 0
    for name in ("train", "validation", "test_public"):
        path = root / f"{name}.json.gz"
        df = pd.read_json(path, orient="records", lines=True, compression="infer")
        invalid = 0
        for value in df["structure"]:
            try:
                Structure.from_dict(value)
            except Exception:
                invalid += 1
        invalid_total += invalid
        frames[name] = df

    all_ids = pd.concat([
        frames["train"][["sample_id"]],
        frames["validation"][["sample_id"]],
        frames["test_public"][["sample_id"]],
    ], ignore_index=True).sample_id.astype(str)
    labeled = pd.concat([
        frames["train"][TARGET],
        frames["validation"][TARGET],
    ], ignore_index=True)
    numeric = pd.to_numeric(labeled, errors="coerce")

    systems = set()
    elements = set()
    for df in frames.values():
        for raw in df.get("chemical_system", pd.Series(dtype=str)).dropna().astype(str):
            parts = [x for x in raw.split("-") if x]
            systems.add("-".join(sorted(parts)))
            elements.update(parts)

    metadata = json.loads((root / "dataset_metadata.json").read_text(encoding="utf-8"))
    expected_hashes = metadata.get("public_file_sha256") or {}
    hash_report = {
        name: {
            "expected": expected,
            "actual": sha256(root / name),
            "matches": sha256(root / name) == expected,
        }
        for name, expected in expected_hashes.items()
    }

    report = {
        "snapshot_id": metadata.get("snapshot_id"),
        "splits": {
            name: {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "duplicate_ids_within_split": int(df.sample_id.astype(str).duplicated(keep=False).sum()),
                "invalid_structures": int(sum(
                    1 for value in df.structure
                    if _invalid(value)
                )),
                "missing_target": int(pd.to_numeric(df[TARGET], errors="coerce").isna().sum()) if TARGET in df else None,
            }
            for name, df in frames.items()
        },
        "duplicate_id_rows_across_all_splits": int(all_ids.duplicated(keep=False).sum()),
        "invalid_structure_count": int(invalid_total),
        "target_statistics_public_labeled": {
            "count": int(numeric.notna().sum()),
            "missing": int(numeric.isna().sum()),
            "minimum": float(numeric.min()),
            "q10": float(numeric.quantile(0.1)),
            "median": float(numeric.median()),
            "mean": float(numeric.mean()),
            "q90": float(numeric.quantile(0.9)),
            "maximum": float(numeric.max()),
            "std_ddof1": float(numeric.std(ddof=1)),
        },
        "elements": sorted(elements),
        "element_count": len(elements),
        "chemical_system_count": len(systems),
        "public_hashes": hash_report,
        "all_public_hashes_match": bool(hash_report) and all(x["matches"] for x in hash_report.values()),
    }
    print(json.dumps(report, indent=2))


def _invalid(value: dict) -> bool:
    try:
        Structure.from_dict(value)
        return False
    except Exception:
        return True


if __name__ == "__main__":
    main()

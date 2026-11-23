"""Run the four fresh v1.1 Gate-B native smoke angles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy
import rdkit

from tools.science import (
    MAPPED_SMILES,
    TARGET_DIHEDRAL_MAPS,
    run_native_scan,
    sha256_file,
    write_json,
    write_reference,
    xtb_version_record,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_or_build_reference(case_root: Path) -> dict:
    path = case_root / "reference_structure.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return write_reference(case_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    case_root = ROOT / "benchmark_case"
    smoke_root = case_root / "smoke_attempts_v1_1"
    smoke_root.mkdir(parents=True, exist_ok=True)
    reference = _load_or_build_reference(case_root)
    records = []
    for angle in (0, 90, 180, 270):
        attempt = smoke_root / f"angle_{angle:03d}" / "attempt_001"
        if attempt.is_dir() and args.resume:
            record = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
        else:
            attempt.parent.mkdir(parents=True, exist_ok=True)
            record = run_native_scan(
                reference_path=case_root / "reference.xyz",
                angle_deg=angle,
                attempt_root=attempt,
                timeout_seconds=180,
            )
        records.append(record)
    evidence = {
        "schema_version": "1.0",
        "gate": "Gate B",
        "task_id": "adaptive_xtb_torsion_profile_agent_v1_1",
        "benchmark_version": "1.1.0",
        "protocol": "adaptive_torsion_sampling_tool_loop/1.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mapped_smiles": MAPPED_SMILES,
        "target_dihedral_maps": list(TARGET_DIHEDRAL_MAPS),
        "reference": reference,
        "reference_xyz_sha256": sha256_file(case_root / "reference.xyz"),
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "numpy": numpy.__version__,
            "scipy": "not installed in isolated xtb-benchmark runtime; not required",
            "xtb": xtb_version_record(),
        },
        "smoke_angles": records,
        "passed": all(record.get("profile_eligible") is True for record in records),
    }
    output = ROOT / "reports" / "gate_b_v1_1_native_smoke.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("V1_1_SMOKE_REPORT_ALREADY_EXISTS")
    write_json(output, evidence, exclusive=True)
    for record in records:
        print(
            f"angle={record['angle_deg']:3d} return={record['return_code']} "
            f"converged={record['converged']} energy={record['total_energy_hartree']} "
            f"final={record['final_angle_deg']} eligible={record['profile_eligible']}"
        )
    print(f"reference_sha256={evidence['reference_xyz_sha256']}")
    print(f"passed={evidence['passed']}")
    return 0 if evidence["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

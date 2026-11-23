#!/usr/bin/env python3
"""Run four fresh package-safe xTB 6.7.1 smoke points for v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_multicase_suite_v3 import (  # noqa: E402
    base_molecule,
    molecule_from_xyz,
    run_angle,
    xtb_identity,
)
from scripts.run_suite_models_v3 import _definition, _locate_frozen_xtb  # noqa: E402


ANGLES = (15, 105, 195, 285)
CASE_ID = "2_fluorobiphenyl_central_torsion_v1"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError("POST_RECHARGE_SMOKE_OUTPUT_EXISTS")
    output.mkdir(parents=True)

    suite = json.loads((ROOT / "organizer_private" / "suite.json").read_text(encoding="utf-8"))
    case = next(item for item in suite["cases"] if item["case_id"] == CASE_ID)
    definition, reference_path, _ = _definition(case)
    reference = molecule_from_xyz(reference_path, base_molecule(definition))
    executable = _locate_frozen_xtb()
    identity = xtb_identity(executable)
    case_root = output / CASE_ID
    rows = [
        run_angle(
            case_root=case_root,
            definition=definition,
            reference_molecule=reference,
            reference_path=reference_path,
            angle=angle,
            executable=executable,
            identity=identity,
            timeout=180,
        )
        for angle in ANGLES
    ]
    report = {
        "schema_version": "3.0",
        "benchmark_version": "3.0.0",
        "case_id": CASE_ID,
        "angles_deg": list(ANGLES),
        "xtb_identity": identity,
        "points": [
            {
                "angle_deg": row["angle_deg"],
                "return_code": row["return_code"],
                "timed_out": row["timed_out"],
                "converged": row["converged"],
                "profile_eligible": row["profile_eligible"],
                "total_energy_hartree": row["total_energy_hartree"],
                "final_angle_deg": row["final_angle_deg"],
                "angle_error_deg": row["angle_error_deg"],
                "result_sha256": hashlib.sha256(
                    canonical_bytes(row)
                ).hexdigest().upper(),
            }
            for row in rows
        ],
        "passed": bool(
            identity.get("version") == "6.7.1"
            and all(
                row.get("return_code") == 0
                and row.get("timed_out") is False
                and row.get("converged") is True
                and row.get("profile_eligible") is True
                for row in rows
            )
        ),
    }
    report_path = output / "smoke_report.json"
    with report_path.open("xb") as handle:
        handle.write(canonical_bytes(report))
    print(json.dumps({"passed": report["passed"], "output": str(report_path)}, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

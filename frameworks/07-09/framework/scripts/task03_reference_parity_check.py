#!/usr/bin/env python3
"""Cheap Task 03 parity check on frozen private runtime reference values.

This does not replace the locked xTB/DIPRO runtime Oracle. It verifies that the
Harbor scientific formula and the stored successful reference examples agree
exactly for every reference molecule.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tasks" / "task-03-five-ring-xtb-dipro-marcus" / "tests" / "task_snapshot" / "input" / "original" / "src"
REFERENCE = (
    ROOT
    / "tasks"
    / "task-03-five-ring-xtb-dipro-marcus"
    / "tests"
    / "task_snapshot"
    / "private_data"
    / "reference_examples"
    / "satisfied_molecules.json"
)


def main() -> int:
    sys.path.insert(0, str(SRC))
    from xtb_runner import marcus_hole_rate
    from scoring import rate_performance_score

    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    molecules = payload["molecules"]
    cfg = {
        "rate_score_floor_s_inv": 1.0e14,
        "rate_score_ceiling_s_inv": 1.0e15,
        "rate_threshold_s_inv": 1.0e14,
    }
    mismatches = []
    for molecule in molecules:
        rate = marcus_hole_rate(float(molecule["lambda_h_eV"]), float(molecule["Vh_DIPRO_eV"]), cfg)
        score = rate_performance_score(rate, cfg)
        target_hit = rate > cfg["rate_threshold_s_inv"]
        expected_score = min(100.0, 100.0 * rate / cfg["rate_score_ceiling_s_inv"])
        if (
            not math.isclose(rate, float(molecule["marcus_hole_rate_s_inv"]), rel_tol=1.0e-12)
            or bool(target_hit) != bool(molecule["target_hit"])
            or not math.isclose(score, expected_score, rel_tol=1.0e-12)
        ):
            mismatches.append(
                {
                    "index": molecule["index"],
                    "computed_rate_s_inv": rate,
                    "stored_rate_s_inv": molecule["marcus_hole_rate_s_inv"],
                    "computed_target_hit": target_hit,
                    "stored_target_hit": molecule["target_hit"],
                    "computed_score": score,
                    "expected_score": expected_score,
                }
            )

    result = {
        "reference_molecules": len(molecules),
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    print(json.dumps(result, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

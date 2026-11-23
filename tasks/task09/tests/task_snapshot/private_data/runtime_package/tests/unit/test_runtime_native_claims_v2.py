from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable

import pytest

from tools.tool_registry import ToolSession


def _native_record(
    *,
    reference_path: Path,
    angle_deg: int,
    attempt_root: Path,
    timeout_seconds: int,
) -> dict:
    del timeout_seconds
    attempt_root.mkdir(parents=True, exist_ok=False)
    (attempt_root / "input.xyz").write_bytes(reference_path.read_bytes())
    signed = angle_deg if angle_deg <= 180 else angle_deg - 360
    (attempt_root / "xcontrol.inp").write_text(
        "$constrain\n"
        "  force constant=1.0\n"
        f"  dihedral: 6, 7, 8, 9, {signed:.1f}\n"
        "$end\n",
        encoding="ascii",
        newline="\n",
    )
    (attempt_root / "xtbopt.xyz").write_bytes(reference_path.read_bytes())
    (attempt_root / "stdout.log").write_text(
        "GEOMETRY OPTIMIZATION CONVERGED\n", encoding="ascii"
    )
    (attempt_root / "stderr.log").write_bytes(b"")
    (attempt_root / "attempt_manifest.json").write_text(
        '{"algorithm":"SHA-256","files":[],"schema_version":"1.0"}',
        encoding="ascii",
    )
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest().upper()
    input_sha = hashlib.sha256(
        (attempt_root / "input.xyz").read_bytes()
    ).hexdigest().upper()
    executable = "XTB_FIXTURE"
    return {
        "method": "GFN2-xTB",
        "phase": "gas",
        "charge": 0,
        "uhf": 0,
        "threads": 1,
        "requested_final_angle_tolerance_deg": 2.0,
        "argv": [
            executable,
            "input.xyz",
            "--gfn",
            "2",
            "--chrg",
            "0",
            "--uhf",
            "0",
            "--opt",
            "tight",
            "--input",
            "xcontrol.inp",
        ],
        "shell": False,
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "xtb": {
            "executable": executable,
            "version": "6.7.1",
            "binary_sha256": (
                "E11F074B519C3B226F3FAB4902F36555C4BFB204CC0365BDB196313124E3DC1B"
            ),
            "version_exit_code": 0,
        },
        "return_code": 0,
        "timed_out": False,
        "converged": True,
        "total_energy_hartree": -100.0
        + 0.001 * (1.0 - math.cos(math.radians(4.0 * angle_deg))),
        "final_angle_deg": float(angle_deg),
        "angle_error_deg": 0.0,
        "structure_validation": {
            "atom_count_ok": True,
            "atom_order_ok": True,
            "composition_ok": True,
            "formal_charge_ok": True,
            "connectivity_ok": True,
        },
        "profile_eligible": True,
        "reference_sha256": reference_sha,
        "prepared_input": {
            "reference_sha256": reference_sha,
            "input_sha256": input_sha,
        },
    }


def _claiming_native(
    mutate: Callable[[dict], None],
) -> Callable[..., dict]:
    def runner(**kwargs) -> dict:
        record = _native_record(**kwargs)
        if kwargs["angle_deg"] == 0:
            mutate(record)
            # The defect under test is a native adapter that leaves its
            # aggregate claim true despite contradictory primitive facts.
            record["profile_eligible"] = True
        return record

    return runner


def _phase0_action() -> dict:
    return {
        "action": "run_scan_batch",
        "angles_deg": [0, 60, 120, 180, 240, 300],
        "finalize": False,
        "confidence": None,
        "rationale": "Exercise runtime-native eligibility recomputation.",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update({"structure_validation": {}}),
        lambda record: record.update(
            {"final_angle_deg": 45.0, "angle_error_deg": 0.0}
        ),
        lambda record: record.update({"return_code": True}),
        lambda record: record.update({"total_energy_hartree": float("nan")}),
    ],
    ids=[
        "empty-structure-validation",
        "false-angle-error",
        "boolean-return-code",
        "nonfinite-energy",
    ],
)
def test_runtime_recomputes_native_eligibility_instead_of_trusting_claim(
    tmp_path: Path,
    mutate: Callable[[dict], None],
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id=f"native-claim-{tmp_path.name}",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_claiming_native(mutate),
    )

    result = session.dispatch(_phase0_action(), assistant_turn=1)

    point = next(
        item for item in result["point_results"] if item["angle_deg"] == 0
    )
    attempts = session.attempt_summaries[0]
    assert result["status"] == "BATCH_COMPLETE"
    assert point["status"] == "failed"
    assert point["attempt_count"] == 2
    assert [item["profile_eligible"] for item in attempts] == [False, False]
    assert result["state"]["valid_unique_angles"] == 5
    assert result["state"]["failed_angles_deg"] == [0]


def test_runtime_recomputes_circular_angle_error_for_safe_projection(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="native-angle-projection",
        raw_provenance_root=tmp_path / "raw",
        native_runner=_claiming_native(
            lambda record: record.update(
                {"final_angle_deg": 359.0, "angle_error_deg": 359.0}
            )
        ),
    )

    result = session.dispatch(_phase0_action(), assistant_turn=1)

    point = next(
        item for item in result["point_results"] if item["angle_deg"] == 0
    )
    assert point["status"] == "valid"
    assert point["attempt_count"] == 1
    assert point["angle_error_deg"] == pytest.approx(1.0)


def test_runtime_does_not_require_native_aggregate_eligibility_claim(
    tmp_path: Path,
) -> None:
    def native_without_aggregate(**kwargs) -> dict:
        record = _native_record(**kwargs)
        record["profile_eligible"] = False
        return record

    run_root = tmp_path / "run"
    run_root.mkdir()
    session = ToolSession(
        run_root,
        run_id="native-aggregate-is-advisory",
        raw_provenance_root=tmp_path / "raw",
        native_runner=native_without_aggregate,
    )

    result = session.dispatch(_phase0_action(), assistant_turn=1)

    assert result["status"] == "BATCH_COMPLETE"
    assert result["state"]["valid_unique_angles"] == 6
    assert all(
        item["attempt_count"] == 1 for item in result["point_results"]
    )

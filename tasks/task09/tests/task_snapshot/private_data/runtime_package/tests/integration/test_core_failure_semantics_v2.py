from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from framework.agent_runner import FormalAgentRunner
from framework.trajectory import load_jsonl


MODEL = "Pro/moonshotai/Kimi-K2.6"
TEST_EXECUTION_FREEZE_BINDING = {
    "schema_version": "execution-freeze-binding/2.0",
    "content_sha256": "A" * 64,
    "commitment_file_sha256": "B" * 64,
    "file_count": 1,
    "total_bytes": 1,
}


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Transport:
    def complete(self, **_kwargs):
        action = {
            "action": "run_scan_batch",
            "angles_deg": [0, 60, 120, 180, 240, 300],
            "finalize": False,
            "confidence": None,
            "rationale": "Use broad initial coverage.",
        }
        return {
            "model": MODEL,
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            action, separators=(",", ":")
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }


def test_total_native_timeout_is_external_blocked_not_model_fail(
    tmp_path: Path,
) -> None:
    clock = _Clock()

    def slow_native(
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
        clock.advance(901.0)
        executable = "XTB_FIXTURE"
        reference_sha = hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest().upper()
        input_sha = hashlib.sha256(
            (attempt_root / "input.xyz").read_bytes()
        ).hexdigest().upper()
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
            + 0.001 * (1.0 - math.cos(math.radians(4 * angle_deg))),
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

    destination = tmp_path / "run"
    result = FormalAgentRunner(
        label="kimi",
        run_root=destination,
        transport=_Transport(),
        native_runner=slow_native,
        raw_provenance_root=tmp_path / "raw",
        execution_freeze_binding=TEST_EXECUTION_FREEZE_BINDING,
        retry_sleep=lambda _seconds: None,
        monotonic=clock,
    ).run()

    assert result["trusted_runtime"]["status"] == "BLOCKED"
    assert (
        result["trusted_runtime"]["safe_status"]
        == "TOTAL_NATIVE_TIMEOUT_EXCEEDED"
    )
    events = load_jsonl(destination / "trajectory.jsonl")
    assert events[-1]["event"] == "run_failure"
    assert events[-1]["payload"]["status"] == "BLOCKED"
    assert (
        events[-1]["payload"]["safe_status"]
        == "TOTAL_NATIVE_TIMEOUT_EXCEEDED"
    )
    manifest = json.loads(
        (destination / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "BLOCKED"
    assert manifest["blocker"] == "TOTAL_NATIVE_TIMEOUT_EXCEEDED"

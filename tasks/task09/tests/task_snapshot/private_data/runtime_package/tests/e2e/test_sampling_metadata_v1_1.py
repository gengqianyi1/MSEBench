from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import tools.tool_registry as registry
from evaluator.evaluate import _budget_facts, _tool_pairs
from framework.agent_runner import RunContext
from framework.budget import recompute_budget
from framework.schemas import CASE_ID, FORMAL_MODELS
from framework.trajectory import load_jsonl
from tools.science import canonical_json_bytes, sha256_file


CANONICAL_FIELDS = (
    "new_unique_angles_this_round",
    "cumulative_unique_angles",
    "unique_angles_remaining",
    "scan_feedback_rounds",
    "profile_fit_rounds",
    "current_sampling_phase",
    "finalization_only",
    "phase_new_unique_angle_limit",
    "phase_new_unique_angles_remaining",
    "phase_fit_pending_analysis",
    "phase_fit_profile_id",
    "initial_sampling_complete",
    "initial_sampling_new_unique_angles",
    "third_feedback_round_reachable",
    "eligibility_gaps",
    "completed_sampling_rounds",
    "sampling_turns",
    "first_sampling_turn_new_angles",
)


def _fake_native(
    *,
    reference_path: Path,
    angle_deg: int,
    attempt_root: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    attempt_root.mkdir(parents=True, exist_ok=False)
    (attempt_root / "result.json").write_bytes(
        canonical_json_bytes({"return_code": 0, "angle_deg": angle_deg})
    )
    (attempt_root / "attempt_manifest.json").write_bytes(
        canonical_json_bytes({"angle_deg": angle_deg})
    )
    return {
        "return_code": 0,
        "timed_out": False,
        "converged": True,
        "total_energy_hartree": -20.0 + angle_deg * 0.00001,
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
        "reference_sha256": sha256_file(reference_path),
        "prepared_input": {"input_sha256": "A" * 64},
        "attempt_manifest_sha256": sha256_file(
            attempt_root / "attempt_manifest.json"
        ),
        "xtb": {"binary_sha256": "B" * 64},
    }


def test_trajectory_trusted_runtime_and_evaluator_metadata_are_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(registry, "run_native_scan", _fake_native)
    requested = FORMAL_MODELS["kimi"]
    run_root = tmp_path / "run"
    context = RunContext.create(
        run_root,
        run_id="v1-1-metadata-e2e",
        requested_model_id=requested,
        mode="TEST_FAKE_NATIVE",
    )
    call_index = 0

    def assistant(turn: int) -> None:
        context.record_assistant(
            turn=turn,
            actual_model_id=requested,
            content="",
            tool_count=0,
        )

    def call(turn: int, tool: str, arguments: dict[str, object]) -> dict[str, object]:
        nonlocal call_index
        call_index += 1
        return context.call_tool(
            turn=turn,
            call_id=f"c{call_index:02d}",
            name=tool,
            arguments=arguments,
        )

    assistant(1)
    for angle in (0, 5, 10, 15, 20, 25):
        assert call(1, "run_scan_point", {"angle_deg": angle})["ok"]
    first_fit = call(1, "fit_periodic_profile", {"method": "periodic_cubic"})
    first_id = first_fit["profile"]["profile_id"]
    assert call(1, "analyze_profile", {"profile_id": first_id})["ok"]

    final_fit: dict[str, object] | None = None
    final_analysis: dict[str, object] | None = None
    for turn, angles in (
        (2, (30, 35, 40)),
        (3, (45, 50, 55)),
        (4, (60, 65, 70)),
    ):
        assistant(turn)
        for angle in angles:
            assert call(turn, "run_scan_point", {"angle_deg": angle})["ok"]
        final_fit = call(
            turn, "fit_periodic_profile", {"method": "periodic_cubic"}
        )
        final_id = final_fit["profile"]["profile_id"]
        final_analysis = call(
            turn, "analyze_profile", {"profile_id": final_id}
        )
        assert final_analysis["ok"]

    assert final_fit is not None and final_analysis is not None
    assistant(5)
    sixteenth = call(5, "run_scan_point", {"angle_deg": 75})
    assert sixteenth["budget"]["finalization_only"] is True
    scientific = final_analysis["analysis"]
    submission = {
        "case_id": CASE_ID,
        "final_status": "completed",
        "profile_id": final_fit["profile"]["profile_id"],
        "global_minimum_deg": scientific["global_minimum"]["angle_deg"],
        "local_minima_deg": [
            item["angle_deg"] for item in scientific["local_minima"]
        ],
        "local_maxima_deg": [
            item["angle_deg"] for item in scientific["local_maxima"]
        ],
        "forward_barrier_kcal_mol": scientific["forward_barrier"],
        "reverse_barrier_kcal_mol": scientific["reverse_barrier"],
        "confidence": 0.8,
        "rationale": "Guarded v1.1 sampling phases completed.",
    }
    accepted = call(5, "submit_final", submission)
    assert accepted["ok"] is True
    envelope = context.finalize_success(turn=5)

    events = load_jsonl(run_root / "trajectory.jsonl")
    replay = recompute_budget(events).to_dict()
    trusted = envelope["trusted_runtime"]["process_recomputed_by_runner"]
    assert trusted["live_reconstruction_agrees"] is True
    assert trusted["first_sampling_turn_new_angles"] == 6
    assert trusted["initial_sampling_new_unique_angles"] == 6
    assert trusted["cumulative_unique_angles"] == 16
    assert trusted["scan_feedback_rounds"] == 3
    assert trusted["profile_fit_rounds"] == 4
    assert trusted["finalization_only"] is True

    calls, paired = _tool_pairs(events)
    assert paired is True
    evaluator = _budget_facts(
        calls,
        attempts={
            attempt_id: {
                "angle_deg": record["angle_deg"],
                "valid": record["profile_eligible"],
            }
            for attempt_id, record in context.session.attempt_records.items()
        },
        profiles={
            path.stem: {"valid": True}
            for path in context.session.profiles_root.glob("profile_*.json")
        },
    )
    for field in CANONICAL_FIELDS:
        assert trusted[field] == replay[field] == evaluator[field]

    evaluator_module = importlib.import_module("evaluator.evaluate")

    def validated_attempt(
        directory: Path,
        *,
        reference_symbols: list[str],
        expected_bonds: set[tuple[int, int]],
    ) -> dict[str, object]:
        record = context.session.attempt_records[directory.name]
        return {
            "attempt_id": directory.name,
            "angle_deg": record["angle_deg"],
            "valid": True,
            "profile_eligible": True,
            "energy_hartree": record["total_energy_hartree"],
            "manifest_valid": True,
            "manifest_sha256": record["provenance"][
                "attempt_manifest_sha256"
            ],
            "invariants_valid": True,
            "structure_valid": True,
            "native_valid": True,
            "energy_valid": True,
            "control_valid": True,
        }

    monkeypatch.setattr(
        evaluator_module,
        "_validate_attempt",
        validated_attempt,
    )
    evaluation = evaluator_module.evaluate_run(run_root)
    reconstruction = evaluation["independent_reconstruction"]
    assert reconstruction["process_binding_valid"] is True
    assert reconstruction["budget"]["first_sampling_turn_new_angles"] == 6
    assert reconstruction["budget"]["scan_feedback_rounds"] == 3
    assert evaluation["eligible"] is True

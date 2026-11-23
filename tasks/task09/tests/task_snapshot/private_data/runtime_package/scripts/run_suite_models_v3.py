#!/usr/bin/env python3
"""Run the exact formal models over all four parameterized v3 torsion cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import rdkit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.evaluate_multicase_run_v3 import (  # noqa: E402
    canonical_bytes,
    evaluate_case,
)
from evaluator.evaluate_suite import aggregate_suite  # noqa: E402
from evaluator.profile_recompute_v2 import recompute_automatic_profile  # noqa: E402
from framework.agent_runner import _response_parts, parse_single_json_action  # noqa: E402
from framework.provider_retry import DEFAULT_PROVIDER_RETRY_POLICY  # noqa: E402
from framework.schemas import FORMAL_MODELS, ModelConfig  # noqa: E402
from framework.siliconflow_adapter import (  # noqa: E402
    ProviderCallError,
    SiliconFlowChatTransport,
    load_runtime_credential,
)
from framework.trajectory import TrajectoryRecorder  # noqa: E402
from scripts.build_multicase_suite_v3 import (  # noqa: E402
    HARTREE_TO_KCAL_MOL,
    base_molecule,
    locate_xtb,
    molecule_from_xyz,
    run_angle,
    xtb_identity,
)


PROTOCOL = "adaptive_torsion_sampling_tool_loop/3.0"
TASK_ID = "adaptive_xtb_torsion_profile_agent_v3"
MODEL_ORDER = ("deepseek", "kimi", "glm")
SYSTEM_PROMPT = """You are selecting adaptive sampling angles for one frozen gas-phase GFN2-xTB torsion case. Return exactly one JSON object with keys action, angles_deg, finalize, confidence, and rationale. action must be run_scan_batch. Choose only new integer multiples of 5 in [0,360). Follow the phase-specific batch size and finalize rules in the user state. Use only the case specification and native feedback supplied there. Do not request or mention an oracle, hidden data, paths, files, source code, evaluator, credentials, shell, Python, network, scores, or hashes."""
BENCHMARK_NAME = "Adaptive xTB Torsion Profile Agent Benchmark"
OPTIONAL_USAGE_FIELDS = (
    "reasoning_tokens",
    "cached_input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _locate_frozen_xtb() -> Path:
    try:
        return locate_xtb()
    except RuntimeError:
        completed = subprocess.run(
            ["conda", "info", "--envs", "--json"],
            capture_output=True,
            timeout=30,
            check=True,
            shell=False,
        )
        payload = json.loads(completed.stdout.decode("utf-8"))
        matches = [
            Path(value)
            for value in payload.get("envs", [])
            if isinstance(value, str) and Path(value).name == "xtb-benchmark"
        ]
        for prefix in matches:
            for candidate in (
                prefix / "Library" / "bin" / "xtb.exe",
                prefix / "Scripts" / "xtb.exe",
                prefix / "bin" / "xtb",
            ):
                if candidate.is_file():
                    return candidate.resolve(strict=True)
        raise RuntimeError("xTB executable not found in frozen conda environment")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_NOT_OBJECT")
    return value


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _response_parts_v3(response: Any) -> tuple[str, str, dict[str, int] | None]:
    """Project standard and optional provider usage without retaining raw data."""

    actual, content, usage = _response_parts(response)
    if usage is None:
        return actual, content, None
    projected = dict(usage)
    raw_usage = _field(response, "usage")
    prompt_details = _field(raw_usage, "prompt_tokens_details")
    completion_details = _field(raw_usage, "completion_tokens_details")
    optional = {
        "reasoning_tokens": _nonnegative_int(_field(completion_details, "reasoning_tokens")),
        "cached_input_tokens": _nonnegative_int(_field(prompt_details, "cached_tokens")),
        "cache_read_input_tokens": _nonnegative_int(_field(raw_usage, "cache_read_input_tokens")),
        "cache_creation_input_tokens": _nonnegative_int(
            _field(raw_usage, "cache_creation_input_tokens")
        ),
    }
    projected.update({key: value for key, value in optional.items() if value is not None})
    return actual, content, projected


def _suite() -> tuple[list[dict[str, Any]], Path]:
    path = ROOT / "organizer_private" / "suite.json"
    suite = _json(path)
    cases = suite.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("V3_SUITE_MUST_HAVE_FOUR_CASES")
    records = [dict(case) for case in cases if isinstance(case, Mapping)]
    if len(records) != 4 or len({case.get("case_id") for case in records}) != 4:
        raise ValueError("V3_SUITE_CASES_INVALID")
    for case in records:
        oracle = ROOT / str(case["oracle_path"])
        if not oracle.is_file() or hashlib.sha256(oracle.read_bytes()).hexdigest().upper() != str(
            case["oracle_sha256"]
        ).upper():
            raise ValueError("V3_ORACLE_COMMITMENT_MISMATCH")
    return records, path


def _definition(case: Mapping[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    source = ROOT / str(case["case_root"])
    structure = _json(source / "reference_structure.json")
    spec = _json(source / "case_spec.json")
    definition = {
        "molecule": case["molecule"],
        "mapped_smiles": structure["mapped_smiles"],
        "target_maps": structure.get(
            "target_dihedral_atom_maps", structure.get("target_dihedral_maps")
        ),
        "reference_angle_deg": structure.get(
            "reference_angle_deg", structure.get("reference_torsion_deg")
        ),
        "seed": structure["reference_seed"],
    }
    if not isinstance(definition["target_maps"], list) or len(definition["target_maps"]) != 4:
        raise ValueError("CASE_DIHEDRAL_DEFINITION_INVALID")
    return definition, source / "reference.xyz", source


def _copy_public_case_materials(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("case_spec.json", "reference.xyz", "reference_structure.json"):
        origin = source / name
        if not origin.is_file():
            raise FileNotFoundError(name)
        shutil.copy2(origin, destination / name)


def _analysis(angles: list[int], energies_hartree: list[float]) -> tuple[dict[str, Any], list[float]]:
    minimum = min(energies_hartree)
    relative = [(value - minimum) * HARTREE_TO_KCAL_MOL for value in energies_hartree]
    profile = recompute_automatic_profile(angles, relative)
    curve = np.asarray(profile["predicted_energies"], dtype=float)
    curvature = np.abs(np.roll(curve, -1) - 2 * curve + np.roll(curve, 1))
    unsampled = [angle for angle in range(0, 360, 5) if angle not in set(angles)]
    recommended = sorted(unsampled, key=lambda angle: (-curvature[angle // 5], angle))[:8]
    compact = {
        "selected_method": profile["selected_method"],
        "selected_order": profile["selected_order"],
        "sampled_relative_energies_kcal_mol": [round(value, 8) for value in relative],
        "predicted_global_minimum_angle_deg": int(np.argmin(curve)) * 5,
        "predicted_global_maximum_angle_deg": int(np.argmax(curve)) * 5,
        "predicted_energy_range_kcal_mol": round(float(np.max(curve) - np.min(curve)), 8),
        "high_curvature_unsampled_angles_deg": recommended,
        "best_loo_rmse_kcal_mol": round(
            min(float(item["loo_rmse_kcal_mol"]) for item in profile["candidates"]), 8
        ),
    }
    return compact, [float(value) for value in curve]


def _state(
    *, case_spec: Mapping[str, Any], phase: int, selected: list[int], feedback: Mapping[str, Any] | None,
    validation_error: str | None,
) -> str:
    counts = (6, (2, 3), (2, 3))
    payload = {
        "task_id": TASK_ID,
        "protocol": PROTOCOL,
        "case_spec": dict(case_spec),
        "phase": phase,
        "phase_rule": {
            "required_new_angle_count": counts[phase],
            "finalize": phase == 2,
            "confidence": "low|medium|high" if phase == 2 else None,
        },
        "selected_angles_deg": selected,
        "latest_native_feedback": dict(feedback) if isinstance(feedback, Mapping) else None,
        "previous_action_error": validation_error,
        "response_schema": {
            "action": "run_scan_batch",
            "angles_deg": "array of new grid angles",
            "finalize": phase == 2,
            "confidence": "low|medium|high when finalizing, otherwise null",
            "rationale": "1-400 character sampling rationale",
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _valid_action(action: Any, *, phase: int, selected: list[int]) -> tuple[bool, str]:
    required = {"action", "angles_deg", "finalize", "confidence", "rationale"}
    if not isinstance(action, Mapping) or set(action) != required:
        return False, "ACTION_SCHEMA_INVALID"
    angles = action.get("angles_deg")
    if not isinstance(angles, list) or any(
        isinstance(angle, bool) or not isinstance(angle, int) for angle in angles
    ):
        return False, "ANGLE_LIST_INVALID"
    allowed_count = len(angles) == 6 if phase == 0 else len(angles) in {2, 3}
    if not allowed_count:
        return False, "BATCH_SIZE_INVALID"
    if len(angles) != len(set(angles)) or set(angles) & set(selected):
        return False, "ANGLES_NOT_NEW_AND_UNIQUE"
    if any(angle < 0 or angle >= 360 or angle % 5 for angle in angles):
        return False, "ANGLE_DOMAIN_INVALID"
    if action.get("finalize") is not (phase == 2):
        return False, "FINALIZE_PHASE_INVALID"
    if phase < 2 and action.get("confidence") is not None:
        return False, "CONFIDENCE_MUST_BE_NULL"
    if phase == 2 and action.get("confidence") not in {"low", "medium", "high"}:
        return False, "CONFIDENCE_INVALID"
    rationale = action.get("rationale")
    if not isinstance(rationale, str) or not 1 <= len(rationale) <= 400:
        return False, "RATIONALE_INVALID"
    return True, "PASS"


def _provider_turn(
    config: ModelConfig,
    messages: list[dict[str, str]],
    *, api_key: str, recorder: TrajectoryRecorder, turn: int,
) -> tuple[str, str, dict[str, int] | None, int]:
    policy = DEFAULT_PROVIDER_RETRY_POLICY
    cumulative = 0.0
    for attempt in range(1, policy.maximum_total_attempts + 1):
        try:
            # Native xTB work can leave a pooled HTTP connection idle for
            # several minutes.  Bind every application attempt to a fresh,
            # fixed-model client so stale sockets cannot poison later turns or
            # all retries.  Retry count, delays, model identity and request
            # configuration remain frozen.
            transport = SiliconFlowChatTransport(config, api_key)
            response = transport.complete(
                messages=messages,
                tools=[],
                timeout_seconds=config.timeout_seconds,
                response_format={"type": "json_object"},
            )
            actual, content, usage = _response_parts_v3(response)
            return actual, content, usage, attempt - 1
        except ProviderCallError as exc:
            category = str(exc.safe.get("status", "UNKNOWN_PROVIDER_ERROR"))
            retry = policy.should_retry(category=category, attempt_number=attempt)
            delay = 0.0
            if retry:
                delay = policy.plan_delay(
                    attempt_number=attempt,
                    cumulative_delay_seconds=cumulative,
                    jitter_unit=0.5,
                )
            recorder.append(
                "provider_attempt_failed",
                {
                    "turn": turn,
                    "attempt": attempt,
                    "category": category,
                    "retryable": retry,
                    "planned_delay_seconds": delay,
                    "raw_response_persisted": False,
                },
            )
            if not retry:
                raise
            cumulative += delay
            time.sleep(delay)
    raise RuntimeError("PROVIDER_RETRIES_EXHAUSTED")


def _render_dialogue(dialogue: list[dict[str, Any]]) -> str:
    lines = ["# Formal v3 dialogue", ""]
    for item in dialogue:
        lines.append(f"## Turn {item['turn']}")
        lines.append("")
        lines.append("### Canonical state")
        lines.append("")
        lines.append("```json")
        lines.append(item["state"])
        lines.append("```")
        lines.append("")
        lines.append("### Model action")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(item.get("action"), ensure_ascii=False, sort_keys=True, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_case(
    *, label: str, case: Mapping[str, Any], case_root: Path, executable: Path,
    identity: Mapping[str, Any], api_key: str, campaign_id: str,
) -> dict[str, Any]:
    if case_root.exists() or case_root.is_symlink():
        raise FileExistsError("FORMAL_CASE_OUTPUT_EXISTS")
    case_root.mkdir(parents=True)
    case_id = str(case["case_id"])
    run_id = f"{campaign_id}-{label}-{case_id}"
    recorder = TrajectoryRecorder(run_id=run_id, protocol=PROTOCOL)
    config = ModelConfig.for_label(label)
    definition, reference_path, source_root = _definition(case)
    artifact_case = case_root / "artifacts" / case_id
    _copy_public_case_materials(source_root, artifact_case)
    reference_molecule = molecule_from_xyz(reference_path, base_molecule(definition))
    case_spec = _json(source_root / "case_spec.json")
    started_at = _utc_now()
    started = time.monotonic()
    recorder.append(
        "run_start",
        {
            "task_id": TASK_ID,
            "case_id": case_id,
            "protocol": PROTOCOL,
            "logical_model": label,
            "requested_model_id": config.model,
            "execution_mode": "REAL_API_LIVE_XTB",
        },
    )
    selected: list[int] = []
    energies: list[float] = []
    latest_feedback: dict[str, Any] | None = None
    final_curve: list[float] | None = None
    final_action: dict[str, Any] | None = None
    actual_model: str | None = None
    dialogue: list[dict[str, Any]] = []
    provider_turns = provider_retries = input_tokens = output_tokens = total_tokens = 0
    optional_usage_totals: dict[str, int | None] = {
        key: None for key in OPTIONAL_USAGE_FIELDS
    }
    validation_error: str | None = None
    correction_used = False
    runner_status = "FAILED"
    provider_status = "RESPONDED"
    failure: dict[str, Any] | None = None
    phase = 0
    try:
        while phase < 3 and provider_turns < 5:
            state = _state(
                case_spec=case_spec,
                phase=phase,
                selected=selected,
                feedback=latest_feedback,
                validation_error=validation_error,
            )
            provider_turns += 1
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state},
            ]
            actual, content, usage, retries = _provider_turn(
                config,
                messages,
                api_key=api_key,
                recorder=recorder,
                turn=provider_turns,
            )
            provider_retries += retries
            actual_model = actual
            if usage:
                input_tokens += usage["input_tokens"]
                output_tokens += usage["output_tokens"]
                total_tokens += usage["total_tokens"]
                for key in OPTIONAL_USAGE_FIELDS:
                    value = usage.get(key)
                    if value is not None:
                        optional_usage_totals[key] = (
                            (optional_usage_totals[key] or 0) + value
                        )
            recorder.append(
                "provider_response_accepted",
                {
                    "turn": provider_turns,
                    "actual_model_id": actual,
                    "exact_identity": actual == config.model,
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest().upper(),
                    "content_chars": len(content),
                    "usage": usage,
                    "raw_response_persisted": False,
                },
            )
            if actual != config.model:
                raise RuntimeError("EXACT_MODEL_IDENTITY_MISMATCH")
            try:
                action = parse_single_json_action(content)
            except Exception:
                action = None
            valid, error = _valid_action(action, phase=phase, selected=selected)
            dialogue.append({"turn": provider_turns, "state": state, "action": action})
            if not valid:
                recorder.append(
                    "protocol_violation",
                    {"turn": provider_turns, "category": error, "correction_available": not correction_used},
                )
                if correction_used:
                    raise RuntimeError("MODEL_ACTION_PROTOCOL_FAILURE")
                correction_used = True
                validation_error = error
                continue
            validation_error = None
            safe_action = dict(action)
            recorder.append("model_action", {"turn": provider_turns, "phase": phase, "action": safe_action})
            for angle in safe_action["angles_deg"]:
                result = run_angle(
                    case_root=artifact_case,
                    definition=definition,
                    reference_molecule=reference_molecule,
                    reference_path=artifact_case / "reference.xyz",
                    angle=angle,
                    executable=executable,
                    identity=dict(identity),
                    timeout=180,
                )
                recorder.append(
                    "native_point",
                    {
                        "phase": phase,
                        "angle_deg": angle,
                        "profile_eligible": result.get("profile_eligible"),
                        "result_sha256": hashlib.sha256(
                            canonical_bytes(result)
                        ).hexdigest().upper(),
                    },
                )
                if result.get("profile_eligible") is not True:
                    raise RuntimeError("NATIVE_XTB_POINT_UNAVAILABLE")
                selected.append(angle)
                energies.append(float(result["total_energy_hartree"]))
            latest_feedback, final_curve = _analysis(selected, energies)
            recorder.append(
                "profile_feedback",
                {"phase": phase, "selected_angles_deg": list(selected), "analysis": latest_feedback},
            )
            final_action = safe_action
            phase += 1
        if phase != 3 or final_action is None or final_curve is None:
            raise RuntimeError("PROVIDER_TURN_BUDGET_EXHAUSTED")
        recorder.append(
            "final_submitted",
            {"case_id": case_id, "selected_angles_deg": selected, "confidence": final_action["confidence"]},
        )
        recorder.append("run_complete", {"status": "COMPLETE", "native_xtb_attempts": len(selected)})
        runner_status = "COMPLETE"
        provider_status = "AVAILABLE"
    except ProviderCallError as exc:
        # Include failures from the terminal provider turn.  Successful-turn
        # retries are returned directly above; a turn that ultimately raises
        # can only be recovered from its append-only attempt receipts.
        provider_retries = sum(
            1
            for event in recorder.events
            if event.get("event") == "provider_attempt_failed"
            and event.get("payload", {}).get("retryable") is True
        )
        provider_status = (
            "UNAVAILABLE_AFTER_RETRIES"
            if exc.safe.get("retryable") is True or provider_retries > 0
            else "UNAVAILABLE_PERMANENT"
        )
        runner_status = "BLOCKED"
        failure = {"category": str(exc.safe.get("status", "UNKNOWN_PROVIDER_ERROR"))}
        if not recorder.terminal:
            recorder.append("run_failure", {"status": "BLOCKED", **failure})
    except Exception as exc:
        provider_status = "RESPONDED" if actual_model else "UNAVAILABLE"
        runner_status = "FAILED"
        failure = {"category": str(exc) if str(exc).isupper() else type(exc).__name__.upper()}
        if not recorder.terminal:
            recorder.append("run_failure", {"status": "FAILED", **failure})

    recorder.write(case_root / "trajectory.jsonl")
    ended_at = _utc_now()
    duration = round(time.monotonic() - started, 6)
    if runner_status == "COMPLETE" and final_curve is not None and final_action is not None:
        final_answer = {
            "schema_version": "3.0",
            "case_id": case_id,
            "final_status": "completed",
            "selected_angles_deg": selected,
            "confidence": final_action["confidence"],
            "rationale": final_action["rationale"],
            "predicted_energies_kcal_mol": final_curve,
        }
    else:
        final_answer = {
            "schema_version": "3.0",
            "case_id": case_id,
            "final_status": runner_status.lower(),
            "selected_angles_deg": selected,
            "failure": failure,
        }
    _write_json(case_root / "final_answer.json", final_answer)
    (case_root / "dialogue.md").write_text(_render_dialogue(dialogue), encoding="utf-8", newline="\n")
    profile_artifact_sha256 = (
        hashlib.sha256(
            canonical_bytes(
                {
                    "case_id": case_id,
                    "selected_angles_deg": selected,
                    "predicted_energies_kcal_mol": final_curve,
                }
            )
        ).hexdigest().upper()
        if final_curve is not None
        else None
    )
    artifact_hashes = {
        "trajectory_jsonl_sha256": hashlib.sha256(
            (case_root / "trajectory.jsonl").read_bytes()
        ).hexdigest().upper(),
        "final_answer_json_sha256": hashlib.sha256(
            (case_root / "final_answer.json").read_bytes()
        ).hexdigest().upper(),
        "dialogue_md_sha256": hashlib.sha256(
            (case_root / "dialogue.md").read_bytes()
        ).hexdigest().upper(),
        "case_spec_json_sha256": hashlib.sha256(
            (artifact_case / "case_spec.json").read_bytes()
        ).hexdigest().upper(),
        "reference_xyz_sha256": hashlib.sha256(
            (artifact_case / "reference.xyz").read_bytes()
        ).hexdigest().upper(),
        "reference_structure_json_sha256": hashlib.sha256(
            (artifact_case / "reference_structure.json").read_bytes()
        ).hexdigest().upper(),
    }
    metadata = {
        "schema_version": "3.0",
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": "3.0.0",
        "task_id": TASK_ID,
        "case_id": case_id,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "provider": "siliconflow",
        "endpoint": config.base_url,
        "logical_model": label,
        "requested_model_id": config.model,
        "actual_model_id": actual_model,
        "exact_model_identity": actual_model == config.model,
        "temperature": config.temperature,
        "seed": None,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration,
        "provider_turns": provider_turns,
        "provider_retries": provider_retries,
        "token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
        "reasoning_token_usage": {
            "reasoning_tokens": optional_usage_totals["reasoning_tokens"],
            "provider_field_available": optional_usage_totals["reasoning_tokens"] is not None,
        },
        "cache_usage": {
            "cached_input_tokens": optional_usage_totals["cached_input_tokens"],
            "cache_read_input_tokens": optional_usage_totals["cache_read_input_tokens"],
            "cache_creation_input_tokens": optional_usage_totals["cache_creation_input_tokens"],
            "provider_field_available": any(
                optional_usage_totals[key] is not None
                for key in OPTIONAL_USAGE_FIELDS
                if key != "reasoning_tokens"
            ),
        },
        "native_xtb_attempts": len(selected),
        "runner_status": runner_status,
        "provider_status": provider_status,
        "terminal_status": runner_status,
        "failure": failure,
        "environment_versions": {
            "python": sys.version.split()[0],
            "rdkit": rdkit.__version__,
            "xtb": identity.get("version"),
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
        },
        "xtb_identity": dict(identity),
        "oracle_commitment": {
            "profile_sha256": str(case["oracle_sha256"]).upper(),
            "suite_json_sha256": hashlib.sha256(
                (ROOT / "organizer_private" / "suite.json").read_bytes()
            ).hexdigest().upper(),
        },
        "profile_artifact_sha256": profile_artifact_sha256,
        "artifact_hashes": artifact_hashes,
        "raw_provider_response_persisted": False,
    }
    _write_json(case_root / "metadata.json", metadata)
    first = evaluate_case(case_root, case, project_root=ROOT)
    second = evaluate_case(case_root, case, project_root=ROOT)
    if canonical_bytes(first) != canonical_bytes(second):
        raise RuntimeError("V3_EVALUATOR_NONDETERMINISTIC")
    _write_json(case_root / "evaluation.json", first)
    replay = {
        "schema_version": "3.0",
        "canonical_match": True,
        "evaluation_sha256": hashlib.sha256(canonical_bytes(first)).hexdigest().upper(),
        "case_id": case_id,
        "logical_model": label,
    }
    _write_json(case_root / "replay_verification.json", replay)
    status = {
        "schema_version": "3.0",
        "logical_model": label,
        "case_id": case_id,
        "provider_status": first["provider_status"],
        "run_status": first["status"],
        "eligible": first["eligible"],
        "scientific_score": first["scientific_score"],
        "process_score": first["process_score"],
        "official_score": first["official_score"],
        "selected_angles_deg": first["selected_angles_deg"],
        "unique_valid_angles": first["unique_valid_angles"],
        "feedback_rounds": first["feedback_rounds"],
        "metrics": first["metrics"],
        "wall_time_seconds": duration,
        "requested_model_id": config.model,
        "actual_model_id": actual_model,
        "exact_model_identity": actual_model == config.model,
        "provider_turns": provider_turns,
        "provider_retries": provider_retries,
        "native_xtb_attempts": len(selected),
        "replay_match": True,
        "failure": failure,
    }
    _write_json(case_root / "status.json", status)
    print(json.dumps(status, ensure_ascii=True, sort_keys=True), flush=True)
    return status


def run_campaign(output_root: Path, models: tuple[str, ...]) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("CAMPAIGN_OUTPUT_EXISTS")
    cases, _ = _suite()
    output_root.mkdir(parents=True)
    locked_files = (
        Path(__file__).resolve(),
        ROOT / "framework" / "schemas.py",
        ROOT / "framework" / "siliconflow_adapter.py",
        ROOT / "evaluator" / "evaluate_multicase_run_v3.py",
        ROOT / "evaluator" / "evaluate_suite.py",
        ROOT / "evaluator" / "profile_recompute_v2.py",
        ROOT / "organizer_private" / "suite.json",
    )
    campaign_lock = {
        "schema_version": "3.0",
        "campaign_id": output_root.name,
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": "3.0.0",
        "task_id": TASK_ID,
        "protocol": PROTOCOL,
        "models": {label: FORMAL_MODELS[label] for label in models},
        "temperature": ModelConfig.for_label(models[0]).temperature,
        "seed": None,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest().upper(),
        "locked_file_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest().upper()
            for path in locked_files
        },
        "case_oracle_commitments": {
            str(case["case_id"]): str(case["oracle_sha256"]).upper()
            for case in cases
        },
    }
    _write_json(output_root / "campaign_lock.json", campaign_lock)
    executable = _locate_frozen_xtb()
    identity = xtb_identity(executable)
    api_key = load_runtime_credential(ROOT)
    rows: list[dict[str, Any]] = []
    expected = [str(case["case_id"]) for case in cases]
    for label in models:
        model_root = output_root / label
        for case in cases:
            rows.append(
                run_case(
                    label=label,
                    case=case,
                    case_root=model_root / str(case["case_id"]),
                    executable=executable,
                    identity=identity,
                    api_key=api_key,
                    campaign_id=output_root.name,
                )
            )
        records = [
            _json(model_root / case_id / "evaluation.json") for case_id in expected
        ]
        suite_evaluation = aggregate_suite(records, expected)
        _write_json(model_root / "suite_evaluation.json", suite_evaluation)
    campaign = {
        "schema_version": "3.0",
        "models": list(models),
        "case_count_per_model": 4,
        "run_count": len(rows),
        "all_replays_match": all(row["replay_match"] for row in rows),
        "all_exact_model_identities": all(
            row["exact_model_identity"] or row["run_status"] == "BLOCKED" for row in rows
        ),
        "campaign_lock_sha256": hashlib.sha256(
            (output_root / "campaign_lock.json").read_bytes()
        ).hexdigest().upper(),
        "rows": rows,
    }
    _write_json(output_root / "campaign_summary.json", campaign)
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=MODEL_ORDER)
    args = parser.parse_args()
    models = tuple(args.model) if args.model else MODEL_ORDER
    # Do not call Path.resolve() here.  On Windows it expands a SUBST drive
    # back to the long physical path; xTB 6.7.1 can then silently ignore its
    # control file once the native working-directory path exceeds the classic
    # path limit.  An absolute caller-supplied drive path is already unambiguous.
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    campaign = run_campaign(output, models)
    print(json.dumps({key: campaign[key] for key in ("run_count", "all_replays_match", "all_exact_model_identities")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .candidate_scoring import _pathology_reason
except ImportError:
    from candidate_scoring import _pathology_reason


def _load_records(candidate_path: Path) -> list[dict[str, Any]]:
    with gzip.open(candidate_path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _relax_structure(structure: Any, constraints: Mapping[str, Any], optimizer: Any) -> tuple[Any, dict[str, Any]]:
    config = constraints["selected_chgnet"]
    result = optimizer.relax(
        structure,
        relax_cell=False,
        fmax=float(config["fmax_ev_ang"]),
        steps=int(config["maximum_steps"]),
        verbose=False,
    )
    final_structure = result["final_structure"]
    trajectory = result["trajectory"]
    forces = np.asarray(trajectory.forces[-1], dtype=float)
    max_force = float(np.linalg.norm(forces, axis=1).max())
    pathology = _pathology_reason(final_structure, constraints)
    stable = max_force <= float(config["fmax_ev_ang"]) and pathology is None
    return final_structure, {
        "stable": stable,
        "max_force_ev_ang": max_force,
        "steps": len(trajectory.energies),
        "initial_energy_ev_atom": float(trajectory.energies[0]) / len(structure),
        "final_energy_ev_atom": float(trajectory.energies[-1]) / len(structure),
        "final_pathology": pathology,
    }


def run_selected_relaxation(
    *,
    candidate_ids: Sequence[str],
    candidate_path: Path,
    constraints_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if len(candidate_ids) != 5 or len(set(candidate_ids)) != 5:
        raise ValueError("exactly five distinct candidate_ids are required")
    records = _load_records(candidate_path)
    by_id = {str(record.get("candidate_id")): record for record in records}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"candidate_ids not found in submitted library: {missing}")

    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    from chgnet.model.dynamics import StructOptimizer
    from chgnet.model.model import CHGNet
    from pymatgen.core import Structure

    optimizer = StructOptimizer(model=CHGNet.load())
    for candidate_id in candidate_ids:
        record = by_id[candidate_id]
        structure = Structure.from_dict(record["structure"])
        initial_pathology = _pathology_reason(structure, constraints)
        item: dict[str, Any] = {
            "candidate_id": candidate_id,
            "initial_pathology": initial_pathology,
            "stable": False,
        }
        if initial_pathology is None:
            try:
                final_structure, relaxation = _relax_structure(structure, constraints, optimizer)
                relaxed_file = output_dir / f"{candidate_id}_relaxed.json"
                relaxed_file.write_text(
                    json.dumps(final_structure.as_dict(), indent=2) + "\n",
                    encoding="utf-8",
                )
                item.update(relaxation)
                item["relaxed_structure_file"] = str(relaxed_file.name)
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)

    report = {
        "protocol_version": constraints["selected_chgnet"]["protocol_version"],
        "candidate_ids": list(candidate_ids),
        "stable_count": sum(bool(item["stable"]) for item in results),
        "results": results,
    }
    (output_dir / "chgnet_relaxation.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def score_selected_candidates(
    *,
    selected_candidates: Sequence[Mapping[str, Any]],
    candidate_path: Path,
    constraints_path: Path,
    output_dir: Path,
    tool_attempts: int,
) -> dict[str, Any]:
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    maximum = float(constraints["scoring_weights"]["selected_chgnet_stability"])
    per_candidate = maximum / 5.0
    if tool_attempts < 1:
        return {
            "score": 0.0,
            "max_score": maximum,
            "stable_count": 0,
            "tool_attempts": tool_attempts,
            "reason": "one CHGNet tool attempt is required",
        }

    candidate_ids = [str(item.get("candidate_id", "")) for item in selected_candidates]
    if len(candidate_ids) != 5 or len(set(candidate_ids)) != 5:
        return {
            "score": 0.0,
            "max_score": maximum,
            "stable_count": 0,
            "tool_attempts": tool_attempts,
            "reason": "submission must identify five distinct candidates",
        }

    records = _load_records(candidate_path)
    predictions = sorted(
        (
            float(record["predicted_exfoliation_energy_mev_atom"]),
            str(record["candidate_id"]),
        )
        for record in records
        if np.isfinite(float(record.get("predicted_exfoliation_energy_mev_atom", np.nan)))
    )
    prediction_values = np.asarray([value for value, _ in predictions], dtype=float)
    minimum_std = float(constraints["selected_chgnet"]["minimum_prediction_std_mev_atom"])
    minimum_unique = int(constraints["selected_chgnet"]["minimum_unique_prediction_values"])
    prediction_std = float(np.std(prediction_values)) if len(prediction_values) else 0.0
    unique_prediction_values = len(set(prediction_values.tolist()))
    if prediction_std < minimum_std or unique_prediction_values < minimum_unique:
        return {
            "score": 0.0,
            "max_score": maximum,
            "stable_count": 0,
            "tool_attempts": tool_attempts,
            "prediction_std_mev_atom": prediction_std,
            "unique_prediction_values": unique_prediction_values,
            "reason": "candidate predictions are constant, near-constant, or insufficiently ranked",
        }
    pool_size = max(5, int(len(records) * float(constraints["selected_chgnet"]["low_prediction_fraction"])))
    eligible_ids = {candidate_id for _, candidate_id in predictions[:pool_size]}
    report = run_selected_relaxation(
        candidate_ids=candidate_ids,
        candidate_path=candidate_path,
        constraints_path=constraints_path,
        output_dir=output_dir,
    )
    for item in report["results"]:
        item["low_prediction_pool_eligible"] = item["candidate_id"] in eligible_ids
        item["credited"] = bool(item["stable"] and item["low_prediction_pool_eligible"])
    credited = sum(bool(item["credited"]) for item in report["results"])
    excess_penalty = 0.0
    if tool_attempts == 2:
        excess_penalty = 5.0
    elif tool_attempts == 3:
        excess_penalty = 10.0
    elif tool_attempts > 3:
        excess_penalty = maximum
    return {
        "score": max(0.0, per_candidate * credited - excess_penalty),
        "max_score": maximum,
        "stable_count": report["stable_count"],
        "credited_count": credited,
        "tool_attempts": tool_attempts,
        "excess_tool_call_penalty": excess_penalty,
        "low_prediction_pool_size": pool_size,
        "prediction_std_mev_atom": prediction_std,
        "unique_prediction_values": unique_prediction_values,
        "results": report["results"],
    }

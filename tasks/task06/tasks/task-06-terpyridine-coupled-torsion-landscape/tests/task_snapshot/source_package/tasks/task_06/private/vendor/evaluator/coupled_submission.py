"""Semantic validation for the Task 06 coupled-landscape answer.

JSON Schema owns types and local value bounds.  This module deliberately
contains only cross-reference checks; it does not infer coordinates, select a
route, or supply scientific hints to the model-visible runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return duplicate


def validate_coupled_submission(payload: Mapping[str, Any]) -> list[str]:
    """Return deterministic cross-reference errors for a schema-valid answer."""

    errors: list[str] = []
    coordinates = payload.get("coordinate_hypotheses") or []
    points = payload.get("sampled_points") or []
    basins = payload.get("basins") or []
    transitions = payload.get("transitions") or []

    coordinate_ids = [str(item.get("coordinate_id", "")) for item in coordinates]
    point_ids = [str(item.get("point_id", "")) for item in points]
    basin_ids = [str(item.get("basin_id", "")) for item in basins]
    transition_ids = [str(item.get("transition_id", "")) for item in transitions]
    coordinate_set = set(coordinate_ids)
    point_set = set(point_ids)
    basin_set = set(basin_ids)

    for label, values in (
        ("coordinate_id", coordinate_ids),
        ("point_id", point_ids),
        ("basin_id", basin_ids),
        ("transition_id", transition_ids),
    ):
        for duplicate in sorted(_duplicates(values)):
            errors.append(f"duplicate {label}: {duplicate}")

    for coordinate in coordinates:
        coordinate_id = str(coordinate.get("coordinate_id", ""))
        atom_ids = list(coordinate.get("atom_ids") or [])
        central_bond = list(coordinate.get("central_bond_atom_ids") or [])
        if len(set(atom_ids)) != len(atom_ids):
            errors.append(f"coordinate {coordinate_id}: non-unique atom_ids")
        if len(atom_ids) == 4 and central_bond != atom_ids[1:3] and central_bond != atom_ids[2:0:-1]:
            errors.append(f"coordinate {coordinate_id}: central bond must be the middle atom pair")

    for point in points:
        point_id = str(point.get("point_id", ""))
        value_ids = [str(item.get("coordinate_id", "")) for item in point.get("coordinate_values") or []]
        for duplicate in sorted(_duplicates(value_ids)):
            errors.append(f"point {point_id}: duplicate coordinate_id {duplicate}")
        for coordinate_id in value_ids:
            if coordinate_id not in coordinate_set:
                errors.append(f"point {point_id}: unknown coordinate_id {coordinate_id}")

    # A single scan job/frame may be declared only once, and 0/360-style
    # aliases within one scan job are the same periodic sample.  Independent
    # jobs may revisit a coordinate for validation, so the job ID remains part
    # of the key.
    seen_evidence: dict[tuple[str, int | None], str] = {}
    seen_periodic: dict[tuple[str, tuple[tuple[str, float], ...]], str] = {}
    for point in points:
        point_id = str(point.get("point_id", ""))
        job_id = str(point.get("source_job_id", ""))
        frame_index = point.get("frame_index")
        evidence_key = (job_id, frame_index)
        previous = seen_evidence.get(evidence_key)
        if previous is not None:
            errors.append(
                f"duplicate periodic sample: {point_id} reuses job/frame from {previous}"
            )
        else:
            seen_evidence[evidence_key] = point_id
        values = tuple(
            sorted(
                (
                    str(item.get("coordinate_id", "")),
                    round(float(item.get("actual_angle_deg", 0.0)) % 360.0, 6),
                )
                for item in point.get("coordinate_values") or []
                if isinstance(item, Mapping)
            )
        )
        periodic_key = (job_id, values)
        previous = seen_periodic.get(periodic_key)
        if previous is not None:
            errors.append(
                f"duplicate periodic sample: {point_id} is symmetry-equivalent to {previous}"
            )
        else:
            seen_periodic[periodic_key] = point_id

    points_by_id = {str(item.get("point_id", "")): item for item in points}
    for basin in basins:
        basin_id = str(basin.get("basin_id", ""))
        representative = str(basin.get("representative_point_id", ""))
        if representative not in point_set:
            errors.append(f"basin {basin_id}: unknown point_id {representative}")
        supporting = [str(value) for value in basin.get("supporting_point_ids") or []]
        for point_id in supporting:
            if point_id not in point_set:
                errors.append(f"basin {basin_id}: unknown point_id {point_id}")
        if representative in points_by_id:
            point = points_by_id[representative]
            if str(basin.get("representative_job_id", "")) != str(point.get("source_job_id", "")):
                errors.append(f"basin {basin_id}: representative job does not match its point")
            if str(basin.get("representative_geometry_path", "")) != str(point.get("geometry_path", "")):
                errors.append(f"basin {basin_id}: representative geometry path does not match its point")
            if str(basin.get("representative_geometry_sha256", "")) != str(point.get("geometry_sha256", "")):
                errors.append(f"basin {basin_id}: representative geometry hash does not match its point")

    for transition in transitions:
        transition_id = str(transition.get("transition_id", ""))
        for field in ("from_basin_id", "to_basin_id"):
            basin_id = str(transition.get(field, ""))
            if basin_id not in basin_set:
                errors.append(f"transition {transition_id}: unknown basin_id {basin_id}")
        ordered = [str(value) for value in transition.get("ordered_point_ids") or []]
        for point_id in ordered:
            if point_id not in point_set:
                errors.append(f"transition {transition_id}: unknown point_id {point_id}")
        highest = str(transition.get("highest_point_id", ""))
        if highest not in ordered:
            errors.append(f"transition {transition_id}: highest point must occur in ordered_point_ids")

    coupling = payload.get("coupling_assessment") or {}
    for point_id in coupling.get("supporting_point_ids") or []:
        if str(point_id) not in point_set:
            errors.append(f"coupling_assessment: unknown point_id {point_id}")

    evidence_jobs = set(str(value) for value in payload.get("evidence_job_ids") or [])
    for point in points:
        job_id = str(point.get("source_job_id", ""))
        if job_id and job_id not in evidence_jobs:
            errors.append(f"point {point.get('point_id', '')}: source job is absent from evidence_job_ids")

    return sorted(set(errors))

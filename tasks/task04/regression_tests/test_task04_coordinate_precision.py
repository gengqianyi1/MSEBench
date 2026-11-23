"""Submission XYZ formatting must not weaken trusted opt/Hessian binding."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/task-04-histidine-microstate-ensemble"
SNAPSHOT = TASK / "tests/task_snapshot"
ORACLE = SNAPSHOT / "organizer_private/oracle_harbor/repetition_01"


@pytest.fixture(scope="module")
def science():
    import sys

    sys.path.insert(0, str(SNAPSHOT / "source_package"))
    from tasks.task_04.private import grader
    from tasks.task_04.private.evaluator.evidence import load_trusted_jobs
    from tasks.task_04.private.evaluator.geometry import XYZGeometry, read_xyz

    jobs, errors = load_trusted_jobs(ORACLE)
    assert not errors
    answer = json.loads((ORACLE / "accepted_submission/final_answer.json").read_text())
    return grader, jobs, answer, XYZGeometry, read_xyz


def write_xyz(path, elements, coordinates, decimals):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [str(len(elements)), "coordinate serialization regression fixture"]
    rows.extend(
        f"{element} " + " ".join(f"{float(value):.{decimals}f}" for value in xyz)
        for element, xyz in zip(elements, coordinates)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def case(science, tmp_path):
    grader, jobs, answer, geometry_type, _ = science
    state = copy.deepcopy(answer["states"][0])
    payload = state["conditions"]["gas"]
    geometry = jobs[payload["optimization_job_id"]].optimized_geometry
    path = tmp_path / payload["geometry_path"]
    write_xyz(path, geometry.elements, geometry.coordinates, 14)
    return grader, dict(jobs), state, geometry, path, geometry_type


@pytest.mark.parametrize("decimals", [6, 8, 14])
def test_serialized_submission_retains_original_scientific_evidence(case, tmp_path, decimals):
    grader, jobs, state, geometry, path, _ = case
    write_xyz(path, geometry.elements, geometry.coordinates, decimals)
    result, error = grader._assessment(state, "gas", tmp_path, jobs)
    assert error is None
    assert result is not None
    # The accepted scientific geometry is the original job, never its rounded copy.
    assert np.array_equal(result["geometry"].coordinates, geometry.coordinates)
    assert result["gibbs_energy_hartree"] == -34.226516298359


@pytest.mark.parametrize("offset,accepted", [(0.9e-6, True), (1.1e-6, False)])
def test_submission_coordinate_tolerance_has_a_finite_boundary(case, tmp_path, offset, accepted):
    grader, jobs, state, geometry, path, _ = case
    coordinates = geometry.coordinates.copy()
    coordinates[0, 0] += offset
    write_xyz(path, geometry.elements, coordinates, 14)
    result, error = grader._assessment(state, "gas", tmp_path, jobs)
    assert (result is not None) is accepted
    assert (error is None) is accepted


@pytest.mark.parametrize("mutation", ["element", "atom_order", "atom_count"])
def test_submission_rounding_does_not_accept_atom_changes(case, tmp_path, mutation):
    grader, jobs, state, geometry, path, _ = case
    elements = list(geometry.elements)
    coordinates = geometry.coordinates.copy()
    if mutation == "element":
        elements[0] = "C"
    elif mutation == "atom_order":
        # The two O atoms have identical element symbols but distinct coordinates.
        coordinates[[0, 1]] = coordinates[[1, 0]]
    else:
        elements = elements[:-1]
        coordinates = coordinates[:-1]
    write_xyz(path, elements, coordinates, 14)
    result, error = grader._assessment(state, "gas", tmp_path, jobs)
    assert result is None
    assert error is not None


def test_hessian_input_binding_does_not_inherit_submission_tolerance(case, tmp_path):
    grader, jobs, state, geometry, _, geometry_type = case
    hessian_id = state["conditions"]["gas"]["thermochemistry_job_id"]
    coordinates = geometry.coordinates.copy()
    coordinates[0, 0] += 2e-8  # Above trusted 1e-8, well below submission 1e-6.
    jobs[hessian_id] = replace(
        jobs[hessian_id], input_geometry=geometry_type(geometry.elements, coordinates)
    )
    result, error = grader._assessment(state, "gas", tmp_path, jobs)
    assert result is None
    assert error is not None


@pytest.mark.parametrize("hessian_id", ["missing_job", "job_034"])
def test_wrong_hessian_reference_is_rejected(case, tmp_path, hessian_id):
    grader, jobs, state, _, _, _ = case
    state["conditions"]["gas"]["thermochemistry_job_id"] = hessian_id
    result, error = grader._assessment(state, "gas", tmp_path, jobs)
    assert result is None
    assert error is not None


def test_harbor_grader_rounding_preserves_golden_score(science, tmp_path):
    _, _, _, _, read_xyz = science
    spec = importlib.util.spec_from_file_location(
        "task04_rounding_protocol", TASK / "tests/protocol_grader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "rounded_oracle"
    shutil.copytree(ORACLE, run)
    for path in (run / "accepted_submission").rglob("*.xyz"):
        geometry = read_xyz(path)
        write_xyz(path, geometry.elements, geometry.coordinates, 6)
    metrics = {
        "agent_turns_used": 1,
        "scientific_turns_used": 1,
        "finalization_turns_used": 0,
        "tool_calls_used": 1,
        "termination_reason": "submitted",
    }
    result = module.grade_evidence(run, SNAPSHOT, metrics, 160)
    assert result["hard_zero"] is False
    assert result["score"] == 100.0
    assert result["score"] == result["task_local_score"]
    assert result["score"] == result["scientific_basis_score"]
    assert result["score"] == result["comparison_score"]

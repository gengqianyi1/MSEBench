from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK06 = ROOT / "tasks/task-06-terpyridine-coupled-torsion-landscape"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )








def _task06_result(root: Path) -> dict:
    grader = _load(TASK06 / "tests/protocol_grader.py", "task06_review_negative")
    return grader.grade_evidence(root, TASK06 / "tests/task_snapshot")


def _task06_fixture(tmp_path: Path) -> Path:
    source = TASK06 / "tests/task_snapshot/organizer_private/oracle/repetition_01"
    root = tmp_path / "task06"
    shutil.copytree(source, root)
    return root


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_periodic_angle",
        "invalid_grid_count",
        "missing_periodic_coverage",
        "symmetry_equivalent_boundary",
        "energy_inconsistency",
        "forged_frame_hash",
    ],
)
def test_task06_review_negative_ci_rejects_invalid_scan_claims(
    tmp_path: Path, mutation: str
) -> None:
    root = _task06_fixture(tmp_path)
    final_path = root / "accepted_submission/final_answer.json"
    final = _json(final_path)

    if mutation in {"duplicate_periodic_angle", "symmetry_equivalent_boundary"}:
        duplicate = copy.deepcopy(final["sampled_points"][0])
        duplicate["point_id"] = f"{mutation}_point"
        if mutation == "symmetry_equivalent_boundary":
            duplicate["coordinate_values"][0]["actual_angle_deg"] += 360.0
        final["sampled_points"].append(duplicate)
    elif mutation == "invalid_grid_count":
        jobs_path = root / "jobs.json"
        jobs = _json(jobs_path)
        jobs[0]["scan_point_count"] += 1
        _write_json(jobs_path, jobs)
    elif mutation == "missing_periodic_coverage":
        removed = {point["point_id"] for point in final["sampled_points"][:12]}
        final["sampled_points"] = final["sampled_points"][12:]
        final["coupling_assessment"]["supporting_point_ids"] = [
            point_id
            for point_id in final["coupling_assessment"]["supporting_point_ids"]
            if point_id not in removed
        ]
        final["answer_status"] = "complete"
    elif mutation == "energy_inconsistency":
        final["sampled_points"][0]["relative_energy_kcal_mol"] += 1.0
    elif mutation == "forged_frame_hash":
        final["sampled_points"][0]["geometry_sha256"] = "0" * 64

    _write_json(final_path, final)
    result = _task06_result(root)
    assert result["score"] < 100.0, (mutation, result)
    if mutation in {"duplicate_periodic_angle", "symmetry_equivalent_boundary"}:
        assert any("duplicate periodic sample" in error for error in result["contract_errors"])

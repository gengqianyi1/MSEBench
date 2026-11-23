from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK05 = ROOT / "tasks/task-05-caffeine-water-binding-decomposition"


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


def _task05_result(root: Path, *, outer: dict | None = None) -> dict:
    grader = _load(TASK05 / "tests/protocol_grader.py", "task05_review_negative")
    return grader.grade_evidence(
        root / "submission",
        TASK05 / "tests/task_snapshot",
        harness_manifest=outer,
    )


def _task05_fixture(tmp_path: Path) -> Path:
    source = TASK05 / "tests/task_snapshot/organizer_private/oracle/repetition_01"
    root = tmp_path / "task05"
    shutil.copytree(source, root)
    return root


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_job",
        "duplicate_job",
        "wrong_atom_order",
        "closure_error",
        "budget_exceeded",
        "forged_hash",
    ],
)
def test_task05_review_negative_ci_rejects_invalid_scientific_evidence(
    tmp_path: Path, mutation: str
) -> None:
    root = _task05_fixture(tmp_path)
    submission = root / "submission"
    manifest_path = submission / "run_manifest.json"
    manifest = _json(manifest_path)
    outer = None

    if mutation == "missing_job":
        shutil.rmtree(submission / manifest["jobs"][0]["workdir"])
    elif mutation == "duplicate_job":
        manifest["jobs"].append(copy.deepcopy(manifest["jobs"][0]))
        _write_json(manifest_path, manifest)
    elif mutation == "wrong_atom_order":
        target = next((submission / "xtb_jobs").rglob("input.xyz"))
        lines = target.read_text(encoding="utf-8").splitlines()
        first, second = lines[2].split(), lines[3].split()
        first[0], second[0] = second[0], first[0]
        lines[2], lines[3] = " ".join(first), " ".join(second)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif mutation == "closure_error":
        final_path = submission / "final_answer.json"
        final = _json(final_path)
        final["poses"][0]["binding_energy_kcal_mol"] += 1.0
        _write_json(final_path, final)
    elif mutation == "budget_exceeded":
        grader = _load(TASK05 / "tests/protocol_grader.py", "task05_outer_negative")
        outer = grader._outer_manifest(submission, grader._metrics({}))
        outer["budget_ledger"]["totals"]["xtb_jobs"] = 17
        outer["budget_ledger"]["events"][-1]["cumulative"]["xtb_jobs"] = 17
    elif mutation == "forged_hash":
        manifest["jobs"][0]["files_sha256"]["input.xyz"] = "0" * 64
        _write_json(manifest_path, manifest)

    result = _task05_result(root, outer=outer)
    assert result["score"] < 100.0, (mutation, result)
    assert result.get("hard_gates") or result.get("contract_errors") or result["score"] < 99.0







from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "task-18-redox-reorganization"
PUBLIC = TASK / "environment" / "public"
GRADER_ROOT = TASK / "tests" / "task_snapshot" / "private_grader"
GRADER = GRADER_ROOT / "grader.py"
HARTREE_TO_EV = 27.211386245988


def _load_grader():
    spec = importlib.util.spec_from_file_location("task18_grader_under_test", GRADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(GRADER_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(GRADER_ROOT))
    return module


def _energy_stdout(energy: float) -> str:
    return f"xTB test fixture\n | TOTAL ENERGY      {energy: .12f} Eh\n"


def _perturbed_xyz(source: Path, delta: float) -> str:
    lines = source.read_text(encoding="utf-8").splitlines()
    fields = lines[2].split()
    fields[1] = f"{float(fields[1]) + delta:.10f}"
    lines[2] = " ".join(fields)
    return "\n".join(lines) + "\n"


def _vibspectrum(xyz_text: str, imaginary: bool = False) -> str:
    atom_count = int(xyz_text.splitlines()[0])
    values = [0.0] * 6 + [100.0 + index for index in range(3 * atom_count - 6)]
    if imaginary and len(values) > 6:
        values[6] = -50.0
    return "$vibrational spectrum\n" + "\n".join(
        f"{index:6d} a {value:12.2f} 0.0 YES"
        for index, value in enumerate(values, start=1)
    ) + "\n"


def _write_job(
    run: Path,
    *,
    operation: str,
    input_path: str,
    input_text: str,
    charge: int,
    uhf: int,
    energy: float,
    output_text: str | None = None,
    imaginary: bool = False,
) -> str:
    jobs_path = run / "jobs.json"
    records = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else []
    job_id = f"job_{len(records) + 1:03d}"
    directory = run / "jobs" / job_id
    directory.mkdir(parents=True)
    input_name = "job_input.xyz"
    (directory / input_name).write_text(input_text, encoding="utf-8")
    (directory / "stdout.txt").write_text(_energy_stdout(energy), encoding="utf-8")
    (directory / "stderr.txt").write_text("", encoding="utf-8")
    if operation in {"opt", "opt_hess"}:
        optimized = output_text or input_text
        (directory / "xtbopt.xyz").write_text(optimized, encoding="utf-8")
        (directory / ".xtboptok").write_text("", encoding="utf-8")
    if operation in {"hess", "opt_hess"}:
        spectrum_geometry = output_text or input_text
        (directory / "hessian").write_text("$hessian\n 1.0\n$end\n", encoding="utf-8")
        (directory / "vibspectrum").write_text(
            _vibspectrum(spectrum_geometry, imaginary=imaginary), encoding="utf-8"
        )
    command = ["/opt/conda/bin/xtb", input_name, "--chrg", str(charge), "--uhf", str(uhf)]
    command += ["--ohess", "tight"] if operation == "opt_hess" else [f"--{operation}"]
    records.append(
        {
            "job_id": job_id,
            "operation": operation,
            "input": input_path,
            "input_snapshot": input_name,
            "command": command,
            "status": "completed",
            "return_code": 0,
            "output_directory": f"jobs/{job_id}",
        }
    )
    jobs_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return job_id


def _build_fixture(tmp_path: Path, candidate_count: int = 3) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    submission = run / "accepted_submission"
    reference = tmp_path / "reference"
    (run / "jobs").mkdir(parents=True)
    submission.mkdir(parents=True)
    (reference / "geometries").mkdir(parents=True)
    candidate_results = []
    search_assessment = []
    reference_candidates = {}
    energies = {
        "donor_01": (-100.00, -99.60, -99.58, -99.99),
        "donor_02": (-101.00, -100.62, -100.59, -100.98),
        "donor_03": (-102.00, -101.55, -101.51, -101.99),
    }
    for index in range(candidate_count):
        candidate_id = f"donor_{index + 1:02d}"
        seed_path = PUBLIC / "input" / f"{candidate_id}.xyz"
        seed = seed_path.read_text(encoding="utf-8")
        neutral = _perturbed_xyz(seed_path, 0.002 * (index + 1))
        cation = _perturbed_xyz(seed_path, 0.012 * (index + 1))
        alternate_neutral = _perturbed_xyz(seed_path, 0.12 + 0.01 * index)
        alternate_cation = _perturbed_xyz(seed_path, 0.18 + 0.01 * index)
        e_neutral, e_cation, e_cation_at_neutral, e_neutral_at_cation = energies[candidate_id]
        neutral_job = _write_job(
            run,
            operation="opt_hess",
            input_path=f"{candidate_id}.xyz",
            input_text=seed,
            charge=0,
            uhf=0,
            energy=e_neutral,
            output_text=neutral,
        )
        cation_job = _write_job(
            run,
            operation="opt_hess",
            input_path=f"{candidate_id}_cation_start.xyz",
            input_text=seed,
            charge=1,
            uhf=1,
            energy=e_cation,
            output_text=cation,
        )
        cation_at_neutral = _write_job(
            run,
            operation="sp",
            input_path=f"job_results/{neutral_job}/xtbopt.xyz",
            input_text=neutral,
            charge=1,
            uhf=1,
            energy=e_cation_at_neutral,
        )
        neutral_at_cation = _write_job(
            run,
            operation="sp",
            input_path=f"job_results/{cation_job}/xtbopt.xyz",
            input_text=cation,
            charge=0,
            uhf=0,
            energy=e_neutral_at_cation,
        )
        alternate_neutral_job = _write_job(
            run,
            operation="opt",
            input_path=f"{candidate_id}_neutral_alt.xyz",
            input_text=alternate_neutral,
            charge=0,
            uhf=0,
            energy=e_neutral + 0.01,
            output_text=alternate_neutral,
        )
        alternate_cation_job = _write_job(
            run,
            operation="opt",
            input_path=f"{candidate_id}_cation_alt.xyz",
            input_text=alternate_cation,
            charge=1,
            uhf=1,
            energy=e_cation + 0.01,
            output_text=alternate_cation,
        )
        lambda_c = (e_cation_at_neutral - e_cation) * HARTREE_TO_EV
        lambda_n = (e_neutral_at_cation - e_neutral) * HARTREE_TO_EV
        result = {
            "candidate_id": candidate_id,
            "lambda_h_ev": lambda_c + lambda_n,
            "lambda_c_ev": lambda_c,
            "lambda_n_ev": lambda_n,
            "vertical_oxidation_proxy_ev": (e_cation_at_neutral - e_neutral) * HARTREE_TO_EV,
            "adiabatic_oxidation_proxy_ev": (e_cation - e_neutral) * HARTREE_TO_EV,
            "neutral_optimization_job_id": neutral_job,
            "cation_optimization_job_id": cation_job,
            "cation_at_neutral_job_id": cation_at_neutral,
            "neutral_at_cation_job_id": neutral_at_cation,
            "neutral_hessian_job_id": neutral_job,
            "cation_hessian_job_id": cation_job,
            "evidence_job_ids": [neutral_job, cation_job, cation_at_neutral, neutral_at_cation],
        }
        candidate_results.append(result)
        search_assessment.append(
            {
                "candidate_id": candidate_id,
                "neutral_search_job_ids": [neutral_job, alternate_neutral_job],
                "cation_search_job_ids": [cation_job, alternate_cation_job],
                "unresolved_risk": "low",
                "basis": "Two distinct starts in each charge state support the adopted basin within this fixture.",
            }
        )
        neutral_path = reference / "geometries" / f"{candidate_id}_neutral.xyz"
        cation_path = reference / "geometries" / f"{candidate_id}_cation.xyz"
        neutral_path.write_text(neutral, encoding="utf-8")
        cation_path.write_text(cation, encoding="utf-8")
        reference_candidates[candidate_id] = {
            **{key: result[key] for key in (
                "lambda_h_ev",
                "lambda_c_ev",
                "lambda_n_ev",
                "vertical_oxidation_proxy_ev",
                "adiabatic_oxidation_proxy_ev",
            )},
            "neutral_basins": [
                {"geometry_path": f"geometries/{candidate_id}_neutral.xyz", "energy_hartree": e_neutral}
            ],
            "cation_basins": [
                {"geometry_path": f"geometries/{candidate_id}_cation.xyz", "energy_hartree": e_cation}
            ],
        }
    # The organizer reference is always complete even when a mutation submits only
    # part of the scientific answer.  This is what makes a partial scientific
    # failure score partially rather than becoming a malformed-reference hard zero.
    for index in range(candidate_count, 3):
        candidate_id = f"donor_{index + 1:02d}"
        seed_path = PUBLIC / "input" / f"{candidate_id}.xyz"
        neutral = _perturbed_xyz(seed_path, 0.002 * (index + 1))
        cation = _perturbed_xyz(seed_path, 0.012 * (index + 1))
        e_neutral, e_cation, e_cation_at_neutral, e_neutral_at_cation = energies[candidate_id]
        lambda_c = (e_cation_at_neutral - e_cation) * HARTREE_TO_EV
        lambda_n = (e_neutral_at_cation - e_neutral) * HARTREE_TO_EV
        neutral_path = reference / "geometries" / f"{candidate_id}_neutral.xyz"
        cation_path = reference / "geometries" / f"{candidate_id}_cation.xyz"
        neutral_path.write_text(neutral, encoding="utf-8")
        cation_path.write_text(cation, encoding="utf-8")
        reference_candidates[candidate_id] = {
            "lambda_h_ev": lambda_c + lambda_n,
            "lambda_c_ev": lambda_c,
            "lambda_n_ev": lambda_n,
            "vertical_oxidation_proxy_ev": (e_cation_at_neutral - e_neutral) * HARTREE_TO_EV,
            "adiabatic_oxidation_proxy_ev": (e_cation - e_neutral) * HARTREE_TO_EV,
            "neutral_basins": [
                {"geometry_path": f"geometries/{candidate_id}_neutral.xyz", "energy_hartree": e_neutral}
            ],
            "cation_basins": [
                {"geometry_path": f"geometries/{candidate_id}_cation.xyz", "energy_hartree": e_cation}
            ],
        }
    oxidation_ranking = ["donor_02", "donor_01", "donor_03"]
    lambda_ranking = ["donor_01", "donor_02", "donor_03"]
    pareto = ["donor_01", "donor_02"]
    answer = {
        "task_id": "task_18_redox_reorganization",
        "status": "completed" if candidate_count == 3 else "partial",
        "candidate_results": candidate_results,
        "oxidation_ease_ranking": oxidation_ranking,
        "low_reorganization_ranking": lambda_ranking,
        "pareto_front_candidate_ids": pareto,
        "recommendations": {
            "oxidation_access_priority": oxidation_ranking[0],
            "low_reorganization_priority": lambda_ranking[0],
        },
        "search_assessment": search_assessment,
        "decision_assessment": {
            "classification": "robust" if candidate_count == 3 else "insufficient_evidence",
            "basis": "The two independent rankings and Pareto relation follow the adopted evidence and searched alternatives.",
        },
        "geometry_response_summary": "The candidates show different oxidation relaxation energies; donor_01 and donor_02 define the trade-off while donor_03 is dominated.",
        "limitations": [
            "GFN2-xTB is a semiempirical gas-phase electronic model with property-specific ionization limits.",
            "Electronic coupling, outer-sphere reorganization, solvent free energy, and crystal packing are omitted.",
        ],
    }
    (submission / "final_answer.json").write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    (submission / "report.md").write_text(
        "# Evidence-based result\n\nThe two charge states were searched from distinct geometries. "
        "Adopted minima have converged Hessian evidence. The Pareto trade-off separates easier oxidation "
        "from lower structural reorganization, and the dominated candidate is not promoted.\n",
        encoding="utf-8",
    )
    (reference / "reference.json").write_text(
        json.dumps(
            {
                "schema_version": "task18-reference/4.1",
                "task_version": "4.1.0",
                "grader_version": "4.2.0",
                "candidate_values": reference_candidates,
                "oxidation_ease_ranking": oxidation_ranking,
                "low_reorganization_ranking": lambda_ranking,
                "pareto_front_candidate_ids": pareto,
                "recommendations": answer["recommendations"],
                "decision_classification": "robust",
                "tolerances": {
                    "energy_ev_full": 0.01,
                    "energy_ev_zero": 0.30,
                    "basin_energy_ev_full": 0.01,
                    "basin_rmsd_angstrom": 0.35,
                    "pareto_tie_ev": 0.01,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return submission, run, reference


@pytest.mark.parametrize("mutation", ["missing", "malformed", "schema_invalid"])
def test_invalid_final_contract_preserves_bounded_organizer_evidence_credit(
    tmp_path: Path, mutation: str
) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    final = submission / "final_answer.json"
    if mutation == "missing":
        final.unlink()
    elif mutation == "malformed":
        final.write_text("{", encoding="utf-8")
    else:
        value = json.loads(final.read_text(encoding="utf-8"))
        value["task_id"] = "wrong"
        final.write_text(json.dumps(value), encoding="utf-8")
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert result["score"] == pytest.approx(20.0)
    assert result["hard_zero"] is False
    assert result["submission_contract_valid"] is False
    assert result["contract_errors"] == [
        "missing_final_answer" if mutation == "missing" else "invalid_final_contract"
    ]
    assert result["sections"]["charge_state_basins_and_validation"] > 0
    assert result["sections"]["autonomous_search_and_resources"] > 0
    assert result["sections"]["numerical_redox_properties"] == 0
    assert result["sections"]["pareto_and_scientific_decision"] == 0
    assert result["sections"]["evidence_reliability_and_scope"] == 0


def test_no_trusted_xtb_job_remains_a_hard_zero(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    (run / "jobs.json").unlink()

    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)

    assert result["score"] == 0
    assert result["hard_zero"] is True
    assert result["hard_zero_reason"] == "no_trusted_task18_job"


def test_missing_report_only_removes_report_dependent_scope_credit(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    (submission / "report.md").unlink()

    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)

    assert result["score"] == pytest.approx(96.0)
    assert result["hard_zero"] is False
    assert result["submission_contract_valid"] is False
    assert result["contract_errors"] == ["missing_report"]
    assert result["sections"]["evidence_reliability_and_scope"] == pytest.approx(4.0)


def test_complete_trusted_fixture_scores_100(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert result["score"] == pytest.approx(100.0)
    assert all(
        result[key] == pytest.approx(100.0)
        for key in ("task_local_score", "scientific_basis_score", "comparison_score")
    )


def test_invalid_conformer_search_attenuates_numeric_decision_and_efficiency(
    tmp_path: Path,
) -> None:
    """Removing real multi-start support must not retain global-result credit."""

    submission, run, reference = _build_fixture(tmp_path)
    final_path = submission / "final_answer.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    for assessment, candidate in zip(
        final["search_assessment"], final["candidate_results"], strict=True
    ):
        assessment["neutral_search_job_ids"] = [
            candidate["neutral_optimization_job_id"]
        ]
        assessment["cation_search_job_ids"] = [
            candidate["cation_optimization_job_id"]
        ]
    final_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)

    assert result["sections"]["numerical_redox_properties"] == pytest.approx(21.0)
    assert result["sections"]["pareto_and_scientific_decision"] == pytest.approx(4.0)
    assert result["sections"]["autonomous_search_and_resources"] == pytest.approx(9.0)
    assert all(
        check["search"]["valid"] is False
        for check in result["derived"]["per_candidate_checks"].values()
    )


def test_unmatched_cation_basins_attenuate_numeric_decision_and_efficiency(
    tmp_path: Path,
) -> None:
    """Correct arithmetic on the wrong cation basins remains local partial credit."""

    submission, run, reference = _build_fixture(tmp_path)
    reference_path = reference / "reference.json"
    reference_value = json.loads(reference_path.read_text(encoding="utf-8"))
    for candidate in reference_value["candidate_values"].values():
        candidate["cation_basins"][0]["energy_hartree"] += 1.0
    reference_path.write_text(
        json.dumps(reference_value, indent=2) + "\n", encoding="utf-8"
    )

    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)

    assert result["sections"]["charge_state_basins_and_validation"] == pytest.approx(21.0)
    assert result["sections"]["numerical_redox_properties"] == pytest.approx(25.5)
    assert result["sections"]["pareto_and_scientific_decision"] == pytest.approx(4.0)
    assert result["sections"]["autonomous_search_and_resources"] == pytest.approx(9.0)
    assert all(
        check["cation_basin"]["matched"] is False
        for check in result["derived"]["per_candidate_checks"].values()
    )


def test_scientific_partial_failure_is_not_automatic_zero(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path, candidate_count=1)
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert 0 < result["score"] < 100
    assert result["hard_zero"] is False


def test_wrong_charge_breaks_only_dependent_science(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    jobs = json.loads((run / "jobs.json").read_text(encoding="utf-8"))
    jobs[2]["command"][jobs[2]["command"].index("--chrg") + 1] = "0"
    (run / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert 0 < result["score"] < 100
    assert result["hard_zero"] is False


def test_wrong_reported_energy_loses_numerical_points(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    final = submission / "final_answer.json"
    value = json.loads(final.read_text(encoding="utf-8"))
    value["candidate_results"][0]["lambda_h_ev"] += 0.30
    final.write_text(json.dumps(value), encoding="utf-8")
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert result["sections"]["numerical_redox_properties"] < 34
    assert result["score"] < 100


def test_wrong_pareto_front_loses_decision_points_only(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    final = submission / "final_answer.json"
    value = json.loads(final.read_text(encoding="utf-8"))
    value["pareto_front_candidate_ids"] = ["donor_01", "donor_02", "donor_03"]
    final.write_text(json.dumps(value), encoding="utf-8")
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert result["sections"]["pareto_and_scientific_decision"] < 16
    assert result["sections"]["numerical_redox_properties"] == pytest.approx(34.0)


def test_false_hessian_evidence_loses_basin_validation_not_entire_task(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    (run / "jobs" / "job_001" / "vibspectrum").unlink()
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert 0 < result["score"] < 100
    assert result["hard_zero"] is False
    assert result["sections"]["charge_state_basins_and_validation"] < 30
    assert result["sections"]["numerical_redox_properties"] == pytest.approx(30.0)
    assert result["sections"]["pareto_and_scientific_decision"] == pytest.approx(4.0)
    assert result["sections"]["autonomous_search_and_resources"] == pytest.approx(9.0)


def test_fabricated_role_job_id_loses_binding_and_dependent_science(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    final = submission / "final_answer.json"
    value = json.loads(final.read_text(encoding="utf-8"))
    value["candidate_results"][0]["cation_at_neutral_job_id"] = "job_999"
    value["candidate_results"][0]["evidence_job_ids"].append("job_999")
    final.write_text(json.dumps(value), encoding="utf-8")
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert result["score"] < 100
    assert result["hard_zero"] is False
    assert result["sections"]["evidence_reliability_and_scope"] < 8


def test_score_is_finite_and_unmapped(tmp_path: Path) -> None:
    submission, run, reference = _build_fixture(tmp_path)
    result = _load_grader().grade_task18(submission, run, reference, PUBLIC)
    assert math.isfinite(result["score"])
    assert result["score"] == result["task_local_score"]
    assert result["score"] == result["scientific_basis_score"]
    assert result["score"] == result["comparison_score"]

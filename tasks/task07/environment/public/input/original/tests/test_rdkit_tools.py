from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools import (
    TOOL_REGISTRY,
    calculate_properties,
    calculate_qed,
    calculate_similarity,
    canonicalize_smiles,
    check_scaffold,
    check_substructure,
    evaluate_candidate,
    validate_smiles,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "task.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_public_tool_registry_is_complete() -> None:
    assert set(TOOL_REGISTRY) == {
        "validate_smiles",
        "canonicalize_smiles",
        "calculate_qed",
        "calculate_properties",
        "calculate_similarity",
        "check_scaffold",
        "check_substructure",
        "evaluate_candidate",
    }


@pytest.mark.parametrize(
    ("smiles", "error_fragment"),
    [
        (None, "string"),
        ("", "empty"),
        ("not-a-smiles", "parse"),
        ("CC.O", "multi-component"),
        ("C[Si](C)C", "disallowed"),
    ],
)
def test_validate_smiles_rejects_unsafe_inputs_without_raising(
    smiles: object, error_fragment: str, config: dict
) -> None:
    result = validate_smiles(smiles, config)
    assert result["ok"] is False
    assert result["valid"] is False
    assert result["canonical_smiles"] is None
    assert error_fragment.lower() in result["error"].lower()


def test_validate_smiles_rejects_overlong_input(config: dict) -> None:
    result = validate_smiles("C" * 513, config)
    assert result["ok"] is False
    assert "maximum length" in result["error"]


def test_validate_smiles_reports_formal_charge_without_executing_code(config: dict) -> None:
    charged = validate_smiles("C[NH+](C)C", config)
    assert charged["ok"] is True
    assert charged["valid"] is True
    assert charged["formal_charge"] == 1
    assert charged["neutral_formal_charge"] is False


def test_tool_config_paths_cannot_escape_project() -> None:
    outside = ROOT.parent / "outside-task.yaml"
    result = validate_smiles("CC", outside)
    assert result["ok"] is False
    assert "inside the benchmark project" in result["error"]


def test_canonicalize_smiles_is_deterministic(config: dict) -> None:
    first = canonicalize_smiles("C(C)O", config)
    second = canonicalize_smiles("OCC", config)
    assert first["ok"] and second["ok"]
    assert first["canonical_smiles"] == second["canonical_smiles"] == "CCO"


def test_calculate_qed_and_properties_match_known_rdkit_values(config: dict) -> None:
    start = config["task"]["start_smiles"]
    qed_result = calculate_qed(start, config)
    property_result = calculate_properties(start, config)

    assert qed_result["qed"] == pytest.approx(0.769002, abs=1e-6)
    assert property_result["properties"] == {
        "molecular_weight": pytest.approx(207.229, abs=1e-4),
        "logp": pytest.approx(1.8217, abs=1e-4),
        "tpsa": pytest.approx(55.4, abs=1e-4),
        "hbond_donors": 1,
        "hbond_acceptors": 3,
        "rotatable_bonds": 3,
        "ring_count": 1,
        "aromatic_ring_count": 1,
        "halogen_count": 0,
        "formal_charge": 0,
    }


def test_property_and_qed_errors_are_structured(config: dict) -> None:
    assert calculate_qed("bad", config)["qed"] is None
    assert calculate_properties("bad", config)["properties"] is None
    assert calculate_qed("bad", config)["ok"] is False
    assert calculate_properties("bad", config)["ok"] is False


def test_morgan_similarity_is_symmetric_and_rounded(config: dict) -> None:
    molecule_a = "CCO"
    molecule_b = "CCCO"
    forward = calculate_similarity(molecule_a, molecule_b, config)
    reverse = calculate_similarity(molecule_b, molecule_a, config)

    assert forward["ok"] and reverse["ok"]
    assert forward["similarity"] == reverse["similarity"]
    assert 0.0 < forward["similarity"] < 1.0
    assert calculate_similarity(molecule_a, molecule_a, config)["similarity"] == 1.0


def test_similarity_rejects_bad_molecules_and_parameters(config: dict) -> None:
    assert calculate_similarity("bad", "CC", config)["ok"] is False
    assert calculate_similarity("CC", "CC", config, radius=999)["ok"] is False


def test_scaffold_and_substructure_checks(config: dict) -> None:
    start = config["task"]["start_smiles"]
    scaffold_result = check_scaffold(start, config=config)
    substructure_result = check_substructure(start, "c1ccccc1", config)

    assert scaffold_result["ok"] is True
    assert scaffold_result["scaffold_match"] is True
    assert substructure_result["ok"] is True
    assert substructure_result["has_substructure"] is True


@pytest.mark.parametrize("smarts", [None, "", "[", "C("])
def test_malformed_smarts_never_crashes(smarts: object, config: dict) -> None:
    result = check_substructure(config["task"]["start_smiles"], smarts, config)
    assert result["ok"] is False
    assert result["matches"] is False
    assert isinstance(result["error"], str)


def test_evaluate_candidate_reports_all_constraints_and_feasibility(config: dict) -> None:
    candidate = config["task"]["start_smiles"]
    result = evaluate_candidate(candidate, config)

    assert result["ok"] is True
    assert result["valid"] is True
    assert result["feasible"] is True
    assert result["scaffold"] is True
    assert result["required_substructure_retention"] is True
    assert result["required_substructures"] == []
    assert result["qed"] == 0.769002
    assert result["similarity_to_start"] == 1.0
    assert result["violations"] == []
    assert all(result["constraint_flags"].values())


def test_halogen_count_is_publicly_reported_and_hard_gated(config: dict) -> None:
    chlorinated = "CCOC(=O)c1ccc(NC(C)=O)c(Cl)c1"
    methylated = "CCOC(=O)c1ccc(NC(C)=O)c(C)c1"

    chlorinated_result = evaluate_candidate(chlorinated, config)
    methylated_result = evaluate_candidate(methylated, config)

    assert chlorinated_result["properties"]["halogen_count"] == 1
    assert chlorinated_result["constraint_flags"]["halogen_count"] is False
    assert chlorinated_result["feasible"] is False
    assert methylated_result["properties"]["halogen_count"] == 0
    assert methylated_result["constraint_flags"]["halogen_count"] is True
    assert methylated_result["feasible"] is True


def test_aromatic_amide_constraint_is_one_connected_match(config: dict) -> None:
    config = yaml.safe_load(yaml.safe_dump(config))
    config["constraints"]["scaffold"] = {
        "smarts": "O=C(O[*])c1ccc(NC(=O)[#6])cc1"
    }
    config["constraints"]["required_substructures"] = [
        {
            "name": "acetanilide_amide",
            "smarts": "[c][NX3][CX3](=[OX1])[#6]",
        }
    ]
    decoy = "CC(=O)NCCN(CC)CCOC(=O)c1ccc(NC)cc1"
    result = evaluate_candidate(decoy, config)

    assert result["ok"] is True
    assert result["valid"] is True
    assert result["feasible"] is False
    assert result["qed"] is not None
    assert result["scaffold_match"] is False
    assert result["required_substructure_retention"] is False
    assert result["constraint_flags"]["scaffold"] is False
    assert result["constraint_flags"]["required_substructures"] is False


def test_start_candidate_remains_usable_intermediate_feedback(config: dict) -> None:
    result = evaluate_candidate(config["task"]["start_smiles"], config)

    assert result["ok"] is True
    assert result["valid"] is True
    assert result["feasible"] is True
    assert result["error"] is None


def test_evaluate_candidate_safely_rejects_invalid_and_infeasible(config: dict) -> None:
    invalid = evaluate_candidate("CC.[Na+]", config)
    too_small = evaluate_candidate("c1ccccc1", config)
    charged = evaluate_candidate("C[NH+](C)CCOC(=O)c1ccc(NC(C)=O)cc1", config)

    assert invalid["ok"] is False
    assert invalid["valid"] is False
    assert invalid["canonical_smiles"] is None
    assert invalid["qed"] is None
    assert invalid["feasible"] is False

    assert too_small["ok"] is True
    assert too_small["valid"] is True
    assert too_small["feasible"] is False
    assert too_small["constraint_flags"]["molecular_weight"] is False

    assert charged["valid"] is True
    assert charged["feasible"] is False
    assert charged["constraint_flags"]["neutral_formal_charge"] is False


def test_evaluate_candidate_handles_malformed_configured_scaffold(config: dict) -> None:
    altered = yaml.safe_load(yaml.safe_dump(config))
    altered["constraints"]["scaffold"]["smarts"] = "["
    result = evaluate_candidate(config["task"]["start_smiles"], altered)
    assert result["ok"] is False
    assert result["feasible"] is False
    assert result["constraint_flags"]["scaffold"] is False

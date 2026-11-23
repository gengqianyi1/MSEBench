"""Safe, deterministic RDKit tools exposed by the benchmark.

The functions in this module deliberately return JSON-serialisable mappings
instead of RDKit objects.  Invalid user input is data, not an exception: every
public function returns ``ok=False`` and a short ``error`` string when it
cannot complete its calculation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "task.yaml"
MAX_CONFIG_BYTES = 1_048_576
DEFAULT_MAX_SMILES_LENGTH = 512
DEFAULT_MAX_SMARTS_LENGTH = 4096
PROPERTY_DECIMALS = 4
SCORE_DECIMALS = 6


def _error_result(error: str, **values: Any) -> dict[str, Any]:
    return {"ok": False, "error": error, **values}


def _success_result(**values: Any) -> dict[str, Any]:
    return {"ok": True, "error": None, **values}


def _load_config(config: Any = None) -> tuple[dict[str, Any], str | None]:
    """Return a task configuration without raising on caller-controlled input."""

    if isinstance(config, Mapping):
        return dict(config), None

    path: Path
    if config is None:
        path = DEFAULT_CONFIG_PATH
    elif isinstance(config, (str, Path)):
        try:
            path = Path(config)
        except (TypeError, ValueError, OSError):
            return {}, "Configuration path is invalid."
    else:
        return {}, "Configuration must be a mapping or a YAML file path."

    try:
        resolved = path.expanduser().resolve()
        resolved.relative_to(PROJECT_ROOT.resolve())
    except (OSError, RuntimeError, ValueError):
        return {}, "Configuration path must stay inside the benchmark project."

    try:
        if resolved.stat().st_size > MAX_CONFIG_BYTES:
            return {}, f"Configuration exceeds {MAX_CONFIG_BYTES} bytes."
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}, "Configuration could not be loaded."
    if not isinstance(loaded, Mapping):
        return {}, "Configuration root must be a mapping."
    return dict(loaded), None


def _constraints(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("constraints", {})
    return value if isinstance(value, Mapping) else {}


def _max_smiles_length(config: Mapping[str, Any], override: Any = None) -> int:
    value = override
    if value is None:
        output = config.get("output", {})
        if isinstance(output, Mapping):
            value = output.get("max_smiles_length", DEFAULT_MAX_SMILES_LENGTH)
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = DEFAULT_MAX_SMILES_LENGTH
    return max(1, min(parsed, 100_000))


def _allowed_atomic_numbers(
    config: Mapping[str, Any], override: Any = None
) -> set[int] | None:
    value = override
    if value is None:
        value = _constraints(config).get("allowed_atomic_numbers")
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return set()
    try:
        return {int(number) for number in value}
    except (TypeError, ValueError, OverflowError):
        return set()


def _parse_smiles(
    smiles: Any,
    config: Mapping[str, Any],
    *,
    max_length: Any = None,
    allowed_atomic_numbers: Any = None,
    require_single_component: Any = None,
) -> tuple[Chem.Mol | None, dict[str, Any]]:
    """Parse and apply the benchmark's input-level molecule restrictions."""

    empty = {
        "valid": False,
        "canonical_smiles": None,
        "component_count": 0,
        "atom_count": 0,
        "formal_charge": None,
        "neutral_formal_charge": False,
        "disallowed_atomic_numbers": [],
    }
    if not isinstance(smiles, str):
        return None, _error_result("SMILES must be a string.", **empty)

    text = smiles.strip()
    if not text:
        return None, _error_result("SMILES must not be empty.", **empty)

    length_limit = _max_smiles_length(config, max_length)
    if len(text) > length_limit:
        return None, _error_result(
            f"SMILES exceeds the maximum length of {length_limit} characters.",
            **empty,
        )

    constraints = _constraints(config)
    if require_single_component is None:
        require_single_component = bool(constraints.get("require_single_component", True))
    else:
        require_single_component = bool(require_single_component)

    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(text, sanitize=True)
    except Exception:
        mol = None
    if mol is None or mol.GetNumAtoms() == 0:
        return None, _error_result("RDKit could not parse and sanitize the SMILES.", **empty)

    try:
        component_count = len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))
    except Exception:
        return None, _error_result("RDKit could not inspect molecule components.", **empty)
    atom_count = int(mol.GetNumAtoms())
    formal_charge = int(Chem.GetFormalCharge(mol))
    if require_single_component and component_count != 1:
        return None, _error_result(
            "Disconnected or multi-component molecules are not allowed.",
            valid=False,
            canonical_smiles=None,
            component_count=component_count,
            atom_count=atom_count,
            formal_charge=formal_charge,
            neutral_formal_charge=formal_charge == 0,
            disallowed_atomic_numbers=[],
        )

    allowed = _allowed_atomic_numbers(config, allowed_atomic_numbers)
    disallowed = sorted(
        {
            int(atom.GetAtomicNum())
            for atom in mol.GetAtoms()
            if allowed is not None and int(atom.GetAtomicNum()) not in allowed
        }
    )
    if disallowed:
        return None, _error_result(
            "Molecule contains disallowed atomic numbers: "
            + ", ".join(str(number) for number in disallowed)
            + ".",
            valid=False,
            canonical_smiles=None,
            component_count=component_count,
            atom_count=atom_count,
            formal_charge=formal_charge,
            neutral_formal_charge=formal_charge == 0,
            disallowed_atomic_numbers=disallowed,
        )

    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None, _error_result(
            "RDKit could not canonicalize the molecule.",
            valid=False,
            canonical_smiles=None,
            component_count=component_count,
            atom_count=atom_count,
            formal_charge=formal_charge,
            neutral_formal_charge=formal_charge == 0,
            disallowed_atomic_numbers=[],
        )
    return mol, _success_result(
        valid=True,
        canonical_smiles=canonical,
        component_count=component_count,
        atom_count=atom_count,
        formal_charge=formal_charge,
        neutral_formal_charge=formal_charge == 0,
        disallowed_atomic_numbers=[],
    )


def _prepare(smiles: Any, config: Any = None, **kwargs: Any) -> tuple[Chem.Mol | None, dict[str, Any], dict[str, Any]]:
    loaded, config_error = _load_config(config)
    if config_error:
        result = _error_result(
            config_error,
            valid=False,
            canonical_smiles=None,
            component_count=0,
            atom_count=0,
            formal_charge=None,
            neutral_formal_charge=False,
            disallowed_atomic_numbers=[],
        )
        return None, result, {}
    mol, result = _parse_smiles(smiles, loaded, **kwargs)
    return mol, result, loaded


def _rounded(value: Any, digits: int) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0 else result


def _descriptor_values(mol: Chem.Mol) -> dict[str, int | float]:
    return {
        "molecular_weight": _rounded(Descriptors.MolWt(mol), PROPERTY_DECIMALS),
        "logp": _rounded(Crippen.MolLogP(mol), PROPERTY_DECIMALS),
        "tpsa": _rounded(rdMolDescriptors.CalcTPSA(mol), PROPERTY_DECIMALS),
        "hbond_donors": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hbond_acceptors": int(rdMolDescriptors.CalcNumHBA(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "halogen_count": int(
            sum(
                atom.GetAtomicNum() in {9, 17, 35, 53, 85}
                for atom in mol.GetAtoms()
            )
        ),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
    }


def validate_smiles(
    smiles: Any,
    config: Any = None,
    *,
    max_length: int | None = None,
    allowed_atomic_numbers: Any = None,
    require_single_component: bool | None = None,
) -> dict[str, Any]:
    """Validate a SMILES against input restrictions in ``configs/task.yaml``."""

    _, result, _ = _prepare(
        smiles,
        config,
        max_length=max_length,
        allowed_atomic_numbers=allowed_atomic_numbers,
        require_single_component=require_single_component,
    )
    return result


def canonicalize_smiles(smiles: Any, config: Any = None) -> dict[str, Any]:
    """Return a canonical isomeric SMILES, or a structured validation error."""

    _, result, _ = _prepare(smiles, config)
    return result


def calculate_qed(smiles: Any, config: Any = None) -> dict[str, Any]:
    """Calculate QED rounded to six decimal places."""

    mol, validation, _ = _prepare(smiles, config)
    if mol is None:
        return {
            **validation,
            "qed": None,
        }
    try:
        value = _rounded(QED.qed(mol), SCORE_DECIMALS)
    except Exception:
        return _error_result(
            "RDKit could not calculate QED.",
            valid=True,
            canonical_smiles=validation["canonical_smiles"],
            qed=None,
        )
    return _success_result(
        valid=True,
        canonical_smiles=validation["canonical_smiles"],
        qed=value,
    )


def calculate_properties(smiles: Any, config: Any = None) -> dict[str, Any]:
    """Calculate all public molecular descriptors used by the benchmark."""

    mol, validation, _ = _prepare(smiles, config)
    if mol is None:
        return {
            **validation,
            "properties": None,
        }
    try:
        properties = _descriptor_values(mol)
    except Exception:
        return _error_result(
            "RDKit could not calculate molecular properties.",
            valid=True,
            canonical_smiles=validation["canonical_smiles"],
            properties=None,
        )
    return _success_result(
        valid=True,
        canonical_smiles=validation["canonical_smiles"],
        properties=properties,
        **properties,
    )


def _raw_similarity_from_mols(
    mol_a: Chem.Mol, mol_b: Chem.Mol, radius: int, n_bits: int
) -> float:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint_a = generator.GetFingerprint(mol_a)
    fingerprint_b = generator.GetFingerprint(mol_b)
    return float(DataStructs.TanimotoSimilarity(fingerprint_a, fingerprint_b))


def _similarity_from_mols(
    mol_a: Chem.Mol, mol_b: Chem.Mol, radius: int, n_bits: int
) -> float:
    return _rounded(
        _raw_similarity_from_mols(mol_a, mol_b, radius, n_bits), SCORE_DECIMALS
    )


def calculate_similarity(
    smiles_a: Any,
    smiles_b: Any,
    config: Any = None,
    *,
    radius: int | None = None,
    n_bits: int | None = None,
) -> dict[str, Any]:
    """Calculate Morgan-fingerprint Tanimoto similarity for two molecules."""

    loaded, config_error = _load_config(config)
    if config_error:
        return _error_result(
            config_error,
            similarity=None,
            canonical_smiles_a=None,
            canonical_smiles_b=None,
        )

    similarity_cfg = _constraints(loaded).get("similarity", {})
    if not isinstance(similarity_cfg, Mapping):
        similarity_cfg = {}
    radius_value = similarity_cfg.get("radius", 2) if radius is None else radius
    bits_value = similarity_cfg.get("n_bits", 2048) if n_bits is None else n_bits
    try:
        radius_value = int(radius_value)
        bits_value = int(bits_value)
    except (TypeError, ValueError, OverflowError):
        return _error_result(
            "Fingerprint radius and bit count must be integers.",
            similarity=None,
            canonical_smiles_a=None,
            canonical_smiles_b=None,
        )
    if not 0 <= radius_value <= 8 or not 16 <= bits_value <= 65_536:
        return _error_result(
            "Fingerprint radius or bit count is outside the safe range.",
            similarity=None,
            canonical_smiles_a=None,
            canonical_smiles_b=None,
        )

    mol_a, result_a = _parse_smiles(smiles_a, loaded)
    if mol_a is None:
        return _error_result(
            "First SMILES is invalid: " + str(result_a["error"]),
            similarity=None,
            canonical_smiles_a=None,
            canonical_smiles_b=None,
        )
    mol_b, result_b = _parse_smiles(smiles_b, loaded)
    if mol_b is None:
        return _error_result(
            "Second SMILES is invalid: " + str(result_b["error"]),
            similarity=None,
            canonical_smiles_a=result_a["canonical_smiles"],
            canonical_smiles_b=None,
        )
    try:
        similarity = _similarity_from_mols(mol_a, mol_b, radius_value, bits_value)
    except Exception:
        return _error_result(
            "RDKit could not calculate fingerprint similarity.",
            similarity=None,
            canonical_smiles_a=result_a["canonical_smiles"],
            canonical_smiles_b=result_b["canonical_smiles"],
        )
    return _success_result(
        similarity=similarity,
        method="morgan_tanimoto",
        radius=radius_value,
        n_bits=bits_value,
        canonical_smiles_a=result_a["canonical_smiles"],
        canonical_smiles_b=result_b["canonical_smiles"],
    )


def _parse_smarts(smarts: Any) -> tuple[Chem.Mol | None, str | None]:
    if not isinstance(smarts, str):
        return None, "SMARTS must be a string."
    text = smarts.strip()
    if not text:
        return None, "SMARTS must not be empty."
    if len(text) > DEFAULT_MAX_SMARTS_LENGTH:
        return None, f"SMARTS exceeds the maximum length of {DEFAULT_MAX_SMARTS_LENGTH} characters."
    try:
        with rdBase.BlockLogs():
            query = Chem.MolFromSmarts(text)
    except Exception:
        query = None
    if query is None or query.GetNumAtoms() == 0:
        return None, "RDKit could not parse the SMARTS pattern."
    return query, None


def check_substructure(smiles: Any, smarts: Any, config: Any = None) -> dict[str, Any]:
    """Check a SMARTS substructure match without exposing RDKit objects."""

    mol, validation, _ = _prepare(smiles, config)
    if mol is None:
        return {
            **validation,
            "matches": False,
            "has_substructure": False,
        }
    query, smarts_error = _parse_smarts(smarts)
    if query is None:
        return _error_result(
            str(smarts_error),
            valid=True,
            canonical_smiles=validation["canonical_smiles"],
            matches=False,
            has_substructure=False,
        )
    try:
        matches = bool(mol.HasSubstructMatch(query, useChirality=True))
    except Exception:
        return _error_result(
            "RDKit could not perform the substructure match.",
            valid=True,
            canonical_smiles=validation["canonical_smiles"],
            matches=False,
            has_substructure=False,
        )
    return _success_result(
        valid=True,
        canonical_smiles=validation["canonical_smiles"],
        smarts=smarts.strip(),
        matches=matches,
        has_substructure=matches,
    )


def check_scaffold(
    smiles: Any,
    scaffold_smarts: Any = None,
    config: Any = None,
) -> dict[str, Any]:
    """Check the configured scaffold, or an explicitly supplied scaffold SMARTS."""

    loaded, config_error = _load_config(config)
    if config_error:
        return _error_result(
            config_error,
            valid=False,
            canonical_smiles=None,
            scaffold_match=False,
            matches=False,
        )
    if scaffold_smarts is None:
        scaffold_cfg = _constraints(loaded).get("scaffold", {})
        if isinstance(scaffold_cfg, Mapping):
            scaffold_smarts = scaffold_cfg.get("smarts")
    result = check_substructure(smiles, scaffold_smarts, loaded)
    matches = bool(result.get("matches", False))
    return {
        **result,
        "scaffold_match": matches,
    }


def _within_range(value: int | float, specification: Any) -> bool:
    if specification is None:
        return True
    if not isinstance(specification, Mapping):
        return False
    try:
        minimum = float(specification.get("min", float("-inf")))
        maximum = float(specification.get("max", float("inf")))
        return minimum <= float(value) <= maximum
    except (TypeError, ValueError, OverflowError):
        return False


def _empty_constraint_flags() -> dict[str, bool]:
    return {
        "valid_smiles": False,
        "single_component": False,
        "allowed_atoms": False,
        "neutral_formal_charge": False,
        "molecular_weight": False,
        "logp": False,
        "tpsa": False,
        "hbond_donors": False,
        "hbond_acceptors": False,
        "rotatable_bonds": False,
        "ring_count": False,
        "halogen_count": False,
        "similarity": False,
        "scaffold": False,
        "required_substructures": False,
    }


def _required_substructure_checks(
    mol: Chem.Mol, specification: Any
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Evaluate every configured required SMARTS without exposing RDKit objects."""

    if specification is None:
        return [], True, None
    if isinstance(specification, (str, bytes)) or not isinstance(specification, list):
        return [], False, "constraints.required_substructures must be a list."

    checks: list[dict[str, Any]] = []
    for index, item in enumerate(specification):
        if not isinstance(item, Mapping):
            return checks, False, (
                f"constraints.required_substructures[{index}] must be a mapping."
            )
        name = item.get("name")
        smarts = item.get("smarts")
        if not isinstance(name, str) or not name.strip():
            return checks, False, (
                f"constraints.required_substructures[{index}].name must be a non-empty string."
            )
        query, smarts_error = _parse_smarts(smarts)
        if query is None:
            return checks, False, (
                f"Required substructure {name!r} is invalid: {smarts_error}"
            )
        try:
            matches = bool(mol.HasSubstructMatch(query, useChirality=True))
        except Exception:
            return checks, False, f"RDKit could not check required substructure {name!r}."
        checks.append(
            {
                "name": name.strip(),
                "smarts": str(smarts).strip(),
                "matches": matches,
            }
        )
    return checks, all(check["matches"] for check in checks), None


def evaluate_candidate(smiles: Any, config: Any = None) -> dict[str, Any]:
    """Evaluate one candidate against every public hard constraint.

    The result is deterministic and safe to pass directly back to an agent.
    Constraint failure is represented by ``feasible=False`` while ``ok`` remains
    true when all requested RDKit calculations themselves completed.
    """

    loaded, config_error = _load_config(config)
    flags = _empty_constraint_flags()
    base: dict[str, Any] = {
        "ok": False,
        "error": config_error,
        "valid": False,
        "canonical_smiles": None,
        "qed": None,
        "properties": None,
        "similarity_to_start": None,
        "scaffold": False,
        "scaffold_match": False,
        "required_substructures": [],
        "required_substructure_retention": False,
        "constraint_flags": flags,
        "constraint_checks": dict(flags),
        "violations": list(flags),
        "feasible": False,
    }
    if config_error:
        return base

    mol, validation = _parse_smiles(smiles, loaded)
    if mol is None:
        base["error"] = validation.get("error")
        return base

    try:
        properties = _descriptor_values(mol)
        qed_value = _rounded(QED.qed(mol), SCORE_DECIMALS)
    except Exception:
        base.update(
            valid=True,
            canonical_smiles=validation["canonical_smiles"],
            error="RDKit could not calculate candidate properties.",
        )
        return base

    constraints = _constraints(loaded)
    calculation_error: str | None = None

    similarity_value: float | None = None
    similarity_raw: float | None = None
    similarity_cfg = constraints.get("similarity", {})
    if not isinstance(similarity_cfg, Mapping):
        similarity_cfg = {}
    start_smiles = None
    task = loaded.get("task", {})
    if isinstance(task, Mapping):
        start_smiles = task.get("start_smiles")
    if start_smiles:
        start_mol, start_validation = _parse_smiles(start_smiles, loaded)
        if start_mol is None:
            calculation_error = "Configured starting SMILES is invalid: " + str(start_validation["error"])
        else:
            try:
                radius = int(similarity_cfg.get("radius", 2))
                n_bits = int(similarity_cfg.get("n_bits", 2048))
                if not 0 <= radius <= 8 or not 16 <= n_bits <= 65_536:
                    raise ValueError
                similarity_raw = _raw_similarity_from_mols(
                    mol, start_mol, radius, n_bits
                )
                similarity_value = _rounded(similarity_raw, SCORE_DECIMALS)
            except (TypeError, ValueError, OverflowError, RuntimeError):
                calculation_error = "Configured similarity parameters are invalid."

    scaffold_match = True
    scaffold_cfg = constraints.get("scaffold")
    if scaffold_cfg is not None:
        scaffold_smarts = scaffold_cfg.get("smarts") if isinstance(scaffold_cfg, Mapping) else None
        query, smarts_error = _parse_smarts(scaffold_smarts)
        if query is None:
            scaffold_match = False
            calculation_error = str(smarts_error)
        else:
            try:
                scaffold_match = bool(mol.HasSubstructMatch(query, useChirality=True))
            except Exception:
                scaffold_match = False
                calculation_error = "RDKit could not perform the scaffold match."

    required_checks, required_retained, required_error = _required_substructure_checks(
        mol, constraints.get("required_substructures")
    )
    if required_error is not None:
        calculation_error = required_error

    require_single = bool(constraints.get("require_single_component", True))
    require_neutral = bool(constraints.get("require_neutral_formal_charge", False))
    flags = {
        "valid_smiles": True,
        "single_component": (validation["component_count"] == 1) if require_single else True,
        "allowed_atoms": not validation["disallowed_atomic_numbers"],
        "neutral_formal_charge": (properties["formal_charge"] == 0) if require_neutral else True,
        "molecular_weight": _within_range(properties["molecular_weight"], constraints.get("molecular_weight")),
        "logp": _within_range(properties["logp"], constraints.get("logp")),
        "tpsa": _within_range(properties["tpsa"], constraints.get("tpsa")),
        "hbond_donors": _within_range(properties["hbond_donors"], constraints.get("hbond_donors")),
        "hbond_acceptors": _within_range(properties["hbond_acceptors"], constraints.get("hbond_acceptors")),
        "rotatable_bonds": _within_range(properties["rotatable_bonds"], constraints.get("rotatable_bonds")),
        "ring_count": _within_range(properties["ring_count"], constraints.get("ring_count")),
        "halogen_count": _within_range(properties["halogen_count"], constraints.get("halogen_count")),
        "similarity": (
            similarity_raw is not None
            and similarity_raw >= float(similarity_cfg.get("min", 0.0))
        )
        if start_smiles
        else similarity_cfg == {},
        "scaffold": scaffold_match,
        "required_substructures": required_retained,
    }
    feasible = calculation_error is None and all(flags.values())
    violations = [name for name, passed in flags.items() if not passed]
    return {
        "ok": calculation_error is None,
        "error": calculation_error,
        "valid": True,
        "canonical_smiles": validation["canonical_smiles"],
        "qed": qed_value,
        "properties": properties,
        "similarity_to_start": similarity_value,
        "scaffold": scaffold_match,
        "scaffold_match": scaffold_match,
        "required_substructures": required_checks,
        "required_substructure_retention": required_retained,
        "constraint_flags": flags,
        "constraint_checks": dict(flags),
        "violations": violations,
        "feasible": feasible,
    }


TOOL_REGISTRY = {
    "validate_smiles": validate_smiles,
    "canonicalize_smiles": canonicalize_smiles,
    "calculate_qed": calculate_qed,
    "calculate_properties": calculate_properties,
    "calculate_similarity": calculate_similarity,
    "check_scaffold": check_scaffold,
    "check_substructure": check_substructure,
    "evaluate_candidate": evaluate_candidate,
}


__all__ = [
    "TOOL_REGISTRY",
    "validate_smiles",
    "canonicalize_smiles",
    "calculate_qed",
    "calculate_properties",
    "calculate_similarity",
    "check_scaffold",
    "check_substructure",
    "evaluate_candidate",
]

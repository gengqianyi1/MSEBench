"""Public RDKit tool surface for the molecular-design benchmark."""

from .rdkit_tools import (
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

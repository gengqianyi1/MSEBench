"""Independent evaluator package for the xTB recovery benchmark."""

from .evaluate import evaluate, evaluate_files, evaluation_bytes, write_evaluation
from .scoring import compute_score
from .trusted_validation import (
    ArtifactFacts,
    TraceFacts,
    TrustedConfiguration,
    TrustedValidationError,
    canonical_json_bytes,
    load_trusted_configuration,
    validate_artifacts,
    validate_trajectory,
)

__all__ = [
    "ArtifactFacts",
    "TraceFacts",
    "TrustedConfiguration",
    "TrustedValidationError",
    "canonical_json_bytes",
    "compute_score",
    "evaluate",
    "evaluate_files",
    "evaluation_bytes",
    "load_trusted_configuration",
    "validate_artifacts",
    "validate_trajectory",
    "write_evaluation",
]


"""Public API for the constrained-QED evaluator.

Imports are intentionally lazy so ``python evaluator/evaluate.py`` can use the
same implementation without module-cycle warnings.
"""

from __future__ import annotations

from typing import Any

from .scoring import ScoringConfigurationError, calculate_score, compute_score


def evaluate_submission(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .evaluate import evaluate_submission as implementation

    return implementation(*args, **kwargs)


def evaluate_submission_file(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .evaluate import evaluate_submission_file as implementation

    return implementation(*args, **kwargs)


def load_config(*args: Any, **kwargs: Any) -> Any:
    from .evaluate import load_config as implementation

    return implementation(*args, **kwargs)


def validate_submission_schema(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .evaluate import validate_submission_schema as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "ScoringConfigurationError",
    "calculate_score",
    "compute_score",
    "evaluate_submission",
    "evaluate_submission_file",
    "load_config",
    "validate_submission_schema",
]

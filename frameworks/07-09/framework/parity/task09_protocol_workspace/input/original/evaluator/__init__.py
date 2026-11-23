"""Independent evaluator package for the torsion-profile benchmark."""

from .scoring import (
    CATEGORY_MAX_POINTS,
    DEFAULT_THRESHOLDS,
    SCORING_SCHEMA_VERSION,
    STANDARD_GRID_DEG,
    ScoringInputError,
    circular_distance_deg,
    compute_score,
    directional_barriers_from_curve,
    linear_error_points,
    one_to_one_circular_matches,
    periodic_local_extrema,
)

__all__ = [
    "CATEGORY_MAX_POINTS",
    "DEFAULT_THRESHOLDS",
    "SCORING_SCHEMA_VERSION",
    "STANDARD_GRID_DEG",
    "ScoringInputError",
    "circular_distance_deg",
    "compute_score",
    "directional_barriers_from_curve",
    "linear_error_points",
    "one_to_one_circular_matches",
    "periodic_local_extrema",
]

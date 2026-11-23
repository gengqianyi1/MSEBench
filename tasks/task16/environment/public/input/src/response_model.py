"""RESPONSE MODEL -- PUBLIC STUB. The real module ships only in the grader package.

This package deliberately does NOT contain the response surface. In the grader package this module
holds the invented per-substance property table, the closed-form synthetic CQA simulator, and the
fitted surrogate that is the scoring ruler, loaded from data/oracle.npz. None of those is here, and
data/oracle.npz is not here either.

Why: the reference optimum is not the only thing worth hiding. An agent that can EVALUATE candidates
offline does not need to know the reference value to win -- it can optimise the response surface
directly and skip the five rounds of reasoning the benchmark is measuring. So the surface sits behind
the same boundary as the reference answer.

Consequence, and it is intended: this package can render the task, validate a formulation against the
declared design space and the change budget, run the tool layer, and record a trajectory -- but it
CANNOT produce a CQA value, a desirability, or a score. Evaluation is run by the party holding the
grader package:

    python scripts/grade_run.py runs/<run_dir>        # inside the grader package

Every name below exists so that importing the harness still works. Calling any of them raises.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

SEED = 20260727
DEFAULT_ORACLE_PATH = "<not shipped in the public package>"
ORACLE_SCHEMA_PATH = "<not shipped in the public package>"
ORACLE_ARRAY_KEYS = ("W0", "b0", "W1", "b1", "W2", "b2",
                     "x_mean", "x_scale", "y_mean", "y_scale")

_MSG = ("the response model is not shipped in the public package. The per-substance property table, "
        "the CQA simulator and the fitted surrogate (data/oracle.npz) live only in the grader "
        "package, so this package cannot measure a formulation, compute a desirability, or score a "
        "run. Use the grader package: python scripts/grade_run.py runs/<run_dir>")


class ResponseModelUnavailable(RuntimeError):
    """Raised by every callable in this stub."""


# The property tables are absent, not empty-by-accident: anything that indexes them fails loudly.
SUB: Dict[str, Dict[str, float]] = {}
MCC_GRADE: Dict[str, Dict[str, float]] = {}


def _props(d: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    raise ResponseModelUnavailable(_MSG)


def cqa_true(d: Dict[str, Any]) -> np.ndarray:
    raise ResponseModelUnavailable(_MSG)


def feature_vector(d: Dict[str, Any]) -> np.ndarray:
    raise ResponseModelUnavailable(_MSG)


def _quiet_matmul(fn):
    raise ResponseModelUnavailable(_MSG)


class Oracle:
    """Not available in the public package. Constructing one raises."""

    def __init__(self, path: str | None = None) -> None:
        raise ResponseModelUnavailable(_MSG)

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        raise ResponseModelUnavailable(_MSG)

    def predict(self, designs: Sequence[Dict[str, Any]]) -> np.ndarray:
        raise ResponseModelUnavailable(_MSG)

    def predict_one(self, design: Dict[str, Any]) -> np.ndarray:
        raise ResponseModelUnavailable(_MSG)


def get_oracle(path: str | None = None) -> Oracle:
    raise ResponseModelUnavailable(_MSG)


def random_designs(n: int, rng: "np.random.Generator") -> List[Dict[str, Any]]:
    raise ResponseModelUnavailable(_MSG)


def train_oracle(n: int = 60000, out_path: str | None = None):
    raise ResponseModelUnavailable(_MSG)

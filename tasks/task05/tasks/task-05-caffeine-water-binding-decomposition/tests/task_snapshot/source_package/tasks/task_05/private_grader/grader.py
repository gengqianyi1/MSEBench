"""Official task-local grader entrypoint.

This thin bridge delegates to the versioned scientific evaluator stored under
``private_data/evaluator_bundle``.  Keeping this standard path real prevents the
published grader fixture from diverging from the grader used by Harness runs.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=1)
def _implementation() -> ModuleType:
    # The bridge must also work when executed directly from private_grader/,
    # where the project root is otherwise absent from sys.path.
    project_root = Path(__file__).resolve().parents[3]
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    path = Path(__file__).resolve().parents[1] / "private_data" / "evaluator_bundle" / "grader.py"
    if not path.is_file():
        raise FileNotFoundError(f"scientific grader implementation is missing: {path}")
    spec = importlib.util.spec_from_file_location(f"chembench_{path.parents[2].name}_scientific_grader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import scientific grader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "grade", None)):
        raise AttributeError(f"scientific grader has no callable grade(context): {path}")
    return module


def grade(context):
    """Grade one immutable submission with the actual scientific evaluator."""
    return _implementation().grade(context)

"""GRADER-SIDE build of the response-model boundary.

This module is the single seam between the public half of the benchmark and the synthetic response
mechanism. Everything upstream of it -- the harness, the parser, the tool layer, the feedback
contract, the desirability objective, the score bands -- imports `responses` from HERE and never from
`agglomeration` directly. That is what makes the isolation checkable with one grep instead of an
argument.

Two builds of this file exist:

  * THIS one (grader): re-exports the real mechanism from src/agglomeration.py, so the grader package
    runs end to end and reproduces every calibrated number.
  * the public one (public_overlay/src/response_model.py): a stub that documents the evaluation
    contract and raises. The public package therefore contains the task, the contract, the framework
    and the tools -- and no way to read or reverse-engineer the response surface, because the
    mechanism, its coefficients and the surrogate weights are simply not in it.

scripts/package_for_delivery.py swaps the two when it builds the public archive, and asserts that no
mechanism symbol survives there.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agglomeration import responses as _responses

BUILD = "grader"
DESCRIPTION = ("synthetic response mechanism (30-bin agglomeration-breakage population balance, "
               "frozen array-bundle surrogate, solubility mass balance) -- grader side only")


def responses(x: Dict[str, Any], use_surrogate: bool = True, CFL: float = 0.30,
              maxstep: int = 240, surrogate_path: Optional[str] = None) -> Dict[str, float]:
    """Measure one batch on the synthetic simulator. See src/agglomeration.responses."""
    return _responses(x, use_surrogate=use_surrogate, CFL=CFL, maxstep=maxstep,
                      surrogate_path=surrogate_path)

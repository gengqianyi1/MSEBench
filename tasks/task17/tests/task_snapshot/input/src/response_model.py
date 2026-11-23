"""PUBLIC build of the response-model boundary. The synthetic response mechanism is NOT in this
package.

This file is the whole of the evaluation isolation, made concrete. In the grader package this module
re-exports the real simulator from `src/agglomeration.py`. In THIS package that module does not exist:
the population balance, the yield mass balance, every mechanistic coefficient and the surrogate weights
are held on the grading side, so an agent with full file access to the public package cannot read the
response surface, cannot invert it, and cannot compute its own score.

What IS in the public package, and is enough to understand and attempt the task:
  * tasks/q1_agglomeration/question.md          the task, the windows, the constraints, the scoring rule
  * tasks/q1_agglomeration/task_config.public.yaml   the machine-readable contract, WITHOUT the
                                                reference optimum, the score anchors or the gate
                                                measurements
  * src/design_grid.py                          the twelve level grids, the change-cost table, the
                                                baseline process, the four specification windows and
                                                the C1/C3-C7 predicates -- i.e. exactly what
                                                question.md prints
  * src/design_space.py                         the Derringer-Suich objective and the score bands
  * src/tools.py                                the diagnostic tool layer offered to the agent
  * src/{main,llm_client,parser,feedback,utils,markdown_report}.py   the harness
  * configs/, requirements.txt                  the runtime environment

What is deliberately ABSENT: src/agglomeration.py, data/ (surrogate weights, training set, model
card), answers/ (the reference enumeration and the expert baselines), the full task_config.yaml with
its `reference:` block, and the shipped model trajectories.

DATA NATURE. The responses this boundary stands in front of are SYNTHETIC. There is no registered
product, no filed process and no experimental measurement anywhere in this benchmark.

--------------------------------------------------------------------------------------------------
GRADER-SIDE EVALUATION CONTRACT
--------------------------------------------------------------------------------------------------
The grading side exposes exactly one operation. Request:

    {"design": {"S1_bridging_liquid": <str>, "S2_bridging_ratio_v_w": <float>,
                "S3_addition_mode": <str>,  "S4_solids_g_per_L": <int>,
                "S5_tip_speed_m_s": <float>, "S6_temperature_C": <int>,
                "S7_agglomeration_min": <int>, "S8_primary_d50_um": <int>,
                "S9_water_fraction": <float>, "S10_classification": <str>,
                "S11_drying": <str>, "S12_HPMC_ppm": <int>}}

All twelve keys required, each value one of the levels declared in src/design_grid.LEVELS.

Response, on success:

    {"d50": <float>,        # um
     "strength": <float>,   # MPa
     "porosity": <float>,   # dimensionless
     "yield_pct": <float>}  # %

plus the population-balance diagnostics listed in src/evaluator._DIAGNOSTIC_KEYS. Deterministic: the
same design always returns the same numbers. On an off-grid or incomplete design the grading side
raises `design_grid.DesignError` and no numbers are returned.

To run the benchmark end to end you need the grader package. This build refuses to invent numbers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

BUILD = "public"
DESCRIPTION = ("stub: the synthetic response mechanism is grader-side only and is not present in "
               "this package")

_MESSAGE = (
    "src/response_model.py is the PUBLIC build: the synthetic response mechanism is not part of this "
    "package.\n"
    "\n"
    "This is deliberate, not a packaging mistake. The response mechanism, its coefficients and the "
    "surrogate weights are held on the grading side so that an agent with file access cannot optimise "
    "against the response surface instead of solving the process problem, and so that the reference "
    "optimum and the score anchors are not readable.\n"
    "\n"
    "The public package therefore contains the task, the input/output contract, the runtime "
    "environment, the harness, the diagnostic tool layer and worked examples -- everything except the "
    "ability to measure a batch. Measurement and scoring happen on the grading side; see the "
    "GRADER-SIDE EVALUATION CONTRACT in this file's docstring for the request/response shape.\n"
    "\n"
    "To reproduce a graded run, use the grader package."
)


class ResponseModelUnavailable(RuntimeError):
    """Raised instead of returning fabricated numbers."""


def responses(x: Dict[str, Any], use_surrogate: bool = True, CFL: float = 0.30,
              maxstep: int = 240, surrogate_path: Optional[str] = None) -> Dict[str, float]:
    raise ResponseModelUnavailable(_MESSAGE)

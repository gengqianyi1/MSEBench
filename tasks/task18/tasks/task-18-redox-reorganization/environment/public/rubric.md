# Task 18 answer-free rubric

The private grader assigns **100 points** in five scientific sections:

| Section | Points | Public meaning |
|---|---:|---|
| Charge-state basins and validation | 30 | Evidence-backed neutral/cation minima, charge-state assignments, stationary-point checks, and geometry-derived basin support. |
| Numerical redox properties | 34 | Adiabatic and vertical oxidation proxies, inner-sphere reorganization components, and rankings recomputed from coherent xTB cycles. |
| Pareto and scientific decision | 16 | Recomputed two-objective Pareto membership, priority-specific recommendations, and scientifically supported interpretation. |
| Autonomous search and resources | 12 | Independent conformer exploration, discriminative validation, and efficient use of the fixed computation budget. |
| Evidence reliability and scope | 8 | Exact evidence-job binding, readable report, and correct limits of the gas-phase GFN2-xTB model. |

## Hard-zero conditions

The whole task receives a **hard-zero** only when either of these organizer-verifiable conditions holds:

1. there is no trusted Task 18 xTB job with the required method, charge/spin role, termination status, and organizer-owned evidence; or
2. the organizer evidence cannot be bound to this attempt (for example, the sealed submission/evidence lineage is absent or invalid).

A missing or schema-invalid `final_answer.json` is not itself a hard-zero when trusted organizer evidence exists. Declaration-dependent numerical, Pareto, recommendation, and evidence-binding sections become unavailable, and evidence-only credit is capped at 20 points. A missing `report.md` removes only report-dependent scope credit. Honest partial work therefore remains scoreable.

This document intentionally contains no private reference geometry, private reference energy, expected ranking, expected Pareto answer, or private numerical tolerances. Those remain in the organizer-only private suite.

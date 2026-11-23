# Candidate-library scoring specification

Version: `candidate-library-score-v3-hard`

The candidate module is worth 50 points. The library must contain exactly
1,000 records; count-based metrics retain 1,000 as their denominator. Invalid
or pathological records receive no dependent credit. Public constraints and
schemas define eligibility, while private calibration anchors prevent direct
optimization against the test set.

## Scored dimensions

| Dimension | Points |
|---|---:|
| Structural validity and continuous geometry/chemistry quality | 8 |
| Internal uniqueness under the frozen `StructureMatcher` | 3 |
| Composition-specific novelty | 3 |
| Anonymous-prototype novelty | 8 |
| Composition/prototype diversity | 3 |
| Reconstructive generation provenance | 5 |
| Five selected CHGNet-local-stability checks | 10 |
| Prediction-model traceability | 5 |
| Per-candidate uncertainty and applicability-domain evidence | 5 |

The structural-quality fraction is `0.5*valid_fraction +
0.25*mean_geometry_quality + 0.25*mean_chemistry_quality`. Uniqueness,
composition novelty, prototype novelty, traceability, and uncertainty fields
are each their eligible count divided by 1,000 and multiplied by the points in
the table. Diversity is the weighted sum published in
`data/candidate_generation_constraints.json`. Generation reproducibility uses
the reconstructive checks implemented in `src/candidate_scoring.py`; selected
CHGNet credit is 2 points per eligible converged candidate. The novelty
hierarchy is enforced: anonymous-prototype novelty is credited
only for candidates already composition-specific novel. Diversity is evaluated
over valid, internally unique, composition-novel representatives.

## Traceability and uncertainty

Each record must contain `prediction_model_sha256` matching the submitted
`model.joblib`, a finite positive `prediction_uncertainty_mev_atom`, and an
`applicability_domain_score` in `[0, 1]`. These are evaluated for completeness
and consistency; placeholder or non-finite values receive no credit.

## Selected CHGNet candidates

The submission identifies exactly five distinct IDs from its own low-prediction
pool. The controlled tool and private evaluator use the frozen CHGNet protocol:
CHGNet 0.4.2/model 0.3.0, fixed lattice, at most 200 optimization steps, and a
final maximum force no greater than 0.05 eV/Angstrom. Credit requires a valid
pre-relaxation structure, successful convergence, and a valid two-dimensional
post-relaxation structure. This is a local structural-stability proxy, not a
claim of DFT, convex-hull, phonon, finite-temperature, or experimental
stability.

## Caps and auditability

Low-coverage caps for validity, uniqueness, and novelty remain active, with the
strictest applicable cap used. Schema failures do not terminate the independent
property-prediction module. Public artifacts disclose required outputs,
eligibility rules, formulas, point dimensions, software protocol, and caps.
Only test labels and private audit samples remain private; no scoring rule or
threshold is withheld.

# Adaptive xTB Torsion Profile Agent Benchmark v3

Version 3 evaluates whether an agent can generalize an adaptive sampling policy
across four different frozen torsional energy surfaces while using at most 12
native points per case. The model chooses scan angles from compact feedback.
The tools perform geometry preparation, native-point retrieval/execution,
automatic periodic fitting, profile analysis, and finalization. This benchmark
does not claim that the model invents or implements the fitting algorithm, and
it is not a general multi-torsion conformer-search benchmark.

The public development/regression case remains 2-fluorobiphenyl and uses the
frozen per-case protocol 2.0 executor. The official v3 organizer suite adds
native 5° profiles for three organizer-hidden molecules spanning distinct
torsional morphologies. Their identities and numerical profiles are disclosed
only in the organizer package.
They span fourfold biphenyl extrema, alkane anti/gauche behavior, a fluorinated
gauche-effect profile, and a high amide C–N torsional barrier. Every hidden
profile contains 72 successful constrained GFN2-xTB points generated with xTB
6.7.1; their energy ranges are approximately 3.80, 4.95, 6.65, and 15.79
kcal/mol across the four cases.

## Public and organizer distributions

The participant archive contains the public case, schemas, tools, public suite
metadata, and a SHA-256 commitment only. It does not contain
`evaluator_private`, `organizer_private`, `organizer_configs`, hidden case
identities, oracle values, best sampling points, organizer trajectories, or
model-answer caches.

The organizer archive contains all four oracles, native provenance, private
evaluation materials, calibration results, and release validation. SHA-256 is
only an integrity commitment; it does not disclose or make oracle numeric
values reversible. Ranking results must be produced by the organizer evaluator.

## Status and score semantics

- `PASS`: a model produced an evaluable run and all eligibility gates passed.
- `FAIL`: a model responded, but scientific or protocol gates failed.
- `BLOCKED`: a provider, network, platform, or external dependency prevented an
  evaluable scientific trajectory.

`BLOCKED` carries null scientific, process, and official scores. It is excluded
from numeric means and makes full-suite coverage ranking-ineligible. Release
completeness is reported separately from model capability.

Each case reports scientific score (curve accuracy plus symmetry-aware extrema
and barriers, maximum 80), process score (adaptive sampling plus reproducibility,
maximum 20), eligibility gates, provider status, run status, official score,
and ranking inclusion. The existing periodic treatment, near-degenerate basin
handling, and adjacent barrier multiset matching are preserved.

## Reproduce and validate

```powershell
conda env create -f environment.yml
conda activate xtb-benchmark
python -m pytest -q
python scripts/finalize_multicase_suite_v3.py
python baselines/run_suite_baselines_v3.py --output-root <output-directory>
```

The tested environment pins Python 3.12.13, NumPy 2.5.1, RDKit 2025.03.5,
xTB 6.7.1, OpenAI 2.46.0, Pydantic 2.13.4, PyYAML 6.0.3,
python-dotenv 1.2.2, and pytest 9.1.1. Native commands use `shell=False`, one
thread, explicit timeouts, and isolated directories.

Truthful calibration applies the same feedback gates used for model runs. The
precommitted fixed twelve-angle policy and incomplete weak-six policy therefore
both score official **0**; fixed-12 has no feedback rounds and cannot receive
feedback/refinement credit. The organizer-only oracle-guided diagnostic scores
72.208436. A separate genuine feedback-driven live xTB baseline scores 23.886201.
The historical 83.714238 fixed-12 figure is an ungated/misattributed snapshot and
is not the current official adaptive baseline score.

Before the 2026-08-23 reviewer-response run, DeepSeek's 85.987921 aggregate was
diagnostic at 3/4 and Kimi's 26.0 aggregate was diagnostic at 1/4; both were
`ranking_eligible=false` and never official full-suite scores. The preregistered
missing-data rule retains completed cases, never imputes a missing case as zero,
requires full coverage for ranking, and forbids comparison between different
partial subsets.

The reviewer-response execution invoked exactly the missing one DeepSeek and
three Kimi cases with the frozen task/grader, budget, retry policy, and model
identities. DeepSeek now has 4/4 coverage and a replayed full-suite score of
**64.490941**. The initially blocked Kimi/butane case was completed incrementally:
two accepted model turns and nine eligible native points were reused by hash, the
live provider was called only for the unfinished third turn, and xTB was run only
for the three newly selected angles. That case scored 100.0, giving Kimi a
replayed 4/4 full-suite score of **31.5**. GLM remains **79.228673** at 4/4.
All three models now have common full coverage and are ranking eligible.

The preregistered missing-data rule remains unchanged and applies to every
preserved failed attempt: `BLOCKED` has null score and is never imputed as zero.
The selected completed authority is additive under
`official_runs/q09/reviewer_response_20260823/`; all historical partial outputs,
network failures, requests, retries, and token records remain unmodified.

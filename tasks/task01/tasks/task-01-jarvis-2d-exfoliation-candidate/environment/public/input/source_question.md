# Benchmark Task: JARVIS-DFT 2D Exfoliation-Energy Prediction and Reproducible Candidate-Library Generation

**Submission contract:** before submitting, read `input/submission_contract.md`.
It lists every artifact checked by the sidecar and the exact workspace-relative
paths. Validate all listed paths before calling `submit_candidate`.

## 1. Objective

Complete two independent but related tasks:

1. Build a reproducible structure-to-property model for predicting the exfoliation energy of two-dimensional materials.
2. Generate a reproducible library containing exactly 1,000 periodic two-dimensional material candidate structures.

Target field:

```text
exfoliation_energy_mev_atom  [meV/atom]
```

Lower exfoliation energy indicates a lower predicted thermodynamic cost of separating a monolayer from its layered bulk parent. The target does not establish synthesizability, thermodynamic stability, dynamical stability, environmental stability, or device suitability.

This is an execution benchmark. The required files must be produced through actual computation with the provided tool.

## 2. Public data

The workspace contains:

```text
data/train.json.gz
data/validation.json.gz
data/test_public.json.gz
data/dataset_metadata.json
data/candidate_generation_constraints.json
```

The three split files are gzip-compressed JSON Lines. Each record contains a `sample_id`, a pymatgen-compatible `structure`, and a `chemical_system`. Training and validation records also contain the target. Public test targets are absent.

`candidate_generation_constraints.json` defines the public hard limits and the fixed parameters used by candidate evaluation. Do not modify public data files.

Public test identifiers are opaque. They do not identify source records or reveal equivalent-representation groups.

## 3. Execution environment

Two executable tools are available:

```text
run_python(script, timeout_seconds)
relax_selected_candidates_chgnet(candidate_ids)
```

`relax_selected_candidates_chgnet` accepts exactly five distinct IDs already
present in `candidate_library.json.gz`. It can be attempted once without
penalty. Later attempts are recorded and rejected. Use the single opportunity
only after completing the library and selecting five low-predicted-exfoliation,
structurally plausible candidates.

Network access, package installation, external label lookup, external pretrained
checkpoints, and filesystem access outside the workspace are prohibited. The
frozen CHGNet checkpoint exposed only through the controlled selection tool is
an evaluator-provided exception; it may not be loaded through `run_python`.

The installed Python environment includes:

```text
numpy
pandas
scipy
scikit-learn
pymatgen
matminer
jarvis-tools
spglib
pyyaml
jsonschema
joblib
torch  # CPU only
chgnet
```

Package availability does not imply that any particular representation, feature family, estimator, tuning strategy, invariance procedure, or ensemble method is preferred. Selecting and implementing the workflow is part of the benchmark.

## 4. Property-prediction requirements

The submission must:

- inspect and audit all public splits;
- parse readable structures as `pymatgen.Structure` objects and record failures;
- train a constant training-only baseline and at least one learned model derived from the supplied structures;
- avoid target leakage and prohibited auxiliary-label lookup;
- report validation MAE, RMSE, Spearman correlation, baseline metrics, model-selection policy, fitted feature count, random seed, and refit policy;
- produce one finite prediction for every public test `sample_id`;
- perform and report a validation-side representation-invariance diagnostic selected by the solver;
- remain robust to hidden physically equivalent structure representations;
- save the fitted model, feature names, training log, predictions, and a `result.json` conforming to `schemas/result_schema.json`.

Private evaluation measures overall regression, chemical-system out-of-distribution performance, low-energy ranking, target-tail robustness, and representation invariance.

The 50 property-prediction points are distributed as follows. Private labels and
frozen calibration anchors are not disclosed.

| Property-prediction component | Direction | Points |
|---|---|---:|
| Overall MAE | Lower is better | 14 |
| Overall RMSE | Lower is better | 4 |
| Overall Spearman correlation | Higher is better | 4 |
| Chemical-system OOD MAE | Lower is better | 8 |
| Chemical-system OOD Spearman correlation | Higher is better | 4 |
| Lowest-energy 10% hit rate | Higher is better | 4 |
| Lowest-energy 10% NDCG | Higher is better | 4 |
| Low-target-tail MAE | Lower is better | 2 |
| High-target-tail MAE | Lower is better | 2 |
| Representation-invariance joint MAE | Lower is better | 4 |

Metric definitions:

- **Overall MAE/RMSE/Spearman** are computed over private rows marked as base
  regression samples. Spearman is the rank correlation between measured and
  predicted exfoliation energies.
- **Chemical-system OOD MAE/Spearman** use the base regression subset whose
  chemical systems are held out from the training split.
- **Lowest-energy 10% hit rate** is the fraction of the measured lowest-energy
  decile recovered in the predicted lowest-energy decile.
- **Lowest-energy 10% NDCG** evaluates the ordering of low-energy materials,
  with larger relevance assigned to measured values further below the private
  10th-percentile cutoff.
- **Low-target-tail MAE** and **high-target-tail MAE** are MAE values on the
  measured bottom and top target deciles, respectively.
- **Representation-invariance joint MAE** is averaged over hidden groups of
  physically equivalent structure representations. Each group combines the
  mean pairwise spread of its predictions with the absolute error of the group
  mean prediction. A constant predictor therefore does not receive perfect
  invariance credit merely for producing identical outputs.

The evaluator-only label file supplies opaque `sample_id` alignment, the measured
target, private split membership, base-regression membership, and equivalence-group
membership. It is not mounted in the agent workspace. Exact labels, group identities,
and frozen score-calibration anchors remain private.

## 5. Candidate-library requirements

Generate exactly 1,000 periodic two-dimensional material candidate records in `candidate_library.json.gz`.

Every candidate must additionally report a positive predictive uncertainty, an applicability-domain score in `[0,1]`, and the SHA-256 of the submitted `model.joblib` used for its prediction. The evaluator checks these fields against the submitted model artifact and uses a frozen private sample to audit prediction traceability. Self-declared low energies without a matching model hash receive no traceability credit.

Each record must conform to `schemas/candidate_record_schema.json` and contain:

- a stable candidate identifier;
- a pymatgen-compatible periodic structure;
- provenance sufficient to reconstruct the candidate;
- the random seed associated with generation;
- a predicted exfoliation energy in meV/atom from the submitted property model;
- parent identifiers when a public structure was used;
- an ordered list of generation operations and their parameters.

The 1,000 candidate predictions must define a non-degenerate ranking. Constant
or near-constant placeholders do not form a valid low-prediction pool.

The evaluator, not the submitting model, determines validity, uniqueness, novelty, geometry, chemistry, diversity, and reproducibility.

### 5.1 Public hard limits

The candidate constraints file defines the authoritative values. The initial benchmark profile uses:

- exactly 1,000 submitted records;
- 2 to 40 sites per submitted cell;
- no more than 4 distinct elements;
- ordered sites only;
- finite lattice and coordinate values;
- elements restricted to the public allowed-element list;
- no network-dependent or manually unreproducible generation step.
- no duplicate/overlapping atoms, degenerate cell, inadequate two-dimensional
  vacuum, or implausible periodic coordination as defined in the public
  constraints file.

Records that fail candidate-level checks remain in the denominator and receive no credit for dependent metrics.

### 5.2 What does not count as a distinct material candidate

The following changes alone do not create a unique candidate:

- changing site order;
- translating the periodic origin;
- wrapping fractional coordinates;
- changing equivalent lattice bases;
- converting between a primitive cell and an integer supercell;
- changing only vacuum thickness;
- applying negligible numerical perturbations.

### 5.3 Candidate-library evaluation

The candidate-library score is based on:

1. combined structural validity, pathology, geometry, and chemistry;
2. internal structural uniqueness;
3. material-structure and higher-weight anonymous-prototype novelty relative to the training set;
4. compositional and structural diversity;
5. generation reproducibility;
6. CHGNet-predicted local structural stability of five selected candidates.

After generating the library, select five structurally plausible candidates
from the lowest 10% of submitted predicted exfoliation energies. Candidate
ordering is not scored. Use the single CHGNet tool opportunity to relax those
five structures and report the same five IDs in
`selected_generated_candidates`. A selected candidate receives stability
credit only when the evaluator independently reproduces a converged relaxation,
the final maximum force is at most 0.05 eV/Angstrom, and the relaxed structure
remains non-pathological and two-dimensional. This is a local structural
stability proxy, not proof of thermodynamic stability or synthesizability.

## 6. Required generator artifacts

Submit:

```text
generator/generate.py
generator/config.json
generator/generation_manifest.json
candidate_library.json.gz
candidate_summary.json
```

The private evaluator reruns the generator using the fixed interface:

```text
python generator/generate.py \
  --input data/train.json.gz \
  --constraints data/candidate_generation_constraints.json \
  --config generator/config.json \
  --output <evaluator-selected-output-path>
```

The generator must not rely on files not declared in the manifest. The regenerated library is compared with the submitted library after deterministic JSON normalization.

The manifest must conform to `schemas/generation_manifest_schema.json` and record:

- the random seed;
- hashes of declared input files;
- relevant package versions;
- generation parameters;
- the expected output count;
- the fixed generator interface version.

## 7. Required submission files

```text
result.json
predictions.csv
model.joblib
feature_names.json
training_log.json
candidate_library.json.gz
candidate_summary.json
generator/generate.py
generator/config.json
generator/generation_manifest.json
selected_chgnet/chgnet_relaxation.json
selected_chgnet/CAND-xxxxxx_relaxed.json  # five files when all relaxations execute
```

`predictions.csv` must contain exactly:

```text
sample_id,predicted_exfoliation_energy_mev_atom
```

`candidate_library.json.gz` should be gzip-compressed JSON Lines with 1,000 records. A parseable short library remains scoreable, but every missing record receives zero under the fixed 1,000-record denominator; submitting fewer records never improves a fraction or average.

Before writing final submission artifacts, inspect all four schemas under
`schemas/`. Validate `result.json`, every candidate record,
`candidate_summary.json`, and `generator/generation_manifest.json` with
`jsonschema`. Do not invent, rename, or omit schema fields.

## 8. Scoring

Total score: 100 points.

- Property-prediction performance: 50 points.
- Candidate-library quality: 50 points.

The property-prediction and candidate-library modules are scored independently.
Missing, incomplete, non-finite, or unalignable predictions receive zero for the
property module without automatically erasing an otherwise scoreable candidate
module. An unreadable candidate library receives zero for the candidate module
without automatically erasing an otherwise scoreable property module. Candidate
record failures and an incomplete library remain in the fixed 1,000-record
denominator.

Public evidence and artifact errors incur up to 6 points of deductions from the
combined score:

| Compliance area | Maximum deduction |
|---|---:|
| Result schema and cross-file consistency | 2 |
| Public-data audit accuracy | 1 |
| Validation evidence | 1 |
| Model and feature reproducibility artifacts | 1 |
| Scientific interpretation and uncertainty | 1 |

Candidate-generator completeness and reproducibility are scored only within the
50-point candidate module and are not deducted a second time as public compliance
errors.

Candidate-record Schema errors do not terminate the independent property
module, but they cap the candidate-library module at 35/50.

Confirmed target leakage, hidden-label access, prohibited network use, public-data
tampering, or filesystem-policy violation invalidates the complete run and results
in zero points.

Candidate-library assessment covers structural quality, uniqueness, composition and prototype novelty, diversity, deterministic generation, submitted-model traceability, uncertainty/applicability evidence, and CHGNet local stability. Exact calibration constants, private audit samples, and continuous thresholds are frozen in the evaluator bundle and are not exposed to the candidate model.

Detailed deterministic rules are documented in `CANDIDATE_SCORING_SPEC.md`. `candidate_summary.json` must conform to `schemas/candidate_summary_schema.json`; every reported value is treated as a claim and independently recomputed by the evaluator.

## 9. Execution resilience

The Harbor evaluation applies a strict LLM-request budget cap of 40 requests for this task.
Runs with at most 40 recorded LLM requests keep their full scientific score; a run whose
recorded request count exceeds 40 is scored 0 for breaching the request budget cap. Complete the full workflow and submit only
after every required artifact validates. A six-hour
wall-time safety limit and the published per-script timeout still apply; these are
runtime safeguards, not scoring penalties. Write expensive and independently
scoreable artifacts incrementally rather than postponing all output until the end.

Write expensive and independently scoreable artifacts incrementally. In particular,
write the generator source/configuration and any generated candidate records before
enforcing exact-count assertions. A short or partially invalid candidate library
must still be submitted: invalid or missing records receive no credit under the
fixed 1,000-record denominator. Do not deliberately terminate the script before
preserving partial, parseable outputs and failure metadata.

## 10. Scientific and integrity constraints

- Use only mounted public inputs.
- Do not search for hidden labels or recover source IDs.
- Do not treat representation variants as additional labeled materials.
- Do not silently discard failed test structures; use a deterministic fallback and log it.
- Do not submit copied training structures as novel candidates.
- Do not claim DFT stability, synthesizability, or experimental feasibility from rule-based candidate checks.
- Record failures and uncertainty honestly.

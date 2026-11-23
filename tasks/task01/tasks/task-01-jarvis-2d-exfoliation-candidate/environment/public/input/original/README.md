# MSEBench JARVIS 2D Exfoliation + Candidate Generation

Benchmark version: **1.3**

This package evaluates two capabilities in one bounded tool-calling run:

1. structure-based exfoliation-energy prediction on a frozen JARVIS-DFT 2D split;
2. deterministic generation of exactly 1,000 periodic 2D material candidate structures.

The property model receives 50 points. The candidate library receives 50 points.
Generated candidates carry model-predicted exfoliation energies so the agent can
select a low-prediction pool; five selected structures are independently
re-relaxed with the frozen CHGNet protocol.

## Candidate score

- combined structural validity/pathology/geometry/chemistry: 8;
- internal uniqueness: 3;
- composition-specific structural novelty: 3;
- anonymous-prototype novelty: 8;
- composition and structure diversity: 3;
- generator reproducibility: 5;
- five selected CHGNet-stable candidates: 10;
- prediction-model traceability: 5;
- uncertainty and applicability-domain completeness: 5.

See `CANDIDATE_SCORING_SPEC.md` for deterministic rules and caveats.

## End-to-end evaluation

1. Create or activate the environment shown below.
2. Put the runtime credential in the API config referenced by the selected
   model config. Do not put credentials in example files or commit them.
3. Run one model:

```bash
python src/main.py --config configs/deepseek.yaml
# or: python src/main.py --config configs/qwen.yaml
# or: python src/main.py --config configs/kimi.yaml
```

The runner copies only the public data files into the agent workspace, including `candidate_generation_constraints.json`.
The default protocol provides 20 agent tool calls plus at most two terminal
delivery calls. LLM credentials are read from the API configuration referenced
by the selected run config; example files contain placeholders only.

```bash
conda env create -f environment-linux.yml
conda activate jarvis_2d_exfoliation_benchmark
docker build -t msebench-jarvis-candidate-v0.5.0:latest .
```

The run creates `runs/<run_id>/final_report.json`, an agent workspace, and the
complete trajectory. A run is fully delivered only when
`runs/<run_id>/workspace/result.json` exists. Perform the public validation
below, then run private scoring to obtain the final score out of 100.

## Public validation

```bash
python scripts/validate_result.py runs/<run_id>/workspace/result.json
```

Public validation verifies prediction coverage, strict schemas, required artifacts,
the candidate file, and generator metadata. Public evidence errors incur up to
6 points of deductions. The 50-point property and candidate modules are scored
independently; a failure in one module does not automatically erase the other.
Confirmed leakage or another integrity violation invalidates the complete run.

## Private scoring

```bash
python scripts/score_submission.py \
  --workspace runs/<run_id>/workspace \
  --labels private/test_labels_private.json.gz \
  --metadata data/dataset_metadata.json \
  --workflow-report runs/<run_id>/final_report.json \
  --output runs/<run_id>/private_score.json
```

The private scorer combines:

- 50 property-prediction points using frozen measured anchors;
- 50 deterministic candidate-library points.

The reproducibility check executes untrusted submitted generator code. Production use must run the scorer in an isolated, network-disabled container with resource limits.
The complete local evaluator bundle contains private labels under `private/`.
They are evaluator-only, excluded from the public manifest and Docker context,
and are never copied into the public agent workspace. Their required format is
documented in `private/README.md`.

The final score is the `score` field in:

```text
runs/<run_id>/private_score.json
```

`final_report.json` describes workflow completion and public compliance;
`private_score.json` is the authoritative complete benchmark score.

## Required agent artifacts

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
```

## Calibration warning

The candidate thresholds in `data/candidate_generation_constraints.json` are explicit and executable, but they should be calibrated using multiple dry runs before a public leaderboard is frozen. In particular, StructureMatcher tolerances, geometry thresholds, diversity saturation targets, and chemistry heuristics can shift score distributions.

## Release validation and citation

```bash
python scripts/build_manifest.py
python scripts/validate_package.py
pytest -q
```

See `REPRODUCIBILITY.md`, `SECURITY_AND_PRIVATE_DATA.md`, `DATASET_CARD.md`,
`CITATION.cff`, and `THIRD_PARTY_LICENSES.md`.

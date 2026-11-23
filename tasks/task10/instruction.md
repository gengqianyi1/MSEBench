# MSEBench Task 10: Experimental band-gap prediction

Build a reproducible composition-based regression pipeline using the labelled rows in `input/data/train.csv`. Predict every row in both `input/data/validation_features.csv` and `input/data/test_features.csv`.

The fixed split comes from the submitted benchmark revision. Validation and test labels are private and are never copied to your workspace. Network access is disabled. Do not attempt to infer or access private files.

Write these files at the workspace root:

1. `predictions.csv` with exactly two columns, `id,prediction_eV`. Include each validation and test ID exactly once. Row order is irrelevant.
2. `pipeline.json` describing the feature representation, estimator, hyperparameters, random seed, and training-data files actually used. Do not report MAE or R2; the grader independently computes all metrics.
3. `final_answer.json` with a concise summary and paths to the two artifacts. <!-- 2026-09-27 fix: clarified artifact roles; see audit note below. -->

Predictions must be finite decimal numbers. Duplicate, missing, or unknown IDs invalidate the prediction protocol. The grader computes validation MAE, test MAE, validation R2, and zero-gap/positive-gap diagnostic MAE using private labels. The three scored metrics use the public piecewise-linear rules in `rubric.json`; the former pass/fail thresholds remain the zero-point anchors, while stronger metrics receive proportionally more credit.

---

## Harbor execution contract

This MSEBench task uses the upstream Matbench dataset `matbench_expt_gap`.

The evaluated model is supplied by the Harbor Agent layer; this task must not call the evaluated LLM directly. Work only inside `/workspace`. Required legacy artifacts: `final_answer.json`, `predictions.csv`, `pipeline.json`.

<!-- 2026-09-27 fix: disclosure added after audit found the instruction neither documented the runtime budget nor separated harness-level artifact requiredness from scientific scoring. Role of final_answer.json: it is a harness-required protocol artifact (see required_artifacts in .chembench_agent.json); the scientific grader never reads it — scoring uses only predictions.csv and pipeline.json. -->

Role of `final_answer.json`: it is a harness-required protocol artifact (declared under `required_artifacts` in `.chembench_agent.json`); the scientific grader does not read or score it. Scientific scoring uses only `predictions.csv` and `pipeline.json`, with all metrics recomputed from private labels.

Budget limits: at most 6 agent rounds and 20 monitored tool calls (see `.chembench_agent.json` / `harness_contract.yaml`; 600 s per tool call, no automatic retries). The harness saves `final_answer.json` only when the model returns `final_answer`. A run that exhausts its round/tool budget without writing valid scored artifacts (`predictions.csv`, `pipeline.json`) has nothing scorable and receives 0 points.

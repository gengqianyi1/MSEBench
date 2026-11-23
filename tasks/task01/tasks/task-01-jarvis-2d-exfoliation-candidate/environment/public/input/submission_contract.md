# Task 01 submission contract

All paths below are relative to the task workspace root (the tool process current
working directory; do not use a literal `/workspace` absolute path). A text
claim that a file was written does not count; each path must exist before
calling `submit_candidate`.

Required paths:

```text
final_answer.json
result.json
native_final_report.json
data/
predictions.csv
candidate_library.json.gz
model.joblib
feature_names.json
training_log.json
generator/
```

Use the schemas and examples under `input/original/` for JSON structure. The
canonical prediction header is:

```text
sample_id,predicted_exfoliation_energy_mev_atom
```

`final_answer.json` should contain the final completion status and rationale;
`result.json` must reference the submitted prediction/library artifacts;
`native_final_report.json` must summarize the validated result; and
`generator/` must contain the executable generator plus its configuration and
manifest. Run `python input/submission_check.py` from the workspace root; it must
print `"ok": true` before submission. Stop if any required path is missing or
invalid.

Before fitting or predicting, perform an explicit finite-value preflight on
the feature matrices and predictions (for example, check
`np.isfinite(X).all()` and `np.isfinite(predictions).all()`). If a descriptor
contains NaN or infinity, repair it deterministically within the workspace,
record the repair and its effect in `training_log.json`, then rerun the fit
and prediction. Do not submit an `incomplete` status merely because the first
fit attempt failed: produce all required artifacts and rerun the checker
after the repair.

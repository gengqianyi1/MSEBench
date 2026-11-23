# Task 02 canonical submission contract

All paths are relative to the task workspace root (the tool process current
working directory; do not use a literal `/workspace` absolute path). The sidecar checks every path below;
writing only `result.json` or describing results in `rationale` is insufficient.

Required paths:

```text
final_answer.json
result.json
native_final_report.json
stage_A_audit.json
plan.json
stage_B_bulk.json
stage_C_local.json
revisions.json
structures/
logs/
seed_evidence/
```

`structures/` and `logs/` must contain the coordinate files and raw formal/
ALPB/complex/SAPT logs referenced by the stage JSON files. `seed_evidence/`
must be retained so anomaly findings remain auditable. `stage_A_audit.json`, `stage_B_bulk.json`, `stage_C_local.json`, and
`result.json` must validate against the schemas in
`input/original/tasks/oligothiophene_smd_sapt_v3/`. The wrapper files must
validate against `input/schemas/final_answer.schema.json`,
`input/schemas/native_final_report.schema.json`, and
`input/schemas/revisions.schema.json`. Minimal valid examples are under
`input/examples/`; when no correction was necessary, use `[]` for
`revisions.json`. Run `python input/submission_check.py` from the
workspace root; it must print `"ok": true`, then run the public stage/result
validators before calling `submit_candidate`.

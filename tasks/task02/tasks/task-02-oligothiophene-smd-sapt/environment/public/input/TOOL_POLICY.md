# Task 02 tool policy

The only sidecar action is `run_python`. Pass Python source in `script` and a
`timeout_seconds` value from 1 to 7200. Scripts run from the task workspace and
must use relative paths.

Network libraries, destructive filesystem/process operations, `exec`, `eval`,
`compile`, all subprocess calls, and direct imports of `pyscf`, `psi4`, or `xtb`
are forbidden. Scientific programs must be invoked only through these functions
from `benchmark_tools`:

```text
run_xtb_microtest
run_xtb_optimization
run_xtb_single_point
run_pyscf_microtest
run_pyscf_formal_single_point
run_psi4_microtest
run_psi4_sapt0
```

Run each program's micro-test before its first production call. Do not modify
`instance.json`, `protocol.yaml`, `benchmark_tools.py`, the stage/result schemas,
or `stage_validation.py`. Write stage artifacts as soon as each stage completes.

The task uses a strict 120-LLM-request budget cap; a run whose recorded
request count is at most 120 keeps its full scientific score, while any run
exceeding 120 recorded requests is scored 0 for breaching the cap. A six-hour
wall-time safety cap and per-script timeout remain. Submit only after
`input/submission_check.py` and the public validators pass; no runner fallback
creates missing artifacts.


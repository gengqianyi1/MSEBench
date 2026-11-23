# Oligothiophene SMD/SAPT Evidence-Audited Benchmark

This directory contains participant-facing protocol and supporting code for MSEBench Task 02. Use the [Task 02 instruction](../../../../instruction.md) for the current requirements and the [tool policy](../TOOL_POLICY.md) for the supported execution interface.

## Participant workspace

- The model receives `question.md`, a whitelist-filtered public task configuration, and the generated `instance.json` in its initial context. Hidden anomaly-generation settings and private scoring-reference fields are never included.
- The workspace also contains `protocol.yaml`, `benchmark_tools.py`, `stage_A_schema.json`, `stage_B_schema.json`, `stage_C_schema.json`, `stage_validation.py`, and `result_schema.json`.
- Tool-call count has no hard limit and does not change the score; a six-hour wall-time safety limit remains.
- The model must independently complete the workflow and produce every required stage file. There is no forced finalization and no degraded delivery path.
- Rejected scripts are recorded. Candidate subprocesses and project-root file access are blocked; candidate code is limited to the workspace and the public `benchmark_tools.py` interface.
- Candidate code may use RDKit directly, but xTB, PySCF, and Psi4 must be called through the copied workspace module `benchmark_tools.py`.
- `stage_B_bulk.json` must pass `validate_stage_B`; existence alone is insufficient.
- The runner does not stop when only `result.json` exists. Stage D is complete only when both `result.json` and `revisions.json` parse and the result schema validates.
- Missing `result.json` is still represented to the evaluator as `{}`; that behavior is unchanged.
- Tool-specific timeout floors prevent a large chemistry batch from being terminated by an unrealistically small model-requested timeout.

## Fixed chemistry protocol

The authoritative execution settings are in:

```text
tasks/oligothiophene_smd_sapt_v3/protocol.yaml
```

At runtime they are copied to `workspace/protocol.yaml`. The standardized public interface is copied to `workspace/benchmark_tools.py` and provides:

```python
run_xtb_microtest
run_xtb_optimization
run_xtb_single_point
run_pyscf_microtest
run_pyscf_formal_single_point
run_psi4_microtest
run_psi4_sapt0
```

Direct candidate imports of xTB/PySCF/Psi4, direct executable access, and all candidate-created subprocesses are rejected. The public wrappers perform controlled chemistry launches.

## Stage B gate

Before Stage C, the runner validates:

- the canonical Stage B JSON schema;
- six conformer attempts and at least one retained conformer;
- `structures/R_g.xyz` and its recomputed coordinate hash;
- `analysis/formal_coordinate_hashes.json` for all five common-reference systems;
- gas, n-hexane, 1,4-dioxane, chloroform, and water energies;
- existence, convergence, metadata, geometry identity, and numerical consistency of all five raw logs;
- one distinct parseable sensitivity log using a permitted sensitivity type;
- numerical and robust-ranking fields.

The candidate can run the same check inside a tool call:

```python
from stage_validation import validate_stage_B
errors = validate_stage_B("stage_B_bulk.json", ".")
assert not errors, errors
```

## Environment and scoring

See [Reproducibility](REPRODUCIBILITY.md) for the packaged participant environment.
The MSEBench organizer operates the official scoring environment. The
[submission contract](../submission_contract.md) describes the public submission
checks. The task instruction governs the scoring rules and request budget.

See [CITATION.cff](CITATION.cff) and [third-party notices](THIRD_PARTY_LICENSES.md)
for attribution.

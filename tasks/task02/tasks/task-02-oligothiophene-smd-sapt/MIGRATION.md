# Task02 Harbor migration

- Source: `task_02_oligothiophene_smd_sapt`, release 1.3, `native_runner`.
- Current status: **harbor-ready**.
- Required A/B/C/D artifacts and the legacy `1.3-postrun-private-replay`
  deterministic score are preserved.

## Protocol equivalence

The organizer-side `chembench-hidden-suite/1.0` adapter implements the frozen
A/B/C/D staged campaign. The model receives at most 60 `run_python` actions:
calls 1-30 are unpenalized, 31-40 add one cumulative point each, and 41-50 add
two cumulative points each. The Agent may take up to 60 turns, but a 51st tool
call is rejected. Required stage files remain the model's responsibility.

## Verified evidence

- Harbor Oracle score: `65.412`.
- Same-model Harbor and legacy replay score: `55.0`; absolute difference `0`.
- Evidence: `parity/task02_offline_oracle.json` and
  `parity/task02_same_model_parity.json`.
- Missing, corrupt, and schema-poor submissions produce deterministic scored
  outcomes without verifier errors.
- `scoring_reference.json` exists only in the verifier snapshot; participant
  release-boundary audit and package validation pass.

Parity tolerance is an absolute score difference of `1e-9`. The canonical Agent
runtime contains no `native_runner.py`; private replay data and grader code are
confined to the separately built verifier image.

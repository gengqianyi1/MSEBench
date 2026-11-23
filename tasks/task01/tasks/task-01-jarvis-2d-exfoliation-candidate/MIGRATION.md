# Task01 Harbor migration

- Source: `task_01_jarvis_2d_exfoliation_candidate`, release 1.3, `native_runner`.
- Current status: **harbor-ready**.
- Required artifacts and the legacy `1.3-required-artifacts-strict` scoring semantics are preserved.

## Protocol equivalence

The organizer-side `chembench-hidden-suite/1.0` adapter exposes only
`run_python` and the one-shot `relax_selected_candidates_chgnet` action. Both
count toward 20 primary actions. At most two additional `run_python` actions
are reserved for schema/output repair; repair actions cannot retrain, regenerate
the candidate library, or invoke CHGNet. The Agent configuration permits at
most 122 tool calls: 20 primary compute actions, two output-only repairs, and
up to 100 read-only file inspections. The sidecar independently enforces those
separate limits.

## Verified evidence

- Harbor Oracle and legacy score: `45.802024`; absolute parity difference `0`.
- Same-model evidence: `parity/task01_same_model_parity.json`.
- Invalid submissions: integrity violation, missing `result.json`, and missing
  workflow report all return structured scored results rather than verifier errors.
- Private measured labels and scoring anchors exist only in the verifier snapshot.
- Participant release-boundary audit and package validation pass.

Parity tolerance is an absolute score difference of `1e-9`. The canonical Agent
runtime contains no `native_runner.py`; private data and grader code are confined
to the separately built verifier image.

> 2026-09-27 update: the primary action budget was unified to 40 across the
> instruction, TOOL_POLICY, adapter and agent prompt (the earlier 20/40 split).
> The agent turn ceiling was raised to 50 to keep it above the action budget.

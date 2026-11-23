# Task 18 Harbor migration

Status: `v4.1-harbor-validated`. Task 18 uses the same organizer-side workspace
sidecar and private replay boundary as Tasks 04–06. A real Harbor 0.21.0 Docker
lifecycle completed on 2026-08-30 with one completed trial, zero errors, reward
1.000, Golden score 100 and consistent canonical replay.

## Public protocol

- Three frozen anonymous neutral donor XYZ inputs are mounted with one
  goal-oriented question, one result Schema and one empty template.
- The Agent receives exactly `list_files`, `read_file`, `write_file`,
  `run_python`, `xtb_run`, `budget_status` and `submit`.
- The budget is 56 compute credits, at most 32 admitted xTB jobs, 90 scientific
  turns, 8 finalization turns, 360 tool calls and 10,800 seconds.
- The public Harness does not generate conformers, identify rotatable bonds,
  select charge-state basins, assign Hessians, reconstruct energy cycles, rank
  candidates or make Pareto decisions.
- Missing or invalid deliverables do not erase factual execution already proved
  by organizer evidence. They make all submission-dependent result and decision
  sections unavailable; evidence-only credit is capped at 20.

## Private replay

`private_suite_adapter.py` snapshots authoritative jobs outside the model
workspace. `protocol_grader.py` resolves only the organizer evidence ID, passes
trusted Agent metrics to the task-local grader, performs two canonical replays
and emits Harbor reward/result files. The 4.1 grader reconstructs molecular
identity, charge/UHF, optimized-geometry lineage, Hessian validity, coherent
four-point cycles, numerical properties, rankings and Pareto decisions without
launching calculations or repairing model output.

## Accepted verification

- Public/private drift: clean; public tools are exactly seven generic primitives.
- Oracle: three independent real-xTB repetitions, 3/3 score 100 and deterministic replay.
- Baseline: score 100 and deterministic replay.
- Three formal model submissions regraded through the Harbor private protocol
  with exact zero score delta and identical section scores: DeepSeek V4 Pro,
  Kimi K2.6 and GLM 5.2.
- DeepSeek's grader 4.1.0 to 4.2.0 replay is bound to the unchanged accepted
  submission and organizer evidence in
  `validation/model_runs/deepseek-v4-pro/regrade_lineage.json`; it records the
  original and regraded scores, file/content hashes, replay time, and zero new
  model API or xTB calls.
- Real Harbor lifecycle: `validation/harbor_lifecycle/task18/deployment_record.json`.
- Runtime image: `chembench/task-18-redox-reorganization:4.1.0`, pinned by digest.
- Private verifier image: `chembench/task-18-redox-reorganization-verifier:4.2.0`,
  pinned by digest.
- Harbor lifecycle verification launched zero model API calls and zero new
  scientific xTB jobs; it replayed the stored organizer-owned Oracle evidence.

Task 18 therefore satisfies the author-side Harbor migration and release gates.

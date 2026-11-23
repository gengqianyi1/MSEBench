# Tasks 10-13 Harbor author handoff (2026-09-06)

This private author package aligns Tasks 10-13 with
`MatBench_Harbor_Combined_framework_tasks_no_textqa_20260905.zip`. It is not a
participant bundle: `tests/task_snapshot/private_data`, private graders, Oracles,
and calibration fixtures must stay in the separate verifier image and must never
be copied or mounted into the model workspace.

## Author decisions

| Requirement | 10 | 11 | 12 | 13 |
|---|---|---|---|---|
| Public tools/state/budgets/retries/termination complete | confirmed | confirmed | confirmed | confirmed |
| Private adapter and independent replay grader complete | confirmed | confirmed | confirmed | confirmed |
| Oracle, three invalid cases, thresholds and calibration documented | confirmed | confirmed | confirmed | confirmed |
| Offline legacy-grader/evidence-replay parity | exact, delta 0 | exact, delta 0 | exact, delta 0 | exact, delta 0 |
| Live same-model behavioral parity | pending | pending | pending | pending |
| Release eligible | no | no | no | no |

These are single-step tasks. The common Harbor Agent manages the evaluated model
and writes the declared artifacts to `/workspace`; it never calls a task-owned
`native_runner`. After termination, the verifier-side
`private_suite_adapter.py` snapshots only allowlisted artifacts into immutable
organizer evidence, and `protocol_grader.py` independently replays the private
scientific grader. Unlike interactive hidden-suite tasks, Tasks 10-13 do not need
a model-visible sidecar endpoint or task-specific interactive sidecar.

## Task contracts

- Task 10: 6 model iterations, 20 Python commands, 600 s/action, 3600 s total,
  no automatic retries. Required files: `final_answer.json`, `predictions.csv`,
  `pipeline.json`. Pass score 50; exact-label Oracle 100.
- Task 11: 6 iterations, 12 commands, 300 s/action, 1800 s total, no retries.
  Required file: `final_answer.json`; optional `candidate.json`. Pass score 60;
  Oracle 100. This remains a static RDKit compliance task, not a physical
  electrolyte-performance claim and not an Agent-leaderboard task.
- Task 12: 6 iterations, 12 commands, 900 s/action, 5400 s total, no retries.
  Required file: `final_answer.json`; optional `candidate.json`. The canonical
  `final_answer.candidate` wrapper and archived flat eight-field envelope are both
  supported. Pass score 60; frozen-benchmark Oracle 100 under the continuous
  scientific scoring revision.
- Task 13: 5 iterations, 15 commands, 600 s/action, 3000 s total, no retries.
  Required file: `final_answer.json`; optional `route.json`. The required key is
  `final_answer.route`. Pass score 60; six-step Oracle 100. This remains a
  deterministic single-target route-validation task, not an open-ended synthesis
  benchmark.

Per-task details are authoritative in `tasks/<task>/MIGRATION.md`,
`harness_contract.yaml`, `instruction.md`, and `tests/PRIVATE_SUITE.md`.

## Ownership boundary

Framework-owned and copied unchanged from the 2026-09-05 combined framework:
`chembench_harness/`, `agents/`, `schemas/`, the Harbor task schema, contract
immutable fields (`schema_version`, `protocol.state_file`, `ownership`,
`release_gates`), and the compatibility runner.

Task-author-owned: task instructions and public task inputs, task-local budgets
and artifact declarations, `MIGRATION.md`, `MIGRATION_MAPPING.yaml`, private
scientific grader/data, `private_suite_adapter.py`, `protocol_grader.py`, Oracle
fixtures, calibration thresholds, and task-specific validation records.

Set `parity_validated`, `release_gates.parity`, and `release_gates.build` from the
technical evidence for matched Harbor and legacy runs and the archived image
digests. The organizer approved 10–13 for public release on 2026-11-08 and
recorded that decision separately from those technical results.

## Validation

Run from the Harbor harness root:

```bash
python scripts/validate_structure.py tasks
python scripts/validate_handoff_10_13.py
python scripts/validate_tasks_10_13_protocol.py
```

The last command requires the pinned RDKit/PyBaMM verifier dependencies. Results
from this delivery are in `revision_records/`.

All four fixed-flow `solution/solve.sh` scripts are included in the task package.
The release ZIP stores these scripts and `tests/test.sh` with Unix mode `0755`;
after extraction they can be invoked directly in Harbor/Linux.

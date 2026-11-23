# Task 12 Harbor migration and author confirmation

Source revision: v2.0.2; grader `2.0.2-pybamm-continuous-envelope-compatible`. Canonical nested and archived flat envelopes remain equivalent. The 1400 mol m-3 floor is a benchmark-internal proxy, not a validated plating criterion.

## Public protocol mapping

- One Harbor invocation equals one legacy generic-Agent run.
- Limits: 6 model iterations, 12 Python commands, 900 s per command, 5400 s total, no automatic retry, and at most 20 file reads per iteration.
- Invalid JSON consumes an iteration; dispatched commands consume budget. Termination is `status=complete` plus `final_answer`, or exhaustion/timeout.
- Public: offline pinned PyBaMM SPMe/Chen2020, model card, command output and state. Private: grading simulation results, threshold decisions and score.
- `native_runner.example.py` is example-only and is not invoked by Harbor.

## Private scoring mapping

The organizer adapter hashes/snapshots only `final_answer.json` and optional `candidate.json`. The replay grader verifies that evidence and independently runs the unchanged private evaluator; reported metrics are never trusted.

Raw score is 0–100; pass is 60; reward is raw/100. The 15/10/10/20/15/10/20 dimensions, four-step continuous experiment, Casadi safe solver and 1e-7 tolerances are unchanged.

## Oracle, tests, calibration and lock

See `tests/PRIVATE_SUITE.md`: the 2C/4.1 V eight-field Oracle and invalid missing-field, out-of-range and non-finite cases. Replay tolerance is ≤ `1e-9` with exact failures/runtime contract. `CLEAN_ROOM_REPLAY.md` records lockfile hashes, expected versions, replay command and runtime fingerprint status requested by the 2026-08-26 recheck.

Framework core, compatibility runner, ownership/release gates and boundaries are unchanged. Offline replay can confirm scientific/envelope parity. Live same-model Harbor parity and a built image digest remain pending without Docker/Harbor/model credentials, so release gates stay false.

## 2026-09-27 scoring-contract disclosure fixes

<!-- 2026-09-27 fix: audit record of grader/rubric/instruction consistency fixes; historical statements above are preserved unmodified. -->

- File-selection priority: the private grader previously read `candidate.json` first and only fell back to `final_answer.json`, inverting the published contract (`final_answer.json` canonical, `candidate.json` optional audit artifact). Fixed in `tests/task_snapshot/private_grader/grader.py`: `final_answer.json` is now read first; `candidate.json` is consulted only when `final_answer.json` is absent.
- Rubric synchronization: the public rubric (`environment/public/rubric.json`, mirrored under `tests/task_snapshot/rubric.json`) listed 15/10/10/20/15/10/20-weighted dimensions, which did not match grader execution. It now states the executed breakdown: three 0-point pass/fail hard gates (`valid_submission_and_protocol`, `electrode_manufacturability`, `simulation_completed`; failing any one zeroes all continuous dimensions) plus fast_charge 30 / discharge_capacity 25 / voltage_safety 20 / plating_proxy 25 continuous points. The historical "15/10/10/20/15/10/20 dimensions ... are unchanged" sentence above is kept for audit but did not describe grader execution.
- Anchor/gate overlap: `input/MODEL_CARD.md` now discloses that the continuous full-score anchors coincide with the hard-gate boundaries, i.e. passing all hard gates is already the full-score band, and continuous interpolation provides gradient only below the gates. Anchor values are unchanged. It is recommended that the author evaluate whether to recalibrate these anchors to restore an optimization gradient.
- Budget disclosure: `instruction.md` now publicly documents the agent budget (6 rounds, 12 tool calls) and that exhausting the round/tool budget without a scorable answer yields 0 points.


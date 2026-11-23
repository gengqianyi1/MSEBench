# Task 10 Harbor migration and author confirmation

## Scope and provenance

Scientific source: Matbench `matbench_expt_gap`; “MSEBench” in paths is only the packaging namespace. Attribution is in `input/data/README.md`. Because the archive has no machine-readable license field, upstream redistribution-license approval remains an organizer release action.

Source revision: v2.1.0; grader `2.1.0-private-continuous-metrics`; terminal artifacts: `predictions.csv`, `pipeline.json`, `final_answer.json`.

## Public protocol mapping

- One Harbor task invocation equals one legacy generic-Agent run.
- Limits: 6 model iterations, 20 Python commands, 600 s per command, 3600 s total, no automatic retry, and at most 20 file reads per iteration.
- Invalid model JSON consumes an iteration. A dispatched command consumes budget even when it fails or times out.
- Termination: `status=complete` plus `final_answer`, or iteration/command/wall-clock exhaustion.
- Public: offline Python/NumPy/pandas/scikit-learn, labelled training CSV, unlabelled validation/test features, command results and `protocol_state.json`. Private labels/metrics never enter the workspace.
- `native_runner.example.py` is non-authoritative example code; Harbor supplies the model.

## Private scoring mapping

`tests/private_suite_adapter.py` snapshots only allowlisted files and commits size/SHA-256 to organizer evidence outside `/workspace`. `tests/protocol_grader.py` verifies that snapshot and replays the unchanged private grader/labels. Candidate-reported metrics are ignored.

Raw score is 0–100; pass is 50; reward is raw/100. The 10/30/35/25 rubric and MAE/R² anchors are unchanged. Test R² and zero/positive-gap MAE remain diagnostics.

## Oracle, tests, calibration, parity

See `tests/PRIVATE_SUITE.md`: exact-label Oracle = 100; invalid cases cover missing, duplicate and unknown IDs. Replay tolerance is raw-score difference ≤ `1e-9` with exact failure-code and recomputed-metric equality.

The 2026-08-26 recheck accepted continuous scoring. Archived model scores 87.51/51.06/69.57 are calibration evidence, not new model runs. Repeated-run uncertainty is recommended for high-discrimination publication claims.

Framework core, schemas, compatibility runner, ownership/release gates and boundaries are unchanged. Offline evidence replay may confirm scorer parity. Live same-model trajectory parity, image digest and external license approval remain release evidence; their gates stay false.


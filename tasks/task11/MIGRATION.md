# Task 11 Harbor migration and author confirmation

Source revision: v2.0.2; grader `2.0.0-rdkit-surrogate-only`. This remains a deterministic static cheminformatics-compliance task, not a real electrochemical/fast-charge benchmark, and remains outside the Agent leaderboard.

## Public protocol mapping

- One Harbor invocation equals one legacy generic-Agent run.
- Limits: 6 model iterations, 12 Python commands, 300 s per command, 1800 s total, no automatic retry, and at most 20 file reads per iteration.
- Invalid JSON consumes an iteration; dispatched commands consume budget regardless of result. Termination is `status=complete` plus `final_answer`, or exhaustion/timeout.
- Public: `protocol_state.json`, fixed model card and offline RDKit descriptors. Private: reserved canonical-SMILES set, score and private failures.
- `native_runner.example.py` is example-only; Harbor supplies the model.

## Private scoring mapping

The organizer-only adapter captures `final_answer.json` and optional `candidate.json`, records hashes, and replays immutable evidence. `protocol_grader.py` invokes the unchanged private RDKit surrogate and reserved set. Reported properties/scores are ignored.

Raw score is 0–100; pass is 60; reward is raw/100. Five 20-point dimensions and hard gates are unchanged.

## Oracle, tests and parity

See `tests/PRIVATE_SUITE.md`: Oracle `COCCOC(=O)C(F)OC` = 100; invalid cases cover parse failure, reserved-set leakage and the three-oxygen-neighbour filter. Replay tolerance is ≤ `1e-9` with exact failure/property evidence.

The 2026-08-26 recheck requires no scientific change while `benchmark_role=static_cheminformatics_compliance` and `leaderboard_eligible=false`. Framework core, compatibility runner, ownership/release gates and boundaries are unchanged. Offline replay checks scorer parity; live same-model trajectory parity and image digest remain pending release evidence.


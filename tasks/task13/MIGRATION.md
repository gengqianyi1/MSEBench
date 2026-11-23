# Task 13 Harbor migration and author confirmation

Source revision: v2.0.2; grader `2.0.2-public-contract-no-target-double-count`. This remains a deterministic single-target route-validation task and stays outside the Agent leaderboard until a frozen hidden multi-target set exists.

## Public protocol mapping

- One Harbor invocation equals one legacy generic-Agent run.
- Limits: 5 model iterations, 15 Python commands, 600 s per command, 3000 s total, no automatic retry, and at most 20 file reads per iteration.
- Invalid JSON consumes an iteration; dispatched commands consume budget. Termination is `status=complete` plus `final_answer`, or exhaustion/timeout.
- Public validator feedback: reaction firing, product match, availability, heavy-atom conservation, atom economy and errors. Private reference route and score stay in the verifier.
- Public validator/private grader share the reaction contract. `native_runner.example.py` is example-only.

## Private scoring mapping

The organizer adapter captures/hashes only `final_answer.json` and optional `route.json`. The replay grader verifies evidence and invokes the unchanged evaluator. Exact target equality occurs only in the 40-point target component, not process validity.

Raw score is 0–100; pass is 60; reward is raw/100; dimensions are 35/25/40.

## Oracle, tests and parity

See `tests/PRIVATE_SUITE.md`: six-step Oracle = 100; invalid cases cover empty route, unknown reaction and valid-prefix/wrong-final route. Replay tolerance is ≤ `1e-9` with exact failure/per-step evidence.

The 2026-08-26 recheck requires no scientific change while non-leaderboard. Framework core, compatibility runner, ownership/release gates and boundaries are unchanged. Offline evidence replay checks scorer parity; live same-model trajectory parity and image digest remain pending release evidence.


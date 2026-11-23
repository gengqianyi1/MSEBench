# Task 14 M1.B design (multi-step, parity-preserving)

## Static status
- Five Harbor steps are declared in `task.toml`; `multi_step_reward_strategy = "final"`.
- The evaluated model never runs from task-owned code.
- Cross-step conversation persistence is temporarily implemented by `/workspace/protocol_state.json`.
- Each round writes exactly one `/workspace/rounds/round-N/candidate.json`.
- Rounds 2-5 setup hooks deterministically evaluate the previous candidate and expose only legacy-equivalent reduced feedback.
- Rounds 1-4 verifier scripts enforce protocol completeness only.
- Round 5 falls back to the root separate verifier; `final_grader.py` reconstructs legacy `submitted_candidate` trajectory records and the legacy private grader recomputes the scientific score.
- Offline feedback replay parity: 20/20 rounds matched.
- Offline final-grader replay parity: 4/4 calibrated corpora matched.

## Synthetic oracle
A deterministic five-step Oracle solution is included under each `steps/round-N/solution/`. It replays
the calibrated DeepSeek trajectory with legacy raw score 47, so the expected final Harbor reward is
0.47. This is a runtime/lifecycle oracle, not an optimal chemistry solution. See `SYNTHETIC_ORACLE.md`.

## Remaining runtime gates
1. Harbor 0.21.0 task load and Docker builds.
2. Synthetic Oracle trial reaches expected final reward 0.47.
3. Shared-verifier residue probe on Harbor 0.21.0; if residue is visible, migrate rounds 1-4 to separate protocol verifiers before release.
4. `ChemBenchAgent` runtime smoke test for `/workspace/protocol_state.json` persistence.
5. Real-model legacy-vs-Harbor repeated parity.
6. Native Harbor resume can be evaluated later; it does not block the deterministic filesystem-state implementation.

Until those gates pass: `protocol_decoupled = false`, `parity_validated = false`, `release_eligible = false`.

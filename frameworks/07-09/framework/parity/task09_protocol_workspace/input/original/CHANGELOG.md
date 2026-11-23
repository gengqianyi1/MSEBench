# Changelog — Adaptive Torsion v3 post-recharge release

- Preserved a complete pre-recharge snapshot and all historical evidence.
- Explicitly pinned the trusted root `.env` so stale inherited credentials cannot win; no credential value is persisted.
- Passed exact-ID recharge preflight for DeepSeek V4 Pro, Kimi K2.6, and GLM 5.2.
- Added complete per-run environment, optional reasoning/cache usage, oracle commitment, profile hash, and artifact hash metadata.
- Diagnosed the non-canonical r1 stale HTTP connection lifecycle and recreated the fixed-model client on every bounded attempt.
- Ran a clean locked r2 campaign over all 12 model×case combinations; 4 PASS and 8 evidenced external BLOCKED, with zero balance blocks.
- Independently replayed all 12 evaluations and validated 216/216 hidden oracle points, xTB 6.7.1 smoke, 483 tests, manifests, and secrets.
- Rebuilt only the Torsion participant/organizer releases; QED v4 and xTB Failure v2 remain frozen.

# Release validation — Adaptive Torsion v3 post-recharge

- Canonical campaign: `working/m/post_recharge_20260805_r2`, 12/12 formal terminals, 4 PASS and 8 evidenced external BLOCKED.
- Exact preflight identities: DeepSeek V4 Pro, Kimi K2.6, and GLM 5.2 all PASS with no fallback.
- Balance gate: zero `INSUFFICIENT_BALANCE` terminals after recharge.
- Replay/integrity: all trajectory chains, stored evaluations, independent replays, profile hashes, artifact hashes, and native manifests match.
- Scientific runtime: xTB 6.7.1; fresh four-point smoke PASS; all four oracle commitments match; hidden native oracle points 216/216.
- Tests/security: 483 passed; working-tree secret findings 0; no project `.env`.
- Packaging: only the Torsion participant and organizer archives are rebuilt. QED v4 and xTB Failure v2 archives remain frozen byte-for-byte.

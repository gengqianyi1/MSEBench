# Release validation — QED v4

- Formal evidence: 15/15 real runs present; exact requested/actual identities; no fallback.
- Evaluator: all stored evaluations equal a fresh deterministic replay.
- Tests: 194 passed.
- Boundary: participant has one public case and no organizer-private case or formal model trajectory.
- Secrets: `.env` is never copied; package scans compare against loaded credential values without printing them.
- Packaging: deterministic member order and per-member SHA-256 MANIFEST are required for both roles.

The participant ZIP is the submission artifact. The organizer ZIP is private verification material.

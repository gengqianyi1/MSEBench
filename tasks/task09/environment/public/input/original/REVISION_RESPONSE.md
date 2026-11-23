# Revision response — Adaptive xTB Torsion Profile v3 post-recharge

The pre-recharge project and archives were snapshotted without deletion. The recharged workspace credential was loaded from the explicitly pinned root `.env` without logging any value; all three exact model IDs passed a one-turn, zero-xTB preflight.

The first post-recharge attempt (`r1`) exposed a multi-turn stale-connection lifecycle defect and was stopped without becoming canonical. The runner was hardened so every bounded application attempt receives a fresh fixed-model client; sampling budget, prompt, retry count/delay, four cases, oracle, native xTB protocol, and scoring were unchanged. Regression coverage increased from 480 to 483 tests.

The clean `r2` campaign ran all 12 combinations from one lock. Four completed scientific PASS cases and eight genuine external provider blocks remain. All blocks are network connection or provider overload evidence with null scores; no balance, permission, fallback, fabricated trajectory, or cross-model reuse remains. The complete result is deliverable under the audit rule allowing fully evidenced external BLOCKED terminals.

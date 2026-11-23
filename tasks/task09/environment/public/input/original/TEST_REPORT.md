# Test report — Adaptive Torsion v3 post-recharge

Pre-campaign and post-campaign full suites passed. Final post-campaign command: `python -m pytest -q`; result: **483 passed in 12.64 s**. This is above the historical 480-test floor.

Additional gates passed: Python syntax compilation; exact-ID three-model preflight; fresh four-point xTB 6.7.1 smoke; four oracle commitments; 216/216 hidden native oracle points; 12/12 trajectory hash-chain validation; 12/12 profile/artifact hash validation; 12/12 stored-versus-replayed evaluator equality; and working-tree secret scan with zero findings. Role-specific staging/ZIP boundary, manifest, extraction, and extracted test gates are enforced by the Torsion-only release builder.

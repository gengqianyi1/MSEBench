# Test report — QED v4

Final command: `python -m pytest -q`.

Result: **194 passed in 5.04 s** on 2026-08-05. The suite covers schema closure, public/private separation, halogen constraints, budget enforcement, score boundaries, adversarial inputs, model runner behavior, hash manifests, and offline replay.

Formal replay additionally verified 15 stored model evaluations byte-for-byte. The earlier dependency-invalid and cap-calibration campaigns are preserved but excluded from the release result.

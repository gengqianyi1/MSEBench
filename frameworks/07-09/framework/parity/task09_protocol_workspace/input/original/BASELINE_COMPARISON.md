# Baseline comparison — Adaptive Torsion v3

Frozen four-case calibration (`validation/baselines_20260805_r1`) is unchanged by the recharge rerun:

| Baseline | Macro official | Per-case behavior |
|---|---:|---|
| Weak 6-angle | 0.000000 | fails minimum adaptive coverage |
| Fixed 12-angle | 83.714238 | 89.2461, 97.5, 97.5, 50.6109 |
| Oracle-guided greedy diagnostic | 72.208436 | organizer calibration only |

The fixed public policy is strong on smooth profiles but materially weaker on the narrow amide barrier, demonstrating cross-morphology difficulty without changing the established periodic/symmetry scoring.

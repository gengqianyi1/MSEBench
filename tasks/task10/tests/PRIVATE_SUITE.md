# Private suite, Oracle, thresholds and calibration

This verifier-only suite is implemented by `scripts/validate_tasks_10_13_protocol.py`; no fixture content is copied to the model workspace.

| Case | Expected raw score | Expected failure |
|---|---:|---|
| `oracle_exact_private_labels` | 100.0 | none |
| `invalid_missing_id` | 0.0 | `MISSING_IDS` |
| `invalid_duplicate_id` | 0.0 | `DUPLICATE_ID`, `MISSING_IDS` |
| `invalid_unknown_id` | 0.0 | `UNKNOWN_IDS` |

Scoring bounds are 0–100, pass threshold 50, reward = raw/100. Continuous anchors: validation MAE 0.30/0.65 eV, test MAE 0.30/1.20 eV, validation R² 0.55/0.75. Replay tolerance is `1e-9` raw points plus exact result/failure evidence. Calibration uses the archived 87.51/51.06/69.57 scores and the 100-point Oracle; no OOD gate is added.


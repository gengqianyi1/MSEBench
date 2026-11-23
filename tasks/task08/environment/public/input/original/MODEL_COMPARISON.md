# Model comparison — xTB Failure v2

Evidence: `working/m/r3`; 18/18 real trajectories, correct per-case evaluator configuration, exact offline replay, and no BLOCKED cases.

| Model | Exact model ID | PASS / FAIL | Macro official | Public raw/official | Hidden raw scores |
|---|---|---:|---:|---|---|
| DeepSeek | `deepseek-ai/DeepSeek-V4-Pro` | 1 / 5 | 16.6667 | 100 / 100 | 22, 24, 22, 24, 24 (official 0) |
| Kimi | `Pro/moonshotai/Kimi-K2.6` | 0 / 6 | 0.0000 | 24 / 0 | 24, 24, 5, 24, 74 (official 0) |
| GLM | `zai-org/GLM-5.2` | 0 / 6 | 0.0000 | 20 / 0 | 24, 24, 24, 22, 24 (official 0) |

The gap between raw and official scores demonstrates the intended scientific gates. These are real FAIL scores, not provider BLOCKED values.

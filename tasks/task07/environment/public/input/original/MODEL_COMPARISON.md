# Model comparison — QED v4

Evidence: `working/formal_runs_v4_20260805_r3`. All 15 case runs received a model response, used the exact requested ID, and replayed exactly.

| Model | Exact model ID | Case results | Macro official | Ranking eligible |
|---|---|---|---:|---|
| DeepSeek | `deepseek-ai/DeepSeek-V4-Pro` | PASS 23.9553, PASS 60.4061, PASS 37.0955, FAIL 0, PASS 24.2727 | 29.1459 | yes |
| Kimi | `Pro/moonshotai/Kimi-K2.6` | PASS 21.7952, PASS 22.7578, FAIL 0, PASS 20.2998, FAIL 0 | 12.9706 | yes |
| GLM | `zai-org/GLM-5.2` | PASS 37.6286, PASS 59.3221, FAIL 0, FAIL 0, FAIL 0 | 19.3901 | yes |

FAIL is a scored scientific/protocol failure, not a provider failure. No case is BLOCKED. The former credential-missing/zero-call table is obsolete.

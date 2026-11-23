# Model comparison — Adaptive Torsion v3 post-recharge

Canonical evidence is `working/m/post_recharge_20260805_r2`. All 12 model×case terminals were produced by one locked runner/evaluator/configuration set and independently replayed. BLOCKED rows are null, excluded from numeric aggregates, and disqualify coverage-based ranking; they are never converted to zero.

## Per-case results

| Model | Case | Status | Official | Scientific | Process | xTB ok/fail | Model calls | Retries | Requested / actual |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| deepseek | `2_fluorobiphenyl_central_torsion_v1` | PASS | 68.21965443164 | 48.21965443164 | 20 | 12/0 | 3 | 2 | `deepseek-ai/DeepSeek-V4-Pro` / `deepseek-ai/DeepSeek-V4-Pro` |
| deepseek | `butane_central_torsion_v1` | BLOCKED / NETWORK_CONNECTION_FAILED | null | null | null | 6/0 | 3 | 5 | `deepseek-ai/DeepSeek-V4-Pro` / `deepseek-ai/DeepSeek-V4-Pro` |
| deepseek | `difluoroethane_fccf_torsion_v1` | BLOCKED / NETWORK_CONNECTION_FAILED | null | null | null | 6/0 | 2 | 3 | `deepseek-ai/DeepSeek-V4-Pro` / `deepseek-ai/DeepSeek-V4-Pro` |
| deepseek | `n_methylacetamide_cn_torsion_v1` | BLOCKED / NETWORK_CONNECTION_FAILED | null | null | null | 9/0 | 3 | 6 | `deepseek-ai/DeepSeek-V4-Pro` / `deepseek-ai/DeepSeek-V4-Pro` |
| kimi | `2_fluorobiphenyl_central_torsion_v1` | BLOCKED / NETWORK_CONNECTION_FAILED | null | null | null | 6/0 | 2 | 3 | `Pro/moonshotai/Kimi-K2.6` / `Pro/moonshotai/Kimi-K2.6` |
| kimi | `butane_central_torsion_v1` | BLOCKED / PROVIDER_OVERLOADED | null | null | null | 6/0 | 2 | 6 | `Pro/moonshotai/Kimi-K2.6` / `Pro/moonshotai/Kimi-K2.6` |
| kimi | `difluoroethane_fccf_torsion_v1` | BLOCKED / NETWORK_CONNECTION_FAILED | null | null | null | 9/0 | 3 | 4 | `Pro/moonshotai/Kimi-K2.6` / `Pro/moonshotai/Kimi-K2.6` |
| kimi | `n_methylacetamide_cn_torsion_v1` | BLOCKED / NETWORK_CONNECTION_FAILED | null | null | null | 9/0 | 3 | 5 | `Pro/moonshotai/Kimi-K2.6` / `Pro/moonshotai/Kimi-K2.6` |
| glm | `2_fluorobiphenyl_central_torsion_v1` | BLOCKED / NETWORK_CONNECTION_FAILED | null | null | null | 0/0 | 1 | 3 | `zai-org/GLM-5.2` / `null` |
| glm | `butane_central_torsion_v1` | PASS | 100 | 80 | 20 | 12/0 | 3 | 0 | `zai-org/GLM-5.2` / `zai-org/GLM-5.2` |
| glm | `difluoroethane_fccf_torsion_v1` | PASS | 99.791666666667 | 80 | 19.791666666667 | 11/0 | 4 | 0 | `zai-org/GLM-5.2` / `zai-org/GLM-5.2` |
| glm | `n_methylacetamide_cn_torsion_v1` | PASS | 66.835672287878 | 46.835672287878 | 20 | 12/0 | 4 | 0 | `zai-org/GLM-5.2` / `zai-org/GLM-5.2` |

## Aggregates

| Model | PASS | FAIL | BLOCKED | Scored mean | Std. dev. | Min | Max | Ranking eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek | 1 | 0 | 3 | 68.21965443164 | 0 | 68.21965443164 | 68.21965443164 | no |
| kimi | 0 | 0 | 4 | null | null | null | null | no |
| glm | 3 | 0 | 1 | 88.875779651515 | 15.584941453102 | 66.835672287878 | 100 | no |

## Baseline context

| Frozen baseline | Four-case macro official | Interpretation |
|---|---:|---|
| Weak 6-angle | 0.000000 | below minimum adaptive coverage |
| Fixed 12-angle | 83.714238 | strong on smooth cases, 50.6109 on the amide stress case |
| Oracle-guided greedy diagnostic | 72.208436 | organizer-only calibration, not a participant baseline |

## Sampling and score diagnostics for valid scientific trajectories

| Model | Case | RMSE | MaxAE | Min F1 | Max F1 | Barrier MAE | Max barrier error |
|---|---|---:|---:|---:|---:|---:|---:|
| deepseek | `2_fluorobiphenyl_central_torsion_v1` | 0.854092803563 | 2.229620031406 | 0.5 | 1 | 0.255898727653 | 0.390411850256 |
| glm | `butane_central_torsion_v1` | 0.083825742127 | 0.209817848101 | 1 | 1 | 0.004499307364 | 0.010130278934 |
| glm | `difluoroethane_fccf_torsion_v1` | 0.032182769129 | 0.081124796811 | 1 | 1 | 0.030056936696 | 0.072114751145 |
| glm | `n_methylacetamide_cn_torsion_v1` | 0.61612712281 | 2.16723896733 | 0.5 | 1 | 1.868277975287 | 1.939775928396 |

DeepSeek's valid public-case trajectory began with a periodic six-angle scaffold and refined near predicted curvature/extrema, producing 12 unique points. GLM's three valid trajectories adapted case-by-case: 12 points for butane, 11 for difluoroethane after one protocol-correction turn, and 12 for N-methylacetamide with refinement around the narrow amide barrier. Kimi produced no complete scientific trajectory because all four provider sessions were externally blocked, so no sampling-policy or score inference is made for it.

Score differences come from independently recomputed curve error, symmetry-aware extrema, barrier error, adaptive coverage, and process gates—not model self-reports. GLM is near-perfect on the two smooth hidden cases but falls to `66.835672287878` on the sharp amide barrier; DeepSeek's public case scores `68.21965443164` because RMSE/MaxAE and minimum-extrema recall are weaker despite a correct global basin and full process score.

There are no `INSUFFICIENT_BALANCE` terminals after recharge. The eight remaining BLOCKED rows carry bounded-retry evidence for `NETWORK_CONNECTION_FAILED` or `PROVIDER_OVERLOADED`. DeepSeek has one scored PASS (`68.21965443164`); Kimi has no scored case; GLM has three scored PASS cases with mean `88.875779651515`. The historical GLM mean `88.892742` is not reused.

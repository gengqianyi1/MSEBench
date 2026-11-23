# 三个 LLM 在 Task 01–03 上的评测结果

生成时间: 2026-09-06

## 结果对比 (final_score)

| 模型 | task01 | task02 | task03 |
|---|---|---|---|
| deepseek-v4-pro | 75.877182 | 33.267 | 0.0 |
| kimi-k2.6 | 40.051821 | 34.361 | 0.0 |
| glm-5.2 | 68.758345 | 33.267 | 0.0 |

## 详细结果

### deepseek-v4-pro (https://api.deepseek.com)

| task | final_score | reward | efficiency | LLM 调用数 |
|---|---|---|---|---|
| task01 (01 JARVIS 2D exfoliation) | 75.877182 | 0.7588 | 1.0 | 39 |
| task02 (02 oligothiophene SMD-SAPT) | 33.267 | 0.3327 | 1.0 | 100 |
| task03 (03 five-ring xTB-DIPRO Marcus) | 0.0 | 0.0 | — | 6 |

### kimi-k2.6 (https://api.siliconflow.cn/v1)

| task | final_score | reward | efficiency | LLM 调用数 |
|---|---|---|---|---|
| task01 (01 JARVIS 2D exfoliation) | 40.051821 | 0.4005 | 1.0 | 29 |
| task02 (02 oligothiophene SMD-SAPT) | 34.361 | 0.3436 | 1.0 | 166 |
| task03 (03 five-ring xTB-DIPRO Marcus) | 0.0 | 0.0 | — | 9 |

### glm-5.2 (https://api.siliconflow.cn/v1)

| task | final_score | reward | efficiency | LLM 调用数 |
|---|---|---|---|---|
| task01 (01 JARVIS 2D exfoliation) | 68.758345 | 0.6876 | 1.0 | 94 |
| task02 (02 oligothiophene SMD-SAPT) | 33.267 | 0.3327 | 1.0 | 96 |
| task03 (03 five-ring xTB-DIPRO Marcus) | 0.0 | 0.0 | — | 7 |

## 说明

- deepseek-v4-pro 使用 https://api.deepseek.com；Kimi-K2.6 与 GLM-5.2 使用 https://api.siliconflow.cn/v1。
- task01 上模型分化明显（deepseek 75.88 ≈ GLM 68.76 > Kimi 40.05）。
- task02（SAPT 证据审计）上三者接近（33–34），区分度低；Kimi 调用数最多（166）。
- task03（xTB-DIPRO Marcus 优化）三个模型均未改善初始构象的 Marcus 速率，得 0.0。
- 原始结果文件（result.json）位于各模型 job 目录的 jobs/<job_name>/result.json；
  调用日志（llm_calls.jsonl）位于各模型 job 目录根下 task0X_llm_calls.jsonl。
- 运行脚本与模型配置见 agents/、scripts/、profiles/、templates/sidecar/。

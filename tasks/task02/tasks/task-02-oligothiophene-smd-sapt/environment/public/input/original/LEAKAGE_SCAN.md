# Task 02 — Public Leakage Scan

Generated: 2026-08-07 | Target: `task_02_public/`

| Scan | Command | Result |
|------|---------|--------|
| scoring_reference.json | `grep -rl "scoring_reference.json" task_02_public/` | ✅ 0 files contain `scoring_reference.json` as actual file (only `src/main.py`, `src/scoring.py`, `evaluation_plan.md`, `task_config.yaml` reference the key name, but the files have been deleted) |
| scoring_reference.json 文件 | `find task_02_public/ -name "scoring_reference.json"` | ✅ 0 files found (deleted, migrated to `task_02_private/`) |
| reference_value | `grep -r "reference_value" task_02_public/` | ✅ 0 matches |
| target_score | `grep -r "target_score" task_02_public/` | ✅ 0 matches |
| hidden_anomaly | `grep -r "hidden_anomaly" task_02_public/tasks/` | ✅ Only `question.md` says "存在异常但数量未告知" (no count/position leaked) |
| 可疑数值 | `grep -rE "[0-9]+\.[0-9]{4,}" task_02_public/tasks/*/question.md` | ✅ Only public constraint ranges (e.g., budget limits), no reference energies/SAPT values |

结论：P0 泄漏已关闭。私有参考位于 wrapper 的 `private_data/`，不在 Agent 可见的 `input/original` 中；公开 `src/main.py` 不再包含私有目录环境变量入口或 public-path fallback。

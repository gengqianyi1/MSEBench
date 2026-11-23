# Task 01 — Public Leakage Scan

Generated: 2026-08-07 | Target: `task_01_public/`

| Scan | Command | Result |
|------|---------|--------|
| 私有标签路径 | `grep -r "private/\\|evaluator_private" task_01_public/` | ✅ Only in `.dockerignore`, `SECURITY_AND_PRIVATE_DATA.md` (documentation), `MANIFEST.json` (listed as `private_label_packaged: false`) |
| hidden labels | `grep -r "test_labels_private\\|hidden_test" task_01_public/` | ✅ 0 matches |
| 答案数值泄漏 | `grep -rE "exfoliation_energy.*\d+\\.\d+" task_01_public/tasks/` | ✅ Only schema documentation in `question.md`, no actual values |
| scoring 阈值 | `grep -r "mae_full_score_threshold\\|mae_zero_score" task_01_public/` | ✅ 0 matches (thresholds in private `private_data/`) |
| Docker context 隔离 | `grep '' task_01_public/.dockerignore` | ✅ Contains `private/` and `runs/` |

结论：无泄漏。

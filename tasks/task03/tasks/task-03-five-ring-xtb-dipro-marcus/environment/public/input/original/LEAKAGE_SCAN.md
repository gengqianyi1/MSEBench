# Task 03 — Public Leakage Scan

Generated: 2026-08-24 | Target: complete public tree `input/original/`

| Scan | Command | Result |
|------|---------|--------|
| reference filenames | Recursively check `satisfied_molecules.*`, `batch_v2.ps1`, `batch_marcus.py`, and `validate_designs.py` | ✅ 0 public files; reference CSV/JSON retained only under wrapper `private_data/reference_examples/` |
| scripted replay assets | Recursively check `api_config_codex_scripted.json`, `codex_scripted.yaml`, and `scripted_responses` | ✅ 0 public files/content |
| all public script/config formats | Scan `.py`, `.ps1`, `.json`, `.yaml`, `.yml`, `.csv`, and `.md`, not only previously named Python scripts | ✅ no embedded reference library found |
| target_rate | `grep -r "target_rate\\|target_hit_count" task_03_public/` | ✅ 0 matches |
| Vh_DIPRO 数值泄漏 | `grep -rE "Vh_DIPRO\s*[=:]\s*\d+\.\d+" task_03_public/tasks/` | ✅ 0 matches in question.md (only usage instruction, no values) |
| Marcus 速率数值 | `grep -rE "marcus_hole_rate.*\d+[eE][+-]?\d+" task_03_public/tasks/` | ✅ 0 matches |
| scoring 阈值/目标 | `grep -r "scoring_target\\|score_cutoff\\|benchmark_target" task_03_public/tasks/` | ✅ 0 matches |
| MANIFEST files 验证 | Check `MANIFEST.json files[]` and enforce denylist in `scripts/validate_package.py` | ✅ no reference/replay assets |

结论：P0 泄漏已关闭。此前漏检的 `scripts/batch_v2.ps1`、`configs/api_config_codex_scripted.json` 和 `configs/codex_scripted.yaml` 已从 public 包删除；参考 CSV/JSON 仅保存在 wrapper 私有目录。发布校验现在会阻止这些文件或等价 scripted 内容再次进入公开包。

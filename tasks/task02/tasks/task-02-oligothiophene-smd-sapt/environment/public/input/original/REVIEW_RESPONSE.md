# 审核意见逐条回复（revision2 同步版）

- 更新日期：2026-08-07（同步 20260807 综合修改意见与复审意见的 P0/P1 项）
- 修订版：1.3（wrapper、task config和`MANIFEST.json`均为1.3）
- 协议名：保留 `v3` 作为稳定协议标识；unbounded control 仅归档，不计入榜单

## 20260807 综合修改意见 P0/P1 逐条应答

| 审核项 | 严重性 | 修改与回复 | 证据路径 | 验证状态 |
|---|---|---|---|---|
| 公开控制/主任务目录泄漏评分参考 | P0 | 私有参考保存在 wrapper 的 `private_data/`；Agent 仅挂载 `input/original`；评分由 post-run `private_grader` 完成 | `native_runner.py`、`private_grader/grader.py`、`tests/test_private_isolation.py` | 已闭环 |
| 主任务、control V3、config 版本与规则未完全一致 | P1 | 主任务、control及三题统一release version 1.3；预算规则统一为 1–30 无罚 / 31–40 每次 +1 / 41–50 每次 +2 | `README.md`、task configs、`MANIFEST.json` | 已闭环 |
| SAPT 数值语义未冻结 | P1 | 冻结近简并规则（前两大吸引项 ΔE ≤ 0.5 kcal/mol 双给）、fragment 定义、几何来源（唯一输入 `structures/R_g.xyz`）、单位/符号、收敛失败策略；提供 A/B/C/D 回归 fixture | `SAPT_NUMERICAL_FIXTURES.md`、`src/scoring.py::_verify_fragment_definition()` | 已闭环（fixture 表 + 代码行号） |

## 复审意见 P0/P1 补充应答

| 审核项 | 严重性 | 修改与回复 | 证据路径 | 验证状态 |
|---|---|---|---|---|
| SAPT 主导吸引项误判 elst/disp 归零 | P0 | 按真实日志最大吸引项判定，0.5 kcal/mol 近简并接受两项 | `src/scoring.py`（`_dominant_attractive`）、`SAPT_NUMERICAL_FIXTURES.md` Fixture A/B | 已解决（保留回归 fixture） |
| 污染判定扫描全文误压正答 | P0 | 只检查 `accepted_inputs`/`used_inputs` 等实际输入字段 | `src/scoring.py`（`_anomaly_contaminated`） | 已解决 |
| 约 11 分绑定未公开键名 | P0 | `docs/evidence_contract.md` 公开 atom mapping、起始几何/二面角、冻结/形变能键 | `docs/evidence_contract.md`、`question.md` | 基本解决（canonical 键名示例待补） |
| water 固定 electrostatics 标签不科学 | P0 | 按提交原始 SAPT 日志判定主导项，不再预设溶剂标签 | `src/scoring.py`、`question.md` | 已解决 |
| 难度标定需冻结评分器重跑 | P0 | 旧1.2三模型结果归档；统一1.3冻结包需要重新运行 | wrapper `runs_revalidation/` | 版本已闭环；1.3运行待执行 |
| LLM judge 披露不足 | P1 | LLM judge 已从计分中移除，100 分均为确定性评分 | `src/evaluator.py`、题目说明 | 已闭环 |
| 环境端到端证据 | P1 | 保留 `scripts/check_environment.py`；PySCF-SMD/Psi4/xTB 空镜像实跑记录待补 | `scripts/check_environment.py`、`Dockerfile` | 部分（发布前验收） |

## 2026-08-07 补充：私有评分链路运行级闭环

- 私有参考仅位于 wrapper 的 `private_data/`；公开 Agent 包和 `MANIFEST.json` 不包含真值文件。
- `src/main.py` 已删除环境变量入口及 public-path legacy fallback，无法读取私有参考；只有 post-run `private_grader` 接收 `private_data_dir`。
- `tests/self_test.py` 在无私有真值模式下以无秘密数值 mock 验证结构、schema、安全和 MANIFEST；在私有模式下验证数值评分，并断言主任务/control 选择不同且存在的私有文件。两种模式均通过。
- `scripts/build_manifest.py`、`tests/self_test.py` 与 `scripts/validate_package.py` 采用同一发布文件规则：排除本机 API 配置与 `runs/**` 历史运行产物，保留 `runs/.gitkeep`，其余发布文件进行 SHA-256 校验。

```bash
python scripts/validate_package.py
python tests/self_test.py
CHEMBENCH_TEST_PRIVATE_SCORING_DIR=/secure/path/task_02_scoring python tests/self_test.py
```

## 未闭环（发布门禁）

| 项 | 说明 | 责任 |
|---|---|---|
| 冻结1.3 scorer 重跑三模型 | 版本升级和沙箱协议变化后需生成与当前包哈希一致的正式结果 | 待执行 |
| LLM judge 区分度复测 | 不再适用：LLM judge 已从100分评分中移除 | 已关闭 |
| canonical 键名示例 | question.md/schema 中补充每个必交键的值来源与示例 | 出题人 |
| 空镜像端到端复现 | PySCF-SMD 编译后干净环境完整运行一次 | 出题人 |

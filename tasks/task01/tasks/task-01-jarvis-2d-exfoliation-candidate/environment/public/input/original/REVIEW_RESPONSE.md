# 审核意见逐条回复（revision2 同步版）

- 更新日期：2026-08-07（同步 20260807 综合修改意见与复审意见的 P0/P1 项）
- 修订版：1.3（wrapper、task config 与 `MANIFEST.json` 均为1.3）
- 目标候选数：保持 1000；工具预算：20 次正常调用 + 2 次修复调用

## 决策与版本

- 候选模块 50 分 / 性质模块 50 分保持不变；CHGNet 稳定性权重已由 25 降为 10，并明确仅为局部稳定性代理。
- 短库不整体清零：仅超过 1000 条才报错；短库按 `completion_fraction = n/1000` 计分。
- 旧 runs（revision1环境及v1.0/v1.1 scorer）不再与1.3分数混用，见 `ARCHIVED_SCORES.md`。

## 20260807 综合修改意见 P0/P1 逐条应答

| 审核项 | 严重性 | 修改与回复 | 证据路径 | 验证状态 |
|---|---|---|---|---|
| 历史成绩与修订评分器未形成单一事实来源 | P1 | 旧轨迹统一标记 `archived_historical`；wrapper、task config和MANIFEST统一为1.3；MANIFEST冻结当前private grader SHA-256 | `ARCHIVED_SCORES.md`、`MANIFEST.json` | 已闭环 |
| 候选门槛的科学/计分依据不完整 | P1 | 冻结 StructureMatcher 容差（ltol=0.2/stol=0.3/angle_tol=5°）、0.5 Å 硬阈值原子重叠判定、999/1000/1001 候选数量边界、病理×数量交互，全部给出临界 fixture | `CANDIDATE_THRESHOLD_FIXTURES.md`、`src/candidate_scoring.py` | 已闭环（fixture 表 + 代码行号） |
| 运行时隔离证据缺失 | P1 | 提供 Docker 挂载白名单、路径越界防护（`_validate_path()`）、网络隔离三层、拒绝访问测试清单 | `ISOLATION_EVIDENCE.md`、`tests/test_tool_executor.py`、`.dockerignore` | 已闭环（文档 + 测试存在） |
| 私有标签/评分参考泄漏 | P0 | 泄漏扫描确认 public 包无 `private/` 路径、无隐藏标签、无答案数值、无评分阈值；`private/` 与 `runs/` 在 .dockerignore 排除 | `LEAKAGE_SCAN.md`、`.dockerignore` | 已闭环（扫描表） |

## 历史项复核（revision1 已解决，保留证据）

| 审核问题 | 修改与回复 | 证据路径 |
|---|---|---|
| 题面、配置预算不一致 | 统一为 20+2 | `README.md`、`configs/*.yaml`、`tasks/jarvis_2d_exfoliation/{question.md,task_config.yaml}` |
| 少于 1000 条被整个候选模块清零 | 仅超过 1000 条报错；短库按固定 1000 分母计分 | `src/candidate_scoring.py`、`tests/test_candidate_validator.py` |
| selected candidates 字段歧义 | schema 明确要求五个对象数组 | `result_schema.json`、`question.md` |
| checksum 为空 | 已登记五项数据文件 sha256 | `data/checksums.sha256` |
| CHGNet 权重过高 | 25 → 10，明确局部代理定位 | `CANDIDATE_SCORING_SPEC.md`、`question.md` |

## 仍未闭环（发布门禁）

| 项 | 说明 | 责任 |
|---|---|---|
| 冻结v1.3 scorer后补齐三模型 | DeepSeek与GLM已有1.3结果；Kimi Task01仍需正式运行或保留可聚合的无效提交记录 | 待执行 |
| Dockerfile 空镜像端到端复现 | 需要一次干净环境从零运行记录 | 出题人（发布前验收） |
| 多模型×多种子校准报告 | 阈值敏感性、CHGNet 成功率分布 | 出题人（正式榜单前） |

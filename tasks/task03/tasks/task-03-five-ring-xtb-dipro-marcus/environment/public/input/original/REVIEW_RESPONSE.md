# 审核意见逐条回复（revision2 同步版）

- 更新日期：2026-08-24
- 修订版：1.3（`MANIFEST.json` benchmark_version = 1.3）

## 2026-08-07 最终推荐机制追加闭环

### 审核意见

> 五轮设计结束后必须由 LLM 从 R1-R5 选择最终推荐，不得新造 R6；只评价推荐分子。硬约束失败、计算失败或不存在的 ID 均为 0。rate 从 1e14 的 10 分到 1e15 的 100 分线性插值。

### 问题分析

V1.1 的 `select_best_rate()` 会替 LLM 自动选最高速率，无法测量模型对五轮证据的最终决策能力；旧 30+70 计分也与新的资格 gate 和线性分数不一致。

### 修改措施

- `src/main.py`：第五轮后发送只含 R1-R5 结构、约束、lambda、真实 Vh、rate、状态与失败原因的推荐请求；解析结果仅接受 R1-R5，不执行第六次分子评估。
- `src/scoring.py`：只按 `recommended_candidate_id` 定位候选；选择无效、硬约束或计算状态失败时计 0；合格 rate 使用 `min(100, 100*rate/1e15)`，低于 1e14 为 0。
- `task_config.yaml`、`question.md`、`MANIFEST.json`、`VERSION_MAP.md`：当前统一为1.3并公开唯一合同。

### 验证结果

实际边界测试：9.9e13→0，1.0e14→10，2.5e14→25，1.0e15→100，2.0e15→100；Harness 私有 grader fixture 选择 R3、rate=2.5e14 得 25 分。
- 历史版本：V1.0（含 `multi_packing_robustness`、四档离散、DIPRO CT 耦合误判）—— 旧轨迹不能用 1.1 scorer 重评，见 `VERSION_MAP.md`

## 20260807 综合修改意见 P0/P1 逐条应答

| 审核项 | 严重性 | 修改与回复 | 证据路径 | 验证状态 |
|---|---|---|---|---|
| 公开 oracle 泄漏（参考答案 + 评分/回放脚本随包分发） | P0 | 除旧记录所列文件外，补充删除漏检的 `batch_v2.ps1` 和两份 scripted 配置；参考结果仅保存在 wrapper `private_data/reference_examples/`；发布校验覆盖全部公开脚本与配置 | `LEAKAGE_SCAN.md`、`scripts/validate_package.py`、`MANIFEST.json` | 已闭环 |
| 版本未收敛（MANIFEST/task config/文档指向不同规则） | P1 | wrapper、manifest、题面、README、复现文档和citation统一为1.3；旧结果均在版本表归档 | `VERSION_MAP.md`、`MANIFEST.json` | 已闭环 |
| 连续数值判定缺少 fixture | P1 | 冻结单位/精度（k_h 3 位有效数字、Vh ~0.001 eV、λ 0.001 eV、T=300 K）、构象/电荷/自旋协议（`--uhf 0/1`、`--chrg 0/1`）、收敛失败判定；提供 A–E 回归 fixture（含 Marcus 公式计算验证，修正为 k=3.78×10¹⁴ s⁻¹） | `DIPRO_MARCUS_TOLERANCE_FIXTURES.md`、`src/xtb_runner.py`、`src/scoring.py` | 已闭环（fixture 表 + 代码行号） |

## 复审意见 P0/P1 补充应答

| 审核项 | 严重性 | 修改与回复 | 证据路径 | 验证状态 |
|---|---|---|---|---|
| DIPRO 把 CT/命令参数误作空穴耦合 | P0 | 删除宽松关键词回退，只接受带单位的 `hole transport (occ. MOs)` 汇总行；明确拒绝 `--dipro 0.1`、threshold、electron/unoccupied 和 CT 行；支持 eV/meV 规范换算 | `src/xtb_runner.py`、`tests/test_parser.py` | 已解决（5个专用回归测试） |
| 硬约束仅为加分项、无区分 | P0 | 改为资格 gate：不满足硬约束即无分（eligibility gate） | `task_config.yaml`、`src/scoring.py` | 已解决 |
| 四档离散评分对 1 meV 量化敏感 | P1 | 改为连续插值评分；披露 Vh 打印精度（3 位，~1 meV）对分数的敏感性 | `src/scoring.py`、`DIPRO_MARCUS_TOLERANCE_FIXTURES.md` Fixture E | 已解决（含稳定性 fixture） |
| 历史结果与当前规则不一致 | P1 | V1.0–V1.2结果明确归档，不进入V1.3榜单 | `VERSION_MAP.md` | 版本标注已闭环；V1.3需重跑 |
| 轨迹可审计性不足（conversation.md 实为 JSON） | P1 | 原生Markdown路径写真实Markdown；wrapper结构化对话使用`.json`后缀；trajectory保留逐轮事件 | `src/main.py`、wrapper `native_runner.py` | 已闭环 |

## 未闭环（发布门禁）

| 项 | 说明 | 责任 |
|---|---|---|
| 冻结1.3后重跑三模型 | DIPRO、packing和几何协议已改变，旧结果仅作历史参考 | 待执行 |
| 科学数值链单元测试 | DIPRO 真实汇总、截断输出、electron/CT 排除和单位换算已覆盖；Marcus/并环边界可继续扩充 | DIPRO 项已关闭；其余为增强项 |
| 轨迹/对话真实格式 | conversation → Markdown 或 .json 后缀；trajectory 每轮结构化事件 | 出题人（P1） |
| 默认配置修复 | `api_config.example.json` 的 `max_tokens` 占位符改为数值并对齐实际运行 | 出题人（P2） |
| 反馈体积压缩 | `evaluation.to_dict()`（36 个 packing trial 详情）改为精简反馈 | 出题人（P2） |

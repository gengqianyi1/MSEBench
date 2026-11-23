# Task 01 — Candidate Threshold Fixtures

## StructureMatcher 容差

| 参数 | 值 | 选择理由 |
|------|-----|---------|
| `ltol` | 0.2 Å | Pymatgen 默认值；对周期 2D 结构的晶胞参数变化不敏感 |
| `stol` | 0.3 | 角度余弦容差；覆盖轻原子微量偏移 |
| `angle_tol` | 5° | 放宽至覆盖 CHGNet relax 产生的微量角度变化 |

## 候选数量

| 情况 | 计分行为 |
|------|---------|
| 恰好 1000 候选 | `completion_fraction = 1.0`，满分 |
| 少于 1000 | `completion_fraction = n/1000`，按比例计分 |
| 超过 1000 | 超量报错，不计分（非截断） |

代码：`src/candidate_scoring.py` `_score_candidate_count()`

## 结构病理检测（关键修改：0.5 Å 为硬阈值）

| 病理类型 | 阈值 | 说明 |
|---------|------|------|
| 原子重叠 | 任意两原子中心距 ≤ 0.5 Å | 判为 structural_pathology，不参与 CHGNet 筛选 |
| 异常价态 | `BVAnalyzer` 解析失败 | Pymatgen 内置验证 |
| 结构不可解析 | `Structure.from_dict()` 抛异常 | 不含原子坐标的候选直接拒绝 |

## 临界 Fixture

### Fixture A：候选数量边界
| 输入 | 期望 |
|------|------|
| 999 候选 | `completion_fraction = 0.999`，候选数量分 = `20 * 0.999 = 19.98` |
| 1000 候选 | `completion_fraction = 1.0`，满分 |
| 1001 候选 | 超量报错，不计分 |

### Fixture B：原子重叠边界（阈值 = 0.5 Å）
| 输入 | 期望 | 说明 |
|------|------|------|
| 候选含 0.5 Å 最短原子间距 | `structural_pathology` | 准确命中阈值 |
| 候选含 0.4 Å 最短原子间距 | `structural_pathology` | 低于阈值 |
| 候选最小原子间距 0.6 Å | 不标记 | 高于阈值，正常键级范围（C-C ~1.5 Å, VdW 间距 ~3.0 Å） |

### Fixture C：候选结构数量 × 结构病理交互
| 输入 | 期望 |
|------|------|
| 1000 候选中 1 含 0.5 Å 重叠 | 结构病理扣分 = 1/1000 = 0.008 分；该候选不进入 CHGNet 5 选池 |

## CHGNet 协议

| 参数 | 值 |
|------|-----|
| relax 版本 | `chgnet 0.4.2` relax v1 协议 |
| 选中候选数 | 5 个（预测剥离能最低的合法候选） |
| 每个候选调用次数 | 1 次 |
| 分数权重 | 10 分 / 100 分总分 |

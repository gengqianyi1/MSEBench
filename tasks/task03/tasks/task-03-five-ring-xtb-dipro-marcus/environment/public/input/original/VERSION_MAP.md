# Task 03 — Version Map

| 版本 | MANIFEST | 规则变化 | 历史结果 |
|------|---------|---------|---------|
| V1.0 | 1.0 | 含 `multi_packing_robustness`、四档离散、DIPRO CT 耦合误判 | DeepSeek 46.76（含已移除子分） |
| V1.1 | 1.1 | 移除 `multi_packing_robustness`、连续插值、DIPRO 锚定 hole transport 行 | 待首次运行 |
| V1.2 | 1.2 | 五轮后强制从 R1-R5 最终推荐；只评推荐分子；硬约束/计算/选择失败为 0；rate 线性计分 | 已归档 |
| V1.3 | 1.3 | 统一结果schema；严格DIPRO hole parser；删除公开oracle/replay；确定性主轴；扩展packing网格 | 当前冻结版本，需重新运行三模型 |

任何V1.0–V1.2轨迹均不得作为V1.3正式成绩；V1.3必须包含独立最终推荐事件。

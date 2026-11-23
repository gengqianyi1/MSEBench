# Task 05：咖啡因—水竞争构型与结合能分解

`data/inputs/` 提供冻结的咖啡因、水和 14 个匿名复合物候选。仅使用公开工具和本题计算预算，确定可信的低能结合构型，量化 interaction 与 deformation 对 binding energy 的贡献，判断是否存在近简并竞争构型，并说明排序的不确定性。

固定物理条件：xTB 6.7.1、气相 GFN2-xTB，复合物与两个片段均为 charge 0、UHF 0，原子顺序不变，不使用 Counterpoise，单线程。片段原子归属以 `fragment_manifest.csv` 为准。采用 frozen-fragment 定义：复合物几何中的两个片段能量与各自孤立平衡几何能量共同确定 interaction、deformation 和 binding energy。

能量约定如下。原始 xTB 总能量以 Hartree 读取，所有报告的能量差以 `kcal/mol` 给出。对于某个优化后的复合物几何 `AB`：

```text
E_int = E_AB - E_A^AB - E_B^AB
E_def = (E_A^AB - E_A^opt) + (E_B^AB - E_B^opt)
E_bind = E_int + E_def
closure residual = E_bind - E_int - E_def
```

其中 `E_A^AB` 与 `E_B^AB` 是从该复合物优化几何逐坐标抽取的冻结片段单点能，`E_A^opt` 与 `E_B^opt` 是各自孤立平衡几何能。稳定相互作用对应 negative E_int；按照此定义，几何形变代价应为 non-negative E_def。`E_bind < 0` 表示相对于两个孤立优化单体的稳定结合。每个保留构型都必须同时报告三个分解量与 closure residual；四舍五入后的绝对闭合残差不得超过 `0.001 kcal/mol`。这一阈值只检查公式和报告精度，不是与私有参考答案比较的准确度容差。

你需要自行规划如何覆盖构型多样性、避免重复计数、分配复合物与片段计算、验证能量闭合、处理异常结果，并判断排名是否稳定。题目不规定筛选、优化、Hessian、保留构型数量、聚类或不确定性分析的固定路线。

模型只需写入 `submission/final_answer.json` 和 `submission/report.md`；Harness 自动生成 manifest 并收集原始 xTB job。允许部分答案，预算用尽本身不会把整题清零。

预算：最多 16 个 xTB job、65 个规划/执行回合、260 次工具调用、9,000 秒科学执行时间；科学预算用尽后另有最多 4 个 finalization 回合和 1800 秒。候选数多于可完整优化和分解的数量，因此需要基于可观察证据自主取舍；预算耗尽不会把已有可信结果整题清零。

评分细节：至少一个可信 xTB job 是硬性前提——若运行证据中不存在任何可信 xTB job，无论提交内容如何，整题计 0 分。

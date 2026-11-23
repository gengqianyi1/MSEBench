# Task 06 审核闭环与独立交付验收

| 原问题 | 实际处理及证据 |
|---|---|
| 坐标、basin、方向势垒定义含糊 | 冻结题面已给操作性定义、周期角距离、单位及匹配语义。 |
| 停止及覆盖要求不清楚 | 冻结公开协议已给预算、覆盖判据、停止与部分报告格式；预算耗尽不等于科学整题零分。 |
| negative CI 不完整 | 保留重复/周期等价点、非法网格、缺覆盖、能量不一致和伪造frame哈希测试；原全套验证日志随包。 |
| 独立运行与评分 | 空存储加载随包环境，Oracle100、科学不完整 smoke2、错误格式0均正常完成，canonical replay一致。未更改科学权重或历史模型分数。 |

## 共同验收

本次在四套独立Docker/containerd存储中分别验证，各自初始镜像数0。仅加载本题随包tar；组织者来自公开依赖新建镜像，没有作者venv/可编辑安装挂载。load、check、Oracle、smoke、invalid五步骤退出码均0。分数、四字段及canonical replay见verification/ACCEPTANCE.json。

模型容器只获得本题public/workspace；私有答案、Grader、组织者证据、Docker socket只属于可信组织者侧。包内不含.env或实际API Key，不会读取作者本机密钥路径。没有改动05/06/18的冻结科学源码，也没有增加科学先验。

本次没有新的付费LLM实验；历史真实运行与scripted smoke严格分开。新模型API兼容性、余额、远端故障仍取决于接收方供应商，不能保证任意远端服务必定成功。

原审阅中的Linux能力差异由明确Linux/WSL支持范围处理，不宣称原生Windows能运行xTB sandbox。Windows ZIP提取与文件校验另在总交付验证中记录。

已有源代码全套Linux146项测试通过的日志保留（其原验证日期不改）；本次新打包逻辑6项测试通过。regression_tests是额外开发回归源码，不是LLM启动依赖；运行这些pytest需在Python3.11测试环境安装pytest及任务环境锁列出的公开依赖。

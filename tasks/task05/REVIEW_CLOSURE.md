# Task 05 审核闭环与独立交付验收

| 原问题 | 实际处理及证据 |
|---|---|
| 能量符号、单位及闭合不清楚 | 冻结 question.md 已明确三能量定义、Hartree与kcal/mol、符号及0.001kcal/mol报告闭合精度；私有参考未公开。 |
| negative CI 不完整 | 保留缺失/重复 job、原子顺序、闭合错误、预算超限、伪造哈希测试；原全套验证日志和本题 adapter 测试随包。 |
| 跨题冗余及本机依赖 | tasks/ 仅本题；共用框架位于根目录并完整随包。没有 Task04/06 私有输入/答案/运行目录；共享代码的通用分支不构成跨题资产依赖。 |
| 独立运行与评分 | 空存储加载随包环境，Oracle100、科学不完整 smoke2、错误格式0均由 Harbor 正常完成，canonical replay一致。 |

## 共同验收

本次在四套独立Docker/containerd存储中分别验证，各自初始镜像数0。仅加载本题随包tar；组织者来自公开依赖新建镜像，没有作者venv/可编辑安装挂载。load、check、Oracle、smoke、invalid五步骤退出码均0。分数、四字段及canonical replay见verification/ACCEPTANCE.json。

模型容器只获得本题public/workspace；私有答案、Grader、组织者证据、Docker socket只属于可信组织者侧。包内不含.env或实际API Key，不会读取作者本机密钥路径。没有改动05/06/18的冻结科学源码，也没有增加科学先验。

本次没有新的付费LLM实验；历史真实运行与scripted smoke严格分开。新模型API兼容性、余额、远端故障仍取决于接收方供应商，不能保证任意远端服务必定成功。

原审阅中的Linux能力差异由明确Linux/WSL支持范围处理，不宣称原生Windows能运行xTB sandbox。Windows ZIP提取与文件校验另在总交付验证中记录。

已有源代码全套Linux146项测试通过的日志保留（其原验证日期不改）；本次新打包逻辑6项测试通过。regression_tests是额外开发回归源码，不是LLM启动依赖；运行这些pytest需在Python3.11测试环境安装pytest及任务环境锁列出的公开依赖。

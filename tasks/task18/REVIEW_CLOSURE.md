# Task 18 审核闭环与独立交付验收

| 原问题 | 实际处理及证据 |
|---|---|
| 公开rubric/hard-zero说明 | 冻结question.md已明确无答案权重及整题零分边界；非完美科学结果不自动归零。 |
| 提交路径不统一 | 冻结协议明确 /workspace/submission/ 与 Harbor artifact映射；本次完整sidecar链验证。 |
| 11400与10800秒不一致 | 已公开解释10800秒科学阶段加600秒finalization。 |
| live lifecycle陈旧字段 | 冻结公开记录已为true，真实三模型Harbor日志随包；当前离线分发以CURRENT_DEPLOYMENT.json及实际归档校验为准。 |
| 三模型重复次数解释 | 保留DeepSeek三次、Kimi/GLM各一次原始结果；不声称单次Kimi/GLM形成稳定排名。 |
| 镜像/统一框架接入 | 完整Harbor0.21.0及插件源码、组织者环境、私有sidecar/verifier全部随包；独立空存储Oracle100、smoke0（仅一个SP、科学不完整）、错误提交0均正常完成并replay一致。smoke0不是模型成绩。 |

## 共同验收

本次在四套独立Docker/containerd存储中分别验证，各自初始镜像数0。仅加载本题随包tar；组织者来自公开依赖新建镜像，没有作者venv/可编辑安装挂载。load、check、Oracle、smoke、invalid五步骤退出码均0。分数、四字段及canonical replay见verification/ACCEPTANCE.json。

模型容器只获得本题public/workspace；私有答案、Grader、组织者证据、Docker socket只属于可信组织者侧。包内不含.env或实际API Key，不会读取作者本机密钥路径。没有改动05/06/18的冻结科学源码，也没有增加科学先验。

本次没有新的付费LLM实验；历史真实运行与scripted smoke严格分开。新模型API兼容性、余额、远端故障仍取决于接收方供应商，不能保证任意远端服务必定成功。

原审阅中的Linux能力差异由明确Linux/WSL支持范围处理，不宣称原生Windows能运行xTB sandbox。Windows ZIP提取与文件校验另在总交付验证中记录。

已有源代码全套Linux146项测试通过的日志保留（其原验证日期不改）；本次新打包逻辑6项测试通过。regression_tests是额外开发回归源码，不是LLM启动依赖；运行这些pytest需在Python3.11测试环境安装pytest及任务环境锁列出的公开依赖。

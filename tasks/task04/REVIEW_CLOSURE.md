# Task 04 审核闭环与独立交付验收

| 原问题 | 实际处理及证据 |
|---|---|
| 真实 Harbor LLM 链路缺失 | verification/real_model_regression 保存真实 Flash 运行：47 turns、90 tools、53/56 credits，提交及评分完成。请求 deepseek-v4-flash，实际服务器名 deepseek-flash。本次未新增模型调用。 |
| 原生工具调用/响应截断 | 当前 WorkspaceSidecarAgent 使用 native function calling，截断不会执行不完整工具参数；本包默认 65536 tokens、900s 请求超时，未扩大科学预算。 |
| XYZ 六位小数被误拒 | Grader2.1.1 对提交 XYZ 采用1e-6 Å，可信 opt→Hessian 原始几何仍1e-8 Å；同一原始证据重新评分44.8203998344，四分数字段相等，未改答案、参考或权重。 |
| 镜像未随包交付 | docker/images.docker.tar 为实际资产；本次独立 Docker+containerd 空存储 load/check、Oracle100及完整链路均通过。 |
| 公开镜像历史发布标签与源码不同 | 仅 harness_contract.yaml 的旧 release_eligible 及 sidecar_deployment.yaml 的历史发布记录有差异；不涉及工具、预算、输入、Schema、科学题面或评分代码。冻结记录保留不伪造原镜像；本包部署唯一有效依据为 CURRENT_DEPLOYMENT.json、deploy/task04/images.lock.json、docker/images.sha256。旧 archive_sha256/release_ready/live_llm 字段不代表本包。 |

## 共同验收

本次在四套独立Docker/containerd存储中分别验证，各自初始镜像数0。仅加载本题随包tar；组织者来自公开依赖新建镜像，没有作者venv/可编辑安装挂载。load、check、Oracle、smoke、invalid五步骤退出码均0。分数、四字段及canonical replay见verification/ACCEPTANCE.json。

模型容器只获得本题public/workspace；私有答案、Grader、组织者证据、Docker socket只属于可信组织者侧。包内不含.env或实际API Key，不会读取作者本机密钥路径。没有改动05/06/18的冻结科学源码，也没有增加科学先验。

本次没有新的付费LLM实验；历史真实运行与scripted smoke严格分开。新模型API兼容性、余额、远端故障仍取决于接收方供应商，不能保证任意远端服务必定成功。

原审阅中的Linux能力差异由明确Linux/WSL支持范围处理，不宣称原生Windows能运行xTB sandbox。Windows ZIP提取与文件校验另在总交付验证中记录。

已有源代码全套Linux146项测试通过的日志保留（其原验证日期不改）；本次新打包逻辑6项测试通过。regression_tests是额外开发回归源码，不是LLM启动依赖；运行这些pytest需在Python3.11测试环境安装pytest及任务环境锁列出的公开依赖。

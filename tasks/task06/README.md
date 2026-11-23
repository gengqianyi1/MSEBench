# MSEBench Task 06

本目录是 MSEBench 0.2.0 组装工作目录中的任务模块。完整代码与 Hugging Face 资源组装后，包含题面、输入、评分实现、测试标签、参考解和固定评分样例。预构建镜像按根目录 images/ 中的逐文件清单分发。

## 文件与依赖

- [题目定义](tasks/task-06-terpyridine-coupled-torsion-landscape/instruction.md)
- [任务契约](tasks/task-06-terpyridine-coupled-torsion-landscape/harness_contract.yaml)
- [镜像清单](../../images/task06/images.yaml)
- [统一运行指南](../../docs/RUN_GUIDE.md)
- [安装与组装](../../DEPLOY.md)

本题使用 [统一引擎](../../src/msebench/orchestrator.py) 和 [共享协议](../../src/msebench_harness/__init__.py)。共享源码位于整合工作目录，评分代码和快照按 task.json 的 task_dir 定位。运行前按部署说明组装完整工作目录，并加载本题和共享镜像。

## 运行

以下命令从组装工作目录的根目录执行。模型配置使用 default profile，以及用户的 MSEBENCH_API_BASE、MSEBENCH_MODEL_ID、MSEBENCH_API_KEY 环境变量。

```bash
bash load_images.sh --task 6
msebench --workspace "$PWD" --task 6 --verify
msebench --workspace "$PWD" --task 6
```

--verify 的检查随题族变化。完整模型与 verifier 链路需要 Linux amd64 Docker 环境。输出默认位于根目录 runs/ 和 results/，可由 MSEBENCH_RUNS_ROOT、MSEBENCH_RESULTS_ROOT 指定。

## 评分与验收状态

保留完整提交、工具执行证据和调用计数记录，重新评分命令见统一运行指南。Agent 与评分材料仍按任务契约隔离，用户可以在宿主机查看公开的评分源码和参考数据。

0.2.0 为待 Linux Docker 端到端验收的候选。历史独立包的测试记录对应其原始版本，当前验收以根目录 [发布清单](../../RELEASE_CHECKLIST.md) 为准。来源和适用许可见 [逐题溯源表](../../docs/DATA_PROVENANCE.md) 与任务内第三方声明。

本题的 run.sh 支持 load、smoke、oracle、invalid、model。其 load 分支直接调用 Docker 加载任务镜像；带归档校验的入口是根目录 load_images.sh。原有提交重新评分需要对应的 organizer_evidence。部署依赖和构建待办记录在 [CURRENT_DEPLOYMENT.json](CURRENT_DEPLOYMENT.json)。

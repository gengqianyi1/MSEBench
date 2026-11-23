# MSEBench Task 14–17 共享框架

本目录提供 MSEBench 0.2.0 中 Task 14–17 的 Agent 适配、协议、sidecar 模板和辅助脚本。实际任务在根目录 tasks/task14 至 tasks/task17，组装程序恢复本目录 tasks/ 下的相对链接。

- [当前运行说明](RUN.md)
- [统一部署](../../DEPLOY.md)
- [完整运行指南](../../docs/RUN_GUIDE.md)
- [镜像构建配方](docker_build/README.md)
- [当前验收记录](../../RELEASE_CHECKLIST.md)

模型、数据和镜像先按统一部署说明准备。全部命令从组装工作目录根部执行。

```bash
bash load_images.sh
msebench --workspace "$PWD" --task 14,15,16,17 --verify
msebench --workspace "$PWD" --task 14,15,16,17
```

--verify 使用该题族的 llm-mock 路径，完整模型评分另行执行。运行产物写入 frameworks/14-17/jobs/，统一入口将结果归集至根目录 results/。Task 16、17 的 sidecar 由各题 environment/docker-compose.yaml 配置。

2026 年 9 月的 revision8–10 标识属于历史迁移版本。当前候选的完整 Linux Docker、真实模型与重建等价性验收记录在根目录发布清单。当前分发采用 images/ 下的逐任务 tar 和锁文件。

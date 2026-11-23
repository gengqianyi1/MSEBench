# MSEBench 0.2.0 Task 14–17 运行步骤

目标环境为 Linux amd64 Docker、Compose v2 和 Python 3.12，宿主依赖按根目录安装说明准备。以下命令均从组装工作目录根部执行。

1. 按 [部署说明](../../DEPLOY.md) 下载并组装同版本代码和资源。
2. 校验并加载镜像。

```bash
PYTHON=python bash verify_package.sh
bash load_images.sh
```

3. 配置用户自己的模型接口。

```bash
export MSEBENCH_API_BASE="https://your-provider.example/v1"
export MSEBENCH_MODEL_ID="your-model"
export MSEBENCH_API_KEY=""
msebench --workspace "$PWD" --task 14,15,16,17 --verify
msebench --workspace "$PWD" --task 14,15,16,17
```

模型生成访问用户配置的接口。任务环境的网络与工具预算由任务契约规定。Task 16、17 的 sidecar 由 Compose 启动。Docker daemon 必须能以相同绝对路径访问组装目录。

4. 检查 frameworks/14-17/jobs/ 的原始产物和根目录 results/ 的汇总。已有提交重评分见 [运行指南](../../docs/RUN_GUIDE.md)。

5. 从源码重建时使用 [docker_build 配方](docker_build/README.md)。重建需要访问软件源，并应记录新镜像 ID 与评分等价性结果。

当前 0.2.0 候选的完整 Linux Docker 验收仍待完成。[发布清单](../../RELEASE_CHECKLIST.md) 区分静态检查、参考样例回归和容器端到端验证。镜像实际大小以 data_manifest.json 的 bytes 字段为准。

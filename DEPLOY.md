# MSEBench 本地部署与验收

目标环境为 Linux amd64、Python 3.12、Docker Engine、Compose v2。Harbor 宿主依赖随 local extra 安装。所有评测容器使用同一个 Docker daemon。科学任务资源需求依题目而定，Task 01 建议 16 核、32 GB 内存，Task 02 建议 32 核、32 GB 内存。

在 GitHub 代码候选目录中执行，数据版本使用完整 HF 提交 SHA。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[local]'
python scripts/prepare_local.py --repo OWNER/DATASET --revision FULL_DATASET_COMMIT_SHA \
  --images --output ../msebench-workspace
cd ../msebench-workspace
PYTHON=python bash verify_package.sh
bash load_images.sh
python scripts/check_build_plan.py
```

上传前可以用 --source /path/to/huggingface 读取本地候选。按题下载使用 --task 4,5，省略 --images 则使用已经加载的镜像。下载器在临时目录校验资源后再发布工作目录，并重建包内相对链接。

预构建镜像是本版本的评分参考，镜像锁保存准确 ID 和归档 SHA256。重建配方随源码发布，重建后必须重新验收并更新锁，才能采用新镜像 ID。Task 04、05、06、18 的 sidecar 构建先准备对应 verifier，然后在工作目录根部执行对应命令。

```bash
docker build -t msebench/task04-sidecar:local -f tasks/task04/environment/Dockerfile.sidecar .
```

Task 14–17 的完整重建入口是 frameworks/14-17/docker_build/build.sh。Task 19、20 的配方在任务目录和镜像清单中。重建基础环境需要访问公开软件源。版本锁与可用源码的 COPY 依赖必须在组装后核对。

本地模型与重新评分命令见 [运行指南](docs/RUN_GUIDE.md)。目标主机验收应记录全部 20 道题的正常提交、非法提交、镜像 ID、退出码和评分。静态检查与镜像 SHA256 检查各有独立结果。

## 组织者运行镜像待准备项

Task 01–20 中部分题族通过组织者容器调度用户本地的 Docker。当前归档锁没有声明 `chembench/organizer:harbor-0.21.0-20260913` 标签。Task 04、05、06、18 的 CURRENT_DEPLOYMENT.json 已将其列为构建和运行待验收依赖。在 Linux 主机上可按以下源码配方准备该标签，再验证依赖版本和实际评测链路。

```bash
docker build -t chembench/organizer:harbor-0.21.0-20260913 \
  -f tasks/task04/environment/Dockerfile.rebuild tasks/task04/environment
```

该步骤需要公开软件源。构建后的镜像 ID、部署方式与评分结果应记入发布验收记录。当前候选的完整离线部署验收包含此待办。

# 题 14–17（revision10）Docker 环境：从头构建

## MSEBench 整合版位置

此目录位于 `MSEBench_release/frameworks/14-17/docker_build/`。直接运行 `bash build.sh`；默认题包为上一级 `frameworks/14-17`，其 `tasks/` 符号链接指向发布包根目录的 `tasks/task14`–`task17`。下面提及的原作者独立 zip 路径属于历史说明，整合版部署请以根目录 `DEPLOY.md` 为准。

对应题包：`14_17_曹阳_revision10_20260916.zip`。

题包本身随附预构建的离线镜像（`docker/chembench_14_17_images_linux-amd64.tar.gz`，用 `docker/load_images.sh` 载入）。
本目录提供**不依赖该离线包、从基础镜像开始重新构建**同一组镜像的配方：一个 compose 文件 + 四个 Dockerfile。
构建出的镜像名与题包 `task.toml` 等处的引用逐字一致，构建完成后按 `RUN.md` 第 2 步起照常运行即可。

## 目录

| 文件 | 作用 |
|---|---|
| `docker-compose.build.yaml` | 12 个镜像的构建清单（镜像名、配方、构建上下文、构建参数） |
| `dockerfiles/task-environment.Dockerfile` | 4 个 agent 环境镜像（= 题包 `tasks/<题>/environment/Dockerfile`） |
| `dockerfiles/task-verifier.Dockerfile` | 4 个 verifier 镜像（= 题包 `tasks/<题>/tests/Dockerfile`） |
| `dockerfiles/sidecar.Dockerfile` | 题 16/17 私有 response sidecar（= 题包 `sidecar/Dockerfile`） |
| `dockerfiles/runner.Dockerfile` | Harbor 运行器（= 题包 `docker/runner/Dockerfile`） |
| `egress-control-sidecar/` | Harbor 0.21.0 自带的断网控制 sidecar 配方（Apache-2.0，原样拷贝，仅基础镜像可替换） |
| `runner-context/requirements.txt` | 运行器的 Python 依赖（题包根目录 `requirements.txt` 的副本，`build.sh` 每次自动同步） |
| `build.sh` | 一键：构建 → 让题包认可本地镜像 → 校验 |
| `adopt_local_build.py` | `build.sh` 的第二步（见下文"为什么需要 adopt"） |
| `.env.example` | 镜像源 / 平台 / 题包路径等可选参数 |

四个 Dockerfile 与题包内原配方逐行对应，区别只有一处：基础镜像、conda channel、PyPI、apt 源可以通过构建参数替换。
不传参数时安装的内容与随包镜像完全相同（版本全部锁定在各题的 `environment.lock.yml` 里）。

## 构建出的镜像（12 个）

| 角色 | 镜像 | 基础镜像 / 依赖 |
|---|---|---|
| agent 环境 ×4 | `chembench/task{14,15,16,17}-environment:2.1.0-r10` | `mambaorg/micromamba:2.8.1`；python 3.10.19、numpy 2.2.6、pyyaml 6.0.3、openai 2.15.0；14/15 另有 rdkit 2025.9.6 |
| verifier ×4 | `chembench/task{14,15,16,17}-verifier:2.1.0-r10` | 同上 + 评分脚本 + `task_snapshot`（**私有**） |
| sidecar ×2 | `chembench/task{16,17}-sidecar:2.1.0-r10` | 同 verifier 环境 + `private_response_server.py` + `task_snapshot`（**私有**） |
| 断网控制 | `harbor-prebuilt:harbor-docker-egress-control-sidecar--b1e6760734a9e6f6` | `gogost/gost:3.2.7-nightly.20260602`（按 digest 锁定） |
| runner | `chembench/harbor-runner:2.1.0-r10` | `python:3.12-slim-bookworm` + docker-ce-cli/compose/buildx + harbor 0.21.0 等 |

verifier 与 sidecar 镜像内含参考最优解与分档阈值，**只供组织方使用，切勿推送到公开 registry，也不可用作 agent 镜像。**

## 前提

- Docker Engine ≥ 23、Docker Compose v2、buildx 插件（BuildKit）。`additional_contexts` 依赖 BuildKit。
- 构建时需要能访问：Docker Hub、conda-forge、PyPI、deb.debian.org、download.docker.com；不通的部分在 `.env` 里换成镜像源（见下）。
- 题包已解压到与本目录并列的 `14_17_曹阳_revision10_20260916/`（否则设置 `CHEMBENCH_PKG`）。
- 题包与本目录所在路径要能被 Docker daemon 以**同一绝对路径**看到（与 `RUN.md` 第 1 节"路径要求"相同）。
- 目标平台：组织方主机为 `linux/amd64`。`build.sh` 默认按本机 daemon 平台构建。

## 用法

```bash
cd 第五轮交付
unzip -q 14_17_曹阳_revision10_20260916.zip        # 已解压可跳过
cd 14_17_revision10_docker_build
cp .env.example .env                                 # 可选：需要镜像源时编辑
./build.sh
```

最后一行应为 `status = PASS`。之后进入题包，从 `RUN.md` 第 2 步（进入 runner 容器）继续，**跳过第 1 步的 `load_images.sh`**。

只想用 compose 本身构建（不做 adopt）：

```bash
docker compose -f docker-compose.build.yaml build                      # 全部
docker compose -f docker-compose.build.yaml build task16-sidecar       # 单个服务
```

这个 compose 文件只用于 `build`，服务是镜像配方，不需要 `up`。

### 镜像源（Docker Hub / conda-forge / PyPI 不通时）

在 `.env` 里设置（全部可选，留空即官方源）：

| 变量 | 作用 | 示例 |
|---|---|---|
| `MICROMAMBA_IMAGE` `PYTHON_IMAGE` `GOST_IMAGE` | 基础镜像改为内网仓库地址 | `registry.example.com/mambaorg/micromamba:2.8.1` |
| `CONDA_CHANNEL` | 替换锁文件里唯一的 channel `conda-forge` | `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge` |
| `PIP_INDEX_URL` | pip 源（任务镜像里的 openai、runner 里的 harbor） | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| `DEBIAN_MIRROR` | runner 的 Debian apt 源 | `https://mirrors.tuna.tsinghua.edu.cn` |
| `DOCKER_APT_REPO` | runner 的 Docker CE apt 源 | `https://mirrors.aliyun.com/docker-ce/linux/debian` |

基础镜像也可以不改变量，直接在 Docker daemon 的 `daemon.json` 里配置 `registry-mirrors`。
需要走 HTTP 代理时，`HTTP_PROXY` / `HTTPS_PROXY` 作为 Docker 预定义构建参数会自动生效。

## 为什么需要 adopt

题包的运行脚本（`./chembench oracle|llm|llm-mock|preflight`）开跑前会按 `docker/image_manifest.json`
逐个核对镜像的**层摘要**，用来拦截上一版残留的旧镜像。重新构建的镜像装的是同一组锁定版本，
但层摘要必然不同（时间戳、基础镜像重新拉取），不处理的话每次运行都会报
`local layers differ from the shipped image` 并退出。

`adopt_local_build.py`（由 `build.sh` 在刚构建的 runner 镜像里执行）做三件事：

1. 确认 12 个镜像都在、平台与本机 daemon 一致；
2. 若本机平台不是 linux/amd64，把断网控制镜像补打成 Harbor 在本平台实际查找的名字；
3. 把原清单备份为 `docker/image_manifest.shipped.json`，再把 `docker/image_manifest.json` 的 `images[]`
   改写为本地镜像的层摘要，最后运行题包自己的 `offline_images.py check --all-roles`。

改回随包离线镜像：

```bash
cd ..
mv docker/image_manifest.shipped.json docker/image_manifest.json
docker/load_images.sh
```

## 验证记录（2026-10-07，作者本机 Docker Desktop 29.1.3 / Compose v2.40.3 / BuildKit 0.26.2）

为了不改动交付的题包，测试时用的是题包副本（`CHEMBENCH_PKG` 指向副本）。

| 项 | 结果 |
|---|---|
| `./build.sh`（原生 linux/arm64，12 个镜像） | 全部构建成功，用时 3 min 47 s；adopt 后 `offline_images.py check --all-roles` → `status = PASS` |
| `CHEMBENCH_PLATFORM=linux/amd64 ./build.sh`（组织方平台，模拟执行） | 12 个镜像全部构建成功，用时 3 min 20 s；conda 解析 `conda-forge/linux-64`，镜像平台 linux/amd64，x86_64 下 `import rdkit` 正常 |
| 镜像内容 | task14 环境：python 3.10.19 / rdkit 2025.09.6 / numpy 2.2.6 / openai 2.15.0，与锁文件一致；task15 环境 `/workspace` 与题包 `environment/public/` 逐文件 sha256 相同；verifier 的 `/tests`、`/task` 内容完整；runner 的 harbor 0.21.0、Python 3.12，pip 依赖版本与随包 runner 相同 |
| 用重建镜像跑 Harbor Oracle（`--local-public-network`，Docker Desktop 不支持强制断网） | 14：0.47 / 47.0；15：0.33 / 33.0；16：1.00 / 100.0；17：1.00 / 100.0，均为 COMPLETE。与 `RUN.md` 第 4.1 节期望值完全一致 |
| `./chembench llm-mock task-16 …`（真实 agent + 私有 sidecar + 模拟模型） | `status = PASS` |
| 镜像源 | `CONDA_CHANNEL`、`PIP_INDEX_URL`（清华）、`DEBIAN_MIRROR`（清华）、`DOCKER_APT_REPO`（阿里云）均在构建日志里确认生效，构建成功 |

没有验证的部分：在一台原生 linux/amd64 主机上跑完整构建，以及基础镜像完全不在本地缓存时的首次拉取（本机已有 `mambaorg/micromamba:2.8.1` 和 `python:3.12-slim-bookworm`）。

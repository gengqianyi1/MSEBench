# 题 16/17 私有 response sidecar：镜像、接线与 digest

这份东西回答的是交接文件里那条"请补正式私有 sidecar 镜像 digest"。**digest 是这条链的第 5 步，
本包补齐的是此前不存在的第 1–3 步**，第 4 步受宿主内核限制、第 5 步需要框架方先定归属。

## 一、为什么 16/17 需要 sidecar

`environment/public/.chembench_agent.json` 声明 `protocol_mode: private_response_round`。
`agents/chembench_agent.py` 的该分支每轮都要 POST `/session` 与 `/evaluate` 拿测量值：

```python
if not sidecar.get('session_id'):
    started = self._post_json(url + '/session', {})   # 没有 try/except
```

**没有回退路径**：端点不存在就直接抛异常。所以在补上本目录之前，题 16/17 只有 Oracle 路径
（`-a oracle`，重放固定 candidate、从不调用 sidecar）能跑，**真实模型评测根本无法开始**。

## 二、此前缺什么

复核 v2.0 的结果：

| 环节 | 交付前状态 |
|---|---|
| ① sidecar 镜像定义 | 全仓库**没有任何** sidecar Dockerfile |
| ② Harbor 服务声明 | 全仓库**没有任何** docker-compose 文件；`task.toml` 无 service |
| ③ URL 注入 | 没有任何地方设置 `CHEMBENCH_TASK16/17_SIDECAR_URL` |
| ④ `no-network` 下可达性 | 未验证 |
| ⑤ 不可变 digest | `AUTHOR_REQUIRED_SHA256` |

框架方自己在 `TASK04_06_AUTHOR_ACTION.md` 的 "Required deliverable B — reproducible Harbor
wiring" 里一字不差地要求过 ①–④，只是没有对 07/16/17 提。另需说明：01/02 的
"Docker sidecar E2E" 证据来自 `scripts/e2e_task01_task02_mock_llm.py`，它在**宿主**上
`docker run -d` 起 sidecar、用 `FakeEnvironment` 和脚本化假模型、把 URL 设在宿主进程环境里。
它证明协议可用，**不证明 Harbor 能把 sidecar 接到容器化 agent 上**——本仓库此前没有这个先例。

## 三、本包补的内容

| 文件 | 作用 |
|---|---|
| `sidecar/Dockerfile` | 框架模板 server + 该题 `tests/task_snapshot/`，依赖锁沿用该题 `tests/environment.lock.yml` |
| `sidecar/build_sidecar.sh` | 从包根构建并打印待冻结的 digest 与平台 |
| `tasks/task-16.../environment/docker-compose.yaml` | 声明 `task16-sidecar` 服务并向 `main` 注入 URL |
| `tasks/task-17.../environment/docker-compose.yaml` | 同上（8717） |

**构建上下文是包根**：server 是框架资产（`templates/sidecar/private_response_server.py`，
两题共用、不含模型系数），数据是题目资产，两者不在同一子树。

**网络**：`network_mode: "service:main"` 让 sidecar 进入 agent 容器的网络命名空间，因此
`127.0.0.1:8716/8717` 恰好就是 `.chembench_agent.json` 里已有的默认值，两种配置下都成立——
开启 Harbor egress control 时 `main` 本身在 egress 命名空间里，sidecar 跟随进入；未开启时
`main` 是普通命名空间，sidecar 加入它。sidecar 因此也继承 main 的出网策略，`no-network` 下同样出不了网。

**这个镜像含 `private_data/`（枚举参考最优解与分档阈值），绝不可推送到公开 registry，
也绝不可作为 agent 环境的基础镜像。**

**镜像随包离线提供，运行期不构建、不拉取**：compose 服务写的是 `image: chembench/task1X-sidecar:2.1.0-r10`
与 `pull_policy: never`，镜像由 `docker/load_images.sh` 从包内离线镜像包载入。项目方主机不能访问
Docker Hub（2026-09-14 实际报错 `Get "https://registry-1.docker.io/v2/": context deadline exceeded`），
而本 Dockerfile 基于 Docker Hub 上的 `mambaorg/micromamba` 并从 conda-forge 装依赖，运行时构建在那台
主机上必然失败。`build:` 段保留为配方：镜像缺失且主机能联网时 compose 仍会就地构建
（`pull_policy: never` + `build:` 的这一行为已实测）。

> **一个必须知道的坑**：`build.context` 的相对路径以 **project directory** 为基准，
> 不是以 compose 文件所在目录为基准（Harbor 源码注释里的说法在这一点上会误导人）。
> Harbor 调用时传 `--project-directory <task>/environment`，所以 `../../..` 正好是包根。
> 手工 `docker compose` 若不传该参数，会报 `lstat .../sidecar: no such file or directory`。
> 已实测：带该参数时 `docker compose ... build task17-sidecar` 输出 `Built`；不带则失败。

## 四、digest 怎么定：两条必须先说清的规则

1. **必须是评测主机架构的 digest。** 同一份 Dockerfile 在 arm64 与 x86_64 上构建出的镜像
   digest 不同。用 `PLATFORM=linux/amd64 ./sidecar/build_sidecar.sh 16` 对齐评测机。
2. **本地 `docker build` 只产生 image ID，没有 repo digest。** repo digest 要等镜像推送到
   registry 之后才存在。所以在框架方决定这些镜像放哪个私有 registry 之前，能冻结的只有
   image ID（可复现锚点），不是分发地址。

因此 `release.private_image_digest` 现在仍是 `AUTHOR_REQUIRED_SHA256`：**不是因为没构建，
而是因为在归属和 registry 确定之前，写进去的任何值都不是它该有的东西。**
框架方给出 registry 与目标架构后，跑一次 `build_sidecar.sh` 即可填入并把
`release_ready` 置 `true`。

**revision9 起的分发方式是离线镜像包**（`docker/`，linux/amd64）。在私有 registry 确定之前，镜像身份以
`docker/image_manifest.json` 中的层摘要 `layers_digest` 为准，`docker/load_images.sh` 载入后自动逐个核对；
它在 classic 与 containerd 两种镜像存储下取值相同，image ID 则不同。

## 五、验证

```bash
docker/load_images.sh                 # 载入离线镜像包并核对
./chembench sidecar-wiring-check       # 从载入的镜像起 main + sidecar；缺镜像时报错，--build 仅用于联网主机
```

`task16_17_sidecar_wiring_check.py` 用 compose 起 `main` + sidecar，**从 agent 容器内部**
用 agent 配置里声明的那个 URL 依次打 `/health`、`/session`、`/evaluate`，并检查响应里
不含 `reference` / `score` / `achievement` 等禁用键。它测的是"接线通不通"，不需要模型。

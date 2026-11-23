# 环境配置说明 · Task 01 · tests (verifier)

本文件描述 Task 01 评测镜像 `harbor-migration/task01:verifier` 的构建与环境配置。
该镜像同时充当 hidden-suite sidecar / grader（见 `environment/public/sidecar_deployment.yaml`）。
内容依据本目录实际文件与 `images/task01/images.lock.json`、`images/task01/verifier.recipe.yaml` 整理。

## 1. 镜像标识

| 项 | 值 |
|---|---|
| tag | `harbor-migration/task01:verifier` |
| image_id | `sha256:ea61a7a0c112acc80c2fc32581a61a37d5daa896a6c763ae9e42b1d6b7d807af` |
| tar | `images/task01/task01-verifier.tar` |
| bytes | `954863616` |
| tar_sha256 | `d57f2601277d0ed5264ad0031531ddca0659ea8db762fc1e470841b85eff8d3b` |
| platform | `linux/amd64` |

## 2. 基础镜像

- `FROM mambaorg/micromamba:2.8.1`
- 离线来源：`images/shared_images/micromamba-2.8.1.tar`

## 3. 构建上下文（本目录）

| 路径 | 用途 | Dockerfile 指令 |
|---|---|---|
| `Dockerfile` | 构建脚本 | — |
| `environment.lock.yml` | conda 锁文件 | `COPY → /tmp/environment.lock.yml` |
| `task_snapshot/` | 题面快照（`/task`） | `COPY → /task/` |
| `run_legacy_grader.py` | 旧版判分器 | `COPY → /tests/` |
| `test_llm_request_accounting.py` | LLM 请求计量测试 | `COPY → /tests/` |
| `private_suite_adapter.py` | hidden-suite 适配器 | `COPY → /tests/` |
| `sidecar.py` | sidecar 服务入口 | `COPY → /tests/` |
| `test.sh` | 评测入口脚本 | `COPY → /tests/`（chmod +x） |
| `private_data/` | 私有数据（oracle 包等） | 由上述脚本按需读取 |

## 4. conda 环境

与 environment 镜像同锁文件 `environment.lock.yml`：`chembench` 环境，`python=3.11.15`、`numpy=2.4.6`、
`pandas=2.3.3`、`scipy=1.17.1`、`scikit-learn=1.9.0`、`pymatgen=2026.5.4`、`matminer=0.10.1`、
`jarvis-tools=2026.4.2`、`spglib=2.7.0`、`jsonschema=4.26.0`、`openai=2.46.0`；
pip 段 `torch==2.13.0+cpu`、`chgnet==0.4.2`。

## 5. 运行时配置（`verifier.recipe.yaml: runtime_config`）

| 项 | 值 |
|---|---|
| User | `mambauser` |
| WorkingDir | `/workspace` |
| Entrypoint | `/usr/local/bin/_entrypoint.sh` |
| Cmd | `bash` |
| PATH | `/opt/conda/envs/chembench/bin:...` |
| 挂载/端口 | 证据卷 `/organizer_evidence`；host `127.0.0.1:8710`（见 `sidecar_deployment.yaml`） |
| 启动命令 | `micromamba run -n chembench python /tests/sidecar.py --adapter /tests/private_suite_adapter.py --snapshot /task --evidence-dir /organizer_evidence --host 0.0.0.0 --port 8710` |

## 6. 构建 / 加载 / 校验

```bash
docker load -i images/shared_images/micromamba-2.8.1.tar

# 从本目录重建
docker build -t harbor-migration/task01:verifier .

# 权威安装（image_id 与 lock 一致）
docker load -i images/task01/task01-verifier.tar

# 冒烟校验
docker run --rm harbor-migration/task01:verifier bash -lc 'ls /tests; ls /task'
```

## 7. 备注

- 该镜像按 `sidecar_deployment.yaml` 的 `private_image_digest` 做完整性绑定；需要 digest 一致时请用 tar 加载。
- 本题 verifier **没有模型 ID 白名单**（`FROZEN_MODEL_IDS` 属 organizer 07–09 族，不在本题）；`private_suite_adapter.py` / `sidecar.py` 仅含工具名与响应字段白名单，换模型无需重建。

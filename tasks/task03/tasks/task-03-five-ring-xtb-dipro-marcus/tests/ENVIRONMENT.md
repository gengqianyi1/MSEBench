# 环境配置说明 · Task 03 · tests (verifier)

本文件描述 Task 03 评测镜像 `harbor-migration/task03:verifier` 的构建与环境配置。
该镜像用于 hidden-suite sidecar 与最终判分（`final_grader.py`）。内容依据本目录实际文件与
`images/task03/images.lock.json`、`images/task03/verifier.recipe.yaml` 整理。

## 1. 镜像标识

| 项 | 值 |
|---|---|
| tag | `harbor-migration/task03:verifier` |
| image_id | `sha256:3bbba0933b85f890252697522f750e2d6156289bf72d715385eadded8049c471` |
| tar | `images/task03/task03-verifier.tar` |
| bytes | `334807552` |
| tar_sha256 | `88aebd216acca6b5b90d28a56ebcd0174228bfaac43912da77c1e4cc58492de9` |
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
| `task03_private_sidecar.py` | 私有 sidecar | `COPY → /tests/` |
| `test_task03_private_sidecar.py` | sidecar 测试 | `COPY → /tests/` |
| `final_grader.py` | 最终判分器 | `COPY → /tests/` |
| `test.sh` | 评测入口脚本 | `COPY → /tests/`（chmod +x） |

## 4. conda 环境

与 environment 镜像同锁文件 `environment.lock.yml`：`chembench` 环境，
`python=3.10.20`、`numpy=2.2.6`、`pyyaml=6.0.3`、`rdkit=2026.03.1`、`xtb=6.7.1=h8876d29_4`、
pip `openai==2.43.0`。

## 5. 运行时配置（`verifier.recipe.yaml: runtime_config`）

| 项 | 值 |
|---|---|
| User | `mambauser` |
| WorkingDir | `/workspace` |
| Entrypoint | `/usr/local/bin/_entrypoint.sh` |
| Cmd | `bash` |
| PATH | `/opt/conda/envs/chembench/bin:...` |
| 判分调用 | `micromamba run -n chembench python /tests/final_grader.py`（挂载 workspace/logs） |

## 6. 构建 / 加载 / 校验

```bash
docker load -i images/shared_images/micromamba-2.8.1.tar

docker build -t harbor-migration/task03:verifier .         # 重建

docker load -i images/task03/task03-verifier.tar           # 权威安装

docker run --rm harbor-migration/task03:verifier bash -lc 'ls /tests; ls /task'
```

## 7. 备注

- 该镜像也作为 `steps/round-*/tests/Dockerfile` 的 `FROM` 基础镜像使用。
- 需要 digest 一致时用 tar 加载（重建为功能等价）。

# 环境配置说明 · Task 02 · tests (verifier)

本文件描述 Task 02 评测镜像 `harbor-migration/task02:verifier` 的构建与环境配置。
该镜像同时充当 hidden-suite sidecar / grader。内容依据本目录实际文件与
`images/task02/images.lock.json`、`images/task02/verifier.recipe.yaml` 整理。

## 1. 镜像标识

| 项 | 值 |
|---|---|
| tag | `harbor-migration/task02:verifier` |
| image_id | `sha256:9754f9f4418c05c47b733bb7d4b3b90015fa0752fc5318561ed858c2e0ae125a` |
| tar | `images/task02/task02-verifier.tar` |
| bytes | `1451123200` |
| tar_sha256 | `159320b0f6ce37bababb45aa519b5e99f455787420bc87c1e485ade8b3cf5b60` |
| platform | `linux/amd64` |

## 2. 基础镜像

- `FROM mambaorg/micromamba:2.8.1`
- 离线来源：`images/shared_images/micromamba-2.8.1.tar`

## 3. 构建上下文（本目录）

| 路径 | 用途 | Dockerfile 指令 |
|---|---|---|
| `Dockerfile` | 构建脚本 | — |
| `environment.lock.yml` | conda 锁文件 | `COPY → /tmp/` |
| `wheels/pyscf-2.8.0-*.whl` | PySCF 离线 wheel（含 sha256 校验） | `COPY → /tmp/` + pip 安装 |
| `task_snapshot/` | 题面快照（`/task`） | `COPY → /task/` |
| `run_legacy_grader.py` | 旧版判分器 | `COPY → /tests/` |
| `private_suite_adapter.py` | hidden-suite 适配器 | `COPY → /tests/` |
| `sidecar.py` | sidecar 服务入口 | `COPY → /tests/` |
| `check_scientific_error_fixtures.py` | 科学错误 fixture 检查 | `COPY → /tests/` |
| `test.sh` | 评测入口脚本 | `COPY → /tests/`（chmod +x） |
| `fixtures/`、`private_data/` | 私有 fixture / oracle | 由脚本按需读取 |

## 4. conda 环境

同 environment 锁文件：`chembench`（`python=3.10.20`、`rdkit=2026.03.4`、`xtb=6.7.1`、`psi4=1.11`、
`numpy=2.2.6`、`scipy=1.15.2`、`h5py=3.16.0`、`pandas=2.3.3`、`openai=2.45.0` 等），
并同样安装 `pyscf==2.8.0` wheel。构建末尾自带校验：
`python -c "import pyscf; from pyscf.solvent import smd; print('PySCF', pyscf.__version__, 'SMD_OK')"`。

## 5. 运行时配置（`verifier.recipe.yaml: runtime_config`）

| 项 | 值 |
|---|---|
| User | `mambauser` |
| WorkingDir | `/workspace` |
| Entrypoint | `/usr/local/bin/_entrypoint.sh` |
| Cmd | `sleep 600` |
| PATH | `/opt/conda/envs/chembench/bin:...` |
| 端口 | sidecar 服务（见 `environment/public/sidecar_deployment.yaml`） |

## 6. 构建 / 加载 / 校验

```bash
docker load -i images/shared_images/micromamba-2.8.1.tar

docker build -t harbor-migration/task02:verifier .         # 重建（需联网 + wheels/）

docker load -i images/task02/task02-verifier.tar           # 权威安装

docker run --rm harbor-migration/task02:verifier bash -lc 'ls /tests'
```

## 7. 备注

- `wheels/` 必须保留。本题 verifier **没有模型 ID 白名单**（`FROZEN_MODEL_IDS` 属 organizer 07–09 族，不在本题）。
- 需要 digest 一致时用 tar 加载（重建为功能等价）。

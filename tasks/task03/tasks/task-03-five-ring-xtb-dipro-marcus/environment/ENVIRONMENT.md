# 环境配置说明 · Task 03 · environment

本文件描述 Task 03 运行镜像 `harbor-migration/task03:environment` 的构建与环境配置。
内容依据本目录实际文件（`Dockerfile` / `environment.lock.yml` / `environment.yaml` / `public/`）
与 `images/task03/images.lock.json`、`images/task03/environment.recipe.yaml` 实测整理。

## 1. 镜像标识

| 项 | 值 |
|---|---|
| tag | `harbor-migration/task03:environment` |
| image_id | `sha256:a38be5ea0f40d38d245b50c32eb8a2c2f798979ed36466a65f2c31780bf610ab` |
| tar | `images/task03/task03-environment.tar` |
| bytes | `334743040` |
| tar_sha256 | `484c667be6e11edbba52029ac8ed4f9c5df5738e95f920d357651860fce4bd28` |
| platform | `linux/amd64` |

## 2. 基础镜像

- `FROM mambaorg/micromamba:2.8.1`
- 离线来源：`images/shared_images/micromamba-2.8.1.tar`

## 3. 构建上下文（本目录）

| 路径 | 用途 | Dockerfile 指令 |
|---|---|---|
| `Dockerfile` | 构建脚本 | — |
| `environment.lock.yml` | conda 锁文件 | `COPY → /tmp/environment.lock.yml` |
| `environment.yaml` | 环境声明（资源/网络/健康检查元数据） | — |
| `public/` | 公开件（题面/协议/输入/rubric 等） | `COPY → /workspace/` |

## 4. conda 环境

- 环境名 `chembench`，channel `conda-forge`
- `python=3.10.20`、`numpy=2.2.6`、`pyyaml=6.0.3`、`rdkit=2026.03.1`、`xtb=6.7.1=h8876d29_4`
- pip 段：`openai==2.43.0`

## 5. 运行时配置（`environment.recipe.yaml: runtime_config`）

| 项 | 值 |
|---|---|
| User | `root` |
| WorkingDir | `/workspace` |
| Entrypoint | `/usr/local/bin/_entrypoint.sh` |
| Cmd | `sleep 300` |
| PATH | `/opt/conda/envs/chembench/bin:...` |
| 其他 Env | `MAMBA_DEFAULT_ENV=chembench` 等 |

## 6. 资源与网络策略（`environment.yaml`）

| 项 | 值 |
|---|---|
| cpus | 4 |
| memory_mb | 8192 |
| storage_mb | 20480 |
| gpus | 0 |
| network_mode | `no-network`（评测模型调用在宿主侧） |
| workdir | `/workspace` |

## 7. 健康检查

```bash
python -c 'import numpy, rdkit' && xtb --version
```

## 8. 构建 / 加载 / 校验

```bash
docker load -i images/shared_images/micromamba-2.8.1.tar

# 从本目录重建（构建态需联网 conda-forge + PyPI）
docker build -t harbor-migration/task03:environment .

# 权威安装
docker load -i images/task03/task03-environment.tar

# 冒烟校验
docker run --rm harbor-migration/task03:environment \
  micromamba run -n chembench python -c "import rdkit,numpy;print('env OK')"
docker run --rm harbor-migration/task03:environment \
  micromamba run -n chembench xtb --version
```

## 9. 备注

- 重建产物 `image_id` 与 lock 不同（功能等价）；需要 digest 一致时用 tar 加载。
- 评测 sidecar/grader 见同题 `tests/ENVIRONMENT.md`（`harbor-migration/task03:verifier`）。

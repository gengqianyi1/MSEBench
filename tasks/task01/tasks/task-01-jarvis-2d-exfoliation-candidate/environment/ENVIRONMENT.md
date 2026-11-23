# 环境配置说明 · Task 01 · environment

本文件描述 Task 01 运行镜像 `harbor-migration/task01:environment` 的构建与环境配置。
内容依据本目录实际文件（`Dockerfile` / `environment.lock.yml` / `environment.yaml` / `public/`）
与 `images/task01/images.lock.json`、`images/task01/environment.recipe.yaml` 实测整理。

## 1. 镜像标识

| 项 | 值 |
|---|---|
| tag | `harbor-migration/task01:environment` |
| image_id | `sha256:a7b3a5226180502d53036680aa0a87c275810e15ee348aa1d78a27756ef59231` |
| tar | `images/task01/task01-environment.tar` |
| bytes | `953507328` |
| tar_sha256 | `636da6b7077c861a2d62c278cc1470e01a8f002db84138d1534b9c6172f255ad` |
| platform | `linux/amd64` |

## 2. 基础镜像

- `FROM mambaorg/micromamba:2.8.1`
- 离线来源：`images/shared_images/micromamba-2.8.1.tar`（tag `mambaorg/micromamba:2.8.1`）

## 3. 构建上下文（本目录）

| 路径 | 用途 | Dockerfile 指令 |
|---|---|---|
| `Dockerfile` | 构建脚本 | — |
| `environment.lock.yml` | conda 锁文件（精确版本） | `COPY → /tmp/environment.lock.yml` |
| `environment.yaml` | 环境声明（资源/网络/健康检查元数据，不进入镜像） | — |
| `public/` | 公开件（题面/协议/输入/工具清单/rubric 等） | `COPY → /workspace/` |

## 4. conda 环境

- 环境名 `chembench`，channel `conda-forge`
- `python=3.11.15`、`numpy=2.4.6`、`pandas=2.3.3`、`scipy=1.17.1`、`scikit-learn=1.9.0`、
  `pymatgen=2026.5.4`、`matminer=0.10.1`、`jarvis-tools=2026.4.2`、`spglib=2.7.0`、
  `pyyaml=6.0.3`、`jsonschema=4.26.0`、`joblib=1.5.3`、`pyarrow=25.0.0`、`openai=2.46.0`、
  `pytest=9.1.1`、`pip=26.1.2`、`setuptools=80.10.2`
- pip 段（`--index-url https://download.pytorch.org/whl/cpu` + `--extra-index-url https://pypi.org/simple`）：
  `torch==2.13.0+cpu`、`chgnet==0.4.2`

## 5. 运行时配置（`environment.recipe.yaml: runtime_config`）

| 项 | 值 |
|---|---|
| User | `root` |
| WorkingDir | `/workspace` |
| Entrypoint | `/usr/local/bin/_entrypoint.sh` |
| Cmd | `bash` |
| PATH | `/opt/conda/envs/chembench/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` |
| 其他 Env | `MAMBA_DEFAULT_ENV=chembench`、`MAMBA_USER=mambauser`、`ENV_NAME=base`、`MAMBA_ROOT_PREFIX=/opt/conda` |

## 6. 资源与网络策略（`environment.yaml`）

| 项 | 值 |
|---|---|
| cpus | 4 |
| memory_mb | 16384 |
| storage_mb | 40960 |
| gpus | 0 |
| network_mode | `no-network`（评测模型调用在宿主侧，任务容器不联网） |
| workdir | `/workspace` |

## 7. 健康检查

```bash
python -c 'import numpy, pandas, sklearn, pymatgen, matminer, jarvis'
```

## 8. 构建 / 加载 / 校验

```bash
# 加载基础镜像（首次）
docker load -i images/shared_images/micromamba-2.8.1.tar

# 从本目录重建（功能等价；构建态需联网 conda-forge / PyTorch / PyPI）
docker build -t harbor-migration/task01:environment .

# 或从 tar 安装（与 lock 的 image_id 完全一致，权威方式）
docker load -i images/task01/task01-environment.tar

# 冒烟校验
docker run --rm harbor-migration/task01:environment \
  micromamba run -n chembench python -c \
  "import numpy,pymatgen,matminer,jarvis,chgnet,torch;print('env OK')"
```

## 9. 备注

- 重建产物 `image_id` 与 lock 不同（构建 history/时间戳差异），属**功能等价**；需要 digest 完全一致时用 tar 加载。
- 构建态需联网；运行态 `no-network`。
- 本目录仅定义题目运行环境；评测 sidecar/grader 见同题 `tests/ENVIRONMENT.md`（`harbor-migration/task01:verifier`）。

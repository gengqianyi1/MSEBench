# 环境配置说明 · Task 02 · environment

本文件描述 Task 02 运行镜像 `harbor-migration/task02:environment` 的构建与环境配置。
内容依据本目录实际文件（`Dockerfile` / `environment.lock.yml` / `environment.yaml` / `public/` / `wheels/`）
与 `images/task02/images.lock.json`、`images/task02/environment.recipe.yaml` 实测整理。

## 1. 镜像标识

| 项 | 值 |
|---|---|
| tag | `harbor-migration/task02:environment` |
| image_id | `sha256:402d8946feec64505a62d1f290ea0f293d6ce5a0373052ed388f84fca9b4c174` |
| tar | `images/task02/task02-environment.tar` |
| bytes | `1451036672` |
| tar_sha256 | `3800410ebfe45937c765d834d0f3af34521a9ff76411245559140a9ce28307e5` |
| platform | `linux/amd64` |

## 2. 基础镜像

- `FROM mambaorg/micromamba:2.8.1`
- 离线来源：`images/shared_images/micromamba-2.8.1.tar`

## 3. 构建上下文（本目录）

| 路径 | 用途 | Dockerfile 指令 |
|---|---|---|
| `Dockerfile` | 构建脚本 | — |
| `environment.lock.yml` | conda 锁文件（含 `pyscf` 占位 pin，构建时被剔除） | `COPY → /tmp/environment.lock.yml` |
| `environment.yaml` | 环境声明（资源/网络/健康检查元数据） | — |
| `public/` | 公开件（题面/协议/stage schema/输入等） | `COPY → /workspace/` |
| `wheels/pyscf-2.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | PySCF 2.8.0 离线 wheel | `COPY → /tmp/`，`sha256sum -c` 校验后 pip 安装 |
| `requirements-pip.txt` | pip 依赖提示 | — |

> 构建要点：conda-forge 无 linux-64 的 pyscf 2.8.0，Dockerfile 先 `sed '/^- pyscf=/d'` 删除该 pin 生成
> `environment.nopyscf.yml` 再 solve，随后用 **sha256 校验过的 wheel** 经 pip 装入 `chembench` 环境。

## 4. conda 环境

- 环境名 `chembench`，channel `conda-forge`
- `python=3.10.20`、`rdkit=2026.03.4`、`xtb=6.7.1`、`psi4=1.11`、`numpy=2.2.6`、`scipy=1.15.2`、
  `h5py=3.16.0`、`pandas=2.3.3`、`pyyaml=6.0.3`、`jsonschema=4.26.0`、`psutil=7.2.2`、`openai=2.45.0`、
  `pyberny=0.6.3`、`geometric=1.1.1`
- 构建工具链：`cmake>=3.24`、`ninja`、`make`、`git`、`c/cxx/fortran-compiler`、`pkg-config`、
  `openblas`（`libblas`/`liblapack` 指向 openblas）
- 额外 pip 安装：`pyscf==2.8.0`（来自 `wheels/`，sha256 = `4BAA59297DE8A2992E8AFA5B20A12164B6128C14CCC09431D05F73E3EEFDF008`）

## 5. 运行时配置（`environment.recipe.yaml: runtime_config`）

| 项 | 值 |
|---|---|
| User | `root` |
| WorkingDir | `/workspace` |
| Entrypoint | `/usr/local/bin/_entrypoint.sh` |
| Cmd | `bash` |
| PATH | `/opt/conda/envs/chembench/bin:...` |
| 其他 Env | `MAMBA_DEFAULT_ENV=chembench` 等 |

## 6. 资源与网络策略（`environment.yaml`）

| 项 | 值 |
|---|---|
| cpus | 4 |
| memory_mb | 16384 |
| storage_mb | 40960 |
| gpus | 0 |
| network_mode | `no-network`（评测模型调用在宿主侧） |
| workdir | `/workspace` |

## 7. 健康检查

```bash
python -c 'import numpy, scipy, rdkit, pyscf; from pyscf.solvent import smd' && xtb --version
```

## 8. 构建 / 加载 / 校验

```bash
docker load -i images/shared_images/micromamba-2.8.1.tar

# 从本目录重建（务必整目录作上下文，以保证 wheels/ 可用；构建态需联网）
docker build -t harbor-migration/task02:environment .

# 权威安装
docker load -i images/task02/task02-environment.tar

# 冒烟校验
docker run --rm harbor-migration/task02:environment \
  micromamba run -n chembench python -c \
  "import pyscf;from pyscf.solvent import smd;import rdkit,numpy;print('env OK',pyscf.__version__)"
```

## 9. 备注

- **不可删除 `wheels/`**：Dockerfile 内做 wheel sha256 校验，缺失即构建失败。
- 重建产物 `image_id` 与 lock 不同（功能等价）；需要 digest 一致时用 tar 加载。
- 评测 sidecar/grader 见同题 `tests/ENVIRONMENT.md`（`harbor-migration/task02:verifier`）。

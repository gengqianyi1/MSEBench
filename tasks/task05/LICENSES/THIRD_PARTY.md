# 第三方软件与使用边界

本包用于 ChemBench 组织者内部评测，不授予作者题目、参考答案、评分器的公开再分发或商业复用权。第三方软件仍按其各自许可证使用，本声明不限制第三方原有权利。

- Harbor 0.21.0：公开 PyPI 发行版；包内框架适配源码位于 agents、verifiers、chembench_harness、templates、scripts。
- Python 3.14.4（组织者）及任务镜像内 Python/xTB/NumPy/SciPy/PyYAML/jsonschema/psutil：公开 conda-forge/PyPI 来源。详细版本见环境锁、SBOM 和镜像内 conda-meta。
- xTB 6.7.1：LGPL-3.0-or-later，原始源码 https://github.com/grimme-lab/xtb/tree/v6.7.1 。未修改 xTB，不包含 Gaussian、ORCA 或作者商业项目。
- micromamba 2.8.1：公开 mambaorg/micromamba 基础镜像；保留镜像内版权声明。
- Docker CLI 29.1.3：Apache-2.0，官方发行 https://download.docker.com/linux/static/stable/x86_64/docker-29.1.3.tgz 。这里只随包提供客户端，不替代宿主机 Docker Engine。
- Docker Compose 2.40.3：Apache-2.0，Ubuntu 官方 archive 的 docker-compose-v2 包；原版权声明保留在组织者镜像 /opt/organizer/licenses。
- Python 依赖逐包版本、来源及可获取许可证见 public_python_dependencies/；发行镜像继续保留上游许可证。镜像文件和 vendor 均为公开软件，不是本机私有 venv/site-packages 的复制。

本包含 organizer-only Golden/reference/Grader。只供可信组织者持有；切勿把整个包挂载到模型 workspace。模型只通过公开题面、通用工具和 sidecar 协议工作。

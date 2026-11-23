# images/ — MatBench 镜像总目录

每个镜像一个 tar（OCI/layout，containerd store 导出），逐目录附 `images.lock.json`
（tag → image_id + tar 字节数 + sha256），顶层 `manifest.json` 汇总全族。

## 布局

| 目录 | 内容 |
|---|---|
| `shared_images/` | 跨题共用：harbor-runner、gost、egress-control-sidecar×3、python:3.11-slim、micromamba |
| `task01..task03/` | harbor 族 environment + verifier |
| `task04/05/06/18/` | chembench 族 environment + sidecar + verifier（tag 与 `tasks/taskNN/deploy/taskNN/images.lock.json` 一致，已断言同 digest） |
| `task07/08/09/` | organizer 族 environment + sidecar |

> C 档约束：`chembench/*` 与 `harbor-prebuilt:*` 等 tag 名是冻结运行时标识，不更名。
> egress-control-sidecar 三个后缀版本跨族同名不同内容，三份全保留，按族按需加载。

## 加载命令

```bash
# 一次性加载全部（约 25 个镜像）
docker load -i images/shared_images/*.tar   # 注意：逐个循环更稳，见下
for tar in images/shared_images/*.tar images/task*/*.tar; do docker load -i "$tar"; done

# 按族 / 按题加载
for tar in images/task04/*.tar; do docker load -i "$tar"; done   # 题 04
for tar in images/task01/*.tar; do docker load -i "$tar"; done   # 题 01

# 校验（导出时已双重校验：tar 内含目标 tag + reload 后 image_id 不变）
python3 -c "import json; print(json.load(open('images/manifest.json'))['families'].keys())"
```

## 重新导出

```bash
python3 scripts/export_images.py {chembench|harbor|organizer|shared}
```

## YAML 清单与自动安装接口（2026-09-27）

`images/task_images.yaml`（全量）与 `images/taskNN/images.yaml`（每任务）由
`scripts/gen_task_images_yaml.py` 从本目录 lock 生成，声明每个任务的镜像：
`tag / image_id / tar / bytes / tar_sha256`，并在可匹配时附 Dockerfile 构建配方
（`build: {context, dockerfile}`）。

```bash
python3 scripts/image_manager.py list   [--task NN] [--json]   # 查看 task→镜像映射
python3 scripts/image_manager.py verify [--task NN|--all]       # 校验 tar 存在 + sha256
python3 scripts/image_manager.py load   [--task NN|--all]       # 从 tar 安装镜像
python3 scripts/image_manager.py build  [--task NN|--all]       # 从 Dockerfile 构建
python3 scripts/image_manager.py ensure [--task NN|--all]       # 幂等：已存在则跳过，否则 load→build
```

- `--no-sha` 跳过 sha256（大目录快速校验）；`--role environment|sidecar|verifier|tests` 限定角色；`--dry-run` 只打印动作。
- 重新生成清单：`python3 scripts/gen_task_images_yaml.py`。
- 覆盖：20 个任务和 8 个共享镜像。
- 完整离线归档校验：在仓库根目录运行 `python3 scripts/verify_images.py`，按锁文件逐个核对 56 个 tar。

### ⚠️ 关于 `build`（Dockerfile 重建）——不作为全量重建手段

实测结论（2026-09-27）：`image_manager.py build --all` **不能**重建全部任务镜像。

- **无配方（15/55）**：全部 `sidecar` 与全部 `shared` 镜像没有可匹配的 Dockerfile → 直接 skip。
- **配方不完全（有配方也会失败）**：部分 Dockerfile 依赖 build-context 内的 `vendor/*`
  （如 `tasks/task04/environment/Dockerfile` 需要 `vendor/docker-29.1.3.tgz`），
  而交付包**未随任务目录提供**这些文件 → `docker build` 失败（已实测 task04-environment）。
- **digest 不一致**：即使构建成功，产物 `image_id` 也与交付 lock 不同，
  会破坏 organizer 启动器（`frameworks/07-09`）的镜像完整性校验（task08 曾因此无法运行）。

**正确安装方式**：用 `ensure` / `load`（从 tar 安装，`image_id` 与 lock 一致）：
```bash
python3 scripts/image_manager.py ensure --all
```
`build` 仅适用于在补齐 vendor/基础镜像后的本地实验，不作为分发等价重建。

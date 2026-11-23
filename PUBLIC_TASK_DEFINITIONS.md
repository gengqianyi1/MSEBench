# MSEBench 本地评测发布流程

v0.2.0 采用用户本地运行和评分。GitHub 提供完整引擎、题目运行与评分代码、测试代码和构建配方，Hugging Face 提供题目数据、标签、参考解、测试快照及 Docker 归档。下载组装后保持现有题族要求的目录结构。

## 审核与导出

组织方已批准公开题目评测所需的全部材料，包括原 private、solution、oracle、test_snapshot 和 verifier/sidecar 内容。源码目录中的这些名称仍表示评测运行时的职责。Agent 容器按原契约隔离评分材料。

先将发布范围冻结为逐文件 SHA256 清单，再导出新的候选目录。

```bash
python scripts/export_local_release.py --inventory
python scripts/export_local_release.py release-out/v0.2.0
python scripts/export_local_release.py release-out/v0.2.0 --verify
```

release/local_approvals.json 记录授权范围、代码和数据的目标目录、源文件 SHA256、可执行位及组装所需的相对链接。导出器校验源文件后生成两平台的同版 data_manifest.json 和父目录 release_manifest.json。数据清单还保存原有任务许可和来源说明。凭据模板、组织方接口地址与展示名称的处理发生在导出时，最终 SHA256 对应实际交付内容。

旧 v0.1.0 的导出规则和候选作为历史版本保留，当前发布采用 export_local_release.py。v0.2.0 的状态为 candidate，容器端到端结果需在目标 Linux 主机补齐。

## 用户组装

```bash
python scripts/prepare_local.py --repo OWNER/DATASET --revision FULL_DATASET_COMMIT_SHA \
  --images --output ../msebench-workspace
```

本地预演用 --source /path/to/huggingface。按题下载可以指定 --task 4,5，完整下载默认包含全部题目，--images 加入预构建镜像。下载器先在临时目录验证所有资源的大小和摘要，成功后发布工作目录并重建包内相对链接。用户配置模型接口后，通过 msebench --workspace 指向组装目录即可运行。

## 镜像、来源和验收

全套 56 个锁定镜像包括 verifier、sidecar、任务环境和共享依赖。镜像本体放在 HF 的 images/，源码端保存镜像清单和构建入口。Task 04、05、06、18 的新增 sidecar Dockerfile 从对应 verifier 扩展，Task 16、17 的多上下文构建由现有 Compose 配方提供。

原始标签和协议标识用于锁定评分实现。项目展示名称为 MSEBench，Python 包和命令为 msebench。Matbench 等上游数据名称保留来源归属。MIT 引擎许可及适用的任务、第三方许可随文件交付，组织方自有任务数据的额外再利用许可沿用现有决定。

本次按 2026-11-23 用户决定先公开候选版，待完成项目随后补齐。完整验收应在新目录完成安装、数据组装、SHA256 校验、合法与非法提交回归，以及全部 20 道题的 Linux Docker 端到端验证。记录代码版本、HF 提交 SHA、镜像 ID、执行命令、退出码和评分。数据公开后，本地结果用于复现公开评测协议。

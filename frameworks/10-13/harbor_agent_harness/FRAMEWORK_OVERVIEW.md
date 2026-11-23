# MatBench Harbor Combined：整体框架说明

## 1. 定位与边界

本目录是 Task 01–17 的独立 ChemBench/Harbor 数据集。它由每道题的任务目录和根目录的共享运行框架共同组成；仅复制 `tasks/` 不能完整运行。

运行链路为：

```text
Harbor CLI / runner
  -> 任务 task.toml、environment 与 tests 容器
  -> participant agent（ChemBenchAgent 或专用 agent）
  -> /workspace 中的任务协议、工具调用或提交文件
  -> task-local verifier / grader
  -> Harbor 读取 reward 与 trial result
```

任务的评分逻辑、私有参考数据和 validator 保留在各自 Task 目录中。根框架负责任务发现、协议执行、agent 调度、预算/状态管理和开发验证；它不替代任何 task-local grader。

## 2. 顶层目录职责

| 目录或文件 | 职责 |
|---|---|
| `tasks/` | Task 01–17；每题含 `task.toml`、instruction、环境、tests、私有参考数据和协议文件。 |
| `agents/chembench_agent.py` | 当前通用 participant agent，处理多数 ChemBench task-native protocol。 |
| `agents/sidecar_oracle.py` | Task 04–06 的组织者 Oracle/lifecycle agent；不是供 LLM 评测使用的 participant agent。 |
| `chembench_harness/` | 协议、状态、预算、gateway、grader 和校验的共享 Python 实现。 |
| `scripts/` | 结构校验、开发与运行辅助脚本。 |
| `templates/`、`schemas/` | 共享配置模板与 schema。 |
| `datasets/` | development、parity、release 数据集定义。 |
| `chembench`、`pyproject.toml` | 本地 ChemBench 命令入口和 Python 包定义。 |

## 3. 任务协议分组

| Task | `protocol_mode` / 类型 | 当前 participant agent 支持状态 |
|---|---|---|
| 01–02 | `hidden_suite_case` | 支持 |
| 03 | `task03_design_round` | 支持 |
| 04–06 | `workspace_sidecar_case` | **未接入通用 participant agent**；仅 `SidecarOracleAgent` 可作组织者链路测试。 |
| 07–08 | `hidden_suite_case` | 支持 |
| 09 | `adaptive_hidden_suite_case` | 支持 |
| 10–13 | 通用 JSON/Python 协议（无该字段） | 支持 |
| 14 | `candidate_only_round` | 支持 |
| 15 | `task15_diagnostic_round` | 支持 |
| 16–17 | `private_response_round` | 支持 |

“支持”表示当前 `ChemBenchAgent` 有对应协议分支；并不等同于已完成逐题、端到端的模型评测认证。

## 4. 标准运行入口

Harbor 0.21.0 应由评测环境提供；不要把本机虚拟环境路径写入任务包：

```text
请将 `HARBOR_BIN` 设置为目标环境中的 `harbor` 可执行文件路径。
```

从本目录根目录运行时，应使用任务专用 agent，而不是 Harbor 的通用 `oracle`：

```powershell
$harbor = $env:HARBOR_BIN
& $harbor run -p tasks/task-10-matbench-bandgap-pipeline `
  -a agents.chembench_agent:ChemBenchAgent -e docker
```

Harbor 通用 `-a oracle` 仅适合实际带有 `solution/solve.sh` 的题目。对采用 ChemBench 原生协议的任务使用它，会产生“缺少 `solution/solve.sh`”等不代表题目损坏的错误。

## 5. Task 04–06 的特殊 sidecar 架构

04–06 不允许 participant 直接访问私有计算或参考答案。它们使用下列受控路径：

```text
participant agent
  -> task sidecar（公开工具、预算和会话控制）
  -> organizer evidence（受保护证据）
  -> task verifier 回放证据、校验 submission
  -> reward
```

任务配置指定 `CHEMBENCH_TASK04_SIDECAR_URL`、`CHEMBENCH_TASK05_SIDECAR_URL` 或 `CHEMBENCH_TASK06_SIDECAR_URL`。每题还具有 `sidecar_deployment.yaml`、受保护 sidecar 镜像/端口与 `/organizer_evidence` 证据挂载约定。

当前合并包已保留这些题目文件和 `SidecarOracleAgent`，但通用 `ChemBenchAgent` 尚未实现 `workspace_sidecar_case` 的 HTTP 会话、工具调用、提交及证据处理分支。因此，04–06 **不能在当前状态直接作为通用 ChemAgentBench participant 任务进行正式 LLM 批量评测**。

要完成接入，应保留原 verifier 和 sidecar 协议，并在 `ChemBenchAgent` 增加该 protocol 分支，同时在 Harbor 运行编排中启动 sidecar、注入 URL/token/证据挂载。详见 `TASK04_06_AUTHOR_ACTION.md`。

## 6. 已完成的框架一致性检查

- 根 `ChemBenchAgent` 与原 Task 01–03 已验证框架版本字节一致。
- 共享 `chembench_harness` 与 Task 10–13 来源框架的核心 Python 文件一致。
- Task 07–17 来源的旧 agent 是当前 agent 的方法子集；当前 agent保留了其协议能力并增加了额外功能。
- `scripts/validate_structure.py tasks` 已报告 17 个任务、无结构错误。
- Harbor 0.21.0 已能解析 Task 01 的通用 agent 配置，以及 Task 04 的 `SidecarOracleAgent` 配置。

这不是 17 道题均已通过完整 Oracle→verifier→reward 运行的声明。实际运行边界和已知限制见 `VALIDATION_REPORT.md`。

## 7. 配套文档

- `README.md`：仓库简介与入口。
- `TASK_MAPPING.md`：Task 01–17 来源与迁移映射。
- `FRAMEWORK_COMPATIBILITY_AUDIT.md`：共享框架为何必须保留及其核验结果。
- `FRAMEWORK_DIFF_MATRIX.md`：当前框架与各来源 harness 的差异矩阵。
- `TASK04_06_AUTHOR_ACTION.md`：04–06 接入通用 participant agent 的改造清单。
- `VALIDATION_REPORT.md`：已经实际执行的校验、结果和限制。

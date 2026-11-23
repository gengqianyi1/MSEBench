# 运行时框架（Frameworks）

题目 = `tasks/taskNN/`（题定义）；框架 = `frameworks/NN-NN/`（各族运行时 runner + agents + harness 副本 + 构建上下文）。

| 框架 | 服务题号 | family | 运行入口 | 说明 |
|---|---|---|---|---|
| `01-03/` | 1-3 | harbor | `common/scripts/run_llm_evaluation.py` | 题面/协议烘焙在镜像内 |
| `07-09/` | 7-9 | organizer | `start_task.py`（sidecar 编排） | 隐藏套件逐 case；`runtime` 符号链接→ext4 |
| `10-13/` | 10-13 | chembench | `run_task.py`（revision6 单步预算） | `harbor_agent_harness/` 共享位 |
| `14-17/` | 14-17 | chembench | `./chembench llm-mock`（harbor-runner 2.1.0） | 自包含 harbor 包 |
| `19-20/` | 19-20 | chembench | `scripts/run_llm.py` | 题定义直排 |

命名规则：统一 `NN-NN`（题目范围）。族类型见 `task.json.family`，不要靠目录名推断。

## 框架内 `tasks/` 是符号链接，不是副本

每个框架的 `tasks/`（或 `framework/tasks/`、`harbor_agent_harness/tasks/`）下是
指向 `../../tasks/taskNN` 的**符号链接**——冻结 runner 按 `tasks/task-*` glob 寻址，故需保留该别名层，
不占额外空间。

## harness 副本 = 离线兜底（路线 A）

`frameworks/*/matbench_harness` 与 `frameworks/*/chembench_harness` 是 `src/msebench_harness` 的
字节级冻结快照。**权威源在 `src/`**；改动 src 后用 `scripts/compare_harness_copies.py` 批量刷新副本。

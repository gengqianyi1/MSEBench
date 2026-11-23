# task_15_lipophilic_sulfonamide_edit_budget migration

- Source execution mode: `native_runner`
- Harbor migration status: **protocol-refactor-complete** (2026-09-06)
  - 多轮接线、轮次 verifier 隔离与 `trajectory.jsonl` 重建已完成，并由 Docker 内 Harbor 0.21.0
    `-a oracle` 全链路实跑证实：五轮 step verifier 各 reward 1.0，末轮私有 grader 正常打分。
  - 此前的 `protocol-refactor-required` 是 2026-09-03 之前的状态，它会随 `run_legacy_grader.py`
    写进每份 `details.json` 的 `migration_status` 字段，交接文件曾据此判定本题未完成迁移。
  - 仍未完成的是**发布证据**（真实模型 parity、镜像 digest），不是协议迁移；
    `release_eligible` 因此保持 `false`。
- Legacy grader: `2.1.0-harness`
- Required artifacts: final_answer.json, trajectory.jsonl, run_summary.json

## Remaining parity/refactor work

Exact five-round candidate/feedback protocol is task semantics. Must become Harbor multi-step or a protocol sidecar; unrestricted shell agent would change the action space.

The canonical task runtime contains **no `native_runner.py`**. The full legacy private grader/data is baked only into the separate verifier image under `/task`.

## 2026-09-01 作者确认轮（曹阳）

- `AUTHOR_CONFIRMATION.md`：项目说明五项确认逐条作答；总览见仓库根 `AUTHOR_CONFIRMATION_14_17.md`。
- `tests/invalid_submissions/`：新增 5 个无效提交 fixture（合同要求 ≥3，此前为 0）。
- 验证：`./chembench task14-17-check`（Oracle 全链路 + 无效提交 + 私有边界，边界检查带自测）。
- 修掉两个会让 Harbor 评分失效的缺陷：
  1. `private_grader/grader.py` 用固定 `parents[3]` 定位 `harness` 模块，在 verifier 镜像
     （快照挂 `/task`）下解析到 `/`，每次评分都先抛 `ModuleNotFoundError`。改为向上搜索标志文件。
  2. `tests/final_grader.py` 未保护 `json.loads`，一个截断的候选文件让 verifier 非零退出并被记为
     `verifier_error`（基础设施故障）。改为按缺轮处理，分数作为下界照常算出。
- `oracle_status` 由 `synthetic-pending-runtime` 改为 `verified_offline`（14/15），
  `release_gates.oracle` 随之由 `task.toml` 经 sync 置为 true。
- 仍为 `false`：`build` / `parity` / `release_eligible` —— 需要 Harbor + Docker 运行时。

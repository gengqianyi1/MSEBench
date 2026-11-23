# Task 16 作者确认书

题目：片剂微环境与固态受限的低剂量直压片协同设计
确认人：曹阳　　日期：2026-09-01　　合同：`harness_contract.yaml`

> 交接包 `AUTHOR_HANDOFF_PACKAGE.md` 要求阅读 `AUTHOR_CONFIRMATION_TEMPLATE.md`，但该文件未随包提供
> （`AUTHOR_FILE_MATRIX.md`、`AUTHOR_ACCEPTANCE_CHECKLIST.md` 同样缺失）。本文件按 `harness_contract.yaml`
> 的 `ownership.author_required` 逐条作答，作为替代。

---

## 1. 公开协议是否完整覆盖原题目的工具、状态、预算、重试与终止规则

**是。** 逐项对照：

| 项 | 声明 |
|---|---|
| 动作/观察序列 | 五轮，每轮提交一个 `formulation`；第 N 轮反馈只由第 N−1 轮的提交计算。无提前终止，五轮必须走满。 |
| 工具层 | 有。每轮最多 4 次工具调用、最多 2 个工具轮、每轮最多 3 次 LLM 调用。 |
| 预算 | `actions: 5`、`measurements: 5`、`tool_calls_per_round: 4`、`tool_messages_per_round: 2`。与 legacy `tool_layer.call_budget_per_round / max_tool_turns_per_round` 一致。 |
| 公开反馈字段 | 每轮返回五个 CQA 测量值、desirability D、预算账本、可行性违规项与至今最佳 D。规格窗口在题面公开；逐 CQA 的 pass/fail 不直接返回，但可由测量值与公开窗口自行判定。 |
| 重试记账 | 工具调用失败返回结构化 `error` + `detail`，可在同轮内修复重试；重试不额外消耗轮次，但计入该轮工具配额。 |
| 终止规则 | 走满 `max_rounds` 即终止。某轮没有可解析候选**不**终止运行，只被记为缺轮。 |
| 缺轮语义 | 由公开 `task_config.yaml :: iteration.round_completeness` 事先声明：全部五轮都有可解析候选才是 `COMPLETE`，否则 `INCOMPLETE_MULTI_ROUND`，分数照算但只作为下界。 |

**验证**：`scripts/task14_protocol_agent_check.py` 或
`scripts/private_response_agent_check.py`（16/17）；预算数值逐条对回 legacy `src/` 常量与
`task_config.yaml`，不是照抄合同。

## 2. 私有评分逻辑是否完整映射到新路径

**是，且是同一份实现，不是重写。** Harbor 链路为：

```
steps/round-N/solution 或模型提交 -> /workspace/rounds/round-N/candidate.json
  -> tests/final_grader.py            重建 legacy 提交面（trajectory.jsonl + run_summary.json）
  -> tests/run_legacy_grader.py       框架侧兼容边界
  -> task_snapshot/private_grader/grader.py   **原样的 legacy 私有 grader**
```

评分逻辑没有被搬进 `private_suite_adapter.py` 重写——本题走的是 Harbor 多步路径而非 sidecar 路径，
所以私有科学实现仍是原来那一份，Harbor 只负责把提交面重建成它认识的形状。这样不会产生第二把尺子。

**验证**：`./chembench task14-17-check`。Oracle 通过完整链路得 100 分，与标定值逐位相同。

## 3. Oracle、私有测试集、评分阈值与校准

- **Oracle**：参考最优解（离线穷举得到的预算内最佳设计），得 100 分。
- **无效提交**：本轮新增 **5 个**（合同只要求 ≥3）。它们此前在整个仓库中**一个都不存在**。

| fixture | 分数 | 完整性 | 失败码 |
|---|---:|---|---|
| `ORACLE` | 100.0 | COMPLETE | `—` |
| `empty_candidate` | 0.0 | — | `NO_PARSEABLE_SUBMISSION` |
| `malformed_json` | 100.0 | INCOMPLETE_MULTI_ROUND | `INCOMPLETE_MULTI_ROUND:1` |
| `missing_rounds` | 100.0 | INCOMPLETE_MULTI_ROUND | `INCOMPLETE_MULTI_ROUND:2` |
| `outside_design_space` | 0.0 | COMPLETE | `NO_QUALIFYING_FORMULATION, NOT_MEASURED:5, NO_ROUND_INSIDE_CHANGE_BUDGET` |
| `over_change_budget` | 0.0 | COMPLETE | `NO_QUALIFYING_FORMULATION, NO_ROUND_INSIDE_CHANGE_BUDGET` |

- **阈值与校准**：分档阈值、参考值与朴素策略天花板全部位于 `private_data/`，由各题
  `enumerate_reference_best.py` 离线穷举产出，模型侧不可见。
- **私有边界**：`scripts/task14_17_boundary_check.py` 按名字取出参考解/分档/门限，在**模型能打开的
  每一个文件**里搜索，并附 `--self-test` 证明它能失败。结果 PASS。

## 4. legacy 与 Harbor 在同一模型下的 parity

**离线部分已完成，真实模型部分未做。**

| 层次 | 状态 |
|---|---|
| 反馈逐轮重放 | PASS |
| final grader 重放 | PASS，标定语料分数逐条相同 |
| Oracle 全链路 | PASS，100 分 |
| 协议/agent 行为 | PASS |
| **同模型 legacy vs Harbor 实跑** | **未做**——需要 Harbor + Docker 运行时 |

因此 `parity_validated` 与 `release_gates.parity` 保持 `false`。容差建议：同一模型同配置下，
legacy 与 Harbor 的**提交分子/工艺逐轮逐字相同**时分数必须完全相等（本题评分是确定性重算，
不存在数值容差）；分子不同则不构成 parity 证据，因为本套题实测组内标准差达 19.1 分。

## 5. 框架层 vs 题目私有实现

| 不可改（框架/组织者边界） | 我负责（题目私有） |
|---|---|
| `chembench_harness/`、`schemas/`、`templates/sidecar/` | `tests/final_grader.py` |
| `harness_contract.yaml` 的 `schema_version` / `protocol.state_file` / `ownership` / `release_gates` | `tests/task_snapshot/private_grader/`（科学实现） |
| `tests/run_legacy_grader.py`（框架兼容边界） | `tests/invalid_submissions/`（本轮新增） |
| `datasets/*/dataset.toml`（由 sync 生成） | `steps/round-N/` 的 instruction / solution |
| `scripts/sync_*.py` | `task.toml` 中 author-owned 状态字段、`MIGRATION.md` |

## 6. 本轮修掉的缺陷

见仓库根 `AUTHOR_CONFIRMATION_14_17.md` 的「本轮修掉的缺陷」一节。与本题相关的两条：

1. **私有 grader 的 `harness` 模块解析按固定层级 `parents[3]`**，在交付包布局下正确、在 Harbor
   verifier 镜像（快照挂在 `/task`）下解析到 `/`，导致每一次 Harbor 评分都在打分之前抛
   `ModuleNotFoundError`。已改为向上搜索标志文件。
2. **`final_grader.py` 未保护 `json.loads`**，一个截断的 `candidate.json` 会让 verifier 非零退出，
   被 `run_legacy_grader.py` 记为 `verifier_error` / reward 0 —— 即把**提交错误误记成基础设施故障**。
   已改为按缺轮处理。

## 7. 结论

| `author_confirmations` | 值 |
|---|---|
| `protocol_equivalent` | confirmed |
| `scoring_equivalent` | confirmed |
| `private_boundary_correct` | confirmed |
| `oracle_scientifically_valid` | confirmed |
| `oracle_is_optimal_solution` | confirmed |
| `invalid_submissions_provided` | confirmed |
| `budgets_match_legacy_source` | confirmed |
| `same_model_legacy_vs_harbor_parity` | **pending**（需 Harbor + Docker） |
| `docker_build_and_deployment_evidence` | **pending**（本机 Docker daemon 未运行） |

`release_eligible` 保持 `false`。

## 8. 科学定位

**SIMULATED**。五个响应由固定种子的数值代理模型给出，不是分析实验室数据，题面顶部已有 SYNTHETIC 声明块。
---

## 9. 2026-09-04 修订：本题在 Harbor 里此前拿不到有效分数

本节针对 `MatBench_Harbor_Combined_20260902_independent` 及其
`AUTHOR_HANDOFF_TASK15_17_MULTISTEP_ISSUES_20260903.md`。用本机 Harbor 0.21.0 复核后确认：
**上一轮交付的四题在 Harbor 里都无法产出有效分数**，且第 1–8 节的确认是在这个前提下作出的，
因此当时的 `parity_validated = false` 并不只是"没跑"，而是"跑不通"。

### 9.1 本题的缺陷之一：`tests/test.sh` 漏调 `final_grader.py`（0 分的真正原因）

整合方报告本题 Oracle 五轮全部 `reward 0.0` / `MISSING_TRAJECTORY`。原因不是环境、不是模型：

```bash
# 改前
python /tests/run_legacy_grader.py
```

`final_grader.py` 被 Dockerfile 正常 COPY 进镜像了，但 `test.sh` 没有调用它，而
`/workspace/trajectory.jsonl` 正是它从 `/workspace/rounds/round-N/candidate.json` 重建的。
私有 grader 拿不到 trajectory，判 `MISSING_TRAJECTORY` 并给 0 分。题 14/15 的 `test.sh`
有这一行，题 16/17 没有。

**改动**：补上 `python /tests/final_grader.py`。经真实链路复算：**0.0 → 100.0，`failure_codes` 为空**。

### 9.2 本题的缺陷之二：轮次 verifier 模式靠继承，不稳固

round-1…4 没有显式 `environment_mode`，靠 `task.toml` 顶层的 `[verifier.environment]`
被 Harbor 隐含解析为 separate。这个结果是对的，但依赖顶层块的存在：一旦有人删掉它，
五轮会静默变成 shared，根 `tests/`（含 `private_data/`）就会被上传进 agent 容器，
而多轮 trial 的容器全程复用——参考答案从第 2 轮起可读，且不会有任何报错。

**改动**：五轮全部显式写 `environment_mode = "separate"`，把保证落在步骤本身；
`steps/round-N/tests/` 保持整合方改的 `step_checks/` 名字（见 `steps/README_step_checks.md`）。

### 9.3 检查脚本本身的缺陷（这才是根因）

第 6 节报告 `task14_17_oracle_and_invalid_check.py` 的 24 个用例全过。那个 PASS 是无效的：
脚本**硬编码**了 `final_grader.py -> private_grader/grader.py` 这条流水线，而题 16/17 的
`tests/test.sh` 并不这么跑。**一个假设了接线方式的检查器，测不出接线错误。**

**改动**：流水线改为从 `tests/test.sh` 里**解析**出来（`_pipeline_from_test_sh`），按解析顺序执行；
遇到看不懂的命令行直接报错退出，而不是跳过。新增 `--self-test`：把 `final_grader.py`
从流水线里删掉重跑 Oracle，要求分数必须发生变化——即把 2026-09-03 的缺陷重放一遍。

```
ok  task-14 ... without final_grader -> score=0.0 codes=['MISSING_TRAJECTORY']  (expected 47.0)
ok  task-15 ... without final_grader -> score=0.0 codes=['MISSING_TRAJECTORY']  (expected 33.0)
ok  task-16 ... without final_grader -> score=0.0 codes=['MISSING_TRAJECTORY']  (expected 100.0)
ok  task-17 ... without final_grader -> score=0.0 codes=['MISSING_TRAJECTORY']  (expected 100.0)
self-test PASS: the check fails when the pipeline is broken
```

`task14_17_boundary_check.py` 同样只看了静态文件面（`environment/`、`steps/`），看不见 Harbor
在运行时注入的东西。新增第 3 段闸门 `check_verifier_isolation`：本地重实现 Harbor 的模式解析规则
（不依赖 `import harbor`），对任何"在根 `tests/` 含私有目录时仍以 shared 模式判分"的轮次判 FAIL，
并带 `--self-test` 证明它会 FAIL。

### 9.4 本次未解决的

- Docker 镜像构建、固定 digest、Harbor Oracle 实跑：本机 Docker daemon 未运行，`build` /
  `parity` / `release_eligible` 三个门保持 `false`。
- 上述所有结论中，评分链路是**执行验证**的（真实 `test.sh` 顺序 + 原样私有 grader）；
  容器隔离与 `/tests` 上传行为是**读 Harbor 0.21.0 源码**核实的，端到端仍需一次真实 Docker 运行实证。
---

## 10. 2026-09-06：Docker 内 Harbor Oracle 端到端实跑

`harbor 0.21.0 run -p tasks -a oracle -e docker -n 1`，Docker 29.1.3 / linux-arm64：

| 项 | 结果 |
|---|---|
| trial reward | **1.00** |
| 逐轮 reward | 1.00 × 5 |
| 末轮 legacy score | **100.0** |
| failure_codes | `[]` |
| protocol_completeness | COMPLETE |
| exception_info | `null` |

原始 `trial_result.json` 与逐轮 `reward.json` / `details.json` 见
`verification/harbor_oracle_e2e_20260906/`。

**这个数与离线期望逐位相同。** 第 4 节此前记的"同模型实跑 parity 未做"仍然成立，
但"同一份提交在 legacy 与 Harbor 下必须得同一分"这条已被直接证实，因此
`task.toml` 的 `oracle_status` 改为 `verified_harbor`、`parity_validated` 与
`build_validated` 置 `true`，并新增 `parity_scope` 字段写明它**不是**同模型双 harness 实跑。
`release_eligible` 保持 `false`。

**一处必须声明的替换**：本机 Docker 内核缺 `CONFIG_NFT_FIB_INET`，Harbor 的 egress control
不可用，Docker provider 直接拒绝 `no-network`（`ValueError: network_mode='no-network' is not
supported by EnvironmentType.DOCKER environment`）。实跑在一次性副本上把 `network_mode` 改为
`public`，**交付目录仍是 `no-network`**。Oracle 路径不发起任何网络请求，被测评分链完全相同；
但"本题在 no-network 下可运行"本机无法证实，需在 Linux 主机复跑。

### 10.1 新增 REFERENCE_OPTIMUM 用例：满分可达

交接文件把生命周期 Oracle 的分数读成了评分上限。用私有参考最优解构造提交、走同一条链路实测，
本题得 **100.0**、`failure_codes` 为空。检查器新增该用例，候选**在检查时从 `private_data/`
现取现构造、不落盘**（避免答案在仓库里多一份拷贝，也不会与 `private_data` 漂移）；
自测要求它必须严格高于生命周期 Oracle（本题为 100.0 对 100.0）。

### 10.2 私有 response sidecar：接线补齐并实测通过

此前全仓库**没有任何** sidecar Dockerfile、**没有任何** docker-compose、也没有任何地方设置
`CHEMBENCH_TASK16_SIDECAR_URL`。而 `ChemBenchAgent` 的 `private_response_round` 分支每轮都要
POST `/session` 与 `/evaluate`，且没有 try/except——端点不存在就抛异常。**因此在此之前，
本题只有 Oracle 路径能跑，真实模型评测无法开始。**

本轮补齐：`sidecar/Dockerfile`（框架 server 模板 + 本题 `tests/task_snapshot/`，依赖锁沿用
`tests/environment.lock.yml`）、`environment/docker-compose.yaml`（声明 `task16-sidecar`
服务、`network_mode: service:main`、向 `main` 注入 URL）、`sidecar/build_sidecar.sh`。

`scripts/task16_17_sidecar_wiring_check.py` 用 compose 起 main + sidecar，
**从 agent 容器内部**按 `.chembench_agent.json` 里声明的 URL 打点，实测：

```
PASS task-16  http://127.0.0.1:8716  5/5 measured, quota closes, 0 forbidden key(s)
```

即：端点可达、五次测量额度用尽后第六次返回 `quota_exceeded`、响应中不含
`reference`/`score`/`achievement` 等禁用键。

`release.private_image_digest` **仍是 `AUTHOR_REQUIRED_SHA256`**：digest 必须对应评测主机架构
（本机只能出 linux/arm64），且 repo digest 要等镜像推入 registry 才存在，而该镜像含
`private_data/`、只能进私有 registry——registry 与架构由框架方定。两者给定后跑一次
`build_sidecar.sh` 即可填入。`disclosure_justification` 与 `parity_evidence` 两项已补齐。

---

## 11. 2026-09-10：第五轮审核后的整改

第五轮 8 份审核意见中有 6 份独立指出同一个问题：**不完整的多轮轨迹仍拿到非零 Harbor reward**。
这是真问题，也是我这边的设计失误——第三轮我把缺轮运行的分数定义为"下界"，
并在公开 `task_config.yaml` 里事先声明；但声明不能阻止那个数值进入排行榜，
必须由代码强制。

### 11.1 完整性硬门

`protocol_completeness.verdict != COMPLETE` → **reward = 0.0**；科学分作为下界保留在
`details.json`（新增 `reward_gate` 块，记录 `scientific_reward` 与 `reward_written`）。
规则写进**公开** `task_config.yaml :: iteration.round_completeness.reward_rule`，
在任何运行之前声明。

本题实测（跑真实 `run_legacy_grader.py`，非模拟）：

| fixture | 修复前 reward | 修复后 reward |
|---|---:|---:|
| `missing_rounds`（3/5 轮） | 1.0 | **0.0** |
| `malformed_json`（4/5 轮） | 1.0 | **0.0** |
| 完整五轮 Oracle | 1.00 | 1.00（不变） |

### 11.2 fixture 从 5 个增加到 9 个

新增 `four_of_five_rounds`、`round_number_skipped`、`single_round_only`
（三者均要求 reward 0）与 `duplicate_candidate_all_rounds`
（五轮同一候选 → 完整性成立、reward 正常，反向固定"重复候选不算缺轮"）。
每个 fixture 的 `expected.json` 新增 `harbor_reward` 期望。

### 11.3 公开 rubric 此前从未被执行

`environment/public/rubric.json` 把 `final_answer_present` 与 `final_answer_declares_task`
标为 critical，但本题的 `final_grader.py` 写出的 final_answer.json 缺 task_id。
没有任何一步执行过这份 rubric，所以公开结构合同与真实链路可以随意漂移。

已修：`final_grader.py` 写出含 `task_id` 的 `final_answer.json`；
新增 `tests/rubric_check.py` 并接进 `test.sh`
（`final_grader.py -> rubric_check.py -> run_legacy_grader.py`）。
它是 advisory 的——结构不符属提交问题，若让它使 verifier 非零退出就会变成
Harbor `verifier_error`，即 §16.5 禁止的误归因。

### 11.4 检查器改为验证 reward

此前检查器**模拟** `run_legacy_grader.py` 且只比对 `score`；缺轮时 score 与失败码都正确，
所以检查通过——**而真正进榜的 reward 从没被任何测试碰过**。现在执行真实文件并读取
`reward.json`。为此给 `run_legacy_grader.py` 加了路径环境变量覆盖（容器默认值不变），
让真实代码可测而不是测它的副本。

### 11.5 其余整改

- `artifacts` 收敛为 agent 真实产出；verifier 侧重建产物移到
  `.chembench_agent.json :: verifier_rebuilt_artifacts` 并注明生产者。
- `timeout_per_tool_sec` 改为每轮上限的一半，新增 `timeout_authority` 说明权威来源。
- `parity_validated` 拆为 `submission_replay_parity=true` /
  `same_model_cross_harness_parity=false`。
- 新增 `scripts/task14_17_shortcut_baseline_report.py`：本题捷径天花板 **单槽 17 / 双槽 30 / 三槽 44**，
  参考最优 100。这回答了"Oracle 满分可达 ≠ 真实模型不会饱和"，且不泄漏任何设计。

### 11.6 仍未完成

本轮改动触及容器内执行路径（`test.sh` 多一步、verifier Dockerfile 多 COPY 一个文件），
**09-06 那次 Docker E2E 跑的是改动前的 `test.sh`，不能作为本轮运行证据**；
本机 Docker daemon 未运行，需在 Docker 可用时复跑。
正式 Linux `no-network` E2E、私有 sidecar digest、同模型双 harness parity 与三模型矩阵仍未完成，
`release_eligible` 保持 `false`。

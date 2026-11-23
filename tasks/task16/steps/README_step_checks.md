# `steps/round-N/step_checks/` 不是 Harbor verifier

每轮的 `step_checks/test.sh` 是本题原始的**逐轮格式闸门**（检查该轮 `candidate.json` 结构是否合法）。
它保留在这里是为了留档，**Harbor 不会执行它**。

原因：Harbor 用 `steps/round-N/tests/` 同时做两件事——找该轮的 `test.sh`，以及在该轮
`environment_mode = "separate"` 时把这个目录当作 **verifier 镜像的构建上下文**。这些目录里只有
`test.sh`、没有 Dockerfile，于是构建直接失败：

    Task environment directory .../steps/round-N/tests has no environment definition.

而本题四轮都必须是 `separate`：shared 模式下 Harbor 会把根 `tests/`（内含
`task_snapshot/private_data/`，即参考最优解）上传到 **agent 容器**的 `/tests`，而多轮 trial
的 agent 容器是全程复用的——参考答案会在第 2 轮起对模型可见。

因此目录改名为 `step_checks/`，Harbor 找不到轮次 `tests/` 就回落到根 `tests/`：脚本是根
`tests/test.sh`，构建上下文是根 `tests/`（有 Dockerfile）。逐轮闸门的功能没有丢失——
`multi_step_reward_strategy = "final"` 下只有末轮 reward 计分，逐轮 reward 本来就不参与评分；
而候选文件的结构合法性由末轮的 `final_grader.py` + 私有 grader 逐轮重新判定。

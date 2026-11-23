# MSEBench 本地运行指南

先按 [部署说明](../DEPLOY.md) 组装工作目录并加载镜像。模型参数从宿主机环境或 configs/llm_config.json 的 default profile 读取，密钥通过环境注入。

```bash
export MSEBENCH_API_BASE="https://your-provider.example/v1"
export MSEBENCH_MODEL_ID="your-model"
export MSEBENCH_API_KEY=""
msebench --workspace "$PWD" --task 4 --verify
msebench --workspace "$PWD" --task 4
msebench --workspace "$PWD" --task all --concurrency 1
```

--verify 在不同题族执行输入、安装、模拟或启动检查，完整评分需要运行模型并经过 verifier。任一任务运行失败时批次返回非零，批次摘要写到 results/run_summary/。运行证据写到 runs/，请在重新评分前保留完整提交目录及对应证据。

```bash
python scripts/score_local.py --task 4 --submission /path/to/submission \
  --evidence /path/to/organizer_evidence --output /path/to/new-score
msebench textqa --input tasks/textqa/textqa1.0.jsonl \
  --predictions predictions.jsonl --output results/textqa/results.jsonl
```

Task 04–09 和 18 的现有提交重新评分必须提供原运行的证据目录。Task 10–13 可以提供带 manifest.json 的 evidence 进行重放。其余题目同样要提供其契约要求的完整产物和调用计数记录。TextQA 的每条预测包含唯一 id 和 answer，英文使用 --language en。

材料任务使用 Docker 隔离 Agent 与 verifier。虽然评分标签已公开，Agent 容器仍仅挂载规定的参与者输入。默认路径是 runs/、results/，可以用 MSEBENCH_RUNS_ROOT 和 MSEBENCH_RESULTS_ROOT 修改。模型生成需要用户提供服务及相应费用。

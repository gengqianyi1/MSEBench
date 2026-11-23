# TextQA 1.0 — 材料科学选择题评测

正式数据 `textqa1.0.jsonl` 共 80 题，`sample.jsonl` 为前 6 题，用于快速验证。
每行包含 `id`、`field`、`material_category`、`question_ch`、`question_en`、`answer`、`explanation_ch`、`explanation_en`。答案为逗号分隔的正确选项，可包含一个或多个选项。

每题独立调用一次模型，统一使用项目 `configs/llm_config.json` 配置并启用流式输出。仅发送所选语言的题干，不发送答案或解析。默认中文，`--language en` 使用英文；不回退语言。旧版 `question`/`answer` 数据仍可使用。

从项目根目录运行：

```bash
export MATBENCH_API_KEY=<key>
python3 tasks/textqa/run_textqa.py --profile glm53_low --concurrency 4
python3 tasks/textqa/run_textqa.py --profile glm53_low --language en
python3 tasks/textqa/run_textqa.py --profile glm53_low --input tasks/textqa/sample.jsonl
```

每题满分 1 分：选项集合完全一致得 1 分；非空答案为正确选项的真子集（只少选、无错选）得 0.5 分；包含任何错误选项得 0 分。不区分大小写或选项顺序，允许逗号、空白等分隔符。重复选项、空输出和非答案文本均得 0 分；不从推理内容猜测答案。支持明确的 `Answer: A,C`、`答案：A,C` 和 `\boxed{A,C}` 格式。旧版非选择题沿用项目原有文本匹配规则。

离线评分无需 API，预测文件每行格式为 `{"id":"textqa_0000","answer":"B,C"}`，ID 必须与输入逐一对应，不得重复、遗漏或添加：

```bash
python3 tasks/textqa/run_textqa.py --predictions predictions.jsonl --language ch
```

输出默认位于 `results/textqa/<时间戳>/results.jsonl` 和同目录的 `summary.json`。`MATBENCH_RESULTS_ROOT` 可指定结果根目录，`--output` 可指定明细文件。明细按输入顺序记录题目、预测、标准答案、提取选项、每题得分 `score`、是否全对 `correct` 和调用错误；汇总包含总分 `total_score`、得分率 `score_rate`（总分/总题数）、半分题数 `partial`、全对率 `accuracy`（全对题数/总题数）、错误数及按 `field`、`material_category` 分类的指标。调用错误计入总题数并得 0 分，存在调用错误时退出码为 1；输入错误退出码为 2。可用 `--max-tokens N` 覆盖输出长度。

本地验证（无需联网）：

```bash
python3 -m unittest discover -s tasks/textqa -p 'test_*.py'
```

# Model profiles

Profiles contain reproducible model and decoding settings, never API keys.

The checked-in `deepseek-v4-pro-formal-20260905.profile.yaml` targets the
official DeepSeek endpoint and `deepseek-v4-pro`; profiles contain no API key.

Run from Ubuntu WSL after loading the five Docker archives listed in
`TASK01_03_DELIVERY_README.md`:

```bash
cd /path/to/Harbor_v5_task01_03_delivery_20260913
export DEEPSEEK_API_KEY='...'
PYTHONPATH="$PWD/common" python3 "$PWD/common/scripts/run_llm_evaluation.py" \
  --profile "$PWD/tasks/task03/profiles/deepseek-v4-pro-formal-20260905.profile.yaml" \
  --output-dir /tmp/harbor_task03_YYYYMMDD --task task03
```

The launcher defaults to `--engine harbor`.  Do not use `--engine direct` for
official scores: it is retained solely for focused infrastructure diagnostics.
Use a private environment file only when needed; it must not be added to Git.

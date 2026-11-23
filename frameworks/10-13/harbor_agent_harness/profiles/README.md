# Model profiles

Profiles contain reproducible model and decoding settings, never API keys.

`deepseek_v4_flash_siliconflow.yaml` is the default official profile for the
currently release-ready common-Agent set: Tasks 01–03.  It uses the user-specified
SiliconFlow OpenAI-compatible endpoint and a JSON tool protocol fallback.

Run from Ubuntu WSL after activating the Harbor virtual environment:

```bash
cd /mnt/d/ChemBench/workspace/MatBench_Harbor_Combined_20260902_independent
export SILICONFLOW_API_KEY='...'
export CHEMBENCH_HARBOR_BIN=/opt/chembench-harbor/bin/harbor
/opt/chembench-harbor/bin/python scripts/run_llm_evaluation.py \
  --profile profiles/deepseek_v4_flash_siliconflow.yaml \
  --output-dir /mnt/d/ChemBench/jobs/deepseek_v4_flash_YYYYMMDD
```

The launcher defaults to `--engine harbor`.  Do not use `--engine direct` for
official scores: it is retained solely for focused infrastructure diagnostics.
Use a private environment file only when needed; it must not be added to Git.

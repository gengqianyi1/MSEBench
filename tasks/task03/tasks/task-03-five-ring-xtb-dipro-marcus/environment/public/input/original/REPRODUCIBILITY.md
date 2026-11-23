# Reproducibility

Release 1.3 uses five fixed agent rounds. RDKit embedding uses the seed in
`task_config.yaml`; xTB jobs use the recorded geometries and command lines.

Recommended resources are 16 CPU cores, 16 GB RAM, and 20 GB free disk.
A complete run commonly takes tens of minutes to several hours, depending on
candidate validity, the expanded packing grid, hardware, and API latency.
`--rdkit-only` is a diagnostic mode and is not equivalent to the
full benchmark.

Build and verify:

```bash
docker build -t five-ring-bench:1.3 .
docker run --rm five-ring-bench:1.3 \
  python scripts/check_environment.py
```

API keys are supplied through a local config or environment variable and are
never part of a release archive. Preserve `config_used.yaml`,
`api_config_used.json`, `trajectory.jsonl`, and `final_report.json`.

# Reproducibility

Release 1.3 uses a frozen public split and evaluator-only property labels.
The data snapshot identifier is independent of the benchmark release number.

Recommended resources are 16 CPU cores, 32 GB RAM, and 30 GB free disk. Agent
runs generally take 15-60 minutes; private candidate scoring can take longer
because selected structures are independently relaxed with CHGNet.

```bash
docker build -t msebench-jarvis-candidate-v0.5.0:latest .
docker run --rm msebench-jarvis-candidate-v0.5.0:latest \
  pytest -q
```

Publish the public dataset checksums, redacted run configuration, trajectory,
tool calls, public validation, and aggregate private score. Do not publish
evaluator-only labels.

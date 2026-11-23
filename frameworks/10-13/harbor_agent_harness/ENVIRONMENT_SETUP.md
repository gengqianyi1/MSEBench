# Reproducible environment and run guide

## One-command validation on native Ubuntu 24.04 or WSL2

After extracting the delivery on the target Linux machine, run from the
delivery root:

```bash
bash run_all.sh validate  # no Docker required
bash run_all.sh docker    # build images and run grader tests
bash run_all.sh all       # additionally run every Harbor oracle
```

The delivery does not install system software with `sudo`. Install Docker and
Harbor `0.21.0` before using `docker` or `all` mode. The script preserves all
output under `run_logs/` and stops at the first failed command.

## Validated baseline

Use Ubuntu 24.04 on WSL2 or native Linux, Harbor 0.21.0, Docker 29.4.3, and Python
3.12. The tasks themselves build their scientific dependencies from their local
Dockerfiles and lock files; do not install those packages globally.

Recommended capacity is at least 4 CPU cores, 16 GB RAM, and 100 GB free disk for
images, build cache and job artifacts. No task in this collection declares a GPU.

## Windows + WSL2

1. Install/import Ubuntu 24.04 as WSL2 and enable Docker Desktop integration for it.
2. Confirm `wsl -l -v` reports version 2 and `docker version` works inside Ubuntu.
3. Install Python 3.12, `python3-venv`, Git and curl in Ubuntu.
4. Install Harbor 0.21.0 into a dedicated virtual environment. This machine uses
   `/opt/chembench-harbor`.
5. Copy the benchmark to `D:\ChemBench\workspace` and write jobs under
   `D:\ChemBench\jobs`.
6. If Docker Hub resolution is broken on the current network, configure a trusted
   reachable mirror in Docker Desktop's daemon JSON and restart Docker Desktop. The
   currently verified host uses:

   ```json
   {"registry-mirrors": ["https://docker.m.daocloud.io"]}
   ```

7. Confirm with `docker info` and a fresh `docker pull alpine:3.20`.

## Native Ubuntu 24.04 server

Install Docker Engine and Compose v2 from Docker's supported repository, add the
evaluation user to the Docker group, install Python 3.12 plus `venv`, and install the
same Harbor 0.21.0 environment. Use native Linux paths for both benchmark and jobs;
no WSL integration is required.

## Validation commands (inside WSL/Linux)

```bash
export BENCH=/mnt/d/ChemBench/workspace/MatBench_Harbor_Combined_20260902_independent
export JOBS=/mnt/d/ChemBench/jobs
export PATH=/opt/chembench-harbor/bin:$PATH
export PYTHONPATH="$BENCH"

harbor --version
docker version
python3 "$BENCH/scripts/validate_structure.py" "$BENCH/tasks"

# One task, framework failure-path regression
harbor run --path "$BENCH/tasks/task-03-five-ring-xtb-dipro-marcus" \
  --agent nop --jobs-dir "$JOBS" --job-name task03_nop --yes

# One task, environment/agent installation only
harbor run --path "$BENCH/tasks/task-01-jarvis-2d-exfoliation-candidate" \
  --agent nop --install-only --jobs-dir "$JOBS" --job-name task01_install --yes

# Oracle (only for a task with a packaged, verified solution)
harbor run --path "$BENCH/tasks/<TASK_DIRECTORY>" --agent oracle \
  --jobs-dir "$JOBS" --job-name <TASK>_oracle --yes
```

On the validated Windows host, run the top-level script inside WSL2. Whole-root
`harbor run --path "$BENCH/tasks"` works logically but is slow over DrvFS because
Harbor hashes large snapshots first.

## Future LLM evaluation

Keep credentials outside the repository and pass an env file to Harbor:

```bash
harbor run --path "$BENCH/tasks/<TASK_DIRECTORY>" \
  --agent "$AGENT_NAME" --model "$MODEL_NAME" --env-file /secure/path/benchmark.env \
  --jobs-dir "$JOBS" --job-name "${MODEL_NAME}_task" --yes
```

Do not include Tasks 04–06 in a common-agent batch until their sidecar remediation
has passed a new full regression. Never commit `.env` or job logs containing secrets.

# MSEBench 0.2.0

MSEBench provides bilingual TextQA and 20 materials-science agent tasks. Task inputs, scoring code, test labels, reference solutions, evaluation snapshots and verifier/sidecar images are published for evaluation on your own computer.

This is a public candidate. Linux Docker end-to-end and rebuilt-image equivalence validation remain pending. The matching dataset is currently being uploaded; a fixed dataset commit will be recorded when the upload finishes.

Use Python 3.12 and a Linux amd64 Docker Engine host with Compose v2. Container paths, host networking and the Docker socket follow the supplied Linux deployment. Scientific task resources vary. Task 01 recommends 16 CPU cores and 32 GB RAM, Task 02 recommends 32 cores and 32 GB RAM. Download sizes are recorded per file in `data_manifest.json`.

## Assemble a local workspace

Install the source checkout and host Harbor dependency.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[local]'
python scripts/prepare_local.py --repo gengqiane/MSEBench --revision FULL_DATASET_COMMIT_SHA \
  --images --output ../msebench-workspace
cd ../msebench-workspace
PYTHON=python bash verify_package.sh
bash load_images.sh
```

The organizer runtime tag used by Tasks 04–06, 10–13 and 18 is not declared in the shipped archive locks. Follow `DEPLOY.md` to build that runtime from the supplied recipe before running those tasks. Building and validating this runtime is a pending Linux acceptance item.

Before publication, assemble from the local data candidate with `--source /path/to/huggingface` instead of `--repo` and `--revision`. `--task 4,5` selects resources for those tasks and the shared runtime. Load a selected task's images and shared dependencies with `bash load_images.sh --task 4`. Omit `--images` if the matching locked images are already loaded. Every downloaded resource is checked by size and SHA256 before the workspace becomes available.

## Run a model locally

Set the endpoint, model and credential supplied by your own provider.

```bash
export MSEBENCH_API_BASE="https://your-provider.example/v1"
export MSEBENCH_MODEL_ID="your-model"
export MSEBENCH_API_KEY=""
msebench --workspace "$PWD" --task 4 --verify
msebench --workspace "$PWD" --task 4
msebench --workspace "$PWD" --task all --concurrency 1
```

Model generation calls the configured endpoint and may incur provider charges. The task runtime applies the published tool and request budgets. Scoring executes locally. The `--verify` checks vary by task family and do not replace a full model-and-verifier run. Outputs go to `runs/` and `results/`; `MSEBENCH_RUNS_ROOT` and `MSEBENCH_RESULTS_ROOT` override these paths.

## Score an existing submission

Preserve the task's full submission directory and runtime evidence when retaining a run. Several graders check tool executions, chemistry logs, request accounting and evidence provenance.

```bash
python scripts/score_local.py --task 4 --submission /path/to/submission \
  --evidence /path/to/organizer_evidence --output /path/to/new-score
```

Tasks 04–09 and 18 require the evidence directory. Tasks 07–09 use the locked sidecar image, which also includes their verifier scripts. Tasks 10–13 accept `--evidence` for a saved evidence replay. The locked image executes the original task test script with networking disabled, preserving detailed output under the score directory. Missing evidence is handled by the task's own grading rules.

## TextQA

```bash
msebench textqa --input tasks/textqa/textqa1.0.jsonl \
  --predictions predictions.jsonl --output results/textqa/results.jsonl
```

Use `--language en` for English. Predictions contain one unique `id` and `answer` per question. The synthetic example under `examples/` scores `score_rate=0.5` and `accuracy=0.0`.

## Reproduction and provenance

The image locks record the archived image tags and SHA256 values. Source build recipes are included. For Task 04, 05, 06 and 18 sidecars, load or build the corresponding verifier first, then use the repository root as Docker context with `tasks/taskNN/environment/Dockerfile.sidecar`. Rebuilt image IDs differ from the archived locks and require a new validated lock before use as the reference release.

Upstream names such as Matbench identify the original data sources. Versioned scoring protocol identifiers remain available for runtime compatibility. MIT applies to the MSEBench engine; task-local licenses and third-party notices apply to their respective files. Data reuse terms are recorded in the source notices and the release scope authorization.

The answer and reference data are public. Scores reproduce the published local protocol and users should record the code version, dataset commit, image IDs, model configuration and budgets with results.

For host-side scientific regression checks, install `.[test]` and run each task's regression suite in its own Python process. Task-private modules use their original import names. Tests that reference `/tests` directly run inside the supplied verifier container. The release checklist records target-host validation separately from host-side checks.

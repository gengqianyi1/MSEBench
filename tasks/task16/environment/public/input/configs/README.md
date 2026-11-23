# configs/ — what each file is and what to edit

Two kinds of file live here, and only these two are read at run time.

| File | Read by | Purpose |
|---|---|---|
| `default*.yaml` | `src/main.py` (`--config`) | which task to run, where to write the run, which API config to use |
| `api_config*.json` | `src/llm_client.py` | endpoint, model name, credential, token budget |
| `api_config*.example.json` | nobody | templates to copy; the only ones committed and shipped |
| `environment.yaml` | nobody | the interpreter, the pins, the endpoints, and the commands that verify the package — **grader package only** |

`environment.yaml` and the repository-root `README.md` are deliberately withheld from the public
archive: both quote difficulty-calibration numbers (gate ceilings by family, the score histogram,
band populations) that `question.md` does not disclose to the model. In the public archive the
document that replaces the root README is `PUBLIC_PACKAGE_NOTE.md`, and everything below that
mentions `environment.yaml` applies to the grader package; the public archive's own endpoint
information lives in the `_endpoint_status` and `_note` fields of the `api_config*.example.json`
templates.

`api_config*.json` (without `.example`) is **gitignored** and excluded from the delivery archive.
Never paste a key into any other file.

## Set up an environment and a key

```bash
cd <package root>
python3.10 -m venv .venv && PY=.venv/bin/python   # any CPython >= 3.10; see environment.yaml
$PY -m pip install -r requirements.txt

cp configs/api_config_deepseek.example.json configs/api_config_deepseek.json
cp configs/api_config_kimi.example.json     configs/api_config_kimi.json
cp configs/api_config_glm.example.json      configs/api_config_glm.json
# then edit each copy and replace PUT_YOUR_..._KEY_HERE with the real key
```

Those three are the official second-round model set — `deepseek-v4-pro`, `kimi-k2.6`, `GLM-5.2`, all
at `temperature: 0.0`. `api_config_qwen.example.json` and `default_qwen.yaml` are retired and kept
only so that first-round run directories stay interpretable; a qwen run must not enter a comparison
table. **Only the deepseek endpoint is verified** — read `_endpoint_status` in the kimi and glm
templates before spending a run on either.

Check the credential and the endpoint before spending a run (one cheap call):

```bash
$PY scripts/test_llm_connection.py --config configs/default_deepseek.yaml
```

## Run a model

```bash
$PY src/main.py --config configs/default_deepseek.yaml     # 5 rounds, one formulation per round
```

Always run from the package root: every path inside a `default*.yaml` is relative to it.

## Switch to a different model

Either edit `model` in an existing `api_config_<name>.json`, or add a provider in three steps:

1. `cp configs/api_config.example.json configs/api_config_<name>.json`, then set `base_url`, `model`
   and `api_key`.
2. `cp configs/default_deepseek.yaml configs/default_<name>.yaml`, and point
   `llm.api_config_file` at the file from step 1.
3. `$PY src/main.py --config configs/default_<name>.yaml`.

Any OpenAI-compatible chat-completions endpoint works — `src/llm_client.py` uses the `openai`
package and nothing vendor-specific. `configs/environment.yaml` lists the official endpoints, their
verification status, the retired ones and the model names that are known not to work.

## The fields in an api_config

| Field | Meaning |
|---|---|
| `model` | model name as the endpoint spells it |
| `base_url` | OpenAI-compatible base URL |
| `api_key` | credential; the harness refuses to start while it is still the placeholder |
| `temperature` | passed straight through |
| `max_tokens` | completion budget for the first attempt |
| `max_tokens_schedule` | budget per attempt; the harness escalates to the next entry **only** for an empty response caused by exhausted reasoning tokens, or for a response truncated mid-JSON. A transport failure such as a timeout is retried at the *same* budget, because a bigger budget makes the request slower and would guarantee another timeout |
| `timeout`, `max_retries`, `retry_delay_seconds` | client-side retry behaviour |
| `raise_on_empty_response` | `false` in delivery: an empty response costs the model that round instead of aborting the run |
| `save_full_api_response` | store the raw response object in `llm_api_debug.jsonl` |

Reasoning models need a real budget: at 8192 completion tokens `deepseek-v4-pro` reliably spends the
whole allowance on reasoning and returns empty content, so its config asks for 16384. All three
official configs now ask for 16384 for that reason. Beware the opposite failure on a capped gateway:
the DashScope compatible endpoint cuts a response stream at 298 s, so a budget large enough to avoid
an empty response can be large enough to trigger a 504 instead. That trade-off is what makes the
kimi endpoint an open question this round — see `environment.yaml`.

## The fields in a default\*.yaml

| Field | Meaning |
|---|---|
| `llm.api_config_file` | which `api_config*.json` to load |
| `task.task_dir` / `question_file` / `task_config_file` | the task to run; `tasks/q1_microenvironment` |
| `runtime.run_dir` | where run directories are created (`runs/`) |
| `runtime.random_seed` | seeds the harness; the oracle itself is deterministic |
| `parsing.allow_markdown_fence` | accept a JSON object wrapped in a ``` fence |
| `logging.*` | output filenames inside the run directory |

`runtime.max_rounds` in these files is informational. The authoritative round count is
`iteration.max_rounds` in `tasks/q1_microenvironment/task_config.yaml`, which is what `src/main.py`
loops over.

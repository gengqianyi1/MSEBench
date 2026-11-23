# Task 01 tool policy

Only the two sidecar tools declared to the model may be called.

## `run_python`

Pass an object with `script` (Python source) and `timeout_seconds` (1-900).
The script runs with the task workspace as its current directory. Use relative
paths. Reading and writing workspace files, NumPy, pandas, scipy, scikit-learn,
joblib, pymatgen, and other installed offline libraries are allowed during the
40-action primary workflow.

The sandbox rejects network imports (`socket`, `urllib`, `requests`, `ftplib`,
`smtplib`, `telnetlib`, `http`, `httpx`, `aiohttp`, `urllib3`, `websocket`, and
`paramiko`), direct `chgnet` import, dynamic loading through `importlib` or
`__import__`, destructive filesystem/process calls (`os.system`, removal,
permission/ownership changes, process killing, and `shutil.rmtree`), and
subprocess binaries other than Python. Do not try to bypass these controls.

Example native call arguments:

```json
{"script":"from pathlib import Path\nprint(Path('input/submission_contract.md').read_text())","timeout_seconds":120}
```

## `relax_selected_candidates_chgnet`

This is the only allowed CHGNet interface. Call it at most once, after
`candidate_library.json.gz` exists, with exactly five submitted candidate IDs:

```json
{"candidate_ids":["id1","id2","id3","id4","id5"]}
```

`run_python` and `relax_selected_candidates_chgnet` share a 40-action primary
budget. After it is exhausted, at most two `run_python` calls may repair only
schema/output defects; they may not train, regenerate candidates, or invoke
CHGNet. `read_file` is not a primary action and is limited to 100 calls.

The task uses a strict 40-LLM-request budget cap. The Harbor control plane
records every provider request, including failed attempts and retries; a run
whose recorded request count is at most 40 keeps its full scientific score, while
any run exceeding 40 recorded requests is scored 0 for breaching the cap. A six-hour
wall-time safety cap and per-script timeout remain. Submit only after
`input/submission_check.py` passes; no runner fallback creates missing artifacts.

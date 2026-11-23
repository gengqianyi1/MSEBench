# task_08_xtb_failure_diagnosis_restart migration

- Source execution mode: `native_runner`
- Harbor migration status: **harbor-ready-candidate**
- Legacy grader: `xtb-failure-suite-v2-replay-1.1`
- Required artifacts: final_answer.json, benchmark_result.json, benchmark_run/campaign_summary.json, benchmark_run/*/suite_evaluation.json

## Harbor protocol

Hidden structured-tool xTB suite now runs behind an organizer-side `chembench-hidden-suite/1.0`
stateful adapter. Harbor owns model calls; the sidecar charges tool quota before execution,
reveals one case at a time, returns only allowlisted diagnostic responses, and persists
evidence under a token/case two-level path. The verifier independently replays those bundles.

## Verified gates

- Native sidecar -> restricted Agent -> protocol grader: 6/6 cases, score 100.
- Docker Oracle replay through the new protocol grader: 6/6, score 100.
- Security probes: forged evidence ID, path traversal, tampered organizer evidence, and
  workspace self-reported-result injection are all rejected or ignored.

The private sidecar digest and offline archive are pinned in `docker/image_manifest.json`.
The synchronized `task.toml` and harness contract record the current release state.

The canonical task runtime contains **no `native_runner.py`**. The full legacy private grader/data is baked only into the separate verifier image under `/task`.

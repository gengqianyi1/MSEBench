# task_07_qed_molecular_design migration

- Source execution mode: `native_runner`
- Harbor migration status: **harbor-ready-candidate**
- Legacy grader: `qed-suite-v4-replay-1.0`
- Required artifacts: final_answer.json, benchmark_result.json, benchmark_run/campaign_summary.json, benchmark_run/*/suite_evaluation.json

## Harbor protocol migration

The five-case campaign now uses `chembench-hidden-suite/1.0`. Harbor owns model calls; an
organizer-side service reveals one case at a time, charges tool quota before execution, and
returns only allowlisted RDKit diagnostics. The verifier independently replays tool results and
reconstructs the suite score.

Verified gates include five-case restricted-Agent replay, a calibrated attainable reference
optimum scoring 100.0, a legal-but-wrong fixture scoring 0.0, direct-versus-reconstructed
score parity, private-case isolation, forged-response resistance, and the pinned offline
sidecar archive recorded by `docker/image_manifest.json`. The synchronized release contract
is the current publication state.

The canonical task runtime contains **no `native_runner.py`**. The full legacy private grader/data is baked only into the separate verifier image under `/task`.

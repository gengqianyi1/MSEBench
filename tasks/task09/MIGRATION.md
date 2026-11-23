# task_09_adaptive_xtb_torsion_profile migration

- Source execution mode: `native_runner`
- Harbor migration status: **harbor-ready-candidate**
- Legacy grader: `adaptive-torsion-suite-v3-replay-1.0`
- Required artifacts: final_answer.json, benchmark_result.json, benchmark_run/campaign_summary.json, benchmark_run/*/suite_evaluation.json

## Harbor protocol

Adaptive torsion campaign now runs behind an organizer-side
`chembench-hidden-suite/1.0` stateful adapter. Harbor owns model turns; the
adapter enforces the 5-degree grid, batch phases, unique-angle budget, feedback
rounds, and terminal `run_scan_batch(finalize=true)` semantics. Evidence is
written under a token/case two-level path and independently replayed by the v3
multicase evaluator.

## Verified gates

- Restricted Agent -> native sidecar: 4/4 cases, 3 actions each, 12 valid unique
  angles per case.
- Protocol grader replay through the v3 multicase evaluator: 4/4 eligible;
  calibrated reference optimum 100.0 and legal-but-wrong fixture 11.8.
- Security probes: forged evidence ID, path traversal, tampered organizer
  evidence, and workspace self-reported-result injection are rejected or ignored.

The private sidecar digest and offline archive are pinned in
`docker/image_manifest.json`; the synchronized release contract is current.

The canonical task runtime contains **no `native_runner.py`**. The full legacy private grader/data is baked only into the separate verifier image under `/task`.

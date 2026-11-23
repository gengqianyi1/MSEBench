# Task03 Harbor migration

- Source: `task_03_five_ring_xtb_dipro_marcus`, release 1.3, `native_runner`.
- Current status: **release validation pending**. The task is not release-eligible
  until a sidecar deployment, Linux/no-network Oracle, and parity evidence are
  supplied for this frozen version.
- The five candidate rounds plus final recommendation contract is preserved.

## Protocol and scientific equivalence

Each round returns evaluator-generated RDKit/xTB/DIPRO/Marcus feedback before
the next model turn. The final verifier independently recomputes all five
candidates and scores only the selected existing candidate; Agent-provided
scores are never trusted. Invalid, mismatched, or ineligible recommendations
receive structured zero scores.

The Linux runtime is pinned to `xtb=6.7.1=h8876d29_4`. The newer
`gfortran_ha02e54d_5` build was rejected because its DIPRO path segfaulted.

## Verified evidence

- Positive runtime Oracle: 5/5 successful candidates, eligible recommendation,
  score `21.004302890552584`, exit code 0, no OOM.
- Frozen offline calibration Oracle: `100.0`; all nine reference molecules
  reproduce the frozen Marcus and scoring formulas.
- Same DeepSeek five-round output, recomputed with the fixed runtime: 5/5
  successful candidates and score `11.675565284728997` in both the independent
  Harbor recomputation and the legacy grader adapter.
- The old score `0.0` from the crashing xTB build is invalid and must not be
  reused in model comparisons.
- Forged self-reported score, invalid recommendation, recommendation mismatch,
  and ineligible recommendation checks all pass.

Parity tolerance is an absolute score difference of `1e-9`. Evidence is stored
under `parity/task03_positive_container_recheck3_20260830/` and
`parity/task03_fixed_build_same_model_recheck_20260830/`.

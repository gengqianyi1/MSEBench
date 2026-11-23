# Task 06 Harbor migration

Status: `harbor-validated`; offline protocol, Oracle, private-boundary and same-model scoring parity are verified, and the complete Harbor container lifecycle has passed.

## Public protocol

- One neutral terpyridine XYZ, generic atom mapping, route-neutral question, generic xTB scan syntax and answer schema are public.
- The frozen minimal runtime already exposes exactly seven tools: `list_files`, `read_file`, `write_file`, `run_python`, `xtb_run`, `budget_status` and `submit`.
- Limits are 12 compute credits/jobs, 72 requested scan points, 55 scientific turns, 8 finalization-only turns and 240 tool calls. SP, opt, Hessian and scan each cost one admitted job; failed jobs retain cost.
- Coordinate discovery, scan scheduling, basin identification, path/barrier construction, coupling analysis, validation, recovery and conclusions remain model responsibilities.
- A missing or invalid answer contract can hard-zero the task, but a scientifically incomplete or failed search receives evidence-backed partial credit rather than an automatic zero.

## Private replay

The sidecar writes authoritative jobs and submission snapshots to organizer evidence. The unchanged Grader 1.22.0 reconstructs both torsions, basin matches, scan coverage, barriers, paths and coupling from raw xTB frames. The protocol grader performs two canonical replays and ignores model-supplied score or identity names.

## Oracle and parity evidence

- Three independent real-xTB Golden repetitions score 100.0 and replay identically.
- Legacy versus Harbor private replay is exact: DeepSeek V4 Pro 41.04, Kimi K2.6 39.04 and GLM 5.2 72.64; all score deltas are 0.
- The high GLM score is preserved as a genuine model result; no calibration or model-specific remapping is applied.

## Harbor deployment validation

- Harbor 0.21.0 job `task06-harbor-lifecycle-20260829` completed 1/1 trial with zero exceptions, reward 1.000, score 100 and canonical replay equality.
- The full sidecar-Agent-organizer-evidence-separate-verifier path preserved Task 06 partial-credit semantics; scientific incompleteness is not converted into an automatic zero.
- Private verifier image: `chembench/task-06-terpyridine-coupled-torsion-landscape-verifier:1.22.0` at `sha256:300d145322e8342a5edeea26f36a1d06f469f3a8262f10e791aae47ab7c89dde`.
- The evidence tree hash and complete run/verifier logs are recorded in `validation/harbor_lifecycle/task06/deployment_record.json`. No model API call or new scientific xTB calculation was made. Release eligibility is enabled.

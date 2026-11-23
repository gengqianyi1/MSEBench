# Task 04 Harbor migration

Status: `harbor-validated`; offline protocol, Oracle, private-boundary and same-model scoring parity are verified, and the complete Harbor container lifecycle has passed.

## Public protocol

- One public neutral-formula L-histidine seed, generic atom mapping, question, schemas and templates are mounted in the model workspace.
- The model receives only `list_files`, `read_file`, `write_file`, `run_python`, `xtb_run`, `budget_status` and `submit`.
- The frozen budget is 56 compute credits with SP/opt/Hessian costs 1/1/4, 240 tool calls and 160 Agent turns. Failed admitted jobs retain cost. Scan is disabled.
- Candidate generation, protonation/microstate reasoning, resource allocation, identity diagnosis, Hessian selection, recovery, clustering, population analysis and conclusions are not implemented by the public Harness.
- Turn exhaustion triggers an honest partial snapshot. It does not erase independently supported scientific credit. Provider/API failure is an unsuccessful run and is retried by the organizer, not resumed through a task-local ledger.

## Private replay

The sidecar owns the minimal xTB runtime and stores authoritative jobs outside the model workspace. `protocol_grader.py` resolves only the organizer-issued evidence ID, reconstructs trajectory counts from organizer metrics, and invokes the unchanged geometry-derived Grader 2.1.0 against the private reference. The grader derives identities and populations from optimized geometries and xTB thermochemistry rather than model labels.

## Oracle and parity evidence

- Three independent real-xTB Golden repetitions replay to 100.0 with canonical equality.
- The Harbor Oracle derivation preserves every job, geometry, energy and population. It only adds the explicit finite-search/near-degeneracy limitation required by the current frozen schema; the original Golden copies remain immutable.
- Legacy versus Harbor private replay: DeepSeek V4 Pro 43.1009090909, Kimi K2.6 45.4397727273 and GLM 5.2 27.726969697; all score deltas are 0.

## Harbor deployment validation

- Harbor 0.21.0 job `task04-harbor-lifecycle-20260829-final` completed 1/1 trial with zero exceptions, reward 1.000, score 100 and canonical replay equality.
- The final run used a clean single-token `/organizer_evidence` mount; its evidence tree hash and complete run/verifier logs are recorded in `validation/harbor_lifecycle/task04/deployment_record.json`.
- Private verifier image: `chembench/task-04-histidine-microstate-ensemble-verifier:2.1.0` at `sha256:0416a631720fcef96161bff14e53e82b8d5b22ff0ea971ef33aa341514554b5e`.
- The deployment check used the frozen Golden and made no model API call or new scientific xTB calculation. Release eligibility is enabled.

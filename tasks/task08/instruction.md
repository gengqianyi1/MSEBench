# xTB Failure Diagnosis and Restart Agent Benchmark v2

Task ID: `xtb_failure_diagnosis_restart_agent_v2`  
Protocol: `xtb_failure_recovery_tool_loop/1.0`

Diagnose one visible failed or scientifically incomplete xTB workflow, then
recover the job when the frozen specification permits it. The required target
is a GFN2-xTB geometry optimization followed by a Hessian, with all scientific
invariants and artifact provenance preserved.

The participant package contains one public case. Official organizer scoring
uses six equally weighted cases spanning distinct failure mechanisms. Hidden
case inputs and diagnosis references are absent from the participant archive.

## Evidence and tools

The case workspace contains `job_spec.json`, a frozen input geometry, an
initial-attempt record, native logs, scheduler/process metadata, and hash-bound
manifests. File names, scheduler completion, exit zero, and marker files are
not by themselves scientific-success evidence.

Available tools are `list_case_files`, `read_text_range`, `tail_text`,
`parse_xtb_log`, `inspect_scheduler_record`, `validate_xyz`,
`inspect_artifact_provenance`, `prepare_restart`, `run_xtb`,
`validate_attempt`. The terminal submission is not made through those
tools: it is made by calling `submit_candidate`, which carries the
`submit_final` contract's strict object. There is no arbitrary shell, Python,
network, credential, project-source, parent-directory, or evaluator access.

Only successful `read_text_range` calls are citation-eligible. Parser and tail
results locate passages; final evidence must reproduce the returned bounded
reference exactly. The path must remain relative to the case workspace.

## Budgets and invariants

Machine-readable limits in `configs/task.yaml` are authoritative: 24 tool
calls, 14 agent turns, three generated attempts, and 900 seconds total. Native
calls are capped at 240 seconds. Geometry cycle limits are 2–200; a Hessian
uses a null optimization-cycle limit; thread count is fixed at one.

The legacy snapshot field `task.yaml:max_tool_calls: 1` counts the runner's
single outer model invocation only. It does not limit this protocol's
structured actions. The authoritative agent-visible limits are the 24 actions
and 14 turns above and in `harness_contract.yaml`.

Permitted repair dimensions are safe source selection, geometry cycle limit,
thread count, walltime, and executing a required Hessian after validated
optimization. Molecular composition/order, charge, UHF, method, solvation,
and required phases must not change.

## Scoring

The per-case rubric is 40 points for diagnosis and evidence, 20 for a
hash-bound restart, 30 for independently validated recovery, and 10 for
compliance. Process style receives no standalone points. A wrong scientific
core answer is capped at 20 even if mechanical execution succeeds. Hard gates
still exclude fabricated execution, invalid provenance, unsafe paths, missing
required phases, schema failure, budget violation, or tampered evidence.

Suite aggregation reports PASS, FAIL, and BLOCKED separately. BLOCKED cases do
not enter the numeric mean; a run is ranking-eligible only with complete case
coverage. This prevents environment failures from masquerading as model
scores.

## Final submission

Call `submit_candidate` exactly once, with the required candidate fields
directly as arguments — flat, with no outer envelope. The required flat
fields are `case_id`, `final_status`, `diagnosis`, `restart_summary`,
`result_summary`, and `rationale`; the strict object contains case ID,
terminal status, structured diagnosis and conflict resolution, bounded
evidence, restart changes/invariants, result claims, and rationale. The
restart source is an object containing source attempt ID, relative path,
frame index, file SHA-256, and frame SHA-256. Runner-derived
`trusted_runtime` remains outside the model submission and is independently
reconstructed during replay. This call is terminal: do not wrap it in a
`final_candidate` object and do not continue working after it.

---

## ChemBench v2.1 Harbor execution contract

The evaluated model is supplied by the Harbor Agent layer; this task must not call the evaluated LLM directly. Work only inside `/workspace`. Required legacy artifacts: `final_answer.json`, `benchmark_result.json`, `benchmark_run/campaign_summary.json`, `benchmark_run/*/suite_evaluation.json`.

**Migration status:** Harbor protocol, parity, leakage, and 100-point Oracle gates verified for release. See `MIGRATION.md`.

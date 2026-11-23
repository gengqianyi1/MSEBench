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
`validate_attempt`, and `submit_final`. There is no arbitrary shell, Python,
network, credential, project-source, parent-directory, or evaluator access.

Only successful `read_text_range` calls are citation-eligible. Parser and tail
results locate passages; final evidence must reproduce the returned bounded
reference exactly. The path must remain relative to the case workspace.

## Budgets and invariants

Machine-readable limits in `configs/task.yaml` are authoritative: 24 tool
calls, 14 agent turns, three generated attempts, and 900 seconds total. Native
calls are capped at 240 seconds. Geometry cycle limits are 2–200; a Hessian
uses a null optimization-cycle limit; thread count is fixed at one.

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

Call `submit_final` exactly once. The strict object contains case ID, terminal
status, structured diagnosis and conflict resolution, bounded evidence,
restart changes/invariants, result claims, and rationale. The restart source
is an object containing source attempt ID, relative path, frame index, file
SHA-256, and frame SHA-256. Runner-derived `trusted_runtime` remains outside
the model submission and is independently reconstructed during replay.

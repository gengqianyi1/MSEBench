# xTB Failure Diagnosis and Restart Agent Benchmark v2

This benchmark measures whether a tool-using agent can distinguish process
completion from scientific completion, diagnose a failed or incomplete xTB
workflow, choose a provenance-safe restart, and complete a GFN2-xTB geometry
optimization followed by a Hessian without changing the frozen scientific
specification.

Version 2 replaces the reviewed single-case release with a six-case suite. The
participant distribution contains the public maximum-cycle case only. The
organizer distribution additionally contains five hidden cases generated with
native xTB 6.7.1: SCC non-convergence, malformed coordinates, an unsupported
option, a native timeout, and a wrong-phase/incomplete workflow. Hidden inputs,
configurations, and diagnosis references are never included in the participant
archive.

## Provider retry budget scope

`max_provider_retries` is a per-provider-request transport retry allowance. The
organizer evaluator therefore validates the cumulative case count against
`max_agent_turns * max_provider_retries` (14 × 5 = 70). A cumulative count above
five is reported separately as infrastructure degradation; it does not by itself
erase a scientifically valid result.

## Evaluation contract

The agent has eleven structured tools and no arbitrary shell, Python, network,
credential, project-source, parent-directory, or evaluator access. Scientific
intent is defined by `job_spec.json`; scheduler state, exit code, file names,
marker files, native log state, and artifact provenance remain distinct facts.
Only bounded `read_text_range` results are citation eligible.

Each case is worth 100 points:

- diagnosis and evidence: 40;
- restart strategy and provenance: 20;
- independently validated task recovery: 30;
- compliance and efficiency: 10.

There are no points for process style. A wrong scientific core answer is capped
at 20 before eligibility gates. Fabricated execution, unsafe paths, schema or
budget failure, invalid provenance, changed invariants, or missing optimization
or Hessian phases produce an official score of zero.

Suite results report `PASS`, `FAIL`, and `BLOCKED` separately. `BLOCKED` means
the required environment or credential was unavailable; it is excluded from
the numeric mean and makes the suite ineligible for ranking. It is never
converted into a model score.

## Reproduce locally

Create the pinned environment:

```powershell
conda env create -f environment.yml
conda activate xtb-benchmark
python scripts/validate_config.py
python -m pytest -q
```

The tested stack is Python 3.12.13, xTB 6.7.1, RDKit 2025.03.5, OpenAI 2.46.0,
Pydantic 2.13.4, PyYAML 6.0.3, python-dotenv 1.2.2, and pytest 9.1.1. The xTB
identity policy checks the exact version and normalized required help tokens on
every platform. Platform hashes are recorded as references, not used as an
incorrect cross-platform binary lock.

Create an initial reviewed identity candidate with
`python scripts/freeze_xtb_identity.py freeze`. A later environment change must
use `python scripts/freeze_xtb_identity.py refreeze`; it writes a separate
candidate and never silently replaces the current commitment.

Evaluate and aggregate already generated outputs without calling a model API or
rerunning xTB:

```text
python evaluator/evaluate.py --final-answer <final_answer.json> --trajectory <trajectory.jsonl> --artifacts <case_workspace> --config <case_config.yaml> --output <evaluation.json>
python evaluator/evaluate_suite.py --suite <suite.json> --evaluations <evaluation_dir> --output <suite_evaluation.json>
```

`configs/suite_public.json` is participant-visible. The organizer suite and
private oracle are committed by SHA-256 in `configs/oracle_commitment.json`.
The evaluator does not require the private oracle to score a public case.

## Validation evidence

The Revision 3 test run passed `170 passed, 3 skipped`. Two skips exercise
symlinks unavailable on this Windows host; the third is an opt-in additional
live-xTB test. Native xTB was used to generate every frozen failure case. The
model-independent strong reference passed all six cases at 100. A correct
diagnosis-only reference received raw 47 but official 0 on every case because
recovery gates failed. A fixed deliberately wrong-science policy was capped at
20 per case (18–20 observed ungated) and official 0. These are evaluator calibration
baselines, not claims about language-model performance.

The current organizer authority is the release-root `official_runs/q08/` view.
Offline replay of unchanged archived trajectories with grader
`xtb-failure-suite-v2-replay-1.1` gives DeepSeek 19.5, Kimi 35.3333
(`INFRA_DEGRADED`), and GLM 18.5. The bundled `MODEL_COMPARISON.md` and related
`m/r3` report values are historical first-round snapshots retained for provenance;
they are not the current score authority.

The post-audit 18-case aggregation (three models × six cases) finds 8 eligible
and 10 ineligible cases, 10 official-zero cases, 14 scientific-core caps, and 6
`INFRA_DEGRADED` cases. Five correlated gates fail in all 10 ineligible cases
(`complete_lifecycle`, decisive evidence, final schema, current artifacts, and
Hessian completion); the budget gate fails only 3. Point loss is distributed
across diagnosis, lifecycle, evidence/schema, provenance, Hessian, and recovery
components. There is therefore no sole gate whose relaxation is justified by the
three low suite totals, and grader v1.1 is unchanged. The machine-readable count
and deduction table is delivery evidence
`revision_records-1/evidence/task_08_failure_aggregation_r3/task_08_failure_aggregation.json`.

## Key files

- `TASK.md` and `configs/task.yaml`: agent-visible instructions and authority.
- `benchmark_case/`: public native case and hash-bound evidence.
- `tools/`, `framework/`, `evaluator/`: restricted execution and independent scoring.
- `baselines/run_suite_baselines_v2.py`: calibration suite runner.
- `reports/`: design, scientific, reproducibility, and model-status records.
- `LICENSES.md`: code and scientific-software license/citation notices.

The participant and organizer archives are built from explicit release
allowlists. Caches, developer retries, raw provider responses, local paths,
credentials, and prior archives are excluded. Every released member is bound by
`MANIFEST.json`, and archive SHA-256 values are recorded outside each archive to
avoid self-reference.

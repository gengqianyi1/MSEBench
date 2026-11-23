# QED Molecular Design Agent Benchmark v4

Version 4 is a five-case constrained molecular-design suite. One acetanilide
ester case is public; four organizer-only cases use different starting
molecules, scaffold families, initial QED values, property windows, similarity
thresholds, and local substitution spaces. The hidden families are aryl
sulfonamide, heteroaryl amide ester, benzyl oxazolidinone, and fluoroaryl urea.

The model must propose and measure multiple local edits, use tool feedback in at
least two candidate rounds, and submit an already measured candidate. RDKit
recomputes validity, canonical SMILES, QED, molecular weight, Crippen LogP,
TPSA, Lipinski HBD/HBA, rotatable bonds, ring count, halogen count,
substructure constraints, and raw Morgan-radius-2/2048-bit Tanimoto similarity.
Threshold decisions use unrounded values; rounding is presentation-only.

QED in this benchmark is the frozen RDKit quantitative estimate of
drug-likeness used only as an optimization surrogate. It is not a measurement or
claim of drug efficacy, clinical activity, safety, or developability.

## Score design

Each case contributes at most 90 points: 85 for final feasible chemical quality
on a case-specific QED interval and 5 for a genuine feedback-driven revision
that improves over an earlier measured candidate. Calling tools or satisfying a
minimum count does not itself add points. Process requirements are eligibility
gates. The remaining 10 suite points reward stability across completed cases,
which is distinct from any one case's final QED.

`BLOCKED` provider runs are excluded from capability means and make full-suite
ranking ineligible. `FAIL` remains an evaluable zero/partial diagnostic outcome.
The evaluator reports scientific and process components separately.

## Output contract

The only model-authored terminal response is:

```json
{"final_candidate":{"name":"short_identifier","smiles":"one_connected_molecule","rationale":"brief explanation"}}
```

`TASK.md`, `prompts/agent_prompt.md`, `configs/task.yaml`, and the runner all use
that exact structure. The runner then creates a separate
`runner_final_answer/2.0` persisted envelope containing the terminal candidate,
candidate history, and trusted run metadata. The model never authors those
trusted fields.

## Public/private boundary

The participant archive contains the public config and
`configs/suite_public.yaml`; it contains no `organizer_private` tree, hidden
case config, hidden reference candidate, organizer trajectory, or search cache.
The organizer archive contains all five configs and calibration evidence. The
public suite descriptor discloses only counts and aggregation rules.

## Reproduce

```text
python -m pytest -q
python baselines/run_suite_baselines.py --suite organizer_private/suite.yaml --output <dir>/deterministic_beam --mode deterministic_beam
python baselines/run_suite_baselines.py --suite organizer_private/suite.yaml --output <dir>/shallow_local --mode shallow_local
python scripts/run_suite_models_v4.py --output <new-dir>/formal_models_20260805
```

The recalibrated deterministic one-round RDKit beam achieved 31.3753/100
(case macro 21.8814 plus stability 9.4939). It uses the same public per-case
information and never reads reference molecules or cross-case priors. A public
halogen-count constraint prevents one fixed F/Cl-decoration edit from solving
all five families while retaining independently verified non-halogen feasible
improvements. The beam is a weak offline calibration comparator, not a
language-model result.

The formal campaign command loads credentials only through the configured
environment-variable name. Never place a key in YAML. It creates a new,
exclusive output root, records exact requested/actual model IDs, and verifies
every stored evaluation by offline replay.

# QED Molecular Design Agent Benchmark v4

## Objective

For each injected case, use the public RDKit tools to propose and measure local
molecular edits. Submit one already measured, feasible candidate whose QED is
strictly greater than that case's starting molecule. The benchmark measures
planning, tool use, feedback-driven revision, and trade-offs across public hard
constraints. It is not a single QED calculation or a model-training task.

The organizer suite contains five scaffold-diverse cases. The participant
package exposes the development case and the suite counts; the organizer
injects each hidden case through the same schema. The current case YAML is the
authoritative source for its start, thresholds, budgets, and seed.

## Public development case

Starting SMILES:

```text
CCOC(=O)c1ccc(NC(=O)C)cc1
```

Hard constraints:

- one connected, neutral, RDKit-valid molecule;
- allowed atomic numbers: 6, 7, 8, 9, 16, 17;
- molecular weight 190–290;
- Crippen LogP 1.0–3.0;
- TPSA 45–70;
- H-bond donors exactly 1 and acceptors 3–4;
- rotatable bonds 2–5 and ring count 1–2;
- halogen count exactly 0;
- Morgan radius-2/2048-bit Tanimoto similarity to the start at least 0.50;
- match the configured anilide-bearing aromatic scaffold SMARTS.

`halogen_count` is the number of F, Cl, Br, I, and At atoms computed from the
sanitized RDKit molecule. It is a public hard constraint. Cases 1–4 require
zero; case 5 requires exactly its one starting fluorine. This prevents one
fixed halogen-decoration edit from solving every case while retaining verified
non-halogen feasible improvements.

## Public tools

- `validate_smiles`
- `canonicalize_smiles`
- `calculate_qed`
- `calculate_properties`
- `calculate_similarity`
- `check_scaffold`
- `check_substructure`
- `evaluate_candidate`

The tools never edit a molecule. The agent must propose new SMILES and use the
returned measurements. `evaluate_candidate.feasible` reports current hard
constraints only; strict QED improvement is applied to the final selection.

## Process budget and eligibility

- at most 12 public tool calls;
- at most 6 unique `evaluate_candidate` molecules;
- at most 12 agent turns and 900 seconds;
- at least 3 successfully replayable public RDKit calls;
- at least 2 distinct evaluated candidates in at least 2 feedback rounds;
- the final candidate must have been evaluated in this trajectory.

Unknown, rejected, malformed, or failed calls consume budget but do not satisfy
minimum successful-call requirements. Counts and candidate history are rebuilt
from the event trajectory; model-authored counts are not trusted.

## Terminal output

Finish by calling `submit_candidate` exactly once with the final candidate's
fields directly as arguments — flat, with no outer envelope. The parameter
schema is `name`, `smiles`, `rationale` (all required, all strings):

```json
{
  "name": "short_identifier",
  "smiles": "one_connected_molecule",
  "rationale": "brief explanation"
}
```

The call is terminal: do not wrap it in a `final_candidate` object, and do not
continue working after it. If native tool calls are unavailable for your
provider, return the same object as
`{"candidate": {...}}` (a `{"final_candidate": {...}}` envelope is accepted as
a legacy alias, but the flat arguments above are preferred).

The runner creates the trusted persisted envelope, trajectory, dialogue,
metadata, and evaluation. The evaluator independently recomputes chemistry,
process evidence, and eligibility.

## Scoring

Each case contributes at most 90 points: 85 for final feasible chemical quality
on its configured QED floor-to-target interval and 5 for a genuine later
measured revision. Tool counts add no points; they are eligibility gates. The
suite adds up to 10 cross-case stability points. `BLOCKED` is reserved for an
external provider failure with no evaluable model trajectory and is excluded
from capability means; an evaluable scientific or protocol failure is `FAIL`.

The suite score is
`macro(case scores) + 10 * min(case scores) / max(macro(case scores), 1)`,
clipped to 100. The second term is `cross_case_stability_score`; it rewards
consistent performance across the five cases and is part of the official
reward. `PASS` means the submission is protocol-valid and numerically
scorable, not that it met a scientific score threshold. Process checks are
eligibility gates and therefore no numeric `process_score` is reported.
`release_complete` and `ranking_eligible` are compatibility field names for
terminal suite coverage; neither field is task publication approval.

## Reproduction

```text
python -m pytest -q
python baselines/run_suite_baselines.py --suite organizer_private/suite.yaml --output <new-dir>/beam --mode deterministic_beam
python scripts/run_suite_models_v4.py --output <new-dir>/formal_models_20260805
```

Model configuration contains environment-variable names only. Never store an
API key in YAML, a trajectory, a report, or an archive.

---

## ChemBench v2.1 Harbor execution contract

The evaluated model is supplied by the Harbor Agent layer; this task must not call the evaluated LLM directly. Work only inside `/workspace`. Required legacy artifacts: `final_answer.json`, `benchmark_result.json`, `benchmark_run/campaign_summary.json`, `benchmark_run/*/suite_evaluation.json`.

**Migration status:** Harbor protocol, parity, leakage, and reference-optimum gates verified for release. See `MIGRATION.md`.

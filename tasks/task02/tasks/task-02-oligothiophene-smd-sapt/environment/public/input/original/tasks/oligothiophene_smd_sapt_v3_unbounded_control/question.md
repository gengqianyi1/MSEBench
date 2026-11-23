# Benchmark Task V3.0: Evidence-Audited Solvent Effects Under a Fixed Compute Budget

This benchmark evaluates whether you can reconstruct a labeled oligomer, audit imperfect evidence, execute a fixed computational protocol, and make scientifically bounded conclusions from raw outputs.

At runtime, the workspace contains:

- `instance.json`: the public molecular instance, seeds, thresholds, and compute budget;
- `protocol.yaml`: the mandatory external-tool protocol and resource settings;
- `benchmark_tools.py`: the only permitted interface for xTB, PySCF, and Psi4;
- `stage_A_schema.json`, `stage_B_schema.json`, and `stage_C_schema.json`: exact stage contracts;
- `result_schema.json`: the final-result JSON contract;
- `seed_evidence/`: supplied artifacts. One or more may be stale, mislabeled, geometrically inconsistent, or numerically inconsistent; the anomaly count is intentionally not disclosed.

All executable work must use:

```text
run_python(script, timeout_seconds)
```

## Delivery and scoring contract

The stage JSON files are the completion records used by the runner and scorer. Computations, logs, or intermediate files that are not summarized in the required stage JSON do not complete that stage.

You are responsible for managing the tool-call budget and writing each JSON file as soon as its stage is complete. The runner does not reserve calls, repair outputs, create missing JSON files, or provide a degraded final submission.

A complete submission requires all of the following:

```text
stage_A_audit.json
plan.json
stage_B_bulk.json
stage_C_local.json
result.json
revisions.json
```

The benchmark does not accept a partial or downgraded `result.json`. `result.json` must represent the completed workflow and validate against `result_schema.json`.

## Tool-call budget

- This control run has no effective tool-call limit and no tool-call penalty.
- The runner remains subject to its wall-time limit.
- Rejected calls are recorded. Inspect the rejection rules below before submitting code.

## Scoring

The maximum score is 100 points:

| Scored area | Points |
|---|---:|
| Input audit, molecular identity, solvent identity, and hidden anomalies | 16 |
| Conformer search and evidence-driven experimental design | 12 |
| Fixed-common-geometry M05-2X/6-31G*/SMD evidence and result checks | 30 |
| n-hexane/water endpoint geometry and evidence lineage | 10 |
| SAPT dominant attractive interaction analysis, with valid evidence as an eligibility gate | 14 |
| Workflow integrity and reproducibility | 8 |
| Independent scientific scope and uncertainty | 10 |

The first 90 points are determined from submitted files, raw logs, coordinates, provenance, and execution records. The final 10 points are assigned by an independent evaluator for the scientific scope and interpretation requested below. Candidate and evaluator model identities must differ.

The deterministic rubric is non-overlapping. Four-solvent SMILES/structure/alias validation contributes 2 points total, 0.5 point per solvent. The SMD section independently parses the five raw formal logs, checks the fixed common geometry and protocol, scores extraction of the five raw total energies without comparing their absolute values with a reference, recomputes same-run solvation-energy differences, and scores the complete solvent order. The first-ranked entry may only be described as the solvent with the lowest calculated solvation free energy under this protocol; it is not a separately scored solvent choice and must not be generalized as the best solvent. The SAPT section does not compare component vectors or totals with a withheld numerical tolerance. Its only scored result is whether the dominant attractive interaction is correctly identified for each endpoint. A parseable SAPT0 log, correct fragment identity, and geometry lineage are eligibility requirements and carry no separate points. Tool-call efficiency is not scored in this unbounded control run.

### Detailed point allocation

| Section | Sub-item | Points |
|---|---|---:|
| Input audit and identity | Oligomer formula/connectivity/charge/multiplicity | 3 |
|  | Atom mapping, inter-repeat bond, torsion and terminal-atom definitions | 3 |
|  | Four solvent SMILES–molecular-graph–program-alias checks, 0.5 each | 2 |
|  | Hidden anomaly localization, type, evidence, downstream isolation, and false-positive control | 8 |
| Conformer search and design | At least four real, broadly distributed torsion-start geometries | 3 |
|  | Independent retained minima and justified `R_g`/sensitivity conformer selection | 3 |
|  | Parseable sensitivity calculation aimed at a stated uncertainty | 3 |
|  | Resource-aware scheduling and failure isolation | 3 |
| Fixed-geometry SMD | Fixed protocol and one common `R_g` coordinate hash | 4 |
|  | Five raw total energies extracted from the corresponding program logs | 6 |
|  | Four solvation-energy differences recomputed from logs and compared with withheld references | 8 |
|  | Correct complete solvent ordering | 8 |
|  | Parseable HOMO/LUMO and dipole evidence | 4 |
| Endpoint geometry and lineage | Correct endpoint bond graph and atom mapping | 3 |
|  | Recomputed RMSD/twist/end-to-end metrics agree with reported values | 3 |
|  | Explicit complexes use the corresponding selected endpoint geometry | 2 |
|  | Geometric diversity plus separate frozen-interaction/deformation bookkeeping | 2 |
| SAPT | Correctly supported dominant attractive interaction for n-hexane | 7 |
|  | Correctly supported dominant attractive interaction for water | 7 |
| Workflow integrity | Input/output hashes and program/protocol provenance | 2 |
|  | No semantic modification of frozen stage files | 2 |
|  | Append-only revisions record | 1 |
|  | Complete schema-valid stage/final files | 1 |
|  | Avoidable script/API errors and successful recovery | 2 |
| Independent science | Continuum scope, orbital-gap scope, population scope, geometry-specific SAPT scope, and quantitative next calculation, 2 each | 10 |

SAPT component values and SAPT totals are not compared with withheld numerical tolerances. Each endpoint receives 7 points only when a parseable valid SAPT0 log, correct fragment/geometry lineage, the raw-component dominance, the reported dominance, and the withheld endpoint classification all agree. These evidence checks are gates, not separately scored SAPT items.

No tool-call penalty is applied in this unbounded control run. All scientific, evidence-integrity, schema, and stage requirements remain unchanged.

Evidence-integrity failures can cap the final score even when section subtotals are higher:

| Condition | Final-score cap |
|---|---:|
| Reported numerical claims contradict raw evidence | 25 |
| `result.json` fails schema validation | 30 |
| A hidden anomaly contaminates a downstream branch that is nevertheless claimed as validated | 55 |
| A frozen earlier-stage file is modified, or later-stage outputs bypass the staged workflow | 60 |
| Tool-call limit is exceeded, or production use precedes the required micro-test | 70 |

When several caps apply, the lowest cap controls. Missing, incomplete, inconsistent, or non-reproducible evidence receives no credit for the associated scoring item. This section states the public allocation and integrity policy only. Withheld SMD reference values and endpoint SAPT-dominance labels are available only to the scorer; no reference numerical values or expected endpoint labels are supplied to the candidate.

## Rejected behavior

The static safety checker rejects the entire submitted script before execution when it detects any of the following. A rejected call still consumes budget.

### Network access

Static or dynamic imports rooted at any of these modules are rejected:

```text
socket, urllib, requests, ftplib, smtplib, telnetlib, http
```

### Direct external-chemistry access

Direct imports of `xtb`, `pyscf`, or `psi4`, direct access to their executables, and candidate-created subprocesses are rejected. Use only the public functions in `benchmark_tools.py`; those wrappers perform the controlled external-chemistry launches.

### Destructive or process-control calls

The following calls, including equivalent attribute forms ending in these method names, are rejected:

```text
os.system
os.popen
os.remove
os.rmdir
os.removedirs
os.unlink
os.chmod
os.chown
os.kill
shutil.rmtree
*.unlink
*.rmdir
*.chmod
*.chown
*.kill
```

Use non-destructive versioned paths instead of deleting or overwriting evidence.

### Subprocess restrictions

Candidate scripts may not start secondary subprocesses. Perform Python work in the current `run_python` script and use `benchmark_tools.py` for xTB, PySCF, and Psi4. The following are rejected:

- every candidate `subprocess.*` call;
- `shell=True` or shell command execution;
- unknown tools other than `run_python`;
- malformed JSON tool arguments;
- scripts exceeding the configured byte limit.

Do not install packages or call `pip`, `conda`, shell validators, or external validation commands. Workspace validators are Python modules; import and call their public functions directly inside `run_python`. Candidate file reads and writes are restricted to the workspace; runner/scorer files outside the workspace are not accessible.

### Protected benchmark inputs and stage integrity

The following runner-owned files are immutable for the entire run:

```text
instance.json
protocol.yaml
benchmark_tools.py
stage_A_schema.json
stage_B_schema.json
stage_C_schema.json
result_schema.json
stage_validation.py
```

If a script changes or deletes one of these files, the runner restores it, marks the call rejected, and the call still consumes budget.

Once the runner advances to a later stage, earlier stage JSON files are also immutable. Corrections must be appended to `revisions.json`. Modifying a frozen stage file is recorded and penalized.

## Standard external-tool protocol

Read `protocol.yaml` before any chemistry calculation. Do not replace its methods, convergence settings, grid settings, charge, multiplicity, thread limits, output conventions, or job timeout classes.

Use only these functions from `benchmark_tools.py`:

```python
run_xtb_microtest(...)
run_xtb_optimization(...)
run_xtb_single_point(...)
run_pyscf_microtest(...)
run_pyscf_formal_single_point(...)
run_psi4_microtest(...)
run_psi4_sapt0(...)
```

Canonical production signatures are:

```python
run_xtb_optimization(input_xyz, output_dir, solvent=None, timeout_seconds=1800)
run_xtb_single_point(input_xyz, output_dir, solvent=None, timeout_seconds=1800)
run_pyscf_formal_single_point(geometry_file, solvent, log_file, sensitivity_mode=None)
run_psi4_sapt0(monomer_a_xyz, monomer_b_xyz, output_dir, solvent_label="water", timeout_seconds=7200)
```

The helper reads `protocol.yaml` and `instance.json` and rejects unsupported solvents or sensitivity modes. Method, basis, charge, multiplicity, grids, convergence, threads, memory, and optimization settings are not candidate-selectable, except for the explicitly permitted sensitivity mode.

Direct xTB, PySCF, or Psi4 use is not permitted. The helper module fixes working directories, input placement, environment variables, command construction, and benchmark metadata.

Before the first production use of each external program, run its corresponding micro-test and save the output under `microtests/`.

The runner may raise an undersized requested timeout to the fixed timeout floor for the detected job class. The effective timeout is reported in the tool result. The total benchmark wall-time limit still applies.

## Global evidence constraints

- Do not hard-code computed energies, formulae, rankings, classifications, or scientific-answer booleans.
- A numerical claim receives credit only when the scorer can reproduce it from a referenced raw file.
- Preserve failed attempts with structured evidence.
- Every supported anomaly must be detected from the files themselves and excluded, repaired, or regenerated before downstream use. Do not assume that finding one anomaly completes the audit, and do not invent unsupported anomalies.

## Molecular reconstruction

Read the repeat unit, connection sequence, oligomer length, charge, and multiplicity from `instance.json`. Reconstruct the capped oligomer rather than assuming a supplied final structure is correct.

Save independently verifiable evidence for:

- labeled dummy-site interpretation;
- every inter-repeat bond;
- removal of all dummy atoms;
- terminal hydrogen capping;
- sanitization and connected-component count;
- formula, canonical representation, atom counts, charge, and multiplicity;
- atom mapping for the inter-repeat bond, one sulfur per repeat, and terminal carbons.

Save the validated structure as SDF and XYZ.

## Solvent identity audit

For n-hexane, 1,4-dioxane, chloroform, and water, verify that the supplied SMILES encodes the named molecular graph and that the SMD name and xTB ALPB alias refer to the same solvent. Record this under `solvent_identity_audit` in `stage_A_audit.json`. This item is worth 2 points total, 0.5 point per solvent. Equivalent SMILES are accepted after canonical graph comparison; literal string identity is not required.

## Formal bulk-solvation calculation

Use the exact formal method and solvent names in `protocol.yaml` and `instance.json`. Gas and all solvent calculations must use the same saved atom order and fixed coordinates.

Every formal log must contain actual program output and these metadata lines:

```text
BENCHMARK_PROGRAM: ...
BENCHMARK_FUNCTIONAL: ...
BENCHMARK_BASIS: ...
BENCHMARK_REFERENCE: ...
BENCHMARK_SOLVENT: gas-or-solvent-name
BENCHMARK_CHARGE: ...
BENCHMARK_MULTIPLICITY: ...
BENCHMARK_GEOMETRY_FILE: relative/path.xyz
BENCHMARK_TOTAL_ENERGY_HARTREE: ...
BENCHMARK_CONVERGED: true-or-false
```

Compute coordinate SHA-256 hashes using the serialization in `instance.json`. Derive all differences from parsed raw logs.

Run the complete common-reference set for:

```text
gas
n-hexane
1,4-dioxane
chloroform
water
```

Also run one permitted sensitivity calculation and explain why it is informative. Report the complete calculated solvation-free-energy ordering, identify its first entry only as the solvent with the lowest calculated solvation free energy under this protocol, and state whether the complete ordering is preserved by the sensitivity calculation. Do not describe this result as selection of a best solvent.

## Stage B canonical JSON contract

`stage_B_bulk.json` must validate against `stage_B_schema.json`. In particular, it must contain this canonical structure:

```json
{
  "stage": "B",
  "gas_conformer_search": {
    "attempts": [],
    "retained_conformers": [],
    "selected_R_g": {
      "geometry_file": "structures/R_g.xyz",
      "coordinate_sha256": "..."
    }
  },
  "formal_calculations": {
    "method": {},
    "geometry_file": "structures/R_g.xyz",
    "energies_hartree": {
      "gas": 0.0,
      "n-hexane": 0.0,
      "1,4-dioxane": 0.0,
      "chloroform": 0.0,
      "water": 0.0
    },
    "log_files": {
      "gas": "logs/formal/gas.log",
      "n-hexane": "logs/formal/n-hexane.log",
      "1,4-dioxane": "logs/formal/1,4-dioxane.log",
      "chloroform": "logs/formal/chloroform.log",
      "water": "logs/formal/water.log"
    },
    "solvation_energies_kcal_mol": {}
  },
  "sensitivity_calculation": {
    "type": "...",
    "rationale": "...",
    "log_files": [],
    "ranking_preserved": true
  },
  "ranking": {
    "lowest_solvation_free_energy_solvent": "...",
    "ranking_robust_under_sensitivity": true,
    "near_tie": false
  },
  "provenance": []
}
```

Before Stage C, the runner calls `validate_stage_B(stage_B_bulk.json)`. Validation requires:

- parseable JSON matching `stage_B_schema.json`;
- all five formal energies and all five log paths;
- existing and parseable raw formal logs for all five systems;
- one common `structures/R_g.xyz` geometry reference;
- `analysis/formal_coordinate_hashes.json` and matching geometry provenance. This file must contain:

  ```json
  {
    "geometry_file": "structures/R_g.xyz",
    "coordinate_sha256": "...",
    "systems": {
      "gas": "...",
      "n-hexane": "...",
      "1,4-dioxane": "...",
      "chloroform": "...",
      "water": "..."
    }
  }
  ```

  Every system hash must equal the common `R_g` coordinate hash;
- at least one distinct parseable sensitivity log;
- a permitted sensitivity type;
- numerical and robust ranking fields.

A merely existing but incomplete Stage B file does not advance the workflow.

You can run the same validator yourself inside `run_python`:

```python
from stage_validation import validate_stage_B
errors = validate_stage_B("stage_B_bulk.json", ".")
assert not errors, errors
```

Do not spawn a separate process for this check; call the imported validator function in the same Python script.

## Endpoint conformational analysis

Use the same retained gas conformers as starts for both endpoint solvents. Optimize with the exact implicit-solvent protocol in `protocol.yaml`.

Recompute from selected coordinate files:

- Kabsch-aligned heavy-atom RMSD;
- mapped inter-repeat dihedral;
- twist from planarity;
- terminal-carbon end-to-end distance.

Do not compare absolute implicit-solvent total energies across different solvents. State what sampled local minima do and do not establish.

## Explicit complexes and SAPT

For each endpoint solvent:

- generate the fixed number of structurally distinct 1:1 starting arrangements;
- quantify diversity geometrically;
- optimize and apply intactness/dissociation checks;
- distinguish frozen-geometry interaction energy from monomer deformation energy;
- select representatives using energetic and structural evidence;
- run the fixed SAPT protocol in `protocol.yaml`.

Save raw logs and geometries. Identify the largest-magnitude attractive component separately for each successful representative from the parsed raw components. SAPT numerical component values are not compared with a withheld reference tolerance; the scored scientific result is the correctly supported dominant attractive interaction for n-hexane and water.

## Required workflow

### Stage A — Preflight, reconstruction, and evidence audit

Write:

```text
stage_A_audit.json
plan.json
structures/solute_validated.sdf
structures/solute_validated.xyz
analysis/atom_mapping.json
analysis/accepted_input_hashes.json
microtests/...
```

Audit every file under `seed_evidence/`, report every supported anomaly you find, explain file-level evidence, and record how each affected artifact was excluded, repaired, or regenerated. The number of anomalous files is not disclosed.

### Stage B — Gas conformers and bulk-solvation robustness

Write:

```text
stage_B_bulk.json
structures/R_g.xyz
analysis/formal_coordinate_hashes.json
logs/formal/...
```

Stage B is complete only after `validate_stage_B` succeeds.

### Stage C — Endpoint conformational and local-interaction analysis

Write:

```text
stage_C_local.json
logs/alpb/...
logs/complexes/...
logs/sapt/...
```

Include common starts, endpoint selections, recomputed geometry metrics, complex diversity, frozen interaction and deformation terms, representative selection, SAPT outputs, and failures.

For each endpoint, the Stage C record must make the evidence chain machine-auditable: identify the selected conformer coordinate file and its raw optimization evidence; report metrics derived from saved coordinates; associate each explicit-complex arrangement with its geometry and integrity status; record a geometric diversity assessment; identify the selected representative and the evidence used for that selection; report frozen interaction and deformation as separate quantities; and associate each SAPT result with its raw log and fragment geometry. Field organization is flexible, but file references and numerical claims must be unambiguous. The benchmark does not prescribe which conformer, arrangement, component, or scientific conclusion should be selected.

### Stage D — Decision and uncertainty

Write:

```text
result.json
revisions.json
```

Assemble the completed workflow from immutable stage files and raw evidence. `result.json` must validate against `result_schema.json`.

## Scientific questions

Answer from evidence without assuming an expected conclusion:

1. What physical scope does the bulk-solvation calculation cover, and what would be required to infer an experimental solubility ordering?
2. What quantity is represented by the orbital-energy difference, and what calculation would be needed for an optical excitation?
3. What can the selected implicit-solvent local minima establish about conformational response, and what sampling would be needed for populations?
4. What can each SAPT representative establish about local interactions, and what evidence would be needed for a dominant solution motif?
5. What single next calculation would most reduce the remaining uncertainty, and what quantitative metric and threshold would change the conclusion?

Persuasive prose cannot replace raw evidence. A calculation failure must be repaired within the budget; the final workflow must still be complete.

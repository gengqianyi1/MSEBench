# MSEBench Task: Five-Round xTB-DIPRO Molecular Optimization for High Marcus Hole Hopping Rate

## Task overview

You are participating in a fixed-round closed-loop molecular design benchmark.

Your task is to optimize small-molecule organic semiconductor backbones for high computed hole hopping rate. In each round, you will propose exactly one structurally valid candidate molecule. An external evaluator will then run the standardized RDKit/xTB/DIPRO/Marcus workflow and return computed feedback.

You do **not** need to run RDKit, xTB, DIPRO, or Marcus-rate calculations yourself. You only need to propose the next molecule and explain the intended structural modification.

Molecular design stops after exactly **5 design iterations**. After round 5, the evaluator summarizes R1-R5 and you must select one existing `candidate_id` as the final recommendation. You may not create a sixth molecule.

---

## Design objective

Design a neutral organic molecule with a **five-ring fused backbone** and maximize the computed Marcus hole hopping rate:

```text
marcus_hole_rate_s_inv
```

The primary optimization direction is:

```text
maximize marcus_hole_rate_s_inv
```

The target-hit threshold is:

```text
marcus_hole_rate_s_inv > 1.0e14 s^-1
```

Higher rates above this threshold are better.

---

## Required molecular class

Each proposed molecule must be a small-molecule organic semiconductor backbone containing a five-ring fused core.

For this task, the machine-evaluated five-ring fused-core constraint means:

1. The molecule contains a single connected fused-ring backbone.
2. The largest fused-ring cluster contains exactly **5 rings**.
3. Rings in the fused cluster are connected through shared ring bonds, not merely through single bonds or spiro atoms.
4. No additional rings are allowed outside the five-ring fused core.
5. The molecule should represent a plausible small-molecule organic semiconductor backbone rather than an unrelated drug-like, ionic, polymeric, or highly flexible molecule.

Planarity and pi-conjugation are chemical design guidance because they can
improve electronic coupling; they are not separate hard gates.  The evaluator
does not infer a binary conjugation label or reject a candidate using a hidden
planarity threshold.  Geometry and electronic effects instead enter through
the disclosed RDKit/xTB/DIPRO/Marcus calculation.

Do not propose disconnected fragments, salts, polymers, wildcard structures, metal-containing structures, or charged molecules.

---

## Hard constraints

A candidate is considered constraint-valid only if all of the following are satisfied by the external evaluator:

1. **SMILES validity**
   - The submitted SMILES must be valid according to RDKit.

2. **Single component and Neutral molecule**
   - The molecule must be one connected molecular component.
   - Mixtures, salts, and dot-disconnected SMILES are invalid.
   - Total formal charge must be 0.

3. **Allowed elements**
   - Allowed atoms: C, H, N, O, S, Se, F, Cl.
   - Any other element is invalid.
   - The molecule must not contain `*` or query atoms.

4. **Five-ring fused core**
   - RDKit ring analysis must identify exactly one fused-ring cluster.
   - The largest fused-ring cluster must contain exactly 5 rings.
   - No rings may occur outside this largest fused-ring cluster.

5. **Size range**
   - Heavy atom count must be less than 50.
   - Molecular weight must be less than 750 Da.

6. **Uniqueness**
   - Do not repeat a previously evaluated canonical SMILES.

---

## Interaction protocol

There are exactly **5 rounds**.

In each round:

1. You submit exactly one candidate molecule.
2. The external evaluator computes molecular validity, fused-ring constraints, xTB-derived properties, DIPRO coupling, and Marcus rate.
3. The evaluator returns structured feedback.
4. For rounds 2-5, use the previous computed feedback to decide the next modification.

The benchmark continues until 5 rounds are complete, even if a target hit is found early.

---

## What the external evaluator computes

For each candidate, the evaluator performs the following standardized workflow.

### 1. RDKit validation

The evaluator will:

- parse the SMILES;
- canonicalize the molecule;
- check formal charge;
- check allowed elements;
- check number of components;
- check ring structure;
- identify fused-ring clusters;
- verify the five-ring fused-core constraint;
- add hydrogens;
- generate multiple 3D conformers;
- perform force-field pre-optimization and select the lowest-energy RDKit conformer;
- write monomer XYZ files.

### 2. xTB geometry optimization

The evaluator will run GFN2-xTB geometry optimization for the neutral molecule.

If geometry generation or xTB optimization fails, the candidate remains recorded but does not count as a successful computed candidate.

### 3. Hole reorganization energy

The evaluator computes hole reorganization energy using a four-point protocol:

```text
lambda_h_eV =
[E_plus(Q0) - E_plus(Qplus)] + [E_neutral(Qplus) - E_neutral(Q0)]
```

where:

```text
Q0     = optimized neutral geometry
Qplus  = optimized cation geometry
E_neutral(Q0)     = neutral single-point energy at Q0
E_neutral(Qplus)  = neutral single-point energy at Qplus
E_plus(Q0)        = cation single-point energy at Q0
E_plus(Qplus)     = cation single-point energy at Qplus
```

If `lambda_h_eV <= 0`, the candidate is treated as a failed computed candidate.

### 4. Lowest-energy slipped cofacial dimer packing search

For each successfully optimized monomer, the evaluator constructs a grid of slipped cofacial dimers and selects the lowest-energy packing by xTB single-point dimer energy.

The monomer is aligned by principal axes using the heavy-atom coordinates:

```text
x-axis = largest principal axis
y-axis = second principal axis
z-axis = smallest principal axis, approximate molecular-plane normal
```

The packing search scans the following displacements:

```text
dx in {0.0, 1.0, 1.5, 2.0} Angstrom
dy in {-1.0, 0.0, 1.0} Angstrom
dz in {3.0, 3.2, 3.4, 3.6, 3.8, 4.0} Angstrom
```

For each grid point, the second monomer is generated as:

```text
monomer B = monomer A + (dx, dy, dz)
```

The dimer with the lowest successful xTB single-point total energy is selected for the DIPRO transfer-integral calculation. This is a standardized hypothetical packing-search model used only for fair comparison across candidates.

### 5. DIPRO transfer integral

The evaluator computes the intermolecular hole transfer integral using xTB DIPRO on the selected lowest-energy slipped cofacial dimer.

The reported coupling is:

```text
Vh_DIPRO_eV = abs(raw_hole_transfer_integral_eV)
Vh_DIPRO_meV = 1000 * Vh_DIPRO_eV
```

If DIPRO fails, the candidate is recorded but does not count as a successful computed candidate.

### 6. Marcus hole hopping rate

The evaluator computes the hole hopping rate at 300 K:

```text
T = 300 K
kB = 8.617333262145e-5 eV K^-1
hbar = 6.582119569e-16 eV s
```

Formula:

```text
k_h = (2*pi/hbar) * Vh^2 / sqrt(4*pi*lambda_h*kB*T) * exp[-lambda_h/(4*kB*T)]
```

where:

```text
Vh = Vh_DIPRO_eV
lambda_h = lambda_h_eV
```

The final rate is reported as:

```text
marcus_hole_rate_s_inv
```

---

## Evaluator feedback after each round

After each round, the evaluator will return structured computed feedback including:

```text
candidate_id
canonical_smiles
constraint_flags
ring_count_total
largest_fused_ring_cluster_size
heavy_atom_count
molecular_weight
lambda_h_eV
Vh_DIPRO_eV
Vh_DIPRO_meV
marcus_hole_rate_s_inv
selected_packing_dx_A
selected_packing_dy_A
selected_packing_dz_A
selected_packing_energy_hartree
packing_search_n_trials
packing_search_n_successful_trials
target_hit
best_rate_so_far_s_inv
best_candidate_id_so_far
```

Do not invent these values yourself. Use only the values returned by the evaluator.

---

## Required output format for each design round

Return only one JSON object and no extra text.

```json
{
  "candidate": {
    "candidate_id": "R1",
    "parent_candidate_id": null,
    "proposal_mode": "direct_smiles",
    "smiles": "...",
    "structural_modification": "...",
    "modification_reasoning": "..."
  }
}
```

For rounds 2-5, `parent_candidate_id` should refer to the previous or best candidate that your modification is based on.

---

## Field requirements

- `candidate.candidate_id`: string, use `R1`, `R2`, `R3`, `R4`, or `R5`.
- `candidate.parent_candidate_id`: string or null.
- `candidate.proposal_mode`: either `"direct_smiles"` or `"rdkit_edit_plan"`; this label is informational only — scoring always uses `smiles` and the remaining fields.
- `candidate.smiles`: exactly one molecular SMILES string.
- `candidate.structural_modification`: concise description of the structural change.
- `candidate.modification_reasoning`: concise explanation of why this change may improve one or more of:
  - hole reorganization energy;
  - DIPRO hole transfer integral;
  - Marcus hole hopping rate;
  - constraint satisfaction.

Do not include computed values unless they were returned by the evaluator in a previous round.

---

## Target-hit definition

A candidate is a target hit only if all of the following are true:

```text
valid = true
status = "success"
all hard constraints satisfied
five-ring fused core constraint satisfied
xTB reorganization-energy workflow succeeded
DIPRO calculation succeeded
lambda_h_eV > 0
Vh_DIPRO_eV > 0
marcus_hole_rate_s_inv > 1.0e14
```

---

## Scoring summary

The final score is 100 points and evaluates only the candidate explicitly recommended after round 5.

### Final recommendation gate (benchmark version 1.3)

After all five evaluator results, return exactly one JSON object such as `{"recommended_candidate_id":"R1"}`. Select only R1, R2, R3, R4, or R5. The summary includes each structure, hard-constraint status, `lambda_h_eV`, true hole coupling `Vh_DIPRO_eV`, Marcus rate, status, and failure reason. A missing/unknown ID, a new molecule, a hard-constraint failure, or any computation failure receives 0 points.

For an eligible recommendation, rates below `1.0e14 s^-1` receive 0. From `1.0e14` through `1.0e15 s^-1`, the score is linear: `min(100, 100 * rate / 1.0e15)`. Therefore `1.0e14` scores 10, `2.5e14` scores 25, and `1.0e15` or above scores 100. This is the only active scoring rule.

---

## Important reminders

- Submit exactly one molecule per round.
- Complete exactly 5 rounds.
- Do not repeat canonical SMILES.
- Keep the molecule neutral, organic, single-component, and within allowed elements.
- Focus on a five-ring fused conjugated semiconductor backbone.
- Do not report calculated `lambda_h_eV`, `Vh_DIPRO_eV`, or `marcus_hole_rate_s_inv`; the evaluator will compute them.
- Your only responsibility is to propose the next molecular modification based on evaluator feedback.

---

## MSEBench Harbor execution contract (protocol v2.1)

The evaluated model is supplied by the Harbor Agent layer; this task must not call the evaluated LLM directly. Work only inside `/workspace`. Agent-owned artifacts are the five round candidate files and `final_answer.json`. The verifier independently creates `result.json`; agents must not submit or rely on a self-reported result.

**Migration status:** release validation pending. The legacy scientific objective and scoring semantics are preserved; see `MIGRATION.md` for the remaining sidecar, Oracle, and parity release evidence.

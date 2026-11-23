# Molecular Design Benchmark: Coupled Descriptor Windows and Compensating Edits

## Background

You are participating in an iterative molecular design benchmark.

A starting molecule is provided as a SMILES string. Your task is to design a structurally
related molecule with improved QED (drug-likeness) while **simultaneously** satisfying a set
of competing hard constraints: property windows, synthetic accessibility, structural-alert
avoidance, and structural similarity to the starting molecule.

This benchmark evaluates whether a model can:
- understand the starting molecule and a multi-objective optimization goal;
- propose valid molecular structures;
- use evaluator feedback over multiple rounds;
- improve QED **without** breaking any other hard constraint.

The core difficulty is that the objectives are **coupled**. The starting molecule already sits
inside all three property windows. The part of the molecule that holds QED down is also the part
carrying a large share of the mass and of the polar surface, so the edits that raise QED most
directly push molecular weight toward its floor and LogP toward its ceiling at the same time. A QED
gain
therefore has to be paid for with a second, offsetting structural change that holds the molecule
inside every window at the same time. Edits that raise QED can also introduce structural alerts or
inflate the synthetic-accessibility score.

## What Is QED?

QED stands for Quantitative Estimate of Drug-likeness. It is a **surrogate** property: a published
desirability function over eight computed descriptors, evaluated here with RDKit `QED.qed`. It
ranges from 0 to 1 and higher means more drug-like *by that definition*. You do not need to compute
it yourself; the evaluator computes it after each submitted SMILES.

Every number in this task is a surrogate computed deterministically by RDKit from your SMILES
string. None is a measurement, none is simulated, and none of them is a measure of candidate quality
in a real drug programme: there is no potency, selectivity, ADMET or synthesis data anywhere in this
benchmark.

## What Is the SA Score?

The SA (synthetic accessibility) score is computed with the RDKit Contrib SA_Score model
(Ertl & Schuffenhauer). It ranges from about 1 (easy to synthesize) to 10 (hard to
synthesize). Adding strained rings, many stereocenters, or unusual functionality raises it.

## What Are Structural Alerts?

Structural alerts are substructures associated with assay interference, reactivity, or
toxicity. This benchmark uses the RDKit `FilterCatalog` combining three catalogs:

- PAINS (pan-assay interference compounds);
- Brenk (unwanted/reactive functionality);
- NIH (problematic functional groups).

A valid candidate must trigger **zero** alerts across all three catalogs.

## Objective

Given the starting molecule, propose a modified molecule that satisfies all hard constraints
and has QED higher than the starting molecule.

The optimization target is:

- Property: QED
- Direction: maximize

## Hard Constraints

A successful candidate must satisfy **all** of the following constraints:

1. **SMILES validity and admissibility** — the submission must be exactly **one neutral organic
   molecule** that RDKit can parse. Salts, mixtures, counter-ions, bare inorganic ions, isotope
   labels, atom-map numbers, wildcard atoms, radicals and elements outside
   {H, C, N, O, F, P, S, Cl, Br, I} are **rejected**, and a rejected submission is worth nothing.
   The evaluator reports a machine-readable reason code when it rejects an input.

2. **Molecular weight** — RDKit `Descriptors.MolWt` between **280 and 320** (narrow window).

3. **LogP** — Crippen LogP (`Crippen.MolLogP`) between **2.0 and 3.0** (narrow window).

4. **TPSA** — `rdMolDescriptors.CalcTPSA` between **30 and 100 A^2** (narrow window).

5. **Synthetic accessibility** — SA_Score **<= 3.5** (lower is easier to make).

6. **No structural alerts** — **0** PAINS + Brenk + NIH alerts combined.

7. **Structural similarity to the starting molecule** — Morgan fingerprint Tanimoto
   similarity **>= 0.44** (radius 2, nBits 2048).

Constraint 7 is an **edit budget**. Tanimoto similarity falls as more of the molecule is
rewritten, so it limits how much of the starting scaffold a single candidate may replace. Rewriting
the whole molecule into something more drug-like is not a legal move.

## Starting-Molecule Requirement

The submitted molecule must be based on the starting molecule. Do not submit an unrelated
molecule, a large scaffold-hop, a salt, a mixture, a metal-containing compound, a polymer,
or disconnected fragments. Similarity >= 0.44 is the operational check for relatedness;
admissibility (constraint 1) is enforced separately and independently.

## Interaction Rules

- There are **5 rounds**.
- In each round, submit exactly **one** molecule.
- Each round runs under a per-round agent budget declared in the public run configuration
  (`.chembench_agent.json`): at most **5 conversational turns and 5 tool calls per round**
  (`max_iterations` and `max_tool_calls`). The budget is hard: a round that ends with no
  parseable candidate — budget spent, reply unparseable, or no submission at all — records no
  candidate for that round, the whole run is INCOMPLETE and it receives reward 0 (see Scoring
  Summary).
- After each round, an external evaluator returns **computed property values only**:
  - SMILES validity;
  - molecular weight, LogP, TPSA;
  - QED;
  - SA score;
  - the number and names of any triggered structural alerts;
  - similarity to the starting molecule;
  - the best QED you have reached so far.
- The evaluator does **not** tell you which constraints you passed or failed, and does **not**
  tell you whether a submission counted as a success. Every window is specified in this document;
  you must compare the returned values against them yourself and track your own progress.
- Use the returned values to decide your next submission.
- Do not include explanatory text outside the required JSON object.
- Do not submit more than one molecule in a single round.

## Required Output Format

Return only a JSON object with exactly the following top-level structure:

```json
{
  "candidate": {
    "name": "candidate_round_1",
    "smiles": "CCN(CCC(N)=O)C(=O)Nc1ccccc1",
    "rationale": "Briefly describe the relationship between the submitted molecule and the starting molecule."
  }
}
```

## Field Requirements

- `candidate.name`: string, a short identifier.
- `candidate.smiles`: string, exactly one molecule.
- `candidate.rationale`: string, one or two concise sentences.

## Scoring Summary

The final benchmark score is 100 points, and **all 100 come from the objective**.

All five rounds are mandatory for ranking. A run that does not carry a parseable candidate for
every declared round is INCOMPLETE and receives reward 0 (its scientific score is retained only as
a diagnostic lower bound).

### Constraint satisfaction is a gate, not a scoring category

There are **no points for satisfying the constraints**. A molecule that violates a hard constraint is
not a partial success — it is not a usable molecule, so it contributes nothing. A molecule that
satisfies all of them earns no bonus for doing so; it simply becomes eligible to be scored.

This is stricter than awarding partial credit per constraint, and it is deliberate: staying inside
the stated windows is the entry requirement, and the score is reserved for what you actually
achieved. Your per-constraint results are still computed and reported in the final report as
diagnostics.

### Optimality: 100 points

Your best **usable** molecule over the whole trajectory — feasible **and** with QED above the
starting molecule's — is compared against a reference target that the grading side holds and does
not disclose. If no submission was usable, the score is **zero**, however many constraints were
satisfied:

```text
achievement = (your_best_usable_QED - starting_QED) / (reference_QED - starting_QED)
```

Your score is a **monotone graded function of `achievement`**: the more of the reachable headroom
your best usable design captures, the higher the band you land in. The reference value, the band
thresholds and the points per band are held by the isolated grading side and are deliberately not
given to you — there is nothing to reverse-engineer, and no strategy other than pushing QED as high
as you can while staying usable.

The reference is the best value found by an offline enumeration, **not** a proof of global
optimality. Molecular space is unbounded and you may legitimately do better; achievement is clipped
at 1.0, so meeting or exceeding the reference earns full marks rather than being penalised.

Your objective is to maximize the final score.

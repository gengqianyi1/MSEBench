# Surrogate-Property Benchmark: Multi-Site Coordinated Optimization of an Anilide Sulfonamide Under a Fingerprint-Similarity Edit Budget

## What this task is, and what it is not

This is a **surrogate-property optimization** task. The objective and every constraint are numbers
**computed deterministically by RDKit from a SMILES string**. Nothing here is measured, assayed,
simulated or fitted to project data:

- The objective, **QED**, is a published desirability composite fitted to the property distributions
  of marketed drugs. It is a coarse *surrogate* for drug-likeness, not a measure of candidate quality.
- **Tanimoto similarity** on Morgan fingerprints is a substructure-overlap statistic. Here it is used
  as an **edit budget**, not as a claim about pharmacological or synthetic relatedness.
- **SA_Score** is a fragment-frequency heuristic, not a synthetic-route assessment.

There is no potency, selectivity, metabolic, toxicity or experimental data anywhere in this task, and
none is simulated. What is being tested is **constrained multi-site molecular editing under
quantitative feedback on an objective you cannot evaluate by inspection**. Doing well here does not
demonstrate medicinal-chemistry judgement or real lead-optimization capability, and it is not meant to.

## Background

You are given one starting molecule from an anilide sulfonamide series. Your task is to raise its
**QED** (RDKit `QED.qed`) as far as you can over five rounds, while staying inside a set of hard
constraints.

One of those constraints is a **similarity floor**, and it is the point of the exercise. A candidate
must keep Morgan/Tanimoto similarity **≥ 0.4** to the starting molecule. Similarity falls as more of
the molecule changes, so the floor is effectively a **budget on how much you are allowed to change**.
Every edit spends part of that budget, and once it is gone your candidate is rejected no matter how
good its QED is.

This benchmark evaluates whether a model can:

- recognise that a similarity floor is an edit budget, and plan the whole redesign before spending it
  rather than converging on it one tweak at a time;
- work inside three property windows that this series pushes against from different directions;
- use quantitative feedback across rounds on an objective it cannot compute by inspection;
- choose, parameterise and recover from a small set of diagnostic tools under a per-round budget.

## Why This Is Not Straightforward

**QED cannot be evaluated by inspection.** It is a fitted geometric mean of eight desirability
functions (molecular weight, LogP, TPSA, hydrogen-bond donors and acceptors, rotatable bonds,
aromatic rings, and a structural-alert term), each a published nonlinear ramp. You cannot estimate it
from the structure the way you can estimate LogP or TPSA — the returned number is the only way to
find out where a design actually sits. **No tool will compute it for you.**

**The starting molecule is already good, and almost every ordinary edit makes things worse or is
rejected.** Its QED is 0.8090. Here is where the moves a chemist reaches for land:

| Ordinary move | Outcome |
|---|---|
| acetamido → hydroxyl | below the MW floor, and TPSA out of range |
| sulfonamide → sulfone (drop the N) | below the MW floor |
| shorten the linker C3 → C2 | below the MW floor |
| drop the ring fluorine | below the MW floor |
| pyrrolidine → dimethylamino | below the MW floor |
| drop the acetamido entirely | below the MW floor, and TPSA out of range |
| acetamido → nitrile | below the MW floor |
| ring fluorine → chlorine | LogP above its ceiling |
| acetamido → methoxy | below the MW floor, and TPSA out of range |
| acetamido → aniline NH₂ | below the MW floor, and a structural alert |
| N-methyl → N-H on the sulfonamide | below the MW floor, and a structural alert |
| pyrrolidine → piperidine | feasible, QED falls |
| pyrrolidine → morpholine | feasible, QED falls |
| acetamido → propionamido | feasible, QED falls |

**None of the fourteen moves in that table earns anything** — each one either leaves a property window
broken or lowers QED, so none of them is a usable design. Read that as a statement about those
fourteen moves and nothing more. The per-move QED and MW values are withheld deliberately: handed to
you as a table they become a calibrated map of the local surface, and reading a map is not the
constrained-editing decision this task measures. The pattern behind the outcomes is not an accident,
and it is the part worth knowing: the start sits about 12 Da above the MW floor while QED on this
molecule improves mainly by shedding mass, so the improving moves and the feasible moves are almost
disjoint sets. Mass removed in one place has to be paid back in another, and the piece you add back
must not itself undo the QED gain.

Single-position edits are **not** uniformly worthless, and you should not assume they are. Some
one-step change in this series is both feasible and an improvement — but a single edit captures only
a small share of the reachable headroom, and one or two positions will not get you much further.
Capturing most of it requires changing several positions in a coordinated way and still landing
inside the budget.

**Spending the budget safely is not enough either.** Feasible designs exist across the whole
similarity range, from just above the 0.4 floor up past 0.75, but they are not equally good: the
designs that capture a large share of the headroom are concentrated in the shell closest to the
floor, where one careless extra edit tips a candidate into rejection. Staying comfortably far from
the floor is therefore its own way of scoring little. How the feasible designs are distributed across
those shells, and exactly how much headroom each shell can reach, are grader-side measurements and
are not disclosed to you.

## Starting Molecule

```text
CC(=O)Nc1ccc(F)c(S(=O)(=O)N(C)CCCN2CCCC2)c1
```

Starting properties computed by RDKit:

| Property | Value |
|---|---:|
| QED | 0.8090 |
| Molecular weight | 357.45 |
| LogP | 1.8905 |
| TPSA | 69.72 |
| H-bond donors | 1 |
| Rotatable bonds | 7 |
| SA score | 2.13 |
| Structural alerts | 0 |

The starting molecule already sits inside every window.

## Objective

- Property: **QED** (RDKit `QED.qed`)
- Direction: **maximize**

## What counts as a submittable molecule

Your `smiles` string must describe **exactly one neutral organic molecule**. The evaluator rejects
anything else outright — it will not strip a counter-ion or neutralise a charge for you, and a
rejected submission spends the round for nothing. Specifically:

1. **One component only.** No salts, no mixtures, no counter-ions. A `.` joining two fragments is a
   rejection (`multiple_fragments`).
2. **Net formal charge zero.** A bare anion or cation is rejected (`net_charge_nonzero`). A net-zero
   zwitterion is acceptable.
3. **Allowed elements only:** H, C, N, O, F, P, S, Cl, Br, I. Anything else is rejected
   (`disallowed_element`).
4. **At least one carbon** (`no_carbon` otherwise).
5. **No isotope labels** (`isotope_label`), **no atom-map numbers** (`atom_map_number`), **no
   wildcard/query atoms** (`wildcard_atom`), **no radicals** (`radical_electron`).
6. **Chemically possible valences.** An impossible valence fails to parse (`unparseable_smiles`).

Accepted molecules are standardised before measurement, so two different SMILES strings for the same
molecule always produce the same numbers.

## What counts as a usable design

A design is **usable** only if **both** of the following hold:

1. it satisfies **every** hard constraint listed below, **and**
2. its QED is **strictly greater than the starting molecule's 0.8090**.

Condition 2 matters as much as condition 1. **Resubmitting the starting molecule scores zero.** So
does any design that stays inside every window but does not raise QED — a feasible design that did not
improve the objective is not a partial success, it is worth nothing. The starting molecule already
satisfies all seven constraints, so "all constraints satisfied" is the entry requirement, never the
achievement. Confirming feasibility with a tool tells you a design is *eligible*, not that it is
*good*.

## Hard Constraints

A candidate is **usable** only if it satisfies **all** of the following. A candidate that violates
any of them is discarded entirely — see the scoring section.

1. **Valid, submittable SMILES** — parseable by RDKit and admissible under the rules above.

2. **Molecular weight** — `Descriptors.MolWt` between **345.0 and 385.0**.

3. **LogP** — `Crippen.MolLogP` between **1.0 and 2.3**.

4. **TPSA** — `rdMolDescriptors.CalcTPSA` between **62.0 and 88.0**.

5. **Synthetic accessibility** — SA_Score **≤ 3.5** (RDKit Contrib; 1 = easy, 10 = hard).

6. **No structural alerts** — **0** alerts across the combined RDKit `FilterCatalog`
   PAINS + Brenk + NIH catalogs.

7. **Similarity to the starting molecule** — Morgan fingerprint Tanimoto similarity **≥ 0.4**
   (radius 2, nBits 2048). This is the edit budget.

## Tools

You have four **diagnostic** tools. You decide whether to use them at all, which ones, on which
molecules, and how to react when one fails.

- **`validate_structure`** — is this string a submittable molecule at all, and if not, why not?
  - arguments: `smiles` (string, required)
  - returns: admissibility flag, canonical SMILES, component count, net formal charge, element list,
    and the rejection reason codes when it is inadmissible
- **`similarity_to_start`** — Morgan/Tanimoto similarity to the starting molecule: how much of the
  edit budget a design has spent.
  - arguments: `smiles` (string, required)
  - returns: `similarity_to_start` (float, 0–1)
- **`structural_alerts`** — run the combined PAINS + Brenk + NIH catalogs on a molecule.
  - arguments: `smiles` (string, required)
  - returns: `n_alerts` (int) and the alert names
- **`constraint_checklist`** — pass/fail against every hard-constraint window above.
  - arguments: `smiles` (string, required)
  - returns: one **boolean** per constraint plus `all_satisfied`

**No tool returns QED, and no tool returns the numeric value of molecular weight, LogP, TPSA,
hydrogen-bond donors or acceptors, rotatable bonds, or aromatic rings.** The objective is yours to
reason about; the tools only tell you whether a design is admissible, how much budget it has spent,
what it trips, and whether it is inside the windows.

### Tool budget

- **6 tool calls per round**, across **at most 2 tool messages per round**. The budget resets each
  round.
- A call that fails because you got the tool name or the arguments wrong **still costs one call**.
- Once the budget is exhausted, further calls are refused with `quota_exceeded`.
- After your second tool message you **must** submit. A third tool message forfeits the round:
  that round records no candidate, the whole run is INCOMPLETE and it receives reward 0 (see
  Scoring Summary).
- Beyond this task-layer quota, each round runs under the per-round agent budget declared in the
  public run configuration (`.chembench_agent.json`): at most **5 conversational turns and 5 tool
  calls per round** (`max_iterations` and `max_tool_calls`). The budget is hard: a round that ends
  with no parseable candidate — either budget spent, replies that never parse, or no submission at
  all — records no candidate for that round, the whole run is INCOMPLETE and it receives reward 0
  (see Scoring Summary).

### Tool errors

Errors come back as data, not as a crash, and you are expected to recover from them:

| code | means |
|---|---|
| `unknown_tool` | no tool by that name; the available names are returned to you |
| `malformed_tool_call` | the call object was not `{"tool": ..., "arguments": {...}}` |
| `invalid_arguments` | `smiles` was missing, empty, or not a string |
| `invalid_molecule` | the molecule is inadmissible; the reason codes are returned |
| `quota_exceeded` | the round's tool budget is gone; submit instead |

## Interaction Rules

- There are **5 rounds**. In each round you submit exactly **one** molecule.
- Every message you send must be **exactly one JSON object** and nothing else — either a tool message
  or a submission.
- After each round, the evaluator returns **computed property values only**:
  - molecular weight, LogP, TPSA, QED, SA score;
  - the number and names of any triggered structural alerts;
  - similarity to the starting molecule;
  - the best QED you have reached so far.
- The evaluator does **not** tell you which constraints you passed or failed, and does **not** tell
  you whether a submission counted as usable. Every constraint is stated in this document; compare the
  returned values against them yourself, or spend a `constraint_checklist` call.
- Do not include explanatory text outside the required JSON object.

## Required Output Format

A tool message:

```json
{
  "tool_calls": [
    {"tool": "constraint_checklist", "arguments": {"smiles": "CC(=O)Nc1ccc(C)c(S(=O)(=O)N(C)CCCN2CCCC2)c1"}}
  ]
}
```

A submission:

```json
{
  "candidate": {
    "name": "candidate_round_1",
    "smiles": "CC(=O)Nc1ccc(C)c(S(=O)(=O)N(C)CCCN2CCCC2)c1",
    "rationale": "Briefly state which edits you made and how the three property windows stay satisfied."
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
achieved.

### Optimality: 100 points

Your best **usable** molecule over the whole trajectory — feasible **and** with QED above 0.8090 — is
compared against a reference target that the grading side holds and does not disclose. If no
submission was usable, the score is **zero**, however many constraints were satisfied:

```text
achievement = (your_best_usable_QED - starting_QED) / (reference_QED - starting_QED)
```

Your score is a **monotone graded function of `achievement`**: the more of the reachable headroom your
best usable design captures, the higher the band you land in. The reference value, the band thresholds
and the points per band are held by the isolated grading side and are deliberately not given to you —
there is nothing to reverse-engineer, and no strategy other than pushing QED as high as you can while
staying usable.

The reference is the best value found by an offline enumeration, **not** a proof of global optimality.
Molecular space is unbounded and you may legitimately do better; achievement is clipped at 1.0, so
meeting or exceeding the reference earns full marks rather than being penalised.

Your objective is to maximize the final score.

# Pharmaceutical Development Benchmark (SIMULATED): Microenvironment- and Solid-State-Limited Co-Design of a Low-Dose, pH-Labile Direct-Compression Tablet Under a Composition-Overlap Floor

> ## What this exercise is, and is not
>
> **This is a synthetic benchmark instance. Nothing in it describes a real product.** The API, the
> excipients' behaviour, the five response equations and the "current formulation" are all invented
> for this exercise. When you submit a formulation, its five critical quality attributes are returned
> by a **numerical surrogate model fitted to fixed-seed samples of a closed-form simulator, plus
> fixed-seed synthetic noise** — not by an analytical laboratory, and not by any model validated
> against a registered product's development data. There is no dossier, no batch record and no
> regulatory meaning behind any number here. Your score measures search and formulation-reasoning
> behaviour on that synthetic surface.
>
> The regulatory framing below (post-approval change control, release specifications, releasability)
> is the *scenario* the exercise is set in and defines the rules you must satisfy. It is not a claim
> that the numbers are real.

## Background

A 5 mg dose of a pH-labile amine-salt API bearing an ester linkage is manufactured as a 200 mg
immediate-release, direct-compression tablet. The API is 2.5 % w/w of the core. The starting
formulation and its two process set points are given below.

Accelerated stability has produced a degradant growth signal. The quality group's assessment is that
the batch record is not at fault: **all five critical quality attributes (CQAs) currently sit on the
unfavourable side of their release and stability specification windows at the same time.**

| # | CQA | Lower limit L | Target T | Upper limit U | Baseline formulation |
|---|---|---:|---:|---:|---:|
| 1 | Microenvironment pH (20 % w/v aqueous slurry of the final blend) | 4.16 | 5.43 | 6.70 | 6.2514 |
| 2 | Equilibrium moisture content at 25 °C / 60 % RH (% w/w) | 1.96 | 2.98 | 3.99 | 3.6293 |
| 3 | Radial tensile strength (MPa) | 1.42 | 2.11 | 2.81 | 1.6655 |
| 4 | Flow function coefficient ffc of the final blend (dimensionless) | 4.35 | 5.61 | 6.87 | 4.7989 |
| 5 | Dissolution Q30, % dissolved at 30 min (paddle, QC medium) | 77.3 | 86.9 | 96.5 | 80.7007 |

**Every one of the five windows is two-sided.** No attribute is "more is better", and each has a
distinct, documented failure mode at *both* ends:

- **Microenvironment pH.** The API's pH–degradation rate profile is U-shaped: both the upper and the
  lower end of the window accelerate loss of assay, by different routes. The pH has to be brought
  *toward the target*, not merely pushed away from whichever limit it currently sits near.
- **Equilibrium moisture content.** Above the ceiling, hydrolysis and a Maillard route open up. Below
  the floor the blend becomes static-prone; for a 2.5 % w/w low-dose API that means content-uniformity
  failure, and it also promotes capping. A very dry blend is a deviation, not an improvement.
- **Radial tensile strength.** Below the floor, friability and coating losses; above the ceiling, the
  tablet no longer meets the disintegration and dissolution profile the dossier was filed with.
- **Flow function coefficient.** Below the floor, fill-weight variation on a rotary press becomes
  unacceptable. Above the ceiling, a freely flowing coarse blend lets a 2.5 % low-dose API segregate
  out of the ordered mix during hopper residence — again a content-uniformity failure.
- **Dissolution Q30.** The product is a generic and must stay inside the reference product's
  dissolution envelope. A Q30 that is too high is as non-compliant as one that is too low.

Under an approved post-approval change management protocol you may make a **limited** change to the
baseline qualitative and quantitative composition and to the two baseline process set points.
Anything larger is a new registration, not a variation, and is out of scope.

## Objective

Maximise the equal-weight Derringer–Suich overall desirability of the five CQAs:

```text
D = ( d_1 · d_2 · d_3 · d_4 · d_5 ) ^ (1/5)

           ( T_i - y_i ) / ( T_i - L_i )      for  y_i <= T_i
    u_i =  ( y_i - T_i ) / ( U_i - T_i )      for  y_i >  T_i
           # distance from the target in half-window units, measured on whichever side of the
           # target you are on, so the two halves of a window may differ in width:
           # u_i = 0 exactly at the target, u_i = 1 exactly at the printed L_i or U_i

               1 / ( 1 + (a · u_i) ^ M )                          for  u_i <= 1
    h(u_i) =
               e · exp( -kappa · M · (1 - e) · (u_i - 1) )         for  u_i >  1

    d_i    = floor + (1 - floor) · h(u_i)

    M = 6.3      e = 2.0e-4      kappa = 2.6      floor = 1.0e-6
    a = ((1 - e)/e) ^ (1/M),  which is what fixes d_i = e at the printed limit
```

with L_i, T_i, U_i exactly as tabulated above. `d_i` is 1 at the target, decreases monotonically as
you move away from it in either direction, and is **strictly positive everywhere**. Its value along
the ruler:

| u_i (half-windows from target) | 0 | 0.1 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | 0.75 | 1.0 (at L or U) | 1.25 | ≥ 1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d_i | 1.000 | 0.9975 | 0.8351 | 0.5540 | 0.2825 | 0.0604 | 0.01552 | 0.001225 | 2.010e-4 | 4.334e-6 | ~1e-6 |

Two consequences worth reading twice. First, the top of the curve is broad but the flanks are steep:
the exponent M = 6.3 means the interval between a quarter and a half of a half-window costs a factor
of 36 in d_i, so **being merely inside a window is worth almost nothing — the value is in the middle
third of the window.** Second, **a CQA outside its window is not annihilated, it is heavily
penalised.** At the limit itself d_i is 2.0e-4 and a further quarter of a half-window out costs
another factor of 46. So a design with four CQAs at d = 0.6 and the fifth exactly on a limit measures
D = 0.121, against D = 0.304 had the fifth been brought to d = 0.02 inside the window: an
out-of-specification design can still be scored, but a specification breach is bought at a price
that a well-centred design will normally beat. A formulation that cannot be released is still a
formulation that cannot be released.

The starting formulation measures d = (0.003105, 0.003226, 0.003182, 0.003199, 0.003136), i.e.
**D = 0.0031693**. It is inside every window and sits at u ≈ 0.64 half-windows from target on all
five at once — inside every specification, close to an edge of every specification.

### D is the optimisation signal. It is not a release criterion.

Read this together with the release gate in **Hard Constraints** below. `d_i` stays strictly positive
outside a window so that the per-round feedback tells you *how badly* you missed, instead of
collapsing to zero and telling you nothing. But a formulation with any CQA outside its window
**cannot be released, is not a success, and earns exactly nothing**, no matter how high its D. A high
D bought by sacrificing one specification is worth zero points. D ranks the *releasable* candidates
against each other; that is its whole job here.

## Why This Is Not Straightforward

**The five CQAs cannot be computed by inspection.** They are reported by a fitted multi-output
surrogate model that ships with this exercise — a synthetic instrument, described in the box at the
top. It is a table of weights, not a formula you can evaluate on paper: no arrangement of the numbers
in this document yields the pH, the moisture uptake, the tensile strength, the ffc or the Q30 of a
formulation you have not submitted. Submitting a formulation and reading back its measured CQAs is
the only way to find out where it sits. Your formulation chemistry tells you the *direction* an
excipient change is likely to push each CQA; it will not tell you the magnitude, and in this instance
several of the directions oppose each other.

**The geometric mean is dominated by whichever CQA you left near an edge.** Four attributes brought
to d = 0.6 and one left at d = 0.02 gives D = 0.304 — no better than putting all five at d = 0.30.
Push that fifth attribute just onto its limit and D falls to 0.121; a quarter of a half-window beyond
the limit and it is 0.056. Progress therefore comes from raising the *worst* attribute, and on this
formulation the moves that raise one attribute lower others.

**The change budget is small, and it is spent in two currencies at once.** You may change at most 4 of
the 12 declared conditions, at most 1 of them an excipient identity change, and the mass-weighted
composition overlap with the baseline formulation must stay at or above 0.88. The main filler is
q.s. to 200 mg, so *every* change to any amount also changes the filler content by the same mass, and
the overlap metric charges you for both movements. Re-dialling an amount is cheap in proportion to the
mass you move; substituting an excipient identity costs that component's whole mass on both sides of
the ratio.

**The obvious strategies have been measured offline, and they fall short.** The whole change-budget
neighbourhood was enumerated with the same evaluator that will grade you. What that enumeration
established is stated below *without* its numbers: a table of ceilings per family would tell you how
many conditions to move and how much overlap to spend, which is the search you are being asked to do.

- Restricting yourself to a single condition, to two conditions, or to the two process set points
  alone each leaves most of the reachable credit unclaimed. So does keeping the composition overlap
  comfortably above its floor instead of spending it.
- Formulations that are releasable **and** better than the starting point are numerous rather than
  rare. Partial progress is genuinely scored, so there is no reason to submit a near-duplicate of the
  starting formulation to stay safe.
- Formulations that improve D while breaching at least one specification are numerous too, and every
  single one of them scores 0. Improving the objective is not the same as clearing the release gate,
  and this is the most common way to score nothing here.
- The top of the scale is narrow. Treat the graded bands as the target, not a plateau.

Scoring well requires several changes chosen to pay for each other, with the composition overlap
spent close to its floor, **and all five windows still closed**. Repairing the CQAs one at a time,
one round per attribute, does not get there.

## The 12 conditions you may set

The core is fixed at 200 mg with 5 mg (2.5 % w/w) API, and dibasic calcium phosphate anhydrous is the
q.s. main filler, making up whatever balance the other components leave. All percentages are % of core
weight. Two of the twelve variables, `T9_blend_min` and `T11_press_MPa`, are process set points: they
do not appear in the composition vector and so cost nothing in composition overlap, but each of them
still consumes one of your four permitted changes.

| Variable | What it sets | Allowed levels | Baseline |
|---|---|---|---|
| `T1_mcc_grade` | grade of microcrystalline cellulose | PH101, PH102, PH200, PH301 | PH102 |
| `T2_mcc_pct` | microcrystalline cellulose, % | 28, 31, 34, 37, 40, 43, 46, 49 | 40 |
| `T3_acid_id` | acidifier identity | none, FUM, TAR, CIT | CIT |
| `T4_acid_pct` | acidifier, % | 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 | 1.0 |
| `T5_meg_pct` | meglumine, % | 0, 0.25, 0.5, 1.0, 2.0 | 0 |
| `T6_dis_id` | superdisintegrant identity | XPVP, CCS, LHPC, SSG | CCS |
| `T7_dis_pct` | superdisintegrant, % | 1, 2, 3, 4, 5, 6, 7 | 6 |
| `T8_mgst_pct` | magnesium stearate, % | 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5 | 1.0 |
| `T9_blend_min` | lubricant blending time, min | 2, 3, 5, 8, 10, 15, 20 | 10 |
| `T10_sil_pct` | colloidal silicon dioxide, % | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 | 0.2 |
| `T11_press_MPa` | main compression pressure, MPa | 100, 120, 140, 160, 180, 200, 220, 240, 260 | 180 |
| `T12_man_pct` | mannitol, % | 0, 2, 4, 6, 8, 10 | 0 |

Every code that appears in a grid above, in grid order, and the substance it denotes:

| Variable | Code | Substance |
|---|---|---|
| `T1_mcc_grade` | PH101 | microcrystalline cellulose, grade PH101 |
| `T1_mcc_grade` | PH102 | microcrystalline cellulose, grade PH102 |
| `T1_mcc_grade` | PH200 | microcrystalline cellulose, grade PH200 |
| `T1_mcc_grade` | PH301 | microcrystalline cellulose, grade PH301 |
| `T3_acid_id` | none | no acidifier |
| `T3_acid_id` | FUM | fumaric acid |
| `T3_acid_id` | TAR | tartaric acid |
| `T3_acid_id` | CIT | citric acid anhydrous |
| `T6_dis_id` | XPVP | crospovidone |
| `T6_dis_id` | CCS | croscarmellose sodium |
| `T6_dis_id` | LHPC | low-substituted hydroxypropyl cellulose |
| `T6_dis_id` | SSG | sodium starch glycolate |
| `T5_meg_pct` | — | meglumine |
| `T12_man_pct` | — | mannitol Pearlitol 100SD |

All four microcrystalline cellulose grades, all three acidifiers, all four superdisintegrants,
meglumine and mannitol are qualified for this product and available on site; the grid is the complete
list of what you may use, at the levels shown.

The baseline formulation therefore has the composition, in mg per 200 mg core: API 5.0,
microcrystalline cellulose PH102 80.0, citric acid anhydrous 2.0, croscarmellose sodium 12.0,
magnesium stearate 2.0, colloidal silicon dioxide 0.4, dibasic calcium phosphate anhydrous 98.6
(49.3 % of the core).

Values are matched numerically, so `40` and `40.0` are the same level. There are about
2.3 × 10⁹ admissible complete designs, of which the change budget admits a small fraction.

## Hard Constraints

A submission is **usable** only if it satisfies all of the following. A submission that violates any
of them is discarded entirely and contributes nothing — see the scoring section.

Constraints 1 to 9 are the change-control and manufacturability rules. **Constraint 10 is the release
gate and it is the one that decides whether you scored at all.**

1. **On-grid.** All 12 variables must be given, each at one of its listed levels. No new excipient, no
   variable that is not listed, no value that is not a listed level.
2. **Acidifier consistency.** The acidifier identity and its amount must agree: an acid named at a
   non-zero level, or `none` at 0 %. Naming an acid at 0 % is not a formulation.
3. **Mass closure.** The core is 200 mg with 5 mg API and the main filler q.s. to 100 %. The q.s.
   filler must remain at least 10 % of the core weight.
4. **Composition overlap ≥ 0.88.** With `w` the mass-fraction vector of your formulation over all
   components and `w0` that of the baseline formulation,

   ```text
   Overlap(w, w0) = Σ_i min(w_i, w0_i) / Σ_i max(w_i, w0_i)
   ```

   A different excipient is a *different component* of the vector. The two process set points do not
   appear in the vector.
5. **At most 4 changed conditions.** At most 4 of the 12 variables may differ from the baseline set
   point, counting process set points.
6. **At most 1 excipient identity change.** Changing `T3_acid_id`, changing `T6_dis_id`, or raising
   `T12_man_pct` above 0 each count as one identity change.
7. **Lubricant range.** `T8_mgst_pct` must be between 0.3 and 1.5 % (the floor is the qualified
   anti-sticking minimum).
8. **Acidifier total.** `T4_acid_pct` must not exceed 4 % (palatability and tolerability).
9. **Validated process ranges.** `T9_blend_min` and `T11_press_MPa` must stay inside the validated
   ranges, which is exactly what their level lists are.
10. **All five CQAs inside specification — the release gate.** Every one of the five measured CQAs
   must lie strictly inside its printed `[L, U]` window: `L_i < y_i < U_i`, for all five at once. A
   formulation that misses one window cannot be released, so it is **not** a partial success: it
   scores 0 and it does not count as a success, whatever its desirability D is. There is no partial
   credit for four windows out of five. The five windows are printed in full in the table at the top
   of this document, and the evaluator returns your five measured CQA values every round — comparing
   them against those windows is your job.

Constraints 3–8 are plain arithmetic on the composition table. You are expected to check them
yourself before you submit; the `validate_formulation` tool will also do it for you. Constraint 10 is
the only one you cannot check before submitting, because it needs the measurement.

## Interaction Rules

- There are **5 rounds**. In each round you submit exactly **one** formulation.
- **Before you submit, you may call diagnostic tools.** Whether to call them, which ones, how many
  times and with what arguments is entirely your decision; a round with no tool call is perfectly
  valid. The tool declarations, the exact JSON call format and the per-round budget are given to you
  with this task. In outline:

  ```text
  validate_formulation   is this a legal submission? on-grid check with per-slot errors, acid
                         consistency, q.s. filler %, composition in % and mg, changed-condition and
                         identity-change counts, composition overlap with the starting formulation,
                         and the boolean hard-constraint checklist for constraints 1-9.
  composition_overlap    mass-weighted composition overlap between two formulations, with the
                         per-component terms, so you can plan how to spend the overlap floor.
  excipient_note         role and generic compatibility / risk caution for ONE excipient code.
  ```

  **No tool measures a formulation.** None of them returns a CQA value, a per-CQA desirability, the
  aggregate desirability D, a score, or any verdict on the specification windows — including
  constraint 10. They restate legality and composition arithmetic you could do yourself from the
  tables above, plus generic excipient cautions that state no direction of effect on any CQA.
  Submitting a formulation remains the only way to learn its five CQAs.
- Tool calls can fail, and a failure is returned to you rather than hidden: an unknown tool name, a
  malformed argument object, a missing slot, an off-grid level, an unknown excipient code, or a spent
  per-round budget each come back as a structured error with a `recoverable` flag. Fix the call and
  retry, or stop calling and submit. **If you keep requesting tools after you have been told your
  budget for the round is spent, you submit nothing that round: that round is voided — it records no
  candidate, the whole run is INCOMPLETE and it receives reward 0 (see Scoring Summary).**
- Each round also runs under the per-round agent budget declared in the public run configuration
  (`.chembench_agent.json`): at most **5 conversational turns and 5 tool calls per round**
  (`max_iterations` and `max_tool_calls`). The budget is hard: a round that ends with no
  parseable candidate — either budget spent, an unparseable reply, or no submission at all —
  records no candidate for that round, the whole run is INCOMPLETE and it receives reward 0
  (see Scoring Summary).
- After each round the evaluator returns:
  - the five **measured CQA values** for the formulation you submitted, keyed
    `microenvironment_pH`, `emc_60rh_pct`, `tensile_strength_MPa`, `ffc`, `q30_pct` — that is rows 1
    to 5 of the specification table, in that order;
  - its **desirability D**;
  - your own **budget bookkeeping**: composition overlap with the baseline formulation, the number
    and names of changed conditions, the number of excipient identity changes, and the q.s. filler
    percentage;
  - the list of **feasibility constraints violated**, if any;
  - the **best D you have reached so far** in this run.
- The evaluator does **not** return: which CQAs are inside or outside their windows, the per-CQA
  desirability breakdown, your score, the reference target, or any pass/fail verdict on the design —
  and in particular it does **not** tell you whether you cleared the release gate. Every
  specification window is printed above; compare the returned CQA values against them yourself.
- A formulation that is off-grid or cannot close mass is not measured at all — no CQAs are returned
  for it. A formulation that is on-grid but over the change budget *is* measured, and still scores
  nothing.
- Rounds are not free. The measurement is the only source of information you do not already have, so
  use early rounds to learn the sensitivities you cannot derive, not to re-submit near-duplicates.
- Return no text outside the required JSON object.

## Required Output Format

Every reply is exactly one JSON object and nothing else. It is either a tool request — the format is
given to you with the tool declarations — or a submission with exactly the following top-level
structure:

```json
{
  "candidate": {
    "name": "candidate_round_1",
    "rationale": "One or two sentences: which conditions you changed and how all five CQAs stay inside their windows.",
    "formulation": {
      "T1_mcc_grade": "PH102",
      "T2_mcc_pct": 40.0,
      "T3_acid_id": "CIT",
      "T4_acid_pct": 1.0,
      "T5_meg_pct": 0.0,
      "T6_dis_id": "CCS",
      "T7_dis_pct": 6.0,
      "T8_mgst_pct": 1.0,
      "T9_blend_min": 10.0,
      "T10_sil_pct": 0.2,
      "T11_press_MPa": 180.0,
      "T12_man_pct": 0.0
    }
  }
}
```

The values shown are the starting formulation, i.e. the point you must improve on, not a suggestion.

## Field Requirements

- `candidate.name`: string, a short identifier.
- `candidate.rationale`: string, one or two concise sentences.
- `candidate.formulation`: object with **all 12** keys exactly as spelled above. Identity slots take
  the code string; every other slot takes a number.

## Scoring Summary

The benchmark score is 100 points, and **all 100 come from the objective**.

All five rounds are mandatory for ranking. A run that does not carry a parseable candidate for
every declared round is INCOMPLETE and receives reward 0 (its scientific score is retained only as
a diagnostic lower bound).

### Feasibility and releasability are gates, not scoring categories

There are **no points for satisfying the constraints**. Clearing them does not earn a bonus; it makes
you eligible to be scored. A submission is scored only if it clears all four gates, in this order:

1. it is a member of the design space — all twelve conditions given, every one at a listed level,
   acidifier identity and amount consistent, mass closes;
2. it is inside the change budget — at most 4 changed conditions, at most 1 excipient identity
   change, composition overlap ≥ 0.88, q.s. filler ≥ 10 %;
3. **all five CQAs are strictly inside their specification windows** (constraint 10, the release
   gate);
4. its desirability D is higher than the starting formulation's D = 0.0031693.

Missing any one of the four scores **0** for that submission. In particular a formulation that
improves D but leaves one CQA outside its window scores 0 and does not count as a success: a
formulation that cannot be released is not a partial success, and a large share of the in-budget
formulations are exactly that case. Per-constraint results are still reported back in the final
report as diagnostics.

### Optimality: 100 points

Your best **qualifying** formulation over the whole run — qualifying meaning it cleared all four
gates above — is compared against a reference target obtained offline by exhaustively enumerating the
change-budget neighbourhood. The reference target is itself a fully releasable design:

```text
achievement = ( your_best_usable_D - D_baseline ) / ( D_reference - D_baseline )
```

Your score is a **monotone graded function of `achievement`**: the more of the reachable headroom
your best qualifying formulation captures, the higher the band you land in. The bands are
deliberately fine-grained at the bottom, so a small but real improvement is distinguished from none
at all rather than lumped together with it. If no submission cleared all four gates the score is
**0**, however many constraints were satisfied.

The reference value is held by the isolated grading side and is not given to you. The band
thresholds and the points per band, by contrast, are published to you: they are the
`scoring.optimality_bands` ladder of the public task configuration that ships with this task
(`input/tasks/q1_microenvironment/task_config.yaml`) and accompanies your first-round prompt.
What stays grading-side is the one number that turns that published ladder into a score — the
reference target `D_reference` — so there is nothing beyond the published ladder to
reverse-engineer, and no strategy other than pushing D as high as you can while keeping every
window closed.

The reference is the best value found by that enumeration, not a proof of global optimality.
`achievement` is clipped at 1.0, so meeting or beating the reference earns full marks rather than being
penalised.

Your objective is to maximise the final score.

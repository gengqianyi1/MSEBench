# Particle-Engineering Benchmark: Four-Window Coordinated Redesign of a Simulated Quasi-Emulsion Spherical Agglomeration Process Under a Controlled-Change Budget

## What This Task Is Built On — Read This First

**Every number in this task comes from a synthetic simulator. None of it is experimental data.**

* There is **no real drug substance, no registered product and no filed process** here. The "baseline
  process" below is a designed starting point, not a process that was ever run.
* The four attributes returned to you each round are **simulator outputs**, not measurements. They are
  produced by a deterministic mechanistic model plus one small surrogate network that was fitted to
  that same model with **synthetic** heteroscedastic noise added at training time (fixed seed
  `20260726`). No noise is added when you are evaluated: the same process always returns the same four
  numbers.
* The simulator's internal mechanism, its coefficients and the surrogate's weights are **withheld from
  you** and held on the grading side. You are given the process context, the design space, the
  specification windows, the scoring rule and your own measurements — and nothing that would let you
  invert the response surface instead of reasoning about the process.
* The simulator is defined **only** on the twelve discrete level grids listed below. An off-grid value
  is refused, not interpolated.

Consequently: doing well here is evidence about coordinated multi-variable decision-making under a
hard change budget with narrow numerical feedback. It is **not** evidence about real spherical
agglomeration, and no number you see may be quoted as a property of any real material.

## Background (the process context, for reasoning — not a claim about any real product)

The scenario: a drug substance crystallises as thin needles. Out of the final crystalliser the powder
has a volume-median size of 12 µm and a needle aspect ratio near 9: it does not flow, it fills a die
inconsistently, and it cannot be directly compressed. The chosen fix is **spherical agglomeration**
by the quasi-emulsion solvent-diffusion (bridging-liquid) route — a standard particle-engineering
unit operation:

1. The primary needles are suspended in a water-rich continuous phase in which the API is poorly
   soluble.
2. A **bridging liquid**, partly miscible with the continuous phase and able to wet the crystal
   surface, is dosed in. It forms capillary liquid bridges between crystals.
3. In the shear field of the impeller the bridged crystals roll up into spherical agglomerates.
   Shear does two things at once: it consolidates (densifies) the agglomerates and it breaks them,
   so the size distribution approaches a steady state instead of growing without limit.
4. The slurry is wet-classified — coarse lumps and/or fines removed on a 850 µm / 106 µm cut, fines
   optionally recycled to the vessel — then washed and dried.

The process is described by **twelve declared variables**. Each of them has a textbook mechanism, and
each mechanism has more than one consequence:

| variable | what it sets, mechanistically |
|---|---|
| `S1_bridging_liquid` | interfacial tension, wetting angle, API solubility in the bridge, and mutual miscibility with the continuous phase; its boiling point also caps the operating temperature |
| `S2_bridging_ratio_v_w` | the **bridging-liquid saturation** of the void space between crystals. Below the pendular onset no agglomerates form; past the droplet limit the batch becomes an over-wetted paste |
| `S3_addition_mode` | how uniformly the bridging liquid is distributed through the suspension, and how many nuclei are created while it is being dosed |
| `S4_solids_g_per_L` | the collision frequency, and the slurry viscosity that degrades bridging-liquid distribution; it also fixes the solvent charge per kg and the speed needed to keep solids suspended |
| `S5_tip_speed_m_s` | simultaneously the orthokinetic collision rate, the consolidation of the wet agglomerate, and the breakage stress acting on it |
| `S6_temperature_C` | API solubility in the bridging liquid, and how strongly the bridging liquid partitions into the continuous phase |
| `S7_agglomeration_min` | how far the agglomeration–breakage balance advances toward its steady state |
| `S8_primary_d50_um` | set upstream by the cooling ramp: a coarser grade is blockier (lower aspect ratio), packs to a different bed porosity, and offers a different number of contact points per unit volume |
| `S9_water_fraction` | how much of the dosed bridging liquid stays in the crystal bed rather than partitioning into the continuous phase, and how much API is dissolved and lost |
| `S10_classification` | which size fractions leave in the product and what re-enters the vessel; recycled fines act as additional nuclei and also soak up bridging liquid |
| `S11_drying` | capillary shrinkage (which densifies and strengthens the agglomerate), thermal-stress cracking (which regenerates fines), and how much bridging liquid is left in the product, which keeps it plastic and drives a lumping/rework loss |
| `S12_HPMC_ppm` | a spheronisation aid: it changes interfacial tension and wetting and it adsorbs on the solid surface |

None of this tells you where the optimum is. Every one of the twelve variables moves **all four** of
the simulated attributes, and all four attributes are specified on **both** sides.

## Why This Is Not Straightforward

**The four attributes cannot be computed by inspection.** The simulator integrates a discretised
agglomeration–breakage balance and evaluates a fitted network; there is no closed form, and its
internals are not disclosed to you. There is no back-of-the-envelope substitute. The simulated numbers
returned to you after each round are the only way to learn where a design actually sits.

**All four attributes of the baseline process are out of specification at once, each on a different
side.** Comparing the simulated values below against the windows:

| attribute | baseline process | window (plateau) | must move |
|---|---:|---|---|
| `d50` | 274.2 µm | 320 – 390 | up |
| `strength` | 1.611 MPa | 0.95 – 1.28 | down |
| `porosity` | 0.3330 | 0.352 – 0.385 | up |
| `yield_pct` | 83.75 % | 89.5 – 92.5 | up |

**No variable is a private handle for one attribute.** Because each of the twelve enters all four
response models, repairing the attribute that looks worst tends to push at least one of the other
three out of its window — and two of the four carry double weight.

**One move at a time will not get you there.** The change budget below is exactly wide enough for one
coordinated multi-variable change and not wide enough for that change plus slack. Repairing the worst
attribute on its own, or holding back part of the budget to be safe, both leave most of the available
credit on the table. How much they leave has been measured offline against this same scorer; those
measurements are held on the grading side and are not disclosed to you, because they would narrow your
search for you.

**The hard constraints are not the puzzle.** They delete a corner of the grid — a low-boiling bridging
liquid at high temperature, a high solids loading at the lowest tip speed — and nothing more. The large
majority of designs inside the change budget are fully runnable. The difficulty is the coupling, not
the feasible region: many in-budget designs improve on the baseline, so partial credit is genuinely
reachable, but the top of the scale is narrow. You have five measurements, so treat the graded bands as
the target and not the plateau.

## The Baseline Process

This is your starting point. It is a **synthetic** baseline, not a real or filed process; the four
attributes above are its simulated values.

```json
{"S1_bridging_liquid": "EtOAc", "S2_bridging_ratio_v_w": 0.35, "S3_addition_mode": "30_min",
 "S4_solids_g_per_L": 100, "S5_tip_speed_m_s": 1.4, "S6_temperature_C": 25,
 "S7_agglomeration_min": 45, "S8_primary_d50_um": 12, "S9_water_fraction": 0.75,
 "S10_classification": "remove_fines", "S11_drying": "50C_vac_6h", "S12_HPMC_ppm": 0}
```

## The Twelve Variables, Their Allowed Values and What Changing Them Costs

Every submission must set all twelve. A value that is not on its list is not a member of the design
space: no new material, no new unit operation, no interpolated set point.

| variable | unit | allowed values | change cost |
|---|---|---|---:|
| `S1_bridging_liquid` | bridging-liquid identity | `heptane`, `cyclohexane`, `heptane_EtOAc_1_1`, `toluene`, `MTBE`, `EtOAc` | 3 |
| `S2_bridging_ratio_v_w` | L bridging liquid per kg API (v/w) | 0.26, 0.30, 0.35, 0.41, 0.47, 0.55 | 2 |
| `S3_addition_mode` | bridging-liquid addition profile | `one_shot`, `10_min`, `30_min`, `60_min`, `90_min`, `3_pulses` | 1 |
| `S4_solids_g_per_L` | g API per L continuous phase | 40, 70, 100, 140, 190, 260 | 2 |
| `S5_tip_speed_m_s` | impeller tip speed, m/s | 0.5, 0.7, 0.9, 1.4, 2.0, 2.8, 3.6 | 1 |
| `S6_temperature_C` | agglomeration temperature, °C | 5, 15, 25, 35, 45 | 1 |
| `S7_agglomeration_min` | hold time after addition, min | 10, 25, 45, 75, 120 | 1 |
| `S8_primary_d50_um` | primary-crystal volume-median size, µm (set by the upstream cooling ramp) | 5, 9, 12, 20, 32 | 1 |
| `S9_water_fraction` | water mass fraction of the continuous phase | 0.55, 0.65, 0.75, 0.85 | 2 |
| `S10_classification` | wet classification / recycle scheme | `none`, `remove_coarse`, `remove_fines`, `remove_both`, `remove_both_recycle`, `remove_both_recycle_high` | 2 |
| `S11_drying` | drying programme | `20C_N2_30h`, `25C_N2_24h`, `40C_vac_12h`, `50C_vac_6h`, `60C_vac_4h`, `40C_stepped_10h` | 3 |
| `S12_HPMC_ppm` | HPMC in the continuous phase, ppm | 0, 100, 300, 800 | 3 |

A change is priced by what it costs in regulatory and plant terms, not by how much it moves the
responses: **3** for a material of record or a new unit operation, **2** for a quantity of record or
a downstream scheme, **1** for an in-process set point. Changing all twelve would cost **22**.

## Objective

Maximise the weighted Derringer–Suich desirability `D` of the isolated, dried product over four
two-sided specification windows.

| attribute | unit | plateau (`d = 1`) | acceptable range (`d = 0.25` at the limit) | weight |
|---|---|---|---|---:|
| `d50` | µm, volume-median diameter of the dried agglomerates | 320 – 390 | 295 – 415 | 2 |
| `strength` | MPa, agglomerate crush strength by micro-compression | 0.95 – 1.28 | 0.75 – 1.48 | 2 |
| `porosity` | dimensionless intra-agglomerate porosity | 0.352 – 0.385 | 0.340 – 0.397 | 1 |
| `yield_pct` | % unit yield of dried agglomerates, API basis | 89.5 – 92.5 | 87.9 – 94.1 | 1 |

Per attribute, with plateau `[pl, pu]` and acceptable limits `[al, au]`:

```text
t = (pl - y) / (pl - al)   below the plateau
  = (y - pu) / (au - pu)   above the plateau
  = 0                      inside the plateau

d = 1 / (1 + (C t)^m),   m = 2.5,   C = 3^(1/m)

D = exp( sum_i w_i * ln d_i / sum_i w_i ),   w = (2, 2, 1, 1) for (d50, strength, porosity, yield)
```

So `d = 1` on the plateau, `d = 0.25` exactly at the acceptable limit, and `d` decays smoothly beyond
it without ever reaching zero: a partly repaired batch keeps partial credit, and `D` never collapses
to zero for a merely bad process. Both limits of every window are real specifications —

* `d50` too small leaves the flow and die-filling problem unsolved; too large fails blend content
  uniformity and makes dissolution diffusion-limited inside the agglomerate.
* `strength` too low means the agglomerates abrade back to fines in drying, transfer and blending;
  too high means they do not deform during compression, and tablet hardness and dissolution both
  fail.
* `porosity` too low traps mother liquor that cannot be washed out (residual solvent and isomer
  limits); too high makes the agglomerates friable and the bulk density too low.
* `yield_pct` too low is uneconomic; too high means the fines were not actually removed and are being
  carried into the product together with the mother liquor they entrap — the upper limit is the
  measurable expression of a purity constraint.

You have the windows, the formula and the four simulated values of every batch you submit, so you can
compute `D` for anything that has been simulated. You cannot compute the four simulated values.

## Hard Constraints

A batch that violates **C1** is not a member of the design space and is not measured at all. A batch
that violates any of **C3–C7** cannot be run, so it is not measured either. A batch that violates
**C2** *is* measured and reported, but scores zero. There is no partial credit for any of them — see
*Scoring Summary*.

**C1 — Design-space membership.** All twelve variables present, each set to one of its listed values.

**C2 — Controlled-change budget.** The weighted change cost of your process against the baseline
process, summed over the variables you moved, must be at most **6 of the 22** points that changing
everything would cost. Equivalently, at least **72.7 %** of the baseline process must survive on
that weighting. This is a budget, not a target: spend it as you see fit, but once it is gone a better
design is worth nothing.

**C3 — Boiling-point safety margin.** The agglomeration temperature must be at least 10 °C below the
atmospheric boiling point of the chosen bridging liquid:

| bridging liquid | heptane | cyclohexane | heptane_EtOAc_1_1 | toluene | MTBE | EtOAc |
|---|---:|---:|---:|---:|---:|---:|
| boiling point, °C | 98.4 | 80.7 | 84.0 | 110.6 | 55.2 | 77.1 |

**C4 — Just-suspended speed.** Zwietering-type criterion: the tip speed must satisfy
`S5_tip_speed_m_s >= 0.42 * (S4_solids_g_per_L / 40)^0.30`.

**C5 — Total solvent.** Continuous phase + bridging liquid + wash, `1000/S4_solids_g_per_L +
S2_bridging_ratio_v_w + 2` L per kg API, must be at most **26** L/kg.

**C6 — Total processing time.** Bridging-liquid addition + agglomeration hold + drying must be at
most **34** h. Addition durations: `one_shot` 0 min, `10_min` 10, `30_min` 30, `60_min` 60, `90_min`
90, `3_pulses` 40. Drying durations: `20C_N2_30h` 30 h, `25C_N2_24h` 24 h, `40C_vac_12h` 12 h,
`50C_vac_6h` 6 h, `60C_vac_4h` 4 h, `40C_stepped_10h` 10 h.

**C7 — Scheme self-consistency.** A fines-recycle scheme is only admissible if the scheme removes
fines in the first place.

## Interaction Rules

* There are **5 rounds**, and all five are always used: there is no early stop and no bonus for
  finishing early.
* In each round, submit exactly **one** process.
* **Optional diagnostic tools.** Before committing a process, you may call diagnostic tools. You decide
  whether to call any, which ones, how many times, and with what arguments. Tool calls are not
  measurements and cost you no round. The tool catalogue, the exact call format and the per-round call
  limits are supplied with the round-1 prompt. The tools report whether a process is a legal member of
  the design space, whether it can be run (the C3–C7 checklist with numeric margins), what it costs
  against your change budget, and how it compares with another process. **No tool returns any quality
  attribute, any prediction of one, the desirability `D`, your achievement or your score** — tools tell
  you whether a process is legal, runnable and affordable, never whether it is good. Tool calls can
  fail (unknown tool, malformed arguments, off-grid process, exhausted quota); a failure returns an
  `error` with a `detail`, and you may repair the call and retry in the same round.
* **Per-round budgets.** Every round runs under the per-round call limits supplied with the round-1
  prompt and under the per-round agent budget declared in the public run configuration
  (`.chembench_agent.json`): at most **5 conversational turns and 5 tool calls per round**
  (`max_iterations` and `max_tool_calls`). The budget is hard: a round that ends with no
  parseable candidate — budget spent, reply unparseable, or no submission at all — records no
  candidate for that round, the whole run is INCOMPLETE and it receives reward 0 (see Scoring
  Summary).
* After each round the evaluator returns, for the batch you submitted:
  * the four **simulated attributes** (`d50`, `strength`, `porosity`, `yield_pct`) and its
    desirability **`D`**;
  * a **per-attribute window report**: for each attribute, its plateau and acceptable limits, whether
    it is below / inside / above the plateau, the normalised distance `t` to the plateau, its
    desirability `d_i` and its weight. This is not extra information — it is exactly what you would get
    by applying the formula above to the four returned values yourself. It is returned so that a round
    is spent on process judgement rather than on arithmetic;
  * your **change-budget bookkeeping**: change points spent, surviving overlap with the baseline
    process, which variables you moved, total solvent in L/kg, total processing hours;
  * the **names of any hard constraints you violated**;
  * a **history table** of every round so far, and how many measurements remain;
  * the best `D` you have reached so far in the run.
* The evaluator does **not** return: your achievement fraction, your score, the reference optimum, any
  verdict on whether the run is going well, or any sensitivity, ranking or suggestion about which
  variable to move next. It also tells you nothing about the simulator's mechanism — which variable
  drives which attribute, or in which direction. You get simulated values, not the model that produced
  them.
* Your score is the **best usable process anywhere in the trajectory**, not the last one, so a later
  round that explores and fails costs you nothing.
* Return only the JSON object — either a tool-call object or one candidate. No commentary outside it.

## Required Output Format

Return only a JSON object with exactly the following top-level structure:

```json
{
  "candidate": {
    "name": "candidate_round_1",
    "rationale": "Briefly state which variables you moved, what the change costs, and how the four windows are meant to end up satisfied.",
    "process": {
      "S1_bridging_liquid": "EtOAc",
      "S2_bridging_ratio_v_w": 0.35,
      "S3_addition_mode": "30_min",
      "S4_solids_g_per_L": 100,
      "S5_tip_speed_m_s": 1.4,
      "S6_temperature_C": 25,
      "S7_agglomeration_min": 45,
      "S8_primary_d50_um": 12,
      "S9_water_fraction": 0.75,
      "S10_classification": "remove_fines",
      "S11_drying": "50C_vac_6h",
      "S12_HPMC_ppm": 0
    }
  }
}
```

(The values shown are the baseline process, so that the shape of the object is unambiguous;
submitting it unchanged spends no budget and improves nothing.)

## Field Requirements

* `candidate.name`: string, a short identifier.
* `candidate.rationale`: string, one or two concise sentences.
* `candidate.process`: object with exactly the twelve keys above, spelled exactly as listed.
  * String-valued variables (`S1`, `S3`, `S10`, `S11`) must be one of the listed labels.
  * Numeric variables (`S2`, `S4`, `S5`, `S6`, `S7`, `S8`, `S9`, `S12`) must be numbers equal to one
    of the listed levels — not a range, not a comment, not a value in between.
  * No extra keys. An unknown key, a missing key or an off-grid value makes the submission a
    non-member of the design space, and that round returns no measurement at all.

## Scoring Summary

The final benchmark score is 100 points, and **all 100 come from the objective**.

All five rounds are mandatory for ranking. A run that does not carry a parseable candidate for
every declared round is INCOMPLETE and receives reward 0 (its scientific score is retained only as
a diagnostic lower bound).

### Constraint satisfaction is a gate, not a scoring category

There are **no points for satisfying the constraints**. A process that breaks a safety or capacity
limit cannot be run, and one that overspends the change budget cannot be filed; neither is a partial
success, so neither contributes anything. A process that satisfies all of them earns no bonus for
doing so — it simply becomes eligible to be scored. Your per-constraint results are still computed and
reported back in the final report as diagnostics.

### Optimality: 100 points

Your best **usable** process over the whole trajectory — one that satisfies every hard constraint,
including the change budget, and has a higher `D` than the baseline process — is compared against a
reference optimum obtained by exhaustively enumerating and grading the entire change-budget
neighbourhood offline:

```text
achievement = (your_best_usable_D - D_of_the_baseline_process) / (reference_D - D_of_the_baseline_process)
```

Your score is a **monotone graded function of `achievement`**: the more of the reachable headroom
your best usable process captures, the higher the band you land in. If no submission was usable —
every hard constraint satisfied *and* `D` above the baseline process — the score is **0**, however
many constraints were satisfied.

The reference value, the band thresholds and the points per band are all held by the isolated
grading side and are deliberately not given to you, in this document or in any round's feedback.
There is nothing to reverse-engineer, and no strategy other than pushing `D` as high as you can
while staying inside every constraint. In particular, do not try to infer a "good enough" stopping
point: you cannot see where the bands fall, so the only sound policy is to keep improving.

The reference is the exact maximum over the enumerated budget neighbourhood, **not** a proof of
global optimality over the 6.5 × 10⁸-design grid; achievement is clipped at 1.0, so meeting or
exceeding it earns full marks rather than being penalised.

Your objective is to maximise the final score.

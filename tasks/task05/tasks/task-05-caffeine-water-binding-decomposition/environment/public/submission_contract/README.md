# Minimal submission contract

Write `submission/final_answer.json` according to `final_answer.schema.json` and a human-readable `submission/report.md`. Claims that are not represented in the structured JSON do not receive scientific-result or uncertainty credit. The report is explanatory and has no independent numeric score.

Use `answer_status: partial` and `not_determined` when evidence is incomplete. Every job ID and geometry path must resolve to Harness-registered evidence from this run.

## Public energy convention

Raw xTB total energies are read in Hartree; reported differences use `kcal/mol`.

```text
E_int = E_AB - E_A^AB - E_B^AB
E_def = (E_A^AB - E_A^opt) + (E_B^AB - E_B^opt)
E_bind = E_int + E_def
closure residual = E_bind - E_int - E_def
```

Here `A^AB` and `B^AB` are frozen fragments extracted coordinate by coordinate from the optimized complex. A stabilizing interaction has negative E_int, deformation has non-negative E_def, and favorable binding has negative E_bind. The absolute closure residual after reporting/rounding must be at most `0.001 kcal/mol`. This is a public arithmetic-consistency rule, not a disclosed private-reference tolerance.

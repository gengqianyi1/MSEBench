# Task 03 — DIPRO/Marcus tolerance fixtures

## Frozen units and protocol

| Quantity | Unit / setting |
|---|---|
| Marcus hole rate | s^-1 |
| DIPRO hole coupling | eV internally; meV input is converted to eV |
| Hole reorganization energy | eV |
| Temperature | 300 K |
| Neutral xTB calculation | charge 0, UHF 0 |
| Cation xTB calculation | charge +1, UHF 1 |

The parser accepts only an explicitly labelled occupied-MO hole-transport
summary. It has no numeric fallback.

## Required regression cases

| Case | Input | Expected result |
|---|---|---|
| A | `total |J(AB,eff)| for hole transport (occ. MOs) : 0.160 eV` | 0.160 eV |
| B | electron/unoccupied-MO or CT rows only | `None` |
| C | command echo containing `--dipro 0.1` or a DIPRO threshold | `None` |
| D | `hole transport (occ. MOs): 1->2 V=44 meV` | 0.044 eV |
| E | missing hole summary | `None`, candidate status `dipro_failed` |

For lambda_h = 0.250 eV, Vh = 0.356 eV, T = 300 K, the frozen
nonadiabatic Marcus expression gives approximately 3.78e14 s^-1. When
Vh/lambda_h >= 1, the evaluator records a regime warning because the value is
being used as a standardized benchmark proxy rather than an experimental
mobility prediction.

Continuous scoring uses zero below 1e14 s^-1 and `min(100, 100*k/1e15)` at or
above the floor. Thus 1e14, 2.5e14, and 1e15 s^-1 yield 10, 25, and 100 points.

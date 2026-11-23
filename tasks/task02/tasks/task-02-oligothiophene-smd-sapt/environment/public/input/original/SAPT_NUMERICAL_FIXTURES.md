# Task 02 — SAPT Numerical Fixtures

## 近简并处理规则

前两大吸引项 |ΔE| ≤ 0.5 kcal/mol → 二者均可视为正确主导项。

## Fragment 定义
- 噻吩单体 = S 原子所在的环为单元
- SMD 中 fragment 与气相相同
- 实现: `src/scoring.py` `_verify_fragment_definition()`

## 几何来源
- 唯一 SAPT 输入 = `structures/R_g.xyz`
- 由 `analysis/formal_coordinate_hashes.json` 验

## 单位/符号
| 项 | 单位 | 符号 |
|----|------|------|
| elst | kcal/mol | 负=吸引 |
| exch | kcal/mol | 正=排斥 |
| ind | kcal/mol | 负=吸引 |
| disp | kcal/mol | 负=吸引 |

## 回归 Fixture

### A: 近简并 (Δ=0.42 ≤ 0.5)
```
Electrostatics    -8.23 kcal/mol
Dispersion        -7.81 kcal/mol
```
→ 主导项: `["Electrostatics", "Dispersion"]`，通过。

### B: 单主导 (Δ=9.20 > 0.5)
```
Electrostatics   -12.40 kcal/mol
Dispersion        -3.20 kcal/mol
```
→ 主导项: `["Electrostatics"]`，通过。

### C: 未收敛
```
(no SAPT0 RESULT block)
SCF converged, SAPT failed after 50 iterations.
```
→ `CONVERGENCE_FAILURE`，不计分。

### D: Fragment 验证
```
A=C4H2S (thiophene), B=C6H4S2 (extended)
```
→ `_verify_fragment_definition()` → True。

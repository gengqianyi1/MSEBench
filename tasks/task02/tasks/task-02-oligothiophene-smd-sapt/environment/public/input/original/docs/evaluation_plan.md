# Evaluation Plan V3.0

The benchmark is scored as 100 deterministic evidence points. No LLM judge is used.

| Section | Points |
|---|---:|
| Input audit and identity | 16 |
| Conformer search and design | 12 |
| Fixed-geometry SMD | 30 |
| Endpoint geometry and lineage | 10 |
| SAPT evidence and dominant attraction | 14 |
| Workflow integrity and reproducibility | 8 |
| Scientific scope and uncertainty | 10 |

## Key policies

- Solvent SMILES/graph/name/alias validation is 0.5 point for each of four solvents.
- Multiple anomalies are injected deterministically into distinct files; their count is hidden from the candidate. Scoring includes recall, type accuracy, file-level evidence, downstream isolation, and false-positive control.
- The SMD scorer parses raw logs and checks them against the private fixed-reference dataset in `scoring_reference.json`.
- SAPT component values are never compared with a private numerical tolerance. Each endpoint receives 7 points only for the correctly supported dominant attractive component; protocol, fragment identity, and geometry lineage are eligibility gates with no separate points.
- Numerical winner, SAPT dominance, population limitations, and geometry-specific SAPT limitations are each scored in only one layer.
- Tool-call count has no hard limit and does not change the score; a six-hour wall-time safety limit remains.
- Missing an anomaly does not automatically erase unrelated valid branches. A score cap applies only when an anomalous artifact actually contaminates a branch claimed as validated.

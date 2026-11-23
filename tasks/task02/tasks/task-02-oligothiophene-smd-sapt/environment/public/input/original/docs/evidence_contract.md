# Raw Evidence Contract V3.0

## Formal SMD logs

All five gas/solvent calculations must use `structures/R_g.xyz` and the fixed M05-2X/6-31G*/RKS/SMD protocol. Raw log energy, convergence, method metadata, solvent name, charge, multiplicity, and geometry path are parsed. The scorer derives ΔE from the logs and compares the raw evidence with a private reference.

## Solvent identities

Stage A must audit n-hexane, 1,4-dioxane, chloroform, and water. Equivalent SMILES are compared as canonical molecular graphs. Each entry must also match its SMD name and ALPB alias.

`analysis/atom_mapping.json` uses integer arrays named `inter_repeat_dihedral`, `sulfur_atoms`, and `terminal_carbons`. Each conformer-search attempt records `starting_geometry_file` and `initial_torsion_deg`. Each explicit-complex record reports `frozen_interaction_energy_kcal_mol` and `deformation_energy_kcal_mol`. These canonical names are part of the public evidence contract.

## Hidden anomalies

One or more seed artifacts may be inconsistent. The count is not public. Each reported finding must include artifact, anomaly type, reproducible evidence, and downstream action. Unsupported false positives reduce the anomaly score.

## Endpoint evidence

Endpoint coordinates must retain the validated solute atom order and heavy-atom bond graph. Geometry metrics are recomputed from the saved coordinates. Explicit complexes must identify the endpoint geometry from which the solute was taken, preserve fragment files, and keep frozen interaction and deformation quantities separate. `run_xtb_single_point` is provided for fixed-geometry energy bookkeeping.

## SAPT evidence

Use `run_psi4_sapt0`. The sole SAPT scoring result is the dominant attractive component, 7 points per endpoint. A parseable raw output, fixed SAPT0/jun-cc-pVDZ metadata, intact fragment identity, and geometry lineage to the selected representative are eligibility gates and carry no separate points. No private numerical tolerance is applied to SAPT components or total. The reported dominant attractive component must agree with the parsed raw components and the endpoint-specific expected classification.

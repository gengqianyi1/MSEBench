# Route contract

Target: `CC(=O)N(CCOC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1)Cc1ccccc1`.

Purchasable inventory (unlimited quantity): `COC(=O)c1cc(Br)cc(Br)c1`, `Cc1ccc(B(O)O)cc1`, `OB(O)c1ccccc1`, `OCCBr`, `NCc1ccccc1`, `CC(=O)Cl`, `COC(=O)c1ccc(Br)cc1`, `COC(=O)c1cccc(Br)c1`, `COC(=O)c1ccccc1`, `COC(=O)c1ccc(Br)c(Br)c1`, `COC(=O)c1c(Br)cccc1Br`, `Cc1cccc(B(O)O)c1`, `Cc1ccccc1B(O)O`, `COc1ccc(B(O)O)cc1`, `Fc1ccc(B(O)O)cc1`, `Clc1ccc(B(O)O)cc1`, `Nc1ccccc1`, `NCCc1ccccc1`, `CNc1ccccc1`, `ClCc1ccccc1`, `BrCCBr`, `CI`, `CCO`, `CC(=O)Br`, `O=C(O)c1ccccc1`.

Reaction names: `suzuki_coupling`, `ester_hydrolysis`, `amide_coupling`, `n_acylation`, `sn2_n_alkylation`, `fischer_esterification`, `sulfonamide_formation`, `reductive_amination`.

Versioned forward reaction SMARTS (the exact executable contract):

| Name | SMARTS |
|---|---|
| `suzuki_coupling` | `[c:1][Br,I].[c:2][B]([OH])[OH]>>[c:1][c:2]` |
| `ester_hydrolysis` | `[C:1](=[O:2])[OX2][CH3]>>[C:1](=[O:2])[OX2H]` |
| `amide_coupling` | `[C:1](=[O:2])[OX2H1].[NX3;H2,H1;!$(NC=O):3]>>[C:1](=[O:2])[N:3]` |
| `n_acylation` | `[NX3;H2,H1;!$(NC=O):1].[C:2](=[O:3])[Cl]>>[N:1][C:2]=[O:3]` |
| `sn2_n_alkylation` | `[NX3;H2,H1;!$(NC=O):1].[C;X4:2][Cl,Br,I]>>[N:1][C:2]` |
| `fischer_esterification` | `[C:1](=[O:2])[OX2H].[OX2H1:3][C:4]>>[C:1](=[O:2])[O:3][C:4]` |
| `sulfonamide_formation` | `[S:1](=[O:2])(=[O:3])[Cl].[NX3;H2,H1:4]>>[S:1](=[O:2])(=[O:3])[N:4]` |
| `reductive_amination` | `[C:1]=[O:2].[NX3;H2,H1;!$(NC=O):3]>>[C:1][N:3]` |

Rules: 2-6 ordered steps; every reactant is inventory or an earlier verified product; named RDKit reaction must generate the declared product; each product heavy-element count cannot exceed declared-reactant counts; RDKit product-MW/sum(reactant-MW) >=0.50 per step; final product equals target.

Equivalence uses RDKit canonical isomeric SMILES. Different stereoisomers are different when specified. Salts/multiple fragments are not stripped or neutralized. Only declared reactants provide heavy atoms; catalysts, solvents and implicit reagents cannot provide unlisted heavy atoms. Named reaction stoichiometry is the SMARTS contract, while each listed inventory item is available without quantity limit.

Score: 35 verified-step progress, 25 process-validity flags, 40 exact target. Exact target is not included in the 25-point process-validity flags, so it is not counted twice.

Public feedback loop: save either a route array or `{"route": [...]}` and run `python input/tools/reaction_validator.py route.json`. The validator and private grader share the same executable chemistry source. The public tool does not contain the target check or private reference route.

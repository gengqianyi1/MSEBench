# Public input provenance and method references

- Molecular identity anchors: [PubChem Caffeine, CID 2519](https://pubchem.ncbi.nlm.nih.gov/compound/2519) and [PubChem Water, CID 962](https://pubchem.ncbi.nlm.nih.gov/compound/962).
- Method: Bannwarth, Ehlert, Grimme, “GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method”, *J. Chem. Theory Comput.* 2019, [DOI: 10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176).

The fourteen complex poses are deterministic rigid-body placements generated from the two frozen XYZ monomers using NumPy only. They sample both inequivalent caffeine carbonyl acceptors, bent edge approaches, and weak pi approaches, but public filenames and XYZ comments do not reveal private construction families. Post-optimization contact families, collapse, competition, and near-degeneracy are determined only from real xTB evidence.

Fragment A atoms precede fragment B atoms, and `fragment_manifest.csv` is the only allowed membership definition. The task reports gas-phase electronic association energy under a frozen-fragment convention; it is not a rigorous EDA or binding free energy. This package contains no structure generator, RDKit dependency, reference energy, Golden output, or private-project code.

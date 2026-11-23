# Public data contract

Source dataset: Matbench/matminer `matbench_expt_gap`, originally archived from `https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz`. This package uses the submitted revision's deterministic composition split and stable IDs; it does not download data at grading time.

The archived source did not include a machine-readable redistribution-license field. Upstream Matbench/materials-project licensing and attribution requirements therefore remain authoritative and must be checked before external redistribution; this package does not infer a license.

| File | Rows | Columns | SHA-256 |
|---|---:|---|---|
| `train.csv` | 3252 | `id,composition,gap_expt_eV` | `837b992b4d9a52b98693e3a3eea65ae6620ca39f494346333ae756961cb05acd` |
| `validation_features.csv` | 692 | `id,composition` | `da085c8b0ac04ee3305f4c4dfbc756e5d31f481541e50415addc44d7bafea2db` |
| `test_features.csv` | 660 | `id,composition` | `f17dfe40b00c1472068f84c7df25a328477a47a0c43f737e0577f1966d0a451f` |

Validation and test labels are not public inputs. Their private file is versioned separately and is never copied into the Agent workspace.

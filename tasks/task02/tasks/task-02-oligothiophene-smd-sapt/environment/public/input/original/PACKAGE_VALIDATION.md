# Package Validation — Release 1.3

The following checks were actually executed on the packaged source tree.

## Passed checks

- All Python files compiled with `py_compile`.
- All JSON and YAML files parsed successfully.
- `python tests/self_test.py` passes with a no-secret mock for structural, schema, safety, and manifest checks.
- The public Agent runner contains no private-reference loader. Optional private fixtures are accepted only by the self-test through `CHEMBENCH_TEST_PRIVATE_SCORING_DIR`; production private scoring is performed by the wrapper's post-run grader.
- Deterministic instance generation was tested across multiple seeds; each tested instance contained two anomalies in two distinct files, while the public evidence note disclosed only that one or more anomalies may exist.
- Repeating the same seed reproduced the same anomaly pair.
- The initial prompt was checked to exclude hidden anomaly counts/types and private scoring-reference data.
- The four-solvent identity rubric was tested at 0.5 point per solvent, 2 points total.
- Synthetic fixed-common-geometry SMD logs matching the private reference received the complete 30-point SMD section.
- SAPT logs with deliberately non-reference numerical component vectors received full SAPT credit only when their evidence passed the eligibility gates and the dominant attractive components were Dispersion for n-hexane and Electrostatics for water. Protocol/geometry lineage earned no separate SAPT points. Breaking the water lineage reduced the SAPT section from 14 to 7 rather than awarding partial lineage credit. This confirms that no private SAPT numerical tolerance is used.
- Stage A, B, and C schema validators accepted synthetic complete evidence and rejected missing required fields.
- The new `run_xtb_single_point` API compiled, was accepted by the safety checker, and was classified as xTB production.
- Direct xTB/PySCF/Psi4 imports and destructive file calls were rejected by static checks.
- Manifest membership was checked as an exact set, not only for missing files, and every listed file other than `MANIFEST.json` was checked against its recorded SHA-256 digest.
- Real API-key configuration files and historical `runs/**` outputs are excluded from the release manifest (except `runs/.gitkeep`) under the same rule used by manifest construction and self-test verification.

## Environment limitation

This validation host does not provide xTB, PySCF, or Psi4 executables. The expensive external calculations were therefore not rerun here. Wrapper code, parsers, schemas, private-reference logic, and synthetic evidence chains were tested. A final end-to-end chemistry run should still be executed in the target Docker/Conda environment.

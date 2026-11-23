# Task 04: neutral L-histidine microstate ensemble

Starting from the single neutral L-histidine seed (`C6H9N3O2`, fixed atom order), autonomously search chemically meaningful microstates and conformers.

Determine the 298.15 K equilibrium populations in gas and ALPB(water), compare the environments, and assess finite-search uncertainty.

Use fixed xTB 6.7.1, GFN2-xTB, charge 0, UHF 0, single-threaded execution, retained L stereochemistry, and an mRRHO rotor cutoff of 50 cm-1. Electronic energy may screen candidates; a population-bearing state needs an optimized geometry and complete Hessian/thermochemistry in that condition.

Choose candidate generation, screening, optimization, Hessian use, retries, grouping, and conclusions. No hints are available.

The compute budget is 56 credits. SP and optimization cost 1 credit each, a Hessian costs 4 credits, and scans are disabled. Failed or timed-out admitted jobs keep their cost. Budget exhaustion rejects new jobs and does not invalidate otherwise valid work.

Wall-clock budget: the scientific phase is capped at 10,800 seconds, followed when needed by a finalization phase of at most 1,800 seconds. New xTB computations are rejected during finalization; use it to turn the evidence you already have into a complete or partial submission.

Scoring detail: at least one trusted xTB job with valid protocol and termination status is a hard prerequisite — without one, the whole task scores zero. The resource section also rewards turn efficiency: approaches that consume turns close to the turn ceiling earn less resource credit than equally valid work completed in fewer turns.

Write all deliverables at the submission root. Read the public schemas with the
file tool using `root=public` and the paths below:

- `final_answer.json`, validated by `path=schema/final_answer.schema.json`
- `candidate_manifest.json`, validated by `path=schema/candidate_manifest.schema.json`

Every `geometry_path` is relative to the submission root. Copy every referenced
XYZ into the submission; `generated_candidates/*.xyz` is the recommended generic
location. The final `submit` call accepts the entire submission directory.

For a characterized minimum, directly copying the optimized XYZ is recommended.
If re-exporting it, keep at least six decimal places in angstroms and preserve
the atom count, element sequence, and atom order. Each submitted coordinate must
agree with the referenced optimization output within 1e-6 angstrom. Scientific
identity, energies, and populations are evaluated from the original xTB evidence,
not from the rounded submission copy.

Scientific scope boundary: this task evaluates a **fixed neutral-formula**
microstate ensemble in gas phase and ALPB(water). It is not a complete
pH-dependent aqueous acid/base distribution. 边界说明：这里评估固定中性分子式的微观状态集合，
不等价于完整的 pH 依赖水相酸碱分布。

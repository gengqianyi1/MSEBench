# Task 18: conformer-resolved oxidation trade-offs

Starting from the three anonymous neutral donor geometries in `input/`, compare their response to one-electron oxidation using the available xTB calculations.

For every candidate you can support, estimate its gas-phase GFN2-xTB adiabatic electronic oxidation-energy proxy and inner-sphere hole-reorganization energy. Rank the candidates separately by ease of oxidation and by low reorganization, identify the candidates on the two-objective Pareto front, and recommend a candidate for each of two priorities: easier oxidation access and lower structural reorganization. Explain how charge-state geometry relaxation affects the result and whether finite conformer exploration is sufficient to support the decision.

The supplied XYZ files are starting geometries, not certified equilibrium minima for either charge state. Choose conformer generation, calculation sequence, validation effort, retries, evidence selection, resource allocation, and uncertainty treatment yourself. The Harness supplies no scientific route.

Fixed physical settings are xTB 6.7.1, GFN2-xTB, gas phase, one compute thread, and unchanged atom count, elements, and atom order. The neutral state is charge `0`, UHF `0`; the one-electron oxidized state is charge `+1`, UHF `1`. Report energy differences in eV.

The main calculation budget is 56 credits and at most 32 admitted xTB jobs. Costs are: single point 1, optimization 1, Hessian 4, and direct optimization-plus-Hessian 5 credits. Failed or timed-out admitted jobs retain their cost. Budget exhaustion rejects new calculations but does not erase credit supported by completed evidence. The Agent has 90 scientific turns, 360 tool calls, and a 10,800-second scientific phase plus a 1800-second finalization allowance, for the 12,600-second Harbor timeout. If needed, up to 8 finalization-only turns remain after the scientific phase; new xTB jobs are rejected during finalization.

Write exactly these model-authored deliverables under `/workspace/submission/`:

- `final_answer.json`, validated by `schema/final_answer.schema.json`
- `report.md`

Harbor resolves these filenames under /workspace/submission/; consequently `task.toml` and `.chembench_agent.json` list the framework-relative artifact names `final_answer.json` and `report.md` rather than repeating the absolute prefix.

The answer-free scoring section weights and hard-zero conditions are published in `rubric.md`. Private reference values and private numerical tolerances are not model-visible.

Scoring detail: at least one trusted Task 18 xTB job with the required method, charge/spin role, and termination status is a hard prerequisite — the whole task receives zero when the organizer evidence contains no such job.

Use the role-specific job fields and `evidence_job_ids` to identify the completed jobs you adopted. The Harness owns the authoritative job records and hashes. Honest `partial` or `failed` submissions remain eligible for evidence-backed partial credit. A missing or invalid deliverable does not erase factual execution and validation already proved by organizer-owned xTB evidence, but all result, ranking, Pareto, recommendation, evidence-binding, and written-scope points that depend on that deliverable become unavailable.

Scientific scope: the result is a gas-phase GFN2-xTB electronic screening model. It is not an experimental oxidation potential, mobility, or electron-transfer rate and does not include crystal packing, electronic coupling, solvent free energy, or outer-sphere reorganization.

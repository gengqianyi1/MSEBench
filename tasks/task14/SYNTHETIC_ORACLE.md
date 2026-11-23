# Task 14 synthetic oracle

Status: **synthetic oracle added; Harbor runtime validation pending**.

This oracle is not a claim of scientific optimality. It is a deterministic Harbor lifecycle
sanity check built from an existing calibrated legacy replay trajectory:

- source corpus: `deepseek-v4-pro_q1_multiobjective_mol_design_20260802_004103`
- legacy expected raw score: **47 / 100**
- expected normalized Harbor reward after the final verifier: **0.47**
- rounds: 5

Each Harbor step has a native `steps/round-N/solution/` directory. The Oracle agent copies the
corresponding fixed `candidate.json` into `/workspace/rounds/round-N/candidate.json`. The normal
Task 14 setup hooks, public feedback evaluator, per-step verifier, final `final_grader.py`, and legacy
private grader are then exercised unchanged.

A successful Harbor oracle trial therefore validates the runtime chain:

`step solution -> candidate -> protocol verifier -> next-step setup/feedback -> ... -> final separate verifier -> reward`.

It does **not** validate the evaluated LLM, `ChemBenchAgent` conversation persistence, or old-vs-new
model parity. Those remain separate gates.

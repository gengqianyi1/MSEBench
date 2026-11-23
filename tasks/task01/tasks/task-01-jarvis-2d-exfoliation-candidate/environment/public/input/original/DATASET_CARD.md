# Dataset Card

## Snapshot

- Dataset: JARVIS-DFT 2D materials
- Benchmark version: `1.0`
- Snapshot ID: `jarvis2d-exfoliation-candidate-v0.4.0`
- Target: `exfoliation_energy_mev_atom`
- Unit: `meV/atom`

## Counts

- training: 538;
- validation: 60;
- base private test structures: 150;
- public test rows including representation variants: 226.

## Candidate generation

The agent generates 1,000 additional structures and supplies a predicted
exfoliation energy for each. The property predictions define a lowest-10%
selection pool but are not compared with unavailable candidate labels.
Candidate quality is assessed through deterministic structural-validity,
pathology, uniqueness, novelty, diversity, and reproducibility checks.
Five agent-selected candidates are independently re-relaxed by the evaluator
with the frozen CHGNet protocol as a local structural-stability proxy.

The public allowed-element list and candidate thresholds are frozen in `data/candidate_generation_constraints.json`.

## Limitations

Rule-based checks and CHGNet relaxation do not prove thermodynamic or dynamic
stability, synthesizability, or experimental feasibility. Candidate thresholds
require empirical calibration before leaderboard release.

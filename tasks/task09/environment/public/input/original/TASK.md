# Adaptive xTB Torsion Profile Agent Benchmark v3

Official suite ID: `adaptive_xtb_torsion_profile_agent_v3`  
Suite protocol: `adaptive_torsion_sampling_tool_loop/3.0`

Choose a cost-efficient sequence of torsion angles from tool feedback. The
organizer evaluates the same public action contract over four independently
frozen molecular cases. The participant package exposes the
2-fluorobiphenyl development case; hidden molecule identities and oracle
profiles are unavailable to the model.

## Capability boundary

You are responsible for the adaptive sampling policy: which angles to measure
after each compact feedback round and when to finalize. The runtime performs
constrained geometry preparation, xTB execution or trusted native-point replay,
validation, automatic fit selection, curve analysis, and final artifact
creation. Do not claim that you designed the curve-fitting algorithm.

## Public action protocol

Use `run_scan_batch` for normal decisions and `inspect_failed_attempt` only
after both native attempts for an accepted angle fail.

- Phase 0: choose exactly six distinct legal angles.
- Phase 1: after automatic fit/analysis feedback, choose two or three new angles.
- Phase 2: choose two or three new angles and set `finalize=true`.
- A bounded fourth batch is recovery-only if native failures leave fewer than
  ten valid unique angles.

Legal angles are the 5° grid from 0 through 355. At most 12 unique angles and
four batches are allowed; at least 10 valid angles and two feedback-driven
rounds are required. Each accepted angle starts from the frozen case reference
and preserves GFN2-xTB, gas phase, charge 0, UHF 0, one thread, target atom
maps, composition, and bond connectivity.

The model sees compact structured state only. It has no arbitrary shell,
Python, network, filesystem, source, evaluator, oracle, credential, or package
access. Automatic fit/analysis output is feedback, not a hidden instruction
about which angle must be selected.

## Evaluation

Scientific profile quality contributes 80 points: 55 for curve RMSE/MaxAE and
25 for periodic, symmetry-aware minima, extrema, and adjacent barriers.
Adaptive sampling and reproducibility contribute 20 points. Eligibility gates
are reported separately and can set official score to zero without erasing the
ungated diagnostic components.

Every result also declares provider status and one of `PASS`, `FAIL`, or
`BLOCKED`. A blocked external execution has null scores and is excluded from
capability averages. Official ranking requires complete organizer-suite
coverage.

The public case's internal executor retains the compatible per-case task ID
`adaptive_xtb_torsion_profile_agent_v2` and protocol 2.0 in its frozen evidence.
The v3 suite wrapper, aggregation, hidden cases, and release boundary are new;
legacy v2 trajectories are not valid v3 suite results.

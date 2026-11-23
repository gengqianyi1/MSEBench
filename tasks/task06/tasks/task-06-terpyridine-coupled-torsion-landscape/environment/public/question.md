# Task 06: coupled conformational energy landscape

Starting from the single neutral `C15H11N3` structure, determine the smallest set of internal rotational coordinates needed to describe its low-energy conformational changes. Discover the principal conformational basins within `6.0 kcal/mol` of the global minimum, establish evidence-backed interconversion paths and directional barriers between the principal basins, and quantify whether the internal rotations can be treated independently.

Use only evidence produced in this run. Keep the atom count, elements, and atom order fixed. The physical settings are xTB 6.7.1, GFN2-xTB, gas phase, charge `0`, UHF `0`, and one compute thread.

You may use at most 12 xTB jobs and 72 requested scan points in total. Each SP, optimization, Hessian, or scan invocation consumes one job. Failed and timed-out admitted jobs retain their cost. Exhausting the scientific budget prevents new calculations but does not erase credit supported by completed evidence. The limit is 55 scientific turns followed, when needed, by 8 finalization-only turns in which no new xTB job is accepted, plus 240 tool calls overall. The scientific phase is additionally capped at three hours (10,800 seconds) of wall-clock time; the finalization phase gets at most 1,800 seconds, and no new xTB job is accepted during it. Work efficiently while keeping scientific correctness primary.

Choose the candidate coordinates, starting conformations, calculation types, sampling locations, validation strategy, resource allocation, and uncertainty treatment yourself. The Harness supplies no scientific route or identity labels.

## Operational definitions and stopping rule

A **minimal torsional coordinate set** is the smallest set of rotatable internal dihedrals for which the submitted evidence can (a) distinguish every reported low-energy minimum and (b) represent the reported interconversion paths. Removing any retained coordinate must merge at least two evidence-backed basins or make at least one reported path/barrier unsupported. Coordinates that were tested but do not meet that criterion must use the `tested_not_retained` role.

**Basin equivalence** is periodic and geometry based. Compare torsions with periodic angular distance, `min(|a-b|, 360-|a-b|)`. Two optimized structures may be treated as the same basin only when every retained torsion differs by at most `20 deg` and their heavy-atom Kabsch RMSD is at most `0.30 angstrom`; use the lower-energy structure as representative. Reported point angles must agree with angles re-derived from the cited geometry within `5 deg`. These are matching/identity rules, not reference basin locations.

A **directional barrier** is defined on one explicitly ordered, evidence-backed path. If `Emax` is the highest energy among its ordered points, then `forward barrier = Emax - E(from basin)` and `reverse barrier = Emax - E(to basin)`, both in `kcal/mol`. The reported highest point, forward barrier, and reverse barrier must all be derived from the same ordered path. Reversing the path swaps the two directional barriers but does not change `Emax`.

Declare **coverage complete** only when each retained primary periodic coordinate has closed 360-degree sampling with no uncovered periodic gap larger than `60 deg`, every provisional minimum within the requested `6.0 kcal/mol` window has been independently optimized or explicitly rejected by evidence, and the coupling claim is supported by a complete rectangular subset with at least 3 distinct values per retained coordinate spanning at least `120 deg` along each axis. This is a minimum evidence condition, not a prescribed grid or workflow. You may stop earlier when the compute budget is exhausted or further work cannot resolve an observed failure; in that case use `answer_status: partial`, list the unsampled/unresolved periodic region in `largest_unresolved_region_description`, give a conservative `omitted_low_energy_basin_count_upper_bound` (or `null` if it cannot be bounded), and add each unsupported basin/path/coupling claim to `not_determined`. Never duplicate a periodic endpoint (for example, both 0 and 360 degrees) as two independent samples.

Write both required deliverables at the submission root:

- `final_answer.json`, validated by `schema/final_answer.schema.json`
- `report.md`

The readable result mirror for a calculation is `workspace/job_results/<job_id>/`. In `final_answer.json`, identify authoritative evidence as `xtb_jobs/<job_id>/<filename>` and bind it to the actual `job_id`, frame index, and SHA-256 required by the schema. Coordinate and basin names are arbitrary; the private evaluator derives coordinates, energies, and state matches from the authoritative xTB output. Relative point energies must reproduce the cited raw energies within `0.05 kcal/mol`, and each reported directional barrier must reproduce its cited path within `0.15 kcal/mol`. Honest `partial` submissions remain eligible for evidence-backed partial credit.

Scoring detail: at least one trusted xTB job with the required GFN2-xTB protocol is a hard prerequisite for any credit; a run whose evidence contains no such job scores zero overall.

The file `xtb_scan_syntax.md` documents the generic constrained-scan input syntax only.

# Active candidate-set discrimination for two pharmaceutical oxidation impurities

Revision 3.2: the hypothesis space includes all 45 unordered pairs of the ten public candidates. Pair-list frequency is not evidence for an identity; use the measurements. Experiments, budget and scientific scoring weights are unchanged.

When native function calling is enabled, call `submit_candidate` with the candidate fields directly as its arguments, without an outer `candidate` key. The JSON-envelope instruction below applies only to text/JSON fallback. The Agent writes files on your behalf; no file-writing tool is required. The public `submission_contract.json` supplies the function argument schema.

You are discriminating between public candidate-pair hypotheses for two co-eluting dominant oxidative degradation impurities in one forced-degradation sample. This is candidate-set discrimination and active experiment selection, not open-world structure generation. The protected case supplies the parent structure, degradation conditions, initial mixed HRMS/MS/UV evidence, public single-impurity candidates, allowed pair-hypothesis IDs, the experiment menu and the remaining budget.

Use only the protected tools exposed by the Harbor sidecar. You may make at most four experiment calls across at most six Agent turns; the last turn is for submission only. Malformed, unknown, rejected and failed scientific calls consume the experiment quota. A provider or sidecar transport failure is an infrastructure outcome, not a scientific score of zero.

Available tools are `energy_resolved_msms`, `selective_derivatization`, `correlation_window`, and `condition_perturbation`. Tool responses contain measurements and public quality flags only. They never reveal the correct pair, candidate ranking, score, threshold or recommended action.

On the terminal turn, return exactly one JSON envelope of the form `{"candidate": { ... }}` and no surrounding prose. The Harbor Agent sends `candidate` to the sidecar. After the sidecar accepts it, the Agent writes the same envelope to `/workspace/final_answer.json` and writes organizer-attested evidence to `/workspace/cases/case-1/case_result.json`. A rejected or malformed terminal envelope produces an `INVALID_SUBMISSION` diagnostic and is not silently repaired.

The nested `candidate` must have exactly the following shape. Do not substitute aliases, flatten nested fields or change the declared types:

- `pair_hypothesis_id`: one string from the protected case's `pair_hypothesis_ids` enumeration.
- `members`: exactly two objects. Each object has only `candidate_id` (a public candidate ID in the selected pair), `role` (exactly one of `aromatic_oxidation_branch`, `amine_oxidation_branch`, or `sidechain_oxidation_branch`), `abundance_interval` (exactly two ordered numbers between 0 and 1) and `structural_features` (a non-empty array drawn from that candidate's public feature IDs).
- `claims`: an array of objects. Each object has only `claim_id` (non-empty string), `evidence_ids` (a non-empty unique array of initial evidence IDs or returned tool `measurement_id` values), `relation` (one of the protected case's public `relations`) and `candidate_ids` (a non-empty unique array of public candidate IDs).
- `pathways`: exactly two objects, one for each selected member. Each object has only `candidate_id` and `events`. Use `events` (not `event_ids`): it is a non-empty ordered array drawn from the protected case's public `pathway_event_ids` enumeration.
- `adaptive_rationale`: a non-empty string explaining how observed feedback informed experiment choices.
- `excluded_pair_hypotheses`: a unique array drawn from the protected case's public `pair_hypothesis_ids` enumeration.
- `uncertainty` must be an object, not a string. It has only `remaining_pair_hypotheses` (a unique array of public pair IDs), `limitations` (an array of non-empty strings) and `budget_external_confirmation` (a non-empty string).

The protected public case includes the same machine-readable `candidate_schema`, including all public candidate, pair, relation, feature and pathway-event enumerations. Values are validated exactly as submitted; malformed objects and aliases are not repaired.

Scoring uses the public rubric at `/workspace/rubric.json` (dimension weights and the two hard gates) together with the legal argument enumerations in the public tools contract; unregistered parameter combinations return `measurement_unavailable` and still consume experiment quota. Read the rubric before selecting experiments.

The submitted pair, cited evidence and experiment usage are independently reconstructed from organizer evidence. Model-authored counters and measurements are not trusted. Submission terminates the case; no later call is permitted.

Scientific boundary: the protected case and all tool responses are deterministic, frozen, synthetic benchmark data. A score is evidence only of performance on this benchmark; it is not a real pharmaceutical impurity confirmation, regulatory conclusion, laboratory safety determination or authorization for experimental use.

# Mixed catalyst-deactivation diagnosis

Revision 2.2.1 clarifies the submission protocol without changing the scientific score. When native function calling is enabled, call `submit_candidate` with the candidate fields directly as its arguments, without an outer `candidate` key. The JSON-envelope instruction below applies only to text/JSON fallback. The Agent writes files on your behalf. The public `candidate_schema.json` supplies the complete argument types; arrays and objects must not be encoded as strings.

A Pt–Sn/Al2O3 propane-dehydrogenation catalyst shows declining conversion, changing selectivity and rising pressure drop. Diagnose the mixture of `coking`, `sintering`, `sulfur_poisoning`, `pore_blockage`, and `external_mass_transfer`; estimate all five contributions; and propose a safe regeneration plan with a predicted post-regeneration performance interval.

The protected case supplies process history and initial characterization. Use only the Harbor sidecar tools: `tpo`, `xps`, `bet`, `pulse_chemisorption`, `particle_size_test`, and `condition_perturbation`. You may make at most four diagnostic calls in at most five Agent turns. Invalid, rejected and failed scientific calls consume quota. Transport/provider failures remain infrastructure outcomes.

On the terminal turn, return exactly one JSON envelope of the form `{"candidate": { ... }}` and no surrounding prose. The Harbor Agent submits the nested object to the sidecar. On acceptance it materializes `/workspace/final_answer.json` with that envelope and `/workspace/cases/case-1/case_result.json` with organizer-attested replay evidence. A malformed or rejected envelope receives an `INVALID_SUBMISSION` diagnostic and is not silently repaired.

The nested `candidate` must contain:

- `primary_mechanisms`: exactly two distinct mechanism IDs, ordered from primary to secondary;
- `contributions`: all five mechanism IDs mapped to numbers in [0,1], summing to 1;
- `confidence`: number in [0,1];
- `claims`: array of exact objects `{claim_id, evidence_ids, relation, mechanism_ids}`; `evidence_ids` and `mechanism_ids` are non-empty arrays and relation is `supports`, `contradicts`, or `neutral`;
- `regeneration_plan`: array of exact objects `{action_id, temperature_C, atmosphere, oxygen_percent, duration_h, ramp_C_min, safety_controls}`; `safety_controls` must be an array of non-empty strings. The plan must explicitly include `bed thermocouples`, `CO2 offgas monitor`, and `hydrogen LEL monitor` on their applicable action. Oxidation atmosphere must be exactly `2% O2/N2` or `dilute oxygen in nitrogen`; reduction atmosphere must be exactly `10% H2/N2` or `dilute hydrogen in nitrogen`, and the reduction step must have `oxygen_percent` equal to 0;
- `predicted_performance`: exact object `{conversion, selectivity}`, each a two-number `[low, high]` interval bounded to `[0,1]`; overly wide intervals are penalized by interval IoU;
- `uncertainty`: exact object `{alternatives, limitations}`, both arrays of strings.

Do not rename fields, use singular `evidence_id`, or substitute generic keys such as `temperature`, `duration`, or `ramp_rate`.

The grader reconstructs diagnostic usage and measurements from organizer evidence. Submission is terminal, and the sidecar never returns mechanism truth, score or recommended regeneration conditions.

Scoring uses the public rubric at `/workspace/rubric.json` (dimension weights and the diagnostic-coherence multiplier); the legal argument enumerations for each diagnostic tool are in the public tools contract, and unregistered combinations are rejected while still consuming quota. Read the rubric before planning diagnostics.

Scientific boundary: the process history, characterization values and protected diagnostic responses are deterministic, frozen, synthetic benchmark data. The score is not a validated diagnosis of a real industrial catalyst, a regeneration safety permit, an operating instruction or a performance guarantee.

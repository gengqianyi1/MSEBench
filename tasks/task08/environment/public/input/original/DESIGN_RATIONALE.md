# Design rationale — xTB Failure v2

The benchmark measures whether an agent can infer a failure mechanism from neutral, trusted evidence and recover from the correct scientific state. Six cases cover optimization-cycle exhaustion, SCC nonconvergence, malformed coordinates, unsupported options, native timeout, and wrong-phase reasoning.

Process compliance is mainly an eligibility gate. Official credit depends on correct diagnosis and a provenance-bound restart source; duplicated facts and always-true mechanics do not add score. Raw subscores preserve partial-progress information while gates prevent scientifically wrong recovery from appearing successful.

Participant distribution includes one public case and public tools/tests. Private cases, reference diagnoses, organizer configs, and formal trajectories remain organizer-only.

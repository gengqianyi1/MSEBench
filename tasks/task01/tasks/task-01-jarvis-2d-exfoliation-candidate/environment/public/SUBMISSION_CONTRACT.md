# Task 01 submission contract — v1.3

This task uses the strict required-artifact policy (方案 A).

Every path listed in `task.yaml.required_artifacts`, including
`candidate_library.json.gz`, is mandatory. The shared Harness state machine is
authoritative:

```text
all required artifacts present -> SUBMITTED -> grader may run
any required artifact missing  -> INVALID_SUBMISSION / NOT_SCORED / score=null
```

An invalid submission must never be assigned a scientific score, including a
partial property-only score. Historical run files that disagree with this
contract are not valid v1.3 results and must not be aggregated.

The scientific task statement requires `model.joblib`, but does not require
preprocessing components to be serialized as separate `scaler.joblib` or
`selector.joblib` files. Those components may be embedded in the submitted
model pipeline. Version 1.3 removes those two wrapper-level requirements while
keeping `candidate_library.json.gz` mandatory under policy A.

The shared Harness is intentionally unchanged. This task-level contract and its
regression test prevent an invalid archived score from being accepted as a
current result.

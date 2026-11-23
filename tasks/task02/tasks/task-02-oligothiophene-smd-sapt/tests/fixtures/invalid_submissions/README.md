# Task 02 invalid-submission fixtures

These fixtures exercise deterministic rejection paths required by the Harbor
contract. Each directory is a submission workspace fragment; the release test
harness must run the private grader against it and compare the listed failure
code before a release candidate can be accepted.

| Fixture | Expected result |
|---|---|
| `missing_result_json` | score 0, `MISSING_RESULT_JSON` |
| `invalid_result_json` | score 0, `INVALID_RESULT_JSON` |
| `non_object_result_json` | score 0, `INVALID_RESULT_SCHEMA` |

Scientific-error fixtures (unit mix-up, solvent order, and SAPT dominance) are
tested through the deterministic rule grader once the target Linux/amd64 image
is available; their expected score and details must be archived with release
evidence.

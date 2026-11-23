# Task-author migration instructions

## Files that must not be edited by task authors

- `schema_version` and `release_gates` in `harness_contract.yaml`.
- `chembench_harness/` core lifecycle, state, budget, security, and reward code.
- `tests/run_legacy_grader.py` compatibility adapter unless a framework maintainer reviews it.
- Private/public directory boundaries and generated content digests.
- CI validation scripts and release-bundle filters.

Changing these files can silently make scores incomparable or expose private evaluator data.

## Files and fields task authors may edit

- `instruction.md`, provided it still describes the same scientific objective.
- Public task data and public schemas, with version and digest updates.
- `protocol.type`, `protocol.max_rounds`, and `budgets` in the contract.
- A task-local plugin implementing validation, scientific tools, and public feedback.
- The private grader and private fixtures inside the private verifier source tree.
- Resource requests and timeouts, after rerunning parity and cost tests.

Every edit must be recorded in `MIGRATION_MAPPING.yaml` and invalidates affected release gates.

## Information task authors must supply

1. Exact legacy model/action/observation sequence.
2. JSON schema for each permitted action and public observation.
3. Tool call limits, retry accounting, timeouts, deterministic seeds, and failure behavior.
4. Which computed fields are public and which are evaluator-only.
5. Raw scoring formula, score bounds, partial-credit and invalid-submission rules.
6. One scientifically valid Oracle, at least three invalid/adversarial submissions, and expected scores.
7. Same-model legacy-versus-Harbor parity tolerances.
8. Confirmation of scientific equivalence, scoring equivalence, private-data isolation, and Oracle validity.

Until all required items are supplied, `implementation_status` remains
`author_input_required` or `partial`, and the task must not enter the release dataset.

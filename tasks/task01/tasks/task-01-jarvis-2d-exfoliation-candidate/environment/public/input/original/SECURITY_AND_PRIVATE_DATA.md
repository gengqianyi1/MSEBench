# Security And Private Data

The complete local evaluator bundle stores property labels at
`private/test_labels_private.json.gz` so the benchmark can be scored without
an external filesystem dependency. This file is excluded from the public
manifest and Docker build context and must never be copied into an agent
workspace. A public distribution must omit it while retaining
`private/README.md`.

The scoring process reads the private labels only after the agent run has
finished. Labels are matched by opaque `sample_id` and snapshot metadata.

Candidate generator code is untrusted. Reproducibility reruns and CHGNet
rescoring must use a network-disabled container without sensitive host mounts.
API keys must be supplied at runtime and redacted from all published artifacts.

## Publication evaluator-bundle note

For this evaluator release, private assets are supplied in the publication-level `evaluator_private/public_tasks/01_jarvis_2d_exfoliation_candidate/` directory, outside the candidate task root. They are evaluator-only and must not be mounted or copied into the Agent workspace.

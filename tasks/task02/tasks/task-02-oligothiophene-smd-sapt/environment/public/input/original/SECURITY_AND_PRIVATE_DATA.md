# Security And Private Data

Hidden anomaly truth and scoring references are runner-side data. They must
not be copied into the candidate workspace or prompt. Candidate and evaluator
API keys must be supplied at runtime and redacted from all published runs.

Candidate Python is treated as untrusted. Production execution must retain the
workspace boundary, subprocess restrictions, network restrictions, and
chemistry wrapper allowlist implemented by the runner.

## Publication evaluator-bundle note

For release 1.3, private assets live in the wrapper task's `private_data/`
directory, outside `input/original`. The native Agent container mounts only
`input/original` and receives no private-data path or environment variable.
After Agent execution, `private_grader/grader.py` reads the private reference
through the evaluator-only `private_data_dir` supplied by the harness.

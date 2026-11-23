# Security And Private Data

This benchmark requires no hidden labels. LLM credentials are runtime secrets
and must not be committed, copied into run reports, or placed in release
archives. Published run configurations must redact `api_key`.

xTB executes local subprocesses. Run untrusted model outputs in an isolated
environment with a dedicated working directory and no sensitive mounted paths.

## Publication evaluator-bundle note

For this evaluator release, private assets are supplied in the publication-level `evaluator_private/original_tasks/03_five_ring_xtb_dipro_marcus/` directory, outside the candidate task root. They are evaluator-only and must not be mounted or copied into the Agent workspace.

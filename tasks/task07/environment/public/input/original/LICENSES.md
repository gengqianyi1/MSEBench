# License and Redistribution Review

## Project source

The benchmark source code and project documentation are provided under the Apache License, Version 2.0; the complete standard text is in `LICENSE`. The accompanying `NOTICE` identifies the project and 2026 notice year without inventing a copyright owner. Before external distribution, the responsible organization must confirm the actual rights holder and its authority to license and distribute the work.

Apache-2.0 applies only to rights that the applicable Licensor and Contributors are legally able to grant. It does not grant rights in third-party software, trademarks, hosted APIs, model weights, datasets, or model outputs.

## Runtime dependencies

| Dependency | Pinned version | Declared upstream license | Use in this project |
|---|---:|---|---|
| Python | 3.12.7 | Python Software Foundation License | Runtime |
| RDKit | 2025.03.5 | BSD 3-Clause | Molecular parsing, descriptors, fingerprints, QED |
| PyYAML | 6.0.1 | MIT | Safe YAML configuration loading |
| pip | 24.3.1 | MIT | Installation of pinned Python packages |
| OpenAI Python SDK | 2.46.0 | Apache-2.0 | OpenAI-compatible provider transport |
| pytest | 7.4.4 | MIT | Tests only |

The table records direct dependencies only. Each dependency and every transitively resolved package remains governed by its own upstream license; this project does not modify, replace, sublicense, or expand those terms. Before external redistribution, the exact resolved environment must be inventoried again and all required license texts, attribution notices, source-offer obligations, and other conditions must be verified. `LICENSES.md` is an engineering inventory, not legal advice and not a substitute for those upstream materials.

## Model outputs and trajectory provenance

The project package contains no model weights or external molecular datasets. LLM outputs and recorded trajectories are generated artifacts, not automatically project source code, and are not automatically relicensed under Apache-2.0. Every retained trajectory must preserve the requested and returned model identifiers, provider, endpoint, timestamps, tool events, evaluation evidence, and other provenance required by the benchmark. The operator must determine whether the applicable model, provider, account, data, privacy, and redistribution terms permit generating, retaining, reviewing, and sharing each artifact.

## SiliconFlow account and credential responsibility

The operator is solely responsible for lawful access to the SiliconFlow account and API, model availability and model-specific terms, authorization to incur charges, rate limits, acceptable-use requirements, and any provider-side retention or privacy settings. Mention of SiliconFlow or a hosted model is interoperability and provenance information only; it does not imply sponsorship or authorization.

`GJLD_API_KEY` must be supplied only through the process environment. Its value must never be placed in this repository, a project configuration file, `.env` material copied into the package, source code, trajectories, logs, reports, manifests, archives, or other persisted artifacts. Credential owners remain responsible for rotation, revocation, scope, and incident response.

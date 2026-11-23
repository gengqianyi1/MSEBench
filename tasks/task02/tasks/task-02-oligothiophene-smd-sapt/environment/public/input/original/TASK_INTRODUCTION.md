# Design Rationale V3.0

This revision keeps the evidence-audited chemistry objective while making the execution contract explicit and reproducible.

1. Required stage JSON files are the model's responsibility and are the only stage-completion records. The runner does not manufacture missing outputs or force a partial finalization.
2. Tool-call count has no hard limit and does not change the score; a six-hour wall-time safety limit remains.
3. Every static rejection condition is disclosed in the candidate prompt; rejected calls are retained in the audit trace.
4. xTB, PySCF, and Psi4 are exposed through one fixed `benchmark_tools.py` interface backed by `protocol.yaml`, rather than requiring each model to rediscover version-specific invocation details.
5. Stage B advances only after content validation and raw-log verification, not after a filename appears.
6. Tool-specific timeout floors prevent expensive jobs from being accidentally assigned a five-minute outer timeout while preserving the global wall-time limit.
7. Missing `result.json` continues to be evaluated as an empty JSON object.

The benchmark still measures planning: the model must decide how to group work, inspect failures, save stage records promptly, and complete the full workflow within budget. Standardizing APIs removes avoidable software-version trivia without supplying scientific conclusions.

# Template plugins — scaffold only

These plugins (`single_step.py`, `multi_round.py`, `adaptive_campaign.py`,
`controlled_tool.py`) are **scaffolds** that import the deprecated
`chembench_harness` contract layer. They are **not** the runtime protocol
implementations used to run Task 01–17.

The runtime protocol handlers live in `agents/chembench_agent.py` (keyed by
`protocol_mode`). Coverage auditing is in `scripts/harness_coverage.py` and
`scripts/audit_harness_coverage.py`.

Note: the contract taxonomy declares `staged_campaign` and `controlled_suite`
types that have no plugin scaffold here; those runtime paths are implemented as
`hidden_suite_case` in the agent and do not require a plugin file.

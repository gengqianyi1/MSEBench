# Deprecated as a runtime layer — 2026-09-04

This package (`chembench_harness/`) is a **declarative-only** contract layer. It is
**not wired into the runtime execution path**:

- `agents/chembench_agent.py` (the actual participant agent) does **not** import it;
- no task-local grader or sidecar imports it;
- only `templates/plugins/*.py` (template scaffolds) reference it.

The canonical **runtime** protocol vocabulary is `agents/chembench_agent.py`'s
`protocol_mode` dispatch keys. Coverage and release gating now live in
`scripts/harness_coverage.py` + `scripts/audit_harness_coverage.py`; do not treat
`ProtocolEngine`/`TaskPlugin` here as the execution engine.

Keep this directory for historical/declarative reference only. If it is re-wired in
the future, that must be an explicit, verified refactor of the agent onto this
engine — not an assumption.

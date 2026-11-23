# Test report — xTB Failure v2

Final command: `python -m pytest -q`.

Result: **165 passed, 3 skipped in 15.86 s** on 2026-08-05. Two skips are Windows environments without symlink creation; the live native test is deliberately opt-in via `RUN_LIVE_XTB=1`. Native xTB 6.7.1 was separately preflighted and used in the deterministic and formal campaigns.

The tests cover parsers, restart provenance, source geometry identity, cycle limits, UTF-8 handling, redaction, path policy, scientific gates, multi-case aggregation, package isolation, and evaluator replay.

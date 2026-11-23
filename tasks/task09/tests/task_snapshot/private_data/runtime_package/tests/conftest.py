"""v2 acceptance collection policy.

The copied v1.1 tests remain in-tree as historical audit material but are not
v2 protocol tests: they assert the retired seven-tool interface, 16-angle
budget, three-feedback state machine, v1.1 campaign snapshot, or v1.1 archive
name.  Current v2 replacements are collected from the other files.
"""

collect_ignore = [
    "e2e/test_frozen_runs.py",
    "e2e/test_oracle_v1_1_asset_validation.py",
    "e2e/test_sampling_metadata_v1_1.py",
    "integration/test_native_tool_loop.py",
    "integration/test_sampling_phase_loop_v1_1.py",
    "unit/test_campaign_snapshot.py",
    "unit/test_sampling_phase_guard_v1_1.py",
    "unit/test_sampling_reconstruction_v1_1.py",
    "unit/test_tool_policy.py",
    "unit/test_trajectory_budget.py",
    "unit/test_version_identity.py",
]

from __future__ import annotations

from pathlib import Path

import yaml

from framework.schemas import (
    BENCHMARK_VERSION,
    CASE_ID,
    FORMAL_CAMPAIGN_ID,
    PROTOCOL,
    TASK_ID,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v2_task_config_identity_matches_runtime_constants() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "task.yaml").read_text(encoding="utf-8")
    )
    assert payload["task"] == {
        "id": TASK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "case_id": CASE_ID,
        "protocol": PROTOCOL,
        "formal_campaign_id": FORMAL_CAMPAIGN_ID,
    }

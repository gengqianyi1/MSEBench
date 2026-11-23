"""Regression: the committed pre-model receipt must match shipped bytes."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.runtime_support import pre_model_source_fingerprint


ROOT = Path(__file__).resolve().parents[2]


def test_current_release_receipt_matches_allowlisted_source() -> None:
    release = json.loads(
        (ROOT / "configs" / "release.json").read_text(encoding="utf-8")
    )
    assert release["pre_model_gates_passed"] is True
    assert release["pre_model_source_sha256"] == pre_model_source_fingerprint(ROOT)

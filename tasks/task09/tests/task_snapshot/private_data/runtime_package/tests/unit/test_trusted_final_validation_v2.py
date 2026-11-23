from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from framework.trusted_final import TrustedFinalError, validate_final_envelope
from tools.science import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]


def _baseline_envelope() -> dict:
    return json.loads(
        (ROOT / "trajectories" / "baseline" / "final_answer.json").read_text(
            encoding="utf-8"
        )
    )


def _rebind_changed_payload(envelope: dict) -> None:
    runtime = envelope["trusted_runtime"]
    runtime["execution_commitment_payload"]["profile_id"] = "unbound-profile"
    runtime["execution_commitment"] = hashlib.sha256(
        canonical_json_bytes(runtime["execution_commitment_payload"])
    ).hexdigest().upper()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["model_submission"].update(
            {"finalize_action_hash": "G" * 64}
        ),
        lambda value: value["trusted_runtime"].update({"status": "FAIL"}),
        lambda value: value["trusted_runtime"]["model_identity"].update(
            {"actual": "different-model"}
        ),
        lambda value: value["trusted_runtime"]["model_identity"].update(
            {"model_override_used": True}
        ),
        lambda value: value["trusted_runtime"]["trajectory"].update(
            {"event_count": 0}
        ),
        _rebind_changed_payload,
    ],
    ids=[
        "nonhex-finalize-hash",
        "nonpass-runtime",
        "identity-mismatch",
        "model-override",
        "empty-trajectory",
        "self-consistent-but-unbound-commitment-payload",
    ],
)
def test_final_envelope_validator_rejects_unbound_or_invalid_runtime(
    mutate: Callable[[dict], None],
) -> None:
    envelope = deepcopy(_baseline_envelope())
    mutate(envelope)

    with pytest.raises(TrustedFinalError):
        validate_final_envelope(envelope)


def test_frozen_baseline_final_envelope_passes_strict_validation() -> None:
    assert validate_final_envelope(_baseline_envelope())

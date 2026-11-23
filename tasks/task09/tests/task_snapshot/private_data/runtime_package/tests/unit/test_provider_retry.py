from __future__ import annotations

import pytest

from framework.provider_retry import (
    DEFAULT_PROVIDER_RETRY_POLICY,
    NONRETRYABLE_PROVIDER_CATEGORIES,
    PREFLIGHT_PROVIDER_RETRY_POLICY,
    TRANSIENT_PROVIDER_CATEGORIES,
    ProviderRetryPolicy,
    ProviderRetryViolation,
    ProviderTurnRetryBinding,
)


def test_formal_retry_policy_is_four_total_attempts_and_sixty_seconds() -> None:
    policy = DEFAULT_PROVIDER_RETRY_POLICY
    assert policy.mode == "formal"
    assert policy.maximum_total_attempts == 4
    assert policy.maximum_cumulative_delay_seconds == 60.0

    cumulative = 0.0
    delays: list[float] = []
    for attempt in range(1, policy.maximum_total_attempts):
        delay = policy.plan_delay(
            attempt_number=attempt,
            cumulative_delay_seconds=cumulative,
            jitter_unit=1.0,
        )
        delays.append(delay)
        cumulative += delay
    assert delays == [6.0, 12.0, 24.0]
    assert cumulative == 42.0 <= 60.0


def test_retry_is_allowed_only_for_four_frozen_transient_categories() -> None:
    policy = DEFAULT_PROVIDER_RETRY_POLICY
    assert TRANSIENT_PROVIDER_CATEGORIES == {
        "RATE_LIMITED",
        "PROVIDER_OVERLOADED",
        "NETWORK_CONNECTION_FAILED",
        "READ_TIMEOUT",
    }
    for category in TRANSIENT_PROVIDER_CATEGORIES:
        assert policy.should_retry(category=category, attempt_number=1)
        assert not policy.should_retry(category=category, attempt_number=4)
    for category in NONRETRYABLE_PROVIDER_CATEGORIES:
        assert not policy.should_retry(category=category, attempt_number=1)
    assert not policy.should_retry(
        category="UNRECOGNIZED_PROVIDER_FAILURE",
        attempt_number=1,
    )


def test_formal_retry_policy_rejects_excess_attempts_or_delay() -> None:
    with pytest.raises(ValueError):
        ProviderRetryPolicy(maximum_total_attempts=5)
    with pytest.raises(ValueError):
        ProviderRetryPolicy(maximum_cumulative_delay_seconds=60.000001)
    with pytest.raises(ValueError):
        DEFAULT_PROVIDER_RETRY_POLICY.plan_delay(
            attempt_number=4,
            cumulative_delay_seconds=0,
            jitter_unit=0,
        )


def test_preflight_has_one_attempt_and_no_retry_delay() -> None:
    policy = PREFLIGHT_PROVIDER_RETRY_POLICY
    assert policy.mode == "preflight"
    assert policy.maximum_total_attempts == 1
    assert policy.maximum_cumulative_delay_seconds == 0
    for category in TRANSIENT_PROVIDER_CATEGORIES:
        assert not policy.should_retry(category=category, attempt_number=1)
    with pytest.raises(ValueError):
        policy.plan_delay(
            attempt_number=1,
            cumulative_delay_seconds=0,
            jitter_unit=0,
        )


def test_retry_binding_requires_same_model_run_and_canonical_state() -> None:
    digest = "A" * 64
    binding = ProviderTurnRetryBinding(
        requested_model_id="deepseek-ai/DeepSeek-V4-Pro",
        run_id="deepseek-formal-v2",
        canonical_state_sha256=digest,
    )
    binding.verify_attempt(
        requested_model_id="deepseek-ai/DeepSeek-V4-Pro",
        run_id="deepseek-formal-v2",
        canonical_state_sha256=digest.lower(),
    )

    changed = (
        {
            "requested_model_id": "fallback/model",
            "run_id": "deepseek-formal-v2",
            "canonical_state_sha256": digest,
        },
        {
            "requested_model_id": "deepseek-ai/DeepSeek-V4-Pro",
            "run_id": "different-run",
            "canonical_state_sha256": digest,
        },
        {
            "requested_model_id": "deepseek-ai/DeepSeek-V4-Pro",
            "run_id": "deepseek-formal-v2",
            "canonical_state_sha256": "B" * 64,
        },
    )
    for attempt in changed:
        with pytest.raises(ProviderRetryViolation):
            binding.verify_attempt(**attempt)


def test_retry_binding_forbids_fallback_and_model_override() -> None:
    with pytest.raises(ProviderRetryViolation):
        ProviderTurnRetryBinding(
            requested_model_id="zai-org/GLM-5.2",
            run_id="glm-formal-v2",
            canonical_state_sha256="C" * 64,
            fallback_used=True,
        )
    binding = ProviderTurnRetryBinding(
        requested_model_id="zai-org/GLM-5.2",
        run_id="glm-formal-v2",
        canonical_state_sha256="C" * 64,
    )
    with pytest.raises(ProviderRetryViolation):
        binding.verify_attempt(
            requested_model_id="zai-org/GLM-5.2",
            run_id="glm-formal-v2",
            canonical_state_sha256="C" * 64,
            model_override_used=True,
        )

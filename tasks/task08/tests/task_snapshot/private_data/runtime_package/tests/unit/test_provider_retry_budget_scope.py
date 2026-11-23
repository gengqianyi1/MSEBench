from evaluator.trusted_validation import (
    _provider_retry_event_ceiling,
    _provider_retries_within_budget,
)


def test_provider_retry_limit_is_per_request_and_case_ceiling_is_derived() -> None:
    budget = {"max_agent_turns": 14, "max_provider_retries": 5}

    assert _provider_retry_event_ceiling(budget) == 70
    assert _provider_retries_within_budget(19, budget)
    assert _provider_retries_within_budget(70, budget)
    assert not _provider_retries_within_budget(71, budget)

"""Frozen retry policy and immutable binding for one SiliconFlow turn.

The OpenAI-compatible SDK is always configured with ``max_retries=0``.  This
module is therefore the only retry authority.  Formal retries are allowed only
for four transient categories, use at most four total attempts, and may plan no
more than 60 seconds of cumulative delay.  A binding prevents a retry from
changing the model, run, or canonical state and explicitly forbids fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .schemas import (
    FORMAL_MAXIMUM_CUMULATIVE_RETRY_DELAY_SECONDS,
    FORMAL_MAXIMUM_TOTAL_ATTEMPTS_PER_TURN,
)


TRANSIENT_PROVIDER_CATEGORIES = frozenset(
    {
        "RATE_LIMITED",
        "PROVIDER_OVERLOADED",
        "NETWORK_CONNECTION_FAILED",
        "READ_TIMEOUT",
    }
)
NONRETRYABLE_PROVIDER_CATEGORIES = frozenset(
    {
        "INSUFFICIENT_BALANCE",
        "AUTHENTICATION_FAILED",
        "MODEL_PERMISSION_DENIED",
        "MODEL_NOT_FOUND",
        "IDENTITY_MISMATCH",
        "PROTOCOL_VIOLATION",
        "UNKNOWN_PROVIDER_ERROR",
    }
)
KNOWN_PROVIDER_CATEGORIES = (
    TRANSIENT_PROVIDER_CATEGORIES | NONRETRYABLE_PROVIDER_CATEGORIES
)

_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")


class ProviderRetryViolation(RuntimeError):
    """A retry attempted to change frozen request identity or policy."""


@dataclass(frozen=True)
class ProviderTurnRetryBinding:
    """Immutable identity of the exact provider turn being retried."""

    requested_model_id: str
    run_id: str
    canonical_state_sha256: str
    fallback_used: bool = False
    model_override_used: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.requested_model_id, str)
            or not _SAFE_ID.fullmatch(self.requested_model_id)
            or not isinstance(self.run_id, str)
            or not _SAFE_ID.fullmatch(self.run_id)
            or not isinstance(self.canonical_state_sha256, str)
            or not _SHA256.fullmatch(self.canonical_state_sha256)
        ):
            raise ProviderRetryViolation("provider retry binding is invalid")
        if self.fallback_used is not False or self.model_override_used is not False:
            raise ProviderRetryViolation("provider fallback or model override is forbidden")

    def verify_attempt(
        self,
        *,
        requested_model_id: str,
        run_id: str,
        canonical_state_sha256: str,
        fallback_used: bool = False,
        model_override_used: bool = False,
    ) -> None:
        """Fail closed if a later attempt differs from the frozen first attempt."""

        if (
            requested_model_id != self.requested_model_id
            or run_id != self.run_id
            or canonical_state_sha256.casefold()
            != self.canonical_state_sha256.casefold()
            or fallback_used is not False
            or model_override_used is not False
        ):
            raise ProviderRetryViolation(
                "provider retry changed model, run, canonical state, or fallback policy"
            )


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """Bounded exponential backoff with deterministic injectable jitter."""

    mode: str = "formal"
    maximum_total_attempts: int = FORMAL_MAXIMUM_TOTAL_ATTEMPTS_PER_TURN
    initial_delay_seconds: float = 5.0
    backoff_multiplier: float = 2.0
    maximum_single_delay_seconds: float = 30.0
    jitter_fraction: float = 0.20
    maximum_cumulative_delay_seconds: float = (
        FORMAL_MAXIMUM_CUMULATIVE_RETRY_DELAY_SECONDS
    )

    def __post_init__(self) -> None:
        if self.mode not in {"formal", "preflight"}:
            raise ValueError("unknown provider retry policy mode")
        if isinstance(self.maximum_total_attempts, bool) or not isinstance(
            self.maximum_total_attempts, int
        ):
            raise ValueError("provider retry attempt budget is invalid")
        if self.mode == "formal":
            if not 1 <= self.maximum_total_attempts <= FORMAL_MAXIMUM_TOTAL_ATTEMPTS_PER_TURN:
                raise ValueError("formal provider turn exceeds four total attempts")
        elif self.maximum_total_attempts != 1:
            raise ValueError("provider preflight permits exactly one attempt")

        numeric = (
            self.initial_delay_seconds,
            self.backoff_multiplier,
            self.maximum_single_delay_seconds,
            self.jitter_fraction,
            self.maximum_cumulative_delay_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("provider retry timing must be finite")
        if self.initial_delay_seconds < 0 or (
            self.mode == "formal" and self.initial_delay_seconds == 0
        ):
            raise ValueError("formal initial provider retry delay must be positive")
        if self.backoff_multiplier <= 1:
            raise ValueError("provider retry multiplier must exceed one")
        if self.maximum_single_delay_seconds < 0 or (
            self.mode == "formal" and self.maximum_single_delay_seconds == 0
        ):
            raise ValueError("formal maximum provider retry delay must be positive")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("provider retry jitter fraction is invalid")
        upper = FORMAL_MAXIMUM_CUMULATIVE_RETRY_DELAY_SECONDS
        if not 0 <= self.maximum_cumulative_delay_seconds <= upper:
            raise ValueError("provider retry cumulative delay exceeds 60 seconds")
        if self.mode == "formal" and self.maximum_cumulative_delay_seconds == 0:
            raise ValueError("formal provider retry cumulative delay must be positive")
        if self.mode == "preflight" and (
            self.initial_delay_seconds != 0
            or self.maximum_single_delay_seconds != 0
            or self.jitter_fraction != 0
            or self.maximum_cumulative_delay_seconds != 0
        ):
            raise ValueError("provider preflight cannot schedule retry delay")

    @classmethod
    def preflight(cls) -> "ProviderRetryPolicy":
        return cls(
            mode="preflight",
            maximum_total_attempts=1,
            initial_delay_seconds=0.0,
            maximum_single_delay_seconds=0.0,
            jitter_fraction=0.0,
            maximum_cumulative_delay_seconds=0.0,
        )

    def should_retry(self, *, category: str, attempt_number: int) -> bool:
        if category not in KNOWN_PROVIDER_CATEGORIES:
            return False
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or not 1 <= attempt_number <= self.maximum_total_attempts
        ):
            raise ValueError("provider retry attempt number is invalid")
        return (
            self.mode == "formal"
            and category in TRANSIENT_PROVIDER_CATEGORIES
            and attempt_number < self.maximum_total_attempts
        )

    def plan_delay(
        self,
        *,
        attempt_number: int,
        cumulative_delay_seconds: float,
        jitter_unit: float,
    ) -> float:
        """Return the next delay clamped to single and cumulative bounds."""

        if self.mode != "formal":
            raise ValueError("provider preflight does not retry")
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or not 1 <= attempt_number < self.maximum_total_attempts
        ):
            raise ValueError("provider retry attempt number is invalid")
        if (
            isinstance(cumulative_delay_seconds, bool)
            or not isinstance(cumulative_delay_seconds, (int, float))
            or not math.isfinite(float(cumulative_delay_seconds))
            or not 0
            <= cumulative_delay_seconds
            <= self.maximum_cumulative_delay_seconds
        ):
            raise ValueError("provider retry cumulative delay is invalid")
        if (
            isinstance(jitter_unit, bool)
            or not isinstance(jitter_unit, (int, float))
            or not math.isfinite(float(jitter_unit))
            or not 0 <= jitter_unit <= 1
        ):
            raise ValueError("provider retry jitter source must return [0, 1]")

        exponential = self.initial_delay_seconds * (
            self.backoff_multiplier ** (attempt_number - 1)
        )
        base = min(exponential, self.maximum_single_delay_seconds)
        jitter_room = max(0.0, self.maximum_single_delay_seconds - base)
        jitter = min(
            base * self.jitter_fraction * float(jitter_unit),
            jitter_room,
        )
        remaining = (
            self.maximum_cumulative_delay_seconds
            - float(cumulative_delay_seconds)
        )
        return round(max(0.0, min(base + jitter, remaining)), 6)


DEFAULT_PROVIDER_RETRY_POLICY = ProviderRetryPolicy()
PREFLIGHT_PROVIDER_RETRY_POLICY = ProviderRetryPolicy.preflight()


__all__ = [
    "DEFAULT_PROVIDER_RETRY_POLICY",
    "KNOWN_PROVIDER_CATEGORIES",
    "NONRETRYABLE_PROVIDER_CATEGORIES",
    "PREFLIGHT_PROVIDER_RETRY_POLICY",
    "ProviderRetryPolicy",
    "ProviderRetryViolation",
    "ProviderTurnRetryBinding",
    "TRANSIENT_PROVIDER_CATEGORIES",
]
